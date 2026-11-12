import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from demo.gradio_demo import VibeVoiceDemo, create_demo_interface


class GradioStopTests(unittest.TestCase):
    def setUp(self):
        self.demo = VibeVoiceDemo.__new__(VibeVoiceDemo)
        self.demo.device = "cpu"
        self.demo.inference_steps = 5
        self.demo.available_voices = {"Alice": "unused.wav"}
        self.demo.loaded_adapter_root = None
        self.demo.current_streamer = None
        self.demo.stop_generation = False
        self.demo.is_generating = False
        self.demo.processor = Mock(return_value={})
        self.interface = create_demo_interface(self.demo)
        self.addCleanup(self.interface.close)
        self.callbacks = {block.fn.__name__: block.fn for block in self.interface.fns.values() if block.fn}

    def start(self, request=None):
        args = (1, "Speaker 1: Hello.", "Alice", None, None, None, 1.3, 5, 42, True)
        callback = self.callbacks["generate_podcast_wrapper"]
        return callback(*args) if request is None else callback(*args, request=request)

    def test_generate_click_goes_directly_to_the_queued_handler(self):
        button = next(block for block in self.interface.blocks.values()
                      if "generate-btn" in (getattr(block, "elem_classes", None) or []))
        handler = next(block for block in self.interface.fns.values()
                       if block.fn and block.fn.__name__ == "generate_podcast_wrapper")
        self.assertEqual(handler.targets, [(button._id, "click")])
        self.assertTrue(handler.queue)

    def test_stop_at_first_update_is_not_reset_before_synthesis(self):
        self.demo._generate_with_streamer = Mock(side_effect=lambda *args: args[2].end())
        owner = SimpleNamespace(session_hash='first')
        result = self.start(owner)
        first = next(result)
        self.assertTrue(first[5]["visible"])
        self.callbacks["stop_generation_handler"](owner)
        final = list(result)[-1]
        self.demo.processor.assert_not_called()
        self.assertIn("stopped", final[2])
        self.assertTrue(final[4]["visible"])
        self.assertFalse(final[5]["visible"])

    def test_stop_waits_for_the_worker_and_ignores_another_session(self):
        entered = threading.Event()
        release = threading.Event()

        def worker(*args):
            entered.set()
            release.wait(8)
            args[2].end()

        self.demo._generate_with_streamer = worker
        owner = SimpleNamespace(session_hash="first")
        other = SimpleNamespace(session_hash="second")
        result = self.start(owner)
        next(result)
        with patch("demo.gradio_demo.time.sleep"), ThreadPoolExecutor(max_workers=1) as pool:
            remaining = pool.submit(list, result)
            try:
                self.assertTrue(entered.wait(4))
                self.callbacks["stop_generation_handler"](other)
                self.callbacks["stop_generation_handler"]()
                self.assertFalse(self.demo.stop_generation)
                self.callbacks["stop_generation_handler"](owner)
                self.assertTrue(self.demo.stop_generation)
                with self.assertRaises(TimeoutError):
                    remaining.result(timeout=0.25)
            finally:
                release.set()
            final = remaining.result(timeout=5)[-1]
        self.assertIn("stopped", final[2])
        self.assertTrue(final[4]["visible"])
        self.assertIsNone(self.demo.current_streamer)

    def test_worker_cleanup_preserves_the_streaming_error(self):
        finished = threading.Event()
        release = threading.Event()

        def worker(_inputs, _cfg, streamer, *_args):
            try:
                streamer.put(torch.zeros((1, 8)), torch.tensor([0]))
                for _ in range(100):
                    if self.demo.stop_generation or release.wait(0.02):
                        break
            finally:
                finished.set()

        self.demo._generate_with_streamer = worker
        try:
            with patch("demo.gradio_demo.time.sleep"), patch(
                "demo.gradio_demo.convert_to_16_bit_wav", side_effect=MemoryError("audio conversion failed")
            ):
                final = list(self.start())[-1]
        finally:
            release.set()
        self.assertTrue(self.demo.stop_generation)
        self.assertTrue(finished.is_set())
        self.assertIn("audio conversion failed", final[2])
        self.assertTrue(final[4]["visible"])


if __name__ == "__main__":
    unittest.main()
