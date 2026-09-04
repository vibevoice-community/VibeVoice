import unittest
from unittest.mock import patch

import numpy as np

from demo.gradio_demo import VibeVoiceDemo


class _FakeProcessor:
    def __call__(self, **kwargs):
        return {}


class _FakeAudioStreamer:
    def __init__(self, **kwargs):
        self._chunks = [
            np.full(120_000, 0.1, dtype=np.float32),
            np.full(120_000, 0.2, dtype=np.float32),
        ]

    def get_stream(self, index):
        return iter(self._chunks)

    def end(self):
        return None


class _FakeThread:
    def __init__(self, target, args):
        self._target = target
        self._args = args

    def start(self):
        return None

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False


class CompleteAudioDeliveryTests(unittest.TestCase):
    @patch("demo.gradio_demo.sf.write")
    @patch("demo.gradio_demo.time.sleep")
    @patch("demo.gradio_demo.threading.Thread", _FakeThread)
    @patch("demo.gradio_demo.AudioStreamer", _FakeAudioStreamer)
    def test_partial_chunks_are_not_published_before_complete_file(
        self,
        sleep_mock,
        write_mock,
    ) -> None:
        demo = VibeVoiceDemo.__new__(VibeVoiceDemo)
        demo.stop_generation = False
        demo.is_generating = False
        demo.current_streamer = None
        demo.available_voices = {"Alice": "alice.wav", "Carter": "carter.wav"}
        demo.inference_steps = 3
        demo.loaded_adapter_root = None
        demo.device = "cpu"
        demo.execution_device = "cpu"
        demo.processor = _FakeProcessor()

        updates = list(
            demo.generate_podcast_streaming(
                num_speakers=2,
                script="Speaker 1: Hello.\nSpeaker 2: Hi.",
                speaker_1="Alice",
                speaker_2="Carter",
                inference_steps=3,
                disable_voice_cloning=True,
            )
        )

        self.assertTrue(updates)
        self.assertTrue(all(streaming_audio is None for streaming_audio, *_ in updates))
        self.assertTrue(all(complete_audio is None for _, complete_audio, *_ in updates[:-1]))
        self.assertIsNotNone(updates[-1][1])
        write_mock.assert_called_once()

    @patch("demo.gradio_demo.sf.write")
    @patch("demo.gradio_demo.time.sleep")
    @patch("demo.gradio_demo.threading.Thread", _FakeThread)
    @patch("demo.gradio_demo.AudioStreamer", _FakeAudioStreamer)
    def test_streaming_option_can_publish_audio_before_complete_file(
        self,
        sleep_mock,
        write_mock,
    ) -> None:
        demo = VibeVoiceDemo.__new__(VibeVoiceDemo)
        demo.stop_generation = False
        demo.is_generating = False
        demo.current_streamer = None
        demo.available_voices = {"Alice": "alice.wav", "Carter": "carter.wav"}
        demo.inference_steps = 3
        demo.loaded_adapter_root = None
        demo.device = "cpu"
        demo.execution_device = "cpu"
        demo.processor = _FakeProcessor()

        updates = list(
            demo.generate_podcast_streaming(
                num_speakers=2,
                script="Speaker 1: Hello.\nSpeaker 2: Hi.",
                speaker_1="Alice",
                speaker_2="Carter",
                inference_steps=3,
                disable_voice_cloning=True,
                stream_audio_during_generation=True,
            )
        )

        self.assertTrue(any(streaming_audio is not None for streaming_audio, *_ in updates[:-1]))
        self.assertIsNotNone(updates[-1][1])
        write_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
