import io
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import gradio as gr
import httpx
from gradio import route_utils, routes

from demo import gradio_demo


class UploadCacheTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(
        os.name == "nt" and gr.__version__ == "5.50.0",
        "Gradio 5.50.0 uses the duplicate upload fallback on Windows",
    )
    async def test_reupload_keeps_cached_wav_available_for_playback(self):
        wav = io.BytesIO()
        with wave.open(wav, "wb") as recording:
            recording.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            recording.writeframes(b"\x00\x10\x00\xf0" * 400)
        audio = wav.getvalue()

        with (
            tempfile.TemporaryDirectory() as cache,
            patch.dict(os.environ, {"GRADIO_TEMP_DIR": cache}),
            patch.object(routes, "move_uploaded_files_to_cache", route_utils.move_uploaded_files_to_cache),
            patch.object(sys, "argv", ["gradio_demo.py"]),
            patch.object(gradio_demo, "VibeVoiceDemo"),
            patch.object(gradio_demo, "create_demo_interface"),
            patch.object(gradio_demo, "set_seed"),
        ):
            gradio_demo.main()
            with gr.Blocks(analytics_enabled=False) as demo:
                gr.Audio()
            demo.max_file_size = None  # Normally set by Blocks.launch().
            transport = httpx.ASGITransport(app=routes.App.create_app(demo))
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
                files = {"files": ("sample.wav", audio, "audio/wav")}
                first = await client.post(f"{routes.API_PREFIX}/upload", files=files)
                self.assertEqual(first.status_code, 200)
                cached = Path(first.json()[0])
                os.utime(cached, (1_600_000_000, 1_600_000_000))
                timestamp = cached.stat().st_mtime_ns

                second = await client.post(f"{routes.API_PREFIX}/upload", files=files)
                self.assertEqual(second.status_code, 200)
                self.assertEqual(second.json(), first.json())
                self.assertEqual(cached.stat().st_mtime_ns, timestamp)

                url = f"{routes.API_PREFIX}/file={quote(cached.as_posix(), safe='/:')}"
                playback = await client.get(url, headers={"Range": "bytes=0-31"})
                self.assertEqual(playback.status_code, 206)
                self.assertEqual(playback.content, audio[:32])

    def test_already_cached_source_is_preserved(self):
        with tempfile.TemporaryDirectory() as cache:
            cached = Path(cache) / "sample.wav"
            cached.write_bytes(b"existing audio")
            gradio_demo.cache_uploaded_files([str(cached)], [str(cached)])
            self.assertEqual(cached.read_bytes(), b"existing audio")

    def test_different_content_uses_gradio_move(self):
        with tempfile.TemporaryDirectory() as cache:
            source = Path(cache) / "upload.tmp"
            cached = Path(cache) / "sample.wav"
            source.write_bytes(b"new audio")
            cached.write_bytes(b"old audio")
            os.utime(source, (1_600_000_000, 1_600_000_000))
            os.utime(cached, (1_600_000_000, 1_600_000_000))
            gradio_demo.cache_uploaded_files([str(source)], [str(cached)])
            self.assertEqual(cached.read_bytes(), b"new audio")
            self.assertFalse(source.exists())
