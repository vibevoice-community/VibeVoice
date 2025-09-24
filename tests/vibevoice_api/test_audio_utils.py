from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from vibevoice_api import audio_utils


@pytest.fixture
def sample_wav() -> np.ndarray:
    """Return a simple mono waveform for encoding tests."""

    # A handful of samples keeps the fixture fast while exercising encoding.
    return np.linspace(-0.5, 0.5, num=32, dtype=np.float32)


@pytest.mark.parametrize(
    ("fmt", "expected_content_type"),
    [
        ("mp3", "audio/mpeg"),
        ("opus", "audio/ogg"),
        ("aac", "audio/aac"),
    ],
)
def test_to_bytes_for_format_ffmpeg_available(monkeypatch, sample_wav, fmt, expected_content_type):
    """When ffmpeg is available, transcoding should be attempted for compressed formats."""

    fake_bytes = f"{fmt}-payload".encode()
    observed_args = {}

    monkeypatch.setattr(audio_utils, "_ffmpeg_available", lambda ffmpeg_path=None: True)

    def fake_transcode(wav_bytes: bytes, sample_rate: int, out_fmt: str, ffmpeg_path=None):
        observed_args["call"] = (wav_bytes, sample_rate, out_fmt, ffmpeg_path)
        assert out_fmt == fmt
        return fake_bytes, expected_content_type

    monkeypatch.setattr(audio_utils, "_ffmpeg_transcode", fake_transcode)

    data, content_type = audio_utils.to_bytes_for_format(sample_wav, 16000, fmt)

    assert data == fake_bytes
    assert content_type == expected_content_type
    assert observed_args["call"][2] == fmt


@pytest.mark.parametrize("fmt", ["mp3", "opus", "aac"])
def test_to_bytes_for_format_ffmpeg_missing(monkeypatch, sample_wav, fmt):
    """A helpful error is raised when ffmpeg is unavailable."""

    monkeypatch.setattr(audio_utils, "_ffmpeg_available", lambda ffmpeg_path=None: False)

    transcode_called = {"value": False}

    def fake_transcode(*args, **kwargs):
        transcode_called["value"] = True
        return b"", "audio/unknown"

    monkeypatch.setattr(audio_utils, "_ffmpeg_transcode", fake_transcode)

    with pytest.raises(ValueError) as excinfo:
        audio_utils.to_bytes_for_format(sample_wav, 16000, fmt)

    assert "requires ffmpeg" in str(excinfo.value)
    assert not transcode_called["value"]
