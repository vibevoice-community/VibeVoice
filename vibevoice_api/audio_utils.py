from __future__ import annotations

import io
import wave
from typing import Tuple, Optional

import numpy as np
import subprocess
import shutil
import tempfile

from vibevoice_api.config import CONFIG
from vibevoice_api import observability as obs


def float_to_pcm16(wav: np.ndarray) -> bytes:
    """Convert float32/float64 waveform in [-1, 1] to 16-bit PCM bytes."""
    if wav.dtype != np.float32 and wav.dtype != np.float64:
        wav = wav.astype(np.float32)
    # clip for safety
    wav = np.clip(wav, -1.0, 1.0)
    # scale to int16
    int16 = (wav * 32767.0).astype(np.int16)
    return int16.tobytes()


def encode_wav_pcm16_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    """Return a WAV file (RIFF) with 16-bit PCM from a mono float wav array."""
    # ensure mono 1D
    if wav.ndim > 1:
        if wav.shape[0] == 1:
            wav = wav.squeeze(0)
        else:
            # if multiple channels provided, take first channel
            wav = wav[..., 0]
    pcm = float_to_pcm16(wav)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def apply_speed(wav: np.ndarray, speed: float) -> np.ndarray:
    """Naively change playback speed by resampling via linear interpolation.

    speed > 1.0 speeds up (shorter), < 1.0 slows down (longer).
    """
    if speed is None or abs(speed - 1.0) < 1e-3:
        return wav
    # Clamp to a sensible range (aligned with OpenAI spec)
    if speed <= 0:
        return wav
    speed = float(max(0.25, min(4.0, speed)))
    # linear interpolation resample
    n = wav.shape[-1]
    new_n = max(1, int(round(n / speed)))
    x_old = np.linspace(0.0, 1.0, n, endpoint=False)
    x_new = np.linspace(0.0, 1.0, new_n, endpoint=False)
    return np.interp(x_new, x_old, wav.astype(np.float64)).astype(np.float32)


def _ffmpeg_available(ffmpeg_path: Optional[str] = None) -> bool:
    path = ffmpeg_path or CONFIG.ffmpeg_path
    return shutil.which(path) is not None


def _ffmpeg_transcode(
    wav_bytes: bytes,
    sample_rate: int,
    out_fmt: str,
    ffmpeg_path: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Transcode given WAV bytes to requested format using ffmpeg.

    Returns (data, content_type). Raises RuntimeError on failure.
    """
    ffmpeg = ffmpeg_path or CONFIG.ffmpeg_path

    # Build args per format
    out_fmt = out_fmt.lower()
    if out_fmt == "mp3":
        args = [ffmpeg, "-loglevel", "error", "-f", "wav", "-i", "pipe:0", "-f", "mp3", "-ac", "1", "pipe:1"]
        content_type = "audio/mpeg"
    elif out_fmt == "flac":
        args = [ffmpeg, "-loglevel", "error", "-f", "wav", "-i", "pipe:0", "-f", "flac", "-ac", "1", "pipe:1"]
        content_type = "audio/flac"
    elif out_fmt == "opus":
        # Ogg container with libopus
        args = [
            ffmpeg,
            "-loglevel",
            "error",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-f",
            "ogg",
            "-c:a",
            "libopus",
            "-ac",
            "1",
            "pipe:1",
        ]
        content_type = "audio/ogg"
    elif out_fmt == "aac":
        # ADTS stream
        args = [
            ffmpeg,
            "-loglevel",
            "error",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-f",
            "adts",
            "-c:a",
            "aac",
            "-ac",
            "1",
            "pipe:1",
        ]
        content_type = "audio/aac"
    else:
        raise RuntimeError(f"ffmpeg unsupported format: {out_fmt}")

    try:
        # observe that we are using ffmpeg
        try:
            obs.add_hint(f"ffmpeg_encode:{out_fmt}")
        except Exception:
            pass

        proc = subprocess.run(
            args,
            input=wav_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"ffmpeg not found at '{ffmpeg}'. Set VIBEVOICE_FFMPEG or adjust PATH.") from e
    except subprocess.CalledProcessError as e:
        try:
            obs.add_hint(f"ffmpeg_error:{out_fmt}")
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode(errors='ignore')}") from e

    return proc.stdout, content_type


def to_bytes_for_format(wav: np.ndarray, sample_rate: int, fmt: str) -> Tuple[bytes, str]:
    """Return (data_bytes, content_type) for requested format.

    Supports wav, pcm natively; mp3, flac, opus, aac via optional ffmpeg.
    """
    f = (fmt or "wav").lower()
    if f == "wav":
        return encode_wav_pcm16_bytes(wav, sample_rate), "audio/wav"
    if f == "pcm":
        return float_to_pcm16(wav), "audio/pcm"
    if f in {"mp3", "opus", "aac"}:
        if not _ffmpeg_available():
            try:
                obs.add_hint(f"ffmpeg_missing:{f}")
            except Exception:
                pass
            raise ValueError(
                f"response_format '{f}' requires ffmpeg; set VIBEVOICE_FFMPEG or install ffmpeg in PATH"
            )
        wav_bytes = encode_wav_pcm16_bytes(wav, sample_rate)
        return _ffmpeg_transcode(wav_bytes, sample_rate, f)
    raise ValueError(f"Unsupported response_format: {fmt}")


def ffmpeg_stream_cmd(out_fmt: str, sample_rate: int, ffmpeg_path: Optional[str] = None) -> Tuple[list, str]:
    """Build ffmpeg command for streaming encoding from PCM16 to requested format.

    Returns (cmd_list, content_type).
    """
    ffmpeg = ffmpeg_path or CONFIG.ffmpeg_path
    of = out_fmt.lower()
    br = (CONFIG.ffmpeg_bitrate or "").strip()
    fl = (CONFIG.flac_level or "").strip()

    if of == "mp3":
        args = [ffmpeg, "-loglevel", "error", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0", "-f", "mp3"]
        if br:
            args += ["-b:a", br]
        args += ["pipe:1"]
        return args, "audio/mpeg"
    # flac streaming removed
    if of == "opus":
        container = (CONFIG.opus_container or "webm").lower()
        vbr = (CONFIG.opus_vbr_mode or "vbr").lower()
        vbr_args = []
        if vbr in ("off", "cbr"):
            vbr_args = ["-vbr", "off"]
        elif vbr in ("constrained", "cvbr"):
            vbr_args = ["-vbr", "constrained"]
        else:
            vbr_args = ["-vbr", "on"]
        if container == "webm":
            app = (CONFIG.opus_application or "audio").lower()
            args = [ffmpeg, "-loglevel", "error", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0", "-f", "webm", "-c:a", "libopus", "-application", app] + vbr_args
            if br:
                args += ["-b:a", br]
            if CONFIG.opus_frame_duration:
                args += ["-frame_duration", str(CONFIG.opus_frame_duration)]
            args += ["pipe:1"]
            return args, "audio/webm"
        else:
            app = (CONFIG.opus_application or "audio").lower()
            args = [ffmpeg, "-loglevel", "error", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0", "-f", "ogg", "-c:a", "libopus", "-application", app] + vbr_args
            if br:
                args += ["-b:a", br]
            if CONFIG.opus_frame_duration:
                args += ["-frame_duration", str(CONFIG.opus_frame_duration)]
            args += ["pipe:1"]
            return args, "audio/ogg"
    if of == "aac":
        args = [ffmpeg, "-loglevel", "error", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0", "-f", "adts", "-c:a", "aac"]
        if (CONFIG.aac_mode or "cbr").lower() == "vbr" and CONFIG.aac_q:
            args += ["-q:a", str(CONFIG.aac_q)]
        elif br:
            args += ["-b:a", br]
        if CONFIG.aac_profile:
            args += ["-profile:a", CONFIG.aac_profile]
        args += ["pipe:1"]
        return args, "audio/aac"
    raise ValueError(f"Unsupported streaming format: {out_fmt}")
