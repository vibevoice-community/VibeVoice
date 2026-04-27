#!/usr/bin/env python3
"""
Heuristic checker for repeated watermark-like content across audio files.

This script looks for two broad signals:
1. Audible watermark/disclaimer:
   repeated content near the beginning or end of generated clips.
2. Imperceptible watermark:
   repeated high-frequency spectral structure across otherwise different clips.

It does not prove the presence or absence of a watermark. It reports similarity
metrics that can help you decide whether multiple files share a stable pattern.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Iterable, List

import numpy as np
from scipy import signal
from scipy.io import wavfile

try:
    import soundfile as sf
except ImportError:
    sf = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether multiple audio files appear to share a common watermark-like pattern."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Audio files and/or directories. Directories are searched recursively for wav/flac/mp3/m4a/ogg.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Target sample rate for analysis. Default: 24000",
    )
    parser.add_argument(
        "--prefix-sec",
        type=float,
        default=1.5,
        help="Analyze this much audio from the start of each clip. Default: 1.5",
    )
    parser.add_argument(
        "--suffix-sec",
        type=float,
        default=1.5,
        help="Analyze this much audio from the end of each clip. Default: 1.5",
    )
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=20.0,
        help="Cap each clip to this many seconds for full-clip analysis. Default: 20.0",
    )
    parser.add_argument(
        "--highpass-hz",
        type=float,
        default=8000.0,
        help="Lower edge of the high-frequency band used for imperceptible watermark checks. Default: 8000",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=2048,
        help="FFT size for spectral analysis. Default: 2048",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length for spectral analysis. Default: 512",
    )
    parser.add_argument(
        "--lag-sec",
        type=float,
        default=0.25,
        help="Maximum absolute lag used in short-segment cross-correlation. Default: 0.25",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save the full report as JSON.",
    )
    return parser.parse_args()


def iter_audio_files(inputs: Iterable[str]) -> List[Path]:
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    paths: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() in exts:
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in exts))
    deduped = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def load_audio(path: Path, sample_rate: int, max_duration_sec: float) -> np.ndarray:
    audio, original_sr = read_audio(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = to_float32(audio)
    if original_sr != sample_rate:
        audio = resample_audio(audio, original_sr, sample_rate)
    limit = int(max_duration_sec * sample_rate)
    if limit > 0 and len(audio) > limit:
        audio = audio[:limit]
    return normalize_audio(audio)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    if sf is not None:
        audio, sr = sf.read(path.as_posix(), always_2d=False)
        return np.asarray(audio), int(sr)
    if path.suffix.lower() != ".wav":
        raise RuntimeError("soundfile is not installed, so only WAV files are supported.")
    sr, audio = wavfile.read(path.as_posix())
    return np.asarray(audio), int(sr)


def to_float32(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float32)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(info.min), info.max)
        return (audio.astype(np.float32) / float(scale)).clip(-1.0, 1.0)
    return audio.astype(np.float32)


def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return audio
    gcd = math.gcd(original_sr, target_sr)
    up = target_sr // gcd
    down = original_sr // gcd
    return signal.resample_poly(audio, up, down).astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def take_prefix(audio: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    count = max(1, int(seconds * sample_rate))
    return audio[:count]


def take_suffix(audio: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    count = max(1, int(seconds * sample_rate))
    return audio[-count:]


def safe_unit_norm(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm <= 1e-12:
        return np.zeros_like(x)
    return x / norm


def max_corr_with_lag(a: np.ndarray, b: np.ndarray, max_lag_samples: int) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    length = min(len(a), len(b))
    if length < 16:
        return 0.0
    a = a[:length] - np.mean(a[:length])
    b = b[:length] - np.mean(b[:length])
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        return 0.0
    corr = signal.correlate(a, b, mode="full", method="fft") / (a_norm * b_norm)
    center = len(corr) // 2
    start = max(0, center - max_lag_samples)
    end = min(len(corr), center + max_lag_samples + 1)
    if start >= end:
        return 0.0
    return max(float(np.max(corr[start:end])), 0.0)


def band_limited(audio: np.ndarray, sample_rate: int, highpass_hz: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    spectrum[freqs < highpass_hz] = 0
    filtered = np.fft.irfft(spectrum, n=len(audio))
    return normalize_audio(filtered.real.astype(np.float32))


def spectral_fingerprint(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    highpass_hz: float,
) -> np.ndarray:
    if audio.size < 32:
        return np.zeros(16, dtype=np.float32)
    freqs, _, spec = signal.stft(
        audio,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(spec)
    band = magnitude[freqs >= highpass_hz]
    if band.size == 0:
        return np.zeros(16, dtype=np.float32)
    mean_spectrum = np.mean(band, axis=1)
    log_spec = np.log1p(mean_spectrum)
    if log_spec.size >= 9:
        smooth = signal.savgol_filter(log_spec, window_length=9, polyorder=2, mode="interp")
    else:
        smooth = log_spec
    residual = log_spec - smooth
    residual = residual - np.mean(residual)
    norm = np.linalg.norm(residual)
    if norm <= 1e-12:
        return np.zeros_like(residual, dtype=np.float32)
    return (residual / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    length = min(len(a), len(b))
    a = a[:length]
    b = b[:length]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def summarize_scores(values: List[float]) -> dict:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": round(float(np.min(values)), 4),
        "mean": round(float(np.mean(values)), 4),
        "max": round(float(np.max(values)), 4),
    }


def heuristic_flags(prefix_mean: float, suffix_mean: float, band_mean: float) -> List[str]:
    flags = []
    if prefix_mean >= 0.75:
        flags.append("strong_start_repeat")
    elif prefix_mean >= 0.55:
        flags.append("possible_start_repeat")
    if suffix_mean >= 0.75:
        flags.append("strong_end_repeat")
    elif suffix_mean >= 0.55:
        flags.append("possible_end_repeat")
    if band_mean >= 0.92:
        flags.append("strong_high_band_repeat")
    elif band_mean >= 0.82:
        flags.append("possible_high_band_repeat")
    if not flags:
        flags.append("no_strong_shared_pattern_detected")
    return flags


def main() -> int:
    args = parse_args()
    files = iter_audio_files(args.inputs)
    if len(files) < 2:
        print("Need at least 2 audio files.")
        return 1

    lag_samples = max(1, int(args.lag_sec * args.sample_rate))
    records = []
    for path in files:
        audio = load_audio(path, args.sample_rate, args.max_duration_sec)
        prefix = take_prefix(audio, args.sample_rate, args.prefix_sec)
        suffix = take_suffix(audio, args.sample_rate, args.suffix_sec)
        high_band = band_limited(audio, args.sample_rate, args.highpass_hz)
        fingerprint = spectral_fingerprint(
            high_band,
            args.sample_rate,
            args.n_fft,
            args.hop_length,
            args.highpass_hz,
        )
        records.append(
            {
                "path": path.as_posix(),
                "samples": int(len(audio)),
                "duration_sec": round(len(audio) / args.sample_rate, 3),
                "prefix": prefix,
                "suffix": suffix,
                "high_band": high_band,
                "fingerprint": fingerprint,
            }
        )

    pair_reports = []
    prefix_scores = []
    suffix_scores = []
    high_band_scores = []
    for left, right in combinations(records, 2):
        prefix_corr = max_corr_with_lag(left["prefix"], right["prefix"], lag_samples)
        suffix_corr = max_corr_with_lag(left["suffix"], right["suffix"], lag_samples)
        high_band_corr = max_corr_with_lag(left["high_band"], right["high_band"], lag_samples)
        fingerprint_cos = cosine_similarity(left["fingerprint"], right["fingerprint"])
        prefix_scores.append(prefix_corr)
        suffix_scores.append(suffix_corr)
        high_band_scores.append(fingerprint_cos)
        pair_reports.append(
            {
                "pair": [left["path"], right["path"]],
                "prefix_corr": round(prefix_corr, 4),
                "suffix_corr": round(suffix_corr, 4),
                "high_band_wave_corr": round(high_band_corr, 4),
                "high_band_spectrum_cosine": round(fingerprint_cos, 4),
            }
        )

    prefix_mean = float(np.mean(prefix_scores)) if prefix_scores else 0.0
    suffix_mean = float(np.mean(suffix_scores)) if suffix_scores else 0.0
    band_mean = float(np.mean(high_band_scores)) if high_band_scores else 0.0
    flags = heuristic_flags(prefix_mean, suffix_mean, band_mean)

    report = {
        "settings": {
            "sample_rate": args.sample_rate,
            "prefix_sec": args.prefix_sec,
            "suffix_sec": args.suffix_sec,
            "max_duration_sec": args.max_duration_sec,
            "highpass_hz": args.highpass_hz,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "lag_sec": args.lag_sec,
        },
        "files": [
            {
                "path": record["path"],
                "duration_sec": record["duration_sec"],
                "samples": record["samples"],
            }
            for record in records
        ],
        "summary": {
            "pair_count": len(pair_reports),
            "prefix_corr": summarize_scores(prefix_scores),
            "suffix_corr": summarize_scores(suffix_scores),
            "high_band_spectrum_cosine": summarize_scores(high_band_scores),
            "flags": flags,
        },
        "pairs": pair_reports,
    }

    print("Shared watermark-like pattern check")
    print(f"Files analyzed: {len(records)}")
    print(f"Pair comparisons: {len(pair_reports)}")
    print("")
    print("Summary")
    print(f"  Prefix correlation: {report['summary']['prefix_corr']}")
    print(f"  Suffix correlation: {report['summary']['suffix_corr']}")
    print(f"  High-band spectrum cosine: {report['summary']['high_band_spectrum_cosine']}")
    print(f"  Heuristic flags: {', '.join(flags)}")
    print("")
    print("Per-pair details")
    for item in pair_reports:
        left = Path(item["pair"][0]).name
        right = Path(item["pair"][1]).name
        print(
            f"  {left} <-> {right}: "
            f"prefix={item['prefix_corr']}, "
            f"suffix={item['suffix_corr']}, "
            f"high_band_wave={item['high_band_wave_corr']}, "
            f"high_band_spec={item['high_band_spectrum_cosine']}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("")
        print(f"Saved JSON report to {args.json_out.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
