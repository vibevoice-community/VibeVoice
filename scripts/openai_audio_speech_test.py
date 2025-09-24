#!/usr/bin/env python3
"""
Simple test script using the official openai-python client to call the local
VibeVoice OpenAI-compatible audio API and save the output audio.

Examples:
  python scripts/openai_audio_speech_test.py \
    --base_url http://127.0.0.1:8000/v1 \
    --model_path "F:/VibeVoice-Large" \
    --voice Alice \
    --text "Hello from VibeVoice!" \
    --format wav \
    --out outputs/openai_test/out.wav

  python scripts/openai_audio_speech_test.py \
    --base_url http://127.0.0.1:8000/v1 \
    --model_path "F:/VibeVoice-Large" \
    --voice ignored \
    --voice_path demo/voices/en-Alice_woman.wav \
    --text "Custom reference voice" \
    --format mp3 \
    --out outputs/openai_test/out.mp3

  python scripts/openai_audio_speech_test.py \
    --base_url http://127.0.0.1:8000/v1 \
    --model_path vibevoice/VibeVoice-1.5B \
    --voice ignored \
    --voice_data_path demo/voices/en-Alice_woman.wav \
    --text "Voice via data URL" \
    --format flac \
    --out outputs/openai_test/out.flac

Notes:
  - API auth is disabled by default in this repo. If you enable it, set
    OPENAI_API_KEY in .env or pass --api_key.
  - For mp3/flac/opus/aac you need ffmpeg available to the server.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
from typing import Optional


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_openai_on_path() -> None:
    root = _repo_root()
    src = os.path.join(root, "openai-python", "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def _load_dotenv_if_present() -> None:
    def _load(path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", s)
                    if not m:
                        continue
                    key, val = m.group(1), m.group(2)
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ.setdefault(key, val)
        except Exception:
            pass

    _load(os.path.abspath('.env'))
    _load(os.path.join(_repo_root(), '.env'))


def _read_file_b64(path: str, mime: str = "audio/wav") -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def main(argv: Optional[list[str]] = None) -> int:
    _load_dotenv_if_present()
    _ensure_openai_on_path()

    parser = argparse.ArgumentParser(description="Call local VibeVoice API via official openai-python client")
    parser.add_argument("--base_url", default=os.environ.get("VIBEVOICE_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "sk-test"))
    parser.add_argument("--model_path", default=os.environ.get("VIBEVOICE_MODEL", "vibevoice/VibeVoice-1.5B"))
    parser.add_argument("--voice", default="Alice")
    parser.add_argument("--speakers", nargs="*", default=None, help="Multi-speaker list: alias/path/dataURL for Speaker 1..N")
    parser.add_argument("--voice_path", default=None)
    parser.add_argument("--voice_data_path", default=None)
    parser.add_argument("--text", default="Hello from VibeVoice using OpenAI client.")
    parser.add_argument("--format", default="wav", choices=["wav", "pcm", "mp3", "flac", "opus", "aac"])
    parser.add_argument("--instructions", default=None, help="Style/system prompt instructions for the TTS model")
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--ddpm_steps", type=int, default=None)
    parser.add_argument("--instructions_strategy", default=None, choices=["system_only","preprompt_only","system_and_preprompt"]) 
    parser.add_argument("--instructions_repeat", type=int, default=None)
    parser.add_argument("--out", default=os.path.join("outputs", "openai_test", "out.wav"))
    args = parser.parse_args(argv)

    # Lazy import after sys.path injection
    from openai import OpenAI

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    client = OpenAI(base_url=args.base_url.rstrip("/"), api_key=args.api_key)

    extra_body = {}
    if args.voice_path:
        extra_body["voice_path"] = args.voice_path
    if args.voice_data_path:
        # best-effort MIME guess
        mime = "audio/wav"
        ext = os.path.splitext(args.voice_data_path)[1].lower()
        if ext in {".mp3"}: mime = "audio/mpeg"
        elif ext in {".flac"}: mime = "audio/flac"
        elif ext in {".ogg", ".opus"}: mime = "audio/ogg"
        elif ext in {".aac", ".m4a"}: mime = "audio/aac"
        extra_body["voice_data"] = _read_file_b64(args.voice_data_path, mime=mime)

    if args.speakers:
        extra_body["speakers"] = args.speakers
    if args.cfg_scale is not None:
        extra_body["cfg_scale"] = args.cfg_scale
    if args.ddpm_steps is not None:
        extra_body["ddpm_steps"] = args.ddpm_steps
    if args.instructions_strategy:
        extra_body["instructions_strategy"] = args.instructions_strategy
    if args.instructions_repeat is not None:
        extra_body["instructions_repeat"] = args.instructions_repeat

    print("Calling audio.speech.create ...")
    speech = client.audio.speech.create(
        model=args.model_path,
        voice=args.voice,
        input=args.text,
        response_format=args.format,
        instructions=args.instructions if args.instructions else None,
        speed=args.speed if args.speed is not None else None,
        extra_body=extra_body if extra_body else None,
    )

    data = speech.read()
    with open(args.out, "wb") as f:
        f.write(data)
    print(f"Saved to {args.out} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
