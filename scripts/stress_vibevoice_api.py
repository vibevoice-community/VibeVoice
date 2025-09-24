#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import os
import random
import string
import sys
import time
from typing import Optional, Dict, Any
import re


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_openai_client_on_path() -> None:
    # No longer add local openai-python; rely on pip-installed openai package
    return


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


def _random_text(n: int = 80) -> str:
    words = ["hello", "vibe", "voice", "podcast", "sample", "test", "demo", "music", "news", "science"]
    s = []
    for _ in range(n // 6):
        s.append(random.choice(words))
    return " ".join(s).capitalize() + "."


def _one_request(client, model_path: str, voices_dir: str, formats: list[str]) -> Dict[str, Any]:
    # Randomly pick a mode
    mode = random.choice(["name", "path", "data"])
    voice_files = [f for f in os.listdir(voices_dir) if f.lower().endswith(".wav")]
    voice_file = os.path.join(voices_dir, random.choice(voice_files)) if voice_files else None

    kwargs = dict(
        model=model_path,
        input=_random_text(120),
        response_format=random.choice(formats),
        speed=random.choice([0.75, 1.0, 1.25, 1.5, 2.0]),
    )

    if mode == "name":
        kwargs["voice"] = "Alice"
    elif mode == "path" and voice_file:
        kwargs["voice"] = "ignored"
        kwargs["extra_body"] = {"voice_path": voice_file}
    elif mode == "data" and voice_file:
        with open(voice_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        kwargs["voice"] = "ignored"
        kwargs["extra_body"] = {"voice_data": f"data:audio/wav;base64,{b64}"}
    else:
        kwargs["voice"] = "Alice"

    t0 = time.perf_counter()
    try:
        speech = client.audio.speech.create(**kwargs)
        data = speech.read()
        dur = time.perf_counter() - t0
        return {"ok": True, "bytes": len(data), "latency_s": dur, "fmt": kwargs["response_format"], "mode": mode}
    except Exception as e:
        dur = time.perf_counter() - t0
        return {"ok": False, "error": str(e), "latency_s": dur, "fmt": kwargs["response_format"], "mode": mode}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stress test VibeVoice API with concurrent requests")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000")
    parser.add_argument("--model_path", default="vibevoice/VibeVoice-1.5B")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--include_compressed", action="store_true")
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args(argv)

    _load_dotenv_if_present()
    _ensure_openai_client_on_path()
    from openai import OpenAI

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "sk-test"
    client = OpenAI(base_url=args.base_url.rstrip("/"), api_key=api_key)
    voices_dir = os.path.join(_repo_root(), "demo", "voices")

    formats = ["wav", "pcm"]
    if args.include_compressed:
        formats += ["mp3", "opus", "aac"]  # flac removed

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(_one_request, client, args.model_path, voices_dir, formats) for _ in range(args.requests)]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    ok = sum(1 for r in results if r.get("ok"))
    total = len(results)
    latencies = [r["latency_s"] for r in results if r.get("ok")]
    avg = sum(latencies) / len(latencies) if latencies else 0.0
    p95 = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0

    print(f"ok={ok}/{total} avg={avg:.2f}s p95={p95:.2f}s")
    errors = [r for r in results if not r.get("ok")]
    for e in errors[:5]:
        print("ERR:", e)

    return 0 if ok == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
