#!/usr/bin/env python3
"""
End-to-end tests for VibeVoice OpenAI-compatible Audio API.

Covers:
- GET /health
- GET /
- POST /audio/speech  (wav, voice name)
- POST /audio/speech  (wav, voice_path)
- POST /audio/speech  (wav, voice_data base64)
- POST /v1/audio/speech (mp3, if ffmpeg available)

This script can optionally start the server, wait for readiness, run tests,
and write outputs under outputs/api_test/.

Usage:
  python scripts/test_vibevoice_api.py --start --model_path vibevoice/VibeVoice-1.5B --port 8000

Requires:
- This repo (so it can import openai-python/src if not installed)
- ffmpeg (optional) for non-wav/pcm encodings
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple
import re


LOG = logging.getLogger("api_test")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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
                    # strip surrounding quotes if any
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ.setdefault(key, val)
        except Exception:
            pass

    # load from CWD and repo root if present
    _load(os.path.abspath('.env'))
    _load(os.path.join(_repo_root(), '.env'))


def _http_get(url: str, timeout: float = 2.0) -> Tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            code = resp.getcode()
            data = resp.read()
            return code, data
    except Exception as e:
        return -1, str(e).encode()


def _wait_for_health(base_url: str, timeout_s: float = 120.0) -> bool:
    url = base_url.rstrip("/").replace("/v1", "") + "/health"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        code, data = _http_get(url, timeout=2.0)
        if code == 200 and data.strip() == b"ok":
            return True
        time.sleep(1.0)
    return False


@dataclass
class TestResult:
    name: str
    ok: bool
    detail: str = ""
    artifact: Optional[str] = None


def _save_artifact(path: str, data: bytes) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def run_tests(base_url: str, model_path: str, out_dir: str, voices_dir: str, api_key: str | None) -> list[TestResult]:
    results: list[TestResult] = []

    # Prepare OpenAI client from pip
    _ensure_openai_client_on_path()
    try:
        from openai import OpenAI
    except Exception as e:
        results.append(TestResult("import_openai", False, f"Please pip install openai: {e}"))
        return results

    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "sk-test")

    # GET /health
    code, data = _http_get(base_url.rstrip("/").replace("/v1", "") + "/health")
    results.append(TestResult("GET /health", code == 200 and data.strip() == b"ok", f"code={code}, data={data[:64]!r}"))

    # GET /
    code, data = _http_get(base_url.rstrip("/").replace("/v1", ""))
    try:
        payload = json.loads(data.decode()) if code == 200 else {}
    except Exception:
        payload = {}
    results.append(TestResult("GET /", code == 200 and "name" in payload, f"code={code}, payload_keys={list(payload.keys())}"))

    # Locate a demo voice
    demo_voice = os.path.join(voices_dir, "en-Alice_woman.wav")
    if not os.path.exists(demo_voice):
        # pick first wav
        wavs = [f for f in os.listdir(voices_dir) if f.lower().endswith(".wav")]
        if wavs:
            demo_voice = os.path.join(voices_dir, wavs[0])

    # 1) POST /audio/speech wav with voice name
    try:
        speech = client.audio.speech.create(
            model=model_path,
            voice="Alice",
            input="Hello, 我的主人需要我做什麼呢?",
            response_format="wav",
        )
        data = speech.read()
        ok = isinstance(data, (bytes, bytearray)) and len(data) > 1000
        art = _save_artifact(os.path.join(out_dir, "tts_wav_voice_name.wav"), data) if ok else None
        results.append(TestResult("POST /audio/speech wav name", ok, detail=f"bytes={len(data)}", artifact=art))
    except Exception as e:
        results.append(TestResult("POST /audio/speech wav name", False, detail=str(e)))

    # 2) POST /audio/speech wav with voice_path
    try:
        extra_body = {"voice_path": demo_voice}
        speech = client.audio.speech.create(
            model=model_path,
            voice="ignored-when-voice_path",
            input="Testing voice path with direct file.",
            response_format="wav",
            extra_body=extra_body,
        )
        data = speech.read()
        ok = isinstance(data, (bytes, bytearray)) and len(data) > 1000
        art = _save_artifact(os.path.join(out_dir, "tts_wav_voice_path.wav"), data) if ok else None
        results.append(TestResult("POST /audio/speech wav voice_path", ok, detail=f"bytes={len(data)}", artifact=art))
    except Exception as e:
        results.append(TestResult("POST /audio/speech wav voice_path", False, detail=str(e)))

    # 3) POST /audio/speech wav with voice_data (base64)
    try:
        with open(demo_voice, "rb") as f:
            vb64 = base64.b64encode(f.read()).decode()
        extra_body = {"voice_data": f"data:audio/wav;base64,{vb64}"}
        speech = client.audio.speech.create(
            model=model_path,
            voice="ignored-when-voice_data",
            input="Testing voice_data upload.",
            response_format="wav",
            extra_body=extra_body,
        )
        data = speech.read()
        ok = isinstance(data, (bytes, bytearray)) and len(data) > 1000
        art = _save_artifact(os.path.join(out_dir, "tts_wav_voice_data.wav"), data) if ok else None
        results.append(TestResult("POST /audio/speech wav voice_data", ok, detail=f"bytes={len(data)}", artifact=art))
    except Exception as e:
        results.append(TestResult("POST /audio/speech wav voice_data", False, detail=str(e)))

    # 4) POST /audio/speech pcm with speed
    try:
        speech = client.audio.speech.create(
            model=model_path,
            voice="Alice",
            input="Testing PCM with speed 1.5.",
            response_format="pcm",
            speed=1.5,
        )
        data = speech.read()
        ok = isinstance(data, (bytes, bytearray)) and len(data) > 1000
        art = _save_artifact(os.path.join(out_dir, "tts_pcm_speed15.pcm"), data) if ok else None
        results.append(TestResult("POST /audio/speech pcm speed", ok, detail=f"bytes={len(data)}", artifact=art))
    except Exception as e:
        results.append(TestResult("POST /audio/speech pcm speed", False, detail=str(e)))

    # 5) POST /v1/audio/speech mp3 (if ffmpeg available)
    ffmpeg = os.environ.get("VIBEVOICE_FFMPEG", "ffmpeg")
    if shutil.which(ffmpeg):
        try:
            client_v1 = OpenAI(base_url=base_url.rstrip("/") + "/v1" if not base_url.rstrip("/").endswith("/v1") else base_url.rstrip("/"), api_key=api_key or "sk-test")
            speech = client_v1.audio.speech.create(
                model=model_path,
                voice="Alice",
                input="Testing mp3 via ffmpeg.",
                response_format="mp3",
            )
            data = speech.read()
            ok = isinstance(data, (bytes, bytearray)) and len(data) > 1000
            art = _save_artifact(os.path.join(out_dir, "tts_mp3_voice_name.mp3"), data) if ok else None
            results.append(TestResult("POST /v1/audio/speech mp3", ok, detail=f"bytes={len(data)}", artifact=art))
        except Exception as e:
            results.append(TestResult("POST /v1/audio/speech mp3", False, detail=str(e)))
    else:
        results.append(TestResult("POST /v1/audio/speech mp3", True, detail="skipped: ffmpeg not found"))

    return results


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="VibeVoice API end-to-end test script")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base_url", default=None, help="Override full base URL; default http://host:port")
    parser.add_argument("--model_path", default="vibevoice/VibeVoice-1.5B")
    parser.add_argument("--start", action="store_true", help="Start server as subprocess")
    parser.add_argument("--timeout", type=float, default=600.0, help="Max time for end-to-end run")
    parser.add_argument("--api_key", default=None, help="API key (Bearer) for the server")
    args = parser.parse_args(argv)

    _load_dotenv_if_present()
    root = _repo_root()
    voices_dir = os.path.join(root, "demo", "voices")
    out_dir = os.path.join(root, "outputs", "api_test")

    base_url = args.base_url or f"http://{args.host}:{args.port}"

    proc = None
    try:
        if args.start:
            LOG.info("Starting server...")
            env = os.environ.copy()
            # pass through model path
            cmd = [sys.executable, "-m", "vibevoice_api.server", "--model_path", args.model_path, "--port", str(args.port), "--host", args.host]
            proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # wait for health
            if not _wait_for_health(base_url, timeout_s=min(120.0, args.timeout)):
                LOG.error("Server did not become healthy in time")
                if proc:
                    proc.terminate()
                return 1

        LOG.info("Running endpoint tests...")
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        results = run_tests(base_url, args.model_path, out_dir, voices_dir, api_key)

        # Report
        ok_count = sum(1 for r in results if r.ok)
        for r in results:
            status = "PASS" if r.ok else "FAIL"
            art = f" artifact={r.artifact}" if r.artifact else ""
            LOG.info(f"[{status}] {r.name}: {r.detail}{art}")

        all_ok = all(r.ok for r in results)
        return 0 if all_ok else 2
    finally:
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
