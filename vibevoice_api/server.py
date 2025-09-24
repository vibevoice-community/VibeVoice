from __future__ import annotations

import argparse
import logging
import os
import secrets
from typing import Optional

import uuid
import time
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from vibevoice_api.config import CONFIG
from vibevoice_api.tts_engine import synthesize, synthesize_stream_pcm
from vibevoice_api import auth, observability as obs
from vibevoice_api.voice_map import VoiceMapper
import vibevoice_api.config as config_mod


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
                    import re as _re
                    m = _re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", s)
                    if not m:
                        continue
                    key, val = m.group(1), m.group(2)
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ.setdefault(key, val)
        except Exception:
            pass

    # load from CWD and repo root if present
    _load(os.path.abspath('.env'))
    # repo root relative to this file
    _load(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env')))


_load_dotenv_if_present()
CONFIG = config_mod.ServerConfig()


def _normalize_base_path(raw: str | None) -> str:
    if not raw:
        return ""
    path = raw.strip()
    if not path or path == "/":
        return ""
    if not path.startswith("/"):
        path = "/" + path
    while len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return path


API_PREFIX = _normalize_base_path(CONFIG.base_path)


def _join_with_base(path: str) -> str:
    if not path:
        return API_PREFIX or "/"
    if not path.startswith("/"):
        path = "/" + path
    if not API_PREFIX:
        return path
    if path == "/":
        return API_PREFIX
    return f"{API_PREFIX}{path}"


def _normalize_request_path(path: str) -> str:
    if not path:
        return "/"
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return path or "/"


_OPEN_PATHS_RAW = {"/", "/health", "/metrics", "/favicon.ico"}
_OPEN_PATHS = {_normalize_request_path(p) for p in _OPEN_PATHS_RAW}
_OPEN_PATHS.update(_normalize_request_path(_join_with_base(p)) for p in _OPEN_PATHS_RAW)

_ADMIN_KEYS_PREFIX = _normalize_request_path(_join_with_base("/admin/keys"))


def _is_admin_path(path: str) -> bool:
    if not _ADMIN_KEYS_PREFIX or _ADMIN_KEYS_PREFIX == "/":
        return False
    if path == _ADMIN_KEYS_PREFIX:
        return True
    return path.startswith(f"{_ADMIN_KEYS_PREFIX}/")

log = logging.getLogger("vibevoice_api")


class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rid = obs.get_request_id()
        except Exception:
            rid = "-"
        record.request_id = rid
        return True


def _configure_logging() -> None:
    if getattr(_configure_logging, "_done", False):
        return
    handler = logging.StreamHandler()
    handler.addFilter(RequestIDFilter())
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
    handler.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.handlers = [handler]
    _configure_logging._done = True  # type: ignore


_configure_logging()

router = APIRouter(prefix=API_PREFIX or "")
admin_router = APIRouter(prefix=_ADMIN_KEYS_PREFIX)

app = FastAPI(title="VibeVoice OpenAI-Compatible Audio API")

# Allow CORS for browser-based clients (dev-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static web assets (web/)
_here = os.path.abspath(os.path.dirname(__file__))
_web_dir_candidates = [
    os.path.abspath(os.path.join(_here, "..", "web")),
    os.path.abspath(os.path.join(os.getcwd(), "web")),
]
_mounted_static_paths: set[str] = set()
for _p in _web_dir_candidates:
    if os.path.isdir(_p):
        mount_points = [(_join_with_base("/web"), "web")]
        if mount_points[0][0] != "/web":
            mount_points.append(("/web", "web-legacy"))
        for _mount_path, _mount_name in mount_points:
            if _mount_path in _mounted_static_paths:
                continue
            app.mount(_mount_path, StaticFiles(directory=_p), name=_mount_name)
            _mounted_static_paths.add(_mount_path)
        break

# Startup info
log.info(
    "Startup config: base_path=%s, require_api_key=%s, admin_token_configured=%s, logs_dir=%s",
    API_PREFIX or "/",
    str(CONFIG.require_api_key),
    "yes" if CONFIG.admin_token else "no",
    CONFIG.logs_dir,
)


class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to generate audio for")
    model: str = Field(..., description="Model ID or path; ignored if not VibeVoice")
    voice: str = Field("", description="Voice name to use (mapped to demo/voices)")
    # Optional customizations
    voice_path: Optional[str] = Field(None, description="Absolute/relative file path to a reference voice sample (wav/mp3/etc.)")
    voice_data: Optional[str] = Field(
        None,
        description="Base64 string or data URL (data:audio/...;base64,...) containing reference voice audio",
    )
    instructions: Optional[str] = Field(None, description="Additional style instructions (unused)")
    response_format: str = Field("wav", description="Audio format: wav/pcm (native) or mp3/flac/opus/aac (requires ffmpeg)")
    speed: Optional[float] = Field(None, description="Playback speed multiplier")
    stream_format: Optional[str] = Field(None, description="sse or audio (unsupported)")
    # Advanced hyperparameters / strategies
    cfg_scale: Optional[float] = Field(None, description="Classifier-free guidance scale")
    ddpm_steps: Optional[int] = Field(None, description="Diffusion inference steps (per request)")
    instructions_strategy: Optional[str] = Field(None, description="system_only | preprompt_only | system_and_preprompt")
    instructions_repeat: Optional[int] = Field(None, description="Repeat times for preprompt injection")
    # Multi-speaker support (in order): each entry can be an alias, a file path, or a data URL
    speakers: Optional[list[str]] = Field(None, description="List of voices (alias/path/dataURL) for Speaker 1..N")


@router.get("/")
def root() -> JSONResponse:
    return JSONResponse({"name": "vibevoice_api", "version": "0.1.0"})


@router.get("/health")
def health() -> PlainTextResponse:
    return PlainTextResponse("ok")


@router.get("/metrics")
def metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@router.get("/voices/aliases")
def voices_aliases() -> JSONResponse:
    mapper = VoiceMapper(os.getcwd())
    avail = mapper.available()
    names = sorted(avail.keys())
    return JSONResponse({"aliases": names, "count": len(names)})


@router.get("/config/ffmpeg")
def config_ffmpeg() -> JSONResponse:
    cfg = {
        "ffmpeg_path": CONFIG.ffmpeg_path,
        "bitrate": CONFIG.ffmpeg_bitrate,
        "opus_container": CONFIG.opus_container,
        "opus_vbr_mode": CONFIG.opus_vbr_mode,
        "opus_application": CONFIG.opus_application,
        "opus_frame_duration": CONFIG.opus_frame_duration,
        "aac_profile": CONFIG.aac_profile,
        "aac_mode": CONFIG.aac_mode,
        "aac_q": CONFIG.aac_q,
        "sse_chunk_bytes": CONFIG.sse_chunk_bytes,
    }
    return JSONResponse(cfg)


class AdminKeyCreateRequest(BaseModel):
    key: Optional[str] = Field(
        None, description="Existing API key to persist. If omitted, a new key is generated.")
    prefix: Optional[str] = Field(
        "sk-", description="Prefix used when generating a new API key if none is supplied.")


def _require_admin_auth(request: Request) -> Optional[JSONResponse]:
    token = (CONFIG.admin_token or "").strip()
    if not token:
        return JSONResponse(
            status_code=403,
            content={"error": {"message": "Admin token not configured", "type": "admin_disabled"}},
        )
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Admin token required", "type": "invalid_admin_token"}},
            headers={"WWW-Authenticate": "Bearer"},
        )
    provided = auth_header.split(" ", 1)[1].strip()
    if not provided or not secrets.compare_digest(provided, token):
        return JSONResponse(
            status_code=403,
            content={"error": {"message": "Invalid admin token", "type": "invalid_admin_token"}},
        )
    return None


@admin_router.get("")
def admin_list_keys(request: Request) -> JSONResponse:
    auth_error = _require_admin_auth(request)
    if auth_error:
        return auth_error
    hashes = auth.list_api_key_hashes()
    return JSONResponse({"keys": hashes, "count": len(hashes)})


@admin_router.post("", status_code=201)
def admin_create_key(request: Request, payload: Optional[AdminKeyCreateRequest] = None) -> JSONResponse:
    auth_error = _require_admin_auth(request)
    if auth_error:
        return auth_error
    body = payload or AdminKeyCreateRequest()
    key = (body.key or "").strip()
    if not key:
        prefix_value = body.prefix if body and body.prefix is not None else "sk-"
        key = auth.generate_api_key(prefix=prefix_value)
    auth.add_api_key(key)
    key_hash = auth.hash_api_key(key)
    log.info("Admin created API key hash=%s", key_hash)
    return JSONResponse({"key": key, "hash": key_hash}, status_code=201)


@admin_router.delete("/{key_hash}")
def admin_delete_key(key_hash: str, request: Request) -> JSONResponse:
    auth_error = _require_admin_auth(request)
    if auth_error:
        return auth_error
    normalized = (key_hash or "").strip().lower()
    if not normalized:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "key_hash is required", "type": "invalid_request_error"}},
        )
    removed = auth.remove_api_key(normalized, hashed=True)
    if not removed:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "API key not found", "type": "invalid_request_error"}},
        )
    log.info("Admin revoked API key hash=%s", normalized)
    return JSONResponse({"deleted": True, "hash": normalized})


@app.middleware("http")
async def metrics_and_request_id_middleware(request: Request, call_next):
    # assign request id and hints container
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    obs.set_request_id(rid)
    obs.set_hints_container()
    start = time.perf_counter()
    endpoint = request.url.path
    method = request.method
    # API key auth (skip for configured open paths)
    normalized_path = _normalize_request_path(endpoint)
    admin_path = _is_admin_path(normalized_path)
    if CONFIG.require_api_key and normalized_path not in _OPEN_PATHS and not admin_path:
        authz = request.headers.get("Authorization", "")
        if not authz.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"error": {"message": "Unauthorized", "type": "invalid_api_key"}})
        api_key = authz.split(" ", 1)[1].strip()
        if not auth.validate_api_key(api_key):
            return JSONResponse(status_code=401, content={"error": {"message": "Unauthorized", "type": "invalid_api_key"}})

    try:
        response = await call_next(request)
    except Exception as e:
        obs.ERRORS_TOTAL.labels(type=type(e).__name__).inc()
        log.exception("unhandled exception")
        raise
    finally:
        try:
            obs.REQUEST_LATENCY.labels(endpoint, method).observe(time.perf_counter() - start)
        except Exception:
            pass
    # response headers + counters
    response.headers["X-Request-ID"] = rid
    hints = obs.get_hints()
    if hints:
        response.headers["X-Hints"] = " | ".join(hints[:6])
        log.info(f"hints: {hints}")
    try:
        obs.REQUEST_COUNT.labels(endpoint, method, str(response.status_code)).inc()
    except Exception:
        pass
    return response


def _speech_impl(req: SpeechRequest, base_dir: str, endpoint_path: str):
    if not req.input or not isinstance(req.input, str):
        raise HTTPException(status_code=400, detail="'input' must be a non-empty string")

    # Supported formats
    fmt = (req.response_format or "wav").lower()
    allowed = {"wav", "pcm", "mp3", "opus", "aac"}
    if fmt not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported response_format: {req.response_format}")

    speed = None
    speed_clamped = False
    if req.speed is not None:
        try:
            s = float(req.speed)
            if s < 0.25:
                speed_clamped = True
                s = 0.25
            elif s > 4.0:
                speed_clamped = True
                s = 4.0
            speed = s
            if speed_clamped:
                obs.add_hint(f"speed_clamp:{req.speed}->{speed}")
        except Exception:
            speed = None

    # Binary audio streaming path via ffmpeg (chunked transfer)
    if (req.stream_format or "").lower() == "audio" and fmt in {"mp3", "opus", "aac"}:
        from vibevoice_api.audio_utils import ffmpeg_stream_cmd
        import asyncio as _asyncio

        cmd, content_type = ffmpeg_stream_cmd(fmt, CONFIG.sample_rate)
        if not cmd:
            raise HTTPException(status_code=400, detail=f"Unsupported streaming format: {fmt}")

        async def bin_gen():
            # Spawn ffmpeg
            proc = await _asyncio.create_subprocess_exec(
                *cmd,
                stdin=_asyncio.subprocess.PIPE,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )

            async def feeder():
                from vibevoice_api.audio_utils import float_to_pcm16
                async for chunk in synthesize_stream_pcm(
                    root_dir=base_dir,
                    text=req.input,
                    voice=req.voice,
                    voice_path=req.voice_path,
                    voice_data_b64=req.voice_data,
                    model_path=req.model or CONFIG.model_path,
                    speed=speed,
                    instructions=req.instructions,
                    cfg_scale=req.cfg_scale,
                    ddpm_steps=req.ddpm_steps,
                    instructions_strategy=req.instructions_strategy,
                    instructions_repeat=req.instructions_repeat,
                ):
                    pcm = float_to_pcm16(chunk)
                    proc.stdin.write(pcm)
                    try:
                        await proc.stdin.drain()
                    except Exception:
                        break
                try:
                    proc.stdin.close()
                except Exception:
                    pass

            feed_task = _asyncio.create_task(feeder())

            try:
                while True:
                    data = await proc.stdout.read(16384)
                    if not data:
                        break
                    yield data
            finally:
                try:
                    await feed_task
                except Exception:
                    pass
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    await proc.wait()
                except Exception:
                    pass

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Model": req.model or CONFIG.model_path,
            "X-Sample-Rate": str(CONFIG.sample_rate),
            "X-Stream-Format": "audio",
        }
        return StreamingResponse(bin_gen(), media_type=content_type, headers=headers)

    # SSE streaming path
    if (req.stream_format or "").lower() == "sse":
        import json as _json
        import base64 as _b64
        from vibevoice_api.audio_utils import float_to_pcm16

        async def sse_gen():
            # start event with meta
            meta = {"format": "pcm", "sample_rate": CONFIG.sample_rate}
            yield f"event: start\ndata: {_json.dumps(meta)}\n\n"
            try:
                obs.add_hint("sse_pcm")
            except Exception:
                pass
            # stream chunks
            # aggregate PCM bytes to reduce overhead (configurable)
            pcm_buffer = bytearray()
            agg_bytes = max(1024, int(getattr(CONFIG, 'sse_chunk_bytes', 16384) or 16384))
            async for chunk in synthesize_stream_pcm(
                root_dir=base_dir,
                text=req.input,
                voice=req.voice,
                voice_path=req.voice_path,
                voice_data_b64=req.voice_data,
                speakers=req.speakers,
                model_path=req.model or CONFIG.model_path,
                speed=speed,
                instructions=req.instructions,
                cfg_scale=req.cfg_scale,
                ddpm_steps=req.ddpm_steps,
                instructions_strategy=req.instructions_strategy,
                instructions_repeat=req.instructions_repeat,
            ):
                try:
                    pcm = float_to_pcm16(chunk)
                    pcm_buffer.extend(pcm)
                    if len(pcm_buffer) >= agg_bytes:
                        b64 = _b64.b64encode(pcm_buffer).decode()
                        payload = {"type": "audio_chunk", "format": "pcm", "data": b64}
                        yield f"event: chunk\ndata: {_json.dumps(payload)}\n\n"
                        pcm_buffer.clear()
                except Exception as e:
                    # send an error event and stop
                    err = {"error": str(e)}
                    yield f"event: error\ndata: {_json.dumps(err)}\n\n"
                    break
            # end event
            if pcm_buffer:
                b64 = _b64.b64encode(pcm_buffer).decode()
                payload = {"type": "audio_chunk", "format": "pcm", "data": b64}
                yield f"event: chunk\ndata: {_json.dumps(payload)}\n\n"
            yield "event: end\ndata: {}\n\n"

        # persist request log (no prompt body change; same policy)
        try:
            prompt = req.input if CONFIG.log_prompts else None
            if prompt is not None and len(prompt) > CONFIG.prompt_maxlen:
                prompt = prompt[: CONFIG.prompt_maxlen]
            instructions_log = req.instructions if CONFIG.log_prompts else None
            if instructions_log is not None and len(instructions_log) > CONFIG.prompt_maxlen:
                instructions_log = instructions_log[: CONFIG.prompt_maxlen]
            obs.append_json_log(
                "requests.log",
                {
                    "rid": obs.get_request_id(),
                    "endpoint": endpoint_path,
                    "model": req.model,
                    "format": "pcm",
                    "speed": speed,
                    "voice": req.voice,
                    "voice_path": req.voice_path,
                    "has_voice_data": bool(req.voice_data),
                    "prompt": prompt,
                    "instructions": instructions_log,
                    "stream_format": "sse",
                },
            )
            hints = obs.get_hints()
            if hints:
                obs.append_json_log("hints.log", {"rid": obs.get_request_id(), "hints": hints})
        except Exception:
            pass

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Model": req.model or CONFIG.model_path,
            "X-Sample-Rate": str(CONFIG.sample_rate),
            "X-Stream-Format": "sse",
        }
        # attach hints header if any
        try:
            hints = obs.get_hints()
            if hints:
                headers["X-Hints"] = " | ".join(hints[:6])
        except Exception:
            pass
        return StreamingResponse(sse_gen(), media_type="text/event-stream", headers=headers)

    try:
        data, content_type = synthesize(
            root_dir=base_dir,
            text=req.input,
            voice=req.voice,
            voice_path=req.voice_path,
            voice_data_b64=req.voice_data,
            speakers=req.speakers,
            model_path=req.model or CONFIG.model_path,
            response_format=fmt,
            speed=speed,
            instructions=req.instructions,
            cfg_scale=req.cfg_scale,
            ddpm_steps=req.ddpm_steps,
            instructions_strategy=req.instructions_strategy,
            instructions_repeat=req.instructions_repeat,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"synthesis_error: {type(e).__name__}: {e}")

    # per openai-python: Accept: application/octet-stream; return raw bytes
    headers = {
        "Content-Type": content_type,
        "X-Model": req.model or CONFIG.model_path,
        "X-Sample-Rate": str(CONFIG.sample_rate),
    }
    # Preserve client's requested stream_format in headers for transparency
    if req.stream_format:
        headers["X-Stream-Format"] = req.stream_format
        try:
            from vibevoice_api import observability as obs
            obs.add_hint(f"stream_format_ignored:{req.stream_format}")
        except Exception:
            pass
    if speed_clamped:
        headers["X-Speed-Clamped"] = "1"
    # Persist request log with prompt & hints (if enabled)
    try:
        prompt = req.input if CONFIG.log_prompts else None
        if prompt is not None and len(prompt) > CONFIG.prompt_maxlen:
            prompt = prompt[: CONFIG.prompt_maxlen]
        instructions_log = req.instructions if CONFIG.log_prompts else None
        if instructions_log is not None and len(instructions_log) > CONFIG.prompt_maxlen:
            instructions_log = instructions_log[: CONFIG.prompt_maxlen]
        obs.append_json_log(
            "requests.log",
            {
                "rid": obs.get_request_id(),
                "endpoint": endpoint_path,
                "model": req.model,
                "format": fmt,
                "speed": speed,
                "voice": req.voice,
                "voice_path": req.voice_path,
                "has_voice_data": bool(req.voice_data),
                "prompt": prompt,
                "instructions": instructions_log,
            },
        )
        hints = obs.get_hints()
        if hints:
            obs.append_json_log(
                "hints.log",
                {"rid": obs.get_request_id(), "hints": hints},
            )
    except Exception:
        pass

    return StreamingResponse(iter([data]), media_type="application/octet-stream", headers=headers)


@router.post("/audio/speech")
def audio_speech(req: SpeechRequest, request: Request):
    base_dir = os.getcwd()
    return _speech_impl(req, base_dir, request.url.path)


# Note: STT endpoints intentionally not provided.


def _register_legacy_aliases(application: FastAPI) -> None:
    if not API_PREFIX:
        return
    legacy_routes = [
        ("/", root, ["GET"]),
        ("/health", health, ["GET"]),
        ("/metrics", metrics, ["GET"]),
        ("/voices/aliases", voices_aliases, ["GET"]),
        ("/config/ffmpeg", config_ffmpeg, ["GET"]),
        ("/audio/speech", audio_speech, ["POST"]),
    ]
    for path, handler, methods in legacy_routes:
        application.add_api_route(path, handler, methods=methods, include_in_schema=False)


app.include_router(router)
app.include_router(admin_router)
_register_legacy_aliases(app)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run VibeVoice OpenAI-compatible Audio API server")
    parser.add_argument("--host", default=CONFIG.host)
    parser.add_argument("--port", type=int, default=CONFIG.port)
    parser.add_argument("--model_path", default=CONFIG.model_path, help="HF path or local path")
    args = parser.parse_args(argv)

    # export to env so the loader uses it, if provided
    if args.model_path:
        os.environ.setdefault("VIBEVOICE_MODEL", args.model_path)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
