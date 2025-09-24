from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerConfig:
    host: str = os.environ.get("VIBEVOICE_API_HOST", "0.0.0.0")
    port: int = int(os.environ.get("VIBEVOICE_API_PORT", "8000"))
    base_path: str = os.environ.get("VIBEVOICE_API_BASE_PATH", "/v1")
    # Path or HF ID of the model to load by default
    model_path: str = os.environ.get("VIBEVOICE_MODEL", "microsoft/VibeVoice-1.5b")
    # Device preference: auto, cuda, mps, cpu
    device_preference: str = os.environ.get("VIBEVOICE_DEVICE", "auto")
    # Number of diffusion steps for inference
    ddpm_steps: int = int(os.environ.get("VIBEVOICE_DDPM_STEPS", "10"))
    # Default audio sample rate returned by the model
    sample_rate: int = int(os.environ.get("VIBEVOICE_SAMPLE_RATE", "24000"))
    # Optional: path to ffmpeg binary for extra encodings (mp3/flac/opus/aac)
    ffmpeg_path: str = os.environ.get("VIBEVOICE_FFMPEG", "ffmpeg")
    # Validation toggle for local shards (1=on, 0=off)
    validate_shards: bool = os.environ.get("VIBEVOICE_VALIDATE_SHARDS", "1") not in {"0", "false", "False"}
    # Max concurrent inferences per loaded model
    max_concurrency: int = int(os.environ.get("VIBEVOICE_MAX_CONCURRENCY", "2"))
    # Logging directory & prompt logging
    logs_dir: str = os.environ.get("VIBEVOICE_LOG_DIR", os.path.join(os.getcwd(), "logs"))
    log_prompts: bool = os.environ.get("VIBEVOICE_LOG_PROMPTS", "1") not in {"0", "false", "False"}
    prompt_maxlen: int = int(os.environ.get("VIBEVOICE_PROMPT_MAXLEN", "4096"))
    instructions_maxlen: int = int(os.environ.get("VIBEVOICE_INSTRUCTIONS_MAXLEN", "2000"))
    # Default to system_only to prevent style text being spoken
    instructions_strategy: str = os.environ.get("VIBEVOICE_INSTRUCTIONS_STRATEGY", "system_only")
    instructions_repeat: int = int(os.environ.get("VIBEVOICE_INSTRUCTIONS_REPEAT", "1"))
    # SSE chunk aggregation bytes
    sse_chunk_bytes: int = int(os.environ.get("VIBEVOICE_SSE_CHUNK_BYTES", "16384"))
    # ffmpeg encoding knobs
    ffmpeg_bitrate: str = os.environ.get("VIBEVOICE_FFMPEG_BITRATE", "")  # e.g., "128k"
    flac_level: str = os.environ.get("VIBEVOICE_FLAC_LEVEL", "")  # 0-12
    # Opus container: webm (MSE-friendly) or ogg
    opus_container: str = os.environ.get("VIBEVOICE_OPUS_CONTAINER", "webm")
    # Opus VBR mode: vbr | constrained | cbr
    opus_vbr_mode: str = os.environ.get("VIBEVOICE_OPUS_VBR", "vbr")
    opus_application: str = os.environ.get("VIBEVOICE_OPUS_APPLICATION", "audio")  # audio|voip|lowdelay
    opus_frame_duration: str = os.environ.get("VIBEVOICE_OPUS_FRAME_DURATION", "")  # e.g., 20
    # AAC profile: aac_low | aac_he | aac_he_v2 | mpeg2_aac_low
    aac_profile: str = os.environ.get("VIBEVOICE_AAC_PROFILE", "")
    aac_mode: str = os.environ.get("VIBEVOICE_AAC_MODE", "cbr")  # cbr|vbr
    aac_q: str = os.environ.get("VIBEVOICE_AAC_Q", "")  # quality value for VBR (ffmpeg -q:a)
    # Auth
    # API key is DISABLED by default now. Set VIBEVOICE_REQUIRE_API_KEY=1 to enable.
    require_api_key: bool = os.environ.get("VIBEVOICE_REQUIRE_API_KEY", "0") not in {"0", "false", "False"}
    admin_token: str = os.environ.get("VIBEVOICE_ADMIN_TOKEN", "")
    keystore_path: str = os.environ.get("VIBEVOICE_KEYSTORE", os.path.join(os.getcwd(), "logs", "keys.json"))


CONFIG = ServerConfig()
