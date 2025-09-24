from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, AsyncIterator
import asyncio
import json
import base64
import io
import json
import base64
import io
import os

import numpy as np
from copy import deepcopy
import torch

from vibevoice_api.audio_utils import apply_speed, to_bytes_for_format, float_to_pcm16, ffmpeg_stream_cmd
from vibevoice_api.config import CONFIG
from vibevoice_api.voice_map import VoiceMapper
from vibevoice_api import observability as obs

from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AsyncAudioStreamer


@dataclass
class LoadedModel:
    processor: VibeVoiceProcessor
    model: VibeVoiceForConditionalGenerationInference
    device: str
    torch_dtype: torch.dtype
    sample_rate: int
    semaphore: threading.Semaphore


_engine_lock = threading.Lock()
_model_cache: Dict[str, LoadedModel] = {}


def _select_device() -> Tuple[str, torch.dtype, str]:
    pref = CONFIG.device_preference.lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16, "flash_attention_2"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", torch.float32, "sdpa"
        return "cpu", torch.float32, "sdpa"
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda", torch.bfloat16, "flash_attention_2"
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32, "sdpa"
    if pref == "cpu":
        return "cpu", torch.float32, "sdpa"
    # fallback
    return "cpu", torch.float32, "sdpa"


def _find_index_file(dir_path: str) -> Optional[str]:
    candidates = [
        os.path.join(dir_path, "model.safetensors.index.json"),
        os.path.join(dir_path, "pytorch_model.bin.index.json"),
        os.path.join(dir_path, "pytorch_model.safetensors.index.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _resolve_model_dir(model_path: str) -> str:
    """Return the directory that contains the HF index/model files.

    If `model_path` is a directory but doesn't contain an index file, and it has a
    single subdirectory that does, return that subdirectory. Otherwise return the
    original `model_path`.
    """
    if not os.path.isdir(model_path):
        return model_path
    if _find_index_file(model_path):
        return model_path
    # probe immediate subdirectories
    try:
        subdirs = [os.path.join(model_path, d) for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
    except Exception:
        return model_path
    with_index = [d for d in subdirs if _find_index_file(d)]
    if len(with_index) == 1:
        return with_index[0]
    return model_path


def _validate_local_model_dir(model_dir: str) -> None:
    """Validate that a local model directory has required shard files.

    Checks common Hugging Face index files and ensures all referenced shards exist.
    Raises ValueError with a helpful message if something is missing.
    """
    import glob
    if not os.path.isdir(model_dir):
        return

    idx_path = _find_index_file(model_dir)
    if not idx_path:
        # Might be unsharded (single file) or will be downloaded by HF; nothing to validate
        return

    try:
        with open(idx_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
    except Exception:
        return

    weight_map = idx.get("weight_map") or {}
    if not weight_map:
        return

    expected_files = sorted(set(weight_map.values()))
    missing = [fn for fn in expected_files if not os.path.exists(os.path.join(model_dir, fn))]
    if missing:
        msg = (
            "Model shards missing. Expected all shards from index file but could not find: "
            + ", ".join(missing)
        )
        if CONFIG.validate_shards:
            raise ValueError(msg)
        else:
            import logging
            logging.getLogger("vibevoice_api").warning(f"{msg} (validation disabled; attempting to load anyway)")


def _load_model(model_path: str) -> LoadedModel:
    with _engine_lock:
        if model_path in _model_cache:
            return _model_cache[model_path]

        device, torch_dtype, attn_impl = _select_device()

        # Resolve possibly nested local model path and validate shards to provide a clearer error early
        try:
            load_path = _resolve_model_dir(model_path)
            if os.path.isdir(load_path):
                _validate_local_model_dir(load_path)
        except ValueError as e:
            # bubble up a clearer error rather than deep stacktrace
            raise

        import logging
        logging.getLogger("vibevoice_api").info(
            f"Loading VibeVoice model from '{load_path}' on device={device}, dtype={torch_dtype}, attn={attn_impl}"
        )
        processor = VibeVoiceProcessor.from_pretrained(load_path)

        try:
            if device == "mps":
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    load_path,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_impl,
                    device_map=None,
                )
                model.to("mps")
            elif device == "cuda":
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    load_path,
                    torch_dtype=torch_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl,
                )
            else:  # cpu
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    load_path,
                    torch_dtype=torch_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl,
                )
        except FileNotFoundError as e:
            # Provide clearer hint for missing shard
            missing = getattr(e, "filename", "")
            raise ValueError(
                f"Missing model shard when loading '{load_path}'. {missing or ''}. "
                f"Ensure all shard files listed in the index are present."
            ) from e
        except Exception:
            # retry with sdpa if flash not available
            obs.ATTN_FALLBACK_TOTAL.labels("flash_attention_2", "sdpa").inc()
            obs.add_hint("attn_fallback:flash_attention_2->sdpa")
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                load_path,
                torch_dtype=torch_dtype,
                device_map=(device if device in ("cuda", "cpu") else None),
                attn_implementation="sdpa",
            )
            if device == "mps":
                model.to("mps")

        model.eval()
        model.set_ddpm_inference_steps(num_steps=CONFIG.ddpm_steps)

        loaded = LoadedModel(
            processor=processor,
            model=model,
            device=device,
            torch_dtype=torch_dtype,
            sample_rate=CONFIG.sample_rate,
            semaphore=threading.Semaphore(max(1, int(CONFIG.max_concurrency))),
        )
        _model_cache[model_path] = loaded
        return loaded


def synthesize(
    *,
    root_dir: str,
    text: str,
    voice: Optional[str],
    voice_path: Optional[str] = None,
    voice_data_b64: Optional[str] = None,
    speakers: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    response_format: str = "wav",
    speed: Optional[float] = None,
    instructions: Optional[str] = None,
    cfg_scale: Optional[float] = None,
    ddpm_steps: Optional[int] = None,
    instructions_strategy: Optional[str] = None,
    instructions_repeat: Optional[int] = None,
) -> Tuple[bytes, str]:
    """Synthesize speech from text using VibeVoice and return bytes + content type.

    Returns:
        (data, content_type)
    """
    model_id = model_path or CONFIG.model_path
    state = _load_model(model_id)

    # Build a minimal script for single-speaker TTS (optionally inject preprompt)
    inst_strategy = (instructions_strategy or CONFIG.instructions_strategy or "system_and_preprompt").lower()
    inst_repeat = max(1, int(instructions_repeat or CONFIG.instructions_repeat or 1))
    pre = ""
    if instructions and inst_strategy in {"system_and_preprompt", "preprompt_only"}:
        # Inject as valid script lines so the parser won't warn
        unit = f"Speaker 1: [Style: {instructions}. Follow strictly.]\n"
        pre = unit * inst_repeat
    script = pre + f"Speaker 1: {text.strip()}\n"

    # Resolve reference voice(s): precedence -> speakers list -> explicit data -> explicit path -> encoded in voice -> mapper by name
    resolved_voice_path: Optional[str] = None
    resolved_voice_wav: Optional[np.ndarray] = None
    resolved_speakers: List[Any] = []

    def _decode_data_url_or_b64(s: str) -> np.ndarray:
        try:
            import soundfile as sf
        except Exception as e:
            raise RuntimeError(
                "Decoding voice_data requires 'soundfile'. Install it (pip install soundfile) or pass a file path."
            ) from e

        payload = s
        if s.strip().startswith("data:"):
            idx = s.find("base64,")
            if idx == -1:
                raise ValueError("Only base64 data URLs are supported for voice_data")
            payload = s[idx + len("base64,") :]
        try:
            raw = base64.b64decode(payload, validate=False)
        except Exception as e:
            raise ValueError("Invalid base64 in voice_data") from e
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if data.ndim > 1:
            # take first channel
            data = data[:, 0]
        return data.astype(np.float32)

    # speakers list (if provided): allow alias/path/data for each
    if speakers:
        mapper = VoiceMapper(root_dir)
        for item in speakers:
            s = (item or "").strip()
            if not s:
                continue
            try:
                if s.startswith("data:"):
                    decoded = _decode_data_url_or_b64(s)
                    resolved_speakers.append(state.processor.audio_processor.preprocess_audio(decoded))
                else:
                    if os.path.exists(s) or s.startswith("file://") or s.startswith("path:"):
                        p = s
                        if p.startswith("file://"):
                            p = p[7:]
                        if p.startswith("path:"):
                            p = p[5:]
                        if not os.path.isabs(p):
                            p = os.path.abspath(p)
                        resolved_speakers.append(p)
                    else:
                        resolved_speakers.append(mapper.resolve(s))
            except Exception:
                continue

    # 1) explicit voice_data_b64
    if voice_data_b64:
        try:
            decoded = _decode_data_url_or_b64(voice_data_b64)
            # normalize/resample via audio_processor
            resolved_voice_wav = state.processor.audio_processor.preprocess_audio(decoded)
        except Exception as e:
            raise RuntimeError(f"Failed to decode voice_data: {type(e).__name__}: {e}")

    # 2) explicit voice_path
    if resolved_voice_wav is None and voice_path:
        # absolute or relative
        path = voice_path
        if path.startswith("file://"):
            path = path[7:]
        if path.startswith("path:"):
            path = path[5:]
        if not os.path.isabs(path):
            # relative to current working dir
            path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"voice_path not found: {path}")
        resolved_voice_path = path

    # 3) voice encodes a path or data url
    if resolved_voice_wav is None and resolved_voice_path is None and voice:
        v = voice.strip()
        if os.path.exists(v):
            resolved_voice_path = os.path.abspath(v)
        elif v.startswith("file://"):
            p = v[7:]
            if os.path.exists(p):
                resolved_voice_path = os.path.abspath(p)
        elif v.startswith("path:"):
            p = v[5:]
            if os.path.exists(p):
                resolved_voice_path = os.path.abspath(p)
        elif v.startswith("data:"):
            decoded = _decode_data_url_or_b64(v)
            resolved_voice_wav = state.processor.audio_processor.preprocess_audio(decoded)

    # 4) fallback mapper by friendly name
    if resolved_voice_wav is None and resolved_voice_path is None:
        mapper = VoiceMapper(root_dir)
        mapped = mapper.resolve(voice)
        resolved_voice_path = mapped

    # Build voice_samples payload for processor
    voice_samples: Optional[List[Any]] = None
    if resolved_speakers:
        voice_samples = resolved_speakers
    elif resolved_voice_wav is not None:
        voice_samples = [resolved_voice_wav]
    elif resolved_voice_path:
        voice_samples = [resolved_voice_path]

    # Optionally override system prompt with instructions
    original_system_prompt = getattr(state.processor, "system_prompt", None)
    if instructions and inst_strategy in {"system_and_preprompt", "system_only"}:
        try:
            hint_added = False
            # bound instructions length to avoid very long prompts
            inst = str(instructions)
            maxlen = int(getattr(CONFIG, 'instructions_maxlen', 2000) or 2000)
            if len(inst) > maxlen:
                orig_len = len(inst)
                inst = inst[:maxlen]
                try:
                    from vibevoice_api import observability as obs
                    obs.add_hint(f"instructions_clamped:orig={orig_len}>max={maxlen}")
                    hint_added = True
                except Exception:
                    pass
            combined = (original_system_prompt or "") + f"\nStrict style instructions (HIGH PRIORITY):\n{inst}\nFollow strictly.\n"
            state.processor.system_prompt = combined
            if not hint_added:
                try:
                    from vibevoice_api import observability as obs
                    obs.add_hint("instructions_applied")
                    obs.add_hint(f"instructions_strategy:{inst_strategy}")
                    obs.add_hint(f"instructions_repeat:{inst_repeat}")
                except Exception:
                    pass
        except Exception:
            # ignore instructions if any issue
            pass

    # Prepare inputs
    inputs = state.processor(
        text=[script],
        voice_samples=[voice_samples] if voice_samples else [None],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # restore system prompt
    try:
        state.processor.system_prompt = original_system_prompt
    except Exception:
        pass

    # Move tensors
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(state.device)

    with state.semaphore:
        with torch.inference_mode():
            # Optionally override diffusion steps per request
            old_steps = getattr(state.model, 'ddpm_inference_steps', None)
            if ddpm_steps:
                try:
                    state.model.set_ddpm_inference_steps(int(ddpm_steps))
                except Exception:
                    pass
            # Clone scheduler per request to avoid step_index race conditions
            tls_set = getattr(state.model.model, 'set_thread_local_noise_scheduler', None)
            tls_clear = getattr(state.model.model, 'clear_thread_local_noise_scheduler', None)
            tls_get = getattr(state.model.model, 'get_thread_local_noise_scheduler', None)
            try:
                if callable(tls_get) and callable(tls_set):
                    base_sched = tls_get()
                    tls_set(deepcopy(base_sched))
                with obs.observe_latency(obs.SYNTHESIS_LATENCY, (response_format or "wav").lower()):
                    obs.ACTIVE_INFERENCES.inc()
                    try:
                        outputs = state.model.generate(
                            **inputs,
                            max_new_tokens=None,
                            cfg_scale=float(cfg_scale) if cfg_scale is not None else 1.3,
                            tokenizer=state.processor.tokenizer,
                            generation_config={"do_sample": False},
                            verbose=False,
                        )
                    finally:
                        obs.ACTIVE_INFERENCES.dec()
            finally:
                if callable(tls_clear):
                    tls_clear()
                if old_steps is not None:
                    try:
                        state.model.set_ddpm_inference_steps(old_steps)
                    except Exception:
                        pass

    speech = outputs.speech_outputs[0]
    if isinstance(speech, torch.Tensor):
        wav = speech.detach().float().cpu().numpy()
    else:
        wav = np.asarray(speech, dtype=np.float32)

    wav = wav.squeeze()
    if speed is not None:
        try:
            wav = apply_speed(wav, float(speed))
        except Exception:
            # ignore speed if anything goes wrong
            pass

    data, content_type = to_bytes_for_format(wav, state.sample_rate, response_format)
    return data, content_type


async def synthesize_stream_pcm(
    *,
    root_dir: str,
    text: str,
    voice: Optional[str],
    voice_path: Optional[str] = None,
    voice_data_b64: Optional[str] = None,
    speakers: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    speed: Optional[float] = None,
    instructions: Optional[str] = None,
    cfg_scale: Optional[float] = None,
    ddpm_steps: Optional[int] = None,
    instructions_strategy: Optional[str] = None,
    instructions_repeat: Optional[int] = None,
) -> AsyncIterator[np.ndarray]:
    """Asynchronously generate audio and yield PCM float chunks as numpy arrays.

    Yields:
        np.ndarray: mono float32 waveform chunk in [-1, 1].
    """
    model_id = model_path or CONFIG.model_path
    state = _load_model(model_id)

    inst_strategy = (instructions_strategy or CONFIG.instructions_strategy or "system_and_preprompt").lower()
    inst_repeat = max(1, int(instructions_repeat or CONFIG.instructions_repeat or 1))
    pre = ""
    if instructions and inst_strategy in {"system_and_preprompt", "preprompt_only"}:
        pre = (f"Speaker 1: [Style: {instructions}. Follow strictly.]\n") * inst_repeat
    script = pre + f"Speaker 1: {text.strip()}\n"

    mapper = VoiceMapper(root_dir)
    resolved_voice_path: Optional[str] = None
    resolved_voice_wav: Optional[np.ndarray] = None
    resolved_speakers: List[Any] = []

    def _decode_data_url_or_b64(s: str) -> np.ndarray:
        try:
            import soundfile as sf
        except Exception as e:
            raise RuntimeError(
                "Decoding voice_data requires 'soundfile'. Install it (pip install soundfile) or pass a file path."
            ) from e
        payload = s
        if s.strip().startswith("data:"):
            idx = s.find("base64,")
            if idx == -1:
                raise ValueError("Only base64 data URLs are supported for voice_data")
            payload = s[idx + len("base64,") :]
        raw = base64.b64decode(payload, validate=False)
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")  # type: ignore[name-defined]
        if data.ndim > 1:
            data = data[:, 0]
        return data.astype(np.float32)

    # explicit speakers list
    if speakers:
        for item in speakers:
            s = (item or "").strip()
            if not s:
                continue
            try:
                if s.startswith("data:"):
                    decoded = _decode_data_url_or_b64(s)
                    resolved_speakers.append(state.processor.audio_processor.preprocess_audio(decoded))
                else:
                    if os.path.exists(s) or s.startswith("file://") or s.startswith("path:"):
                        p = s
                        if p.startswith("file://"):
                            p = p[7:]
                        if p.startswith("path:"):
                            p = p[5:]
                        if not os.path.isabs(p):
                            p = os.path.abspath(p)
                        resolved_speakers.append(p)
                    else:
                        resolved_speakers.append(mapper.resolve(s))
            except Exception:
                continue

    if voice_data_b64:
        decoded = _decode_data_url_or_b64(voice_data_b64)
        resolved_voice_wav = state.processor.audio_processor.preprocess_audio(decoded)

    if resolved_voice_wav is None and voice_path:
        path = voice_path
        if path.startswith("file://"):
            path = path[7:]
        if path.startswith("path:"):
            path = path[5:]
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"voice_path not found: {path}")
        resolved_voice_path = path

    if resolved_voice_wav is None and resolved_voice_path is None and voice:
        v = voice.strip()
        if os.path.exists(v):
            resolved_voice_path = os.path.abspath(v)
        elif v.startswith("file://"):
            p = v[7:]
            if os.path.exists(p):
                resolved_voice_path = os.path.abspath(p)
        elif v.startswith("path:"):
            p = v[5:]
            if os.path.exists(p):
                resolved_voice_path = os.path.abspath(p)
        elif v.startswith("data:"):
            decoded = _decode_data_url_or_b64(v)
            resolved_voice_wav = state.processor.audio_processor.preprocess_audio(decoded)

    if resolved_voice_wav is None and resolved_voice_path is None:
        mapped = mapper.resolve(voice)
        resolved_voice_path = mapped

    voice_samples: Optional[List[Any]] = None
    if resolved_speakers:
        voice_samples = resolved_speakers
    elif resolved_voice_wav is not None:
        voice_samples = [resolved_voice_wav]
    elif resolved_voice_path:
        voice_samples = [resolved_voice_path]

    # instructions into system prompt (temp override)
    original_system_prompt = getattr(state.processor, "system_prompt", None)
    if instructions and inst_strategy in {"system_and_preprompt", "system_only"}:
        try:
            inst = str(instructions)
            maxlen = int(getattr(CONFIG, 'instructions_maxlen', 2000) or 2000)
            if len(inst) > maxlen:
                orig_len = len(inst)
                inst = inst[:maxlen]
                try:
                    obs.add_hint(f"instructions_clamped:orig={orig_len}>max={maxlen}")  # type: ignore[name-defined]
                except Exception:
                    pass
            combined = (original_system_prompt or "") + f"\nStrict style instructions (HIGH PRIORITY):\n{inst}\nFollow strictly.\n"
            state.processor.system_prompt = combined
            try:
                obs.add_hint("instructions_applied")  # type: ignore[name-defined]
                obs.add_hint(f"instructions_strategy:{inst_strategy}")  # type: ignore[name-defined]
                obs.add_hint(f"instructions_repeat:{inst_repeat}")  # type: ignore[name-defined]
            except Exception:
                pass
        except Exception:
            pass

    inputs = state.processor(
        text=[script],
        voice_samples=[voice_samples] if voice_samples else [None],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    try:
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(state.device)

        async_streamer = AsyncAudioStreamer(batch_size=1, stop_signal="__END__")

        def run_generate():
            tls_set = getattr(state.model.model, 'set_thread_local_noise_scheduler', None)
            tls_clear = getattr(state.model.model, 'clear_thread_local_noise_scheduler', None)
            tls_get = getattr(state.model.model, 'get_thread_local_noise_scheduler', None)
            with state.semaphore:
                with torch.inference_mode():
                    try:
                        old_steps = getattr(state.model, 'ddpm_inference_steps', None)
                        if ddpm_steps:
                            try:
                                state.model.set_ddpm_inference_steps(int(ddpm_steps))
                            except Exception:
                                pass
                        if callable(tls_get) and callable(tls_set):
                            base_sched = tls_get()
                            tls_set(deepcopy(base_sched))
                        _ = state.model.generate(
                            **inputs,
                            max_new_tokens=None,
                            cfg_scale=float(cfg_scale) if cfg_scale is not None else 1.3,
                            tokenizer=state.processor.tokenizer,
                            generation_config={"do_sample": False},
                            verbose=False,
                            audio_streamer=async_streamer,
                        )
                    finally:
                        if callable(tls_clear):
                            tls_clear()
                        if old_steps is not None:
                            try:
                                state.model.set_ddpm_inference_steps(old_steps)
                            except Exception:
                                pass

        loop = asyncio.get_running_loop()
        # run generate in thread executor to avoid blocking event loop
        gen_task = loop.run_in_executor(None, run_generate)
        # stream chunks
        async for chunk in async_streamer.get_stream(0):
            arr = chunk.detach().float().cpu().numpy()
            arr = arr.squeeze()
            # apply speed if requested (naive resample)
            if speed is not None:
                try:
                    arr = apply_speed(arr, float(speed))
                except Exception:
                    pass
            yield arr
        # ensure task completed
        try:
            await gen_task
        except Exception:
            pass
    finally:
        try:
            state.processor.system_prompt = original_system_prompt
        except Exception:
            pass
