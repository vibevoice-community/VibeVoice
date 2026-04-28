import copy
import json
import os
import re
import threading
import time
import traceback
from typing import Any, Optional

import gradio as gr
import numpy as np
import torch
from transformers.utils import logging

from vibevoice.acceleration import (
    DEFAULT_DDPM_STEPS,
    build_model_load_kwargs,
    configure_torch_runtime,
    format_dtype,
    maybe_compile_language_model,
)
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.modular.streamer import AudioStreamer
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SAMPLE_RATE = 24000
VOICES_DIR = os.path.join(SCRIPT_DIR, "demo", "voices")
STREAMING_VOICES_DIR = os.path.join(VOICES_DIR, "streaming_model")
TEXT_EXAMPLES_DIR = os.path.join(SCRIPT_DIR, "demo", "text_examples")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

configure_torch_runtime()

_model = None
_processor = None
_current_model_key = None
_current_load_meta = None
_current_compile_status = "disabled"
_current_runtime_mode = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_available_models() -> list[str]:
    models: list[str] = []
    root = os.path.join(SCRIPT_DIR, "vibevoice")
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                models.append(f"vibevoice/{name}")
    defaults = [
        "vibevoice/VibeVoice-1.5B",
        "vibevoice/VibeVoice-0.5b",
        "microsoft/VibeVoice-Realtime-0.5B",
    ]
    merged: list[str] = []
    for item in defaults + models:
        if item not in merged:
            merged.append(item)
    return merged


def get_available_voices() -> list[str]:
    voices: list[str] = []
    if os.path.isdir(VOICES_DIR):
        for name in sorted(os.listdir(VOICES_DIR)):
            full_path = os.path.join(VOICES_DIR, name)
            if os.path.isfile(full_path) and name.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                voices.append(full_path)
    return voices


def get_available_streaming_voices() -> list[str]:
    voices: list[str] = []
    if os.path.isdir(STREAMING_VOICES_DIR):
        for name in sorted(os.listdir(STREAMING_VOICES_DIR)):
            full_path = os.path.join(STREAMING_VOICES_DIR, name)
            if os.path.isfile(full_path) and name.lower().endswith(".pt"):
                voices.append(full_path)
    return voices


def get_text_examples() -> list[str]:
    examples: list[str] = []
    if os.path.isdir(TEXT_EXAMPLES_DIR):
        for name in sorted(os.listdir(TEXT_EXAMPLES_DIR)):
            if name.lower().endswith(".txt"):
                examples.append(os.path.join(TEXT_EXAMPLES_DIR, name))
    return examples


def load_example_text(example_path: str) -> str:
    if not example_path or not os.path.exists(example_path):
        return ""
    with open(example_path, "r", encoding="utf-8") as handle:
        return handle.read()


def parse_txt_script(txt_content: str) -> list[str]:
    lines = txt_content.strip().splitlines()
    scripts: list[str] = []
    current_speaker = None
    current_text = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        match = re.match(r"^Speaker\s+(\d+):\s*(.*)$", line, re.IGNORECASE)
        if match:
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
            continue

        if current_speaker:
            current_text = f"{current_text} {line}".strip()
        else:
            current_speaker = "1"
            current_text = line

    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")

    return scripts


def resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    candidate = os.path.join(SCRIPT_DIR, path)
    if os.path.exists(candidate):
        return candidate
    return path


def resolve_local_model_dir(model_name_or_path: str) -> Optional[str]:
    candidate = resolve_path(model_name_or_path)
    if os.path.isdir(candidate):
        return candidate
    return None


def detect_model_mode(model_name_or_path: str, requested_mode: str) -> str:
    if requested_mode in {"standard", "streaming"}:
        return requested_mode

    local_dir = resolve_local_model_dir(model_name_or_path)
    if local_dir:
        config_path = os.path.join(local_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as handle:
                    config = json.load(handle)
                if config.get("model_type") == "vibevoice_streaming":
                    return "streaming"
            except Exception:
                pass
        return "standard"

    lowered = model_name_or_path.lower()
    if "realtime" in lowered or "streaming" in lowered or "0.5b" in lowered:
        return "streaming"
    return "standard"


def current_autocast_dtype() -> Optional[torch.dtype]:
    if get_device() != "cuda" or _current_load_meta is None:
        return None
    dtype = _current_load_meta.get("torch_dtype")
    if dtype in (torch.float16, torch.bfloat16):
        return dtype
    return None


def get_streaming_load_meta(attn_impl: str) -> dict[str, Any]:
    device = get_device()
    if device == "cuda":
        torch_dtype = torch.bfloat16
        actual_attn = "flash_attention_2" if attn_impl == "auto" else attn_impl
    elif device == "mps":
        torch_dtype = torch.float32
        actual_attn = "sdpa"
    else:
        torch_dtype = torch.float32
        actual_attn = "sdpa" if attn_impl == "auto" else attn_impl

    return {
        "torch_dtype": torch_dtype,
        "attn_impl": actual_attn,
        "attn_reason": f"streaming model on {device}",
        "is_quantized": False,
    }


def model_status_summary(model_name: str, ddpm_steps: int) -> str:
    mode = _current_runtime_mode or "unknown"
    if _current_load_meta is None:
        return f"Model loaded: {model_name}\nMode: {mode}\nDDPM: {ddpm_steps}"

    parts = [
        f"Model loaded: {model_name}",
        f"Mode: {mode}",
        (
            f"4bit: {'yes' if _current_load_meta['is_quantized'] else 'no'} | "
            f"attention: {_current_load_meta['attn_impl']} | DDPM: {ddpm_steps}"
        ),
        (
            f"dtype: {format_dtype(_current_load_meta['torch_dtype'])} | "
            f"compile: {_current_compile_status}"
        ),
        f"reason: {_current_load_meta['attn_reason']}",
    ]

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        parts.append(f"VRAM: allocated {allocated:.2f} GB / reserved {reserved:.2f} GB")

    return "\n".join(parts)


def load_standard_model(
    model_name: str,
    use_4bit: bool,
    attn_impl: str,
    ddpm_steps: int,
    use_compile: bool,
) -> tuple[Any, Any, dict[str, Any], str]:
    model_path = resolve_path(model_name)
    processor = VibeVoiceProcessor.from_pretrained(model_path)
    load_kwargs, load_meta = build_model_load_kwargs(
        model_path=model_path,
        use_4bit=use_4bit,
        attn_impl=attn_impl,
    )
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        **load_kwargs,
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=ddpm_steps)
    model, compile_status = maybe_compile_language_model(model, use_compile)
    return model, processor, load_meta, compile_status


def load_streaming_model(
    model_name: str,
    attn_impl: str,
    ddpm_steps: int,
) -> tuple[Any, Any, dict[str, Any], str]:
    model_path = resolve_path(model_name)
    device = get_device()
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
    load_meta = get_streaming_load_meta(attn_impl)
    actual_attn = load_meta["attn_impl"]
    torch_dtype = load_meta["torch_dtype"]

    try:
        if device == "mps":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation=actual_attn,
                device_map=None,
            )
            model.to("mps")
        elif device == "cuda":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="cuda",
                attn_implementation=actual_attn,
            )
        else:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="cpu",
                attn_implementation=actual_attn,
            )
    except Exception:
        if actual_attn == "flash_attention_2":
            fallback_attn = "sdpa"
            load_meta["attn_impl"] = fallback_attn
            load_meta["attn_reason"] = "flash_attention_2 load failed; fell back to sdpa"
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=(device if device in ("cuda", "cpu") else None),
                attn_implementation=fallback_attn,
            )
            if device == "mps":
                model.to("mps")
        else:
            raise

    model.eval()
    model.set_ddpm_inference_steps(num_steps=ddpm_steps)
    return model, processor, load_meta, "disabled"


def load_model(
    model_name: str,
    requested_mode: str,
    use_4bit: bool,
    attn_impl: str,
    ddpm_steps: int,
    use_compile: bool,
    progress=gr.Progress(),
):
    global _model, _processor, _current_model_key, _current_load_meta, _current_compile_status, _current_runtime_mode

    resolved_mode = detect_model_mode(model_name, requested_mode)
    cache_key = (
        f"{model_name}|mode={resolved_mode}|4bit={use_4bit}|attn={attn_impl}|"
        f"compile={use_compile}|ddpm={ddpm_steps}"
    )
    if _model is not None and _current_model_key == cache_key:
        _model.set_ddpm_inference_steps(num_steps=ddpm_steps)
        return f"{model_status_summary(model_name, ddpm_steps)}\nReused existing model instance."

    progress(0.1, desc="Releasing old model...")
    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    _current_model_key = None
    _current_load_meta = None
    _current_compile_status = "disabled"
    _current_runtime_mode = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        progress(0.2, desc="Loading processor...")
        if resolved_mode == "streaming":
            progress(0.5, desc="Loading realtime streaming model...")
            model, processor, load_meta, compile_status = load_streaming_model(
                model_name=model_name,
                attn_impl=attn_impl,
                ddpm_steps=ddpm_steps,
            )
        else:
            progress(0.5, desc="Loading standard model...")
            model, processor, load_meta, compile_status = load_standard_model(
                model_name=model_name,
                use_4bit=use_4bit,
                attn_impl=attn_impl,
                ddpm_steps=ddpm_steps,
                use_compile=use_compile,
            )

        _model = model
        _processor = processor
        _current_model_key = cache_key
        _current_load_meta = load_meta
        _current_compile_status = compile_status
        _current_runtime_mode = resolved_mode
        return model_status_summary(model_name, ddpm_steps)
    except Exception:
        _model = None
        _processor = None
        _current_model_key = None
        _current_load_meta = None
        _current_compile_status = "disabled"
        _current_runtime_mode = None
        return f"Model load failed:\n{traceback.format_exc()}"


def resolve_voice_path(voice_preset: str, voice_upload: Optional[str]) -> Optional[str]:
    if voice_upload and os.path.exists(voice_upload):
        return voice_upload
    if voice_preset and os.path.exists(voice_preset):
        return voice_preset
    return None


def resolve_streaming_prompt_path(prompt_preset: str, prompt_upload: Optional[str]) -> Optional[str]:
    return resolve_voice_path(prompt_preset, prompt_upload)


def convert_chunk_to_numpy(audio_chunk: Any) -> np.ndarray:
    if torch.is_tensor(audio_chunk):
        audio_chunk = audio_chunk.detach().cpu()
        if audio_chunk.dtype == torch.bfloat16:
            audio_chunk = audio_chunk.float()
        audio_np = audio_chunk.numpy().astype(np.float32)
    else:
        audio_np = np.asarray(audio_chunk, dtype=np.float32)

    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    return audio_np


def generate_speech(
    text: str,
    voice_preset: str,
    voice_upload: Optional[str],
    cfg_scale: float,
    ddpm_steps: int,
    seed: int,
    enable_prefill: bool,
    do_sample: bool,
    enhance_audio: bool,
    progress=gr.Progress(),
):
    global _model, _processor

    if _model is None or _processor is None:
        return None, "Please load a model first."
    if _current_runtime_mode != "standard":
        return None, "The loaded model is a streaming model. Use the Streaming tab instead."
    if not text or not text.strip():
        return None, "Please enter text to synthesize."

    voice_path = resolve_voice_path(voice_preset, voice_upload)
    if not voice_path:
        return None, "Please choose or upload a reference voice file."

    try:
        _model.set_ddpm_inference_steps(num_steps=ddpm_steps)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        progress(0.1, desc="Parsing text...")
        scripts = parse_txt_script(text)
        if not scripts:
            return None, "Unable to parse the script text."

        full_script = "\n".join(scripts).replace("\u2019", "'")
        inputs = _processor(
            text=[full_script],
            voice_samples=[[voice_path]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        device = get_device()
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(device, non_blocking=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        progress(0.3, desc="Generating audio...")
        start_time = time.time()
        generation_config = {"do_sample": do_sample}
        autocast_dtype = current_autocast_dtype()

        with torch.inference_mode():
            if autocast_dtype is not None:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    outputs = _model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=_processor.tokenizer,
                        generation_config=generation_config,
                        verbose=False,
                        show_progress_bar=False,
                        is_prefill=enable_prefill,
                    )
            else:
                outputs = _model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=_processor.tokenizer,
                    generation_config=generation_config,
                    verbose=False,
                    show_progress_bar=False,
                    is_prefill=enable_prefill,
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        generation_time = time.time() - start_time
        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            return None, "Model returned no audio output."

        audio_np = convert_chunk_to_numpy(outputs.speech_outputs[0])
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"webui_{timestamp}.wav")
        _processor.save_audio(
            outputs.speech_outputs[0],
            output_path=output_path,
            normalize=enhance_audio,
        )

        audio_duration = len(audio_np) / SAMPLE_RATE
        rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")
        input_tokens = int(inputs["input_ids"].shape[1])
        output_tokens = int(outputs.sequences.shape[1])
        generated_tokens = output_tokens - input_tokens

        stats_lines = [
            "Generation complete.",
            f"Time: {generation_time:.2f}s",
            f"Audio duration: {audio_duration:.2f}s",
            f"RTF: {rtf:.3f}x",
            f"Tokens: input {input_tokens} | generated {generated_tokens}",
            f"CFG: {cfg_scale} | DDPM: {ddpm_steps} | Seed: {seed}",
            f"Voice enhancement: {'on' if enhance_audio else 'off'}",
            f"Voice: {os.path.basename(voice_path)}",
            f"Segments: {len(scripts)}",
            f"Saved: {output_path}",
        ]

        if _current_load_meta is not None:
            stats_lines.append(
                f"Runtime: {_current_load_meta['attn_impl']} | {format_dtype(_current_load_meta['torch_dtype'])} | compile={_current_compile_status}"
            )

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3
            stats_lines.append(f"VRAM: current {allocated:.2f} GB | peak {peak:.2f} GB")

        return output_path, "\n".join(stats_lines)
    except Exception:
        return None, f"Generation failed:\n{traceback.format_exc()}"


def _run_streaming_generation(
    inputs: dict[str, Any],
    cached_prompt: dict[str, Any],
    cfg_scale: float,
    audio_streamer: AudioStreamer,
    result_holder: dict[str, Any],
):
    try:
        autocast_dtype = current_autocast_dtype()
        with torch.inference_mode():
            if autocast_dtype is not None:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    outputs = _model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=_processor.tokenizer,
                        generation_config={"do_sample": False},
                        verbose=False,
                        show_progress_bar=False,
                        all_prefilled_outputs=copy.deepcopy(cached_prompt),
                        audio_streamer=audio_streamer,
                    )
            else:
                outputs = _model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=_processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    show_progress_bar=False,
                    all_prefilled_outputs=copy.deepcopy(cached_prompt),
                    audio_streamer=audio_streamer,
                )
        result_holder["outputs"] = outputs
    except Exception:
        result_holder["error"] = traceback.format_exc()
    finally:
        audio_streamer.end()


def generate_streaming_speech(
    text: str,
    prompt_preset: str,
    prompt_upload: Optional[str],
    cfg_scale: float,
    ddpm_steps: int,
    seed: int,
):
    global _model, _processor

    if _model is None or _processor is None:
        yield None, None, "Please load a model first."
        return
    if _current_runtime_mode != "streaming":
        yield None, None, "The loaded model is not a streaming model. Load `VibeVoice-0.5b` or `microsoft/VibeVoice-Realtime-0.5B` first."
        return
    if not text or not text.strip():
        yield None, None, "Please enter text to synthesize."
        return

    prompt_path = resolve_streaming_prompt_path(prompt_preset, prompt_upload)
    if not prompt_path:
        yield None, None, "Please choose or upload a streaming voice prompt `.pt` file."
        return

    try:
        _model.set_ddpm_inference_steps(num_steps=ddpm_steps)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        scripts = parse_txt_script(text)
        if not scripts:
            yield None, None, "Unable to parse the script text."
            return

        full_script = "\n".join(scripts).replace("\u2019", "'")
        device = get_device()
        cached_prompt = torch.load(
            prompt_path,
            map_location=(device if device != "mps" else "cpu"),
            weights_only=False,
        )

        inputs = _processor.process_input_with_cached_prompt(
            text=full_script,
            cached_prompt=cached_prompt,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        for key, value in inputs.items():
            if torch.is_tensor(value):
                target_device = device if device != "mps" else "mps"
                inputs[key] = value.to(target_device)

        if device == "mps":
            cached_prompt = torch.load(prompt_path, map_location="mps", weights_only=False)

        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=1.0)
        result_holder: dict[str, Any] = {"outputs": None, "error": None}
        start_time = time.time()
        worker = threading.Thread(
            target=_run_streaming_generation,
            args=(inputs, cached_prompt, cfg_scale, audio_streamer, result_holder),
            daemon=True,
        )
        worker.start()

        total_samples = 0
        chunk_count = 0
        yield None, None, "Streaming generation started..."

        for audio_chunk in audio_streamer.get_stream(0):
            audio_np = convert_chunk_to_numpy(audio_chunk)
            chunk_count += 1
            total_samples += len(audio_np)
            elapsed = time.time() - start_time
            streamed_duration = total_samples / SAMPLE_RATE
            yield (
                SAMPLE_RATE,
                audio_np,
            ), None, (
                "Streaming generation in progress...\n"
                f"Voice prompt: {os.path.basename(prompt_path)}\n"
                f"Chunk: {chunk_count}\n"
                f"Streamed audio: {streamed_duration:.2f}s\n"
                f"Elapsed: {elapsed:.2f}s"
            )

        worker.join()
        if result_holder["error"]:
            yield None, None, f"Streaming generation failed:\n{result_holder['error']}"
            return

        outputs = result_holder["outputs"]
        if outputs is None or not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            yield None, None, "Streaming generation produced no final audio."
            return

        audio_np = convert_chunk_to_numpy(outputs.speech_outputs[0])
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"webui_stream_{timestamp}.wav")
        _processor.save_audio(outputs.speech_outputs[0], output_path=output_path)

        generation_time = time.time() - start_time
        audio_duration = len(audio_np) / SAMPLE_RATE
        rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")
        prefilled_text_tokens = int(inputs["tts_text_ids"].shape[1])
        output_tokens = int(outputs.sequences.shape[1])
        prompt_tokens = int(cached_prompt["tts_lm"]["last_hidden_state"].size(1))
        generated_tokens = output_tokens - prefilled_text_tokens - prompt_tokens

        stats_lines = [
            "Streaming generation complete.",
            f"Time: {generation_time:.2f}s",
            f"Audio duration: {audio_duration:.2f}s",
            f"RTF: {rtf:.3f}x",
            f"Tokens: text {prefilled_text_tokens} | generated speech {generated_tokens}",
            f"CFG: {cfg_scale} | DDPM: {ddpm_steps} | Seed: {seed}",
            f"Voice prompt: {os.path.basename(prompt_path)}",
            f"Streamed chunks: {chunk_count}",
            f"Saved: {output_path}",
        ]

        if _current_load_meta is not None:
            stats_lines.append(
                f"Runtime: {_current_load_meta['attn_impl']} | {format_dtype(_current_load_meta['torch_dtype'])}"
            )

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3
            stats_lines.append(f"VRAM: current {allocated:.2f} GB | peak {peak:.2f} GB")

        yield None, output_path, "\n".join(stats_lines)
    except Exception:
        yield None, None, f"Streaming generation failed:\n{traceback.format_exc()}"


def get_gpu_info() -> str:
    if not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            return "CUDA unavailable. MPS available."
        return "CUDA unavailable. WebUI will run on CPU."

    lines = [f"CUDA: {torch.version.cuda}"]
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_mem = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(idx) / 1024**3
        lines.append(f"GPU {idx}: {props.name} | total {total_mem:.1f} GB | used {allocated:.2f} GB")
    return "\n".join(lines)


def refresh_vram() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable."
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return f"Allocated {allocated:.2f} GB | Reserved {reserved:.2f} GB | Peak {peak:.2f} GB"


def clear_vram_cache() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable."
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return "VRAM cache cleared."


def unload_model() -> str:
    global _model, _processor, _current_model_key, _current_load_meta, _current_compile_status, _current_runtime_mode

    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None

    _current_model_key = None
    _current_load_meta = None
    _current_compile_status = "disabled"
    _current_runtime_mode = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return "Model unloaded."


def preview_voice(voice_map: dict[str, str], name: str):
    if name and name in voice_map:
        return voice_map[name]
    return None


def batch_generate(
    voice_map: dict[str, str],
    files,
    voice_name,
    cfg,
    steps,
    seed,
    enhance_audio,
    progress=gr.Progress(),
):
    if _model is None or _processor is None:
        return "Please load a model first."
    if _current_runtime_mode != "standard":
        return "Batch generation currently supports standard models only."
    if not files:
        return "Please upload one or more text files."

    voice_path = voice_map.get(voice_name, "") if voice_name else ""
    if not voice_path:
        return "Please choose a reference voice."

    logs = []
    total = len(files)
    for idx, file_obj in enumerate(files, start=1):
        file_name = os.path.basename(file_obj.name) if hasattr(file_obj, "name") else os.path.basename(str(file_obj))
        file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        progress(idx / total, desc=f"Processing {file_name}...")
        logs.append(f"--- [{idx}/{total}] {file_name} ---")

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                text = handle.read()
            _, stats = generate_speech(text, voice_path, None, cfg, steps, int(seed), True, False, enhance_audio)
            logs.append(stats)
        except Exception as exc:
            logs.append(f"Failed: {exc}")

    logs.append(f"\nBatch generation finished. Files: {total}")
    return "\n".join(logs)


def run_compare(
    voice_map: dict[str, str],
    text,
    voice_name,
    a_cfg,
    a_steps,
    a_seed,
    a_prefill,
    b_cfg,
    b_steps,
    b_seed,
    b_prefill,
    enhance_audio,
):
    if _current_runtime_mode != "standard":
        return None, "Compare is only available for standard models.", None, "Compare is only available for standard models."

    voice_path = voice_map.get(voice_name, "") if voice_name else ""
    a_audio, a_stats = generate_speech(text, voice_path, None, a_cfg, a_steps, int(a_seed), a_prefill, False, enhance_audio)
    b_audio, b_stats = generate_speech(text, voice_path, None, b_cfg, b_steps, int(b_seed), b_prefill, False, enhance_audio)
    return a_audio, a_stats, b_audio, b_stats


def get_model_info() -> str:
    if _model is None:
        return "No model loaded."

    lines = [
        f"Current key: {_current_model_key}",
        f"Runtime mode: {_current_runtime_mode}",
    ]
    try:
        total_params = sum(param.numel() for param in _model.parameters())
        trainable_params = sum(param.numel() for param in _model.parameters() if param.requires_grad)
        dtypes = sorted({str(param.dtype) for param in _model.parameters()})
        lines.append(f"Total params: {total_params / 1e9:.2f}B ({total_params:,})")
        lines.append(f"Trainable params: {trainable_params:,}")
        lines.append(f"Parameter dtypes: {', '.join(dtypes)}")
    except Exception:
        pass

    if _current_load_meta is not None:
        lines.append(f"Attention: {_current_load_meta['attn_impl']}")
        lines.append(f"Dtype: {format_dtype(_current_load_meta['torch_dtype'])}")
        lines.append(f"Compile: {_current_compile_status}")

    return "\n".join(lines)


def list_outputs() -> str:
    if not os.path.isdir(OUTPUT_DIR):
        return "Output directory is empty."

    files = sorted(
        [name for name in os.listdir(OUTPUT_DIR) if name.lower().endswith(".wav")],
        reverse=True,
    )[:20]
    if not files:
        return "No output files yet."
    return "\n".join(files)


def build_ui():
    available_models = get_available_models()
    available_voices = get_available_voices()
    voice_names = [os.path.basename(path) for path in available_voices]
    voice_map = dict(zip(voice_names, available_voices))

    available_streaming_voices = get_available_streaming_voices()
    streaming_voice_names = [os.path.basename(path) for path in available_streaming_voices]

    text_examples = get_text_examples()
    example_names = [os.path.basename(path) for path in text_examples]
    example_map = dict(zip(example_names, text_examples))

    with gr.Blocks(
        title="VibeVoice WebUI",
        theme=gr.themes.Soft(),
        css="""
        .status-box { font-family: monospace; font-size: 13px; }
        .header-text { text-align: center; margin-bottom: 10px; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # VibeVoice WebUI
            Local UI for both standard VibeVoice generation and the 0.5B realtime streaming model.

            Suggested streaming models:
            `vibevoice/VibeVoice-0.5b` or `microsoft/VibeVoice-Realtime-0.5B`
            """,
            elem_classes=["header-text"],
        )

        gpu_info = gr.Textbox(
            label="System Info",
            value=get_gpu_info(),
            interactive=False,
            lines=3,
        )

        with gr.Tabs():
            with gr.Tab("Model"):
                with gr.Row():
                    with gr.Column(scale=2):
                        model_selector = gr.Dropdown(
                            label="Model path or Hugging Face model id",
                            choices=available_models,
                            value=available_models[0] if available_models else "vibevoice/VibeVoice-1.5B",
                            interactive=True,
                            allow_custom_value=True,
                        )
                        model_mode = gr.Dropdown(
                            label="Model mode",
                            choices=["auto", "standard", "streaming"],
                            value="auto",
                            info="Use auto for local configs. Set streaming when loading the realtime 0.5B model from Hugging Face.",
                        )
                        with gr.Row():
                            use_4bit = gr.Checkbox(
                                label="4bit quantization",
                                value=False,
                                info="Only applies to standard models.",
                            )
                            use_compile = gr.Checkbox(
                                label="torch.compile",
                                value=False,
                                info="Only applies to standard models.",
                            )

                        attn_impl = gr.Dropdown(
                            label="Attention backend",
                            choices=["auto", "flash_attention_2", "sdpa", "eager"],
                            value="auto",
                        )
                        ddpm_steps_load = gr.Slider(
                            label="DDPM steps",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=DEFAULT_DDPM_STEPS,
                        )

                    with gr.Column(scale=1):
                        load_btn = gr.Button("Load Model", variant="primary", size="lg")
                        unload_btn = gr.Button("Unload Model", variant="stop")
                        clear_cache_btn = gr.Button("Clear VRAM Cache")
                        refresh_vram_btn = gr.Button("Refresh VRAM")
                        vram_display = gr.Textbox(label="VRAM", interactive=False, lines=1)

                model_status = gr.Textbox(
                    label="Model Status",
                    interactive=False,
                    lines=7,
                    elem_classes=["status-box"],
                )

                load_btn.click(
                    fn=load_model,
                    inputs=[model_selector, model_mode, use_4bit, attn_impl, ddpm_steps_load, use_compile],
                    outputs=[model_status],
                )
                unload_btn.click(fn=unload_model, outputs=[model_status])
                clear_cache_btn.click(fn=clear_vram_cache, outputs=[vram_display])
                refresh_vram_btn.click(fn=refresh_vram, outputs=[vram_display])

            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            example_selector = gr.Dropdown(
                                label="Example text",
                                choices=[""] + example_names,
                                value="",
                                interactive=True,
                                scale=3,
                            )
                            load_example_btn = gr.Button("Load Example", scale=1)

                        text_input = gr.Textbox(
                            label="Input text",
                            placeholder=(
                                "Speaker 1: Hello world\n"
                                "Speaker 2: This is a second speaker\n\n"
                                "Plain text also works and will default to Speaker 1."
                            ),
                            lines=8,
                            max_lines=20,
                        )

                        with gr.Row():
                            voice_selector = gr.Dropdown(
                                label="Preset voice",
                                choices=voice_names,
                                value=voice_names[0] if voice_names else None,
                                interactive=True,
                                scale=2,
                            )
                            voice_upload = gr.Audio(
                                label="Upload voice",
                                type="filepath",
                                scale=2,
                            )

                        voice_preview = gr.Audio(
                            label="Voice preview",
                            interactive=False,
                            type="filepath",
                        )
                        voice_selector.change(
                            fn=lambda name: preview_voice(voice_map, name),
                            inputs=[voice_selector],
                            outputs=[voice_preview],
                        )

                    with gr.Column(scale=2):
                        cfg_scale = gr.Slider(
                            label="CFG scale",
                            minimum=0.1,
                            maximum=5.0,
                            step=0.1,
                            value=2.0,
                        )
                        ddpm_steps = gr.Slider(
                            label="DDPM steps",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=DEFAULT_DDPM_STEPS,
                        )
                        seed = gr.Number(
                            label="Seed",
                            value=42,
                            precision=0,
                        )
                        with gr.Row():
                            enable_prefill = gr.Checkbox(
                                label="Enable prefill voice cloning",
                                value=True,
                            )
                            do_sample = gr.Checkbox(
                                label="Sampling mode",
                                value=False,
                            )
                        enhance_audio = gr.Checkbox(
                            label="声音增强",
                            value=False,
                            info="Save and preview peak-normalized audio for louder playback.",
                        )

                        generate_btn = gr.Button("Generate Audio", variant="primary", size="lg")
                        audio_output = gr.Audio(
                            label="Generated audio",
                            type="filepath",
                            interactive=False,
                        )
                        gen_stats = gr.Textbox(
                            label="Generation stats",
                            interactive=False,
                            lines=10,
                            elem_classes=["status-box"],
                        )

                load_example_btn.click(
                    fn=lambda name: load_example_text(example_map[name]) if name and name in example_map else "",
                    inputs=[example_selector],
                    outputs=[text_input],
                )
                generate_btn.click(
                    fn=generate_speech,
                    inputs=[text_input, voice_selector, voice_upload, cfg_scale, ddpm_steps, seed, enable_prefill, do_sample, enhance_audio],
                    outputs=[audio_output, gen_stats],
                )

            with gr.Tab("Streaming"):
                gr.Markdown(
                    "Load a streaming 0.5B model in the Model tab first, then generate realtime audio with cached `.pt` speaker prompts."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            stream_example_selector = gr.Dropdown(
                                label="Example text",
                                choices=[""] + example_names,
                                value="",
                                interactive=True,
                                scale=3,
                            )
                            stream_example_btn = gr.Button("Load Example", scale=1)

                        stream_text_input = gr.Textbox(
                            label="Streaming text",
                            placeholder="Speaker 1: Welcome to realtime VibeVoice streaming.",
                            lines=8,
                            max_lines=20,
                        )
                        with gr.Row():
                            stream_voice_selector = gr.Dropdown(
                                label="Preset streaming prompt (.pt)",
                                choices=streaming_voice_names,
                                value=streaming_voice_names[0] if streaming_voice_names else None,
                                interactive=True,
                                scale=2,
                            )
                            stream_prompt_upload = gr.File(
                                label="Upload streaming prompt (.pt)",
                                file_types=[".pt"],
                                type="filepath",
                                scale=2,
                            )

                    with gr.Column(scale=2):
                        stream_cfg = gr.Slider(
                            label="CFG scale",
                            minimum=0.1,
                            maximum=5.0,
                            step=0.1,
                            value=1.5,
                        )
                        stream_steps = gr.Slider(
                            label="DDPM steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=5,
                        )
                        stream_seed = gr.Number(
                            label="Seed",
                            value=42,
                            precision=0,
                        )
                        stream_generate_btn = gr.Button("Start Streaming", variant="primary", size="lg")

                stream_audio_output = gr.Audio(
                    label="Realtime audio stream",
                    type="numpy",
                    interactive=False,
                    streaming=True,
                    autoplay=True,
                    show_download_button=False,
                )
                stream_final_output = gr.Audio(
                    label="Final audio file",
                    type="filepath",
                    interactive=False,
                    show_download_button=True,
                )
                stream_stats = gr.Textbox(
                    label="Streaming stats",
                    interactive=False,
                    lines=10,
                    elem_classes=["status-box"],
                )

                stream_example_btn.click(
                    fn=lambda name: load_example_text(example_map[name]) if name and name in example_map else "",
                    inputs=[stream_example_selector],
                    outputs=[stream_text_input],
                )
                stream_generate_btn.click(
                    fn=lambda: (None, None, "Preparing streaming generation..."),
                    outputs=[stream_audio_output, stream_final_output, stream_stats],
                    queue=False,
                ).then(
                    fn=generate_streaming_speech,
                    inputs=[stream_text_input, stream_voice_selector, stream_prompt_upload, stream_cfg, stream_steps, stream_seed],
                    outputs=[stream_audio_output, stream_final_output, stream_stats],
                )

            with gr.Tab("Batch"):
                gr.Markdown("Generate audio from multiple text files with a standard model.")
                with gr.Row():
                    with gr.Column():
                        batch_files = gr.File(
                            label="Upload .txt files",
                            file_types=[".txt"],
                            file_count="multiple",
                        )
                        batch_voice = gr.Dropdown(
                            label="Reference voice",
                            choices=voice_names,
                            value=voice_names[0] if voice_names else None,
                        )
                        with gr.Row():
                            batch_cfg = gr.Slider(label="CFG", minimum=0.1, maximum=5.0, step=0.1, value=2.0)
                            batch_steps = gr.Slider(label="DDPM steps", minimum=1, maximum=100, step=1, value=DEFAULT_DDPM_STEPS)
                            batch_seed = gr.Number(label="Seed", value=42, precision=0)
                        batch_enhance_audio = gr.Checkbox(label="声音增强", value=False)
                        batch_btn = gr.Button("Start Batch", variant="primary")
                    with gr.Column():
                        batch_log = gr.Textbox(
                            label="Batch log",
                            lines=15,
                            interactive=False,
                            elem_classes=["status-box"],
                        )

                batch_btn.click(
                    fn=lambda files, voice_name, cfg, steps, seed_value, enhance_flag: batch_generate(
                        voice_map, files, voice_name, cfg, steps, seed_value, enhance_flag
                    ),
                    inputs=[batch_files, batch_voice, batch_cfg, batch_steps, batch_seed, batch_enhance_audio],
                    outputs=[batch_log],
                )

            with gr.Tab("Compare"):
                gr.Markdown("A/B compare two parameter sets using the same text. Standard models only.")
                with gr.Row():
                    compare_text = gr.Textbox(
                        label="Compare text",
                        value="Speaker 1: Hello, this is a parameter comparison test.",
                        lines=3,
                    )
                    compare_voice = gr.Dropdown(
                        label="Reference voice",
                        choices=voice_names,
                        value=voice_names[0] if voice_names else None,
                    )
                    compare_enhance_audio = gr.Checkbox(label="声音增强", value=False)

                with gr.Row():
                    with gr.Column():
                        a_cfg = gr.Slider(label="CFG", minimum=0.1, maximum=5.0, step=0.1, value=1.0)
                        a_steps = gr.Slider(label="DDPM steps", minimum=1, maximum=100, step=1, value=DEFAULT_DDPM_STEPS)
                        a_seed = gr.Number(label="Seed", value=42, precision=0)
                        a_prefill = gr.Checkbox(label="Prefill", value=True)
                        a_audio = gr.Audio(label="Result A", type="filepath", interactive=False)
                        a_stats = gr.Textbox(label="Stats A", lines=6, interactive=False, elem_classes=["status-box"])

                    with gr.Column():
                        b_cfg = gr.Slider(label="CFG", minimum=0.1, maximum=5.0, step=0.1, value=1.5)
                        b_steps = gr.Slider(label="DDPM steps", minimum=1, maximum=100, step=1, value=DEFAULT_DDPM_STEPS)
                        b_seed = gr.Number(label="Seed", value=42, precision=0)
                        b_prefill = gr.Checkbox(label="Prefill", value=True)
                        b_audio = gr.Audio(label="Result B", type="filepath", interactive=False)
                        b_stats = gr.Textbox(label="Stats B", lines=6, interactive=False, elem_classes=["status-box"])

                compare_btn = gr.Button("Run Compare", variant="primary")
                compare_btn.click(
                    fn=lambda text, voice_name, ac, ast, ase, ap, bc, bst, bse, bp, enhance_flag: run_compare(
                        voice_map, text, voice_name, ac, ast, ase, ap, bc, bst, bse, bp, enhance_flag
                    ),
                    inputs=[compare_text, compare_voice, a_cfg, a_steps, a_seed, a_prefill, b_cfg, b_steps, b_seed, b_prefill, compare_enhance_audio],
                    outputs=[a_audio, a_stats, b_audio, b_stats],
                )

            with gr.Tab("System"):
                with gr.Row():
                    with gr.Column():
                        sys_gpu = gr.Textbox(label="GPU info", value=get_gpu_info(), lines=3, interactive=False)
                        sys_vram = gr.Textbox(label="VRAM", lines=1, interactive=False)
                        with gr.Row():
                            sys_refresh = gr.Button("Refresh")
                            sys_clear = gr.Button("Clear Cache")
                            sys_unload = gr.Button("Unload Model", variant="stop")

                        sys_refresh.click(fn=lambda: (get_gpu_info(), refresh_vram()), outputs=[sys_gpu, sys_vram])
                        sys_clear.click(fn=clear_vram_cache, outputs=[sys_vram])
                        sys_unload.click(fn=unload_model, outputs=[sys_vram])

                    with gr.Column():
                        model_info = gr.Textbox(label="Model info", lines=7, interactive=False)
                        model_info_btn = gr.Button("Show Model Info")
                        model_info_btn.click(fn=get_model_info, outputs=[model_info])

                        output_list = gr.Textbox(label="Latest .wav files", lines=8, interactive=False)
                        list_outputs_btn = gr.Button("Refresh Output List")
                        list_outputs_btn.click(fn=list_outputs, outputs=[output_list])

        gr.Markdown(
            """
            ---
            Output files are saved to `outputs/`.
            """,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=False,
        inbrowser=True,
    )
