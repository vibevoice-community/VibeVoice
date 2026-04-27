import os
import re
import time
import traceback
from typing import Optional

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
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SAMPLE_RATE = 24000
VOICES_DIR = os.path.join(SCRIPT_DIR, "demo", "voices")
TEXT_EXAMPLES_DIR = os.path.join(SCRIPT_DIR, "demo", "text_examples")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

configure_torch_runtime()

_model = None
_processor = None
_current_model_key = None
_current_load_meta = None
_current_compile_status = "disabled"


def get_available_models() -> list[str]:
    models: list[str] = []
    root = os.path.join(SCRIPT_DIR, "vibevoice")
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                models.append(f"vibevoice/{name}")
    return models or ["vibevoice/VibeVoice-1.5B"]


def get_available_voices() -> list[str]:
    voices: list[str] = []
    if os.path.isdir(VOICES_DIR):
        for name in sorted(os.listdir(VOICES_DIR)):
            if name.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                voices.append(os.path.join(VOICES_DIR, name))
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


def current_autocast_dtype() -> Optional[torch.dtype]:
    if not torch.cuda.is_available() or _current_load_meta is None:
        return None
    dtype = _current_load_meta.get("torch_dtype")
    if dtype in (torch.float16, torch.bfloat16):
        return dtype
    return None


def model_status_summary(model_name: str, ddpm_steps: int) -> str:
    if _current_load_meta is None:
        return f"Model loaded: {model_name}\nDDPM: {ddpm_steps}"

    parts = [
        f"Model loaded: {model_name}",
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


def load_model(
    model_name: str,
    use_4bit: bool,
    attn_impl: str,
    ddpm_steps: int,
    use_compile: bool,
    progress=gr.Progress(),
):
    global _model, _processor, _current_model_key, _current_load_meta, _current_compile_status

    model_path = os.path.join(SCRIPT_DIR, model_name)
    if not os.path.isdir(model_path):
        return f"Model path does not exist: {model_path}"

    cache_key = f"{model_name}|4bit={use_4bit}|attn={attn_impl}|compile={use_compile}"
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        progress(0.2, desc="Loading processor...")
        _processor = VibeVoiceProcessor.from_pretrained(model_path)

        progress(0.4, desc="Loading model weights...")
        load_kwargs, load_meta = build_model_load_kwargs(
            model_path=model_path,
            use_4bit=use_4bit,
            attn_impl=attn_impl,
        )

        _model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            **load_kwargs,
        )
        _model.eval()

        progress(0.8, desc="Applying inference settings...")
        _model.set_ddpm_inference_steps(num_steps=ddpm_steps)

        if use_compile:
            progress(0.9, desc="Compiling language model...")
        _model, compile_status = maybe_compile_language_model(_model, use_compile)

        _current_model_key = cache_key
        _current_load_meta = load_meta
        _current_compile_status = compile_status
        return model_status_summary(model_name, ddpm_steps)
    except Exception:
        _model = None
        _processor = None
        _current_model_key = None
        _current_load_meta = None
        _current_compile_status = "disabled"
        return f"Model load failed:\n{traceback.format_exc()}"


def resolve_voice_path(voice_preset: str, voice_upload: Optional[str]) -> Optional[str]:
    if voice_upload and os.path.exists(voice_upload):
        return voice_upload
    if voice_preset and os.path.exists(voice_preset):
        return voice_preset
    return None


def generate_speech(
    text: str,
    voice_preset: str,
    voice_upload: Optional[str],
    cfg_scale: float,
    ddpm_steps: int,
    seed: int,
    enable_prefill: bool,
    do_sample: bool,
    progress=gr.Progress(),
):
    global _model, _processor

    if _model is None or _processor is None:
        return None, "Please load a model first."
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
        voice_samples = [voice_path]

        progress(0.2, desc="Preparing inputs...")
        inputs = _processor(
            text=[full_script],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
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
        progress(0.9, desc="Saving audio...")

        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            return None, "Model returned no audio output."

        audio_tensor = outputs.speech_outputs[0]
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.detach().cpu().float().numpy()
        else:
            audio_np = np.asarray(audio_tensor, dtype=np.float32)

        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"webui_{timestamp}.wav")
        _processor.save_audio(outputs.speech_outputs[0], output_path=output_path)

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

        return (SAMPLE_RATE, audio_np), "\n".join(stats_lines)
    except Exception:
        return None, f"Generation failed:\n{traceback.format_exc()}"


def get_gpu_info() -> str:
    if not torch.cuda.is_available():
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
    global _model, _processor, _current_model_key, _current_load_meta, _current_compile_status

    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None

    _current_model_key = None
    _current_load_meta = None
    _current_compile_status = "disabled"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return "Model unloaded."


def preview_voice(voice_map: dict[str, str], name: str):
    if name and name in voice_map:
        return voice_map[name]
    return None


def on_generate(
    voice_map: dict[str, str],
    text: str,
    voice_name: str,
    voice_up: Optional[str],
    cfg: float,
    steps: int,
    seed: int,
    prefill: bool,
    sample: bool,
):
    voice_path = voice_map.get(voice_name, "") if voice_name else ""
    return generate_speech(text, voice_path, voice_up, cfg, steps, int(seed), prefill, sample)


def batch_generate(voice_map: dict[str, str], files, voice_name, cfg, steps, seed, progress=gr.Progress()):
    if _model is None or _processor is None:
        return "Please load a model first."
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
            _, stats = generate_speech(text, voice_path, None, cfg, steps, int(seed), True, False)
            logs.append(stats)
        except Exception as exc:
            logs.append(f"Failed: {exc}")

    logs.append(f"\nBatch generation finished. Files: {total}")
    return "\n".join(logs)


def run_compare(voice_map: dict[str, str], text, voice_name, a_cfg, a_steps, a_seed, a_prefill, b_cfg, b_steps, b_seed, b_prefill):
    voice_path = voice_map.get(voice_name, "") if voice_name else ""
    a_audio, a_stats = generate_speech(text, voice_path, None, a_cfg, a_steps, int(a_seed), a_prefill, False)
    b_audio, b_stats = generate_speech(text, voice_path, None, b_cfg, b_steps, int(b_seed), b_prefill, False)
    return a_audio, a_stats, b_audio, b_stats


def get_model_info() -> str:
    if _model is None:
        return "No model loaded."

    lines = [f"Current key: {_current_model_key}"]
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
            Fast local testing UI for VibeVoice models.

            Recommended on this machine:
            `VibeVoice-1.5B + flash_attention_2 + DDPM 8`
            """,
            elem_classes=["header-text"],
        )

        with gr.Row():
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
                            label="Model",
                            choices=available_models,
                            value=available_models[0] if available_models else None,
                            interactive=True,
                        )
                        with gr.Row():
                            use_4bit = gr.Checkbox(
                                label="4bit quantization",
                                value=False,
                                info="Use BitsAndBytes 4bit loading when supported.",
                            )
                            use_compile = gr.Checkbox(
                                label="torch.compile",
                                value=True,
                                info="Usually skip on Windows unless you are testing it.",
                            )

                        attn_impl = gr.Dropdown(
                            label="Attention backend",
                            choices=["auto", "flash_attention_2", "sdpa", "eager"],
                            value="auto",
                            info="Auto picks flash_attention_2 when flash-attn is installed, otherwise SDPA.",
                        )
                        ddpm_steps_load = gr.Slider(
                            label="DDPM steps",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=DEFAULT_DDPM_STEPS,
                            info="Lower is faster. 8-10 is a good starting point.",
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
                    lines=6,
                    elem_classes=["status-box"],
                )

                load_btn.click(
                    fn=load_model,
                    inputs=[model_selector, use_4bit, attn_impl, ddpm_steps_load, use_compile],
                    outputs=[model_status],
                )
                unload_btn.click(fn=unload_model, outputs=[model_status])
                clear_cache_btn.click(fn=clear_vram_cache, outputs=[vram_display])
                refresh_vram_btn.click(fn=refresh_vram, outputs=[vram_display])

            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### Text")
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

                        gr.Markdown("### Reference voice")
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
                        gr.Markdown("### Generation settings")
                        cfg_scale = gr.Slider(
                            label="CFG scale",
                            minimum=0.1,
                            maximum=5.0,
                            step=0.1,
                            value=1.3,
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

                        generate_btn = gr.Button("Generate Audio", variant="primary", size="lg")
                        gr.Markdown("### Output")
                        audio_output = gr.Audio(
                            label="Generated audio",
                            type="numpy",
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
                    fn=lambda text, voice_name, voice_up, cfg, steps, seed_value, prefill, sample: on_generate(
                        voice_map, text, voice_name, voice_up, cfg, steps, seed_value, prefill, sample
                    ),
                    inputs=[text_input, voice_selector, voice_upload, cfg_scale, ddpm_steps, seed, enable_prefill, do_sample],
                    outputs=[audio_output, gen_stats],
                )

            with gr.Tab("Batch"):
                gr.Markdown("Generate audio from multiple text files.")
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
                            batch_cfg = gr.Slider(label="CFG", minimum=0.1, maximum=5.0, step=0.1, value=1.3)
                            batch_steps = gr.Slider(label="DDPM steps", minimum=1, maximum=100, step=1, value=DEFAULT_DDPM_STEPS)
                            batch_seed = gr.Number(label="Seed", value=42, precision=0)
                        batch_btn = gr.Button("Start Batch", variant="primary")
                    with gr.Column():
                        batch_log = gr.Textbox(
                            label="Batch log",
                            lines=15,
                            interactive=False,
                            elem_classes=["status-box"],
                        )

                batch_btn.click(
                    fn=lambda files, voice_name, cfg, steps, seed_value: batch_generate(
                        voice_map, files, voice_name, cfg, steps, seed_value
                    ),
                    inputs=[batch_files, batch_voice, batch_cfg, batch_steps, batch_seed],
                    outputs=[batch_log],
                )

            with gr.Tab("Compare"):
                gr.Markdown("A/B compare two parameter sets using the same text.")
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

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Setting A")
                        a_cfg = gr.Slider(label="CFG", minimum=0.1, maximum=5.0, step=0.1, value=1.0)
                        a_steps = gr.Slider(label="DDPM steps", minimum=1, maximum=100, step=1, value=DEFAULT_DDPM_STEPS)
                        a_seed = gr.Number(label="Seed", value=42, precision=0)
                        a_prefill = gr.Checkbox(label="Prefill", value=True)
                        a_audio = gr.Audio(label="Result A", type="numpy", interactive=False)
                        a_stats = gr.Textbox(label="Stats A", lines=6, interactive=False, elem_classes=["status-box"])

                    with gr.Column():
                        gr.Markdown("#### Setting B")
                        b_cfg = gr.Slider(label="CFG", minimum=0.1, maximum=5.0, step=0.1, value=1.5)
                        b_steps = gr.Slider(label="DDPM steps", minimum=1, maximum=100, step=1, value=DEFAULT_DDPM_STEPS)
                        b_seed = gr.Number(label="Seed", value=42, precision=0)
                        b_prefill = gr.Checkbox(label="Prefill", value=True)
                        b_audio = gr.Audio(label="Result B", type="numpy", interactive=False)
                        b_stats = gr.Textbox(label="Stats B", lines=6, interactive=False, elem_classes=["status-box"])

                compare_btn = gr.Button("Run Compare", variant="primary")
                compare_btn.click(
                    fn=lambda text, voice_name, ac, ast, ase, ap, bc, bst, bse, bp: run_compare(
                        voice_map, text, voice_name, ac, ast, ase, ap, bc, bst, bse, bp
                    ),
                    inputs=[compare_text, compare_voice, a_cfg, a_steps, a_seed, a_prefill, b_cfg, b_steps, b_seed, b_prefill],
                    outputs=[a_audio, a_stats, b_audio, b_stats],
                )

            with gr.Tab("System"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### GPU")
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
                        gr.Markdown("### Model")
                        model_info = gr.Textbox(label="Model info", lines=6, interactive=False)
                        model_info_btn = gr.Button("Show Model Info")
                        model_info_btn.click(fn=get_model_info, outputs=[model_info])

                        gr.Markdown("### Recent outputs")
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
    demo.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=False,
        inbrowser=True,
    )
