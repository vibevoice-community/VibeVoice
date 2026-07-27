import argparse
import copy
import os
import time
import traceback

import torch

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SAMPLE_RATE = 24000
DEFAULT_MODEL = "vibevoice/VibeVoice-0.5b"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime streaming TTS with VibeVoice 0.5B.")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text or multi-speaker script.",
    )
    parser.add_argument(
        "--txt_path",
        type=str,
        default=None,
        help="Path to a .txt script file.",
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default="Carter",
        help="Preset speaker name from demo/voices/streaming_model.",
    )
    parser.add_argument(
        "--speaker_prompt",
        type=str,
        default=None,
        help="Optional path to a cached prompt .pt file. Overrides --speaker_name.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output wav path. Defaults to outputs/<timestamp>_streaming.wav.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL,
        help="Local model directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention backend preference.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=5,
        help="DDPM inference steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser


def resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    candidate = os.path.join(SCRIPT_DIR, path)
    if os.path.exists(candidate):
        return candidate
    return path


def detect_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype_and_attn(device: str, attn_impl: str) -> tuple[torch.dtype, str]:
    if device == "cuda":
        return torch.bfloat16, ("flash_attention_2" if attn_impl == "auto" else attn_impl)
    if device == "mps":
        return torch.float32, "sdpa"
    return torch.float32, ("sdpa" if attn_impl == "auto" else attn_impl)


def read_input_text(text: str | None, txt_path: str | None) -> str:
    if text and text.strip():
        return text.strip()
    if txt_path:
        with open(resolve_path(txt_path), "r", encoding="utf-8") as handle:
            return handle.read().strip()
    raise ValueError("Please provide either --text or --txt_path.")


def get_streaming_voice_map() -> dict[str, str]:
    voices_dir = os.path.join(SCRIPT_DIR, "demo", "voices", "streaming_model")
    voice_map: dict[str, str] = {}
    if not os.path.isdir(voices_dir):
        return voice_map

    for name in sorted(os.listdir(voices_dir)):
        if not name.lower().endswith(".pt"):
            continue
        full_path = os.path.join(voices_dir, name)
        if not os.path.isfile(full_path):
            continue
        stem = os.path.splitext(name)[0]
        voice_map[stem] = full_path
        if "_" in stem:
            voice_map.setdefault(stem.split("_")[0], full_path)
        if "-" in stem:
            short_name = stem.split("-")[-1]
            voice_map.setdefault(short_name, full_path)
            if "_" in short_name:
                voice_map.setdefault(short_name.split("_")[0], full_path)
    return voice_map


def resolve_prompt_path(speaker_prompt: str | None, speaker_name: str) -> str:
    if speaker_prompt:
        prompt_path = resolve_path(speaker_prompt)
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Speaker prompt not found: {prompt_path}")
        return prompt_path

    voice_map = get_streaming_voice_map()
    if speaker_name in voice_map:
        return voice_map[speaker_name]

    speaker_lower = speaker_name.lower()
    for name, path in voice_map.items():
        if speaker_lower in name.lower() or name.lower() in speaker_lower:
            return path

    available = ", ".join(sorted(set(voice_map.keys())))
    raise ValueError(f"Unknown speaker_name '{speaker_name}'. Available presets: {available}")


def load_model(model_path: str, device: str, attn_impl: str, torch_dtype: torch.dtype):
    try:
        if device == "mps":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
                device_map=None,
            )
            model.to("mps")
        elif device == "cuda":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="cuda",
                attn_implementation=attn_impl,
            )
        else:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="cpu",
                attn_implementation=attn_impl,
            )
    except Exception:
        if attn_impl == "flash_attention_2":
            print("flash_attention_2 load failed, retrying with sdpa...")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=(device if device in ("cuda", "cpu") else None),
                attn_implementation="sdpa",
            )
            if device == "mps":
                model.to("mps")
        else:
            raise
    return model


def main() -> None:
    args = build_parser().parse_args()
    model_path = resolve_path(args.model_path)
    txt_path = resolve_path(args.txt_path) if args.txt_path else None
    device = detect_device(args.device)
    torch_dtype, attn_impl = choose_dtype_and_attn(device, args.attn_impl)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        full_script = read_input_text(args.text, txt_path)
        prompt_path = resolve_prompt_path(args.speaker_prompt, args.speaker_name)
    except Exception as exc:
        print(f"Input error: {exc}")
        return

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Model: {model_path}")
    print(f"Attention: {attn_impl}")
    print(f"DType: {torch_dtype}")
    print(f"Prompt: {prompt_path}")

    try:
        print("Loading processor...")
        processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
        print("Loading model...")
        model = load_model(model_path, device, attn_impl, torch_dtype)
        model.eval()
        model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)
    except Exception as exc:
        print(f"Model load failed: {exc}")
        traceback.print_exc()
        return

    target_device = device if device != "auto" else "cpu"
    try:
        cached_prompt = torch.load(
            prompt_path,
            map_location=(target_device if target_device != "mps" else "cpu"),
            weights_only=False,
        )
        inputs = processor.process_input_with_cached_prompt(
            text=full_script,
            cached_prompt=cached_prompt,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(device if device != "mps" else "mps")

        if device == "mps":
            cached_prompt = torch.load(prompt_path, map_location="mps", weights_only=False)
    except Exception as exc:
        print(f"Input preparation failed: {exc}")
        traceback.print_exc()
        return

    autocast_enabled = device == "cuda" and torch_dtype in (torch.float16, torch.bfloat16)
    start_time = time.time()
    try:
        with torch.inference_mode():
            if autocast_enabled:
                with torch.amp.autocast("cuda", dtype=torch_dtype):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=args.cfg_scale,
                        tokenizer=processor.tokenizer,
                        generation_config={"do_sample": False},
                        verbose=False,
                        show_progress_bar=False,
                        all_prefilled_outputs=copy.deepcopy(cached_prompt),
                    )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=args.cfg_scale,
                    tokenizer=processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    show_progress_bar=False,
                    all_prefilled_outputs=copy.deepcopy(cached_prompt),
                )
    except Exception as exc:
        print(f"Generation failed: {exc}")
        traceback.print_exc()
        return

    generation_time = time.time() - start_time
    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        print("Generation produced no audio.")
        return

    if args.output_path:
        output_path = resolve_path(args.output_path)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(SCRIPT_DIR, "outputs", f"{timestamp}_streaming.wav")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    processor.save_audio(outputs.speech_outputs[0], output_path=output_path)

    audio_samples = int(outputs.speech_outputs[0].shape[-1])
    audio_duration = audio_samples / SAMPLE_RATE
    rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")
    text_tokens = int(inputs["tts_text_ids"].shape[1])
    output_tokens = int(outputs.sequences.shape[1])
    prompt_tokens = int(cached_prompt["tts_lm"]["last_hidden_state"].size(1))
    generated_tokens = output_tokens - text_tokens - prompt_tokens

    print("Generation complete.")
    print(f"Output: {output_path}")
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"RTF: {rtf:.3f}x")
    print(f"Tokens: text {text_tokens} | generated speech {generated_tokens} | total {output_tokens}")
    print(f"CFG: {args.cfg_scale} | DDPM: {args.ddpm_steps}")


if __name__ == "__main__":
    main()
