import argparse
import json
import os
import time
import traceback

import torch

from vibevoice.acceleration import (
    DEFAULT_DDPM_STEPS,
    build_model_load_kwargs,
    configure_torch_runtime,
    format_dtype,
    maybe_compile_language_model,
)
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch TTS generation with VibeVoice.")
    parser.add_argument(
        "--json_input",
        type=str,
        required=True,
        help='JSON list like: [{"text": "...", "output_path": "..."}]',
    )
    parser.add_argument(
        "--speaker_voice",
        type=str,
        default="demo/voices/en-Alice_woman.wav",
        help="Reference voice audio path.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="vibevoice/VibeVoice-4bit",
        help="Local model directory.",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        help="Attention backend to use.",
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=DEFAULT_DDPM_STEPS,
        help="Diffusion inference steps. Lower is faster.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.3,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Compile the language model path for repeated non-quantized inference.",
    )
    return parser


def parse_tasks(json_input: str) -> list[dict]:
    tasks = json.loads(json_input)
    if not isinstance(tasks, list):
        raise ValueError("JSON input must be a list.")
    return tasks


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def seed_everything() -> None:
    configure_torch_runtime()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def run_batch_tts() -> None:
    args = build_parser().parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = resolve_path(script_dir, args.model_path)
    speaker_voice_path = resolve_path(script_dir, args.speaker_voice)

    if not os.path.isdir(model_path):
        print(f"Error: model path does not exist: {model_path}")
        return
    if not os.path.exists(speaker_voice_path):
        print(f"Error: reference voice not found: {speaker_voice_path}")
        return

    try:
        tasks = parse_tasks(args.json_input)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Error: invalid JSON input: {exc}")
        return

    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    seed_everything()

    print(f"\nLoading processor and model from: {model_path}")
    try:
        processor = VibeVoiceProcessor.from_pretrained(model_path)
        load_kwargs, load_meta = build_model_load_kwargs(
            model_path=model_path,
            use_4bit=False,
            attn_impl=args.attn_impl,
        )
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(model_path, **load_kwargs)
        model.eval()
        model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)
        model, compile_status = maybe_compile_language_model(model, args.use_compile)
        print(
            "Model loaded."
            f" attn={load_meta['attn_impl']} ({load_meta['attn_reason']})"
            f", dtype={format_dtype(load_meta['torch_dtype'])}"
            f", quantized={'yes' if load_meta['is_quantized'] else 'no'}"
            f", compile={compile_status}"
            f", ddpm_steps={args.ddpm_steps}"
        )
    except Exception as exc:
        print(f"Error: failed to load model: {exc}")
        traceback.print_exc()
        return

    total_tasks = len(tasks)
    autocast_enabled = torch.cuda.is_available() and load_meta["torch_dtype"] in (torch.float16, torch.bfloat16)

    for index, task in enumerate(tasks, start=1):
        text = task.get("text")
        output_path = task.get("output_path")
        if not text or not output_path:
            print(f"Skipping task {index}/{total_tasks}: missing text or output_path.")
            continue

        print(f"\n--- Task {index}/{total_tasks} ---")
        print(f"Text: {text.strip()[:80]}...")
        print(f"Output: {output_path}")

        try:
            inputs = processor(
                text=[f"Speaker 1: {text.strip()}"],
                voice_samples=[[speaker_voice_path]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(DEVICE, non_blocking=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            with torch.inference_mode():
                if autocast_enabled:
                    with torch.amp.autocast("cuda", dtype=load_meta["torch_dtype"]):
                        outputs = model.generate(
                            **inputs,
                            cfg_scale=args.cfg_scale,
                            tokenizer=processor.tokenizer,
                            generation_config={"do_sample": False},
                            is_prefill=True,
                            max_new_tokens=None,
                            verbose=False,
                            show_progress_bar=False,
                        )
                else:
                    outputs = model.generate(
                        **inputs,
                        cfg_scale=args.cfg_scale,
                        tokenizer=processor.tokenizer,
                        generation_config={"do_sample": False},
                        is_prefill=True,
                        max_new_tokens=None,
                        verbose=False,
                        show_progress_bar=False,
                    )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_time = time.time() - start_time

            if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
                print("Generation produced no audio.")
                continue

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            processor.save_audio(outputs.speech_outputs[0], output_path=output_path)

            sample_rate = 24000
            audio_samples = outputs.speech_outputs[0].shape[-1]
            audio_duration = audio_samples / sample_rate
            rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")
            print(
                f"Saved audio. generation_time={generation_time:.2f}s, "
                f"audio_duration={audio_duration:.2f}s, RTF={rtf:.2f}x"
            )
        except Exception as exc:
            print(f"Task failed: {exc}")
            traceback.print_exc()

    print("\nAll tasks finished.")


if __name__ == "__main__":
    run_batch_tts()
