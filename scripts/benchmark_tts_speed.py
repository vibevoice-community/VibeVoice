#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vibevoice.acceleration import (
    build_model_load_kwargs,
    configure_torch_runtime,
    format_dtype,
    maybe_compile_language_model,
)
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VibeVoice speed on local models.")
    parser.add_argument("--model-path", required=True, help="Local model directory.")
    parser.add_argument("--voice", required=True, help="Reference voice WAV path.")
    parser.add_argument("--text", required=True, help="Input text.")
    parser.add_argument("--attn", default="sdpa", choices=["auto", "sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--ddpm-steps", type=int, nargs="+", default=[10, 8])
    parser.add_argument("--compile", action="store_true", help="Compile the language model for repeated runs.")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def prepare_inputs(processor: VibeVoiceProcessor, text: str, voice: str, device: str):
    inputs = processor(
        text=[f"Speaker 1: {text.strip()}"],
        voice_samples=[[voice]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(device, non_blocking=True)
    return inputs


def run_once(model, processor, inputs, use_autocast: bool, autocast_dtype: torch.dtype) -> tuple[float, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    with torch.inference_mode():
        if use_autocast:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                outputs = model.generate(
                    **inputs,
                    cfg_scale=1.3,
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
                cfg_scale=1.3,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                is_prefill=True,
                max_new_tokens=None,
                verbose=False,
                show_progress_bar=False,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    audio_duration = outputs.speech_outputs[0].shape[-1] / 24000
    return elapsed, audio_duration


def main() -> int:
    args = parse_args()
    configure_torch_runtime()

    model_path = Path(args.model_path).resolve()
    voice_path = Path(args.voice).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model path not found: {model_path}")
    if not voice_path.exists():
        raise SystemExit(f"Voice path not found: {voice_path}")

    processor = VibeVoiceProcessor.from_pretrained(model_path.as_posix())
    load_kwargs, load_meta = build_model_load_kwargs(
        model_path=model_path.as_posix(),
        use_4bit=False,
        attn_impl=args.attn,
    )
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(model_path.as_posix(), **load_kwargs)
    model.eval()
    model, compile_status = maybe_compile_language_model(model, args.compile)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = prepare_inputs(processor, args.text, voice_path.as_posix(), device)
    use_autocast = torch.cuda.is_available() and load_meta["torch_dtype"] in (torch.float16, torch.bfloat16)

    print(
        f"Loaded {model_path.name} | attn={load_meta['attn_impl']} | "
        f"dtype={format_dtype(load_meta['torch_dtype'])} | compile={compile_status}"
    )

    results = []
    for ddpm_steps in args.ddpm_steps:
        model.set_ddpm_inference_steps(num_steps=ddpm_steps)
        for _ in range(args.warmup_runs):
            run_once(model, processor, inputs, use_autocast, load_meta["torch_dtype"])

        elapsed_values = []
        audio_duration = None
        for _ in range(args.timed_runs):
            elapsed, audio_duration = run_once(model, processor, inputs, use_autocast, load_meta["torch_dtype"])
            elapsed_values.append(elapsed)

        avg_elapsed = sum(elapsed_values) / len(elapsed_values)
        rtf = avg_elapsed / audio_duration if audio_duration else float("inf")
        row = {
            "model": model_path.name,
            "ddpm_steps": ddpm_steps,
            "attn_impl": load_meta["attn_impl"],
            "dtype": format_dtype(load_meta["torch_dtype"]),
            "compile": compile_status,
            "avg_generation_sec": round(avg_elapsed, 3),
            "audio_duration_sec": round(audio_duration, 3),
            "rtf": round(rtf, 3),
        }
        results.append(row)
        print(
            f"ddpm={ddpm_steps}: avg_generation={row['avg_generation_sec']}s, "
            f"audio_duration={row['audio_duration_sec']}s, rtf={row['rtf']}x"
        )

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved benchmark report to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
