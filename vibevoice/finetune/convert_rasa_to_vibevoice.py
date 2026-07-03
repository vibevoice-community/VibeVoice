#!/usr/bin/env python
"""Convert the Rasa expressive-TTS dataset into a VibeVoice fine-tuning dataset.

The `Rasa` corpus (expressive multi-style speech for Indian languages) ships as
HuggingFace parquet shards whose rows look like::

    filename : "TEL_F_CONV_00564"
    text     : "<utterance>"
    language : "Telugu"
    gender   : "Female"
    style    : "CONV" | "WIKI" | "HAPPY" | "SAD" | ...
    duration : "4.775"                       # seconds, as a string
    wav_path : "/original/abs/path.wav"       # not usable locally
    audio    : Audio feature -> {array, sampling_rate, path}

VibeVoice fine-tuning (see FINETUNING.md) expects a dataset with just two columns:

    text  -> "Speaker 0: <utterance>"    # the "Speaker N:" prefix is REQUIRED
    audio -> a HuggingFace ``Audio`` feature (decoded to {array, sampling_rate})

Every clip is a single-speaker utterance, so its transcript is always labelled
``Speaker 0:``. (VibeVoice's processor is 0-indexed and normalises the lowest
speaker id in each sample to 0, so the lone speaker of a one-utterance sample is
always speaker 0 -- the real voice identity is supplied by the voice prompt, not
the text label.) This script reads Rasa parquet shards (or a hub dataset id),
resamples the audio to 24 kHz, and writes a VibeVoice-ready dataset. Optional
language/gender/style/duration filters are available for subsetting; nothing is
filtered by default, so every voice is kept.

Two output formats are supported:

  * ``--format hf``   (default) a directory of parquet files the trainer loads
    directly via ``--dataset_name <outdir>``.
  * ``--format jsonl`` extracted ``.wav`` files plus a JSONL manifest usable via
    ``--train_jsonl <outdir>/train.jsonl``.

Example
-------
    python -m vibevoice.finetune.convert_rasa_to_vibevoice \
        --input data/rasa \
        --output data/rasa_vibevoice \
        --val-split 0.02
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import List, Optional


TARGET_SR_DEFAULT = 24000


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert the Rasa TTS dataset into a VibeVoice fine-tuning dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        default="data/rasa",
        help="Rasa source: a directory of *.parquet shards, a single .parquet file, "
        "or a HuggingFace hub dataset id.",
    )
    p.add_argument(
        "--output",
        default="data/rasa_vibevoice",
        help="Output directory for the VibeVoice-compatible dataset.",
    )
    p.add_argument(
        "--format",
        choices=["hf", "jsonl"],
        default="hf",
        help="hf: parquet dataset dir (load via --dataset_name). "
        "jsonl: wavs + manifest (load via --train_jsonl).",
    )

    # --- Optional subsetting filters (NOT applied by default; all voices kept). ---
    p.add_argument("--language", default=None, help="Keep only this language (e.g. Telugu). Case-insensitive.")
    p.add_argument("--gender", default=None, help="Keep only this gender (e.g. Female). Case-insensitive.")
    p.add_argument(
        "--style",
        default=None,
        help="Comma-separated styles to keep (e.g. CONV,WIKI,BOOK). Default: keep all.",
    )
    p.add_argument("--min-duration", type=float, default=None, help="Drop clips shorter than this (seconds).")
    p.add_argument("--max-duration", type=float, default=None, help="Drop clips longer than this (seconds).")

    # --- Formatting / output shaping ---
    p.add_argument("--target-sr", type=int, default=TARGET_SR_DEFAULT, help="Resample audio to this rate.")
    p.add_argument("--val-split", type=float, default=0.0, help="Fraction held out as a validation split (0 disables).")
    p.add_argument("--seed", type=int, default=42, help="Shuffle/split seed.")
    p.add_argument("--max-samples", type=int, default=None, help="Cap number of samples (after filtering) for quick tests.")
    p.add_argument(
        "--split",
        default=None,
        help="When --input is a hub id with multiple splits, use only this split "
        "(default: concatenate all).",
    )
    p.add_argument("--num-proc", type=int, default=1, help="Parallel workers for the map/decoding step.")
    return p.parse_args(argv)


def load_source(input_path: str, split: Optional[str]):
    """Load the Rasa dataset from local parquet files or a hub id into one Dataset."""
    from datasets import DatasetDict, concatenate_datasets, load_dataset

    if os.path.exists(input_path):
        if os.path.isdir(input_path):
            files = sorted(glob.glob(os.path.join(input_path, "**", "*.parquet"), recursive=True))
            if not files:
                raise SystemExit(f"No .parquet files found under {input_path!r}.")
        else:
            files = [input_path]
        print(f"Loading {len(files)} parquet file(s) from {input_path!r} ...", file=sys.stderr)
        return load_dataset("parquet", data_files={"data": files})["data"]

    # Treat as a HuggingFace hub dataset id.
    print(f"{input_path!r} is not a local path; loading as a hub dataset id ...", file=sys.stderr)
    ds = load_dataset(input_path)
    if isinstance(ds, DatasetDict):
        if split is not None:
            return ds[split]
        return concatenate_datasets([ds[s] for s in ds])
    return ds


def build_filter(args: argparse.Namespace):
    styles = None
    if args.style:
        styles = {s.strip().upper() for s in args.style.split(",") if s.strip()}

    def keep(row) -> bool:
        if args.language and str(row.get("language", "")).strip().lower() != args.language.strip().lower():
            return False
        if args.gender and str(row.get("gender", "")).strip().lower() != args.gender.strip().lower():
            return False
        if styles is not None and str(row.get("style", "")).strip().upper() not in styles:
            return False
        text = row.get("text")
        if text is None or not str(text).strip():
            return False
        dur = row.get("duration")
        if (args.min_duration is not None or args.max_duration is not None) and dur is not None:
            try:
                d = float(dur)
            except (TypeError, ValueError):
                d = None
            if d is not None:
                if args.min_duration is not None and d < args.min_duration:
                    return False
                if args.max_duration is not None and d > args.max_duration:
                    return False
        return True

    return keep


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    from datasets import Audio

    ds = load_source(args.input, args.split)
    n_raw = len(ds)

    keep = build_filter(args)
    ds = ds.filter(keep, num_proc=args.num_proc if args.num_proc > 1 else None)
    n_filtered = len(ds)
    if n_filtered == 0:
        raise SystemExit("No rows left after filtering. Loosen --language/--gender/--style/--*-duration.")

    # Shuffle then optionally cap for reproducible quick runs.
    ds = ds.shuffle(seed=args.seed)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Decode + resample audio to the target rate on read.
    ds = ds.cast_column("audio", Audio(sampling_rate=args.target_sr))

    # Informational only: how many distinct voices the data spans. This does NOT
    # affect the text label -- every utterance is a single-speaker turn and is
    # always labelled "Speaker 0:" (see module docstring).
    voice_cols = [c for c in ("language", "gender") if c in ds.column_names]
    if voice_cols:
        counts = Counter(zip(*[ds[c] for c in voice_cols]))
        voice_lines = [
            ("/".join(str(x) for x in key), n)
            for key, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
    else:
        voice_lines = []

    keep_meta = [c for c in ("filename", "language", "gender", "style", "duration") if c in ds.column_names]

    def to_vibevoice(row):
        out = {
            "text": "Speaker 0: " + str(row["text"]).strip(),
            "audio": row["audio"],
        }
        for c in keep_meta:
            out[c] = row[c]
        return out

    remove_cols = [c for c in ds.column_names if c not in (["text", "audio"] + keep_meta)]
    ds = ds.map(
        to_vibevoice,
        remove_columns=remove_cols,
        num_proc=args.num_proc if args.num_proc > 1 else None,
        desc="Formatting for VibeVoice",
    )

    # Build train/validation splits.
    if args.val_split and args.val_split > 0 and len(ds) > 1:
        split = ds.train_test_split(test_size=args.val_split, seed=args.seed)
        splits = {"train": split["train"], "validation": split["test"]}
    else:
        splits = {"train": ds}

    os.makedirs(args.output, exist_ok=True)
    if args.format == "hf":
        write_hf(splits, args)
    else:
        write_jsonl(splits, args)

    print_summary(splits, args, n_raw, n_filtered, voice_lines)


def write_hf(splits: dict, args: argparse.Namespace) -> None:
    """Write parquet shards the trainer can load via --dataset_name <output>."""
    for name, split_ds in splits.items():
        out_file = os.path.join(args.output, f"{name}-00000-of-00001.parquet")
        split_ds.to_parquet(out_file)
        print(f"Wrote {len(split_ds):>6} rows -> {out_file}", file=sys.stderr)


def write_jsonl(splits: dict, args: argparse.Namespace) -> None:
    """Write .wav files + a JSONL manifest usable via --train_jsonl."""
    import soundfile as sf

    wav_dir = os.path.join(args.output, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    for name, split_ds in splits.items():
        manifest = os.path.join(args.output, f"{name}.jsonl")
        n = 0
        with open(manifest, "w", encoding="utf-8") as fh:
            for i, row in enumerate(split_ds):
                audio = row["audio"]
                base = (
                    str(row.get("filename") or "").strip()
                    or os.path.splitext(os.path.basename(audio.get("path") or ""))[0]
                    or f"{name}_{i:06d}"
                )
                wav_path = os.path.abspath(os.path.join(wav_dir, f"{base}.wav"))
                sf.write(wav_path, audio["array"], audio["sampling_rate"], subtype="PCM_16")
                fh.write(json.dumps({"text": row["text"], "audio": wav_path}, ensure_ascii=False) + "\n")
                n += 1
        print(f"Wrote {n:>6} rows -> {manifest}  (wavs in {wav_dir})", file=sys.stderr)


def print_summary(splits: dict, args: argparse.Namespace, n_raw: int, n_filtered: int, voice_lines) -> None:
    train_n = len(splits["train"])
    val_n = len(splits.get("validation", []))
    print("\n" + "=" * 68)
    print("Rasa -> VibeVoice conversion complete")
    print("=" * 68)
    print(f"  source rows           : {n_raw}")
    print(f"  after filtering       : {n_filtered}")
    print(f"  train / validation    : {train_n} / {val_n}")
    print(f"  output ({args.format:>5}) dir     : {args.output}")
    print(f"  audio sample rate     : {args.target_sr} Hz")
    print('  text label            : "Speaker 0: <utterance>" (all rows)')
    if voice_lines:
        print(f"  voices in data ({len(voice_lines)}, by language/gender):")
        for key, n in voice_lines:
            print(f"      {key}  (n={n})")
    print("-" * 68)
    print("Train with, e.g.:\n")
    if args.format == "hf":
        data_flags = f"    --dataset_name {args.output} \\\n"
        if val_n:
            data_flags += "    --eval_split_name validation \\\n"
    else:
        data_flags = f"    --train_jsonl {os.path.join(args.output, 'train.jsonl')} \\\n"
        if val_n:
            data_flags += f"    --validation_jsonl {os.path.join(args.output, 'validation.jsonl')} \\\n"
    print(
        "  python -m vibevoice.finetune.train_vibevoice \\\n"
        "    --model_name_or_path vibevoice/VibeVoice-1.5B \\\n"
        f"{data_flags}"
        "    --text_column_name text \\\n"
        "    --audio_column_name audio \\\n"
        "    --voice_prompts_column_name audio \\\n"
        "    --output_dir finetune_rasa \\\n"
        "    --do_train --bf16 True --remove_unused_columns False"
    )
    print("=" * 68)


if __name__ == "__main__":
    main()
