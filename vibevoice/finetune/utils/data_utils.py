"""
Data loading and dataset utilities for VibeVoice finetuning.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from datasets import VerificationMode, load_dataset

from data_vibevoice import VibeVoiceCollator, VibeVoiceDataset

logger = logging.getLogger(__name__)


def load_datasets(data_args, model_args) -> Tuple[Any, Optional[Any]]:
    """Load training and validation datasets."""
    verification_mode = (
        VerificationMode.NO_CHECKS
        if data_args.ignore_verifications
        else VerificationMode.BASIC_CHECKS
    )

    # Load raw dataset
    if data_args.train_jsonl is not None:
        data_files: Dict[str, str] = {"train": data_args.train_jsonl}
        if data_args.validation_jsonl is not None:
            data_files["validation"] = data_args.validation_jsonl
        raw = load_dataset(
            "json",
            data_files=data_files,
            verification_mode=verification_mode,
            cache_dir=model_args.cache_dir,
        )
    else:
        if data_args.dataset_name is None:
            raise ValueError(
                "Provide --dataset_name (HF datasets) or use --train_jsonl/--validation_jsonl for local files."
            )
        raw = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            verification_mode=verification_mode,
            cache_dir=model_args.cache_dir,
        )

    train_ds = raw[data_args.train_split_name]
    eval_ds = None

    return train_ds, eval_ds, raw


def setup_evaluation_dataset(train_ds, raw, data_args, training_args):
    """Setup evaluation dataset if needed."""
    eval_ds = None

    if training_args.do_eval:
        if data_args.eval_split_name and data_args.eval_split_name in raw:
            eval_ds = raw[data_args.eval_split_name]
        elif (
            data_args.eval_split_size
            and data_args.eval_split_size > 0
            and len(train_ds) > 1
        ):
            split = train_ds.train_test_split(
                test_size=data_args.eval_split_size, seed=training_args.seed
            )
            train_ds, eval_ds = split["train"], split["test"]

    return train_ds, eval_ds


def create_datasets(train_ds, eval_ds, data_args):
    """Create VibeVoice dataset wrappers."""
    train_dataset = VibeVoiceDataset(
        train_ds,
        text_column=data_args.text_column_name,
        audio_column=data_args.audio_column_name,
        voice_prompts_column=data_args.voice_prompts_column_name,
    )

    eval_dataset = None
    if eval_ds is not None:
        eval_dataset = VibeVoiceDataset(
            eval_ds,
            text_column=data_args.text_column_name,
            audio_column=data_args.audio_column_name,
            voice_prompts_column=data_args.voice_prompts_column_name,
        )

    return train_dataset, eval_dataset


def get_model_dimensions(processor, model):
    """Get model dimensions needed for data collator."""
    speech_compress_ratio = getattr(processor, "speech_tok_compress_ratio", 3200)
    semantic_dim = getattr(model.config, "semantic_vae_dim", None)

    if semantic_dim is None:
        try:
            semantic_dim = int(
                getattr(model.config.semantic_tokenizer_config, "vae_dim", 128)
            )
        except Exception:
            semantic_dim = 128

    return speech_compress_ratio, semantic_dim


def create_data_collator(processor, model, data_args):
    """Create data collator for VibeVoice training."""
    speech_compress_ratio, semantic_dim = get_model_dimensions(processor, model)

    compute_semantics_flag = (
        hasattr(processor, "semantic_tokenizer")
        and processor.semantic_tokenizer is not None
    )

    data_collator = VibeVoiceCollator(
        processor=processor,
        max_length=data_args.max_length,
        speech_compress_ratio=speech_compress_ratio,
        semantic_vae_dim=semantic_dim,
        compute_semantics=compute_semantics_flag,
        debug_checks=False,
        voice_prompt_drop_rate=data_args.voice_prompt_drop_rate,
    )

    return data_collator


def setup_data_pipeline(data_args, model_args, training_args, processor, model):
    """Complete data pipeline setup."""
    logger.info("Loading datasets...")
    train_ds, eval_ds, raw = load_datasets(data_args, model_args)

    logger.info("Setting up evaluation split...")
    train_ds, eval_ds = setup_evaluation_dataset(
        train_ds, raw, data_args, training_args
    )

    logger.info("Creating dataset wrappers...")
    train_dataset, eval_dataset = create_datasets(train_ds, eval_ds, data_args)

    logger.info("Creating data collator...")
    data_collator = create_data_collator(processor, model, data_args)

    return train_dataset, eval_dataset, data_collator
