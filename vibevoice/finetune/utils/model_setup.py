"""
Model setup and configuration utilities for VibeVoice finetuning.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from vibevoice.modular.modeling_vibevoice import \
    VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

from .model_introspection import (ModelComponents, get_config_value,
                                  safe_get_attr)

logger = logging.getLogger(__name__)


def load_processor(model_args) -> VibeVoiceProcessor:
    """Load and validate VibeVoice processor."""
    processor_path = model_args.processor_name_or_path or model_args.model_name_or_path
    if processor_path is None:
        raise ValueError(
            "--model_name_or_path (or --processor_name_or_path) must be provided"
        )

    processor = VibeVoiceProcessor.from_pretrained(processor_path)

    # Validate required special tokens
    tok = processor.tokenizer
    for required in ["speech_start_id", "speech_diffusion_id", "speech_end_id"]:
        if not hasattr(tok, required) or getattr(tok, required) is None:
            raise RuntimeError(f"Tokenizer missing required special id: {required}")

    return processor


def load_model(model_args, training_args) -> VibeVoiceForConditionalGeneration:
    """Load and configure VibeVoice model."""
    if model_args.model_name_or_path is None:
        raise ValueError(
            "--model_name_or_path is required to load VibeVoice base model"
        )

    # Determine dtype
    dtype = torch.float32
    if training_args.bf16:
        dtype = torch.bfloat16
    elif getattr(training_args, "fp16", False):
        dtype = torch.float16

    model = VibeVoiceForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=dtype,
    )

    return model


def patch_acoustic_encode_for_legacy_indexing(model, logger_):
    """Patch acoustic tokenizer encode method for legacy indexing compatibility."""
    components = ModelComponents(model)
    acoustic = components.acoustic_tokenizer

    if acoustic is None or not hasattr(acoustic, "encode"):
        logger_.warning("No acoustic_tokenizer.encode() found to patch.")
        return

    try:
        base_encode = acoustic.encode

        def encode_wrapped(*args, **kwargs):
            out = base_encode(*args, **kwargs)
            try:
                _ = out[0][0]
                return out
            except Exception:
                pass
            if isinstance(out, dict):
                for k in ("frames", "codes", "tokens", "latents", "hidden_states"):
                    if k in out:
                        return [[out[k]]]
                if len(out) > 0:
                    return [[next(iter(out.values()))]]
            for attr in ("frames", "codes", "tokens", "latents", "hidden_states"):
                if hasattr(out, attr):
                    return [[getattr(out, attr)]]
            try:
                if isinstance(out, torch.Tensor):
                    return [[out]]
            except Exception:
                pass
            return [[out]]

        acoustic.encode = encode_wrapped
        logger_.info(
            "Patched acoustic_tokenizer.encode() to return [[...]] for legacy indexing."
        )
    except Exception as e:
        logger_.warning(f"Failed to patch acoustic_tokenizer.encode(): {e}")


def setup_model_cache_and_tokenizers(model, model_args, training_args):
    """Configure model caching and freeze tokenizers as needed."""
    # Disable cache during training
    if hasattr(model.config, "use_cache") and training_args.do_train:
        model.config.use_cache = False

    # Freeze tokenizers using cleaner API
    components = ModelComponents(model)
    if model_args.freeze_acoustic_tokenizer:
        components.freeze_component("acoustic_tokenizer")
    if model_args.freeze_semantic_tokenizer:
        components.freeze_component("semantic_tokenizer")


def hard_tie_lm_head(model):
    """Force-tie LM head weights to input embeddings."""
    components = ModelComponents(model)
    emb_module = components.input_embeddings
    head_module = components.output_embeddings

    try:
        if (
            emb_module
            and head_module
            and hasattr(emb_module, "weight")
            and hasattr(head_module, "weight")
        ):
            if (
                emb_module.weight.shape == head_module.weight.shape
                and emb_module.weight.data_ptr() != head_module.weight.data_ptr()
            ):
                with torch.no_grad():
                    head_module.weight = emb_module.weight
                logger.info(
                    "Force-tied LM head weight to input embeddings (pointer share)."
                )
    except Exception as e:
        logger.warning(f"Force-tie of LM head failed: {e}")


def freeze_embeddings_and_head(model):
    """Freeze input embeddings and output head."""
    components = ModelComponents(model)

    try:
        emb = components.input_embeddings
        if emb and hasattr(emb, "weight"):
            emb.weight.requires_grad_(False)

        head = components.output_embeddings
        if head and hasattr(head, "weight"):
            head.weight.requires_grad_(False)
    except Exception:
        pass


def log_lm_head_diagnostics(model):
    """Log diagnostics about LM head weight tying."""
    components = ModelComponents(model)

    try:
        in_emb_mod = components.input_embeddings
        out_emb_mod = components.output_embeddings
        in_w = safe_get_attr(in_emb_mod, "weight")
        out_w = safe_get_attr(out_emb_mod, "weight")

        shared_ptr = bool(
            in_w is not None
            and out_w is not None
            and in_w.data_ptr() == out_w.data_ptr()
        )

        values_equal = False
        if in_w is not None and out_w is not None and in_w.shape == out_w.shape:
            try:
                values_equal = bool(torch.allclose(in_w, out_w))
            except Exception:
                values_equal = False

        tie_cfg = get_config_value(
            model.config, "decoder_config", "tie_word_embeddings"
        ) or get_config_value(model.config, "tie_word_embeddings")

        logger.info(
            f"LM head diagnostics -> shared_params={shared_ptr}, values_equal={values_equal}, tie_word_embeddings={tie_cfg}"
        )
        if out_w is not None:
            logger.info(
                f"LM head requires_grad before freeze: {bool(out_w.requires_grad)}"
            )
    except Exception as e:
        logger.warning(f"LM head tie diagnostics failed: {e}")


def validate_special_tokens(model, processor):
    """Validate special token IDs and log diagnostics."""
    try:
        tok = processor.tokenizer
        special_names = ["speech_start_id", "speech_diffusion_id", "speech_end_id"]
        try:
            vocab_size = int(getattr(model.config.decoder_config, "vocab_size", 0))
        except Exception:
            vocab_size = 0
        in_emb_mod = model.get_input_embeddings()
        out_emb_mod = model.get_output_embeddings()
        in_w = getattr(in_emb_mod, "weight", None)
        out_w = getattr(out_emb_mod, "weight", None)
        for name in special_names:
            val = getattr(tok, name, None)
            exists = val is not None
            in_range = exists and isinstance(val, int) and 0 <= val < vocab_size
            equal_row = None
            if (
                in_range
                and in_w is not None
                and out_w is not None
                and in_w.shape == out_w.shape
                and in_w.size(0) > val
            ):
                try:
                    equal_row = bool(torch.allclose(in_w[val], out_w[val]))
                except Exception:
                    equal_row = False
            decoded_str = None
            if exists and isinstance(val, int):
                try:
                    decoded_str = tok.decode([val])
                except Exception:
                    try:
                        decoded_str = tok.convert_ids_to_tokens(val)
                    except Exception:
                        decoded_str = "<decode_failed>"
            logger.info(
                f"Special token check -> {name}={val}, decoded='{decoded_str}', exists={exists}, in_vocab_range={in_range}, emb_vs_head_row_equal={equal_row}"
            )
    except Exception as e:
        logger.warning(f"Special token ID/row validation failed: {e}")


def log_trainable_parameters(model):
    """Log diagnostics about trainable parameters by component."""
    components = ModelComponents(model)

    try:
        lm_lora = components.count_parameters("language_model", trainable_only=True)
        pred_head_train = components.count_parameters(
            "prediction_head", trainable_only=True
        )
        ac_conn_train = components.count_parameters(
            "acoustic_connector", trainable_only=True
        )
        se_conn_train = components.count_parameters(
            "semantic_connector", trainable_only=True
        )
        total_trainable = components.count_parameters(trainable_only=True)

        logger.info(
            f"Trainable by block -> LLM-LoRA: {lm_lora:,} | diff_head: {pred_head_train:,} | ac_conn: {ac_conn_train:,} | se_conn: {se_conn_train:,}"
        )
        logger.info("TOTAL trainable: %s", f"{total_trainable:,}")
    except Exception:
        pass
