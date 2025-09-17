"""
LoRA configuration and setup utilities for VibeVoice finetuning.
"""

import logging
from typing import List, Optional

import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)


def build_lora_config(args) -> LoraConfig:
    """Build LoRA configuration for language model."""
    target_modules = [
        s.strip() for s in args.lora_target_modules.split(",") if s.strip()
    ]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def build_head_lora_config(args) -> LoraConfig:
    """Build LoRA configuration for diffusion head."""
    target_modules = [
        "noisy_images_proj",
        "cond_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "linear",
    ]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )


def should_skip_lm_lora(model_args) -> bool:
    """Determine if LM LoRA should be skipped based on target modules."""
    tm_lower = [
        s.strip().lower()
        for s in model_args.lora_target_modules.split(",")
        if s.strip()
    ]
    return (len(tm_lower) == 0) or all(
        t in ("none", "off", "disable", "disabled") for t in tm_lower
    )


def apply_lm_lora(model, model_args):
    """Apply LoRA to language model if configured."""
    if should_skip_lm_lora(model_args):
        logger.info("Skipping LLM LoRA wrapping (lora_target_modules indicates none).")
        return

    lora_cfg = build_lora_config(model_args)
    model.model.language_model = get_peft_model(model.model.language_model, lora_cfg)


def apply_diffusion_head_lora(model, model_args):
    """Apply LoRA to diffusion head if configured."""
    if not getattr(model_args, "lora_wrap_diffusion_head", False):
        return

    prediction_head = getattr(model.model, "prediction_head", None)
    if prediction_head is None:
        logger.warning("Cannot apply LoRA to diffusion head: prediction_head not found")
        return

    class _HeadForwardShim(nn.Module):
        def __init__(self, base: nn.Module):
            super().__init__()
            self.base = base

        def forward(self, *args, **kwargs):
            if len(args) >= 3:
                noisy_images, timesteps, condition = args[:3]
            else:
                noisy_images = kwargs.get("noisy_images")
                timesteps = kwargs.get("timesteps")
                condition = kwargs.get("condition")
            return self.base(noisy_images, timesteps, condition)

    try:
        shim = _HeadForwardShim(prediction_head)
        model.model.prediction_head = get_peft_model(
            shim, build_head_lora_config(model_args)
        )
        enable_lora_parameters(model.model.prediction_head)
    except Exception as e:
        logger.warning(f"Could not LoRA-wrap diffusion head: {e}")


def enable_lora_parameters(module):
    """Enable gradients for LoRA parameters in a module."""
    for n, p in module.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True


def enable_lm_lora_parameters(model):
    """Enable gradients for LoRA parameters in language model."""
    try:
        enable_lora_parameters(model.model.language_model)
    except Exception:
        logger.warning("Could not re-enable LoRA params on language_model.")


def freeze_all_parameters(model):
    """Freeze all model parameters."""
    for _, p in model.named_parameters():
        p.requires_grad = False
