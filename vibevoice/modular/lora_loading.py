"""Utilities for loading fine-tuned LoRA adapters and connector weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class _LoadReport:
    """Simple structure capturing what assets were loaded."""

    language_model: bool = False
    diffusion_head_lora: bool = False
    diffusion_head_full: bool = False
    acoustic_connector: bool = False
    semantic_connector: bool = False
    adapter_root: Optional[Path] = None


class _DiffusionHeadForwardShim(nn.Module):
    """Wraps the diffusion head to expose the signature expected by PEFT."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, *args, **kwargs):  # type: ignore[override]
        if len(args) >= 3:
            noisy_images, timesteps, condition = args[:3]
        else:
            noisy_images = kwargs.get("noisy_images")
            timesteps = kwargs.get("timesteps")
            condition = kwargs.get("condition")
        return self.base(noisy_images, timesteps, condition)


def _resolve_adapter_root(checkpoint_path: Path) -> Path:
    """Return the directory that actually holds adapter assets."""

    if checkpoint_path.is_file():
        checkpoint_path = checkpoint_path.parent

    if (checkpoint_path / "lora").exists():
        return checkpoint_path / "lora"
    return checkpoint_path


def _load_connector(module: Optional[nn.Module], path: Path, device: torch.device) -> bool:
    if module is None or not path.exists():
        return False

    state_dict = torch.load(path, map_location=device)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Connector load missing keys: {missing}")
    if unexpected:
        logger.warning(f"Connector load unexpected keys: {unexpected}")
    module.to(device)
    return True


def _load_diffusion_head(
    model, adapter_root: Path, device: torch.device, report: _LoadReport
) -> None:
    diff_dir = adapter_root / "diffusion_head"
    adapter_config = diff_dir / "adapter_config.json"
    adapter_model = diff_dir / "adapter_model.bin"
    adapter_model_safetensors = diff_dir / "adapter_model.safetensors"

    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - dependency guaranteed via pyproject, safeguard anyway
        raise RuntimeError(
            "peft is required to load diffusion head adapters but is not installed"
        ) from exc

    if adapter_config.exists() and (adapter_model.exists() or adapter_model_safetensors.exists()):
        logger.info(f"Loading diffusion head LoRA from {diff_dir}")
        shim = _DiffusionHeadForwardShim(model.model.prediction_head)
        peft_head = PeftModel.from_pretrained(shim, diff_dir)
        peft_head.to(device)
        model.model.prediction_head = peft_head
        report.diffusion_head_lora = True
        return

    # Fallback to full state dict if provided
    full_path = diff_dir / "diffusion_head_full.bin"
    if not full_path.exists():
        full_path = adapter_root / "diffusion_head_full.bin"

    if full_path.exists():
        logger.info(f"Loading full diffusion head weights from {full_path}")
        state_dict = torch.load(full_path, map_location=device)
        missing, unexpected = model.model.prediction_head.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Diffusion head load missing keys: {missing}")
        if unexpected:
            logger.warning(f"Diffusion head load unexpected keys: {unexpected}")
        model.model.prediction_head.to(device)
        report.diffusion_head_full = True


def _load_language_model(
    model, adapter_root: Path, device: torch.device, report: _LoadReport
) -> None:
    config_file = adapter_root / "adapter_config.json"
    bin_file = adapter_root / "adapter_model.bin"
    safe_tensors_file = adapter_root / "adapter_model.safetensors"
    if not (config_file.exists() and (bin_file.exists() or safe_tensors_file.exists())):
        return

    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - safeguard
        raise RuntimeError(
            "peft is required to load language model adapters but is not installed"
        ) from exc

    logger.info(f"Loading language model LoRA from {adapter_root}")
    peft_lm = PeftModel.from_pretrained(model.model.language_model, adapter_root)
    peft_lm.to(device)
    model.model.language_model = peft_lm
    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
        except Exception as exc:
            logger.warning(f"Failed to retie weights after loading language LoRA: {exc}")
    report.language_model = True


def load_lora_assets(model, checkpoint_dir: str, device: Optional[torch.device] = None) -> _LoadReport:
    """Load LoRA adapters and connector weights onto an instantiated model.

    Args:
        model: The already instantiated `VibeVoiceForConditionalGenerationInference` model.
        checkpoint_dir: Directory produced during fine-tuning containing a `lora/` folder.
        device: Optional device to place loaded modules on. When ``None`` the function
            infers the device from the model parameters.

    Returns:
        `_LoadReport` summarizing which assets were successfully loaded.
    """

    adapter_root = _resolve_adapter_root(Path(checkpoint_dir))
    if not adapter_root.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_root}")

    inferred_device = device or next(model.parameters()).device
    report = _LoadReport(adapter_root=adapter_root)

    _load_language_model(model, adapter_root, inferred_device, report)
    _load_diffusion_head(model, adapter_root, inferred_device, report)

    ac_path = adapter_root / "acoustic_connector" / "pytorch_model.bin"
    if _load_connector(getattr(model.model, "acoustic_connector", None), ac_path, inferred_device):
        report.acoustic_connector = True

    se_path = adapter_root / "semantic_connector" / "pytorch_model.bin"
    if _load_connector(getattr(model.model, "semantic_connector", None), se_path, inferred_device):
        report.semantic_connector = True

    if not any(report.__dict__.values()):
        logger.warning(
            "No adapter assets were loaded. Ensure the checkpoint directory is correct and contains LoRA weights."
        )

    return report
