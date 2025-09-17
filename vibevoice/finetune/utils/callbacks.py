"""
Training callbacks for VibeVoice finetuning.
"""

import copy
import logging
from typing import Dict, List

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class EmaCallback(TrainerCallback):
    """
    EMA callback for VibeVoice finetuning.
    """

    def __init__(self, attr_path="model.prediction_head", decay=0.999, device="cpu"):
        """
        attr_path: where the head lives under self.model (Trainer wraps your VibeVoiceForConditionalGeneration)
        decay:     EMA decay (0.999 ~ stable, 0.9999 ~ very smooth, slower to adapt)
        """
        self.attr_path = attr_path
        self.decay = float(decay)
        self.device = torch.device(device)
        self.shadow = None
        self._orig = None  # store non-EMA weights when we swap

    def _get_module(self, model):
        # Resolve dotted path like "model.prediction_head"
        mod = model
        for name in self.attr_path.split("."):
            mod = getattr(mod, name)
        return mod

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        head = self._get_module(model)
        self.shadow = {
            k: p.detach().to(self.device).clone() for k, p in head.state_dict().items()
        }

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.shadow is None:
            return
        head = self._get_module(model)
        with torch.no_grad():
            for k, v in head.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(
                    v.detach().to(self.device), alpha=(1.0 - self.decay)
                )

    # ---- Swap helpers ----
    def _swap_in_ema(self, model):
        head = self._get_module(model)
        self._orig = copy.deepcopy(head.state_dict())
        head.load_state_dict(self.shadow, strict=False)

    def _swap_back(self, model):
        if self._orig is None:
            return
        head = self._get_module(model)
        head.load_state_dict(self._orig, strict=False)
        self._orig = None

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # use EMA during eval
        self._swap_in_ema(model)

    def on_evaluate_end(self, args, state, control, model=None, **kwargs):
        self._swap_back(model)

    def on_save(self, args, state, control, model=None, **kwargs):
        # temporarily swap to EMA, let Trainer save, then swap back
        self._swap_in_ema(model)

    def on_save_end(self, args, state, control, model=None, **kwargs):
        self._swap_back(model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # final checkpoint: persist EMA
        self._swap_in_ema(model)


class LoRACallback(TrainerCallback):
    """Debug callback to monitor LoRA parameter changes during training."""

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.prev_param_norms: Dict[str, float] = {}
        self.lora_param_names: List[str] = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        try:
            if model is None:
                return

            named: Dict[str, torch.nn.Parameter] = dict(model.named_parameters())
            self.lora_param_names = [
                n for n in named.keys() if ("lora_A" in n or "lora_B" in n)
            ]

            for n in self.lora_param_names:
                p = named[n]
                self.prev_param_norms[n] = float(p.data.norm().item())

            total = len(self.lora_param_names)
            req_grad = sum(1 for n in self.lora_param_names if named[n].requires_grad)
            num_A = sum(1 for n in self.lora_param_names if "lora_A" in n)
            num_B = sum(1 for n in self.lora_param_names if "lora_B" in n)
            zero_B = sum(
                1
                for n in self.lora_param_names
                if ("lora_B" in n and float(named[n].data.norm().item()) == 0.0)
            )

            logger.info(
                f"LoRA debug: found {total} LoRA params (A={num_A}, B={num_B}); trainable={req_grad}. Initial lora_B_zero={zero_B}."
            )

            if total == 0:
                logger.warning(
                    "LoRA debug: No LoRA parameters found. Check lora_target_modules."
                )
            if req_grad != total:
                logger.warning(
                    "LoRA debug: Some LoRA params are frozen. They should be trainable."
                )
        except Exception as e:
            logger.warning(f"LoRA debug (on_train_begin) failed: {e}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        try:
            if model is None or len(self.lora_param_names) == 0:
                return

            step = int(getattr(state, "global_step", 0) or 0)
            if step % self.log_every_n_steps != 0 and step != 1:
                return

            named: Dict[str, torch.nn.Parameter] = dict(model.named_parameters())
            changed_A = 0
            changed_B = 0
            zero_B = 0
            eps = 1e-12

            for n in self.lora_param_names:
                p = named.get(n, None)
                if p is None:
                    continue

                prev = self.prev_param_norms.get(n, 0.0)
                curr = float(p.data.norm().item())

                if "lora_A" in n and abs(curr - prev) > eps:
                    changed_A += 1
                if "lora_B" in n:
                    if abs(curr - prev) > eps:
                        changed_B += 1
                    if curr == 0.0:
                        zero_B += 1

                self.prev_param_norms[n] = curr

            total_A = sum(1 for n in self.lora_param_names if "lora_A" in n)
            total_B = sum(1 for n in self.lora_param_names if "lora_B" in n)

            logger.info(
                f"LoRA debug step {step}: changed A {changed_A}/{total_A}, changed B {changed_B}/{total_B}, lora_B_zero_now={zero_B}."
            )
        except Exception as e:
            logger.warning(f"LoRA debug (on_step_end) failed: {e}")


def create_callbacks(training_args, model_args=None):
    """Create list of callbacks for training."""
    callbacks = []

    # EMA callback
    ema_cb = EmaCallback(attr_path="model.prediction_head", decay=0.999, device="cpu")
    callbacks.append(ema_cb)

    # LoRA debugging callback
    lora_cb = LoRACallback(
        log_every_n_steps=(int(getattr(training_args, "logging_steps", 50) or 50))
    )
    callbacks.append(lora_cb)

    return callbacks
