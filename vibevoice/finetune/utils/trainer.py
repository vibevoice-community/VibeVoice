"""
Custom trainer class for VibeVoice finetuning.
"""

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from vibevoice.modular.modeling_vibevoice import \
    VibeVoiceForConditionalGeneration

logger = logging.getLogger(__name__)


def mask_for_ce(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    acoustic_input_mask: torch.Tensor,
    pad_id: int = -100,
) -> torch.Tensor:
    """Create cross-entropy loss mask."""
    shifted = labels[:, 1:].contiguous()
    base_mask = (
        attention_mask[:, 1:].contiguous().eq(1)
        if (attention_mask is not None and attention_mask.numel() > 0)
        else torch.ones_like(shifted, dtype=torch.bool)
    )
    label_is_acoustic = acoustic_input_mask[:, 1:].contiguous()
    final_mask = base_mask & (~label_is_acoustic)
    out = shifted.clone()
    out[~final_mask] = pad_id
    return out


class VibeVoiceTrainer(Trainer):
    """Custom trainer for VibeVoice with specialized loss computation."""

    def compute_loss(
        self,
        model: VibeVoiceForConditionalGeneration,
        inputs: Dict[str, Any],
        return_outputs=False,
        num_items_in_batch: Optional[int] = None,
    ):
        """Compute custom loss for VibeVoice training."""
        labels = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        acoustic_input_mask = inputs.get("acoustic_input_mask")

        # Ensure semantic tensors exist and have correct dtype/device
        sem = inputs.get("speech_semantic_tensors", None)
        try:
            target_dtype = next(model.model.semantic_connector.parameters()).dtype
        except Exception:
            target_dtype = model.get_input_embeddings().weight.dtype

        if sem is None:
            sm = inputs.get("speech_masks")
            if sm is not None:
                zeros = torch.zeros(
                    sm.size(0),
                    sm.size(1),
                    getattr(model.config, "semantic_vae_dim", 128),
                    dtype=target_dtype,
                    device=sm.device,
                )
                inputs["speech_semantic_tensors"] = zeros
        else:
            if isinstance(sem, torch.Tensor):
                inputs["speech_semantic_tensors"] = sem.to(dtype=target_dtype)

        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=attention_mask,
            speech_tensors=inputs.get("speech_tensors"),
            speech_masks=inputs.get("speech_masks"),
            speech_semantic_tensors=inputs.get("speech_semantic_tensors"),
            acoustic_input_mask=acoustic_input_mask,
            acoustic_loss_mask=inputs.get("acoustic_loss_mask"),
            speeches_loss_input=inputs.get("speeches_loss_input"),
            ddpm_batch_mul=self.args.ddpm_batch_mul,
        )

        # Log debugging information
        self._log_training_diagnostics(inputs, acoustic_input_mask)

        # Cross-entropy loss
        logits = outputs.logits
        ce_labels = mask_for_ce(
            labels, attention_mask, acoustic_input_mask, pad_id=-100
        )
        shift_logits = logits[:, :-1, :].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        ce_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), ce_labels.view(-1)
        )

        # Optional CE diagnostics
        self._debug_ce_if_enabled(
            shift_logits, ce_labels, attention_mask, acoustic_input_mask
        )

        # Diffusion loss
        diffusion_loss = (
            outputs.diffusion_loss
            if outputs.diffusion_loss is not None
            else torch.tensor(0.0, device=ce_loss.device)
        )

        total = (
            self.args.ce_loss_weight * ce_loss
            + self.args.diffusion_loss_weight * diffusion_loss
        )

        # Log losses and learning rate
        self._log_losses_and_lr(ce_loss, diffusion_loss, model.training)

        return (total, outputs) if return_outputs else total

    def _log_training_diagnostics(self, inputs, acoustic_input_mask):
        """Log training diagnostics about token/latent counts."""
        try:
            al_mask = inputs.get("acoustic_loss_mask")
            sp_masks = inputs.get("speech_masks")
            sp_loss_sel = inputs.get("speeches_loss_input")

            num_tok_total = (
                int(acoustic_input_mask.sum().item())
                if acoustic_input_mask is not None
                else 0
            )
            num_tok_loss = int(al_mask.sum().item()) if al_mask is not None else 0
            num_lat_total = int(sp_masks.sum().item()) if sp_masks is not None else 0
            num_lat_loss = (
                int(((sp_loss_sel & sp_masks).sum().item()))
                if (sp_loss_sel is not None and sp_masks is not None)
                else 0
            )

            self.log(
                {
                    "debug/num_tok_total": float(num_tok_total),
                    "debug/num_tok_loss": float(num_tok_loss),
                    "debug/num_lat_total": float(num_lat_total),
                    "debug/num_lat_loss": float(num_lat_loss),
                }
            )

            if sp_loss_sel is not None and sp_masks is not None and al_mask is not None:
                if num_tok_loss != num_lat_loss:
                    logger.warning(
                        f"Loss selection mismatch: acoustic_loss_mask={num_tok_loss} vs speeches_loss_input={num_lat_loss}"
                    )
        except Exception:
            pass

    def _debug_ce_if_enabled(
        self, shift_logits, ce_labels, attention_mask, acoustic_input_mask
    ):
        """Run CE debugging if enabled in training args."""
        try:
            if not getattr(self.args, "debug_ce_details", False):
                return

            step = int(getattr(self.state, "global_step", 0) or 0)
            every_n = max(
                1, int(getattr(self.args, "debug_ce_every_n_steps", 200) or 200)
            )

            if not (step <= 1 or (step % every_n == 0)):
                return

            with torch.no_grad():
                vocab = shift_logits.size(-1)
                per_token_loss = F.cross_entropy(
                    shift_logits.view(-1, vocab),
                    ce_labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view_as(ce_labels)

                valid_mask = ce_labels.ne(-100)
                num_valid = int(valid_mask.sum().item())
                avg_loss = (
                    float((per_token_loss[valid_mask].mean().item()))
                    if num_valid > 0
                    else float("nan")
                )

                per_ex_avgs = []
                max_examples = max(
                    1, int(getattr(self.args, "debug_ce_max_examples", 1) or 1)
                )
                B = ce_labels.size(0)

                for b in range(min(B, max_examples)):
                    vb = valid_mask[b]
                    if int(vb.sum().item()) > 0:
                        per_ex_avgs.append(float(per_token_loss[b][vb].mean().item()))
                    else:
                        per_ex_avgs.append(float("nan"))

                logger.info(
                    f"CE debug: tokens_in_loss={num_valid}, avg_loss={avg_loss:.4f}, per_example_avgs={[round(x,4) if x==x else None for x in per_ex_avgs]}"
                )
        except Exception as e:
            logger.warning(f"CE detailed debug failed: {e}")

    def _log_losses_and_lr(self, ce_loss, diffusion_loss, is_training):
        """Log losses and learning rate."""
        try:
            prefix = "train" if is_training else "eval"
            self.log(
                {
                    f"{prefix}/ce_loss": ce_loss.detach().item(),
                    f"{prefix}/diffusion_loss": (
                        diffusion_loss.detach().item()
                        if isinstance(diffusion_loss, torch.Tensor)
                        else float(diffusion_loss)
                    ),
                }
            )

            if (
                is_training
                and hasattr(self, "optimizer")
                and self.optimizer is not None
                and len(self.optimizer.param_groups) > 0
            ):
                lr_val = self.optimizer.param_groups[0].get("lr", None)
                if lr_val is not None:
                    self.log({"train/learning_rate_real": float(lr_val)})
        except Exception:
            pass

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        """Custom save method that also saves LoRA adapters and full components."""
        try:
            target_dir = output_dir or self.args.output_dir
            lora_out = os.path.join(target_dir, "lora")
            os.makedirs(lora_out, exist_ok=True)

            # Save LLM PEFT adapters (if LoRA-wrapped)
            language_model = getattr(self.model.model, "language_model", None)
            if hasattr(language_model, "save_pretrained"):
                language_model.save_pretrained(lora_out)

            # Save diffusion head PEFT adapters (if LoRA-wrapped)
            pred_head = getattr(self.model.model, "prediction_head", None)
            if hasattr(pred_head, "save_pretrained"):
                ph_dir = os.path.join(lora_out, "diffusion_head")
                os.makedirs(ph_dir, exist_ok=True)
                pred_head.save_pretrained(ph_dir)

            # ALWAYS save FULL diffusion head state_dict for fallback
            if pred_head is not None and hasattr(pred_head, "state_dict"):
                sd = pred_head.state_dict()
                torch.save(sd, os.path.join(lora_out, "diffusion_head_full.bin"))
                ph_dir = os.path.join(lora_out, "diffusion_head")
                os.makedirs(ph_dir, exist_ok=True)
                torch.save(sd, os.path.join(ph_dir, "diffusion_head_full.bin"))

            # Save connectors (plain state_dicts)
            self._save_connector("acoustic_connector", lora_out)
            self._save_connector("semantic_connector", lora_out)

        except Exception as e:
            logger.warning(f"Failed to save LoRA assets: {e}")

    def _save_connector(self, connector_name: str, lora_out: str):
        """Save a connector component."""
        connector = getattr(self.model.model, connector_name, None)
        if connector is not None:
            conn_dir = os.path.join(lora_out, connector_name)
            os.makedirs(conn_dir, exist_ok=True)
            torch.save(
                connector.state_dict(), os.path.join(conn_dir, "pytorch_model.bin")
            )
