"""
Training script for VibeVoice finetuning.

Adapted from https://github.com/voicepowered-ai/VibeVoice-finetuning/blob/main/src/finetune_vibevoice_lora.py (MIT License)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import VerificationMode, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import HfArgumentParser, Trainer, TrainerCallback
from transformers import TrainingArguments as HfTrainingArguments
from transformers import set_seed
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

from data_vibevoice import VibeVoiceCollator, VibeVoiceDataset

logger = logging.getLogger(__name__)


from vibevoice.finetune.utils.ema_callback import EmaCallback
from vibevoice.finetune.utils.arguments import ModelArguments, DataArguments, VibeVoiceTrainingArguments


def build_lora_config(args: ModelArguments) -> LoraConfig:
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


def build_head_lora_config(args: ModelArguments) -> LoraConfig:
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


def mask_for_ce(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    acoustic_input_mask: torch.Tensor,
    pad_id: int = -100,
) -> torch.Tensor:
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


def _patch_acoustic_encode_for_legacy_indexing(model_obj, logger_):
    try:
        acoustic = getattr(
            getattr(model_obj, "model", model_obj), "acoustic_tokenizer", None
        )
        if acoustic is None or not hasattr(acoustic, "encode"):
            logger_.warning("No acoustic_tokenizer.encode() found to patch.")
            return
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


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, VibeVoiceTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    # Configure gradient clipping
    if not getattr(training_args, "gradient_clipping", False):
        if hasattr(training_args, "max_grad_norm"):
            training_args.max_grad_norm = 0.0
            logger.info(
                "Gradient clipping disabled (set max_grad_norm=0.0). Use --gradient_clipping to enable."
            )
    else:
        if (
            (not hasattr(training_args, "max_grad_norm"))
            or training_args.max_grad_norm is None
            or training_args.max_grad_norm <= 0
        ):
            training_args.max_grad_norm = 1.0
        logger.info(
            f"Gradient clipping enabled: max_grad_norm={training_args.max_grad_norm}"
        )

    # Load processor
    processor_path = model_args.processor_name_or_path or model_args.model_name_or_path
    if processor_path is None:
        raise ValueError(
            "--model_name_or_path (or --processor_name_or_path) must be provided"
        )
    processor: VibeVoiceProcessor = VibeVoiceProcessor.from_pretrained(processor_path)

    # Required special tokens
    tok = processor.tokenizer
    for required in ["speech_start_id", "speech_diffusion_id", "speech_end_id"]:
        if not hasattr(tok, required) or getattr(tok, required) is None:
            raise RuntimeError(f"Tokenizer missing required special id: {required}")

    # Load model
    if model_args.model_name_or_path is None:
        raise ValueError(
            "--model_name_or_path is required to load VibeVoice base model"
        )
    dtype = torch.float32
    if training_args.bf16:
        dtype = torch.bfloat16
    elif getattr(training_args, "fp16", False):
        dtype = torch.float16
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=dtype,
    )
    _patch_acoustic_encode_for_legacy_indexing(model, logger)
    processor.semantic_tokenizer = getattr(model.model, "semantic_tokenizer", None)

    # Diagnostics: LM head tie
    try:
        in_emb_mod = model.get_input_embeddings()
        out_emb_mod = model.get_output_embeddings()
        in_w = getattr(in_emb_mod, "weight", None)
        out_w = getattr(out_emb_mod, "weight", None)
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
        try:
            tie_cfg = getattr(
                getattr(model.config, "decoder_config", model.config),
                "tie_word_embeddings",
                None,
            )
        except Exception:
            tie_cfg = getattr(model.config, "tie_word_embeddings", None)
        logger.info(
            f"LM head diagnostics -> shared_params={shared_ptr}, values_equal={values_equal}, tie_word_embeddings={tie_cfg}"
        )
        if out_w is not None:
            logger.info(
                f"LM head requires_grad before freeze: {bool(out_w.requires_grad)}"
            )
    except Exception as e:
        logger.warning(f"LM head tie diagnostics failed: {e}")

    # Hard-tie LM head
    try:
        emb_module = model.get_input_embeddings()
        head_module = model.get_output_embeddings()
        if hasattr(emb_module, "weight") and hasattr(head_module, "weight"):
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

    # Validate special IDs (info logs only)
    try:
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

    # Disable cache during training
    if hasattr(model.config, "use_cache") and training_args.do_train:
        model.config.use_cache = False

    # Freeze tokenizers
    if model_args.freeze_acoustic_tokenizer and hasattr(
        model.model, "acoustic_tokenizer"
    ):
        for p in model.model.acoustic_tokenizer.parameters():
            p.requires_grad = False
    if model_args.freeze_semantic_tokenizer and hasattr(
        model.model, "semantic_tokenizer"
    ):
        for p in model.model.semantic_tokenizer.parameters():
            p.requires_grad = False

    # LoRA wrap LLM (optional)
    lora_cfg = build_lora_config(model_args)
    tm_lower = [
        s.strip().lower()
        for s in model_args.lora_target_modules.split(",")
        if s.strip()
    ]
    skip_lm_lora = (len(tm_lower) == 0) or all(
        t in ("none", "off", "disable", "disabled") for t in tm_lower
    )
    if not skip_lm_lora:
        model.model.language_model = get_peft_model(
            model.model.language_model, lora_cfg
        )
    else:
        logger.info("Skipping LLM LoRA wrapping (lora_target_modules indicates none).")

    try:
        model.tie_weights()
    except Exception:
        pass

    # Freeze all then enable trainable subsets
    for _, p in model.named_parameters():
        p.requires_grad = False

    try:
        for n, p in model.model.language_model.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                p.requires_grad = True
    except Exception:
        logger.warning("Could not re-enable LoRA params on language_model.")

    # Diffusion head LoRA wrapping (optional)
    if getattr(model_args, "lora_wrap_diffusion_head", False) and hasattr(
        model.model, "prediction_head"
    ):

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
            shim = _HeadForwardShim(model.model.prediction_head)
            model.model.prediction_head = get_peft_model(
                shim, build_head_lora_config(model_args)
            )
            for n, p in model.model.prediction_head.named_parameters():
                if "lora_A" in n or "lora_B" in n:
                    p.requires_grad = True
        except Exception as e:
            logger.warning(f"Could not LoRA-wrap diffusion head: {e}")

    # Train full diffusion head (optional)
    if getattr(model_args, "train_diffusion_head", False) and hasattr(
        model.model, "prediction_head"
    ):
        for p in model.model.prediction_head.parameters():
            p.requires_grad = True

    # Freeze diffusion head layers (optional)
    if model_args.layers_to_freeze is not None and hasattr(
        model.model, "prediction_head"
    ):
        head_params = list(model.model.prediction_head.named_parameters())
        try:
            indices_to_freeze = {
                int(x.strip())
                for x in model_args.layers_to_freeze.split(",")
                if x.strip()
            }
            frozen_count = 0
            for i, (name, param) in enumerate(head_params):
                if i in indices_to_freeze:
                    param.requires_grad = False
                    frozen_count += 1
                    logger.info(f"Froze layer [{i}]: {name}")
            logger.info(
                f"Successfully froze {frozen_count} parameter groups in the diffusion head."
            )
        except Exception as e:
            logger.error(f"Could not parse --layers_to_freeze: {e}")
            raise

    # Connectors
    if getattr(model_args, "train_connectors", False):
        if hasattr(model.model, "acoustic_connector"):
            for p in model.model.acoustic_connector.parameters():
                p.requires_grad = True
        if hasattr(model.model, "semantic_connector"):
            for p in model.model.semantic_connector.parameters():
                p.requires_grad = True
    else:
        if hasattr(model.model, "acoustic_connector"):
            for p in model.model.acoustic_connector.parameters():
                p.requires_grad = False
        if hasattr(model.model, "semantic_connector"):
            for p in model.model.semantic_connector.parameters():
                p.requires_grad = False

    # Freeze embedding + head
    try:
        emb = model.get_input_embeddings()
        if hasattr(emb, "weight"):
            emb.weight.requires_grad_(False)
        head = model.get_output_embeddings()
        if head is not None and hasattr(head, "weight"):
            head.weight.requires_grad_(False)
    except Exception:
        pass

    # Diagnostics
    def _sum_params(named_iter):
        return sum(p.numel() for _, p in named_iter if p.requires_grad)

    try:
        lm_lora = (
            _sum_params(model.model.language_model.named_parameters())
            if hasattr(model.model, "language_model")
            else 0
        )
        pred_head_train = (
            _sum_params(model.model.prediction_head.named_parameters())
            if hasattr(model.model, "prediction_head")
            else 0
        )
        ac_conn_train = (
            _sum_params(model.model.acoustic_connector.named_parameters())
            if hasattr(model.model, "acoustic_connector")
            else 0
        )
        se_conn_train = (
            _sum_params(model.model.semantic_connector.named_parameters())
            if hasattr(model.model, "semantic_connector")
            else 0
        )
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Trainable by block -> LLM-LoRA: {lm_lora:,} | diff_head: {pred_head_train:,} | ac_conn: {ac_conn_train:,} | se_conn: {se_conn_train:,}"
        )
        logger.info("TOTAL trainable: %s", f"{total_trainable:,}")
    except Exception:
        pass

    # Datasets
    verification_mode = (
        VerificationMode.NO_CHECKS
        if data_args.ignore_verifications
        else VerificationMode.BASIC_CHECKS
    )
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

    # Ratios/dims from processor+model
    speech_compress_ratio = getattr(processor, "speech_tok_compress_ratio", 3200)
    semantic_dim = getattr(model.config, "semantic_vae_dim", None)
    if semantic_dim is None:
        try:
            semantic_dim = int(
                getattr(model.config.semantic_tokenizer_config, "vae_dim", 128)
            )
        except Exception:
            semantic_dim = 128

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

    class LoRACallback(TrainerCallback):
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
                req_grad = sum(
                    1 for n in self.lora_param_names if named[n].requires_grad
                )
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

    class VibeVoiceTrainer(Trainer):
        def compute_loss(
            self,
            model: VibeVoiceForConditionalGeneration,
            inputs: Dict[str, Any],
            return_outputs=False,
            num_items_in_batch: Optional[int] = None,
        ):
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
                ddpm_batch_mul=training_args.ddpm_batch_mul,
            )

            # Invariants: token/latent selection equality across views (warn, don't assert)
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
                num_lat_total = (
                    int(sp_masks.sum().item()) if sp_masks is not None else 0
                )
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
                if (
                    sp_loss_sel is not None
                    and sp_masks is not None
                    and al_mask is not None
                ):
                    if num_tok_loss != num_lat_loss:
                        logger.warning(
                            f"Loss selection mismatch: acoustic_loss_mask={num_tok_loss} vs speeches_loss_input={num_lat_loss}"
                        )
            except Exception:
                pass

            # CE Loss
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
            try:
                self._debug_ce(
                    shift_logits, ce_labels, attention_mask, acoustic_input_mask
                )
            except Exception as e:
                logger.warning(f"Failed invoking CE debug: {e}")

            # Diffusion loss
            diffusion_loss = (
                outputs.diffusion_loss
                if outputs.diffusion_loss is not None
                else torch.tensor(0.0, device=ce_loss.device)
            )
            total = (
                training_args.ce_loss_weight * ce_loss
                + training_args.diffusion_loss_weight * diffusion_loss
            )

            # Logs
            try:
                prefix = "train" if model.training else "eval"
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
                    hasattr(self, "optimizer")
                    and self.optimizer is not None
                    and len(self.optimizer.param_groups) > 0
                ):
                    lr_val = self.optimizer.param_groups[0].get("lr", None)
                    if lr_val is not None:
                        self.log({"train/learning_rate_real": float(lr_val)})
            except Exception:
                pass

            return (total, outputs) if return_outputs else total

        def _debug_ce(
            self,
            shift_logits: torch.Tensor,
            ce_labels: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            acoustic_input_mask: Optional[torch.Tensor],
        ):
            try:
                if not getattr(training_args, "debug_ce_details", False):
                    return
                step = int(getattr(self.state, "global_step", 0) or 0)
                every_n = max(
                    1, int(getattr(training_args, "debug_ce_every_n_steps", 200) or 200)
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
                        1, int(getattr(training_args, "debug_ce_max_examples", 1) or 1)
                    )
                    B = ce_labels.size(0)
                    for b in range(min(B, max_examples)):
                        vb = valid_mask[b]
                        if int(vb.sum().item()) > 0:
                            per_ex_avgs.append(
                                float(per_token_loss[b][vb].mean().item())
                            )
                        else:
                            per_ex_avgs.append(float("nan"))
                    logger.info(
                        f"CE debug: tokens_in_loss={num_valid}, avg_loss={avg_loss:.4f}, per_example_avgs={[round(x,4) if x==x else None for x in per_ex_avgs]}"
                    )
            except Exception as e:
                logger.warning(f"CE detailed debug failed: {e}")

        # --------- CRITICAL SAVE OVERRIDES: also dump FULL head/connectors for inference ---------

        def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
            try:
                target_dir = output_dir or self.args.output_dir
                lora_out = os.path.join(target_dir, "lora")
                os.makedirs(lora_out, exist_ok=True)

                # --- LLM PEFT adapters (if LoRA-wrapped) ---
                language_model = getattr(self.model.model, "language_model", None)
                if hasattr(language_model, "save_pretrained"):
                    language_model.save_pretrained(lora_out)

                # --- Diffusion head PEFT adapters (if LoRA-wrapped) ---
                pred_head = getattr(self.model.model, "prediction_head", None)
                if hasattr(pred_head, "save_pretrained"):
                    ph_dir = os.path.join(lora_out, "diffusion_head")
                    os.makedirs(ph_dir, exist_ok=True)
                    pred_head.save_pretrained(ph_dir)

                # --- ALWAYS save FULL diffusion head state_dict for fallback ---
                if pred_head is not None and hasattr(pred_head, "state_dict"):
                    sd = pred_head.state_dict()
                    torch.save(sd, os.path.join(lora_out, "diffusion_head_full.bin"))
                    ph_dir = os.path.join(lora_out, "diffusion_head")
                    os.makedirs(ph_dir, exist_ok=True)
                    torch.save(sd, os.path.join(ph_dir, "diffusion_head_full.bin"))

                # --- Connectors (plain state_dicts) ---
                ac = getattr(self.model.model, "acoustic_connector", None)
                if ac is not None:
                    ac_dir = os.path.join(lora_out, "acoustic_connector")
                    os.makedirs(ac_dir, exist_ok=True)
                    torch.save(
                        ac.state_dict(), os.path.join(ac_dir, "pytorch_model.bin")
                    )

                se = getattr(self.model.model, "semantic_connector", None)
                if se is not None:
                    se_dir = os.path.join(lora_out, "semantic_connector")
                    os.makedirs(se_dir, exist_ok=True)
                    torch.save(
                        se.state_dict(), os.path.join(se_dir, "pytorch_model.bin")
                    )

            except Exception as e:
                logger.warning(f"Failed to save LoRA assets: {e}")

    # ------------- Build the Trainer -------------

    # Resolve which adapters to apply in samples

    ema_cb = EmaCallback(attr_path="model.prediction_head", decay=0.999, device="cpu")

    trainer = VibeVoiceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            ema_cb,
            LoRACallback(
                log_every_n_steps=(
                    int(getattr(training_args, "logging_steps", 50) or 50)
                )
            ),
        ],
    )

    if getattr(training_args, "gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            logger.warning("Failed to enable gradient checkpointing on the model.")

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        lora_out = os.path.join(training_args.output_dir, "lora")
        os.makedirs(lora_out, exist_ok=True)

        # LLM PEFT (if any)
        lm = getattr(model.model, "language_model", None)
        if hasattr(lm, "save_pretrained"):
            lm.save_pretrained(lora_out)

        # Diffusion head PEFT (if any)
        ph = getattr(model.model, "prediction_head", None)
        if hasattr(ph, "save_pretrained"):
            ph_dir = os.path.join(lora_out, "diffusion_head")
            os.makedirs(ph_dir, exist_ok=True)
            ph.save_pretrained(ph_dir)

        # ALWAYS: full diffusion head state_dict fallback
        try:
            if ph is not None and hasattr(ph, "state_dict"):
                sd = ph.state_dict()
                torch.save(sd, os.path.join(lora_out, "diffusion_head_full.bin"))
                ph_dir = os.path.join(lora_out, "diffusion_head")
                os.makedirs(ph_dir, exist_ok=True)
                torch.save(sd, os.path.join(ph_dir, "diffusion_head_full.bin"))
        except Exception as e:
            logger.warning(f"Failed to save FULL diffusion head at end: {e}")

        # Connectors (if trained)
        try:
            ac = getattr(model.model, "acoustic_connector", None)
            if ac is not None:
                ac_dir = os.path.join(lora_out, "acoustic_connector")
                os.makedirs(ac_dir, exist_ok=True)
                torch.save(ac.state_dict(), os.path.join(ac_dir, "pytorch_model.bin"))
        except Exception as e:
            logger.warning(f"Failed to save acoustic_connector: {e}")

        try:
            se = getattr(model.model, "semantic_connector", None)
            if se is not None:
                se_dir = os.path.join(lora_out, "semantic_connector")
                os.makedirs(se_dir, exist_ok=True)
                torch.save(se.state_dict(), os.path.join(se_dir, "pytorch_model.bin"))
        except Exception as e:
            logger.warning(f"Failed to save semantic_connector: {e}")

    if training_args.do_eval and eval_dataset is not None:
        trainer.evaluate()


if __name__ == "__main__":
    main()
