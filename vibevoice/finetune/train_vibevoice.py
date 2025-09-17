"""
Training script for VibeVoice finetuning.

Adapted from https://github.com/voicepowered-ai/VibeVoice-finetuning/blob/main/src/finetune_vibevoice_lora.py (MIT License)
"""

import logging
import os

import torch
from transformers import HfArgumentParser, set_seed

# Import modular components
from vibevoice.finetune.utils.arguments import (DataArguments, ModelArguments,
                             VibeVoiceTrainingArguments)
from vibevoice.finetune.utils.callbacks import create_callbacks
from vibevoice.finetune.utils.data_utils import setup_data_pipeline
from vibevoice.finetune.utils.lora_config import (apply_diffusion_head_lora, apply_lm_lora,
                               enable_lm_lora_parameters,
                               freeze_all_parameters)
from vibevoice.finetune.utils.model_introspection import (ModelComponents,
                                       freeze_diffusion_head_layers,
                                       setup_connector_training,
                                       setup_diffusion_head_training,
                                       setup_tokenizer_freezing)
from vibevoice.finetune.utils.model_setup import (freeze_embeddings_and_head, hard_tie_lm_head,
                               load_model, load_processor,
                               log_lm_head_diagnostics,
                               log_trainable_parameters,
                               patch_acoustic_encode_for_legacy_indexing,
                               setup_model_cache_and_tokenizers,
                               validate_special_tokens)
from vibevoice.finetune.utils.trainer import VibeVoiceTrainer

logger = logging.getLogger(__name__)


def setup_logging(training_args):
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)


def configure_gradient_clipping(training_args):
    """Configure gradient clipping based on training arguments."""
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


def setup_model_components(model, model_args, training_args):
    """Setup all model components including LoRA, freezing, etc."""
    components = ModelComponents(model)

    # Apply LoRA configurations
    apply_lm_lora(model, model_args)
    apply_diffusion_head_lora(model, model_args)

    # Tie weights if needed
    try:
        model.tie_weights()
    except Exception:
        pass

    # Freeze all parameters first
    freeze_all_parameters(model)

    # Re-enable LoRA parameters
    enable_lm_lora_parameters(model)

    # Setup component training based on args
    setup_connector_training(components, model_args)
    setup_diffusion_head_training(components, model_args)
    freeze_diffusion_head_layers(components, model_args)

    # Freeze embeddings and head
    freeze_embeddings_and_head(model)


def setup_model_for_training(model_args, training_args):
    """Load and configure the complete model for training."""
    # Load processor and model
    processor = load_processor(model_args)
    model = load_model(model_args, training_args)

    # Apply patches and setup
    patch_acoustic_encode_for_legacy_indexing(model, logger)
    processor.semantic_tokenizer = getattr(model.model, "semantic_tokenizer", None)

    # Log diagnostics
    log_lm_head_diagnostics(model)

    # Hard-tie LM head
    hard_tie_lm_head(model)

    # Validate special tokens
    validate_special_tokens(model, processor)

    # Setup caching and tokenizers
    setup_model_cache_and_tokenizers(model, model_args, training_args)

    # Setup all model components
    setup_model_components(model, model_args, training_args)

    # Log final parameter counts
    log_trainable_parameters(model)

    return model, processor


def save_final_artifacts(model, training_args):
    """Save final training artifacts."""
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

    # Save connectors if they were trained
    components = ModelComponents(model)
    for connector_name in ["acoustic_connector", "semantic_connector"]:
        connector = getattr(components, connector_name)
        if connector is not None:
            try:
                conn_dir = os.path.join(lora_out, connector_name)
                os.makedirs(conn_dir, exist_ok=True)
                torch.save(
                    connector.state_dict(), os.path.join(conn_dir, "pytorch_model.bin")
                )
            except Exception as e:
                logger.warning(f"Failed to save {connector_name}: {e}")


def main() -> None:
    """Main training function."""
    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, VibeVoiceTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging and configuration
    setup_logging(training_args)
    set_seed(training_args.seed)
    configure_gradient_clipping(training_args)

    # Setup model and processor
    model, processor = setup_model_for_training(model_args, training_args)

    # Setup data pipeline
    train_dataset, eval_dataset, data_collator = setup_data_pipeline(
        data_args, model_args, training_args, processor, model
    )

    # Create callbacks
    callbacks = create_callbacks(training_args, model_args)

    # Build the Trainer
    trainer = VibeVoiceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Enable gradient checkpointing if requested
    if getattr(training_args, "gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            logger.warning("Failed to enable gradient checkpointing on the model.")

    # Training
    if training_args.do_train:
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.info("Training completed. Saving final artifacts...")
        save_final_artifacts(model, training_args)

    # Evaluation
    if training_args.do_eval and eval_dataset is not None:
        logger.info("Running evaluation...")
        trainer.evaluate()


if __name__ == "__main__":
    main()
