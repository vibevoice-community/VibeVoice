"""
VibeVoice finetuning utilities.
"""

from .arguments import (DataArguments, ModelArguments,
                        VibeVoiceTrainingArguments)
from .callbacks import create_callbacks
from .data_utils import setup_data_pipeline
from .ema_callback import EmaCallback
from .lora_config import (apply_diffusion_head_lora, apply_lm_lora,
                          enable_lm_lora_parameters, freeze_all_parameters)
from .model_introspection import (ModelComponents,
                                  freeze_diffusion_head_layers,
                                  setup_connector_training,
                                  setup_diffusion_head_training,
                                  setup_tokenizer_freezing)
from .model_setup import (freeze_embeddings_and_head, hard_tie_lm_head,
                          load_model, load_processor, log_lm_head_diagnostics,
                          log_trainable_parameters,
                          patch_acoustic_encode_for_legacy_indexing,
                          setup_model_cache_and_tokenizers,
                          validate_special_tokens)
from .trainer import VibeVoiceTrainer

__all__ = [
    "ModelArguments",
    "DataArguments",
    "VibeVoiceTrainingArguments",
    "load_processor",
    "load_model",
    "patch_acoustic_encode_for_legacy_indexing",
    "setup_model_cache_and_tokenizers",
    "hard_tie_lm_head",
    "freeze_embeddings_and_head",
    "log_lm_head_diagnostics",
    "validate_special_tokens",
    "log_trainable_parameters",
    "ModelComponents",
    "setup_tokenizer_freezing",
    "setup_connector_training",
    "setup_diffusion_head_training",
    "freeze_diffusion_head_layers",
    "apply_lm_lora",
    "apply_diffusion_head_lora",
    "enable_lm_lora_parameters",
    "freeze_all_parameters",
    "setup_data_pipeline",
    "VibeVoiceTrainer",
    "create_callbacks",
    "EmaCallback",
]
