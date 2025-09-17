from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HfTrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="vibevoice/VibeVoice-1.5B",
        metadata={"help": "Path to VibeVoice base model with config.json"},
    )
    processor_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to processor dir (preprocessor_config.json). Defaults to model path."
        },
    )
    cache_dir: Optional[str] = field(default=None)
    freeze_acoustic_tokenizer: bool = field(default=True)
    freeze_semantic_tokenizer: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={
            "help": "Comma-separated list of target module names in the LLM blocks"
        },
    )
    lora_wrap_diffusion_head: bool = field(
        default=False, metadata={"help": "Wrap diffusion head with PEFT LoRA"}
    )
    train_diffusion_head: bool = field(
        default=False,
        metadata={"help": "Train diffusion prediction head (full fine-tune)"},
    )
    train_connectors: bool = field(
        default=False,
        metadata={"help": "Train acoustic/semantic connectors (full fine-tune)"},
    )
    layers_to_freeze: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated indices of diffusion head layers to freeze (e.g., '0,1,5,7,8')."
        },
    )


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "HF dataset name or 'json' with --train_jsonl for local files"
        },
    )
    dataset_config_name: Optional[str] = field(default=None)
    train_split_name: str = field(default="train")
    eval_split_name: Optional[str] = field(default="validation")
    text_column_name: str = field(default="text")
    audio_column_name: str = field(default="audio")
    voice_prompts_column_name: Optional[str] = field(default="voice_prompts")
    eval_split_size: float = field(default=0.0)
    ignore_verifications: bool = field(default=False)
    max_length: Optional[int] = field(default=None)
    train_jsonl: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to local train JSONL with {text, audio, [voice_prompts]}"
        },
    )
    validation_jsonl: Optional[str] = field(
        default=None, metadata={"help": "Optional path to local validation JSONL"}
    )
    voice_prompt_drop_rate: float = field(
        default=0.0,
        metadata={
            "help": "Probability to drop conditioning voice prompt during training (0.0 keep always, 1.0 drop always)."
        },
    )


@dataclass
class VibeVoiceTrainingArguments(HfTrainingArguments):
    ddpm_batch_mul: int = field(default=1)
    ce_loss_weight: float = field(default=1.0)
    diffusion_loss_weight: float = field(default=1.0)
    debug_ce_details: bool = field(default=False)
    debug_ce_topk: int = field(default=5)
    debug_ce_max_examples: int = field(default=1)
    debug_ce_every_n_steps: int = field(default=200)
    gradient_clipping: bool = field(
        default=False,
        metadata={
            "help": "Enable gradient clipping using max_grad_norm (set via --max_grad_norm, default 1.0). When False, disables clipping by forcing max_grad_norm=0.0."
        },
    )
