#!/usr/bin/env python3
"""
Bayesian Optimization Configuration for Norwegian VibeVoice Training
Agent: bayesopt-agent
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class HyperparameterSpace:
    """Define the hyperparameter search space for Bayesian optimization."""

    # Learning rate range
    lr_min: float = 1e-5
    lr_max: float = 5e-5

    # LoRA rank range
    lora_r_min: int = 8
    lora_r_max: int = 64

    # LoRA alpha range (typically 2-4x rank)
    lora_alpha_min: int = 16
    lora_alpha_max: int = 128

    # Batch size options
    batch_sizes: List[int] = None

    # Gradient accumulation options
    grad_accum_options: List[int] = None

    # Diffusion loss weight range
    diff_loss_weight_min: float = 0.8
    diff_loss_weight_max: float = 2.0

    # CE loss weight range
    ce_loss_weight_min: float = 0.02
    ce_loss_weight_max: float = 0.08

    # Voice prompt drop rate range
    voice_drop_min: float = 0.0
    voice_drop_max: float = 0.5

    # Warmup ratio range
    warmup_ratio_min: float = 0.01
    warmup_ratio_max: float = 0.1

    # Max grad norm range
    max_grad_norm_min: float = 0.5
    max_grad_norm_max: float = 1.5

    # DDPM batch multiplier options
    ddpm_batch_mul_options: List[int] = None

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [2, 4, 6, 8]
        if self.grad_accum_options is None:
            self.grad_accum_options = [16, 24, 32, 48, 64]
        if self.ddpm_batch_mul_options is None:
            self.ddpm_batch_mul_options = [2, 4, 6, 8]


@dataclass
class TrainingConfig:
    """Configuration for a single training run."""

    # Experiment metadata
    experiment_id: str
    stage: str  # "stage1", "stage2", "stage3"
    iteration: int

    # Model and data
    model_name_or_path: str
    processor_name_or_path: str
    dataset_name: str
    output_dir: str

    # Hyperparameters to optimize
    learning_rate: float
    lora_r: int
    lora_alpha: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    diffusion_loss_weight: float
    ce_loss_weight: float
    voice_prompt_drop_rate: float
    warmup_ratio: float
    max_grad_norm: float
    ddpm_batch_mul: int

    # Fixed parameters
    num_train_epochs: int = 3
    gradient_checkpointing: bool = True
    load_in_4bit: bool = True
    bf16: bool = True
    train_diffusion_head: bool = True

    # Other settings
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # Results (filled after training)
    final_loss: Optional[float] = None
    training_time: Optional[float] = None
    samples_per_second: Optional[float] = None
    memory_peak_gb: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingConfig':
        return cls(**d)


class BayesianOptimizer:
    """Semi-manual Bayesian optimization for hyperparameter tuning."""

    def __init__(self, search_space: HyperparameterSpace, stage: str):
        self.search_space = search_space
        self.stage = stage
        self.configs: List[TrainingConfig] = []
        self.results: List[Dict] = []

    def suggest_initial_configs(self, n: int = 5) -> List[TrainingConfig]:
        """Generate initial configurations using grid/random sampling."""
        import random

        configs = []

        # Configuration 1: Conservative baseline
        configs.append(self._create_config(
            iteration=1,
            name="conservative",
            lr=2.5e-5,
            lora_r=16,
            lora_alpha=32,
            batch_size=4,
            grad_accum=32,
            diff_loss=1.4,
            ce_loss=0.04,
            voice_drop=0.2,
            warmup=0.03,
            max_norm=0.8,
            ddpm_mul=4
        ))

        # Configuration 2: Aggressive (faster learning)
        configs.append(self._create_config(
            iteration=2,
            name="aggressive",
            lr=4e-5,
            lora_r=32,
            lora_alpha=64,
            batch_size=6,
            grad_accum=24,
            diff_loss=1.6,
            ce_loss=0.06,
            voice_drop=0.1,
            warmup=0.05,
            max_norm=1.0,
            ddpm_mul=6
        ))

        # Configuration 3: High capacity
        configs.append(self._create_config(
            iteration=3,
            name="high_capacity",
            lr=2e-5,
            lora_r=48,
            lora_alpha=96,
            batch_size=2,
            grad_accum=48,
            diff_loss=1.2,
            ce_loss=0.03,
            voice_drop=0.3,
            warmup=0.04,
            max_norm=0.9,
            ddpm_mul=8
        ))

        # Configuration 4: Fast convergence
        configs.append(self._create_config(
            iteration=4,
            name="fast_converge",
            lr=3.5e-5,
            lora_r=24,
            lora_alpha=48,
            batch_size=8,
            grad_accum=16,
            diff_loss=1.8,
            ce_loss=0.05,
            voice_drop=0.15,
            warmup=0.06,
            max_norm=1.2,
            ddpm_mul=4
        ))

        # Configuration 5: Balanced
        configs.append(self._create_config(
            iteration=5,
            name="balanced",
            lr=3e-5,
            lora_r=20,
            lora_alpha=40,
            batch_size=4,
            grad_accum=32,
            diff_loss=1.5,
            ce_loss=0.045,
            voice_drop=0.25,
            warmup=0.04,
            max_norm=1.0,
            ddpm_mul=5
        ))

        self.configs.extend(configs[:n])
        return configs[:n]

    def _create_config(self, iteration: int, name: str, **kwargs) -> TrainingConfig:
        """Create a training configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"bayesopt_{self.stage}_{name}_iter{iteration}_{timestamp}"

        # Base paths
        if self.stage == "stage1":
            model_path = "marksverdhai/vibevoice-7b-bnb-8bit"
            dataset = "heiertech/vibevoice-mcv-scripted-no-v24"
            output_prefix = "heiertech/vibevoice-7b-nob-qlora-stage1"
        elif self.stage == "stage2":
            model_path = "heiertech/vibevoice-7b-nob-qlora-stage1"
            dataset = "heiertech/vibevoice-mcv-scripted-nb"
            output_prefix = "heiertech/vibevoice-7b-nob-qlora-stage2"
        else:  # stage3
            model_path = "heiertech/vibevoice-7b-nob-qlora-stage2"
            dataset = "TBD"
            output_prefix = "heiertech/vibevoice-7b-nob-qlora-stage3"

        output_dir = f"experiments/bayesopt_{self.stage}_{name}_iter{iteration}"

        return TrainingConfig(
            experiment_id=exp_id,
            stage=self.stage,
            iteration=iteration,
            model_name_or_path=model_path,
            processor_name_or_path="vibevoice/VibeVoice-7B",
            dataset_name=dataset,
            output_dir=output_dir,
            learning_rate=kwargs.get('lr'),
            lora_r=kwargs.get('lora_r'),
            lora_alpha=kwargs.get('lora_alpha'),
            per_device_train_batch_size=kwargs.get('batch_size'),
            gradient_accumulation_steps=kwargs.get('grad_accum'),
            diffusion_loss_weight=kwargs.get('diff_loss'),
            ce_loss_weight=kwargs.get('ce_loss'),
            voice_prompt_drop_rate=kwargs.get('voice_drop'),
            warmup_ratio=kwargs.get('warmup'),
            max_grad_norm=kwargs.get('max_norm'),
            ddpm_batch_mul=kwargs.get('ddpm_mul'),
            num_train_epochs=3 if self.stage == "stage1" else (2 if self.stage == "stage2" else 1)
        )

    def suggest_next_config(self, best_results: List[Dict]) -> TrainingConfig:
        """
        Suggest next configuration based on previous results.
        This is semi-manual - you provide guidance on what to explore.
        """
        # Analyze best performing configs
        if not best_results:
            return self.suggest_initial_configs(1)[0]

        # Find best configuration
        best = min(best_results, key=lambda x: x.get('final_loss', float('inf')))

        # Explore around the best configuration
        iteration = len(self.configs) + 1

        # Vary parameters slightly around best
        import random

        config = self._create_config(
            iteration=iteration,
            name="refined",
            lr=best['learning_rate'] * random.uniform(0.8, 1.2),
            lora_r=int(best['lora_r'] * random.uniform(0.8, 1.2)),
            lora_alpha=int(best['lora_alpha'] * random.uniform(0.8, 1.2)),
            batch_size=best['per_device_train_batch_size'],
            grad_accum=best['gradient_accumulation_steps'],
            diff_loss=best['diffusion_loss_weight'] * random.uniform(0.9, 1.1),
            ce_loss=best['ce_loss_weight'] * random.uniform(0.9, 1.1),
            voice_drop=best['voice_prompt_drop_rate'] * random.uniform(0.8, 1.2),
            warmup=best['warmup_ratio'] * random.uniform(0.8, 1.2),
            max_norm=best['max_grad_norm'] * random.uniform(0.9, 1.1),
            ddpm_mul=best['ddpm_batch_mul']
        )

        self.configs.append(config)
        return config

    def save_config(self, config: TrainingConfig, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            f.write(config.to_json())

    def load_config(self, filepath: str) -> TrainingConfig:
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return TrainingConfig.from_dict(data)


if __name__ == "__main__":
    # Example usage
    space = HyperparameterSpace()
    optimizer = BayesianOptimizer(space, "stage1")

    # Generate initial configurations
    configs = optimizer.suggest_initial_configs(5)

    print("Generated 5 initial configurations for Stage 1:")
    print("=" * 80)

    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {config.experiment_id}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  LoRA r/alpha: {config.lora_r}/{config.lora_alpha}")
        print(f"  Batch Size: {config.per_device_train_batch_size}")
        print(f"  Grad Accum: {config.gradient_accumulation_steps}")
        print(f"  Effective Batch: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
        print(f"  Diffusion/CE Loss: {config.diffusion_loss_weight}/{config.ce_loss_weight}")
        print(f"  Voice Drop Rate: {config.voice_prompt_drop_rate}")
        print(f"  Output: {config.output_dir}")

        # Save config
        import os
        os.makedirs("experiments/configs", exist_ok=True)
        filepath = f"experiments/configs/config_{i}.json"
        optimizer.save_config(config, filepath)
        print(f"  Saved to: {filepath}")

    print("\n" + "=" * 80)
    print("Configurations saved! Use these to run parallel experiments.")
