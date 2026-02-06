#!/usr/bin/env python3
"""
Generate training scripts from Bayesian optimization configurations.
Agent: bayesopt-agent
"""

import json
import os
from pathlib import Path
from bayesopt_config import BayesianOptimizer, HyperparameterSpace, TrainingConfig


def generate_training_script(config: TrainingConfig, output_file: str):
    """Generate a bash script for a training configuration."""

    script = f"""#!/bin/bash
# Bayesian Optimization Experiment
# Experiment ID: {config.experiment_id}
# Stage: {config.stage}
# Iteration: {config.iteration}
# Agent: bayesopt-agent

set -e

echo "Starting experiment: {config.experiment_id}"
echo "================================================================"
echo "Stage: {config.stage}"
echo "Model: {config.model_name_or_path}"
echo "Dataset: {config.dataset_name}"
echo "Output: {config.output_dir}"
echo "================================================================"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Create output directory
mkdir -p "{config.output_dir}"

# Save configuration
cat > "{config.output_dir}/config.json" << 'EOF'
{config.to_json()}
EOF

# Log start time
START_TIME=$(date +%s)
echo "Start time: $(date)"

# Run training
python -m vibevoice.finetune.train_vibevoice \\
    --model_name_or_path "{config.model_name_or_path}" \\
    --processor_name_or_path "{config.processor_name_or_path}" \\
    --dataset_name "{config.dataset_name}" \\
    --text_column_name text \\
    --audio_column_name audio \\
    --voice_prompts_column_name audio \\
    --output_dir "{config.output_dir}" \\
    --per_device_train_batch_size {config.per_device_train_batch_size} \\
    --gradient_accumulation_steps {config.gradient_accumulation_steps} \\
    --learning_rate {config.learning_rate} \\
    --num_train_epochs {config.num_train_epochs} \\
    --logging_steps {config.logging_steps} \\
    --save_steps {config.save_steps} \\
    --eval_steps {config.eval_steps} \\
    --report_to wandb \\
    --run_name "{config.experiment_id}" \\
    --remove_unused_columns False \\
    --bf16 {str(config.bf16)} \\
    --do_train \\
    --gradient_clipping \\
    --gradient_checkpointing {str(config.gradient_checkpointing)} \\
    --ddpm_batch_mul {config.ddpm_batch_mul} \\
    --diffusion_loss_weight {config.diffusion_loss_weight} \\
    --train_diffusion_head {str(config.train_diffusion_head)} \\
    --ce_loss_weight {config.ce_loss_weight} \\
    --voice_prompt_drop_rate {config.voice_prompt_drop_rate} \\
    --lora_r {config.lora_r} \\
    --lora_alpha {config.lora_alpha} \\
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \\
    --lr_scheduler_type cosine \\
    --warmup_ratio {config.warmup_ratio} \\
    --max_grad_norm {config.max_grad_norm} \\
    --load_in_4bit {str(config.load_in_4bit)} \\
    2>&1 | tee "{config.output_dir}/training.log"

TRAIN_EXIT_CODE=${{PIPESTATUS[0]}}

# Log end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "End time: $(date)"
echo "Duration: $DURATION seconds"

# Parse final loss from training log
FINAL_LOSS=$(grep -oP 'train/loss[\'"]?: \\K[0-9.]+' "{config.output_dir}/training.log" | tail -1)

# Create results file
cat > "{config.output_dir}/results.json" << EOF
{{
    "experiment_id": "{config.experiment_id}",
    "stage": "{config.stage}",
    "iteration": {config.iteration},
    "training_time_seconds": $DURATION,
    "final_loss": ${{FINAL_LOSS:-null}},
    "exit_code": $TRAIN_EXIT_CODE,
    "config": $(cat "{config.output_dir}/config.json")
}}
EOF

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "================================================================"
    echo "Training completed successfully!"
    echo "Final loss: $FINAL_LOSS"
    echo "Results saved to: {config.output_dir}/results.json"
    echo "================================================================"
else
    echo "================================================================"
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "Check logs at: {config.output_dir}/training.log"
    echo "================================================================"
    exit $TRAIN_EXIT_CODE
fi
"""

    # Write script file
    with open(output_file, 'w') as f:
        f.write(script)

    # Make executable
    os.chmod(output_file, 0o755)

    print(f"Generated script: {output_file}")


def generate_parallel_runner(stage: str, num_configs: int):
    """Generate a script to run multiple experiments in parallel."""

    script = f"""#!/bin/bash
# Parallel Bayesian Optimization Runner
# Stage: {stage}
# Agent: bayesopt-agent

set -e

echo "Running {num_configs} parallel experiments for {stage}"
echo "================================================================"

# Create logs directory
mkdir -p experiments/logs

# Run experiments in background
PIDS=()

"""

    for i in range(1, num_configs + 1):
        script += f"""
echo "Starting experiment {i}/{num_configs}..."
bash experiments/scripts/train_{stage}_config{i}.sh > experiments/logs/{stage}_config{i}.log 2>&1 &
PIDS+=($!)
sleep 5  # Stagger starts to avoid resource contention
"""

    script += """
echo "================================================================"
echo "All experiments started. Waiting for completion..."
echo "Monitor logs in: experiments/logs/"
echo "================================================================"

# Wait for all to complete
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait $PID; then
        echo "Experiment $((i+1)) completed successfully"
    else
        echo "Experiment $((i+1)) failed"
        FAILED=$((FAILED+1))
    fi
done

echo "================================================================"
echo "All experiments completed"
echo "Successful: $((${#PIDS[@]} - FAILED))/${#PIDS[@]}"
echo "Failed: $FAILED/${#PIDS[@]}"
echo "================================================================"

# Aggregate results
python experiments/analyze_results.py
"""

    output_file = f"experiments/run_{stage}_parallel.sh"
    with open(output_file, 'w') as f:
        f.write(script)

    os.chmod(output_file, 0o755)
    print(f"Generated parallel runner: {output_file}")


def main():
    """Generate all training scripts for Stage 1."""

    print("Bayesian Optimization Training Script Generator")
    print("=" * 80)

    # Create directories
    os.makedirs("experiments/configs", exist_ok=True)
    os.makedirs("experiments/scripts", exist_ok=True)
    os.makedirs("experiments/logs", exist_ok=True)

    # Initialize optimizer
    space = HyperparameterSpace()
    optimizer = BayesianOptimizer(space, "stage1")

    # Generate initial configurations
    configs = optimizer.suggest_initial_configs(5)

    print(f"\nGenerated {len(configs)} configurations for Stage 1")
    print("=" * 80)

    # Generate training scripts
    for i, config in enumerate(configs, 1):
        # Save config
        config_file = f"experiments/configs/{config.stage}_config{i}.json"
        optimizer.save_config(config, config_file)

        # Generate training script
        script_file = f"experiments/scripts/train_{config.stage}_config{i}.sh"
        generate_training_script(config, script_file)

        print(f"\nConfig {i}:")
        print(f"  ID: {config.experiment_id}")
        print(f"  LR: {config.learning_rate}")
        print(f"  LoRA: r={config.lora_r}, α={config.lora_alpha}")
        print(f"  Batch: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
        print(f"  Loss weights: diff={config.diffusion_loss_weight}, ce={config.ce_loss_weight}")
        print(f"  Config: {config_file}")
        print(f"  Script: {script_file}")

    # Generate parallel runner
    generate_parallel_runner("stage1", len(configs))

    print("\n" + "=" * 80)
    print("All scripts generated!")
    print("\nTo run experiments:")
    print("  Individual: bash experiments/scripts/train_stage1_config1.sh")
    print("  Parallel:   bash experiments/run_stage1_parallel.sh")
    print("  Analysis:   python experiments/analyze_results.py")


if __name__ == "__main__":
    main()
