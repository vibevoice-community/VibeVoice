#!/usr/bin/env python3
"""
Fast Bayesian optimization using fast_throughput_test.py
Target: Find optimal config in <30 minutes total.
Agent: bayesopt-agent
"""

import json
import os
import subprocess
import sys
from typing import List, Dict


def generate_search_configs(mode: str = "quick") -> List[Dict]:
    """Generate configs to test."""

    if mode == "quick":
        # 8 configs, ~16 minutes total
        return [
            # Conservative (will definitely work)
            {"batch_size": 2, "grad_accum": 32, "lora_r": 8, "ddpm_mul": 2},
            {"batch_size": 2, "grad_accum": 24, "lora_r": 16, "ddpm_mul": 4},

            # Moderate (likely to work)
            {"batch_size": 4, "grad_accum": 24, "lora_r": 16, "ddpm_mul": 4},
            {"batch_size": 4, "grad_accum": 16, "lora_r": 24, "ddpm_mul": 4},

            # Aggressive (might OOM)
            {"batch_size": 6, "grad_accum": 16, "lora_r": 24, "ddpm_mul": 4},
            {"batch_size": 6, "grad_accum": 24, "lora_r": 16, "ddpm_mul": 6},

            # Very aggressive (probably OOM)
            {"batch_size": 8, "grad_accum": 16, "lora_r": 24, "ddpm_mul": 4},
            {"batch_size": 8, "grad_accum": 8, "lora_r": 32, "ddpm_mul": 6},
        ]

    elif mode == "thorough":
        # 16 configs, ~32 minutes total
        configs = []
        for bs in [2, 4, 6, 8]:
            for r in [8, 16, 24, 32]:
                configs.append({
                    "batch_size": bs,
                    "grad_accum": 128 // bs,  # Keep effective batch ~128
                    "lora_r": r,
                    "ddpm_mul": 4,
                })
        return configs

    else:  # minimal
        # 4 configs, ~8 minutes total
        return [
            {"batch_size": 2, "grad_accum": 24, "lora_r": 16, "ddpm_mul": 4},
            {"batch_size": 4, "grad_accum": 16, "lora_r": 16, "ddpm_mul": 4},
            {"batch_size": 6, "grad_accum": 16, "lora_r": 24, "ddpm_mul": 4},
            {"batch_size": 8, "grad_accum": 8, "lora_r": 24, "ddpm_mul": 4},
        ]


def run_optimization(mode: str = "quick"):
    """Run fast Bayesian optimization."""

    print("=" * 80)
    print("FAST BAYESIAN OPTIMIZATION")
    print("Agent: bayesopt-agent")
    print(f"Mode: {mode}")
    print("=" * 80)

    # Generate configs
    configs = generate_search_configs(mode)
    print(f"\nTesting {len(configs)} configurations")
    print(f"Expected time: ~{len(configs) * 2} minutes\n")

    # Save configs
    os.makedirs("experiments", exist_ok=True)
    config_file = "experiments/test_configs.json"

    with open(config_file, 'w') as f:
        json.dump(configs, f, indent=2)

    print(f"✓ Configs saved to: {config_file}\n")

    # Run fast throughput test
    print("Starting throughput tests...")
    print("=" * 80)

    cmd = [
        "python", "experiments/fast_throughput_test.py",
        "--configs", config_file,
        "--output", "experiments/fast_results.json",
        "--num_batches", "10",
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n✗ Testing failed")
        return None

    # Load and analyze results
    with open("experiments/fast_results.json", 'r') as f:
        results = json.load(f)

    # Find best
    successful = [r for r in results if r['success']]

    if not successful:
        print("\n✗ No successful configurations found!")
        print("All configs caused OOM. Try smaller batch sizes.")
        return None

    best = max(successful, key=lambda x: x['samples_per_sec'])

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)

    cfg = best['config']
    print(f"Batch size:        {cfg['batch_size']}")
    print(f"Grad accumulation: {cfg['grad_accum']}")
    print(f"LoRA rank:         {cfg['lora_r']}")
    print(f"DDPM multiplier:   {cfg['ddpm_mul']}")
    print(f"Effective batch:   {cfg['batch_size'] * cfg['grad_accum']}")
    print()
    print(f"Throughput:        {best['samples_per_sec']:.2f} samples/sec")
    print(f"Memory usage:      {best['memory_gb']:.1f} GB")
    print(f"Est. full runtime: {best['est_runtime_hours']:.1f} hours")
    print("=" * 80)

    # Generate training command
    generate_training_command(cfg, best)

    return best


def generate_training_command(config: Dict, result: Dict):
    """Generate optimized training command."""

    cmd = f"""#!/bin/bash
# OPTIMIZED CONFIGURATION (Fast Bayesian Optimization)
# Agent: bayesopt-agent
# Throughput: {result['samples_per_sec']:.2f} samples/sec
# Memory: {result['memory_gb']:.1f}GB
# Est. Runtime: {result['est_runtime_hours']:.1f}h
#
# Found in <30 minutes using 10-batch stress tests

python -m vibevoice.finetune.train_vibevoice \\
    --model_name_or_path marksverdhai/vibevoice-7b-bnb-8bit \\
    --processor_name_or_path vibevoice/VibeVoice-7B \\
    --dataset_name heiertech/vibevoice-mcv-scripted-no-v24 \\
    --text_column_name text \\
    --audio_column_name audio \\
    --voice_prompts_column_name audio \\
    --output_dir heiertech/vibevoice-7b-nob-qlora-stage1-bayesopt \\
    --per_device_train_batch_size {config['batch_size']} \\
    --gradient_accumulation_steps {config['grad_accum']} \\
    --learning_rate 3e-5 \\
    --num_train_epochs 3 \\
    --logging_steps 10 \\
    --save_steps 100 \\
    --eval_steps 100 \\
    --report_to wandb \\
    --run_name vibevoice-no-stage1-bayesopt \\
    --remove_unused_columns False \\
    --bf16 True \\
    --do_train \\
    --gradient_clipping \\
    --gradient_checkpointing True \\
    --ddpm_batch_mul {config['ddpm_mul']} \\
    --diffusion_loss_weight 1.5 \\
    --train_diffusion_head True \\
    --ce_loss_weight 0.045 \\
    --voice_prompt_drop_rate 0.2 \\
    --lora_r {config['lora_r']} \\
    --lora_alpha {config['lora_r'] * 2} \\
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \\
    --lr_scheduler_type cosine \\
    --warmup_ratio 0.04 \\
    --max_grad_norm 1.0 \\
    --load_in_4bit
"""

    output_file = "experiments/OPTIMIZED_TRAINING_CMD.sh"
    with open(output_file, 'w') as f:
        f.write(cmd)

    os.chmod(output_file, 0o755)

    print(f"\n✓ Training command saved to: {output_file}")
    print(f"\nTo start training:")
    print(f"  bash {output_file}")


def main():
    mode = "quick"

    if len(sys.argv) > 1:
        if "--minimal" in sys.argv:
            mode = "minimal"
        elif "--thorough" in sys.argv:
            mode = "thorough"

    run_optimization(mode)

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review results: cat experiments/fast_results.json")
    print("2. Start training: bash experiments/OPTIMIZED_TRAINING_CMD.sh")
    print("3. Monitor: tail -f heiertech/vibevoice-7b-nob-qlora-stage1-bayesopt/training.log")
    print("=" * 80)


if __name__ == "__main__":
    main()
