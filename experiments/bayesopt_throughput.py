#!/usr/bin/env python3
"""
Bayesian Optimization for throughput/memory efficiency.
Goal: Find fastest config that fits in GPU memory.
Agent: bayesopt-agent
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import itertools


@dataclass
class SearchSpace:
    """Define search space for memory optimization."""
    batch_sizes: List[int] = None
    grad_accums: List[int] = None
    lora_ranks: List[int] = None
    ddpm_muls: List[int] = None

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 6, 8]
        if self.grad_accums is None:
            self.grad_accums = [8, 16, 24, 32, 48]
        if self.lora_ranks is None:
            self.lora_ranks = [8, 16, 24, 32, 48]
        if self.ddpm_muls is None:
            self.ddpm_muls = [2, 4, 6, 8]


class ThroughputOptimizer:
    """Bayesian optimizer for throughput."""

    def __init__(self, search_space: SearchSpace, max_memory_gb: float = 22.0):
        self.search_space = search_space
        self.max_memory_gb = max_memory_gb
        self.results: List[Dict] = []
        self.tested_configs = set()

    def config_to_tuple(self, config: Dict) -> Tuple:
        """Convert config to hashable tuple."""
        return (
            config['batch_size'],
            config['grad_accum'],
            config['lora_r'],
            config['ddpm_mul']
        )

    def test_configuration(self, config: Dict) -> Dict:
        """Test a single configuration."""

        config_tuple = self.config_to_tuple(config)
        if config_tuple in self.tested_configs:
            print(f"Skipping already tested config: {config}")
            return None

        print("\n" + "=" * 80)
        print(f"Testing Configuration {len(self.results) + 1}")
        print("=" * 80)
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Grad accum: {config['grad_accum']}")
        print(f"  LoRA rank: {config['lora_r']}")
        print(f"  DDPM mul: {config['ddpm_mul']}")
        print(f"  Effective batch: {config['batch_size'] * config['grad_accum']}")
        print("=" * 80)

        # Run throughput test
        cmd = [
            "python", "experiments/throughput_test.py",
            "--batch_size", str(config['batch_size']),
            "--grad_accum", str(config['grad_accum']),
            "--lora_r", str(config['lora_r']),
            "--lora_alpha", str(config['lora_r'] * 2),
            "--ddpm_mul", str(config['ddpm_mul']),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            # Parse result from JSON
            result_file = f"experiments/throughput_results/test_bs{config['batch_size']}_ga{config['grad_accum']}_r{config['lora_r']}.json"

            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    test_result = json.load(f)

                self.tested_configs.add(config_tuple)
                self.results.append(test_result)

                if test_result['success']:
                    print(f"✓ SUCCESS")
                    print(f"  Throughput: {test_result['samples_per_second']:.2f} samples/sec")
                    print(f"  Memory: {test_result['memory_peak_gb']:.2f}GB")
                    print(f"  ETA: {test_result['estimated_full_runtime_hours']:.1f}h")
                else:
                    reason = "OOM" if test_result['oom_occurred'] else "Error"
                    print(f"✗ FAILED: {reason}")

                return test_result

        except subprocess.TimeoutExpired:
            print("✗ TIMEOUT (>10 minutes)")
            return {
                'config': config,
                'success': False,
                'oom_occurred': False,
                'error_message': 'Timeout'
            }
        except Exception as e:
            print(f"✗ ERROR: {e}")
            return {
                'config': config,
                'success': False,
                'oom_occurred': False,
                'error_message': str(e)
            }

        return None

    def run_initial_grid_search(self, quick: bool = True):
        """Run initial grid search to map out feasible space."""

        print("\n" + "=" * 80)
        print("PHASE 1: INITIAL GRID SEARCH")
        print("=" * 80)

        if quick:
            # Quick scan: test corners of search space
            configs_to_test = [
                # Small configs (should work)
                {'batch_size': 1, 'grad_accum': 32, 'lora_r': 8, 'ddpm_mul': 2},
                {'batch_size': 2, 'grad_accum': 24, 'lora_r': 8, 'ddpm_mul': 4},

                # Medium configs
                {'batch_size': 4, 'grad_accum': 16, 'lora_r': 16, 'ddpm_mul': 4},
                {'batch_size': 4, 'grad_accum': 24, 'lora_r': 24, 'ddpm_mul': 4},

                # Aggressive configs (might OOM)
                {'batch_size': 6, 'grad_accum': 16, 'lora_r': 24, 'ddpm_mul': 6},
                {'batch_size': 8, 'grad_accum': 8, 'lora_r': 32, 'ddpm_mul': 8},
            ]
        else:
            # Full grid (slower but thorough)
            configs_to_test = []
            for bs in [2, 4, 6]:
                for ga in [16, 24, 32]:
                    for r in [8, 16, 24]:
                        for dm in [2, 4, 6]:
                            configs_to_test.append({
                                'batch_size': bs,
                                'grad_accum': ga,
                                'lora_r': r,
                                'ddpm_mul': dm
                            })

        for config in configs_to_test:
            self.test_configuration(config)

        self.save_results()

    def find_best_configuration(self) -> Optional[Dict]:
        """Find best configuration from tested results."""

        # Filter successful runs
        successful = [r for r in self.results if r['success']]

        if not successful:
            print("\n✗ No successful configurations found!")
            return None

        # Sort by throughput (samples/sec)
        best = max(successful, key=lambda x: x['samples_per_second'])

        print("\n" + "=" * 80)
        print("BEST CONFIGURATION FOUND")
        print("=" * 80)
        config = best['config']['config']
        print(f"  Batch size: {config['per_device_train_batch_size']}")
        print(f"  Grad accum: {config['gradient_accumulation_steps']}")
        print(f"  LoRA rank: {config['lora_r']}")
        print(f"  DDPM mul: {config['ddpm_batch_mul']}")
        print(f"  Effective batch: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
        print()
        print(f"  Throughput: {best['samples_per_second']:.2f} samples/sec")
        print(f"  Memory peak: {best['memory_peak_gb']:.2f}GB / {self.max_memory_gb}GB")
        print(f"  Estimated full runtime: {best['estimated_full_runtime_hours']:.1f} hours")
        print("=" * 80)

        return best

    def suggest_refined_configs(self, best: Dict) -> List[Dict]:
        """Suggest refined configurations around the best one."""

        config = best['config']['config']
        bs = config['per_device_train_batch_size']
        ga = config['gradient_accumulation_steps']
        r = config['lora_r']
        dm = config['ddpm_batch_mul']

        suggestions = []

        # Try increasing batch size (more throughput)
        if bs < max(self.search_space.batch_sizes):
            suggestions.append({
                'batch_size': bs + 2,
                'grad_accum': ga,
                'lora_r': r,
                'ddpm_mul': dm
            })

        # Try increasing LoRA rank (better quality, if memory allows)
        if r < max(self.search_space.lora_ranks):
            suggestions.append({
                'batch_size': bs,
                'grad_accum': ga,
                'lora_r': min(r + 8, max(self.search_space.lora_ranks)),
                'ddpm_mul': dm
            })

        # Try different grad accum (trade-off)
        if ga > min(self.search_space.grad_accums):
            suggestions.append({
                'batch_size': bs,
                'grad_accum': ga // 2,
                'lora_r': r,
                'ddpm_mul': dm
            })

        return suggestions

    def run_refinement(self):
        """Refine around best configuration."""

        best = self.find_best_configuration()
        if not best:
            return

        print("\n" + "=" * 80)
        print("PHASE 2: REFINEMENT")
        print("=" * 80)

        suggestions = self.suggest_refined_configs(best)

        for config in suggestions:
            self.test_configuration(config)

        self.save_results()

    def save_results(self, filename: str = "experiments/bayesopt_summary.json"):
        """Save all results to JSON."""

        summary = {
            'total_tested': len(self.results),
            'successful': len([r for r in self.results if r['success']]),
            'failed_oom': len([r for r in self.results if r.get('oom_occurred')]),
            'failed_other': len([r for r in self.results if not r['success'] and not r.get('oom_occurred')]),
            'results': self.results
        }

        # Find best if available
        successful = [r for r in self.results if r['success']]
        if successful:
            best = max(successful, key=lambda x: x['samples_per_second'])
            summary['best_config'] = best

        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Results saved to: {filename}")

    def generate_training_command(self, best: Dict, stage: str = "stage1") -> str:
        """Generate final training command from best config."""

        config = best['config']['config']

        cmd = f"""
# OPTIMIZED CONFIGURATION (Bayesian Optimization Result)
# Agent: bayesopt-agent
# Throughput: {best['samples_per_second']:.2f} samples/sec
# Memory: {best['memory_peak_gb']:.2f}GB
# Est. Runtime: {best['estimated_full_runtime_hours']:.1f}h

python -m vibevoice.finetune.train_vibevoice \\
    --model_name_or_path {config['model_name_or_path']} \\
    --processor_name_or_path {config['processor_name_or_path']} \\
    --dataset_name {config['dataset_name']} \\
    --text_column_name text \\
    --audio_column_name audio \\
    --voice_prompts_column_name audio \\
    --output_dir heiertech/vibevoice-7b-nob-qlora-{stage}-bayesopt \\
    --per_device_train_batch_size {config['per_device_train_batch_size']} \\
    --gradient_accumulation_steps {config['gradient_accumulation_steps']} \\
    --learning_rate 3e-5 \\
    --num_train_epochs {3 if stage == 'stage1' else 2} \\
    --logging_steps 10 \\
    --save_steps 100 \\
    --eval_steps 100 \\
    --report_to wandb \\
    --run_name vibevoice-no-{stage}-bayesopt \\
    --remove_unused_columns False \\
    --bf16 True \\
    --do_train \\
    --gradient_clipping \\
    --gradient_checkpointing True \\
    --ddpm_batch_mul {config['ddpm_batch_mul']} \\
    --diffusion_loss_weight 1.5 \\
    --train_diffusion_head True \\
    --ce_loss_weight 0.045 \\
    --voice_prompt_drop_rate 0.2 \\
    --lora_r {config['lora_r']} \\
    --lora_alpha {config['lora_alpha']} \\
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \\
    --lr_scheduler_type cosine \\
    --warmup_ratio 0.04 \\
    --max_grad_norm 1.0 \\
    --load_in_4bit
"""

        return cmd


def main():
    print("=" * 80)
    print("BAYESIAN OPTIMIZATION FOR THROUGHPUT")
    print("Agent: bayesopt-agent")
    print("Goal: Find fastest config that fits in GPU memory")
    print("=" * 80)

    # Initialize
    search_space = SearchSpace()
    optimizer = ThroughputOptimizer(search_space, max_memory_gb=22.0)

    # Phase 1: Initial search
    quick_mode = "--quick" in sys.argv
    optimizer.run_initial_grid_search(quick=quick_mode)

    # Phase 2: Refinement
    if "--refine" in sys.argv:
        optimizer.run_refinement()

    # Find and display best
    best = optimizer.find_best_configuration()

    if best:
        # Generate training command
        cmd = optimizer.generate_training_command(best, stage="stage1")

        # Save command
        with open("experiments/OPTIMIZED_TRAINING_CMD.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(cmd)

        os.chmod("experiments/OPTIMIZED_TRAINING_CMD.sh", 0o755)

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("1. Review results: cat experiments/bayesopt_summary.json")
        print("2. Run optimized training: bash experiments/OPTIMIZED_TRAINING_CMD.sh")
        print("=" * 80)


if __name__ == "__main__":
    main()
