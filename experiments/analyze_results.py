#!/usr/bin/env python3
"""
Analyze Bayesian optimization experiment results.
Agent: bayesopt-agent
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import sys


def load_results(results_dir: str = "experiments") -> List[Dict]:
    """Load all experiment results from JSON files."""
    results = []

    # Find all results.json files
    for root, dirs, files in os.walk(results_dir):
        if "results.json" in files:
            filepath = os.path.join(root, "results.json")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")

    return results


def analyze_hyperparameter_impact(results: List[Dict]):
    """Analyze the impact of each hyperparameter on performance."""

    if not results:
        print("No results to analyze")
        return

    # Filter successful runs
    successful = [r for r in results if r.get('exit_code') == 0 and r.get('final_loss') is not None]

    if not successful:
        print("No successful runs to analyze")
        return

    print("\n" + "=" * 80)
    print("HYPERPARAMETER IMPACT ANALYSIS")
    print("=" * 80)

    # Sort by final loss
    sorted_results = sorted(successful, key=lambda x: x['final_loss'])

    print(f"\nTotal successful runs: {len(successful)}")
    print(f"Best loss: {sorted_results[0]['final_loss']:.6f}")
    print(f"Worst loss: {sorted_results[-1]['final_loss']:.6f}")

    # Show top 3 configurations
    print("\n" + "-" * 80)
    print("TOP 3 CONFIGURATIONS")
    print("-" * 80)

    for i, result in enumerate(sorted_results[:3], 1):
        config = result['config']
        print(f"\n#{i} - Loss: {result['final_loss']:.6f}")
        print(f"  Experiment: {result['experiment_id']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  LoRA r/alpha: {config['lora_r']}/{config['lora_alpha']}")
        print(f"  Batch Size: {config['per_device_train_batch_size']}")
        print(f"  Grad Accum: {config['gradient_accumulation_steps']}")
        print(f"  Effective Batch: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
        print(f"  Diffusion Loss Weight: {config['diffusion_loss_weight']}")
        print(f"  CE Loss Weight: {config['ce_loss_weight']}")
        print(f"  Voice Drop Rate: {config['voice_prompt_drop_rate']}")
        print(f"  Warmup Ratio: {config['warmup_ratio']}")
        print(f"  Max Grad Norm: {config['max_grad_norm']}")
        print(f"  DDPM Batch Mul: {config['ddpm_batch_mul']}")
        print(f"  Training Time: {result['training_time_seconds']/3600:.2f} hours")

    # Correlation analysis
    print("\n" + "-" * 80)
    print("PARAMETER CORRELATIONS WITH LOSS")
    print("-" * 80)

    param_keys = [
        'learning_rate', 'lora_r', 'lora_alpha',
        'per_device_train_batch_size', 'gradient_accumulation_steps',
        'diffusion_loss_weight', 'ce_loss_weight', 'voice_prompt_drop_rate',
        'warmup_ratio', 'max_grad_norm', 'ddpm_batch_mul'
    ]

    print("\nNote: This is a simple analysis. For full Bayesian optimization,")
    print("      use a library like scikit-optimize or Optuna.\n")

    # Calculate simple statistics
    for param in param_keys:
        values = [r['config'][param] for r in sorted_results]
        losses = [r['final_loss'] for r in sorted_results]

        # Find best and worst parameter values
        best_idx = losses.index(min(losses))
        worst_idx = losses.index(max(losses))

        best_val = values[best_idx]
        worst_val = values[worst_idx]

        print(f"{param:30} Best: {best_val:8.6g} | Worst: {worst_val:8.6g}")


def generate_next_config_suggestions(results: List[Dict]):
    """Suggest next configurations based on results."""

    successful = [r for r in results if r.get('exit_code') == 0 and r.get('final_loss') is not None]

    if not successful:
        print("\nNo successful runs to base suggestions on.")
        return

    sorted_results = sorted(successful, key=lambda x: x['final_loss'])
    best = sorted_results[0]['config']

    print("\n" + "=" * 80)
    print("SUGGESTED NEXT CONFIGURATIONS")
    print("=" * 80)
    print("\nBased on best performing config, try these variations:\n")

    # Suggestion 1: Increase LoRA capacity
    print("1. Higher LoRA Capacity:")
    print(f"   --lora_r {int(best['lora_r'] * 1.5)}")
    print(f"   --lora_alpha {int(best['lora_alpha'] * 1.5)}")
    print(f"   (Keep other params same as best)")

    # Suggestion 2: Adjust learning rate
    print("\n2. Fine-tune Learning Rate:")
    print(f"   --learning_rate {best['learning_rate'] * 0.8:.6f}  (conservative)")
    print(f"   --learning_rate {best['learning_rate'] * 1.2:.6f}  (aggressive)")

    # Suggestion 3: Optimize batch settings
    eff_batch = best['per_device_train_batch_size'] * best['gradient_accumulation_steps']
    print(f"\n3. Optimize Batch Settings (keep effective batch = {eff_batch}):")
    print(f"   Option A: --per_device_train_batch_size {best['per_device_train_batch_size'] * 2} --gradient_accumulation_steps {best['gradient_accumulation_steps'] // 2}")
    print(f"   Option B: --per_device_train_batch_size {best['per_device_train_batch_size'] // 2} --gradient_accumulation_steps {best['gradient_accumulation_steps'] * 2}")

    # Suggestion 4: Adjust loss weights
    print("\n4. Fine-tune Loss Weights:")
    print(f"   --diffusion_loss_weight {best['diffusion_loss_weight'] * 1.1:.2f}")
    print(f"   --ce_loss_weight {best['ce_loss_weight'] * 0.9:.4f}")

    # Suggestion 5: Voice prompt dropout exploration
    print("\n5. Voice Prompt Dropout Exploration:")
    print(f"   --voice_prompt_drop_rate {max(0.0, best['voice_prompt_drop_rate'] - 0.1):.2f}  (less dropout)")
    print(f"   --voice_prompt_drop_rate {min(1.0, best['voice_prompt_drop_rate'] + 0.1):.2f}  (more dropout)")


def export_results_summary(results: List[Dict], output_file: str = "experiments/results_summary.json"):
    """Export a summary of all results."""

    successful = [r for r in results if r.get('exit_code') == 0 and r.get('final_loss') is not None]
    failed = [r for r in results if r.get('exit_code') != 0 or r.get('final_loss') is None]

    summary = {
        "total_experiments": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "best_loss": min([r['final_loss'] for r in successful]) if successful else None,
        "best_config": sorted(successful, key=lambda x: x['final_loss'])[0] if successful else None,
        "all_results": results
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nResults summary exported to: {output_file}")


def main():
    """Main analysis function."""

    print("=" * 80)
    print("BAYESIAN OPTIMIZATION RESULTS ANALYZER")
    print("Agent: bayesopt-agent")
    print("=" * 80)

    # Load results
    results = load_results()

    if not results:
        print("\nNo results found. Run experiments first.")
        print("Usage: bash experiments/run_stage1_parallel.sh")
        sys.exit(1)

    print(f"\nFound {len(results)} experiment results")

    # Analyze
    analyze_hyperparameter_impact(results)
    generate_next_config_suggestions(results)

    # Export summary
    export_results_summary(results)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
