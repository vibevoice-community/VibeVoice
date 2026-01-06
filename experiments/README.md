# Bayesian Optimization Experiments for Norwegian VibeVoice

**Agent:** bayesopt-agent
**Branch:** unsloth-finetuning
**Goal:** Find optimal hyperparameters for fast Norwegian TTS training

## Overview

This experiment framework uses semi-manual Bayesian optimization to explore hyperparameter space and find the best configuration for training Norwegian VibeVoice models quickly and effectively.

### Training Plan

```
Stage 1: Full Norwegian (bokmål + nynorsk) ← Currently experimenting
├─ Dataset: heiertech/vibevoice-mcv-scripted-no-v24
├─ 3 epochs
├─ Base: marksverdhai/vibevoice-7b-bnb-8bit
└─ Output: heiertech/vibevoice-7b-nob-qlora-stage1-bayesopt*

Stage 2: Bokmål only
├─ Dataset: heiertech/vibevoice-mcv-scripted-nb
├─ 2 epochs
├─ Base: Best Stage 1 checkpoint
└─ Output: heiertech/vibevoice-7b-nob-qlora-stage2-bayesopt*

Stage 3: Østnorsk/Oslo dialect
├─ Dataset: TBD (østnorsk subset)
├─ 1-2 epochs
├─ Base: Best Stage 2 checkpoint
└─ Output: heiertech/vibevoice-7b-nob-qlora-stage3-bayesopt*
```

## Quick Start

### 1. Generate Initial Configurations

```bash
cd experiments
python generate_training_scripts.py
```

This creates 5 initial configurations exploring different hyperparameter regions:

1. **Conservative** - Safe baseline (LR: 2.5e-5, LoRA r=16)
2. **Aggressive** - Faster learning (LR: 4e-5, LoRA r=32)
3. **High Capacity** - Large model capacity (LoRA r=48)
4. **Fast Converge** - Optimized for quick convergence
5. **Balanced** - Middle ground configuration

### 2. Run Experiments

**Option A: Run all in parallel (requires multiple GPUs)**
```bash
bash experiments/run_stage1_parallel.sh
```

**Option B: Run individually**
```bash
# Run first experiment
bash experiments/scripts/train_stage1_config1.sh

# Run second experiment
bash experiments/scripts/train_stage1_config2.sh

# ... etc
```

**Option C: Run sequentially**
```bash
for i in {1..5}; do
    bash experiments/scripts/train_stage1_config$i.sh
done
```

### 3. Analyze Results

```bash
python experiments/analyze_results.py
```

This will:
- Rank configurations by final loss
- Show correlation between hyperparameters and performance
- Suggest next configurations to try
- Export results summary

### 4. Iterate

Based on analysis, generate refined configurations:

```bash
# Edit bayesopt_config.py to create refined configs based on best results
python generate_training_scripts.py --iteration 2

# Run new experiments
bash experiments/run_stage1_parallel.sh
```

## Hyperparameter Search Space

### Learning Rate
- Range: 1e-5 to 5e-5
- Impact: Training speed vs stability
- Initial values: 2e-5, 2.5e-5, 3e-5, 3.5e-5, 4e-5

### LoRA Rank (r) and Alpha (α)
- r range: 8 to 64
- α range: 16 to 128 (typically 2-4× rank)
- Impact: Model capacity vs memory/speed
- Initial values: r=16/α=32, r=24/α=48, r=32/α=64, r=48/α=96

### Batch Configuration
- Batch sizes: 2, 4, 6, 8
- Gradient accumulation: 16, 24, 32, 48, 64
- Effective batch size = batch_size × grad_accum
- Target: 64-256 effective batch size

### Loss Weights
- Diffusion loss: 0.8 to 2.0
- CE loss: 0.02 to 0.08
- Impact: Balance between quality and convergence

### Other Parameters
- Voice prompt drop rate: 0.0 to 0.5
- Warmup ratio: 0.01 to 0.1
- Max grad norm: 0.5 to 1.5
- DDPM batch multiplier: 2 to 8

## Directory Structure

```
experiments/
├── README.md                       # This file
├── bayesopt_config.py              # Configuration generator
├── generate_training_scripts.py   # Script generator
├── analyze_results.py              # Results analyzer
├── run_stage1_parallel.sh          # Parallel runner (generated)
├── configs/                        # JSON configurations
│   ├── stage1_config1.json
│   ├── stage1_config2.json
│   └── ...
├── scripts/                        # Training scripts
│   ├── train_stage1_config1.sh
│   ├── train_stage1_config2.sh
│   └── ...
├── logs/                           # Training logs
│   ├── stage1_config1.log
│   └── ...
├── bayesopt_stage1_*/              # Experiment outputs
│   ├── config.json
│   ├── results.json
│   ├── training.log
│   └── lora/
└── results_summary.json            # Aggregated results
```

## Monitoring Experiments

### Check Running Experiments
```bash
# Watch all logs
tail -f experiments/logs/*.log

# Check specific experiment
tail -f experiments/bayesopt_stage1_conservative_*/training.log
```

### GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Weights & Biases
All experiments log to wandb with run names: `bayesopt_stage1_*`

## Interpreting Results

### Key Metrics

1. **Final Loss** - Primary optimization target
2. **Training Time** - Time to complete 3 epochs
3. **Samples/Second** - Throughput metric
4. **Memory Peak** - Max GPU memory usage

### Good Signs

- Loss consistently decreasing
- Stable training (no NaN or exploding gradients)
- Reasonable training time (<24h per stage)
- Memory usage within GPU limits

### Bad Signs

- Loss plateaus early
- Training crashes or OOM
- Extremely slow training (>48h per stage)
- Unstable loss curves

## Tips for Bayesian Optimization

### Iteration 1: Exploration
- Use diverse initial configurations
- Cover different regions of hyperparameter space
- Don't worry about bad results - they're informative!

### Iteration 2: Exploitation
- Focus on region around best configs
- Small parameter variations
- Test robustness of best configuration

### Iteration 3: Refinement
- Fine-tune the best configuration
- Test edge cases
- Validate on held-out data

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `ddpm_batch_mul`
- Lower `lora_r`

### Training Too Slow
- Increase `per_device_train_batch_size`
- Reduce `gradient_accumulation_steps`
- Reduce `ddpm_batch_mul`
- Lower `lora_r`

### Poor Convergence
- Increase `learning_rate`
- Increase `warmup_ratio`
- Adjust loss weights
- Increase `lora_r`

### Unstable Training
- Decrease `learning_rate`
- Decrease `max_grad_norm`
- Adjust loss weight ratio

## Next Steps

After finding optimal hyperparameters for Stage 1:

1. Apply best config to Stage 2 (bokmål only)
2. Fine-tune for Stage 2 dataset characteristics
3. Apply to Stage 3 (Oslo dialect)
4. Document final configuration in `BEST_CONFIG.md`

## Collaboration

This is running in parallel with other training experiments. Use the `bayesopt` suffix to avoid conflicts:

- Output dirs: `experiments/bayesopt_*`
- Run names: `bayesopt_stage1_*`
- Configs: Separate `experiments/` directory

## Contact

Agent: bayesopt-agent
Method: Semi-manual Bayesian optimization
Goal: Fast, high-quality Norwegian TTS training
