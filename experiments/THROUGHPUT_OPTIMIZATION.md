# Throughput-Based Bayesian Optimization

**Agent:** bayesopt-agent
**Goal:** Find the fastest training configuration that fits in GPU memory

## What This Does

This is **NOT** about finding the best hyperparameters for model quality.
This **IS** about finding the most memory-efficient configuration for maximum training speed.

### The Process

1. **Find longest sample** - Identifies the longest audio in dataset (worst-case memory)
2. **Duplicate it** - Creates a batch of only the longest sample (stress test)
3. **Run 50 batches** - Quick test (~5-10 minutes per config)
4. **Measure throughput** - Samples/second, memory usage, ETA
5. **Bayesian optimization** - Intelligently search for optimal batch size, LoRA rank, etc.
6. **Generate training command** - Output the fastest config that doesn't OOM

## Quick Start

```bash
cd /home/me/VibeVoice

# Option 1: Quick mode (6 configs, ~1 hour)
python experiments/bayesopt_throughput.py --quick

# Option 2: Thorough mode (20+ configs, ~3-4 hours)
python experiments/bayesopt_throughput.py

# Option 3: With refinement (test improvements around best)
python experiments/bayesopt_throughput.py --quick --refine
```

## What Gets Optimized

### Search Space

**Batch Size:** 1, 2, 4, 6, 8
- Higher = faster training (if fits in memory)
- Lower = safer, less memory

**Gradient Accumulation:** 8, 16, 24, 32, 48
- Effective batch = batch_size × grad_accum
- Doesn't affect memory much
- Affects convergence behavior

**LoRA Rank (r):** 8, 16, 24, 32, 48
- Higher = more model capacity, better quality
- Higher = more memory, slower
- This is a trade-off: quality vs speed

**DDPM Batch Multiplier:** 2, 4, 6, 8
- Affects diffusion head training
- Higher = more memory
- Minimal quality impact

### Optimization Goal

**Maximize:** `samples_per_second`

**Constraints:**
- Memory usage < 22GB (RTX 3090)
- No OOM errors
- Must complete 50 batches successfully

## Output

### Results File

`experiments/bayesopt_summary.json`:

```json
{
  "total_tested": 8,
  "successful": 6,
  "failed_oom": 2,
  "best_config": {
    "samples_per_second": 3.45,
    "memory_peak_gb": 20.8,
    "estimated_full_runtime_hours": 8.2,
    "config": {
      "per_device_train_batch_size": 6,
      "gradient_accumulation_steps": 24,
      "lora_r": 24,
      "ddpm_batch_mul": 4
    }
  }
}
```

### Training Command

`experiments/OPTIMIZED_TRAINING_CMD.sh`:

Ready-to-run bash script with optimal configuration:
```bash
#!/bin/bash
python -m vibevoice.finetune.train_vibevoice \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 24 \
    --lora_r 24 \
    ...
```

## Example Run

```bash
$ python experiments/bayesopt_throughput.py --quick

================================================================================
BAYESIAN OPTIMIZATION FOR THROUGHPUT
Agent: bayesopt-agent
Goal: Find fastest config that fits in GPU memory
================================================================================

================================================================================
PHASE 1: INITIAL GRID SEARCH
================================================================================

================================================================================
Testing Configuration 1
================================================================================
  Batch size: 1
  Grad accum: 32
  LoRA rank: 8
  DDPM mul: 2
  Effective batch: 32
================================================================================
[Model loading...]
[Running 50 batches...]
✓ SUCCESS
  Throughput: 2.10 samples/sec
  Memory: 15.2GB
  ETA: 14.3h

================================================================================
Testing Configuration 2
================================================================================
  Batch size: 4
  Grad accum: 16
  LoRA rank: 16
  DDPM mul: 4
  Effective batch: 64
================================================================================
[Model loading...]
[Running 50 batches...]
✓ SUCCESS
  Throughput: 3.20 samples/sec
  Memory: 19.8GB
  ETA: 9.4h

...

================================================================================
BEST CONFIGURATION FOUND
================================================================================
  Batch size: 6
  Grad accum: 24
  LoRA rank: 24
  DDPM mul: 4
  Effective batch: 144

  Throughput: 3.45 samples/sec
  Memory peak: 20.8GB / 22.0GB
  Estimated full runtime: 8.2 hours
================================================================================
```

## Understanding Results

### Key Metrics

**Throughput (samples/sec):**
- Higher is better
- Actual samples processed per second
- Includes forward + backward pass

**Memory Peak (GB):**
- Must be < your GPU memory
- Leave ~1-2GB headroom for safety
- Measured at worst case (longest sample)

**Estimated Runtime (hours):**
- For full 3-epoch training
- Based on 50-batch average
- Assumes ~10k samples in dataset

### Trade-offs

**Batch Size:**
- ↑ = faster training, more memory
- Sweet spot usually 4-8 for RTX 3090

**LoRA Rank:**
- ↑ = better quality, more memory, slower
- 16-24 usually optimal for trade-off
- 8 if you need max speed
- 32-48 if quality > speed

**Gradient Accumulation:**
- Doesn't affect memory much
- Higher = more stable but slower updates
- 16-32 usually good

## Tips

### If Everything OOMs

```bash
# Try even smaller configs
python -c "
from experiments.bayesopt_throughput import ThroughputOptimizer, SearchSpace

space = SearchSpace(
    batch_sizes=[1, 2],
    lora_ranks=[4, 8],
    ddpm_muls=[2]
)
optimizer = ThroughputOptimizer(space, max_memory_gb=22.0)
optimizer.run_initial_grid_search(quick=True)
optimizer.find_best_configuration()
"
```

### If You Want Max Quality

After finding fastest config, test higher LoRA ranks manually:

```bash
# Test if r=32 fits
python experiments/throughput_test.py --batch_size 4 --grad_accum 24 --lora_r 32

# Test if r=48 fits
python experiments/throughput_test.py --batch_size 2 --grad_accum 32 --lora_r 48
```

### If You Have More/Less GPU Memory

Edit `bayesopt_throughput.py`:

```python
# For RTX 4090 (24GB)
optimizer = ThroughputOptimizer(search_space, max_memory_gb=23.0)

# For RTX 3060 (12GB)
optimizer = ThroughputOptimizer(search_space, max_memory_gb=11.0)
```

## Manual Testing

Test a specific config:

```bash
python experiments/throughput_test.py \
    --batch_size 4 \
    --grad_accum 24 \
    --lora_r 16 \
    --lora_alpha 32 \
    --ddpm_mul 4
```

Results in `experiments/throughput_results/test_bs4_ga24_r16.json`

## Norwegian Training Plan Integration

### Stage 1: Full Norwegian

```bash
# 1. Find optimal config
python experiments/bayesopt_throughput.py --quick

# 2. Run training with optimized config
bash experiments/OPTIMIZED_TRAINING_CMD.sh
```

### Stage 2: Bokmål Only

Use same config from Stage 1, just:
- Change dataset to `heiertech/vibevoice-mcv-scripted-nb`
- Reduce epochs from 3 to 2
- Change base model to Stage 1 output

### Stage 3: Oslo Dialect

Use same config, adjust for smaller dataset if needed.

## FAQ

**Q: Why 50 batches?**
A: Enough to measure steady-state performance, fast enough to test many configs (<10min each)

**Q: Why use longest sample?**
A: Worst-case memory test. If it works with longest sample, it works with all samples.

**Q: What if my dataset is different?**
A: Re-run optimization. Different audio lengths = different memory profile.

**Q: Can I trust the ETA?**
A: It's an estimate. Actual time may vary ±20% depending on dataset variance.

**Q: Should I use the fastest config always?**
A: Depends! Fastest isn't always best quality. If time isn't critical, test higher LoRA ranks.

## Next Steps

1. Run optimization: `python experiments/bayesopt_throughput.py --quick`
2. Review results: `cat experiments/bayesopt_summary.json`
3. Check command: `cat experiments/OPTIMIZED_TRAINING_CMD.sh`
4. Run training: `bash experiments/OPTIMIZED_TRAINING_CMD.sh`
5. Monitor: `tail -f heiertech/vibevoice-7b-nob-qlora-stage1-bayesopt/training.log`

---

**Remember:** This finds the fastest config, not necessarily the best quality.
For production, you may want to sacrifice some speed for higher LoRA rank.
