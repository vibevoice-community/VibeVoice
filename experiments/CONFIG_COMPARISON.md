# Bayesian Optimization - Configuration Comparison

**Agent:** bayesopt-agent
**Stage:** Stage 1 - Full Norwegian
**Generated:** 2026-01-05

## Configuration Overview

| Config | Name | Learning Rate | LoRA (r/α) | Batch Size | Grad Accum | Eff. Batch | Diff Loss | CE Loss | Voice Drop |
|--------|------|--------------|------------|------------|------------|------------|-----------|---------|------------|
| 1 | Conservative | 2.5e-5 | 16/32 | 4 | 32 | 128 | 1.4 | 0.040 | 0.20 |
| 2 | Aggressive | 4.0e-5 | 32/64 | 6 | 24 | 144 | 1.6 | 0.060 | 0.10 |
| 3 | High Capacity | 2.0e-5 | 48/96 | 2 | 48 | 96 | 1.2 | 0.030 | 0.30 |
| 4 | Fast Converge | 3.5e-5 | 24/48 | 8 | 16 | 128 | 1.8 | 0.050 | 0.15 |
| 5 | Balanced | 3.0e-5 | 20/40 | 4 | 32 | 128 | 1.5 | 0.045 | 0.25 |

## Detailed Configuration Matrix

### Config 1: Conservative (Baseline)
```
Purpose: Safe, stable baseline
Strategy: Proven hyperparameters, moderate learning
Expected: Reliable convergence, slower training
```
- **Learning Rate:** 2.5e-5 (conservative)
- **LoRA r/α:** 16/32 (standard)
- **Batch Config:** 4 × 32 = 128 (balanced)
- **Loss Weights:** 1.4 / 0.04 (standard ratio)
- **Voice Drop:** 0.20 (moderate regularization)
- **Warmup:** 0.03 (3%)
- **Max Grad Norm:** 0.8
- **DDPM Mul:** 4

**Best for:** Establishing baseline, ensuring stability

---

### Config 2: Aggressive (Fast Training)
```
Purpose: Maximum learning speed
Strategy: High LR + large capacity + big batches
Expected: Fast convergence, risk of instability
```
- **Learning Rate:** 4.0e-5 (high)
- **LoRA r/α:** 32/64 (high capacity)
- **Batch Config:** 6 × 24 = 144 (large effective)
- **Loss Weights:** 1.6 / 0.06 (emphasize both)
- **Voice Drop:** 0.10 (low, rely on data)
- **Warmup:** 0.05 (5%)
- **Max Grad Norm:** 1.0
- **DDPM Mul:** 6

**Best for:** Fast experiments, finding upper LR bound

---

### Config 3: High Capacity (Quality Focus)
```
Purpose: Maximum model expressiveness
Strategy: Large LoRA rank, careful learning
Expected: Best quality, slower training, more memory
```
- **Learning Rate:** 2.0e-5 (conservative for large model)
- **LoRA r/α:** 48/96 (very high capacity)
- **Batch Config:** 2 × 48 = 96 (memory constrained)
- **Loss Weights:** 1.2 / 0.03 (diffusion dominant)
- **Voice Drop:** 0.30 (strong regularization needed)
- **Warmup:** 0.04 (4%)
- **Max Grad Norm:** 0.9
- **DDPM Mul:** 8

**Best for:** Final production model, quality over speed

---

### Config 4: Fast Converge (Optimizer Focus)
```
Purpose: Optimize for quick convergence
Strategy: Higher LR + larger batches + aggressive losses
Expected: Fast training, early convergence
```
- **Learning Rate:** 3.5e-5 (aggressive)
- **LoRA r/α:** 24/48 (moderate capacity)
- **Batch Config:** 8 × 16 = 128 (large batch, low accum)
- **Loss Weights:** 1.8 / 0.05 (high diffusion weight)
- **Voice Drop:** 0.15 (low regularization)
- **Warmup:** 0.06 (6%, longer warmup)
- **Max Grad Norm:** 1.2
- **DDPM Mul:** 4

**Best for:** Finding convergence patterns, throughput testing

---

### Config 5: Balanced (Middle Ground)
```
Purpose: Balance all trade-offs
Strategy: Middle values across all parameters
Expected: Reliable, good starting point for refinement
```
- **Learning Rate:** 3.0e-5 (moderate-high)
- **LoRA r/α:** 20/40 (moderate capacity)
- **Batch Config:** 4 × 32 = 128 (standard)
- **Loss Weights:** 1.5 / 0.045 (balanced)
- **Voice Drop:** 0.25 (moderate regularization)
- **Warmup:** 0.04 (4%)
- **Max Grad Norm:** 1.0
- **DDPM Mul:** 5

**Best for:** General purpose, refinement base

---

## Hypothesis & Expected Outcomes

### Performance Predictions

**Fastest Training:**
1. Config 4 (Fast Converge) - Optimized for throughput
2. Config 2 (Aggressive) - High LR + large batches
3. Config 5 (Balanced) - Efficient middle ground

**Best Quality:**
1. Config 3 (High Capacity) - Largest model capacity
2. Config 1 (Conservative) - Most stable training
3. Config 5 (Balanced) - Good all-around

**Best Trade-off:**
1. Config 5 (Balanced) - Designed for it
2. Config 1 (Conservative) - Reliable fallback
3. Config 4 (Fast Converge) - If quality holds up

### Key Questions to Answer

1. **Learning Rate Sensitivity**
   - How does 2e-5 vs 4e-5 affect convergence?
   - What's the optimal LR for Norwegian data?

2. **LoRA Capacity**
   - Is r=48 worth the extra memory/time?
   - What's the minimum r for good quality?

3. **Batch Configuration**
   - Does 8×16 train faster than 4×32 with same effective batch?
   - Impact of accumulation steps on convergence?

4. **Loss Weight Ratio**
   - Optimal diffusion/CE weight balance for Norwegian?
   - Does higher diffusion weight improve prosody?

5. **Voice Prompt Dropout**
   - Is 0.3 too high for Norwegian dataset?
   - Impact on generalization vs quality?

## Experiment Protocol

### Phase 1: Initial Run (Current)
- Run all 5 configs
- Monitor for crashes/OOM
- Track training speed and loss curves
- Collect results after 100-200 steps

### Phase 2: Analysis
- Compare loss curves
- Identify best performers
- Analyze failure modes
- Calculate efficiency metrics

### Phase 3: Refinement
- Generate 3-5 new configs based on results
- Focus on regions around best configs
- Test edge cases
- Validate best configuration

### Phase 4: Final Selection
- Run best 2-3 configs to completion (3 epochs)
- Evaluate on held-out data
- Select final config for Stage 2

## Running the Experiments

### Individual Runs (Recommended for testing)
```bash
# Test first config
bash experiments/scripts/train_stage1_config1.sh

# Monitor progress
tail -f experiments/bayesopt_stage1_conservative_*/training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Parallel Runs (Requires multiple GPUs)
```bash
# Run all at once
bash experiments/run_stage1_parallel.sh

# Monitor all
tail -f experiments/logs/*.log
```

### Sequential Runs (Single GPU)
```bash
# Run one after another
for i in {1..5}; do
    echo "Running config $i..."
    bash experiments/scripts/train_stage1_config$i.sh
done
```

## Results Template

After each experiment completes, update this section:

### Config 1: Conservative
- **Status:** [ ] Running / [ ] Complete / [ ] Failed
- **Final Loss:** _____
- **Training Time:** _____ hours
- **Memory Peak:** _____ GB
- **Notes:**

### Config 2: Aggressive
- **Status:** [ ] Running / [ ] Complete / [ ] Failed
- **Final Loss:** _____
- **Training Time:** _____ hours
- **Memory Peak:** _____ GB
- **Notes:**

### Config 3: High Capacity
- **Status:** [ ] Running / [ ] Complete / [ ] Failed
- **Final Loss:** _____
- **Training Time:** _____ hours
- **Memory Peak:** _____ GB
- **Notes:**

### Config 4: Fast Converge
- **Status:** [ ] Running / [ ] Complete / [ ] Failed
- **Final Loss:** _____
- **Training Time:** _____ hours
- **Memory Peak:** _____ GB
- **Notes:**

### Config 5: Balanced
- **Status:** [ ] Running / [ ] Complete / [ ] Failed
- **Final Loss:** _____
- **Training Time:** _____ hours
- **Memory Peak:** _____ GB
- **Notes:**

---

## Next Iteration Planning

Based on results, next batch will explore:
- [ ] Fine-tuned LR around best config
- [ ] LoRA rank variations (±25%)
- [ ] Batch size optimization
- [ ] Loss weight refinement
- [ ] Regularization tuning

Use `python experiments/analyze_results.py` for automated suggestions.
