# Quick Start Guide - Bayesian Optimization Experiments

**Agent:** bayesopt-agent
**Time to first results:** ~30 minutes (test run) or 8-24 hours (full run)

## TL;DR

```bash
# 1. You're already on the right branch (unsloth-finetuning)
cd /home/me/VibeVoice

# 2. Scripts are already generated in experiments/

# 3. Run a single experiment (Config 5 - Balanced is a good start)
bash experiments/scripts/train_stage1_config5.sh

# 4. Monitor progress
tail -f experiments/bayesopt_stage1_balanced_*/training.log

# 5. Analyze results when done
python experiments/analyze_results.py
```

## Prerequisites Check

✅ Already done:
- [x] Branch: `unsloth-finetuning`
- [x] Dependencies installed (torch, transformers, unsloth, etc.)
- [x] Virtual environment: `.venv`
- [x] Training scripts generated
- [x] Configs created

## Running Your First Experiment

### Option 1: Test Run (Quick validation - 10 steps)

Edit any config file to test the setup:

```bash
# Edit config to run just 10 steps for testing
cat > experiments/test_run.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate

python -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path marksverdhai/vibevoice-7b-bnb-8bit \
    --processor_name_or_path vibevoice/VibeVoice-7B \
    --dataset_name heiertech/vibevoice-mcv-scripted-no-v24 \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir experiments/test_output \
    --max_steps 10 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --logging_steps 1 \
    --save_steps 10 \
    --bf16 True \
    --do_train \
    --gradient_checkpointing True \
    --ddpm_batch_mul 2 \
    --diffusion_loss_weight 1.5 \
    --train_diffusion_head True \
    --ce_loss_weight 0.045 \
    --voice_prompt_drop_rate 0.2 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --load_in_4bit
EOF

chmod +x experiments/test_run.sh
bash experiments/test_run.sh
```

**Expected:** Completes in ~5-10 minutes, validates setup works

### Option 2: Full Single Config (Recommended first)

Run the balanced configuration (good middle ground):

```bash
bash experiments/scripts/train_stage1_config5.sh
```

**Expected:** 8-12 hours for 3 epochs on RTX 3090

### Option 3: All Configs in Parallel (Multiple GPUs)

```bash
# Requires 2-5 GPUs (or run sequentially)
bash experiments/run_stage1_parallel.sh
```

## Monitoring Your Experiment

### Check Training Log
```bash
# Find your experiment directory
ls -lt experiments/bayesopt_*/ | head -5

# Tail the log (replace with your actual directory)
tail -f experiments/bayesopt_stage1_balanced_iter5_*/training.log
```

### Watch GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check Loss Progress
```bash
# Look for train/loss lines
grep "train/loss" experiments/bayesopt_stage1_*/training.log | tail -20
```

### Weights & Biases (if configured)
Visit: https://wandb.ai/your-username
Look for runs starting with: `bayesopt_stage1_`

## What to Expect

### Startup Phase (5-10 min)
- Model loading
- Dataset downloading/caching
- First forward pass

Output will show:
```
Loading model from: marksverdhai/vibevoice-7b-bnb-8bit
Unsloth: Fast Qwen2 patching...
Loading checkpoint shards: 100%
```

### Training Phase (Hours)
- Regular loss updates every 10 steps
- Checkpoints saved every 100 steps
- Memory usage stable around 18-22 GB

Output will show:
```
{'loss': 2.4561, 'learning_rate': 2.8e-05, 'epoch': 0.12}
{'train/ce_loss': 1.2345, 'train/diffusion_loss': 1.2216}
```

### Completion Phase (Few minutes)
- Final checkpoint saved
- Results written to JSON
- Summary printed

Output will show:
```
Training completed successfully!
Final loss: 1.2345
Results saved to: experiments/bayesopt_*/results.json
```

## Understanding the Output

### Directory Structure
```
experiments/bayesopt_stage1_balanced_iter5_*/
├── config.json          # Hyperparameters used
├── results.json         # Final metrics
├── training.log         # Full training log
├── lora/               # Trained LoRA adapters
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
└── checkpoint-*/       # Intermediate checkpoints
```

### Key Files

**config.json** - Your experiment configuration
**results.json** - Performance metrics to compare
**lora/** - Your trained model (use this for inference)

## Analyzing Results

After one or more experiments complete:

```bash
cd experiments
python analyze_results.py
```

This shows:
- Ranking of all experiments by loss
- Best hyperparameter values
- Suggestions for next experiments
- Correlation analysis

## Common Issues

### Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution:** Edit the config file and:
- Reduce `per_device_train_batch_size` from 4 to 2
- Increase `gradient_accumulation_steps` from 32 to 64
- Reduce `ddpm_batch_mul` from 4 to 2

### Dataset Not Found
```
DatasetNotFoundError: heiertech/vibevoice-mcv-scripted-no-v24
```

**Solution:** Check dataset exists or use alternative:
```bash
--dataset_name mozilla-foundation/common_voice_17_0 \
--dataset_config_name no
```

### Model Loading Fails
```
ValueError: Unrecognized processing class
```

**Solution:** Make sure processor path is set:
```bash
--processor_name_or_path vibevoice/VibeVoice-7B
```

### Very Slow Training
- Check GPU is being used: `nvidia-smi`
- Verify Unsloth loaded: Look for "Unsloth: Fast Qwen2 patching" in log
- Try increasing batch size if memory allows

## Next Steps

### After First Experiment
1. Check `results.json` - Did it complete successfully?
2. Review `training.log` - Was loss decreasing?
3. Compare with other configs (if running parallel)
4. Decide: Continue with this config or try others?

### After All 5 Configs
1. Run analysis: `python experiments/analyze_results.py`
2. Identify best 2-3 configurations
3. Review CONFIG_COMPARISON.md notes
4. Plan iteration 2 with refined parameters

### Preparing for Stage 2
Once you have a winning configuration:
1. Document it in `BEST_CONFIG.md`
2. Adapt for Stage 2 dataset (bokmål only)
3. Reduce epochs from 3 to 2
4. Run Stage 2 experiments

## Getting Help

### Check Status
```bash
# Are any experiments running?
ps aux | grep train_vibevoice

# Check GPU usage
nvidia-smi

# Check disk space
df -h experiments/
```

### Debug Logs
```bash
# Last 50 lines of a log
tail -50 experiments/bayesopt_*/training.log

# Search for errors
grep -i error experiments/bayesopt_*/training.log
grep -i "out of memory" experiments/bayesopt_*/training.log
```

### Quick Validation
```bash
# Test if model loads (without training)
python -c "
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from unsloth import FastLanguageModel
print('Testing model load...')
model, tokenizer = FastLanguageModel.from_pretrained(
    'marksverdhai/vibevoice-7b-bnb-8bit',
    auto_model=VibeVoiceForConditionalGeneration,
    load_in_4bit=True
)
print('✓ Model loads successfully!')
"
```

## Tips for Success

1. **Start Small** - Run test_run.sh first (10 steps)
2. **Monitor Closely** - Watch first 100 steps for issues
3. **Save Results** - Keep notes in CONFIG_COMPARISON.md
4. **Be Patient** - Full training takes hours, not minutes
5. **Compare Fairly** - Let all configs run to completion

## Timeline Estimates

- **Test run (10 steps):** 5-10 minutes
- **Single config (3 epochs):** 8-12 hours
- **All 5 configs (sequential):** 2-3 days
- **All 5 configs (parallel, 5 GPUs):** 8-12 hours
- **Analysis:** 5 minutes

## Success Criteria

Your experiment is successful if:
- ✅ Training completes without errors
- ✅ Loss decreases over time
- ✅ Memory usage stays under GPU limit
- ✅ Results.json is created
- ✅ LoRA adapters are saved

---

**Ready to start?** Run this now:

```bash
bash experiments/scripts/train_stage1_config5.sh
```

Then check back in a few hours! 🚀
