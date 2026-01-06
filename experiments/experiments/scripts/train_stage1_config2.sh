#!/bin/bash
# Bayesian Optimization Experiment
# Experiment ID: bayesopt_stage1_aggressive_iter2_20260105_195256
# Stage: stage1
# Iteration: 2
# Agent: bayesopt-agent

set -e

echo "Starting experiment: bayesopt_stage1_aggressive_iter2_20260105_195256"
echo "================================================================"
echo "Stage: stage1"
echo "Model: marksverdhai/vibevoice-7b-bnb-8bit"
echo "Dataset: heiertech/vibevoice-mcv-scripted-no-v24"
echo "Output: experiments/bayesopt_stage1_aggressive_iter2"
echo "================================================================"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Create output directory
mkdir -p "experiments/bayesopt_stage1_aggressive_iter2"

# Save configuration
cat > "experiments/bayesopt_stage1_aggressive_iter2/config.json" << 'EOF'
{
  "experiment_id": "bayesopt_stage1_aggressive_iter2_20260105_195256",
  "stage": "stage1",
  "iteration": 2,
  "model_name_or_path": "marksverdhai/vibevoice-7b-bnb-8bit",
  "processor_name_or_path": "vibevoice/VibeVoice-7B",
  "dataset_name": "heiertech/vibevoice-mcv-scripted-no-v24",
  "output_dir": "experiments/bayesopt_stage1_aggressive_iter2",
  "learning_rate": 4e-05,
  "lora_r": 32,
  "lora_alpha": 64,
  "per_device_train_batch_size": 6,
  "gradient_accumulation_steps": 24,
  "diffusion_loss_weight": 1.6,
  "ce_loss_weight": 0.06,
  "voice_prompt_drop_rate": 0.1,
  "warmup_ratio": 0.05,
  "max_grad_norm": 1.0,
  "ddpm_batch_mul": 6,
  "num_train_epochs": 3,
  "gradient_checkpointing": true,
  "load_in_4bit": true,
  "bf16": true,
  "train_diffusion_head": true,
  "logging_steps": 10,
  "save_steps": 100,
  "eval_steps": 100,
  "final_loss": null,
  "training_time": null,
  "samples_per_second": null,
  "memory_peak_gb": null
}
EOF

# Log start time
START_TIME=$(date +%s)
echo "Start time: $(date)"

# Run training
python -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path "marksverdhai/vibevoice-7b-bnb-8bit" \
    --processor_name_or_path "vibevoice/VibeVoice-7B" \
    --dataset_name "heiertech/vibevoice-mcv-scripted-no-v24" \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir "experiments/bayesopt_stage1_aggressive_iter2" \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 24 \
    --learning_rate 4e-05 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --report_to wandb \
    --run_name "bayesopt_stage1_aggressive_iter2_20260105_195256" \
    --remove_unused_columns False \
    --bf16 True \
    --do_train \
    --gradient_clipping \
    --gradient_checkpointing True \
    --ddpm_batch_mul 6 \
    --diffusion_loss_weight 1.6 \
    --train_diffusion_head True \
    --ce_loss_weight 0.06 \
    --voice_prompt_drop_rate 0.1 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --load_in_4bit True \
    2>&1 | tee "experiments/bayesopt_stage1_aggressive_iter2/training.log"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# Log end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "End time: $(date)"
echo "Duration: $DURATION seconds"

# Parse final loss from training log
FINAL_LOSS=$(grep -oP 'train/loss['"]?: \K[0-9.]+' "experiments/bayesopt_stage1_aggressive_iter2/training.log" | tail -1)

# Create results file
cat > "experiments/bayesopt_stage1_aggressive_iter2/results.json" << EOF
{
    "experiment_id": "bayesopt_stage1_aggressive_iter2_20260105_195256",
    "stage": "stage1",
    "iteration": 2,
    "training_time_seconds": $DURATION,
    "final_loss": ${FINAL_LOSS:-null},
    "exit_code": $TRAIN_EXIT_CODE,
    "config": $(cat "experiments/bayesopt_stage1_aggressive_iter2/config.json")
}
EOF

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "================================================================"
    echo "Training completed successfully!"
    echo "Final loss: $FINAL_LOSS"
    echo "Results saved to: experiments/bayesopt_stage1_aggressive_iter2/results.json"
    echo "================================================================"
else
    echo "================================================================"
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "Check logs at: experiments/bayesopt_stage1_aggressive_iter2/training.log"
    echo "================================================================"
    exit $TRAIN_EXIT_CODE
fi
