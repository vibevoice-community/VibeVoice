#!/bin/bash

# VibeVoice 7B Fine-tuning with Unsloth + QLoRA (4-bit)
# Uses pre-quantized checkpoint: marksverdhai/vibevoice-7b-bnb-4bit
# Optimized for 24GB VRAM (RTX 3090)

# Activate virtual environment
source .venv/bin/activate

# Set GPU to use (check nvidia-smi first!)
export CUDA_VISIBLE_DEVICES=0

# Set HF token for model download (if needed)
export HF_TOKEN="${AI_HF_TOKEN:-}"

# Run training with 4-bit quantized model
python -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path marksverdhai/vibevoice-7b-bnb-4bit \
    --processor_name_or_path vibevoice/VibeVoice-7B \
    --load_in_4bit True \
    --dataset_name vibevoice/jenny_vibevoice_formatted \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir finetune_vibevoice_7b_qlora \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2.5e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --report_to tensorboard \
    --remove_unused_columns False \
    --bf16 True \
    --do_train \
    --gradient_clipping \
    --gradient_checkpointing True \
    --ddpm_batch_mul 2 \
    --diffusion_loss_weight 1.4 \
    --train_diffusion_head True \
    --ce_loss_weight 0.04 \
    --voice_prompt_drop_rate 0.2 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_r 8 \
    --lora_alpha 32 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_grad_norm 0.8
