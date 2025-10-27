#!/bin/bash

# Training script for VibeVoice with DeepSpeed ZeRO-2 and Full Fine-Tuning
# Optimized for 2x NVIDIA RTX 4090 GPUs (24GB each)
# No LoRA - Full parameter training with BF16
# DeepSpeed ZeRO-2 handles distributed training (shards gradients & optimizer states, not params)

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Enable TF32 for faster matmul on Ampere GPUs (4090)
export NVIDIA_TF32_OVERRIDE=1

deepspeed --num_gpus=2 --module vibevoice.finetune.train_vibevoice \
    --deepspeed deepspeed_config_zero2.json \
    --model_name_or_path vibevoice/VibeVoice-1.5B \
    --dataset_name vibevoice/jenny_vibevoice_formatted \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir finetune_vibevoice_fft \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --report_to wandb \
    --remove_unused_columns False \
    --bf16 True \
    --tf32 True \
    --do_train \
    --gradient_clipping \
    --gradient_checkpointing True \
    --ddpm_batch_mul 4 \
    --diffusion_loss_weight 1.4 \
    --ce_loss_weight 0.04 \
    --voice_prompt_drop_rate 0.2 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_grad_norm 0.8 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True
