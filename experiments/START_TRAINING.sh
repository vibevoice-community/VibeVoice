#!/bin/bash
# Optimized Norwegian Stage 1 Training - Multi-GPU with DDP
# Agent: manual-tuning (optimized from runtime analysis)
# Config: bs=8, ga=6, r=32, lr=4e-5 (tuned for 2x RTX 3090)
# Effective batch size: 8 * 2 GPUs * 6 grad_accum = 96

source .venv/bin/activate

# Multi-GPU training with torchrun (DDP) - better for QLoRA
torchrun --nproc_per_node=2 --master_port=29500 -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path vibevoice/VibeVoice-7B \
    --processor_name_or_path vibevoice/VibeVoice-7B \
    --dataset_name heiertech/vibevoice-mcv-scripted-no-v24 \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir heiertech/vibevoice-7b-nob-qlora-stage1-bayesopt \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 6 \
    --learning_rate 4e-5 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --report_to none \
    --run_name vibevoice-no-stage1-bayesopt-multigpu \
    --remove_unused_columns False \
    --bf16 True \
    --do_train \
    --gradient_clipping \
    --gradient_checkpointing True \
    --ddpm_batch_mul 4 \
    --diffusion_loss_weight 1.5 \
    --train_diffusion_head True \
    --ce_loss_weight 0.045 \
    --voice_prompt_drop_rate 0.2 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.04 \
    --max_grad_norm 1.0 \
    --load_in_4bit \
    --ddp_find_unused_parameters False
