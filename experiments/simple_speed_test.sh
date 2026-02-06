#!/bin/bash
# Simple speed test - just measure throughput with different batch sizes
# Fixed LoRA r=16, only vary batch_size and grad_accum

echo "================================"
echo "Simple 4-Config Speed Test"
echo "Fixed: LoRA r=16, ddpm_mul=4"
echo "Testing: batch size & grad accum"
echo "================================"
echo

# Config 1: bs=4, ga=16
echo "Config 1/4: bs=4, ga=16, eff_batch=64"
python -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path vibevoice/VibeVoice-7B \
    --processor_name_or_path vibevoice/VibeVoice-7B \
    --dataset_name heiertech/vibevoice-mcv-scripted-no-v24 \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir experiments/test_bs4 \
    --max_steps 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-5 \
    --logging_steps 1 \
    --save_steps 1000 \
    --bf16 True \
    --do_train \
    --gradient_checkpointing True \
    --ddpm_batch_mul 4 \
    --diffusion_loss_weight 1.5 \
    --train_diffusion_head True \
    --ce_loss_weight 0.045 \
    --voice_prompt_drop_rate 0.2 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --load_in_4bit \
    2>&1 | grep -E "samples/s|train/loss|OutOfMemory" | tail -5
echo

# Config 2: bs=6, ga=16
echo "Config 2/4: bs=6, ga=16, eff_batch=96"
python -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path vibevoice/VibeVoice-7B \
    --processor_name_or_path vibevoice/VibeVoice-7B \
    --dataset_name heiertech/vibevoice-mcv-scripted-no-v24 \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir experiments/test_bs6 \
    --max_steps 10 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-5 \
    --logging_steps 1 \
    --save_steps 1000 \
    --bf16 True \
    --do_train \
    --gradient_checkpointing True \
    --ddpm_batch_mul 4 \
    --diffusion_loss_weight 1.5 \
    --train_diffusion_head True \
    --ce_loss_weight 0.045 \
    --voice_prompt_drop_rate 0.2 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --load_in_4bit \
    2>&1 | grep -E "samples/s|train/loss|OutOfMemory" | tail -5
echo

# Config 3: bs=8, ga=12
echo "Config 3/4: bs=8, ga=12, eff_batch=96"
python -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path vibevoice/VibeVoice-7B \
    --processor_name_or_path vibevoice/VibeVoice-7B \
    --dataset_name heiertech/vibevoice-mcv-scripted-no-v24 \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir experiments/test_bs8_ga12 \
    --max_steps 10 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 12 \
    --learning_rate 3e-5 \
    --logging_steps 1 \
    --save_steps 1000 \
    --bf16 True \
    --do_train \
    --gradient_checkpointing True \
    --ddpm_batch_mul 4 \
    --diffusion_loss_weight 1.5 \
    --train_diffusion_head True \
    --ce_loss_weight 0.045 \
    --voice_prompt_drop_rate 0.2 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --load_in_4bit \
    2>&1 | grep -E "samples/s|train/loss|OutOfMemory" | tail -5
echo

# Config 4: bs=8, ga=8
echo "Config 4/4: bs=8, ga=8, eff_batch=64"
python -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path vibevoice/VibeVoice-7B \
    --processor_name_or_path vibevoice/VibeVoice-7B \
    --dataset_name heiertech/vibevoice-mcv-scripted-no-v24 \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir experiments/test_bs8_ga8 \
    --max_steps 10 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --logging_steps 1 \
    --save_steps 1000 \
    --bf16 True \
    --do_train \
    --gradient_checkpointing True \
    --ddpm_batch_mul 4 \
    --diffusion_loss_weight 1.5 \
    --train_diffusion_head True \
    --ce_loss_weight 0.045 \
    --voice_prompt_drop_rate 0.2 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --load_in_4bit \
    2>&1 | grep -E "samples/s|train/loss|OutOfMemory" | tail -5

echo
echo "================================"
echo "Test complete! Check which config:"
echo "1. Didn't OOM"
echo "2. Has highest samples/s"
echo "================================"
