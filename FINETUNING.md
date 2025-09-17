# Finetuning

> [!IMPORTANT]
> Finetuning is still **very experimental** and not well tested yet!

More instructions coming soon.

Join the [Discord](https://discord.gg/ZDEYTTRxWG) for support.

Example script:

```bash
python -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path vibevoice/VibeVoice-1.5B \
    --dataset_name vibevoice/jenny_vibevoice_formatted \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir finetune_vibevoice_zac \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2.5e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --report_to wandb \
    --remove_unused_columns False \
    --bf16 True \
    --do_train \
    --gradient_clipping \
    --gradient_checkpointing False \
    --ddpm_batch_mul 4 \
    --diffusion_loss_weight 1.4 \
    --train_diffusion_head True \
    --ce_loss_weight 0.04 \
    --voice_prompt_drop_rate 0.2 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_grad_norm 0.8
```

## Notes

- Currently, only single-speaker finetuning is supported. Podcast fine-tuning is not supported yet.
- This is an unofficial finetuning implementation, it has not been validated by the original authors.
- The `voice_prompts_column_name` parameter is currently set to `audio` in the example above, which means the same audio file is used for both training data and voice prompts. This is appropriate when you don't have separate voice prompt files. However, if your dataset includes dedicated voice prompt files (short audio clips that capture the target speaker's voice characteristics), you should specify a different column name that contains these separate voice prompt files. For podcast-style training (once it is supported), the model typically uses the first utterance from each speaker within the podcast episode as the voice prompt, meaning the voice prompt is extracted from the beginning of the same audio file used for training.
- The dataset text/transcript must be in the format of "Speaker X: text", even if there is only one speaker. Example: `Speaker 1: Hello, how are you?`
- The default dataset is the Jenny (Dioco) dataset. This is a small dataset for testing purposes and each segment is only a few seconds long. The model may struggle to generate long audio with this dataset.