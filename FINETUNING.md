# Finetuning

**NOTE: This document is still a work in progress.**

VibeVoice finetuning is relatively straightforward and highly effective. You can finetune on either single speaker or multi-speaker data.

Finetuning can be used to train a new speaker/voice or adapt VibeVoice to a new language. LoRA finetuning yields excellent results - even on small datasets.

## Before you start

If you run into any issues during the finetuning process, feel free to join the [Discord](https://discord.gg/ZDEYTTRxWG) for support.

Before you start, you will need:

- Python, CUDA, and PyTorch installed
- A GPU with at least 16 GB of VRAM
- A dataset of audio recordings (preferably with at least 1 hour of audio, optimally 5+ hours)

## Getting started

Clone the VibeVoice repository:

```bash
git clone https://github.com/vibevoice-community/VibeVoice.git
cd VibeVoice/
```

Install the dependencies:

```bash
pip install uv
uv pip install -e .
```

## Finetune the model

**IMPORTANT: For single-speaker datasets:** When working with data from only one speaker, you'll need to choose whether to maintain voice cloning functionality in your finetuned model. For single-speaker training, consider setting `voice_prompt_drop_rate` to `1.0`, which disables voice cloning entirely. Many users have found that removing voice cloning constraints allows the model to be more expressive and natural, as it eliminates unnecessary limitations on the model's output capabilities.

Run the following command to finetune the model:

```bash
python -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path vibevoice/VibeVoice-1.5B \
    --dataset_name [PATH_TO_YOUR_DATASET] \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir [PATH_TO_SAVE_YOUR_FINETUNED_MODEL] \
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

## Credits

- Thanks to [VoicePowered AI](https://github.com/voicepowered-ai/VibeVoice-finetuning) for the original finetuning implementation.
- Thanks to members of the community for discovering that setting `voice_prompt_drop_rate` to `1.0` can lead to more natural speech generation.