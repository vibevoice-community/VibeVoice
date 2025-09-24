> [!IMPORTANT]
> This is a community-maintained fork of VibeVoice. Following the removal of the official VibeVoice repository, this fork serves to preserve the codebase and maintain accessibility for the community while also introducing additional functionality (such as unofficial training/fine-tuning implementations)

## 🎙️ VibeVoice: A Frontier Long Conversational Text-to-Speech Model

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://microsoft.github.io/VibeVoice)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Models-orange?logo=huggingface)](https://huggingface.co/vibevoice)
[![Technical Report](https://img.shields.io/badge/Technical-Report-red)](https://arxiv.org/pdf/2508.19205)
[![Colab](https://img.shields.io/badge/Colab-Demo-orange?logo=googlecolab)](https://colab.research.google.com/github/vibevoice-community/VibeVoice/blob/main/demo/VibeVoice_colab.ipynb)

## Community

**Join the unofficial Discord community: https://discord.gg/ZDEYTTRxWG** - share samples, ask questions, discuss fine-tuning, etc.

## Overview

VibeVoice is a novel framework designed for generating **expressive**, **long-form**, **multi-speaker** conversational audio, such as podcasts, from text. It addresses significant challenges in traditional Text-to-Speech (TTS) systems, particularly in scalability, speaker consistency, and natural turn-taking.

A core innovation of VibeVoice is its use of continuous speech tokenizers (Acoustic and Semantic) operating at an ultra-low frame rate of 7.5 Hz. These tokenizers efficiently preserve audio fidelity while significantly boosting computational efficiency for processing long sequences. VibeVoice employs a [next-token diffusion](https://arxiv.org/abs/2412.08635) framework, leveraging a Large Language Model (LLM) to understand textual context and dialogue flow, and a diffusion head to generate high-fidelity acoustic details.

The model can synthesize speech up to **90 minutes** long with up to **4 distinct speakers**, surpassing the typical 1-2 speaker limits of many prior models.

Fine-tuning is now supported, which is incredibly powerful. You can adapt VibeVoice to a new language or a new voice - [try it out](https://github.com/vibevoice-community/VibeVoice/blob/main/FINETUNING.md)

## [Examples](./EXAMPLES.md)

## Evaluation

<p align="left">
  <img src="Figures/MOS-preference.png" alt="MOS Preference Results" height="260px">
  <img src="Figures/VibeVoice.jpg" alt="VibeVoice Overview" height="250px" style="margin-right: 10px;">
</p>


## Updates

- **[2025-09-05]** Microsoft repo restored (without code) with statement about responsible AI use.
- **[2025-09-04]** Community backup created after Microsoft removed original repo and models.
- **[2025-08-26]** The [VibeVoice-7B](https://huggingface.co/vibevoice/VibeVoice-7B) model weights are open-sourced!
- **[2025-08-28]** [Colab Notebook](https://colab.research.google.com/github/microsoft-community/VibeVoice/blob/main/demo/VibeVoice_colab.ipynb) available. Only VibeVoice-1.5B is supported due to GPU memory limitations.

## Roadmap

- [x] Unofficial/community training code
- [ ] HF Transformers integration ([PR](https://github.com/huggingface/transformers/pull/40546))
- [ ] VibePod: End-to-end solution that creates podcasts from documents, webpages, or even a simple topic.

## Model Zoo

| Model | Context Length | Generation Length |  Weight |
|-------|----------------|----------|----------|
| VibeVoice-1.5B | 64K | ~90 min | [HF link](https://huggingface.co/vibevoice/VibeVoice-1.5B) |
| VibeVoice-Large| 32K | ~45 min | [HF link](https://huggingface.co/vibevoice/VibeVoice-7B) |

## Installation

```bash
git clone https://github.com/vibevoice-community/VibeVoice.git
cd VibeVoice/

uv pip install -e .
```

## Usage

### 🚨 Tips

We observed users may encounter occasional instability when synthesizing Chinese speech. We recommend:

- Using English punctuation even for Chinese text, preferably only commas and periods.
- Using the Large model variant, which is considerably more stable.
- If you found the generated voice speak too fast. Please try to chunk your text with multiple speaker turns with same speaker label.

We'd like to thank [PsiPi](https://huggingface.co/PsiPi) for sharing an interesting way for emotion control. Details can be found via [discussion #12](https://huggingface.co/microsoft/VibeVoice-1.5B/discussions/12).

**Option 1: Launch Gradio demo**

```bash
python demo/gradio_demo.py --model_path vibevoice/VibeVoice-1.5B --share
# or python demo/gradio_demo.py --model_path vibevoice/VibeVoice-7B --share
# optionally add --checkpoint_path path/to/checkpoint to load a fine-tuned adapter
# use the in-app "Disable voice cloning" setting (Advanced Settings) to skip speaker conditioning
```

**Option 2: Inference from files directly**

```bash
# We provide some LLM generated example scripts under demo/text_examples/ for demo
# 1 speaker
python demo/inference_from_file.py --model_path vibevoice/VibeVoice-7B --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice

# or more speakers
python demo/inference_from_file.py --model_path vibevoice/VibeVoice-7B --txt_path demo/text_examples/2p_music.txt --speaker_names Alice Frank

# load a fine-tuned LoRA checkpoint
python demo/inference_from_file.py --model_path vibevoice/VibeVoice-7B --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice --checkpoint_path path/to/checkpoint

# disable voice cloning (skip speech prefill)
python demo/inference_from_file.py --model_path vibevoice/VibeVoice-7B --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice --disable_prefill
```


## [Finetuning](./FINETUNING.md)

NOTE: Finetuning is still **very experimental** and not well tested yet!

## FAQ

#### Q1: Is this a pretrained model?
**A:** Yes, it's a pretrained model without any post-training or benchmark-specific optimizations. In a way, this makes VibeVoice very versatile and fun to use.

#### Q2: Randomly trigger Sounds / Music / BGM.
**A:** As you can see from our demo page, the background music or sounds are spontaneous. This means we can't directly control whether they are generated or not. The model is content-aware, and these sounds are triggered based on the input text and the chosen voice prompt.

Here are a few things we've noticed:
*   If the voice prompt you use contains background music, the generated speech is more likely to have it as well. (The Large model is quite stable and effective at this—give it a try on the demo!)
*   If the voice prompt is clean (no BGM), but the input text includes introductory words or phrases like "Welcome to," "Hello," or "However," background music might still appear.
*   Speaker voice related, using "Alice" results in random BGM than others (fixed).
*   In other scenarios, the Large model is more stable and has a lower probability of generating unexpected background music.

In fact, we intentionally decided not to denoise our training data because we think it's an interesting feature for BGM to show up at just the right moment. You can think of it as a little easter egg we left for you.

#### Q3: Text normalization?
**A:** We don't perform any text normalization during training or inference. Our philosophy is that a large language model should be able to handle complex user inputs on its own. However, due to the nature of the training data, you might still run into some corner cases.

#### Q4: Singing Capability.
**A:** Our training data **doesn't contain any music data**. The ability to sing is an emergent capability of the model (which is why it might sound off-key, even on a famous song like 'See You Again'). (The Large model is more likely to exhibit this than the 1.5B).

#### Q5: Some Chinese pronunciation errors.
**A:** The volume of Chinese data in our training set is significantly smaller than the English data. Additionally, certain special characters (e.g., Chinese quotation marks) may occasionally cause pronunciation issues.

#### Q6: Instability of cross-lingual transfer.
**A:** The model does exhibit strong cross-lingual transfer capabilities, including the preservation of accents, but its performance can be unstable. This is an emergent ability of the model that we have not specifically optimized. It's possible that a satisfactory result can be achieved through repeated sampling.

## Credits

- Thanks to [Microsoft](https://github.com/microsoft/VibeVoice) for the original VibeVoice implementation.
- Huge shoutout to [Juan Pablo Gallego](https://github.com/jpgallegoar) from [VoicePowered AI](https://www.voicepowered.ai/) for the unofficial training/fine-tuning code.
- Thanks to [PsiPi](https://huggingface.co/PsiPi) for sharing an interesting way for emotion control. Details can be found via [discussion #12](https://huggingface.co/microsoft/VibeVoice-1.5B/discussions/12).

## License

The source code and models are licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

Note: Microsoft has removed the original repo and models. This fork is based off of the MIT-licensed code from Microsoft.
