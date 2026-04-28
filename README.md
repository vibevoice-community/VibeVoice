# VibeVoice Local Notes

This repo now has a Windows-friendly WebUI startup path for the newer flash-attn test environment.

## Recommended setup on this machine

- OS: Windows 10
- GPU: RTX 5060 Ti
- CUDA runtime path used by the launchers: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
- Fast WebUI environment: `.\.venv311fa`

## Start the new WebUI

PowerShell:

```powershell
.\start_webui_flashattn.ps1
```

Batch:

```bat
start_webui_flashattn.bat
```

Both launchers do the following:

- switch to the repo root
- set `CUDA_HOME` and `CUDA_PATH` to CUDA 12.8 when it exists
- prepend the CUDA `bin` directory to `PATH`
- run `webui.py` with `.\.venv311fa\Scripts\python.exe`

## WebUI recommendations

- Model: `vibevoice/VibeVoice-1.5B`
- Attention backend: `auto`
- DDPM steps: `8` or `10`
- `4bit quantization`: off unless you specifically want low-VRAM testing
- `torch.compile`: off by default on Windows

`auto` will try to use `flash_attention_2` when available in the environment, and fall back to `sdpa` otherwise.

## CLI example

```powershell
$env:CUDA_HOME='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;' + $env:PATH
.\.venv311fa\Scripts\python.exe .\main.py --model_path vibevoice/VibeVoice-1.5B --attn_impl flash_attention_2 --ddpm_steps 8
```

## Benchmark reports

- `outputs/benchmark_15b_flashattn_py311.json`
- `outputs/benchmark_15b_flashattn_py311_warm.json`
- `outputs/benchmark_15b_sdpa_py311fa_warm.json`

## Notes

- The first `flash_attention_2` run can be much slower because Triton may compile kernels on first use.
- After warmup, `flash_attention_2` was slightly faster than `sdpa` on this machine.
- Output audio files are saved to `outputs/`.

python .\mainsteam.py --model_path microsoft/VibeVoice-Realtime-0.5B --txt_path demo/text_examples/1p_vibevoice.txt --speaker_name Carter
