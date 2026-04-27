"""
VibeVoice WebUI - 语音模型调试界面
基于 Gradio 构建的交互式 TTS 调试工具
"""

import os
import re
import time
import tempfile
import traceback
from typing import List, Tuple, Optional

import torch
import numpy as np
import gradio as gr

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# ============= 全局状态 =============
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SAMPLE_RATE = 24000
VOICES_DIR = os.path.join(SCRIPT_DIR, "demo", "voices")
TEXT_EXAMPLES_DIR = os.path.join(SCRIPT_DIR, "demo", "text_examples")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局模型和处理器引用
_model = None
_processor = None
_current_model_path = None


def get_available_models() -> list:
    """扫描可用的模型目录"""
    models = []
    vv_dir = os.path.join(SCRIPT_DIR, "vibevoice")
    if os.path.isdir(vv_dir):
        for name in sorted(os.listdir(vv_dir)):
            full = os.path.join(vv_dir, name)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, "config.json")):
                models.append(f"vibevoice/{name}")
    return models if models else ["vibevoice/VibeVoice-1.5B"]


def get_available_voices() -> list:
    """扫描可用的参考语音文件"""
    voices = []
    if os.path.isdir(VOICES_DIR):
        for f in sorted(os.listdir(VOICES_DIR)):
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                voices.append(os.path.join(VOICES_DIR, f))
    return voices


def get_text_examples() -> list:
    """获取示例文本文件"""
    examples = []
    if os.path.isdir(TEXT_EXAMPLES_DIR):
        for f in sorted(os.listdir(TEXT_EXAMPLES_DIR)):
            if f.endswith(".txt"):
                examples.append(os.path.join(TEXT_EXAMPLES_DIR, f))
    return examples


def parse_txt_script(txt_content: str) -> List[str]:
    """解析文本脚本，支持 Speaker N: 格式和纯文本"""
    lines = txt_content.strip().split('\n')
    scripts = []
    current_speaker = None
    current_text = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^Speaker\s+(\d+):\s*(.*)$', line, re.IGNORECASE)
        if match:
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
        else:
            if current_speaker:
                current_text += " " + line if current_text else line
            else:
                # 没有 Speaker 前缀的纯文本，自动添加 Speaker 1
                current_speaker = "1"
                current_text = line

    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")

    return scripts


def load_model(model_name: str, use_4bit: bool, attn_impl: str, ddpm_steps: int, use_compile: bool, progress=gr.Progress()):
    """加载或切换模型"""
    global _model, _processor, _current_model_path

    model_path = os.path.join(SCRIPT_DIR, model_name)
    if not os.path.isdir(model_path):
        return f"❌ 模型路径不存在: {model_path}"

    # 如果已加载同一模型，跳过
    cache_key = f"{model_name}|4bit={use_4bit}|attn={attn_impl}"
    if _model is not None and _current_model_path == cache_key:
        # 只更新 DDPM 步数
        _model.set_ddpm_inference_steps(num_steps=ddpm_steps)
        return f"✅ 模型已就绪 (更新 DDPM 步数={ddpm_steps})"

    progress(0.1, desc="释放旧模型...")
    # 释放旧模型
    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        progress(0.2, desc="加载 Processor...")
        _processor = VibeVoiceProcessor.from_pretrained(model_path)

        progress(0.4, desc="加载模型权重...")
        load_kwargs = {
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "attn_implementation": attn_impl,
        }

        # 检测是否为预量化的 4bit 模型目录
        is_prequantized_4bit = "4bit" in model_name.lower() or os.path.exists(
            os.path.join(model_path, "quantization_config.json")
        )

        if use_4bit or is_prequantized_4bit:
            if not torch.cuda.is_available():
                return "❌ 4bit 量化需要 CUDA GPU"
            # 检测 bitsandbytes 是否安装
            try:
                import bitsandbytes  # noqa: F401
                _bnb_available = True
            except ImportError:
                _bnb_available = False

            if not _bnb_available:
                return (
                    "❌ 4bit 量化需要安装 bitsandbytes 库，但当前环境未安装。\n\n"
                    "请在终端运行以下命令安装:\n"
                    "  pip install bitsandbytes\n\n"
                    "Windows 用户可尝试:\n"
                    "  pip install bitsandbytes-windows\n"
                    "  或 pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl\n\n"
                    "如果不想安装，请取消勾选「4bit 量化」，选择非 4bit 模型（如 VibeVoice-1.5B）以 bfloat16 加载。"
                )

            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["torch_dtype"] = torch.bfloat16
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        _model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path, **load_kwargs
        )
        _model.eval()

        progress(0.8, desc="设置推理参数...")
        _model.set_ddpm_inference_steps(num_steps=ddpm_steps)

        if use_compile and hasattr(_model, 'model') and hasattr(_model.model, 'language_model'):
            progress(0.9, desc="编译模型 (torch.compile)...")
            _model.model.language_model = torch.compile(
                _model.model.language_model,
                mode="reduce-overhead"
            )

        _current_model_path = cache_key

        # VRAM 统计
        vram_info = ""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            vram_info = f"\n📊 显存: 已分配 {allocated:.2f} GB / 已保留 {reserved:.2f} GB"

        return f"✅ 模型加载成功: {model_name}\n   4bit量化: {'是' if use_4bit else '否'} | 注意力: {attn_impl} | DDPM步数: {ddpm_steps} | 编译: {'是' if use_compile else '否'}{vram_info}"

    except Exception as e:
        _model = None
        _processor = None
        _current_model_path = None
        error_msg = traceback.format_exc()
        return f"❌ 模型加载失败:\n{error_msg}"


def load_example_text(example_path: str) -> str:
    """加载示例文本文件"""
    if not example_path or not os.path.exists(example_path):
        return ""
    with open(example_path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_speech(
    text: str,
    voice_preset: str,
    voice_upload: Optional[str],
    cfg_scale: float,
    ddpm_steps: int,
    seed: int,
    enable_prefill: bool,
    do_sample: bool,
    progress=gr.Progress(),
):
    """核心 TTS 推理函数"""
    global _model, _processor

    if _model is None or _processor is None:
        return None, "❌ 请先加载模型！"

    if not text or not text.strip():
        return None, "❌ 请输入要合成的文本！"

    # 确定参考语音路径
    voice_path = None
    if voice_upload and os.path.exists(voice_upload):
        voice_path = voice_upload
    elif voice_preset and os.path.exists(voice_preset):
        voice_path = voice_preset

    if not voice_path:
        return None, "❌ 请选择或上传参考语音文件！"

    try:
        # 更新 DDPM 步数
        _model.set_ddpm_inference_steps(num_steps=ddpm_steps)

        # 设置随机种子
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        progress(0.1, desc="解析文本...")
        # 解析文本
        scripts = parse_txt_script(text)
        if not scripts:
            return None, "❌ 无法解析文本，请检查格式"

        full_script = '\n'.join(scripts).replace("\u00e2\u0080\u0099", "'")
        voice_samples = [voice_path]

        progress(0.2, desc="处理输入...")
        inputs = _processor(
            text=[full_script],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device, non_blocking=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        progress(0.3, desc="生成语音中...")
        start_time = time.time()

        gen_config = {'do_sample': do_sample}

        with torch.inference_mode():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = _model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=_processor.tokenizer,
                        generation_config=gen_config,
                        verbose=True,
                        is_prefill=enable_prefill,
                    )
            else:
                outputs = _model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=_processor.tokenizer,
                    generation_config=gen_config,
                    verbose=True,
                    is_prefill=enable_prefill,
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_time = time.time() - start_time

        progress(0.9, desc="保存音频...")

        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            return None, "❌ 模型没有生成任何音频输出"

        audio_tensor = outputs.speech_outputs[0]
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.cpu().float().numpy()
        else:
            audio_np = np.array(audio_tensor, dtype=np.float32)

        # 确保音频是一维的
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        audio_duration = len(audio_np) / SAMPLE_RATE
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')

        # 保存到文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"webui_{timestamp}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        _processor.save_audio(outputs.speech_outputs[0], output_path=output_path)

        # Token 统计
        input_tokens = inputs['input_ids'].shape[1]
        output_tokens = outputs.sequences.shape[1]
        generated_tokens = output_tokens - input_tokens

        # VRAM
        vram_info = ""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3
            vram_info = f"\n📊 显存: 当前 {allocated:.2f} GB | 峰值 {peak:.2f} GB"

        stats = (
            f"✅ 生成成功！\n"
            f"⏱️ 生成耗时: {generation_time:.2f} 秒\n"
            f"🎵 音频时长: {audio_duration:.2f} 秒\n"
            f"⚡ RTF (实时因子): {rtf:.3f}x\n"
            f"🔢 输入 Tokens: {input_tokens} | 生成 Tokens: {generated_tokens}\n"
            f"🎯 参数: CFG={cfg_scale}, DDPM步数={ddpm_steps}, Seed={seed}\n"
            f"🗣️ 参考语音: {os.path.basename(voice_path)}\n"
            f"📄 解析片段数: {len(scripts)}\n"
            f"💾 已保存: {output_path}"
            f"{vram_info}"
        )

        return (SAMPLE_RATE, audio_np), stats

    except Exception as e:
        error_msg = traceback.format_exc()
        return None, f"❌ 生成失败:\n{error_msg}"


def get_gpu_info():
    """获取 GPU 信息"""
    if not torch.cuda.is_available():
        return "⚠️ CUDA 不可用，将使用 CPU 推理"

    info_lines = [f"🖥️ CUDA 版本: {torch.version.cuda}"]
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        info_lines.append(
            f"   GPU {i}: {props.name} ({total_mem:.1f} GB 总显存, {allocated:.2f} GB 已用)"
        )
    return "\n".join(info_lines)


def refresh_vram():
    """刷新显存信息"""
    if not torch.cuda.is_available():
        return "CUDA 不可用"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return f"已分配: {allocated:.2f} GB | 已保留: {reserved:.2f} GB | 峰值: {peak:.2f} GB"


def clear_vram_cache():
    """清理显存缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return "✅ 显存缓存已清理"
    return "CUDA 不可用"


def unload_model():
    """卸载模型释放显存"""
    global _model, _processor, _current_model_path
    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    _current_model_path = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return "✅ 模型已卸载，显存已释放"


# ============= 构建 Gradio UI =============

def build_ui():
    available_models = get_available_models()
    available_voices = get_available_voices()
    voice_names = [os.path.basename(v) for v in available_voices]
    voice_map = dict(zip(voice_names, available_voices))
    text_examples = get_text_examples()
    text_example_names = [os.path.basename(t) for t in text_examples]
    text_example_map = dict(zip(text_example_names, text_examples))

    with gr.Blocks(
        title="VibeVoice WebUI",
        theme=gr.themes.Soft(),
        css="""
        .status-box { font-family: monospace; font-size: 13px; }
        .header-text { text-align: center; margin-bottom: 10px; }
        """
    ) as demo:

        gr.Markdown(
            """
            # 🎙️ VibeVoice WebUI
            ### 语音合成模型调试界面
            支持多模型切换、参数调节、语音克隆等功能
            """,
            elem_classes=["header-text"]
        )

        with gr.Row():
            gpu_info = gr.Textbox(
                label="系统信息",
                value=get_gpu_info(),
                interactive=False,
                lines=2,
            )

        with gr.Tabs():
            # ========== TAB 1: 模型管理 ==========
            with gr.Tab("🔧 模型管理"):
                with gr.Row():
                    with gr.Column(scale=2):
                        model_selector = gr.Dropdown(
                            label="选择模型",
                            choices=available_models,
                            value=available_models[0] if available_models else None,
                            interactive=True,
                        )
                        with gr.Row():
                            use_4bit = gr.Checkbox(label="4bit 量化", value=False, info="使用 BitsAndBytes 4bit 量化减少显存")
                            use_compile = gr.Checkbox(label="torch.compile", value=False, info="编译加速（首次推理较慢）")

                        attn_impl = gr.Dropdown(
                            label="注意力实现",
                            choices=["sdpa", "flash_attention_2", "eager"],
                            value="sdpa",
                            info="flash_attention_2 需要安装 flash-attn",
                        )
                        ddpm_steps_load = gr.Slider(
                            label="DDPM 推理步数",
                            minimum=1, maximum=100, step=1, value=30,
                            info="步数越多质量越高，但速度越慢",
                        )

                    with gr.Column(scale=1):
                        load_btn = gr.Button("🚀 加载模型", variant="primary", size="lg")
                        unload_btn = gr.Button("🗑️ 卸载模型", variant="stop")
                        clear_cache_btn = gr.Button("🧹 清理显存缓存")
                        refresh_vram_btn = gr.Button("🔄 刷新显存")
                        vram_display = gr.Textbox(label="显存状态", interactive=False, lines=1)

                model_status = gr.Textbox(
                    label="模型状态",
                    interactive=False,
                    lines=4,
                    elem_classes=["status-box"],
                )

                load_btn.click(
                    fn=load_model,
                    inputs=[model_selector, use_4bit, attn_impl, ddpm_steps_load, use_compile],
                    outputs=[model_status],
                )
                unload_btn.click(fn=unload_model, outputs=[model_status])
                clear_cache_btn.click(fn=clear_vram_cache, outputs=[vram_display])
                refresh_vram_btn.click(fn=refresh_vram, outputs=[vram_display])

            # ========== TAB 2: 语音合成 ==========
            with gr.Tab("🎵 语音合成"):
                with gr.Row():
                    # --- 左列：输入 ---
                    with gr.Column(scale=3):
                        gr.Markdown("### 📝 输入文本")
                        with gr.Row():
                            example_selector = gr.Dropdown(
                                label="加载示例文本",
                                choices=[""] + text_example_names,
                                value="",
                                interactive=True,
                                scale=3,
                            )
                            load_example_btn = gr.Button("📂 加载", scale=1)

                        text_input = gr.Textbox(
                            label="合成文本",
                            placeholder="输入要合成的文本...\n支持格式:\n  Speaker 1: 你好世界\n  Speaker 2: Hello world\n或直接输入纯文本（自动作为 Speaker 1）",
                            lines=8,
                            max_lines=20,
                        )

                        gr.Markdown("### 🗣️ 参考语音")
                        with gr.Row():
                            voice_selector = gr.Dropdown(
                                label="预设语音",
                                choices=voice_names,
                                value=voice_names[0] if voice_names else None,
                                interactive=True,
                                scale=2,
                            )
                            voice_upload = gr.Audio(
                                label="上传语音",
                                type="filepath",
                                scale=2,
                            )

                        # 试听参考语音
                        voice_preview = gr.Audio(
                            label="参考语音预览",
                            interactive=False,
                            type="filepath",
                        )

                        def preview_voice(name):
                            if name and name in voice_map:
                                return voice_map[name]
                            return None
                        voice_selector.change(fn=preview_voice, inputs=[voice_selector], outputs=[voice_preview])

                    # --- 右列：参数 + 输出 ---
                    with gr.Column(scale=2):
                        gr.Markdown("### ⚙️ 生成参数")
                        cfg_scale = gr.Slider(
                            label="CFG Scale (引导强度)",
                            minimum=0.1, maximum=5.0, step=0.1, value=1.3,
                            info="越高越忠实于文本，但可能影响自然度",
                        )
                        ddpm_steps = gr.Slider(
                            label="DDPM 推理步数",
                            minimum=1, maximum=100, step=1, value=30,
                            info="影响音频质量与速度的权衡",
                        )
                        seed = gr.Number(
                            label="随机种子",
                            value=42,
                            precision=0,
                            info="相同种子 + 相同参数 = 可复现结果",
                        )
                        with gr.Row():
                            enable_prefill = gr.Checkbox(
                                label="启用语音克隆 (Prefill)",
                                value=True,
                                info="使用参考语音的音色",
                            )
                            do_sample = gr.Checkbox(
                                label="采样模式",
                                value=False,
                                info="启用随机采样（关闭=贪婪解码）",
                            )

                        generate_btn = gr.Button("🎤 生成语音", variant="primary", size="lg")

                        gr.Markdown("### 🎧 输出")
                        audio_output = gr.Audio(
                            label="生成结果",
                            type="numpy",
                            interactive=False,
                        )
                        gen_stats = gr.Textbox(
                            label="生成统计",
                            interactive=False,
                            lines=10,
                            elem_classes=["status-box"],
                        )

                def on_load_example(name):
                    if name and name in text_example_map:
                        return load_example_text(text_example_map[name])
                    return ""

                load_example_btn.click(
                    fn=on_load_example,
                    inputs=[example_selector],
                    outputs=[text_input],
                )

                def on_generate(text, voice_name, voice_up, cfg, steps, s, prefill, sample):
                    vp = voice_map.get(voice_name, "") if voice_name else ""
                    return generate_speech(text, vp, voice_up, cfg, steps, int(s), prefill, sample)

                generate_btn.click(
                    fn=on_generate,
                    inputs=[text_input, voice_selector, voice_upload, cfg_scale, ddpm_steps, seed, enable_prefill, do_sample],
                    outputs=[audio_output, gen_stats],
                )

            # ========== TAB 3: 批量生成 ==========
            with gr.Tab("📦 批量生成"):
                gr.Markdown("### 批量文本到语音转换\n从文本文件批量生成语音")
                with gr.Row():
                    with gr.Column():
                        batch_files = gr.File(
                            label="上传文本文件 (.txt)",
                            file_types=[".txt"],
                            file_count="multiple",
                        )
                        batch_voice = gr.Dropdown(
                            label="参考语音",
                            choices=voice_names,
                            value=voice_names[0] if voice_names else None,
                        )
                        with gr.Row():
                            batch_cfg = gr.Slider(label="CFG", minimum=0.1, maximum=5.0, step=0.1, value=1.3)
                            batch_steps = gr.Slider(label="DDPM 步数", minimum=1, maximum=100, step=1, value=30)
                            batch_seed = gr.Number(label="Seed", value=42, precision=0)
                        batch_btn = gr.Button("🚀 开始批量生成", variant="primary")
                    with gr.Column():
                        batch_log = gr.Textbox(label="批量生成日志", lines=15, interactive=False, elem_classes=["status-box"])

                def batch_generate(files, voice_name, cfg, steps, s, progress=gr.Progress()):
                    if _model is None or _processor is None:
                        return "❌ 请先在「模型管理」页加载模型"
                    if not files:
                        return "❌ 请上传文本文件"

                    vp = voice_map.get(voice_name, "") if voice_name else ""
                    if not vp:
                        return "❌ 请选择参考语音"

                    logs = []
                    total = len(files)
                    for idx, f in enumerate(files):
                        fname = os.path.basename(f.name) if hasattr(f, 'name') else f
                        fpath = f.name if hasattr(f, 'name') else f
                        progress((idx) / total, desc=f"处理 {fname}...")
                        logs.append(f"\n--- [{idx+1}/{total}] {fname} ---")

                        try:
                            with open(fpath, 'r', encoding='utf-8') as tf:
                                text = tf.read()
                            audio, stats = generate_speech(text, vp, None, cfg, steps, int(s), True, False)
                            logs.append(stats)
                        except Exception as e:
                            logs.append(f"❌ 失败: {e}")

                    logs.append(f"\n{'='*40}\n🎉 批量处理完成 ({total} 个文件)")
                    return "\n".join(logs)

                batch_btn.click(
                    fn=batch_generate,
                    inputs=[batch_files, batch_voice, batch_cfg, batch_steps, batch_seed],
                    outputs=[batch_log],
                )

            # ========== TAB 4: 参数对比 ==========
            with gr.Tab("🔬 参数对比"):
                gr.Markdown("### A/B 参数对比测试\n使用相同文本，对比不同参数下的生成效果")
                with gr.Row():
                    compare_text = gr.Textbox(
                        label="对比文本",
                        value="Speaker 1: 你好，这是一段用于参数对比的测试文本。",
                        lines=3,
                    )
                    compare_voice = gr.Dropdown(
                        label="参考语音",
                        choices=voice_names,
                        value=voice_names[0] if voice_names else None,
                    )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🅰️ 设置 A")
                        a_cfg = gr.Slider(label="CFG", minimum=0.1, maximum=5.0, step=0.1, value=1.0)
                        a_steps = gr.Slider(label="DDPM 步数", minimum=1, maximum=100, step=1, value=10)
                        a_seed = gr.Number(label="Seed", value=42, precision=0)
                        a_prefill = gr.Checkbox(label="Prefill", value=True)
                        a_audio = gr.Audio(label="结果 A", type="numpy", interactive=False)
                        a_stats = gr.Textbox(label="统计 A", lines=6, interactive=False, elem_classes=["status-box"])
                    with gr.Column():
                        gr.Markdown("#### 🅱️ 设置 B")
                        b_cfg = gr.Slider(label="CFG", minimum=0.1, maximum=5.0, step=0.1, value=1.5)
                        b_steps = gr.Slider(label="DDPM 步数", minimum=1, maximum=100, step=1, value=30)
                        b_seed = gr.Number(label="Seed", value=42, precision=0)
                        b_prefill = gr.Checkbox(label="Prefill", value=True)
                        b_audio = gr.Audio(label="结果 B", type="numpy", interactive=False)
                        b_stats = gr.Textbox(label="统计 B", lines=6, interactive=False, elem_classes=["status-box"])

                compare_btn = gr.Button("⚡ 开始对比生成", variant="primary")

                def run_compare(text, voice_name, ac, ast, ase, ap, bc, bst, bse, bp):
                    vp = voice_map.get(voice_name, "") if voice_name else ""
                    a_out = generate_speech(text, vp, None, ac, ast, int(ase), ap, False)
                    b_out = generate_speech(text, vp, None, bc, bst, int(bse), bp, False)
                    return a_out[0], a_out[1], b_out[0], b_out[1]

                compare_btn.click(
                    fn=run_compare,
                    inputs=[compare_text, compare_voice, a_cfg, a_steps, a_seed, a_prefill, b_cfg, b_steps, b_seed, b_prefill],
                    outputs=[a_audio, a_stats, b_audio, b_stats],
                )

            # ========== TAB 5: 系统监控 ==========
            with gr.Tab("📈 系统监控"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### GPU 状态")
                        sys_gpu = gr.Textbox(label="GPU 信息", value=get_gpu_info(), lines=3, interactive=False)
                        sys_vram = gr.Textbox(label="显存状态", lines=1, interactive=False)
                        with gr.Row():
                            sys_refresh = gr.Button("🔄 刷新")
                            sys_clear = gr.Button("🧹 清理缓存")
                            sys_unload = gr.Button("🗑️ 卸载模型", variant="stop")

                        sys_refresh.click(fn=lambda: (get_gpu_info(), refresh_vram()), outputs=[sys_gpu, sys_vram])
                        sys_clear.click(fn=clear_vram_cache, outputs=[sys_vram])
                        sys_unload.click(fn=unload_model, outputs=[sys_vram])

                    with gr.Column():
                        gr.Markdown("### 模型信息")
                        def get_model_info():
                            if _model is None:
                                return "未加载模型"
                            info = [f"当前模型: {_current_model_path}"]
                            try:
                                total_params = sum(p.numel() for p in _model.parameters())
                                trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
                                info.append(f"总参数量: {total_params/1e9:.2f}B ({total_params:,})")
                                info.append(f"可训练参数: {trainable:,}")
                                dtypes = set(str(p.dtype) for p in _model.parameters())
                                info.append(f"数据类型: {', '.join(dtypes)}")
                            except:
                                pass
                            return "\n".join(info)

                        model_info = gr.Textbox(label="模型详情", lines=5, interactive=False)
                        model_info_btn = gr.Button("🔄 获取模型信息")
                        model_info_btn.click(fn=get_model_info, outputs=[model_info])

                        gr.Markdown("### 输出目录")
                        def list_outputs():
                            if not os.path.isdir(OUTPUT_DIR):
                                return "输出目录为空"
                            files = sorted(
                                [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')],
                                reverse=True
                            )[:20]
                            if not files:
                                return "暂无输出文件"
                            return "\n".join(f"  📄 {f}" for f in files)

                        output_list = gr.Textbox(label="最近输出 (最多20个)", lines=8, interactive=False)
                        list_outputs_btn = gr.Button("🔄 刷新列表")
                        list_outputs_btn.click(fn=list_outputs, outputs=[output_list])

        gr.Markdown(
            """
            ---
            <center>VibeVoice WebUI | 基于 Gradio 构建 | 输出保存在 <code>outputs/</code> 目录</center>
            """,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=False,
        inbrowser=True,
    )
