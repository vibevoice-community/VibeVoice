# FILE: 1main.py

import os
import time
import torch
import argparse
import json
import traceback

# 导入 transformers 和 bitsandbytes 相关库
from transformers import BitsAndBytesConfig
from transformers.utils import logging

# 导入 VibeVoice 相关模块
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

# 设置日志级别
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# ============= 全局配置 =============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
# ====================================


def run_batch_tts():
    """
    主函数，用于执行批量文本到语音转换任务。
    此脚本的命令行接口与原始的 main.py 完全兼容。
    """
    # --- 1. 参数解析 (与 main.py 的接口保持一致) ---
    parser = argparse.ArgumentParser(description="使用 VibeVoice 从一批文本任务生成语音。")
    parser.add_argument(
        "--json_input",
        type=str,
        required=True,
        help="一个JSON字符串, 格式为 '[{\"text\": \"...\", \"output_path\": \"...\"}, ...]'"
    )
    # 关键修改：将 speaker_voice 改为可选参数，并提供默认值
    parser.add_argument(
        "--speaker_voice",
        type=str,
        default="demo/voices/新闻原始2.wav",  # <-- 这里设置了默认值
        help="用于参考音色的音频文件路径。如果未提供，将使用默认语音。"
    )
    args = parser.parse_args()

    # --- 自动计算模型和参考语音的绝对路径 ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, "vibevoice/VibeVoice-4bit")

    # 处理参考语音路径：如果是相对路径，则相对于脚本位置
    speaker_voice_path = args.speaker_voice
    if not os.path.isabs(speaker_voice_path):
        speaker_voice_path = os.path.join(script_dir, speaker_voice_path)

    if not os.path.isdir(model_path):
        print(f"❌ 错误: 自动检测的模型路径不存在: '{model_path}'")
        return

    try:
        tasks = json.loads(args.json_input)
        if not isinstance(tasks, list):
            raise ValueError("JSON input 必须是一个任务列表。")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"错误: 无效的 JSON 输入. 请检查格式. {e}")
        return

    # --- 2. 设置环境和随机种子 ---
    print(f"使用设备: {DEVICE}")
    print(f"设置随机种子: {SEED}")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    # --- 3. 模型和处理器初始化 ---
    print(f"\n正在加载处理器和模型: {model_path}")
    print("这个过程可能需要一些时间，请耐心等待...")
    
    try:
        processor = VibeVoiceProcessor.from_pretrained(model_path)
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        model.eval()
        model.set_ddpm_inference_steps(num_steps=10)
        print("✅ TTS 模型已成功加载并准备就绪。")
    except Exception as e:
        print(f"❌ 错误: 模型加载失败. 详细信息: {e}")
        return

    # --- 4. 循环处理所有任务 ---
    if not os.path.exists(speaker_voice_path):
        print(f"❌ 错误: 参考语音文件未找到: {speaker_voice_path}")
        return

    total_tasks = len(tasks)
    for i, task in enumerate(tasks):
        text = task.get("text")
        output_path = task.get("output_path")
        
        if not all([text, output_path]):
            print(f"⏭️ 跳过任务 {i+1}/{total_tasks}，因为缺少 'text' 或 'output_path' 字段。")
            continue
            
        print(f"\n--- 正在处理任务 {i+1}/{total_tasks} ---")
        print(f"   ┣━ 文本内容: \"{text.strip()[:70]}...\"")
        print(f"   ┣━ 输出路径: {output_path}")
        print(f"   ┗━ 参考语音: {speaker_voice_path}")

        try:
            full_script = f"Speaker 1: {text.strip()}"
            voice_samples = [speaker_voice_path]
            
            inputs = processor(
                text=[full_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)
            
            start_time = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    cfg_scale=1.3,
                    tokenizer=processor.tokenizer,
                    generation_config={'do_sample': False},
                    is_prefill=True,
                    max_new_tokens=None,
                )
            generation_time = time.time() - start_time
            
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                processor.save_audio(outputs.speech_outputs[0], output_path=output_path)
                
                sample_rate = 24000
                audio_samples = outputs.speech_outputs[0].cpu().shape[-1]
                audio_duration = audio_samples / sample_rate
                rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
                
                print(f"   ✅ 任务 {i+1}/{total_tasks} 已完成。音频已保存到: {output_path}")
                print(f"      ┣━ 生成耗时: {generation_time:.2f} 秒, 音频时长: {audio_duration:.2f} 秒, RTF: {rtf:.2f}x")
            else:
                print(f"   ❌ 任务 {i+1}/{total_tasks} 执行失败: 模型没有生成任何音频输出。")

        except Exception as e:
            print(f"   ❌ 任务 {i+1}/{total_tasks} 执行时发生未知错误: {e}")
            traceback.print_exc()

    print("\n🎉 所有TTS任务处理完毕。")


if __name__ == "__main__":
    run_batch_tts()