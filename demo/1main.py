import os
import re
import time
import torch
from typing import List, Tuple

# 导入 BitsAndBytesConfig
from transformers import BitsAndBytesConfig

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# ============= 硬编码配置 =============
# 将 MODEL_PATH 修改为您的 Q8 模型路径
MODEL_PATH = "vibevoice/VibeVoice-4bit"
INPUT_TEXT_FILE = "demo/text_examples/1p_abs.txt"
OUTPUT_DIR = "./outputs"
SPEAKER_VOICE = "demo/voices/新闻原始2.wav"
DEVICE = "cuda"
SEED = 42
CFG_SCALE = 1.3
DISABLE_PREFILL = False
# ====================================


def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
    """解析文本脚本,提取说话人和文本"""
    lines = txt_content.strip().split('\n')
    scripts = []
    speaker_numbers = []
    
    speaker_pattern = r'^Speaker\s+(\d+):\s*(.*)$'
    current_speaker = None
    current_text = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)
            
            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
        else:
            if current_text:
                current_text += " " + line
            else:
                current_text = line
    
    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)
    
    return scripts, speaker_numbers


def main():
    print(f"使用设备: {DEVICE}")
    print(f"设置随机种子: {SEED}")
    
    torch.backends.cudnn.benchmark = True           # 让 cudnn 自动选择最快算法（适合固定输入形状）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # 设置随机种子
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # 检查输入文件
    if not os.path.exists(INPUT_TEXT_FILE):
        print(f"错误: 找不到文本文件: {INPUT_TEXT_FILE}")
        return
    
    if not os.path.exists(SPEAKER_VOICE):
        print(f"错误: 找不到语音文件: {SPEAKER_VOICE}")
        return
    
    # 读取并解析文本文件
    print(f"读取脚本: {INPUT_TEXT_FILE}")
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    
    scripts, speaker_numbers = parse_txt_script(txt_content)
    
    if not scripts:
        print("错误: 文本文件中没有找到有效的说话人脚本")
        return
    
    print(f"找到 {len(scripts)} 个说话人片段")
    for i, script in enumerate(scripts):
        print(f"  {i+1}. {script[:80]}...")
    
    # 准备完整脚本和语音样本
    full_script = ' '.join(scripts).replace("â€™", "'")
    voice_samples = [SPEAKER_VOICE]  # 所有说话人使用同一个语音样本
    
    print(f"\n加载处理器和模型: {MODEL_PATH}")
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    
    # ==================== 模型加载方式修改开始 ====================
    print("正在以8位量化模式加载模型...")

    # 1. 定义8位量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,   # 必须添加
        bnb_4bit_use_double_quant=True,          # 小提升（3~5%）
        bnb_4bit_quant_type="nf4",               # 最佳量化类型
    )

    # 2. 加载量化模型
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map="cuda",  # 使用 accelerate 自动分配设备
        attn_implementation="flash_attention_2",
    )
    # ==================== 模型加载方式修改结束 ====================
    model.eval()
   
    model.set_ddpm_inference_steps(num_steps=10)

    # 注意：量化模型通常与 torch.compile 不兼容，因此我们注释掉这一行
    # model = torch.compile(model) 
    
    print(f"语音克隆: {'禁用' if DISABLE_PREFILL else '启用'}")
    
    # 准备输入
    inputs = processor(
        text=[full_script],
        voice_samples=[voice_samples],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    # 生成音频
    start_time = time.time()
    with torch.inference_mode():
      outputs = model.generate(
          **inputs,
          # max_new_tokens=None, # generate 函数内部有默认处理
          # max_new_tokens=2048,
          cfg_scale=CFG_SCALE,
          tokenizer=processor.tokenizer,
          generation_config={'do_sample': False},
          is_prefill=not DISABLE_PREFILL,
          max_new_tokens=None,
          # verbose=True,
      )
      generation_time = time.time() - start_time
      
      print(f"生成时间: {generation_time:.2f} 秒")
      
      # 计算音频时长和指标
      if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
          sample_rate = 24000
          # .cpu() 是个好习惯，以防音频在GPU上
          audio_samples = outputs.speech_outputs[0].cpu().shape[-1]
          audio_duration = audio_samples / sample_rate
          rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
          
          print(f"音频时长: {audio_duration:.2f} 秒")
          print(f"RTF (实时因子): {rtf:.2f}x")
      else:
          print("没有生成音频输出")
          return
      
      # 计算token指标
      input_tokens = inputs['input_ids'].shape[1]
      output_tokens = outputs.sequences.shape[1]
      generated_tokens = output_tokens - input_tokens
      
      print(f"输入tokens: {input_tokens}")
      print(f"生成tokens: {generated_tokens}")
      print(f"总tokens: {output_tokens}")
      
      # 保存输出
      txt_filename = os.path.splitext(os.path.basename(INPUT_TEXT_FILE))[0]
      output_path = os.path.join(OUTPUT_DIR, f"{txt_filename}_generated_q8.wav") # 加个后缀以区分
      os.makedirs(OUTPUT_DIR, exist_ok=True)
      
      processor.save_audio(
          outputs.speech_outputs[0],
          output_path=output_path,
      )
      
      print(f"\n音频已保存到: {output_path}")
      
      # 打印摘要
      print("\n" + "="*50)
      print("生成摘要 (8位量化模式)")
      print("="*50)
      print(f"输入文件: {INPUT_TEXT_FILE}")
      print(f"输出文件: {output_path}")
      print(f"语音样本: {SPEAKER_VOICE}")
      print(f"片段数量: {len(scripts)}")
      print(f"生成时间: {generation_time:.2f} 秒")
      print(f"音频时长: {audio_duration:.2f} 秒")
      print(f"RTF: {rtf:.2f}x")
      print(f"随机种子: {SEED}")
      print("="*50)


if __name__ == "__main__":
    main()