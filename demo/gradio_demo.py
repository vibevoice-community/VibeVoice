import os
import re
import time
import torch
from typing import List, Tuple

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

# ============= 配置 =============
MODEL_PATH = "vibevoice/VibeVoice-1.5b"
INPUT_TEXT_FILE = "demo/text_examples/1p_abs.txt"
OUTPUT_DIR = "./outputs"
SPEAKER_VOICE = "demo/voices/新闻原始2.wav"
DEVICE = "cuda"
SEED = 42
CFG_SCALE = 1.3
COMPILE_MODEL = True  # torch.compile加速
# ================================


def parse_txt_script(txt_content: str) -> List[str]:
    """解析文本脚本"""
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
            current_text += " " + line if current_text else line
    
    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
    
    return scripts


def main():
    # 性能优化设置
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # 读取文本
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        scripts = parse_txt_script(f.read())
    
    full_script = '\n'.join(scripts).replace("â€™", "'")
    print(f"脚本片段数: {len(scripts)}")
    
    # 加载模型
    print("加载模型...")
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=30)
    
    # 编译模型加速
    if COMPILE_MODEL and hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        print("编译模型...")
        model.model.language_model = torch.compile(
            model.model.language_model, 
            mode="reduce-overhead"
        )
    
    # 准备输入
    inputs = processor(
        text=[full_script],
        voice_samples=[[SPEAKER_VOICE]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(DEVICE, non_blocking=True)
    
    torch.cuda.synchronize()
    
    # 生成
    print("生成中...")
    start_time = time.time()
    
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=CFG_SCALE,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False},
            verbose=True,
            is_prefill=True,
        )
    
    torch.cuda.synchronize()
    generation_time = time.time() - start_time
    
    # 保存
    output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(INPUT_TEXT_FILE))[0]}_generated.wav")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processor.save_audio(outputs.speech_outputs[0], output_path=output_path)
    
    # 统计
    audio_duration = outputs.speech_outputs[0].shape[-1] / 24000
    rtf = generation_time / audio_duration
    
    print(f"\n生成完成: {output_path}")
    print(f"生成时间: {generation_time:.2f}s | 音频: {audio_duration:.2f}s | RTF: {rtf:.2f}x")


if __name__ == "__main__":
    main()