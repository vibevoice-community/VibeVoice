"""
Inference script for VibeVoice-7B with trained Norwegian LoRA adapters
"""
import torch
from transformers import AutoProcessor
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from peft import PeftModel
import soundfile as sf
import os

# Configuration
base_model_id = "vibevoice/VibeVoice-7B"
lora_path = "/home/me/VibeVoice/heiertech/vibevoice-7b-nob-qlora-stage1-bayesopt/lora"
output_dir = "/home/me/VibeVoice/inference_outputs"
os.makedirs(output_dir, exist_ok=True)

# Norwegian text examples
examples = [
    "Hei, hvordan har du det?",
    "Dette er en test av den norske talesyntesemodellen.",
    "Været i dag er veldig fint.",
    "Jeg liker å spise pizza.",
    "God morgen! Håper du har en fin dag.",
]

print("Loading base model...")
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = VibeVoiceForConditionalGeneration.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    device_map={'': 0},  # Load everything on GPU 0
    torch_dtype=torch.bfloat16,
)

print("Loading LoRA adapters...")
# Add dummy method to avoid PEFT error
if not hasattr(model, 'prepare_inputs_for_generation'):
    def _dummy_prepare_inputs(input_ids, **kwargs):
        return {"input_ids": input_ids}
    model.prepare_inputs_for_generation = _dummy_prepare_inputs

model = PeftModel.from_pretrained(model, lora_path)
model.eval()

print("Loading processor...")
processor = AutoProcessor.from_pretrained(base_model_id)

print("\nGenerating speech examples...\n")
for idx, text in enumerate(examples, 1):
    print(f"{idx}. Processing: '{text}'")
    
    try:
        # Prepare inputs
        inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True,
        )
        
        # Move to device
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode audio
        audio = processor.decode_audio(outputs[0])
        
        # Save
        output_path = os.path.join(output_dir, f"example_{idx}.wav")
        sf.write(output_path, audio, samplerate=24000)
        print(f"   ✓ Saved to: {output_path}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        continue

print(f"\nDone! Generated {len(examples)} examples in {output_dir}")
