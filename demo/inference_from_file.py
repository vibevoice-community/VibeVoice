import argparse
import os
import re
import traceback
from typing import List, Tuple, Union, Dict, Any, Optional
import time
import torch
import hashlib
import numpy as np

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from transformers import BitsAndBytesConfig
from transformers.utils import logging
import warnings
import os

# Suppress all transformers and model loading verbosity
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LoRA support imports
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PeftModel = None
    PEFT_AVAILABLE = False

logger = logging.get_logger(__name__)
logger.setLevel(logging.WARNING)


class VoiceMapper:
    """Maps speaker names to voice file paths or creates synthetic voices"""
    
    def __init__(self):
        self.setup_voice_presets()
        new_dict = {}
        for name, path in self.voice_presets.items():
            if '_' in name:
                name = name.split('_')[0]
            if '-' in name:
                name = name.split('-')[-1]
            new_dict[name] = path
        self.voice_presets.update(new_dict)

    def setup_voice_presets(self):
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        self.voice_presets = {}
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith('.wav') and os.path.isfile(os.path.join(voices_dir, f))]
        
        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        if self.available_voices:
            print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
            print(f"Available voices: {', '.join(self.available_voices.keys())}")
        else:
            print(f"No voice files found in {voices_dir} - will use synthetic voices")

    def _create_synthetic_voice_sample(self, speaker_idx: int, seed: int = 42) -> np.ndarray:
        """Create synthetic voice sample for a specific speaker"""
        # Set numpy seed for reproducible synthetic voices
        np.random.seed(seed + speaker_idx)  # Different seed for each speaker but deterministic
        
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples, False)
        
        # Create realistic voice-like characteristics for each speaker
        # Use different base frequencies for different speaker types
        base_frequencies = [120, 180, 140, 200]  # Mix of male/female-like frequencies
        base_freq = base_frequencies[speaker_idx % len(base_frequencies)]
        
        # Create vowel-like formants (like "ah" sound) - unique per speaker
        formant1 = 800 + speaker_idx * 100  # First formant
        formant2 = 1200 + speaker_idx * 150  # Second formant
        
        # Generate more voice-like waveform
        voice_sample = (
            # Fundamental with harmonics (voice-like)
            0.6 * np.sin(2 * np.pi * base_freq * t) +
            0.25 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.15 * np.sin(2 * np.pi * base_freq * 3 * t) +
            
            # Formant resonances (vowel-like characteristics)
            0.1 * np.sin(2 * np.pi * formant1 * t) * np.exp(-t * 2) +
            0.05 * np.sin(2 * np.pi * formant2 * t) * np.exp(-t * 3) +
            
            # Natural breath noise (reduced) - now deterministic
            0.02 * np.random.normal(0, 1, len(t))
        )
        
        # Add natural envelope (like human speech pattern)
        # Quick attack, slower decay with slight vibrato (unique per speaker)
        vibrato_freq = 4 + speaker_idx * 0.3  # Slightly different vibrato per speaker
        envelope = (np.exp(-t * 0.3) * (1 + 0.1 * np.sin(2 * np.pi * vibrato_freq * t)))
        voice_sample *= envelope * 0.08  # Lower volume
        
        return voice_sample.astype(np.float32)

    def get_voice_sample(self, speaker_name: str, speaker_idx: int = 0, seed: int = 42) -> Union[str, np.ndarray]:
        """Get voice sample - tries to find file first, falls back to synthetic voice"""
        # Try exact match
        if speaker_name in self.voice_presets:
            print(f"Found voice file for '{speaker_name}': {os.path.basename(self.voice_presets[speaker_name])}")
            return self.voice_presets[speaker_name]
        
        # Try partial match
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                print(f"Found similar voice file for '{speaker_name}': {os.path.basename(path)}")
                return path
        
        # No match found - use synthetic voice (no more default file fallback)
        print(f"No voice file found for '{speaker_name}' - generating synthetic voice (Speaker {speaker_idx + 1})")
        return self._create_synthetic_voice_sample(speaker_idx, seed)

    # Backward compatibility
    def get_voice_path(self, speaker_name: str) -> str:
        """Legacy method - returns file path or creates synthetic voice"""
        result = self.get_voice_sample(speaker_name, 0)
        if isinstance(result, np.ndarray):
            # For synthetic voices, we need to return something - create a temp identifier
            return f"synthetic_voice_0"
        return result


def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
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


def _load_state_dict_into(module, folder):
    """Load state dict into module from safetensors or pytorch_model.bin"""
    if module is None or not isinstance(folder, str):
        return False
    
    # Try safetensors first
    try:
        import safetensors.torch as st
        p = os.path.join(folder, 'model.safetensors')
        if os.path.exists(p):
            try:
                sd = st.load_file(p)
                missing_unexpected = module.load_state_dict(sd, strict=False)
                missing_keys = getattr(missing_unexpected, 'missing_keys', [])
                unexpected_keys = getattr(missing_unexpected, 'unexpected_keys', [])
                print(f"Loaded safetensors from {p}; missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
                return True
            except Exception as e:
                print(f"Failed loading safetensors from {p}: {e}")
    except ImportError:
        st = None
    
    # HF PyTorch format fallback
    p = os.path.join(folder, 'pytorch_model.bin')
    if os.path.exists(p):
        try:
            sd = torch.load(p, map_location='cpu')
            missing_unexpected = module.load_state_dict(sd, strict=False)
            missing_keys = getattr(missing_unexpected, 'missing_keys', [])
            unexpected_keys = getattr(missing_unexpected, 'unexpected_keys', [])
            print(f"Loaded PyTorch weights from {p}; missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
            return True
        except Exception as e:
            print(f"Failed loading PyTorch weights from {p}: {e}")
    return False


def apply_lora_components(model, lora_path: str, use_llm_lora: bool = True, use_diffusion_head_lora: bool = True,
                         use_acoustic_connector_lora: bool = True, use_semantic_connector_lora: bool = True):
    """Apply LoRA to all requested components"""
    
    applied_components = {}
    
    if not os.path.isdir(lora_path):
        print(f"[LoRA] LoRA path not found: {lora_path}")
        return applied_components
    
    print(f"[LoRA] Applying LoRA from: {lora_path}")
    
    if not any([use_llm_lora, use_diffusion_head_lora, use_acoustic_connector_lora, use_semantic_connector_lora]):
        print("All LoRA component toggles disabled; skipping LoRA application")
        return applied_components

    try:
        # Compute module checksums BEFORE loading optional weights
        def _module_sha1(module: torch.nn.Module) -> Optional[str]:
            if module is None:
                return None
            try:
                h = hashlib.sha1()
                with torch.no_grad():
                    for p in module.parameters():
                        if p is None:
                            continue
                        t = p.detach().float().cpu().contiguous()
                        h.update(t.numpy().tobytes())
                return h.hexdigest()
            except Exception as e:
                print(f"Failed to compute checksum for {module.__class__.__name__}: {e}")
                return None

        ph_before = _module_sha1(getattr(model.model, 'prediction_head', None)) if use_diffusion_head_lora else None
        ac_before = _module_sha1(getattr(model.model, 'acoustic_connector', None)) if use_acoustic_connector_lora else None
        se_before = _module_sha1(getattr(model.model, 'semantic_connector', None)) if use_semantic_connector_lora else None

        # Apply LoRA adapter to language model if requested and possible
        if use_llm_lora and PEFT_AVAILABLE:
            base_lm = getattr(model.model, 'language_model', None)
            if base_lm is not None:
                print(f"[LoRA] Applying LLM adapter from: {lora_path}")
                try:
                    lora_wrapped = PeftModel.from_pretrained(base_lm, lora_path, is_trainable=False)
                    device = next(model.parameters()).device
                    dtype = next(model.parameters()).dtype
                    lora_wrapped = lora_wrapped.to(device=device, dtype=dtype)
                    model.model.language_model = lora_wrapped
                    print("[LoRA] LLM adapter loaded")
                    
                    # Check active adapters
                    adapter_names = []
                    try:
                        if hasattr(lora_wrapped, 'peft_config') and isinstance(lora_wrapped.peft_config, dict):
                            adapter_names = list(lora_wrapped.peft_config.keys())
                        elif hasattr(lora_wrapped, 'active_adapter') and lora_wrapped.active_adapter is not None:
                            adapter_names = [lora_wrapped.active_adapter]
                        elif hasattr(lora_wrapped, 'active_adapters') and lora_wrapped.active_adapters:
                            names = lora_wrapped.active_adapters
                            if isinstance(names, (list, tuple)):
                                adapter_names = list(names)
                            elif isinstance(names, str):
                                adapter_names = [names]
                    except Exception as e:
                        print(f"[LoRA] Failed to query PEFT adapters: {e}")
                    
                    print(f"[LoRA] Active LLM adapters: {adapter_names or 'NONE'}")
                    if not adapter_names:
                        print("[LoRA] Warning: No active PEFT adapters detected on language_model")
                    applied_components["llm"] = True
                except Exception as lm_e:
                    print(f"[LoRA] Failed to apply LLM adapter: {lm_e}")
            else:
                print("[LoRA] Skipping LLM adapter: language_model not present on model")
        elif use_llm_lora and not PEFT_AVAILABLE:
            print("[LoRA] Skipping LLM adapter: PEFT not available")
        else:
            print("[LoRA] LLM adapter disabled by user toggle")

        # Load safetensors support
        try:
            import safetensors.torch as st
        except Exception:
            st = None

        # Apply diffusion head LoRA (full weights)
        if use_diffusion_head_lora:
            diff_head_dir = os.path.join(lora_path, 'diffusion_head')
            dh_dir_exists = os.path.isdir(diff_head_dir)
            if dh_dir_exists:
                ok = _load_state_dict_into(getattr(model.model, 'prediction_head', None), diff_head_dir)
                print(f"[LoRA] Loaded diffusion head weights: {ok}")
                if ok:
                    applied_components["diffusion_head"] = True
            else:
                print("[LoRA] Diffusion head directory not found; skipping")

        # Apply connector LoRA (full weights)
        if use_acoustic_connector_lora:
            ac_dir = os.path.join(lora_path, 'acoustic_connector')
            ac_dir_exists = os.path.isdir(ac_dir)
            if ac_dir_exists:
                ok = _load_state_dict_into(getattr(model.model, 'acoustic_connector', None), ac_dir)
                print(f"[LoRA] Loaded acoustic connector weights: {ok}")
                if ok:
                    applied_components["acoustic_connector"] = True
            else:
                print("[LoRA] Acoustic connector directory not found; skipping")

        if use_semantic_connector_lora:
            se_dir = os.path.join(lora_path, 'semantic_connector')
            se_dir_exists = os.path.isdir(se_dir)
            if se_dir_exists:
                ok = _load_state_dict_into(getattr(model.model, 'semantic_connector', None), se_dir)
                print(f"[LoRA] Loaded semantic connector weights: {ok}")
                if ok:
                    applied_components["semantic_connector"] = True
            else:
                print("[LoRA] Semantic connector directory not found; skipping")

        # Compute checksums AFTER loading and compare
        ph_after = _module_sha1(getattr(model.model, 'prediction_head', None)) if use_diffusion_head_lora else None
        ac_after = _module_sha1(getattr(model.model, 'acoustic_connector', None)) if use_acoustic_connector_lora else None
        se_after = _module_sha1(getattr(model.model, 'semantic_connector', None)) if use_semantic_connector_lora else None

        # Check if weights actually changed
        if use_diffusion_head_lora:
            print(f"[LoRA] Checksum(prediction_head): before={ph_before} after={ph_after}")
            if ph_before == ph_after and "diffusion_head" in applied_components:
                print("[LoRA] Warning: prediction_head checksum unchanged; weights may not have been applied")
        if use_acoustic_connector_lora:
            print(f"[LoRA] Checksum(acoustic_connector): before={ac_before} after={ac_after}")
            if ac_before == ac_after and "acoustic_connector" in applied_components:
                print("[LoRA] Warning: acoustic_connector checksum unchanged; weights may not have been applied")
        if use_semantic_connector_lora:
            print(f"[LoRA] Checksum(semantic_connector): before={se_before} after={se_after}")
            if se_before == se_after and "semantic_connector" in applied_components:
                print("[LoRA] Warning: semantic_connector checksum unchanged; weights may not have been applied")
                
    except Exception as e:
        print(f"[LoRA] Application encountered an error: {e}")
        applied_components["error"] = str(e)

    return applied_components


def get_quantization_config(quantize_llm: str):
    """Get quantization config for LLM"""
    if quantize_llm == '4bit':
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return None


def load_model_with_lora(args):
    """
    Load model with optional quantization and apply LoRA
    """
    
    print(f"Loading processor from {args.model_path}")
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)
    
    config = VibeVoiceConfig.from_pretrained(args.model_path)
    config.quantize_llm = args.quantize_llm
    
    # Get quantization config
    quantization_config = get_quantization_config(args.quantize_llm)
    if quantization_config:
        print(f"[Quantization] Loading model with {args.quantization_llm} quantization")
    else:
        print("[Quantization] Loading model without quantization")
    
    # Load model with optional quantization
    print(f"[Model] Loading to {args.device}...")
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        quantization_config=quantization_config
    )
    
    # Debug: List available modules before LoRA application
    if args.lora_path:
        print("\n[DEBUG] Available model modules:")
        for name, module in model.model.named_modules():
            if any(keyword in name.lower() for keyword in ['head', 'diffusion', 'prediction', 'connector']):
                print(f"  {name}: {type(module).__name__}")
    
    # Apply LoRA components using the enhanced function
    applied_lora_components = {}
    if args.lora_path:
        applied_lora_components = apply_lora_components(
            model, 
            args.lora_path,
            use_llm_lora=args.use_llm_lora,
            use_diffusion_head_lora=args.use_diffusion_head_lora,
            use_acoustic_connector_lora=args.use_acoustic_connector_lora,
            use_semantic_connector_lora=args.use_semantic_connector_lora
        )
    
    # IMPORTANT: Set diffusion steps AFTER LoRA application
    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)
    
    if hasattr(model.model, 'language_model'):
        print(f"Language model attention: {model.model.language_model.config._attn_implementation}")
    
    print(f"Applied LoRA components: {list(applied_lora_components.keys())}")
    
    return model, processor, applied_lora_components


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Inference with LoRA Support")
    parser.add_argument("--model_path", type=str, default="microsoft/VibeVoice-1.5b", help="Path to the HuggingFace model directory")
    parser.add_argument("--txt_path", type=str, default="demo/text_examples/1p_abs.txt", help="Path to the txt file containing the script")
    parser.add_argument("--speaker_names", type=str, nargs='+', default=None, help="Speaker names in order. If not provided, synthetic voices will be used.")
    parser.add_argument("--output", type=str, default="./outputs/generated_audio.wav", help="Output audio file path (e.g., outputs/my_audio.wav)")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")), help="Device for inference")
    parser.add_argument("--cfg_scale", type=float, default=1.3, help="CFG scale for generation")
    parser.add_argument("--quantize_llm", type=str, choices=["none", "4bit"], default="none", help="Quantize language model")
    parser.add_argument("--temperature", type=float, default=0.95, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--seed", type=str, default="42", help="Random seed (number) or 'random' for random seed")
    parser.add_argument("--do_sample", action="store_true", default=True, help="Enable sampling")
    parser.add_argument("--lora_path", type=str, default="", help="Path to LoRA adapter folder")
    parser.add_argument("--use_llm_lora", action="store_true", default=True, help="Apply LLM LoRA")
    parser.add_argument("--use_diffusion_head_lora", action="store_true", default=True, help="Apply diffusion head LoRA")
    parser.add_argument("--use_acoustic_connector_lora", action="store_true", default=True, help="Apply acoustic connector LoRA")
    parser.add_argument("--use_semantic_connector_lora", action="store_true", default=True, help="Apply semantic connector LoRA")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device.lower() == "mpx":
        print("Note: device 'mpx' detected, treating it as 'mps'.")
        args.device = "mps"

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available. Falling back to CPU.")
        args.device = "cpu"

    print(f"Using device: {args.device}")
    print(f"LLM quantization: {args.quantize_llm}")
    
    # Handle seed parameter (can be number or "random")
    if args.seed.lower() == "random":
        import random
        actual_seed = random.randint(0, 2**32-1)
        print(f"Using random seed: {actual_seed}")
    else:
        try:
            actual_seed = int(args.seed)
            print(f"Using fixed seed: {actual_seed}")
        except ValueError:
            print(f"Invalid seed value '{args.seed}', using default seed 42")
            actual_seed = 42
        
    print(f"Generation settings: temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, seed={actual_seed}")
    
    if args.lora_path:
        print(f"LoRA path: {args.lora_path}")
        print(f"LoRA components: LLM={args.use_llm_lora}, Diffusion={args.use_diffusion_head_lora}, Acoustic={args.use_acoustic_connector_lora}, Semantic={args.use_semantic_connector_lora}")
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available - LLM LoRA will be skipped")

    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(actual_seed)
        torch.cuda.manual_seed_all(actual_seed)

    voice_mapper = VoiceMapper()
    
    if not os.path.exists(args.txt_path):
        print(f"Error: txt file not found: {args.txt_path}")
        return
    
    print(f"Reading script from: {args.txt_path}")
    with open(args.txt_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    
    scripts, speaker_numbers = parse_txt_script(txt_content)
    
    if not scripts:
        print("Error: No valid speaker scripts found in the txt file")
        return
    
    print(f"Found {len(scripts)} speaker segments:")
    for i, (script, speaker_num) in enumerate(zip(scripts, speaker_numbers)):
        print(f"  {i+1}. Speaker {speaker_num}")
        print(f"     Text preview: {script[:100]}...")
    
    speaker_name_mapping = {}
    if args.speaker_names is None:
        # No speaker names provided - use synthetic voices with generic names
        print("No speaker names provided - using synthetic voices")
        speaker_names_list = []
        for speaker_num in set(speaker_numbers):
            speaker_names_list.append(f"SyntheticSpeaker{speaker_num}")
    else:
        speaker_names_list = args.speaker_names if isinstance(args.speaker_names, list) else [args.speaker_names]
    
    for i, name in enumerate(speaker_names_list, 1):
        speaker_name_mapping[str(i)] = name
    
    # Fill in missing speakers with synthetic names
    for speaker_num in set(speaker_numbers):
        if speaker_num not in speaker_name_mapping:
            speaker_name_mapping[speaker_num] = f"SyntheticSpeaker{speaker_num}"
    
    print(f"\nSpeaker mapping:")
    for speaker_num in set(speaker_numbers):
        mapped_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
        print(f"  Speaker {speaker_num} -> {mapped_name}")
    
    voice_samples = []
    actual_speakers = []
    unique_speaker_numbers = []
    seen = set()
    for speaker_num in speaker_numbers:
        if speaker_num not in seen:
            unique_speaker_numbers.append(speaker_num)
            seen.add(speaker_num)
    
    for speaker_num in unique_speaker_numbers:
        speaker_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
        speaker_idx = int(speaker_num) - 1  # Convert to 0-based index for synthetic voices
        
        voice_sample = voice_mapper.get_voice_sample(speaker_name, speaker_idx, actual_seed)
        
        if isinstance(voice_sample, np.ndarray):
            # Synthetic voice - already a numpy array
            voice_samples.append(voice_sample)
            print(f"Speaker {speaker_num} ('{speaker_name}') -> Synthetic voice (freq: {120 + (speaker_idx % 4) * 20}Hz)")
        else:
            # Voice file - need to load it
            voice_samples.append(voice_sample)
            actual_speakers.append(speaker_name)
            print(f"Speaker {speaker_num} ('{speaker_name}') -> Voice: {os.path.basename(voice_sample)}")
    
    full_script = '\n'.join(scripts)
    full_script = full_script.replace("'", "'")        
    
    # Load model with LoRA
    model, processor, applied_lora_components = load_model_with_lora(args)
    
    if model is None or processor is None:
        print("Failed to load model and processor")
        return
       
    inputs = processor(
        text=[full_script],
        voice_samples=[voice_samples],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    target_device = args.device if args.device != "cpu" else "cpu"
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)

    print(f"Starting generation with cfg_scale: {args.cfg_scale}")

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=args.cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={
            'do_sample': args.do_sample,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k if args.top_k > 0 else None,
        },
        verbose=True,
    )
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")
    
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        sample_rate = 24000
        audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
        
        print(f"Generated audio duration: {audio_duration:.2f} seconds")
        print(f"RTF (Real Time Factor): {rtf:.2f}x")
    else:
        print("No audio output generated")
    
    input_tokens = inputs['input_ids'].shape[1]
    output_tokens = outputs.sequences.shape[1]
    generated_tokens = output_tokens - input_tokens
    
    print(f"Prefilling tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")

    # Handle output file path
    output_path = args.output
    
    # Ensure .wav extension
    if not output_path.lower().endswith('.wav'):
        output_path += '.wav'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    processor.save_audio(
        outputs.speech_outputs[0],
        output_path=output_path,
    )
    print(f"Saved output to {output_path}")
    
    print("\n" + "="*50)
    print("GENERATION SUMMARY")
    print("="*50)
    print(f"Input file: {args.txt_path}")
    print(f"Output file: {output_path}")
    print(f"Speaker names: {args.speaker_names}")
    print(f"Seed used: {actual_seed}")
    print(f"LLM quantization: {args.quantize_llm}")
    if args.lora_path:
        print(f"LoRA path: {args.lora_path}")
        print(f"Applied LoRA components: {list(applied_lora_components.keys())}")
        if "error" in applied_lora_components:
            print(f"LoRA errors: {applied_lora_components['error']}")
    print(f"Number of unique speakers: {len(set(speaker_numbers))}")
    print(f"Number of segments: {len(scripts)}")
    print(f"Prefilling tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")
    print(f"Generation time: {generation_time:.2f} seconds")
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        print(f"Audio duration: {audio_duration:.2f} seconds")
        print(f"RTF (Real Time Factor): {rtf:.2f}x")
    
    print("="*50)


if __name__ == "__main__":
    main()