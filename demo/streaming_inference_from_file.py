"""
VibeVoice Streaming Model Inference from File

This script demonstrates how to use the VibeVoice-Streaming-0.5B model
for text-to-speech synthesis from a text file.

The streaming model uses pre-computed voice embeddings (.pt files) for
low-latency generation, making it suitable for real-time applications.

Usage:
    python demo/streaming_inference_from_file.py \
        --model_path microsoft/VibeVoice-Realtime-0.5B \
        --txt_path demo/text_examples/1p_vibevoice.txt \
        --speaker_name Emma \
        --output_dir ./outputs

Available voice presets (in demo/voices/streaming_model/):
    - Carter, Davis, Emma, Frank, Grace, Mike (English)
    - Samuel (Indian English)
"""

import argparse
import os
import re
import traceback
from typing import List, Tuple, Union, Dict, Any
import time
import torch
import copy

from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VoiceMapper:
    """Maps speaker names to voice file paths for the streaming model."""

    def __init__(self):
        self.setup_voice_presets()

        # Create name aliases without prefixes/suffixes for easier matching
        new_dict = {}
        for name, path in self.voice_presets.items():
            # Remove gender suffix (e.g., "en-Emma_woman" -> "Emma")
            if '_' in name:
                name = name.split('_')[0]

            # Remove language prefix (e.g., "en-Emma" -> "Emma")
            if '-' in name:
                name = name.split('-')[-1]

            new_dict[name] = path
        self.voice_presets.update(new_dict)

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices/streaming_model directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices/streaming_model")

        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return

        # Scan for all .pt files in the voices directory
        self.voice_presets = {}

        pt_files = [f for f in os.listdir(voices_dir)
                    if f.lower().endswith('.pt') and os.path.isfile(os.path.join(voices_dir, f))]

        # Create dictionary with filename (without extension) as key
        for pt_file in pt_files:
            name = os.path.splitext(pt_file)[0]
            full_path = os.path.join(voices_dir, pt_file)
            self.voice_presets[name] = full_path

        # Sort alphabetically for consistent ordering
        self.voice_presets = dict(sorted(self.voice_presets.items()))

        # Filter out voices that don't exist (safety check)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }

        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name."""
        # First try exact match
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]

        # Try partial matching (case insensitive)
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path

        # Default to first voice if no match found
        if self.voice_presets:
            default_voice = list(self.voice_presets.values())[0]
            print(f"Warning: No voice preset found for '{speaker_name}', using default voice")
            return default_voice

        raise ValueError(f"No voice presets available. Please check the voices/streaming_model directory.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="VibeVoice Streaming Model (0.5B) - Text-to-Speech from File",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default settings
    python demo/streaming_inference_from_file.py --txt_path demo/text_examples/1p_vibevoice.txt

    # Specify a different voice
    python demo/streaming_inference_from_file.py --txt_path demo/text_examples/1p_vibevoice.txt --speaker_name Emma

    # Use local model path
    python demo/streaming_inference_from_file.py --model_path /path/to/model --txt_path input.txt
        """
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-Realtime-0.5B",
        help="Path to the HuggingFace model directory or model ID",
    )
    parser.add_argument(
        "--txt_path",
        type=str,
        default="demo/text_examples/1p_vibevoice.txt",
        help="Path to the txt file containing the script",
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default="Emma",
        help="Speaker name for voice selection (e.g., Emma, Carter, Mike)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save output audio files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),
        help="Device for inference: cuda | mps | cpu",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="CFG (Classifier-Free Guidance) scale for generation (default: 1.5)",
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=5,
        help="Number of DDPM inference steps (default: 5, lower is faster)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Using random seed: {args.seed}")

    # Normalize potential 'mpx' typo to 'mps'
    if args.device.lower() == "mpx":
        print("Note: device 'mpx' detected, treating it as 'mps'.")
        args.device = "mps"

    # Validate mps availability if requested
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available. Falling back to CPU.")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    # Initialize voice mapper
    voice_mapper = VoiceMapper()

    # Check if txt file exists
    if not os.path.exists(args.txt_path):
        print(f"Error: txt file not found: {args.txt_path}")
        return

    # Read and parse txt file
    print(f"Reading script from: {args.txt_path}")
    with open(args.txt_path, 'r', encoding='utf-8') as f:
        scripts = f.read().strip()

    if not scripts:
        print("Error: No valid scripts found in the txt file")
        return

    # Normalize quotes
    full_script = scripts.replace("'", "'").replace('"', '"').replace('"', '"')

    print(f"Loading processor & model from {args.model_path}")
    processor = VibeVoiceStreamingProcessor.from_pretrained(args.model_path)

    # Decide dtype & attention implementation based on device
    if args.device == "mps":
        load_dtype = torch.float32  # MPS requires float32
        attn_impl_primary = "sdpa"  # flash_attention_2 not supported on MPS
    elif args.device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl_primary = "flash_attention_2"
    else:  # cpu
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"

    print(f"Using device: {args.device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")

    # Load model with device-specific logic
    try:
        if args.device == "mps":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                attn_implementation=attn_impl_primary,
                device_map=None,
            )
            model.to("mps")
        elif args.device == "cuda":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map="cuda",
                attn_implementation=attn_impl_primary,
            )
        else:  # cpu
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map="cpu",
                attn_implementation=attn_impl_primary,
            )
    except Exception as e:
        if attn_impl_primary == 'flash_attention_2':
            print(f"[ERROR] : {type(e).__name__}: {e}")
            print(traceback.format_exc())
            print("Error loading the model. Trying to use SDPA. Note that flash_attention_2 is recommended for best quality.")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map=(args.device if args.device in ("cuda", "cpu") else None),
                attn_implementation='sdpa'
            )
            if args.device == "mps":
                model.to("mps")
        else:
            raise e

    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)

    if hasattr(model.model, 'language_model'):
        print(f"Language model attention: {model.model.language_model.config._attn_implementation}")

    # Load voice preset
    target_device = args.device if args.device != "cpu" else "cpu"
    voice_sample = voice_mapper.get_voice_path(args.speaker_name)
    print(f"Loading voice preset: {voice_sample}")
    all_prefilled_outputs = torch.load(voice_sample, map_location=target_device, weights_only=False)

    # Prepare inputs for the model
    inputs = processor.process_input_with_cached_prompt(
        text=full_script,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Move tensors to target device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)

    print(f"Starting generation with cfg_scale: {args.cfg_scale}, ddpm_steps: {args.ddpm_steps}")

    # Generate audio
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=args.cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={'do_sample': False},
        verbose=True,
        all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs) if all_prefilled_outputs is not None else None,
    )
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")

    # Calculate audio duration and additional metrics
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        sample_rate = 24000  # 24kHz sample rate
        audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')

        print(f"Generated audio duration: {audio_duration:.2f} seconds")
        print(f"RTF (Real Time Factor): {rtf:.2f}x")
    else:
        print("No audio output generated")
        audio_duration = 0
        rtf = float('inf')

    # Calculate token metrics
    input_tokens = inputs['tts_text_ids'].shape[1]
    output_tokens = outputs.sequences.shape[1]
    generated_tokens = output_tokens - input_tokens - all_prefilled_outputs['tts_lm']['last_hidden_state'].size(1)

    print(f"Prefilling text tokens: {input_tokens}")
    print(f"Generated speech tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")

    # Save output
    txt_filename = os.path.splitext(os.path.basename(args.txt_path))[0]
    output_path = os.path.join(args.output_dir, f"{txt_filename}_streaming_generated.wav")
    os.makedirs(args.output_dir, exist_ok=True)

    processor.save_audio(
        outputs.speech_outputs[0],
        output_path=output_path,
    )
    print(f"Saved output to {output_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("GENERATION SUMMARY (Streaming 0.5B)")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Input file: {args.txt_path}")
    print(f"Output file: {output_path}")
    print(f"Speaker: {args.speaker_name}")
    print(f"CFG Scale: {args.cfg_scale}")
    print(f"DDPM Steps: {args.ddpm_steps}")
    print(f"Prefilling text tokens: {input_tokens}")
    print(f"Generated speech tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"RTF (Real Time Factor): {rtf:.2f}x")
    print("=" * 50)


if __name__ == "__main__":
    main()
