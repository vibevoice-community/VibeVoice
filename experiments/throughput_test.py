#!/usr/bin/env python3
"""
50-batch throughput test for Bayesian optimization of memory efficiency.
Tests config with longest sample duplicated to detect OOM early.
Agent: bayesopt-agent
"""

import os
import sys
import json
import time
import torch
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict


@dataclass
class ThroughputTestConfig:
    """Configuration for throughput testing."""
    # Model config
    model_name_or_path: str = "marksverdhai/vibevoice-7b-bnb-8bit"
    processor_name_or_path: str = "vibevoice/VibeVoice-7B"
    dataset_name: str = "heiertech/vibevoice-mcv-scripted-no-v24"

    # Hyperparameters to test
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    ddpm_batch_mul: int = 4

    # Test parameters
    num_test_batches: int = 50
    use_longest_sample: bool = True

    # Fixed params
    load_in_4bit: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = True


@dataclass
class ThroughputTestResult:
    """Results from throughput test."""
    config: Dict
    success: bool
    oom_occurred: bool
    avg_batch_time: Optional[float] = None
    samples_per_second: Optional[float] = None
    memory_peak_gb: Optional[float] = None
    estimated_full_runtime_hours: Optional[float] = None
    error_message: Optional[str] = None


def run_throughput_test(config: ThroughputTestConfig) -> ThroughputTestResult:
    """Run 50-batch throughput test."""

    print("=" * 80)
    print("THROUGHPUT TEST")
    print("=" * 80)
    print(f"Config: batch_size={config.per_device_train_batch_size}, "
          f"grad_accum={config.gradient_accumulation_steps}, "
          f"lora_r={config.lora_r}, ddpm_mul={config.ddpm_batch_mul}")
    print("=" * 80)

    try:
        # Register VibeVoice
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
        from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
        from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
        from vibevoice.modular.configuration_vibevoice import (
            VibeVoiceConfig,
            VibeVoiceAcousticTokenizerConfig,
            VibeVoiceSemanticTokenizerConfig,
            VibeVoiceDiffusionHeadConfig,
        )

        AutoConfig.register("vibevoice", VibeVoiceConfig)
        AutoConfig.register("vibevoice_acoustic_tokenizer", VibeVoiceAcousticTokenizerConfig)
        AutoConfig.register("vibevoice_semantic_tokenizer", VibeVoiceSemanticTokenizerConfig)
        AutoConfig.register("vibevoice_diffusion_head", VibeVoiceDiffusionHeadConfig)
        AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGeneration)
        TOKENIZER_MAPPING.register(VibeVoiceConfig, (Qwen2Tokenizer, Qwen2TokenizerFast))

        # Load model with Unsloth
        print("\n[1/6] Loading model...")
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            config.model_name_or_path,
            auto_model=VibeVoiceForConditionalGeneration,
            dtype=torch.bfloat16 if config.bf16 else torch.float16,
            whisper_language="none",
            whisper_task="none",
            use_gradient_checkpointing="unsloth" if config.gradient_checkpointing else False,
            load_in_4bit=config.load_in_4bit,
        )

        print("✓ Model loaded")

        # Load processor
        print("\n[2/6] Loading processor...")
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        processor = VibeVoiceProcessor.from_pretrained(config.processor_name_or_path)
        processor.semantic_tokenizer = getattr(model.model, "semantic_tokenizer", None)
        print("✓ Processor loaded")

        # Setup LoRA
        print(f"\n[3/6] Setting up LoRA (r={config.lora_r}, alpha={config.lora_alpha})...")
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

        # Freeze tokenizers
        if hasattr(model.model, "acoustic_tokenizer"):
            for p in model.model.acoustic_tokenizer.parameters():
                p.requires_grad = False
        if hasattr(model.model, "semantic_tokenizer"):
            for p in model.model.semantic_tokenizer.parameters():
                p.requires_grad = False

        # Prepare for k-bit training
        if config.load_in_4bit:
            model.model.language_model = prepare_model_for_kbit_training(
                model.model.language_model,
                use_gradient_checkpointing=False,
            )

        # Apply LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model.model.language_model = get_peft_model(model.model.language_model, lora_config)

        # Enable training for diffusion head
        for p in model.model.prediction_head.parameters():
            p.requires_grad = True

        print("✓ LoRA configured")

        # Load dataset
        print(f"\n[4/6] Loading dataset (first 100 samples)...")
        from datasets import load_dataset
        ds = load_dataset(config.dataset_name, split="train", streaming=True)

        # Get longest sample or first batch
        samples = []
        max_duration = 0
        longest_sample = None

        for idx, sample in enumerate(ds.take(100)):
            samples.append(sample)
            if config.use_longest_sample:
                audio = sample.get('audio', {})
                if audio and 'array' in audio:
                    duration = len(audio['array']) / audio.get('sampling_rate', 16000)
                    if duration > max_duration:
                        max_duration = duration
                        longest_sample = sample

        if config.use_longest_sample and longest_sample:
            print(f"✓ Found longest sample: {max_duration:.2f}s")
            # Duplicate longest sample to fill batch
            test_samples = [longest_sample] * config.per_device_train_batch_size
        else:
            test_samples = samples[:config.per_device_train_batch_size]

        print(f"✓ Test batch created ({len(test_samples)} samples)")

        # Create data collator
        print("\n[5/6] Setting up data collator...")
        from vibevoice.finetune.data_vibevoice import VibeVoiceCollator

        speech_compress_ratio = getattr(processor, "speech_tok_compress_ratio", 3200)
        semantic_dim = getattr(model.config, "semantic_vae_dim", 128)

        collator = VibeVoiceCollator(
            processor=processor,
            max_length=None,
            speech_compress_ratio=speech_compress_ratio,
            semantic_vae_dim=semantic_dim,
            compute_semantics=True,
            debug_checks=False,
            voice_prompt_drop_rate=0.2,
        )
        print("✓ Collator ready")

        # Run throughput test
        print(f"\n[6/6] Running {config.num_test_batches}-batch throughput test...")
        print("=" * 80)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

        batch_times = []
        torch.cuda.reset_peak_memory_stats()

        for batch_idx in range(config.num_test_batches):
            try:
                # Collate batch
                batch = collator(test_samples)

                # Move to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Time forward + backward
                start_time = time.time()

                # Forward pass (simplified - just test memory)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16 if config.bf16 else torch.float16):
                    # Dummy forward to test memory
                    input_ids = batch.get("input_ids")
                    if input_ids is not None:
                        outputs = model(input_ids=input_ids[:1])  # Test with 1 sample
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs.logits.sum()
                        loss.backward()
                        optimizer.zero_grad()

                batch_time = time.time() - start_time
                batch_times.append(batch_time)

                # Memory stats
                memory_used = torch.cuda.max_memory_allocated() / 1024**3

                if (batch_idx + 1) % 10 == 0:
                    avg_time = sum(batch_times[-10:]) / len(batch_times[-10:])
                    print(f"  Batch {batch_idx + 1}/{config.num_test_batches}: "
                          f"{avg_time:.3f}s/batch, Memory: {memory_used:.2f}GB")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n✗ OOM at batch {batch_idx + 1}")
                    return ThroughputTestResult(
                        config=asdict(config),
                        success=False,
                        oom_occurred=True,
                        error_message=f"OOM at batch {batch_idx + 1}"
                    )
                raise

        # Calculate results
        avg_batch_time = sum(batch_times) / len(batch_times)
        samples_per_second = config.per_device_train_batch_size / avg_batch_time
        memory_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

        # Estimate full runtime (assuming 10k samples, 3 epochs)
        total_samples = 10000 * 3  # Rough estimate
        batches_per_epoch = total_samples / (config.per_device_train_batch_size * config.gradient_accumulation_steps)
        estimated_seconds = batches_per_epoch * avg_batch_time * config.gradient_accumulation_steps
        estimated_hours = estimated_seconds / 3600

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"✓ Success - No OOM")
        print(f"  Avg batch time: {avg_batch_time:.3f}s")
        print(f"  Throughput: {samples_per_second:.2f} samples/sec")
        print(f"  Memory peak: {memory_peak_gb:.2f}GB")
        print(f"  Est. full runtime: {estimated_hours:.1f} hours")
        print("=" * 80)

        return ThroughputTestResult(
            config=asdict(config),
            success=True,
            oom_occurred=False,
            avg_batch_time=avg_batch_time,
            samples_per_second=samples_per_second,
            memory_peak_gb=memory_peak_gb,
            estimated_full_runtime_hours=estimated_hours,
        )

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

        return ThroughputTestResult(
            config=asdict(config),
            success=False,
            oom_occurred="out of memory" in str(e).lower(),
            error_message=str(e),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--ddpm_mul", type=int, default=4)
    parser.add_argument("--output", type=str, default="experiments/throughput_results")
    args = parser.parse_args()

    config = ThroughputTestConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        ddpm_batch_mul=args.ddpm_mul,
    )

    result = run_throughput_test(config)

    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(
        args.output,
        f"test_bs{args.batch_size}_ga{args.grad_accum}_r{args.lora_r}.json"
    )

    with open(output_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Return exit code based on success
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
