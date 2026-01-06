#!/usr/bin/env python3
"""
Fast throughput testing - loads model once, tests multiple configs.
Target: <2 minutes per config.
Agent: bayesopt-agent
"""

import os
import sys
import json
import time
import torch
import gc
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class FastTestConfig:
    """Lightweight config for fast testing."""
    batch_size: int
    grad_accum: int
    lora_r: int
    ddpm_mul: int


@dataclass
class FastTestResult:
    """Results from fast test."""
    config: Dict
    success: bool
    oom: bool
    avg_batch_time: Optional[float] = None
    samples_per_sec: Optional[float] = None
    memory_gb: Optional[float] = None
    est_runtime_hours: Optional[float] = None


class FastThroughputTester:
    """Test multiple configs quickly by reusing loaded model."""

    def __init__(self, num_batches: int = 10):
        self.num_batches = num_batches
        self.model = None
        self.processor = None
        self.collator = None
        self.test_batch = None
        self.base_lora_r = None

    def setup_once(self):
        """Setup that only happens once."""
        print("\n" + "=" * 80)
        print("ONE-TIME SETUP (Loading model, dataset, etc.)")
        print("=" * 80)

        # Register VibeVoice
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
        from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
        from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
        from vibevoice.modular.configuration_vibevoice import (
            VibeVoiceConfig, VibeVoiceAcousticTokenizerConfig,
            VibeVoiceSemanticTokenizerConfig, VibeVoiceDiffusionHeadConfig,
        )

        AutoConfig.register("vibevoice", VibeVoiceConfig)
        AutoConfig.register("vibevoice_acoustic_tokenizer", VibeVoiceAcousticTokenizerConfig)
        AutoConfig.register("vibevoice_semantic_tokenizer", VibeVoiceSemanticTokenizerConfig)
        AutoConfig.register("vibevoice_diffusion_head", VibeVoiceDiffusionHeadConfig)
        AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGeneration)
        TOKENIZER_MAPPING.register(VibeVoiceConfig, (Qwen2Tokenizer, Qwen2TokenizerFast))

        # Load processor first (from base model, not quantized)
        print("Loading processor...")
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        self.processor = VibeVoiceProcessor.from_pretrained("vibevoice/VibeVoice-7B")

        # Load base model using standard transformers (Unsloth's loader has issues with quantized models without processor)
        print("Loading model...")
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        # Use base model, not pre-quantized
        # Don't use device_map="auto" - it has issues with speech_bias_factor buffer
        self.model = VibeVoiceForConditionalGeneration.from_pretrained(
            "vibevoice/VibeVoice-7B",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        ).cuda()

        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()

        self.processor.semantic_tokenizer = getattr(self.model.model, "semantic_tokenizer", None)

        # Freeze tokenizers
        for module in [self.model.model.acoustic_tokenizer, self.model.model.semantic_tokenizer]:
            if module:
                for p in module.parameters():
                    p.requires_grad = False

        # Load dataset and find longest sample
        print("Loading dataset and finding longest sample...")
        from datasets import load_dataset

        ds = load_dataset("heiertech/vibevoice-mcv-scripted-no-v24", split="train", streaming=True)

        max_duration = 0
        longest = None
        for idx, sample in enumerate(ds.take(200)):
            audio = sample.get('audio', {})
            if audio and 'array' in audio:
                duration = len(audio['array']) / audio.get('sampling_rate', 16000)
                if duration > max_duration:
                    max_duration = duration
                    longest = sample
            if idx % 50 == 0:
                print(f"  Checked {idx} samples...")

        print(f"✓ Longest sample: {max_duration:.2f}s")

        # Create test batch (duplicate longest sample)
        self.longest_sample = longest

        # Setup collator
        print("Setting up collator...")
        from vibevoice.finetune.data_vibevoice import VibeVoiceCollator

        self.collator = VibeVoiceCollator(
            processor=self.processor,
            max_length=None,
            speech_compress_ratio=getattr(self.processor, "speech_tok_compress_ratio", 3200),
            semantic_vae_dim=getattr(self.model.config, "semantic_vae_dim", 128),
            compute_semantics=True,
            debug_checks=False,
            voice_prompt_drop_rate=0.2,
        )

        print("✓ Setup complete!\n")

    def apply_lora_config(self, lora_r: int, lora_alpha: int):
        """Apply or update LoRA configuration."""
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

        # If this is the first LoRA or rank changed, reapply
        if self.base_lora_r != lora_r:
            print(f"  Applying LoRA r={lora_r}, α={lora_alpha}...")

            # Prepare for k-bit training
            self.model.model.language_model = prepare_model_for_kbit_training(
                self.model.model.language_model,
                use_gradient_checkpointing=False,
            )

            # Apply LoRA
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )

            self.model.model.language_model = get_peft_model(self.model.model.language_model, lora_config)
            self.base_lora_r = lora_r

        # Always enable diffusion head training
        for p in self.model.model.prediction_head.parameters():
            p.requires_grad = True

    def test_config(self, config: FastTestConfig) -> FastTestResult:
        """Test a single configuration quickly."""

        print(f"\n{'─' * 80}")
        print(f"Testing: bs={config.batch_size}, ga={config.grad_accum}, r={config.lora_r}, ddpm={config.ddpm_mul}")
        print(f"{'─' * 80}")

        try:
            # Apply LoRA if needed
            self.apply_lora_config(config.lora_r, config.lora_r * 2)

            # Create batch (duplicate longest sample)
            test_samples = [self.longest_sample] * config.batch_size
            batch = self.collator(test_samples)

            # Move to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Setup optimizer
            self.model.train()
            optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=3e-5)

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            # Run test batches
            batch_times = []
            start = time.time()

            for i in range(self.num_batches):
                batch_start = time.time()

                # Forward + backward
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")

                if input_ids is not None:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        loss = outputs.logits.sum() * 0.0001  # Tiny loss to trigger backward
                        loss.backward()
                        optimizer.zero_grad()

                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                if (i + 1) % 5 == 0:
                    avg = sum(batch_times[-5:]) / 5
                    print(f"  Batch {i+1}/{self.num_batches}: {avg:.3f}s/batch")

            total_time = time.time() - start

            # Results
            avg_batch_time = sum(batch_times) / len(batch_times)
            samples_per_sec = config.batch_size / avg_batch_time
            memory_gb = torch.cuda.max_memory_allocated() / 1024**3

            # Estimate full runtime (30k samples, 3 epochs)
            total_samples = 30000
            batches_needed = total_samples / (config.batch_size * config.grad_accum)
            est_seconds = batches_needed * avg_batch_time * config.grad_accum
            est_hours = est_seconds / 3600

            print(f"✓ Success: {samples_per_sec:.2f} samples/sec, {memory_gb:.1f}GB, ETA {est_hours:.1f}h")

            return FastTestResult(
                config=asdict(config),
                success=True,
                oom=False,
                avg_batch_time=avg_batch_time,
                samples_per_sec=samples_per_sec,
                memory_gb=memory_gb,
                est_runtime_hours=est_hours,
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"✗ OOM")
                torch.cuda.empty_cache()
                return FastTestResult(
                    config=asdict(config),
                    success=False,
                    oom=True,
                )
            raise

        except Exception as e:
            print(f"✗ Error: {e}")
            return FastTestResult(
                config=asdict(config),
                success=False,
                oom=False,
            )

    def test_multiple(self, configs: List[FastTestConfig]) -> List[FastTestResult]:
        """Test multiple configs."""
        results = []

        for idx, config in enumerate(configs, 1):
            print(f"\n{'═' * 80}")
            print(f"CONFIG {idx}/{len(configs)}")
            print(f"{'═' * 80}")

            result = self.test_config(config)
            results.append(result)

            # Cleanup between tests
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, required=True, help="Path to configs JSON file")
    parser.add_argument("--output", type=str, default="experiments/fast_results.json")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to test")
    args = parser.parse_args()

    # Load configs
    with open(args.configs, 'r') as f:
        config_dicts = json.load(f)

    configs = [FastTestConfig(**c) for c in config_dicts]

    print("=" * 80)
    print(f"FAST THROUGHPUT TESTING - {len(configs)} configs")
    print(f"Target: <2 minutes per config ({args.num_batches} batches each)")
    print("=" * 80)

    # Initialize tester
    tester = FastThroughputTester(num_batches=args.num_batches)

    # Setup once
    start_time = time.time()
    tester.setup_once()
    setup_time = time.time() - start_time
    print(f"Setup time: {setup_time:.1f}s")

    # Test all configs
    test_start = time.time()
    results = tester.test_multiple(configs)
    test_time = time.time() - test_start

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Setup: {setup_time:.1f}s")
    print(f"Testing: {test_time:.1f}s ({test_time/len(configs):.1f}s per config)")
    print(f"Successful: {sum(1 for r in results if r.success)}/{len(results)}")
    print(f"OOM: {sum(1 for r in results if r.oom)}/{len(results)}")

    # Find best
    successful = [r for r in results if r.success]
    if successful:
        best = max(successful, key=lambda x: x.samples_per_sec)
        print(f"\nBest: {best.samples_per_sec:.2f} samples/sec, {best.memory_gb:.1f}GB, {best.est_runtime_hours:.1f}h ETA")
        print(f"Config: {best.config}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
