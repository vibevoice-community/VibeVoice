#!/usr/bin/env python3
"""
Find the longest sample in the dataset for OOM testing.
Agent: bayesopt-agent (throughput optimization)
"""

import sys
from datasets import load_dataset
import librosa


def find_longest_sample(dataset_name: str, split: str = "train", max_samples: int = 1000):
    """Find the longest audio sample in the dataset."""

    print(f"Loading dataset: {dataset_name}")
    print(f"Analyzing first {max_samples} samples...")

    try:
        ds = load_dataset(dataset_name, split=split, streaming=True)

        longest_duration = 0
        longest_idx = -1
        longest_sample = None

        for idx, sample in enumerate(ds.take(max_samples)):
            # Get audio duration
            audio = sample.get('audio', {})

            if audio and 'array' in audio:
                sr = audio.get('sampling_rate', 16000)
                duration = len(audio['array']) / sr

                if duration > longest_duration:
                    longest_duration = duration
                    longest_idx = idx
                    longest_sample = sample

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} samples, current max: {longest_duration:.2f}s")

        print("\n" + "=" * 80)
        print("LONGEST SAMPLE FOUND")
        print("=" * 80)
        print(f"Index: {longest_idx}")
        print(f"Duration: {longest_duration:.2f} seconds")
        print(f"Audio length: {len(longest_sample['audio']['array'])} samples")
        print(f"Sample rate: {longest_sample['audio']['sampling_rate']} Hz")

        if 'text' in longest_sample:
            text = longest_sample['text']
            print(f"Text length: {len(text)} characters")
            print(f"Text preview: {text[:200]}...")

        return longest_sample, longest_duration

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


if __name__ == "__main__":
    dataset = "heiertech/vibevoice-mcv-scripted-no-v24"

    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    sample, duration = find_longest_sample(dataset, max_samples=2000)

    if sample:
        print("\n" + "=" * 80)
        print("Use this sample for OOM testing")
        print(f"Estimated memory requirement: ~{duration * 100:.0f} MB per sample")
        print("=" * 80)
