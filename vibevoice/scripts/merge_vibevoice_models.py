#!/usr/bin/env python3
"""
VibeVoice Universal Model Merger

Automatically detects and merges trained components back into the base model:
- LLM LoRA adapters
- Diffusion head (LoRA or full fine-tune)
- Acoustic/Semantic connectors

Supports all training configurations from train_vibevoice.py
"""

import argparse
import logging
import os
import shutil
from typing import Dict, Optional

import torch

from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def detect_trained_components(checkpoint_path: str) -> Dict[str, bool]:
    """Detect which components were trained."""
    
    components = {
        "llm_lora": False,
        "diffusion_head": False,
        "acoustic_connector": False,
        "semantic_connector": False,
    }
    
    # Check LLM LoRA (adapter files in root checkpoint/)
    llm_adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
    llm_adapter_model = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if not os.path.exists(llm_adapter_model):
        llm_adapter_model = os.path.join(checkpoint_path, "adapter_model.bin")
    
    if os.path.exists(llm_adapter_config) and os.path.exists(llm_adapter_model):
        components["llm_lora"] = True
    
    # Check diffusion head
    diffusion_head_dir = os.path.join(checkpoint_path, "diffusion_head")
    if os.path.exists(diffusion_head_dir):
        components["diffusion_head"] = True
    
    # Check acoustic connector
    acoustic_conn_path = os.path.join(checkpoint_path, "acoustic_connector", "pytorch_model.bin")
    if os.path.exists(acoustic_conn_path):
        components["acoustic_connector"] = True
    
    # Check semantic connector
    semantic_conn_path = os.path.join(checkpoint_path, "semantic_connector", "pytorch_model.bin")
    if os.path.exists(semantic_conn_path):
        components["semantic_connector"] = True
    
    return components


def merge_llm_lora(model: VibeVoiceForConditionalGeneration, checkpoint_path: str) -> None:
    """Merge LLM LoRA adapters into base model."""
    
    logger.info("Merging LLM LoRA adapters...")
    
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("peft library required for LoRA merge. Install: pip install peft")
    
    # Load and merge LoRA
    language_model_with_lora = PeftModel.from_pretrained(
        model.model.language_model,
        checkpoint_path
    )
    
    merged_lm = language_model_with_lora.merge_and_unload()
    model.model.language_model = merged_lm
    
    logger.info("✓ LLM LoRA merge completed")


def merge_diffusion_head(model: VibeVoiceForConditionalGeneration, checkpoint_path: str) -> dict:
    """Merge diffusion head weights (LoRA or full fine-tune).
    
    Returns:
        trained_state_dict for verification
    """
    
    logger.info("Merging diffusion head...")
    
    diffusion_head_dir = os.path.join(checkpoint_path, "diffusion_head")
    
    # Try multiple possible weight file locations
    possible_files = [
        os.path.join(diffusion_head_dir, "model.safetensors"),
        os.path.join(diffusion_head_dir, "diffusion_head_full.bin"),
        os.path.join(checkpoint_path, "diffusion_head_full.bin"),
    ]
    
    trained_weights_path = None
    for path in possible_files:
        if os.path.exists(path):
            trained_weights_path = path
            break
    
    if trained_weights_path is None:
        raise ValueError(
            f"Diffusion head weights not found. Searched:\n" +
            "\n".join(f"  - {p}" for p in possible_files)
        )
    
    logger.info(f"Loading from: {trained_weights_path}")
    
    # Load weights
    if trained_weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        trained_state_dict = load_file(trained_weights_path)
    else:
        trained_state_dict = torch.load(trained_weights_path, map_location="cpu")
    
    # Check if LoRA-wrapped (has adapter keys like lora_A, lora_B)
    is_lora = any("lora_" in k for k in trained_state_dict.keys())
    
    if is_lora:
        logger.info("Detected LoRA format, performing LoRA merge...")
        try:
            from peft import PeftModel
            pred_head_with_lora = PeftModel.from_pretrained(
                model.model.prediction_head,
                diffusion_head_dir
            )
            model.model.prediction_head = pred_head_with_lora.merge_and_unload()
        except Exception as e:
            logger.warning(f"LoRA merge failed: {e}. Trying direct state_dict load...")
            model.model.prediction_head.load_state_dict(trained_state_dict, strict=True)
    else:
        logger.info("Detected full fine-tune format, replacing weights...")
        model.model.prediction_head.load_state_dict(trained_state_dict, strict=True)
    
    logger.info("✓ Diffusion head merge completed")
    return trained_state_dict


def merge_connectors(model: VibeVoiceForConditionalGeneration, checkpoint_path: str, 
                     merge_acoustic: bool, merge_semantic: bool) -> None:
    """Merge acoustic and/or semantic connector weights."""
    
    if merge_acoustic:
        logger.info("Merging acoustic connector...")
        acoustic_path = os.path.join(checkpoint_path, "acoustic_connector", "pytorch_model.bin")
        state_dict = torch.load(acoustic_path, map_location="cpu")
        model.model.acoustic_connector.load_state_dict(state_dict, strict=True)
        logger.info("✓ Acoustic connector merge completed")
    
    if merge_semantic:
        logger.info("Merging semantic connector...")
        semantic_path = os.path.join(checkpoint_path, "semantic_connector", "pytorch_model.bin")
        state_dict = torch.load(semantic_path, map_location="cpu")
        model.model.semantic_connector.load_state_dict(state_dict, strict=True)
        logger.info("✓ Semantic connector merge completed")


def verify_merge(
    base_model: VibeVoiceForConditionalGeneration,
    merged_model: VibeVoiceForConditionalGeneration,
    trained_state_dict: Optional[dict],
    component_name: str
) -> None:
    """
    Verify that merge was successful.
    
    Args:
        base_model: Original base model
        merged_model: Model after merge
        trained_state_dict: Trained weights that were merged (if available)
        component_name: Name of component being verified (e.g., "diffusion_head")
    """
    
    logger.info(f"\n=== Verifying {component_name} merge ===")
    
    # Get component modules
    if component_name == "diffusion_head":
        base_module = base_model.model.prediction_head
        merged_module = merged_model.model.prediction_head
    elif component_name == "acoustic_connector":
        base_module = base_model.model.acoustic_connector
        merged_module = merged_model.model.acoustic_connector
    elif component_name == "semantic_connector":
        base_module = base_model.model.semantic_connector
        merged_module = merged_model.model.semantic_connector
    else:
        logger.warning(f"Unknown component: {component_name}, skipping verification")
        return
    
    base_state = base_module.state_dict()
    merged_state = merged_module.state_dict()
    
    # 1. Check that weights actually changed
    logger.info("Checking if weights changed from base model...")
    weights_changed = False
    changed_params = []
    
    for key in base_state.keys():
        if key not in merged_state:
            continue
        if not torch.allclose(base_state[key], merged_state[key], rtol=1e-5, atol=1e-8):
            weights_changed = True
            changed_params.append(key)
    
    if not weights_changed:
        # Weights didn't change - this is OK for connectors (may not have been trained)
        # But it's an error for diffusion_head (should always be trained)
        if component_name == "diffusion_head":
            raise ValueError(f"✗ ERROR: {component_name} weights did not change! Merge may have failed.")
        else:
            logger.info(f"✓ {component_name}: unchanged (was not trained)")
            return
    
    logger.info(f"✓ Weights changed: {len(changed_params)}/{len(base_state)} parameters modified")
    
    # 2. Check trained weights match merged weights (if available)
    if trained_state_dict is not None:
        logger.info("Verifying trained weights match merged model...")
        mismatches = []
        
        for key in trained_state_dict.keys():
            if key not in merged_state:
                mismatches.append(f"{key} (missing in merged)")
                continue
            
            # Convert to same dtype for comparison (handle bf16/fp32 mismatch)
            trained_tensor = trained_state_dict[key].float()
            merged_tensor = merged_state[key].float()
            
            if not torch.allclose(trained_tensor, merged_tensor, rtol=1e-5, atol=1e-8):
                mismatches.append(f"{key} (values differ)")
        
        if mismatches:
            logger.error(f"✗ Weight mismatches found:")
            for mm in mismatches[:5]:  # Show first 5
                logger.error(f"  - {mm}")
            if len(mismatches) > 5:
                logger.error(f"  ... and {len(mismatches) - 5} more")
            raise ValueError(f"✗ ERROR: Trained and merged weights do not match!")
        
        logger.info(f"✓ All trained weights correctly merged: {len(trained_state_dict)} parameters verified")
    
    # 3. Parameter count and structure
    base_params = sum(p.numel() for p in base_module.parameters())
    merged_params = sum(p.numel() for p in merged_module.parameters())
    
    if base_params != merged_params:
        raise ValueError(f"✗ ERROR: Parameter count mismatch: base={base_params:,} vs merged={merged_params:,}")
    
    logger.info(f"✓ Parameter count matches: {merged_params:,}")
    logger.info(f"✓✓✓ {component_name} verification PASSED ✓✓✓")


def verify_models_only(
    base_model_path: str,
    merged_model_path: str
) -> None:
    """
    Verify-only mode: Compare base and merged models without performing merge.
    
    Args:
        base_model_path: Path to base model
        merged_model_path: Path to allegedly merged model
    """
    
    logger.info("=== VERIFY-ONLY MODE ===")
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Merged model: {merged_model_path}")
    
    # Load both models
    logger.info("\nLoading base model...")
    base_model = VibeVoiceForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32
    )
    
    logger.info("Loading merged model...")
    merged_model = VibeVoiceForConditionalGeneration.from_pretrained(
        merged_model_path,
        torch_dtype=torch.float32
    )
    
    # Verify each component
    components_to_check = ["diffusion_head", "acoustic_connector", "semantic_connector"]
    
    for component in components_to_check:
        try:
            verify_merge(base_model, merged_model, None, component)
        except ValueError as e:
            if "did not change" in str(e):
                logger.info(f"✓ {component}: unchanged (likely not trained)")
            else:
                raise
        except Exception as e:
            logger.error(f"✗ {component} verification failed: {e}")
            raise
    
    logger.info("\n✓✓✓ VERIFICATION COMPLETE ✓✓✓")


def merge_vibevoice_model(
    base_model_path: str,
    checkpoint_path: str,
    output_path: str,
    output_format: str = "safetensors"
) -> None:
    """
    Universal merge function for VibeVoice models.
    Automatically detects and merges all trained components.
    """
    
    # Detect what was trained
    logger.info(f"Scanning trained components in: {checkpoint_path}")
    components = detect_trained_components(checkpoint_path)
    
    logger.info("Detected trained components:")
    for name, trained in components.items():
        status = "✓ Found" if trained else "✗ Not found"
        logger.info(f"  {name}: {status}")
    
    if not any(components.values()):
        raise ValueError("No trained components found in checkpoint path!")
    
    # Load base model
    logger.info(f"\nLoading base model from: {base_model_path}")
    base_model = VibeVoiceForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32
    )
    
    # Merge components
    logger.info("\n=== Starting merge process ===")
    
    trained_diffusion_state = None
    
    if components["llm_lora"]:
        merge_llm_lora(base_model, checkpoint_path)
    
    if components["diffusion_head"]:
        trained_diffusion_state = merge_diffusion_head(base_model, checkpoint_path)
    
    if components["acoustic_connector"] or components["semantic_connector"]:
        merge_connectors(
            base_model, 
            checkpoint_path, 
            merge_acoustic=components["acoustic_connector"],
            merge_semantic=components["semantic_connector"]
        )
    
    # Save merged model
    logger.info(f"\n=== Saving merged model to: {output_path} ===")
    os.makedirs(output_path, exist_ok=True)
    
    if output_format == "safetensors":
        base_model.save_pretrained(output_path, safe_serialization=True)
    elif output_format == "bin":
        base_model.save_pretrained(output_path, safe_serialization=False)
    else:
        raise ValueError(f"Unknown output format: {output_format}. Use 'safetensors' or 'bin'")
    
    # Copy config and processor files
    logger.info("Copying config and processor files...")
    files_to_copy = [
        "config.json",
        "preprocessor_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt"
    ]
    
    for file in files_to_copy:
        src = os.path.join(base_model_path, file)
        dst = os.path.join(output_path, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    # Verification
    logger.info("\n=== Verifying merged model ===")
    try:
        # Reload original base model for comparison (the base_model variable was modified during merge)
        logger.info("Reloading original base model for verification...")
        original_base_model = VibeVoiceForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32
        )
        
        logger.info("Loading merged model for verification...")
        test_model = VibeVoiceForConditionalGeneration.from_pretrained(output_path)
        logger.info("✓ Model loads successfully")
        
        # Detailed verification for each merged component
        if components["diffusion_head"]:
            verify_merge(original_base_model, test_model, trained_diffusion_state, "diffusion_head")
        
        if components["acoustic_connector"]:
            verify_merge(original_base_model, test_model, None, "acoustic_connector")
        
        if components["semantic_connector"]:
            verify_merge(original_base_model, test_model, None, "semantic_connector")
        
        logger.info("\n✓✓✓ Merge and verification completed successfully! ✓✓✓")
        
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Universal merger for VibeVoice trained components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge and verify
  python merge_vibevoice_models.py --base_model_path model --checkpoint_path output/lora --output_path merged
  
  # Verify existing merge (no actual merging)
  python merge_vibevoice_models.py --base_model_path model --output_path merged --verify_only
        """
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to base VibeVoice model directory"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to checkpoint directory (usually 'lora/' or 'checkpoint-XXX/lora/'). Not needed with --verify_only"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged model (or path to verify with --verify_only)"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="safetensors",
        choices=["safetensors", "bin"],
        help="Output format: 'safetensors' (recommended) or 'bin'"
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing merge between base_model_path and output_path (no actual merging)"
    )
    
    args = parser.parse_args()
    
    # Verify-only mode
    if args.verify_only:
        verify_models_only(
            base_model_path=args.base_model_path,
            merged_model_path=args.output_path
        )
        return
    
    # Normal merge mode
    if not args.checkpoint_path:
        parser.error("--checkpoint_path is required unless using --verify_only")
    
    merge_vibevoice_model(
        base_model_path=args.base_model_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        output_format=args.output_format
    )


if __name__ == "__main__":
    main()