from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, Tuple

import torch


DEFAULT_DDPM_STEPS = 10
FAST_DDPM_STEPS = 8


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def compile_backend_status() -> str:
    if not module_available("triton"):
        return "skipped: triton not installed"

    try:
        from triton.compiler.compiler import triton_key  # noqa: F401
    except Exception:
        return "skipped: triton backend incompatible with torch.compile"

    return "ok"


def configure_torch_runtime() -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    for fn_name in (
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
        "enable_math_sdp",
    ):
        fn = getattr(torch.backends.cuda, fn_name, None)
        if callable(fn):
            fn(True)

    allow_reduced = getattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp", None)
    if callable(allow_reduced):
        allow_reduced(True)


def preferred_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_attention_implementation(requested: str = "auto") -> Tuple[str, str]:
    requested = (requested or "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available() and module_available("flash_attn"):
            return "flash_attention_2", "flash-attn available"
        if torch.cuda.is_available():
            return "sdpa", "flash-attn not installed; using PyTorch SDPA"
        return "eager", "CUDA unavailable"

    if requested == "flash_attention_2" and not module_available("flash_attn"):
        if torch.cuda.is_available():
            return "sdpa", "flash-attn requested but not installed; fell back to SDPA"
        return "eager", "flash-attn requested but CUDA unavailable; fell back to eager"

    return requested, "user requested"


def is_prequantized_model_dir(model_path: str) -> bool:
    basename = os.path.basename(model_path).lower()
    return "4bit" in basename or os.path.exists(os.path.join(model_path, "quantization_config.json"))


def build_model_load_kwargs(
    model_path: str,
    use_4bit: bool,
    attn_impl: str = "auto",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    configure_torch_runtime()

    resolved_attn, attn_reason = resolve_attention_implementation(attn_impl)
    dtype = preferred_dtype()
    prequantized = is_prequantized_model_dir(model_path)
    should_use_4bit = use_4bit or prequantized

    load_kwargs: Dict[str, Any] = {
        "device_map": ("cuda" if torch.cuda.is_available() else "cpu"),
        "torch_dtype": dtype,
        "attn_implementation": resolved_attn,
    }

    if should_use_4bit and not prequantized:
        if not module_available("bitsandbytes"):
            raise RuntimeError("bitsandbytes is required for 4-bit loading but is not installed.")
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    meta = {
        "attn_impl": resolved_attn,
        "attn_reason": attn_reason,
        "torch_dtype": dtype,
        "is_prequantized": prequantized,
        "is_quantized": should_use_4bit,
        "compile_recommended": (not should_use_4bit and torch.cuda.is_available()),
    }
    return load_kwargs, meta


def maybe_compile_language_model(model: Any, enabled: bool) -> Tuple[Any, str]:
    if not enabled:
        return model, "disabled"

    language_model = getattr(getattr(model, "model", None), "language_model", None)
    if language_model is None:
        return model, "language model not found"

    if getattr(model, "is_quantized", False) or getattr(model, "is_loaded_in_4bit", False):
        return model, "skipped for quantized model"

    backend_status = compile_backend_status()
    if backend_status != "ok":
        return model, backend_status

    try:
        model.model.language_model = torch.compile(language_model, mode="reduce-overhead")
    except Exception as exc:
        return model, f"skipped: torch.compile failed ({exc.__class__.__name__})"
    return model, "compiled"


def maybe_compile_prediction_head(model: Any, enabled: bool) -> Tuple[Any, str]:
    if not enabled:
        return model, 'disabled'

    prediction_head = getattr(getattr(model, 'model', None), 'prediction_head', None)
    if prediction_head is None:
        return model, 'prediction head not found'
    if not module_available('triton'):
        return model, 'skipped: triton not installed'

    compile_fn = getattr(torch, 'compile', None)
    if not callable(compile_fn):
        return model, 'skipped: torch.compile unavailable'

    try:
        compiled_head = compile_fn(prediction_head, mode='reduce-overhead')
        model.model.prediction_head = compiled_head

        parameter = next(prediction_head.parameters())
        config = prediction_head.config
        noisy = torch.zeros(2, config.latent_size, device=parameter.device, dtype=parameter.dtype)
        timesteps = torch.zeros(2, device=parameter.device, dtype=parameter.dtype)
        condition = torch.zeros(2, config.hidden_size, device=parameter.device, dtype=parameter.dtype)
        with torch.inference_mode():
            compiled_head(noisy, timesteps, condition=condition)
        if parameter.device.type == 'cuda':
            torch.cuda.synchronize(parameter.device)
    except Exception as exc:
        model.model.prediction_head = prediction_head
        return model, f'skipped: prediction head compile failed ({exc.__class__.__name__})'
    return model, 'compiled and warmed prediction head'


def format_dtype(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")
