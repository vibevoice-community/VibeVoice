"""
VibeVoice Modular Components

This module provides the core model architectures for VibeVoice:
- Multi-speaker models (1.5B, 7B) for high-quality multi-speaker TTS
- Streaming model (0.5B) for real-time low-latency TTS
"""

# Multi-speaker model components
from .configuration_vibevoice import (
    VibeVoiceConfig,
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceSemanticTokenizerConfig,
    VibeVoiceDiffusionHeadConfig,
)
from .modeling_vibevoice import (
    VibeVoicePreTrainedModel,
    VibeVoiceModel,
)
from .modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)

# Streaming model components (0.5B)
from .configuration_vibevoice_streaming import (
    VibeVoiceStreamingConfig,
)
from .modeling_vibevoice_streaming import (
    VibeVoiceStreamingPreTrainedModel,
    VibeVoiceStreamingModel,
    BinaryClassifier,
    SpeechConnector,
)
from .modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
    VibeVoiceGenerationOutput,
    TTS_TEXT_WINDOW_SIZE,
    TTS_SPEECH_WINDOW_SIZE,
)

# Shared components
from .modular_vibevoice_tokenizer import (
    VibeVoiceTokenizerStreamingCache,
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceSemanticTokenizerModel,
)
from .modular_vibevoice_text_tokenizer import (
    VibeVoiceTextTokenizer,
    VibeVoiceTextTokenizerFast,
)
from .modular_vibevoice_diffusion_head import (
    VibeVoiceDiffusionHead,
)
from .streamer import (
    AudioStreamer,
    AsyncAudioStreamer,
)

# LoRA support
from .lora_loading import (
    load_lora_assets,
)

__all__ = [
    # Multi-speaker configs
    "VibeVoiceConfig",
    "VibeVoiceAcousticTokenizerConfig",
    "VibeVoiceSemanticTokenizerConfig",
    "VibeVoiceDiffusionHeadConfig",
    # Multi-speaker models
    "VibeVoicePreTrainedModel",
    "VibeVoiceModel",
    "VibeVoiceForConditionalGenerationInference",
    # Streaming configs
    "VibeVoiceStreamingConfig",
    # Streaming models
    "VibeVoiceStreamingPreTrainedModel",
    "VibeVoiceStreamingModel",
    "VibeVoiceStreamingForConditionalGenerationInference",
    "VibeVoiceGenerationOutput",
    "BinaryClassifier",
    "SpeechConnector",
    "TTS_TEXT_WINDOW_SIZE",
    "TTS_SPEECH_WINDOW_SIZE",
    # Shared components
    "VibeVoiceTokenizerStreamingCache",
    "VibeVoiceAcousticTokenizerModel",
    "VibeVoiceSemanticTokenizerModel",
    "VibeVoiceTextTokenizer",
    "VibeVoiceTextTokenizerFast",
    "VibeVoiceDiffusionHead",
    "AudioStreamer",
    "AsyncAudioStreamer",
    # LoRA
    "load_lora_assets",
]
