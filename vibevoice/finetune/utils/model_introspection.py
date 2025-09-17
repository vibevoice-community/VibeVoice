"""
Model introspection utilities to reduce hasattr boilerplate.
"""

import logging
from typing import Any, Callable, Dict, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelComponents:
    """Clean interface for accessing model components."""

    def __init__(self, model):
        self.model = model
        self._cache = {}

    @property
    def base_model(self):
        """Get the base model (model.model)."""
        if "base_model" not in self._cache:
            self._cache["base_model"] = getattr(self.model, "model", None)
        return self._cache["base_model"]

    @property
    def language_model(self):
        """Get the language model component."""
        if "language_model" not in self._cache:
            base = self.base_model
            self._cache["language_model"] = (
                getattr(base, "language_model", None) if base else None
            )
        return self._cache["language_model"]

    @property
    def prediction_head(self):
        """Get the prediction head component."""
        if "prediction_head" not in self._cache:
            base = self.base_model
            self._cache["prediction_head"] = (
                getattr(base, "prediction_head", None) if base else None
            )
        return self._cache["prediction_head"]

    @property
    def acoustic_tokenizer(self):
        """Get the acoustic tokenizer component."""
        if "acoustic_tokenizer" not in self._cache:
            base = self.base_model
            self._cache["acoustic_tokenizer"] = (
                getattr(base, "acoustic_tokenizer", None) if base else None
            )
        return self._cache["acoustic_tokenizer"]

    @property
    def semantic_tokenizer(self):
        """Get the semantic tokenizer component."""
        if "semantic_tokenizer" not in self._cache:
            base = self.base_model
            self._cache["semantic_tokenizer"] = (
                getattr(base, "semantic_tokenizer", None) if base else None
            )
        return self._cache["semantic_tokenizer"]

    @property
    def acoustic_connector(self):
        """Get the acoustic connector component."""
        if "acoustic_connector" not in self._cache:
            base = self.base_model
            self._cache["acoustic_connector"] = (
                getattr(base, "acoustic_connector", None) if base else None
            )
        return self._cache["acoustic_connector"]

    @property
    def semantic_connector(self):
        """Get the semantic connector component."""
        if "semantic_connector" not in self._cache:
            base = self.base_model
            self._cache["semantic_connector"] = (
                getattr(base, "semantic_connector", None) if base else None
            )
        return self._cache["semantic_connector"]

    @property
    def input_embeddings(self):
        """Get input embeddings."""
        if "input_embeddings" not in self._cache:
            self._cache["input_embeddings"] = safe_get_attr(
                self.model, "get_input_embeddings", lambda: None
            )()
        return self._cache["input_embeddings"]

    @property
    def output_embeddings(self):
        """Get output embeddings."""
        if "output_embeddings" not in self._cache:
            self._cache["output_embeddings"] = safe_get_attr(
                self.model, "get_output_embeddings", lambda: None
            )()
        return self._cache["output_embeddings"]

    def has_component(self, component_name: str) -> bool:
        """Check if a component exists."""
        return getattr(self, component_name) is not None

    def apply_to_component(self, component_name: str, func: Callable, *args, **kwargs):
        """Apply a function to a component if it exists."""
        component = getattr(self, component_name)
        if component is not None:
            return func(component, *args, **kwargs)
        return None

    def freeze_component(self, component_name: str):
        """Freeze all parameters in a component."""

        def freeze_params(component):
            for p in component.parameters():
                p.requires_grad = False

        self.apply_to_component(component_name, freeze_params)

    def unfreeze_component(self, component_name: str):
        """Unfreeze all parameters in a component."""

        def unfreeze_params(component):
            for p in component.parameters():
                p.requires_grad = True

        self.apply_to_component(component_name, unfreeze_params)

    def count_parameters(
        self, component_name: str = None, trainable_only: bool = True
    ) -> int:
        """Count parameters in a component or the entire model."""
        if component_name:
            component = getattr(self, component_name)
            if component is None:
                return 0
            params = component.parameters()
        else:
            params = self.model.parameters()

        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        else:
            return sum(p.numel() for p in params)


def safe_get_attr(obj: Any, attr_name: str, default: Any = None) -> Any:
    """Safely get an attribute with a default value."""
    return getattr(obj, attr_name, default)


def safe_call_method(
    obj: Any, method_name: str, *args, default: Any = None, **kwargs
) -> Any:
    """Safely call a method if it exists."""
    method = getattr(obj, method_name, None)
    if method and callable(method):
        try:
            return method(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to call {method_name}: {e}")
            return default
    return default


def get_config_value(config, *path, default=None):
    """Get a nested config value safely."""
    current = config
    for key in path:
        if hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
    return current


def setup_tokenizer_freezing(components: ModelComponents, model_args):
    """Setup tokenizer freezing based on arguments."""
    if model_args.freeze_acoustic_tokenizer:
        components.freeze_component("acoustic_tokenizer")

    if model_args.freeze_semantic_tokenizer:
        components.freeze_component("semantic_tokenizer")


def setup_connector_training(components: ModelComponents, model_args):
    """Setup connector training based on arguments."""
    if getattr(model_args, "train_connectors", False):
        components.unfreeze_component("acoustic_connector")
        components.unfreeze_component("semantic_connector")
    else:
        components.freeze_component("acoustic_connector")
        components.freeze_component("semantic_connector")


def setup_diffusion_head_training(components: ModelComponents, model_args):
    """Setup diffusion head training based on arguments."""
    if getattr(model_args, "train_diffusion_head", False):
        components.unfreeze_component("prediction_head")


def freeze_diffusion_head_layers(components: ModelComponents, model_args):
    """Freeze specific diffusion head layers if configured."""
    if model_args.layers_to_freeze is None or not components.has_component(
        "prediction_head"
    ):
        return

    head_params = list(components.prediction_head.named_parameters())
    try:
        indices_to_freeze = {
            int(x.strip()) for x in model_args.layers_to_freeze.split(",") if x.strip()
        }
        frozen_count = 0
        for i, (name, param) in enumerate(head_params):
            if i in indices_to_freeze:
                param.requires_grad = False
                frozen_count += 1
                logger.info(f"Froze layer [{i}]: {name}")
        logger.info(
            f"Successfully froze {frozen_count} parameter groups in the diffusion head."
        )
    except Exception as e:
        logger.error(f"Could not parse --layers_to_freeze: {e}")
        raise
