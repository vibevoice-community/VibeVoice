from transformers import TrainerCallback
import torch
import copy

class EmaCallback(TrainerCallback):
    """
    EMA callback for VibeVoice finetuning.
    """
    def __init__(self, attr_path="model.prediction_head", decay=0.999, device="cpu"):
        """
        attr_path: where the head lives under self.model (Trainer wraps your VibeVoiceForConditionalGeneration)
        decay:     EMA decay (0.999 ~ stable, 0.9999 ~ very smooth, slower to adapt)
        """
        self.attr_path = attr_path
        self.decay = float(decay)
        self.device = torch.device(device)
        self.shadow = None
        self._orig = None  # store non-EMA weights when we swap

    def _get_module(self, model):
        # Resolve dotted path like "model.prediction_head"
        mod = model
        for name in self.attr_path.split("."):
            mod = getattr(mod, name)
        return mod

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        head = self._get_module(model)
        self.shadow = {
            k: p.detach().to(self.device).clone() for k, p in head.state_dict().items()
        }

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.shadow is None:
            return
        head = self._get_module(model)
        with torch.no_grad():
            for k, v in head.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(
                    v.detach().to(self.device), alpha=(1.0 - self.decay)
                )

    # ---- Swap helpers ----
    def _swap_in_ema(self, model):
        head = self._get_module(model)
        self._orig = copy.deepcopy(head.state_dict())
        head.load_state_dict(self.shadow, strict=False)

    def _swap_back(self, model):
        if self._orig is None:
            return
        head = self._get_module(model)
        head.load_state_dict(self._orig, strict=False)
        self._orig = None

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # use EMA during eval
        self._swap_in_ema(model)

    def on_evaluate_end(self, args, state, control, model=None, **kwargs):
        self._swap_back(model)

    def on_save(self, args, state, control, model=None, **kwargs):
        # temporarily swap to EMA, let Trainer save, then swap back
        self._swap_in_ema(model)

    def on_save_end(self, args, state, control, model=None, **kwargs):
        self._swap_back(model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # final checkpoint: persist EMA
        self._swap_in_ema(model)
