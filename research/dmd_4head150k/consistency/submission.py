"""Opt-in state marker; old progressive state dictionaries remain unchanged."""
import torch


def prepare_state(model, state_dict) -> None:
    key = "consistency_sigma_data"
    if key in state_dict:
        value = state_dict[key]
        if value.ndim != 0 or not torch.isfinite(value) or value <= 0:
            raise ValueError("Invalid consistency sigma_data marker")
        if not hasattr(model, key):
            model.register_buffer(key, torch.tensor(0.5, device=model.alpha_bar.device))
    elif hasattr(model, key):
        delattr(model, key)
