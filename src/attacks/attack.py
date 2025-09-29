from typing import Dict
import numpy as np
import torch

def _random_like(x: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(x) * x.std().clamp_min(1e-6)

def apply_attack_to_update(update: Dict, attack_type: str, is_malicious: bool) -> Dict:
    """
    Modify client update in-place if malicious according to chosen attack.
    - random: replace delta with noise
    - poison: sign-flip and scale
    - reward_flip: already applied in env; no change here
    """
    if not is_malicious or attack_type in ("none", "reward_flip"):
        return update

    if update["type"] == "dense":
        delta = update["delta"]
        if attack_type == "random":
            update["delta"] = _random_like(delta)
        elif attack_type == "poison":
            update["delta"] = -5.0 * delta  # scale+flip
        return update

    if update["type"] == "sparse":
        values = update["values"]
        if attack_type == "random":
            update["values"] = torch.randn_like(values) * values.std().clamp_min(1e-6)
        elif attack_type == "poison":
            update["values"] = -5.0 * values
        return update

    return update
