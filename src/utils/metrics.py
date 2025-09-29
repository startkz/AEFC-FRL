from typing import List, Dict, Tuple
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def params_to_vector(model) -> torch.Tensor:
    return parameters_to_vector(model.parameters()).detach().cpu()

def vector_to_params(vec: torch.Tensor, model):
    vector_to_parameters(vec.to(next(model.parameters()).device), model.parameters())

def dense_to_sparse(delta: torch.Tensor, k_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k magnitude sparsification."""
    n = delta.numel()
    k = max(1, int(k_ratio * n))
    _, idx = torch.topk(delta.abs(), k)
    return idx, delta[idx]

def sparse_to_dense(indices: torch.Tensor, values: torch.Tensor, n_total: int) -> torch.Tensor:
    out = torch.zeros(n_total, dtype=values.dtype)
    out[indices] = values
    return out

def compute_comm_percent(updates: List[Dict], dense_lengths: List[int]) -> float:
    """
    Report per-round comm cost as % of sending full dense vector per client.
    Assumes float32 for values and int64 for indices for sparse payloads.
    """
    total_full = float(sum(dense_lengths))  # baseline units
    sent = 0.0
    for u in updates:
        if u["type"] == "dense":
            sent += float(u["num_params"])
        else:
            # count values as 1 unit each; indices cheaper/more expensive in bytes,
            # but we keep a simple ratio consistent with paper's % reporting.
            sent += float(len(u["values"]))  # dominant term
    return 100.0 * sent / total_full if total_full > 0 else 0.0

class moving_average:
    def __init__(self, window=5):
        self.w = window
        self.buf = []

    def add(self, x: float):
        self.buf.append(x)
        if len(self.buf) > self.w:
            self.buf.pop(0)

    def mean(self):
        return float(np.mean(self.buf)) if self.buf else 0.0
