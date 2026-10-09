import numpy as np
from .security import trimmed_mean


def common_support(updates, k):
    """Server-derived common support from coordinate-wise median magnitudes."""
    x = np.vstack([np.asarray(u, dtype=float).reshape(-1) for u in updates])
    center = np.median(x, axis=0)
    k = max(1, min(int(k), center.size))
    idx = np.argpartition(np.abs(center), -k)[-k:]
    return np.sort(idx)


def robust_sparse_aggregate(updates, f, k):
    x = [np.asarray(u, dtype=float).reshape(-1) for u in updates]
    support = common_support(x, k)
    vals = [u[support] for u in x]
    agg_sparse = trimmed_mean(vals, f=f)
    agg = np.zeros_like(x[0])
    agg[support] = agg_sparse
    return agg, support
