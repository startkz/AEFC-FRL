import numpy as np

def select_malicious(num_clients: int, frac: float, seed: int = 0):
    """Return set of malicious client indices."""
    m = int(round(frac * num_clients))
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(num_clients), size=m, replace=False)
    return set(idx.tolist())
