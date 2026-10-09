from dataclasses import dataclass
import numpy as np

@dataclass
class AttackConfig:
    start: int = 24
    end: int = 72
    observation_eps: float = 0.32
    control_eps: float = 0.10
    byzantine_fraction: float = 0.30
    model_replacement_scale: float = 6.0
    stale_delay: int = 7


def active(t: int, cfg: AttackConfig) -> bool:
    return cfg.start <= t < cfg.end


def observation_attack(dim: int, t: int, rng, cfg: AttackConfig, enabled: bool):
    if not enabled or not active(t, cfg): return np.zeros(dim)
    sign = np.where(np.arange(dim) % 2 == 0, 1.0, -1.0)
    return sign * cfg.observation_eps * (0.75 + 0.25*rng.random(dim))


def post_shield_attack(dim: int, t: int, rng, cfg: AttackConfig, enabled: bool):
    if not enabled or not active(t, cfg): return np.zeros(dim)
    return rng.uniform(-cfg.control_eps, cfg.control_eps, size=dim)


def poison_update(update, t: int, rng, cfg: AttackConfig, enabled: bool):
    u = np.asarray(update, dtype=float).copy()
    if not enabled or not active(t, cfg): return u
    return -cfg.model_replacement_scale*u + rng.normal(0, 0.015, size=u.shape)
