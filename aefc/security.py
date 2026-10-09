"""Security mechanisms for bounded-authority federated recovery.

The module is simulator-agnostic. It intentionally separates recovery evidence,
authorization, deployment and execution so each paper claim has an executable
counterpart and a directly logged diagnostic.
"""
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import numpy as np


def _asvec(x):
    return np.asarray(x, dtype=float).reshape(-1)


def trimmed_mean(values: Sequence[np.ndarray], f: int) -> np.ndarray:
    """Coordinate-wise f-trimmed mean on a common coordinate population."""
    x = np.vstack([_asvec(v) for v in values])
    n = x.shape[0]
    if f < 0 or 2 * f >= n:
        raise ValueError(f"trim budget f={f} invalid for n={n}")
    xs = np.sort(x, axis=0)
    if f:
        xs = xs[f:n-f]
    return xs.mean(axis=0)


@dataclass
class Evidence:
    value: np.ndarray
    source: str
    timestamp: int
    model_version: int
    lineage: str
    provenance_valid: bool = True


@dataclass
class PATBUConfig:
    beta_age: float = 0.08
    residual_scale: float = 0.25
    lambda_b: float = 0.72
    obs_gain: float = 0.20
    knowledge_gain: float = 0.08
    trim_f: int = 1
    belief_low: float = -2.0
    belief_high: float = 2.0


class ProvenanceAwareTemporalBelief:
    def __init__(self, dim: int, config: PATBUConfig):
        self.config = config
        self.belief = np.zeros(dim, dtype=float)

    def _weight(self, ev: Evidence, now: int) -> float:
        if not ev.provenance_valid:
            return 0.0
        age = max(0, now - int(ev.timestamp))
        age_term = np.exp(-self.config.beta_age * age)
        residual = np.linalg.norm(_asvec(ev.value) - self.belief)
        consistency = 1.0 / (1.0 + (residual / max(self.config.residual_scale, 1e-9)) ** 2)
        return float(age_term * consistency)

    def update(self, observation: np.ndarray, evidence: Sequence[Evidence], now: int) -> Tuple[np.ndarray, dict]:
        obs = _asvec(observation)
        weighted, weights = [], []
        for ev in evidence:
            w = self._weight(ev, now)
            weights.append(w)
            weighted.append(w * _asvec(ev.value))
        if weighted:
            f = min(self.config.trim_f, max(0, (len(weighted) - 1) // 2))
            k = trimmed_mean(weighted, f=f)
        else:
            k = np.zeros_like(self.belief)
        nxt = (self.config.lambda_b * self.belief +
               self.config.obs_gain * obs +
               self.config.knowledge_gain * k)
        nxt = np.clip(nxt, self.config.belief_low, self.config.belief_high)
        self.belief = nxt
        trust = float(np.clip(np.mean(weights) if weights else 0.0, 0.0, 1.0))
        return nxt.copy(), {"trust": trust, "weights": weights, "knowledge": k.copy()}


@dataclass
class GateConfig:
    rho_max: float = 0.55
    risk_margin: float = 0.05
    q_min: float = 0.18
    lambda_safe: float = 0.62
    lambda_recovery: float = 0.16
    lambda_belief: float = 0.14
    lambda_provenance: float = 0.08


class RiskAwareRecoveryGate:
    def __init__(self, config: GateConfig):
        self.config = config

    def assess(self, safety_shortfall: float, recovery_shortfall: float,
               belief_uncertainty: float, trust: float) -> Tuple[bool, float]:
        c = self.config
        risk = (c.lambda_safe * max(0.0, safety_shortfall) +
                c.lambda_recovery * max(0.0, recovery_shortfall) +
                c.lambda_belief * max(0.0, belief_uncertainty) +
                c.lambda_provenance * (1.0 - np.clip(trust, 0.0, 1.0)))
        allow = (risk <= c.rho_max - c.risk_margin) and (trust >= c.q_min)
        return bool(allow), float(risk)


@dataclass
class AdaptConfig:
    eta: float = 0.04
    clip_norm: float = 0.20
    trust_region: float = 0.035
    q_min: float = 0.18
    validation_slack: float = 0.025


class TrustGatedAdaptation:
    def __init__(self, dim: int, config: AdaptConfig):
        self.config = config
        self.deployed = np.zeros(dim, dtype=float)
        self.certified_checkpoint = self.deployed.copy()

    def propose(self, gradient: np.ndarray, trust: float) -> np.ndarray:
        g = _asvec(gradient)
        norm = np.linalg.norm(g)
        if norm > self.config.clip_norm > 0:
            g = g * (self.config.clip_norm / norm)
        delta = -self.config.eta * np.clip(trust, 0.0, 1.0) * g
        dn = np.linalg.norm(delta)
        if dn > self.config.trust_region > 0:
            delta = delta * (self.config.trust_region / dn)
        return self.deployed + delta

    def deploy(self, candidate: np.ndarray, trust: float, gate_allowed: bool,
               validation_before: float, validation_after: float) -> Tuple[np.ndarray, dict]:
        candidate = _asvec(candidate)
        validation_ok = validation_after <= validation_before + self.config.validation_slack
        accepted = bool(trust >= self.config.q_min and gate_allowed and validation_ok)
        before = self.deployed.copy()
        if accepted:
            self.deployed = candidate.copy()
        step = float(np.linalg.norm(self.deployed - before))
        return self.deployed.copy(), {"accepted": accepted, "validation_ok": validation_ok, "step_norm": step}

    def rollback(self):
        self.deployed = self.certified_checkpoint.copy()


@dataclass
class ShieldConfig:
    state_limit: float = 0.35
    post_attack_budget: float = 0.10
    action_limit: float = 0.80
    dynamics_gain: float = 0.24
    state_disturbance_budget: float = 0.006


class RobustBoxShield:
    """Closed-form robust shield for the executable reduced-order test backend."""
    def __init__(self, config: ShieldConfig):
        self.config = config

    def project(self, state: np.ndarray, proposal: np.ndarray, drift: np.ndarray) -> Tuple[np.ndarray, dict]:
        x = _asvec(state)
        u0 = np.clip(_asvec(proposal), -self.config.action_limit, self.config.action_limit)
        d = _asvec(drift)
        g = self.config.dynamics_gain
        b = self.config.post_attack_budget
        w = self.config.state_disturbance_budget
        L = self.config.state_limit
        lower = (-L + w - x - d) / g + b
        upper = ( L - w - x - d) / g - b
        feasible = bool(np.all(lower <= upper))
        if feasible:
            u = np.minimum(np.maximum(u0, lower), upper)
            u = np.clip(u, -self.config.action_limit, self.config.action_limit)
        else:
            u = np.clip(-(x + d) / max(g, 1e-9), -self.config.action_limit, self.config.action_limit)
        corrected = float(np.linalg.norm(u-u0))
        robust_margin = float(np.min(L - (np.abs(x + d + g*u) + g*b + w)))
        return u, {"feasible": feasible, "intervention_norm": corrected, "robust_margin": robust_margin}
