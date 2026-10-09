"""Executable reduced-order CPS backend used for mechanism integration tests.
It is NOT an IEEE-39 replacement and Results generated from it are labeled
mechanism-level in the manuscript pipeline.
"""
import numpy as np

class ReducedOrderCPS:
    def __init__(self, dim=4, seed=0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros(dim)
        self.target = np.zeros(dim)
        self.t = 0

    def reset(self):
        self.x = self.rng.normal(0, 0.025, self.dim)
        self.t = 0
        return self.x.copy()

    def drift(self):
        A = np.array([[-.11,.03,0,0],[.02,-.08,.02,0],[0,.02,-.10,.03],[0,0,.02,-.09]])
        d = A @ self.x
        if 18 <= self.t < 34:
            d += np.array([0.090,-0.080,0.075,-0.070])
        return d

    def step(self, executed_action):
        u = np.asarray(executed_action, dtype=float)
        noise = self.rng.normal(0, 0.003, self.dim)
        self.x = self.x + self.drift() + 0.24*u + noise
        self.t += 1
        return self.x.copy()

    def clean_observation(self):
        return self.x + self.rng.normal(0,0.004,self.dim)

    def safe(self, limit=0.35):
        return bool(np.all(np.abs(self.x) <= limit))

    def recovered(self, tol=0.08):
        return bool(np.linalg.norm(self.x, ord=np.inf) <= tol)
