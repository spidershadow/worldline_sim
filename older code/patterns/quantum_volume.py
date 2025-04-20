from __future__ import annotations

import math
import numpy as np
from pathlib import Path
import pandas as pd

from worldline_sim.patterns.base import Pattern

__all__ = ["QuantumVolumePattern"]


class QuantumVolumePattern(Pattern):
    """Best reported Quantum Volume per year (IBM metric)."""

    def __init__(self, *, t_star=None):
        super().__init__("quantum_volume", t_star=t_star)

        # Fallback: if standard observations file missing look for quantum_volume.csv
        if not self.observed:
            alt = Path(__file__).parents[1] / "data" / "quantum_volume.csv"
            if alt.exists():
                try:
                    df = pd.read_csv(alt)
                    col = "quantum_volume" if "quantum_volume" in df.columns else "value"
                    if {"year", col}.issubset(df.columns):
                        self.observed = dict(zip(df["year"], df[col]))
                except Exception:
                    pass

        max_obs = max(self.observed.values(), default=16)
        self.L = max_obs * 8  # allow several doublings
        self.k = 0.3
        mid = (min(self.observed, default=2017) + self.T_STAR) / 2
        self.x0 = mid

        self.leak_lambda, self.leak_tau0 = 0.0, 1.0

    def sample(self, year: int, timeline):
        if year in self.observed:
            return self.observed[year]
        baseline = self.L / (1 + math.exp(-self.k * (year - self.x0)))
        noise = self._rng_instance().lognormal(mean=0.0, sigma=0.15)
        return baseline * noise 