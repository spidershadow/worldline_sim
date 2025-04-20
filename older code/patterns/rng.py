from __future__ import annotations

import numpy as np

from worldline_sim.patterns.base import Pattern

__all__ = ["RngPattern"]


class RngPattern(Pattern):
    """Pseudo‑randomness entropy deviation (e.g. REG outputs).

    Simple linear ramp: zero mean in 1950, maximal anomaly at T_STAR.
    """

    def __init__(self, *, t_star=None):
        super().__init__("rng", t_star=t_star)
        self.slope = 1.0 / (self.T_STAR - 1950)

    # ------------------------------------------------------------------
    def sample(self, year: int, timeline):
        if year in self.observed:
            return self.observed[year]
        rng = self._rng_instance()
        base = self.slope * (year - 1950)
        noise = rng.normal(scale=0.05)
        return base + noise

    # ------------------------------------------------------------------
    def constraint(self, year: int, value: float, *, tol: float | None = None) -> bool:
        return super().constraint(year, value, tol=tol) 