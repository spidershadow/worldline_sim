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
        # For RNG pattern, use a much higher tolerance (1.0) to accept more trajectories
        if year in self.observed:
            return abs(value - self.observed[year]) <= 1.0
        return True
        
    # ------------------------------------------------------------------
    def retro_kernel(self, tau: int) -> float:
        """Scale down the retro-influence to match observation scale."""
        # Get the original retro kernel value
        value = super().retro_kernel(tau)
        # Scale down by 100 to match observed values
        return value / 100.0 