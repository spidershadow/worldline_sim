from __future__ import annotations

import numpy as np

from worldline_sim.patterns.base import Pattern

__all__ = ["TechPattern"]


class TechPattern(Pattern):
    """Technology acceleration proxy (e.g. FLOPS cost) with hyperbolic trend.

    The growth rate amplifies with *uap* count – representing speculative
    coupling between disclosure events and technological breakthroughs.
    """

    def __init__(self, *, t_star=None):
        super().__init__("tech", t_star=t_star)
        # Baseline hyperbolic params
        self.a = 1.0  # coefficient
        self.b = 2.0  # exponent (hyperbolic >1 faster than linear)

    # ------------------------------------------------------------------
    def sample(self, year: int, timeline):
        # Observed value shortcut
        if year in self.observed:
            return self.observed[year]

        # Coupling: use UAP count if available else 1
        uap_val = 1.0
        if timeline and "uap" in timeline.data:
            idx = timeline.years.index(year)
            uap_val = timeline.data["uap"][idx]
        # Hyperbolic growth towards T_STAR (distance shrinks denominator)
        dist = max(self.T_STAR - year, 1)
        base = self.a * ((self.T_STAR / dist) ** self.b)
        # Amplify by log(1+uap)
        value = base * np.log1p(uap_val)
        return value

    # ------------------------------------------------------------------
    def constraint(self, year: int, value: float, *, tol: float | None = None) -> bool:
        # Tech proxy must be positive & obey observations.
        return value > 0 and super().constraint(year, value, tol=tol) 