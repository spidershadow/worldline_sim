from __future__ import annotations

import math

import numpy as np

from worldline_sim.patterns.base import Pattern

__all__ = ["EventsPattern"]


class EventsPattern(Pattern):
    """Yearly count of formally registered *global events*.

    We model the frequency of globally salient events listed in the GCP
    formal registry.  Historically the count trends upward slowly, so we
    use a mild logistic curve (bounded growth) with log‑normal noise.
    """

    def __init__(self, *, t_star=None):
        super().__init__("events", t_star=t_star)

        # Logistic parameters based on observed range.
        max_obs = max(self.observed.values(), default=20)
        self.L = max_obs * 3           # asymptotic upper bound
        self.k = 0.05                  # growth rate
        # Midpoint halfway between first data year and T_STAR.
        first_year = min(self.observed, default=1998)
        self.x0 = (first_year + self.T_STAR) / 2

    # ------------------------------------------------------------------
    def sample(self, year: int, timeline):
        if year in self.observed:
            return self.observed[year]

        # Logistic baseline growth.
        baseline = self.L / (1 + math.exp(-self.k * (year - self.x0)))

        # Add multiplicative noise.
        rng = self._rng_instance()
        noise_factor = rng.lognormal(mean=0.0, sigma=0.1)
        return baseline * noise_factor

    # ------------------------------------------------------------------
    def constraint(self, year: int, value: float, *, tol: float | None = None) -> bool:
        # Non‑negative integerish counts.
        return value >= 0 and super().constraint(year, value, tol=tol) 