from __future__ import annotations

import math

import numpy as np

from worldline_sim.patterns.base import Pattern

__all__ = ["UapPattern"]


class UapPattern(Pattern):
    """Simulate global *UAP sighting count* per year using a logistic prior.

    The curve accelerates toward :pyattr:`Pattern.T_STAR` – reflecting an
    underlying disclosure/manifestation process.  A small log‑normal
    noise term keeps trajectories unique across sampling attempts.
    """

    def __init__(self, *, t_star=None):
        super().__init__("uap", t_star=t_star)

        # Hyper‑parameters for the logistic; chosen heuristically.
        self.L = max(self.observed.values(), default=10.0) * 20  # asymptote
        self.k = 0.1  # growth rate
        # Anchor midpoint around halfway to T_STAR or median observed year.
        if self.observed:
            mid_obs_year = int(np.median(list(self.observed)))
            self.x0 = (mid_obs_year + self.T_STAR) / 2
        else:
            self.x0 = (1950 + self.T_STAR) / 2

    # ------------------------------------------------------------------
    def sample(self, year: int, timeline):  # noqa: D401 – simple signature
        # Always honour observations when present.
        if year in self.observed:
            return self.observed[year]

        # Deterministic logistic baseline.
        baseline = self.L / (1 + math.exp(-self.k * (year - self.x0)))

        # Add mild multiplicative noise.
        rng = self._rng_instance()
        noise_factor = rng.lognormal(mean=0.0, sigma=0.05)
        return baseline * noise_factor

    # ------------------------------------------------------------------
    def constraint(self, year: int, value: float, *, tol: float | None = None) -> bool:
        # Re‑use parent check + non‑negativity.
        if not super().constraint(year, value, tol=tol):
            return False
        return value >= 0 