from __future__ import annotations

import math
import numpy as np
import pandas as pd
from pathlib import Path

from worldline_sim.patterns.base import Pattern

__all__ = ["MaxQubitsPattern"]


class MaxQubitsPattern(Pattern):
    """Largest *announced* physical‑qubit count per year.

    Observed data come from the Wikipedia timeline.  We model future
    growth with a logistic curve (bounded, hyper‑exponential in early
    phase) plus mild multiplicative noise.
    """

    def __init__(self, *, t_star=None):
        super().__init__("max_qubits", t_star=t_star)

        # Logistic asymptote ~5× max observed so far (heuristic).
        # Fallback to CSV from gen_data/fetch_quantum if observations file missing.
        if not self.observed:
            alt = Path(__file__).parents[1] / "data" / "max_qubits.csv"
            if alt.exists():
                try:
                    df = pd.read_csv(alt)
                    col = "max_qubits" if "max_qubits" in df.columns else "value"
                    if {"year", col}.issubset(df.columns):
                        self.observed = dict(zip(df["year"], df[col]))
                except Exception:
                    pass

        # Store observation boundaries
        self.first_observed_year = min(self.observed.keys()) if self.observed else None
        self.last_observed_year = max(self.observed.keys()) if self.observed else None

        max_obs = max(self.observed.values(), default=100)
        self.L = max_obs * 5
        self.k = 0.2  # growth rate
        # Ensure first_observed_year is not None before using it
        first_year_for_midpoint = self.first_observed_year if self.first_observed_year is not None else 2000
        self.x0 = (first_year_for_midpoint + self.T_STAR) / 2  # midpoint

        # Default leak parameters so retro driver can modify them.
        self.leak_lambda, self.leak_tau0 = 0.0, 1.0

    # ------------------------------------------------------------------
    def sample(self, year: int, timeline):
        # 1. Return observed value if available
        if year in self.observed:
            return self.observed[year]

        # 2. Handle years *after* last observation (future prediction)
        # Use the logistic curve for future years
        if self.last_observed_year is None or year > self.last_observed_year:
            baseline = self.L / (1 + math.exp(-self.k * (year - self.x0)))
            noise = self._rng_instance().lognormal(mean=0.0, sigma=0.1)
            # Ensure non-negative
            return max(0.0, baseline * noise)

        # 3. Handle years *before* first observation
        if self.first_observed_year is not None and year < self.first_observed_year:
            # Return a floor value for pre-observation era instead of logistic curve
            return 1.0

        # 4. Handle years *between* first and last observation (but not matching exactly)
        # This case is unlikely for dense data but possible. Fall back to logistic.
        baseline = self.L / (1 + math.exp(-self.k * (year - self.x0)))
        noise = self._rng_instance().lognormal(mean=0.0, sigma=0.1)
        return max(0.0, baseline * noise) 