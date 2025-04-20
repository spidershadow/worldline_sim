from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from worldline_sim.patterns.base import Pattern

__all__ = [
    "Timeline",
    "sample_until_valid",
]


@dataclass
class Timeline:
    """Container holding simulated data for a set of *patterns* over *years*."""

    years: list[int]
    data: dict[str, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self):
        """Ensure arrays have proper length and shape."""
        for k, arr in list(self.data.items()):
            if len(arr) != len(self.years):
                raise ValueError(f"Data length mismatch for {k}")
            self.data[k] = np.asarray(arr, dtype=float)

    # ------------------------------------------------------------------
    def is_valid(self, patterns: Iterable[Pattern]) -> bool:
        """Return True when *all* pattern constraints are satisfied."""
        for p in patterns:
            series = self.data.get(p.name)
            if series is None:
                return False
            for idx, yr in enumerate(self.years):
                if not p.constraint(yr, series[idx]):
                    return False
        return True

    # ------------------------------------------------------------------
    def to_frame(self):
        """Return a *pandas.DataFrame* representation (years as index)."""
        import pandas as pd

        return pd.DataFrame(self.data, index=self.years)


# ----------------------------------------------------------------------

def sample_until_valid(patterns: list[Pattern], *, tries: int = 2000) -> Timeline:
    """Generate timelines until *is_valid* or *tries* exhausted."""

    # Determine global year range from observations, else 1950‑T_STAR.
    all_years = set()
    for p in patterns:
        all_years.update(p.observed)
    if not all_years:
        all_years.update(range(1950, Pattern.T_STAR + 1))
    years = sorted(all_years)

    for attempt in range(tries):
        data: dict[str, np.ndarray] = {}
        timeline_stub = Timeline(years, {})
        for p in patterns:
            series = []
            for yr in years:
                val = p.sample(yr, timeline_stub)
                series.append(val)
            data[p.name] = np.asarray(series)
            timeline_stub.data[p.name] = data[p.name]
        tl = Timeline(years, data)
        if tl.is_valid(patterns):
            return tl
    raise RuntimeError(f"Failed to obtain valid timeline after {tries} tries") 