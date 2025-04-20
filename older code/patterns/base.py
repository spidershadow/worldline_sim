class Pattern:
    """Abstract base for co‑evolving signals.

    Subclasses implement :py:meth:`sample` and optionally override
    :py:meth:`constraint`. Observed data, if any, are loaded from
    *data/<name>_observations.csv* at instantiation time (two columns:
    *year,value*).
    """

    # Default horizon year for the hypothetical Singularity event.  When
    # *t_star* is passed to the constructor this class attribute is
    # overridden on the instance so each simulation run can adopt its
    # own candidate Singularity year.
    T_STAR = 2085

    #: Allowed absolute error when enforcing agreement with observations.
    TOL = 1e-3

    def __init__(self, name: str, *, t_star: int | None = None, data_dir=None):
        from pathlib import Path
        import pandas as pd

        # Allow callers to customise the horizon.
        self.T_STAR = int(t_star) if t_star is not None else self.__class__.T_STAR

        self.name = name
        self._rng = None  # lazily created NumPy Generator
        self.observed: dict[int, float] = {}

        data_dir = Path(data_dir or Path(__file__).parents[1] / "data")
        csv_path = data_dir / f"{name}_observations.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if {"year", "value"}.issubset(df.columns):
                    self.observed = dict(
                        zip(df["year"].astype(int), df["value"].astype(float))
                    )
            except Exception as exc:  # pragma: no cover – diagnostics only
                print(f"[Pattern] Failed to read {csv_path}: {exc}")

    # ------------------------------------------------------------------
    def sample(self, year: int, timeline: "Timeline | None") -> float:  # noqa: F821
        """Return a simulated value for *year*.

        When *year* is present in :pyattr:`observed`, the method **must**
        return the observation verbatim. Concrete subclasses should call
        :py:meth:`_observed_or` to handle this requirement.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    def constraint(self, year: int, value: float, *, tol: float | None = None) -> bool:
        """Return *True* when *value* satisfies the hard constraints."""
        tol = tol or self.TOL
        if year in self.observed:
            return abs(value - self.observed[year]) <= tol
        return True

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def _observed_or(self, year: int, default: float) -> float:
        """Return observation for *year* if present, else *default*."""
        return self.observed.get(year, default)

    def _rng_instance(self, seed=None):
        """Return a per‑pattern ``numpy.random.Generator`` instance."""
        import numpy as np

        if seed is not None:
            # Always create a new RNG when seed is provided
            return np.random.default_rng(seed)
        
        if self._rng is None:
            self._rng = np.random.default_rng(hash(self.name) & 0xFFFF_FFFF)
        return self._rng
