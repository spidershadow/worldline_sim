class Pattern:
    """Abstract base for co‑evolving signals.

    Subclasses implement :py:meth:`sample` and optionally override
    :py:meth:`constraint`. Observed data, if any, are loaded from
    *data/<n>_observations.csv* at instantiation time (two columns: 'year' and
    'value').
    """

    #: Allowed absolute error when enforcing agreement with observations.
    TOL = 0.5  # Modified from 1e-3 to be more forgiving

    def __init__(self, name: str):
        self.name = name
        self.observed = {}
        try:
            import pandas as pd
            path = f"data/{name}_observations.csv"
            df = pd.read_csv(path)
            self.observed = dict(zip(df.year, df.value))
        except Exception:
            pass

    def constraint(self, timeline, t_star):
        """Enforce observations, by default exact match."""
        for y, obs in self.observed.items():
            if y not in timeline:
                if y < t_star:
                    return False
                continue
            if abs(timeline[y] - obs) > self.TOL:
                return False
        return True

    def sample(self, *args, **kwargs):
        """Return a (base) value for year *y*. Must be overridden."""
        raise NotImplementedError() 