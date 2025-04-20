__all__ = [
    "base",
    "uap",
    "tech",
    "rng",
    "enhanced_rng",
]

from importlib import import_module
from pathlib import Path


def load_patterns(names: str | list[str], *, t_star: int | None = None):
    """Return instantiated Pattern objects for *names*.

    Example
    -------
    >>> patterns = load_patterns("uap,tech")
    """
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]

    patterns = []
    here = Path(__file__).parent
    for n in names:
        mod = import_module(f"worldline_sim.patterns.{n}")  # noqa: WPS421
        cls_name = next(
            (
                c
                for c in dir(mod)
                if c.lower().replace("_", "").startswith(n.replace("_", ""))
                and c.endswith("Pattern")
            ),
            None,
        )
        if cls_name is None:
            raise ValueError(f"Pattern class not found in module {n}")
        cls = getattr(mod, cls_name)
        patterns.append(cls(t_star=t_star))
    return patterns 