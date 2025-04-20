from __future__ import annotations

import matplotlib.pyplot as plt

from worldline_sim.sim import Timeline

__all__ = ["plot_timeline"]


def plot_timeline(tl: Timeline, *, path: str | None = None):
    """Overlay timeline curves for all patterns and save/show figure."""

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, series in tl.data.items():
        ax.plot(tl.years, series, label=name)
    ax.set_xlabel("Year")
    ax.set_ylabel("Value (arbitrary units)")
    ax.set_title("World‑line Simulation")
    ax.legend()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show(block=False)
    plt.close(fig) 