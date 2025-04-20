from __future__ import annotations

import argparse
from pathlib import Path

from worldline_sim.patterns import load_patterns
from worldline_sim.sim import sample_until_valid
from worldline_sim.viz import plot_timeline


def build_argparser():
    p = argparse.ArgumentParser(description="World‑line co‑evolution simulator")
    p.add_argument("--patterns", default="uap,tech,rng", help="comma list of patterns")
    p.add_argument("--tries", type=int, default=2000, help="sampling attempts")
    p.add_argument("--runs", type=int, default=1, help="number of timelines to generate")
    p.add_argument("--tstar-range", nargs=2, type=int, metavar=("MIN", "MAX"),
                   help="sample Singularity (T*) uniformly in [MIN, MAX] per run; default keeps hard‑coded 2085")
    p.add_argument("--plot", action="store_true", help="show/save timeline plot")
    return p


def main(argv=None):  # noqa: D401 – CLI entry
    args = build_argparser().parse_args(argv)
    out_dir = Path(__file__).parent

    collected_ts = []
    weighted_counts = {}
    all_frames = []

    for idx in range(1, args.runs + 1):
        # Decide T* for this run.
        t_star = None
        if args.tstar_range:
            import random
            lo, hi = args.tstar_range
            t_star = random.randint(lo, hi)

        # Fresh pattern instances each iteration to reset RNG seeds.
        patterns = load_patterns(args.patterns, t_star=t_star)
        tl = sample_until_valid(patterns, tries=args.tries)

        # Quality score: how well does timeline hit 2023 (or latest) observations?
        err = 0.0
        for p in patterns:
            if not p.observed:
                continue
            last_year = max(p.observed)
            idx = tl.years.index(last_year)
            diff = tl.data[p.name][idx] - p.observed[last_year]
            err += diff * diff  # squared error
        weight = 1.0 / (1.0 + err)

        # Store weighted T* stat.
        if t_star is not None:
            weighted_counts[t_star] = weighted_counts.get(t_star, 0.0) + weight

        # Accumulate for joint CSV; add run and t_star columns.
        df = tl.to_frame().copy()
        df.insert(0, "year", df.index)
        df.insert(0, "run", idx)
        df.insert(0, "t_star", t_star if t_star is not None else Pattern.T_STAR)
        all_frames.append(df.reset_index(drop=True))
        extra = f" (T*={t_star})" if t_star is not None else ""
        print(f"[RUN] generated {idx}/{args.runs}{extra}")

        # Only plot when single run requested to avoid clutter.
        if args.plot and args.runs == 1:
            img_path = out_dir / "timeline.png"
            plot_timeline(tl, path=img_path)
            print(f"[RUN] timeline plot saved to {img_path.relative_to(Path.cwd())}")

        # Record T* for later histogram.
        if t_star is not None:
            collected_ts.append(t_star)

    # ------------------------------------------------------------------
    # Write aggregated CSV of all runs.
    if all_frames:
        import pandas as pd
        big_df = pd.concat(all_frames, ignore_index=True)
        csv_path = out_dir / f"forecast_{args.patterns.replace(',', '_')}_{args.runs}runs.csv"
        big_df.to_csv(csv_path, index=False)
        print(f"[RUN] aggregated CSV saved to {csv_path.relative_to(Path.cwd())}")

    # Histogram summary
    if weighted_counts:
        print("\n[RUN] Weighted T* histogram (year : weight)")
        for yr in sorted(weighted_counts):
            print(f"  {yr}: {weighted_counts[yr]:.3f}")

        if args.plot and args.runs > 1:
            import matplotlib.pyplot as plt

            years = sorted(weighted_counts)
            freqs = [weighted_counts[y] for y in years]
            plt.bar(years, freqs, width=0.8, color="tab:blue")
            plt.xlabel("Candidate Singularity year (T*)")
            plt.ylabel("Weighted sum (higher = better fit)")
            plt.title("Weighted distribution of T* across runs")
            img_path = out_dir / "tstar_hist.png"
            plt.tight_layout()
            plt.savefig(img_path, dpi=150)
            plt.close()
            print(f"[RUN] histogram plot saved to {img_path.relative_to(Path.cwd())}")


if __name__ == "__main__":  # pragma: no cover
    main() 