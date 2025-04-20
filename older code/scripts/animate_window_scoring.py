#!/usr/bin/env python3
"""Animate evolving posterior of T* using a two‑sided window‑fit score.

This is a minimal prototype of the scoring idea discussed: for each candidate
Singularity year T*, we generate a retro‑back‑filled timeline and compute a
symmetry error between past and future windows around T* (values at year
T*-τ should match those at T*+τ).  The overall weight for that T* is an
exponentially decayed function of the aggregated squared differences.

An animation shows how the normalized probability distribution over T*
evolves as additional Monte‑Carlo trajectories are accumulated.

Example
-------
python scripts/animate_window_scoring.py \
       --patterns uap --tstar-range 2030 2100 \
       --runs 200 --window 20 --alpha 2.0 --seed 123
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root in path so *worldline_sim* imports work when run ad‑hoc
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worldline_sim.patterns import load_patterns
from worldline_sim.patterns.base import Pattern
from worldline_sim.retro_run import backfill_timeline  # re‑uses existing helpers

# -----------------------------------------------------------------------------
# Scoring helper
# -----------------------------------------------------------------------------

def window_symmetry_error(tl, patterns: List, t_star: int, window: int) -> float:
    """Return aggregated squared difference between mirrored years.

    For each τ = 1..window we compare value at (T*‑τ) with value at (T*+τ).
    The sum across patterns and τ is the error; smaller means higher symmetry.
    """
    years = tl.years
    try:
        t_idx = years.index(t_star)
    except ValueError:
        return float("inf")

    max_offset = min(window, t_idx, len(years) - t_idx - 1)
    err = 0.0
    for p in patterns:
        series = tl.data[p.name]
        for off in range(1, max_offset + 1):
            past_val = series[t_idx - off]
            fut_val = series[t_idx + off]
            err += (past_val - fut_val) ** 2
    return err

def save_timeline_data(tl, patterns: List, t_star: int, output_dir: Path, run_idx: int) -> None:
    """Save timeline data to CSV for later analysis."""
    # Create a DataFrame from the timeline
    df = pd.DataFrame(index=tl.years)
    for p in patterns:
        df[p.name] = tl.data[p.name]
    
    # Add metadata columns
    df['t_star'] = t_star
    df['run_idx'] = run_idx
    
    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"timeline_tstar{t_star}_run{run_idx}.csv"
    df.to_csv(csv_path)
    
    print(f"[WIN] Saved timeline to {csv_path}")
    return csv_path

# -----------------------------------------------------------------------------
# Custom backfill that's more lenient for UAP pattern
# -----------------------------------------------------------------------------

def lenient_backfill_timeline(patterns: List[Pattern], t_star: int, *, max_tau: int, 
                              correction_factor: float = 0.7, 
                              noise_range: float = 0.1, 
                              tolerance: float = 10.0) -> object:
    """Generate a timeline with relaxed constraints for UAP pattern.
    
    This is similar to the original backfill_timeline but more tolerant of constraint violations.
    """
    # Apply lenient constraint on patterns
    for p in patterns:
        p.custom_tolerance = tolerance
        if p.name == "uap":
            p.TOL = tolerance
        elif p.name == "rng":
            p.rng_tolerance = tolerance
    
    # Determine overall year span.
    past_years = set()
    for p in patterns:
        past_years.update(p.observed)
    start_year = min(past_years) if past_years else 1950

    horizon = t_star + max_tau
    years = list(range(start_year, horizon + 1))

    # Prepare dict year→value for each pattern.
    data: dict[str, dict[int, float]] = {p.name: {y: np.nan for y in years} for p in patterns}

    # 1) Assign post‑Singularity years directly from post_value
    for p in patterns:
        # Add some noise to make patterns more varied
        rng = getattr(p, "_rng", np.random.default_rng())
        for y in range(t_star, horizon + 1):
            base_val = p.post_value(y)
            noise = rng.uniform(-noise_range, noise_range) * base_val
            data[p.name][y] = base_val + noise

    # 2) Backward fill
    for y in reversed(range(start_year, t_star)):
        tau_to_future = t_star - y
        for p in patterns:
            # Retro influence from future years up to *max_tau*
            retro = 0.0
            for tau in range(0, max_tau + 1):
                yy = t_star + tau
                if yy > horizon:
                    break
                retro += p.retro_kernel(tau) * data[p.name][yy]
            
            base = p.sample(y, None)  # forward model baseline
            
            # For RNG pattern, apply correction
            if p.name == "rng":
                # Apply customizable correction and noise
                target_val = 0.75 if y == 2020 else (0.65 if y == 2015 else None)
                if target_val is not None and base + retro != 0:
                    correction = (target_val - (base + retro)) * correction_factor
                    base = base + correction
            
            # Add noise to make things more varied
            rng = getattr(p, "_rng", np.random.default_rng()) 
            noise = rng.uniform(-noise_range, noise_range) * abs(base)
            data[p.name][y] = base + retro + noise

    # Convert to Timeline
    array_data = {name: np.array([vals[y] for y in years]) for name, vals in data.items()}
    from worldline_sim.sim import Timeline
    tl = Timeline(years, array_data)

    # We return the timeline even if it's invalid
    return tl

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Animate posterior of T* using window‑fit scoring")
    p.add_argument("--patterns", default="uap", help="comma‑list of patterns to load")
    p.add_argument("--tstar-range", nargs=2, type=int, required=True, metavar=("MIN", "MAX"),
                   help="candidate T* range (inclusive)")
    p.add_argument("--runs", type=int, default=100, help="Monte‑Carlo trajectories to accumulate")
    p.add_argument("--window", type=int, default=20, help="±years to include in symmetry window")
    p.add_argument("--max-tau", type=int, default=70, help="horizon after T* when back‑filling")
    p.add_argument("--leak-lambda", type=float, default=0.2, help="retro leak λ parameter")
    p.add_argument("--leak-tau0", type=float, default=20.0, help="retro decay τ0 parameter")
    p.add_argument("--alpha", type=float, default=5.0, help="weight sharpness: w = exp(-α⋅err)")
    p.add_argument("--correction-factor", type=float, default=0.7, help="correction factor for RNG")
    p.add_argument("--noise-range", type=float, default=0.1, help="noise range as fraction of base value")
    p.add_argument("--tolerance", type=float, default=10.0, help="tolerance for pattern constraints")
    p.add_argument("--seed", type=int, help="base RNG seed")
    p.add_argument("--save-data", action="store_true", help="save data for each timeline")
    p.add_argument("--output-prefix", default=None, help="prefix for output files")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)

    # Setup RNG and timestamp for run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = args.output_prefix or f"window_score_{timestamp}"
    
    # Setup directories
    animation_dir = PROJECT_ROOT / "animation_frames"
    animation_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = PROJECT_ROOT / "data" / f"{output_prefix}"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup RNG with seed
    seed = args.seed if args.seed is not None else np.random.randint(0, 2**32 - 1)
    rng = np.random.default_rng(seed)
    print(f"[WIN] Using base seed: {seed}")

    t_min, t_max = args.tstar_range
    t_candidates = list(range(t_min, t_max + 1))
    weight_sums = np.zeros(len(t_candidates), dtype=float)
    
    # Store errors for later analysis
    all_errors: Dict[int, List[float]] = {t: [] for t in t_candidates}

    # Prepare patterns template (we clone per trajectory to reseed RNGs)
    patterns_template = load_patterns(args.patterns, t_star=t_min)  # t_star placeholder
    for p in patterns_template:
        p.leak_lambda = args.leak_lambda
        p.leak_tau0 = args.leak_tau0
        p.custom_tolerance = args.tolerance
        if p.name == "uap":
            p.TOL = args.tolerance
        elif p.name == "rng":
            p.rng_tolerance = args.tolerance

    # Storage for animation frames (probability distributions)
    frames: List[np.ndarray] = []
    
    # CSV for tracking results
    results_csv = data_dir / f"{output_prefix}_results.csv"
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_idx', 't_star', 'error', 'weight', 'cumulative_weight'])

    for run_idx in range(args.runs):
        # Jitter seed per run for randomness
        run_seed = rng.integers(0, 2 ** 32 - 1)
        print(f"[WIN] Starting run {run_idx + 1}/{args.runs} with seed {run_seed}")

        for t_idx, t_star in enumerate(t_candidates):
            # Clone patterns so each (run,t*) combo gets fresh RNGs
            patterns = load_patterns(args.patterns, t_star=t_star)
            for p, tmpl in zip(patterns, patterns_template):
                p.leak_lambda = tmpl.leak_lambda
                p.leak_tau0 = tmpl.leak_tau0
                p.custom_tolerance = args.tolerance
                p._rng = np.random.default_rng((hash((p.name, t_star, run_seed)) & 0xFFFF_FFFF))

            # Use our lenient backfill instead
            tl = lenient_backfill_timeline(
                patterns, 
                t_star, 
                max_tau=args.max_tau,
                correction_factor=args.correction_factor,
                noise_range=args.noise_range,
                tolerance=args.tolerance
            )
            
            # Skip if somehow still invalid
            if tl is None:
                print(f"[WIN] Invalid timeline for T*={t_star}, run {run_idx}")
                continue

            err = window_symmetry_error(tl, patterns, t_star, window=args.window)
            all_errors[t_star].append(err)
            
            log_w = -args.alpha * err
            weight = math.exp(log_w)
            weight_sums[t_idx] += weight
            
            # Save timeline data if requested
            if args.save_data and run_idx % 10 == 0:  # Save every 10th run to avoid too many files
                csv_path = save_timeline_data(tl, patterns, t_star, data_dir, run_idx)
            
            # Record results for this run and T*
            with open(results_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([run_idx, t_star, err, weight, weight_sums[t_idx]])

        # Normalise for frame
        if weight_sums.sum() > 0:
            prob = weight_sums / weight_sums.sum()
        else:
            prob = np.zeros_like(weight_sums)
        frames.append(prob.copy())
        
        # Save intermediate results
        if (run_idx + 1) % 10 == 0 or run_idx == args.runs - 1:
            print(f"[WIN] Completed run {run_idx + 1}/{args.runs}")
            
            # Save histogram for this checkpoint
            if prob.sum() > 0:
                plt.figure(figsize=(10, 5))
                plt.bar(t_candidates, prob, color="tab:purple")
                plt.xlabel("Candidate Singularity Year T*")
                plt.ylabel("Posterior probability (normalised)")
                plt.title(f"T* Distribution after {run_idx + 1} runs")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                checkpoint_file = animation_dir / f"{output_prefix}_frame_{run_idx:03d}.png"
                plt.savefig(checkpoint_file, dpi=120)
                plt.close()
                print(f"[WIN] Saved checkpoint plot to {checkpoint_file}")

    # Save final error statistics
    all_errors_filled = {t: errors if errors else [0.0] for t, errors in all_errors.items()}
    error_df = pd.DataFrame({t: all_errors_filled[t] for t in t_candidates}).describe().T
    error_stats_file = data_dir / f"{output_prefix}_error_stats.csv"
    error_df.to_csv(error_stats_file)
    print(f"[WIN] Saved error statistics to {error_stats_file}")

    # ---------------------------------------------------------------------
    # Build animation
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Handle case where frames are empty (all weights are 0)
    if not frames or all(f.sum() == 0 for f in frames):
        print("[WIN] WARNING: No valid weights found. Using uniform distribution.")
        frames = [np.ones_like(t_candidates) / len(t_candidates) for _ in range(args.runs)]
    
    bar_container = ax.bar(t_candidates, frames[0], color="tab:purple")
    ax.set_ylim(0, max(max(f) for f in frames) * 1.1 if frames else 1.0)
    ax.set_xlabel("Candidate Singularity Year T*")
    ax.set_ylabel("Posterior probability (normalised)")
    ax.set_title("Evolution of posterior over T* (window‑fit scoring)")
    ax.grid(True, alpha=0.3)

    def update(frame_idx):
        probs = frames[frame_idx]
        for rect, h in zip(bar_container, probs):
            rect.set_height(h)
        ax.set_title(f"Evolution of posterior over T*  |  after {frame_idx + 1} runs")
        return bar_container

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=120,
        blit=False,
    )

    # Save the animation
    output_mp4 = animation_dir / f"{output_prefix}_evolution.mp4"
    anim.save(output_mp4, dpi=120, writer="ffmpeg")
    print(f"[WIN] Animation saved to {output_mp4}")
    
    # Save final frame as a static image
    final_frame = animation_dir / f"{output_prefix}_final.png"
    plt.figure(figsize=(10, 5))
    plt.bar(t_candidates, frames[-1], color="tab:purple")
    plt.xlabel("Candidate Singularity Year T*")
    plt.ylabel("Posterior probability (normalised)")
    plt.title(f"Final T* Distribution (after {args.runs} runs)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(final_frame, dpi=150)
    plt.close()
    print(f"[WIN] Final frame saved as {final_frame}")
    
    # Create summary report
    print("\n--- SUMMARY ---")
    print(f"Total runs: {args.runs}")
    print(f"Patterns: {args.patterns}")
    print(f"Window size: {args.window}")
    print(f"T* range: {t_min}-{t_max}")
    if frames and frames[-1].sum() > 0:
        most_likely_t = t_candidates[np.argmax(frames[-1])]
        print(f"Most likely T*: {most_likely_t} (probability: {frames[-1][np.argmax(frames[-1])]:.4f})")
    print(f"Data saved to: {data_dir}")
    print(f"Animation saved to: {output_mp4}")
    print("---------------\n")


if __name__ == "__main__":  # pragma: no cover
    main() 