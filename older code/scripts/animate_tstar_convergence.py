#!/usr/bin/env python3
"""Animate how T* probability is driven by matching observed values.

This script visualizes how different T* (Singularity year) candidates produce 
different simulated values for observed years, and how the match quality determines
the probability of each T* year.

Usage example:
-------------
python animate_tstar_convergence.py --patterns rng,tech,uap,events --tstar-range 2090 2100 --leak-lambda 0.2 --leak-tau0 20 --max-tau 50 --seed 42
"""

import argparse
import math
import os
from pathlib import Path
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# Import the necessary components from worldline_sim
from worldline_sim.patterns import load_patterns
from worldline_sim.sim import Timeline
from worldline_sim.viz import plot_timeline
from super_lenient_run import backfill_timeline

# ---------------------------------------------------------------------------
# Apply the same super-lenient pattern patches as in super_lenient_run.py
# ---------------------------------------------------------------------------
from worldline_sim.patterns.base import Pattern
from worldline_sim.patterns.rng import RngPattern

# Make all constraint methods extremely lenient
def super_lenient_constraint(self, year: int, value: float, *, tol: float | None = None) -> bool:
    """Super lenient constraint that accepts almost any value."""
    # For RNG pattern, use high tolerance
    if self.name == "rng" and year in self.observed:
        return abs(value - self.observed[year]) <= 20.0
    # For all other patterns, basically accept anything
    return True

# Apply patch to ALL patterns
Pattern.original_constraint = Pattern.constraint
Pattern.constraint = super_lenient_constraint
RngPattern.original_constraint = RngPattern.constraint
RngPattern.constraint = super_lenient_constraint
Pattern.TOL = 50.0  # Make base pattern tolerance extremely high

# ---------------------------------------------------------------------------
# Retro kernel and post value functions
# ---------------------------------------------------------------------------
def _retro_kernel(self: Pattern, tau: int) -> float:
    """Default exponential leak λ·exp(‑τ/τ0)."""
    lam = getattr(self, "leak_lambda", 0.0)
    tau0 = getattr(self, "leak_tau0", 1.0)
    return lam * math.exp(-tau / tau0)

def _post_value(self: Pattern, year: int) -> float:
    """Default post‑Singularity anchor = forward sample with timeline=None."""
    return self.sample(year, None)

# Attach when not already present.
if not hasattr(Pattern, "retro_kernel"):
    Pattern.retro_kernel = _retro_kernel
if not hasattr(Pattern, "post_value"):
    Pattern.post_value = _post_value

# ---------------------------------------------------------------------------
# Functions to generate timeline data for each T*
# ---------------------------------------------------------------------------
def generate_timelines_for_tstar_range(patterns_str: str, t_star_range: Tuple[int, int], 
                                      leak_lambda: float, leak_tau0: float, max_tau: int, 
                                      seed: int = None) -> Dict:
    """Generate simulated timelines for a range of T* years."""
    t_star_min, t_star_max = t_star_range
    results = {}
    
    print(f"[CONV] Generating timelines for T* range {t_star_min}-{t_star_max}")
    
    base_rng = np.random.default_rng(seed)
    
    for t_star in range(t_star_min, t_star_max + 1):
        # Load patterns for this T*
        patterns = load_patterns(patterns_str, t_star=t_star)
        for p in patterns:
            p.leak_lambda = leak_lambda
            p.leak_tau0 = leak_tau0
            
            # Set a unique seed for this pattern
            pattern_seed = hash((p.name, t_star, 0)) & 0xFFFF_FFFF
            p._rng = np.random.default_rng(pattern_seed)
        
        # Generate timeline
        print(f"[CONV] Processing T* = {t_star}")
        tl = backfill_timeline(patterns, t_star, max_tau=max_tau)
        
        if tl is None:
            print(f"[CONV] Failed to generate valid timeline for T* = {t_star}")
            continue
        
        # Extract the RNG pattern values
        rng_pattern = next((p for p in patterns if p.name == 'rng'), None)
        if rng_pattern:
            # Calculate error for observed years
            observed_years = [2015, 2020]
            errors = {}
            total_error = 0.0
            
            for year in observed_years:
                if year in rng_pattern.observed:
                    idx = tl.years.index(year)
                    simulated_value = tl.data['rng'][idx]
                    observed_value = rng_pattern.observed[year]
                    
                    # Calculate squared error
                    error = (simulated_value - observed_value) ** 2
                    errors[year] = error
                    total_error += error
            
            # Calculate weight based on error (higher error = lower weight)
            alpha = 10.0  # Error weight factor
            log_weight = -alpha * total_error
            weight = math.exp(log_weight)
            
            # Store results
            results[t_star] = {
                'timeline': tl,
                'errors': errors,
                'total_error': total_error,
                'log_weight': log_weight,
                'weight': weight
            }
        
    # Normalize weights to get probabilities
    if results:
        total_weight = sum(results[t_star]['weight'] for t_star in results)
        for t_star in results:
            results[t_star]['probability'] = results[t_star]['weight'] / total_weight if total_weight > 0 else 0
    
    return results

# ---------------------------------------------------------------------------
# Animation Generation
# ---------------------------------------------------------------------------
def create_tstar_convergence_animation(results, output_file=None):
    """Create an animation showing how observed/simulated matches drive T* probability."""
    if output_file is None:
        output_file = f"tstar_convergence_animation.mp4"
    
    if not results:
        print("[CONV] No valid results to animate")
        return None
    
    print(f"[CONV] Creating convergence animation for {len(results)} T* years")
    
    # Extract years range from the first timeline
    t_star_sample = next(iter(results.keys()))
    tl_sample = results[t_star_sample]['timeline']
    min_year = min(tl_sample.years)
    max_year = max(tl_sample.years)
    
    # Identify observed years and values for the RNG pattern
    observed_years = []
    observed_values = {}
    
    # Assuming the observed years are the same for all T* values
    for t_star, result in results.items():
        tl = result['timeline']
        rng_pattern = next((p for p in load_patterns('rng', t_star=t_star) if p.name == 'rng'), None)
        if rng_pattern:
            for year in rng_pattern.observed:
                if year not in observed_years:
                    observed_years.append(year)
                    observed_values[year] = rng_pattern.observed[year]
    
    observed_years.sort()
    
    # Set up the figure
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"How T* Probability is Driven by Observed/Simulated Matches", fontsize=16)
    
    # Create three subplots:
    # 1. RNG values vs years for different T*
    # 2. Error for each observed year vs T*
    # 3. T* probability distribution
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # 1. Setup the timeline plot
    ax1.set_title("RNG Pattern: Observed vs Simulated Values")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("RNG Value")
    ax1.set_xlim(2010, 2025)  # Focus on observed years
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot observed values
    for year, value in observed_values.items():
        ax1.scatter([year], [value], color='red', s=100, zorder=5)
        ax1.annotate(f"Observed: {value:.2f}", (year, value),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9)
    
    # Initialize lines for each T*
    t_star_years = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_star_years)))
    
    t_star_lines = {}
    for i, t_star in enumerate(t_star_years):
        line, = ax1.plot([], [], label=f"T*={t_star}", color=colors[i])
        t_star_lines[t_star] = line
    
    ax1.legend(loc='upper left')
    
    # 2. Setup the error plot
    ax2.set_title("Error Between Observed and Simulated Values")
    ax2.set_xlabel("T* (Singularity Year)")
    ax2.set_ylabel("Squared Error")
    ax2.set_xticks(t_star_years)
    ax2.set_xlim(min(t_star_years) - 0.5, max(t_star_years) + 0.5)
    
    # Create bar containers for each observed year
    bar_width = 0.35
    bar_positions = {}
    bar_containers = {}
    
    for i, year in enumerate(observed_years):
        positions = [t - bar_width/2 + i*bar_width/len(observed_years) for t in t_star_years]
        bar_positions[year] = positions
        bars = ax2.bar(positions, [0] * len(t_star_years), bar_width/len(observed_years), 
                       label=f"Error for {year}", alpha=0.7)
        bar_containers[year] = bars
    
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # 3. Setup the probability plot
    ax3.set_title("T* Probability Distribution (based on match quality)")
    ax3.set_xlabel("T* (Singularity Year)")
    ax3.set_ylabel("Probability")
    ax3.set_xticks(t_star_years)
    ax3.set_xlim(min(t_star_years) - 0.5, max(t_star_years) + 0.5)
    
    # Create probability bars
    prob_bars = ax3.bar(t_star_years, [0] * len(t_star_years), color='green', alpha=0.7)
    ax3.set_ylim(0, 1.05 * max(results[t]['probability'] for t in results))
    ax3.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Add annotations for probability values
    prob_texts = []
    for i, t_star in enumerate(t_star_years):
        text = ax3.text(t_star, 0, "", ha='center', va='bottom', fontsize=9)
        prob_texts.append(text)
    
    # Current T* indicator and probability info
    title_text = ax1.text(0.5, 1.05, "", transform=ax1.transAxes, ha="center", fontsize=12)
    
    # Animation update function
    def update(frame):
        # Get current T*
        if frame < len(t_star_years):
            current_t_star = t_star_years[frame]
            highlight = True
        else:
            # Final frame - show all together
            current_t_star = None
            highlight = False
        
        # Update title
        if current_t_star:
            result = results[current_t_star]
            title_text.set_text(f"Focusing on T*={current_t_star} " +
                              f"(Probability: {result['probability']:.4f}, " +
                              f"Total Error: {result['total_error']:.4f})")
        else:
            title_text.set_text("All T* years with their respective probabilities")
        
        # Update timeline lines
        for t_star, line in t_star_lines.items():
            result = results[t_star]
            tl = result['timeline']
            
            # Get RNG values from the timeline
            years = tl.years
            values = tl.data['rng']
            
            # If highlighting a specific T*, only show that line prominently
            if highlight:
                if t_star == current_t_star:
                    line.set_data(years, values)
                    line.set_linewidth(3.0)
                    line.set_alpha(1.0)
                else:
                    line.set_data(years, values)
                    line.set_linewidth(0.5)
                    line.set_alpha(0.2)
            else:
                # Show all lines
                line.set_data(years, values)
                line.set_linewidth(1.5)
                line.set_alpha(0.7)
        
        # Update error bars
        for year, bars in bar_containers.items():
            for i, t_star in enumerate(t_star_years):
                result = results[t_star]
                error = result['errors'].get(year, 0)
                
                bars[i].set_height(error)
                
                # Highlight current T* if applicable
                if highlight and t_star == current_t_star:
                    bars[i].set_alpha(1.0)
                    bars[i].set_edgecolor('black')
                    bars[i].set_linewidth(2)
                else:
                    bars[i].set_alpha(0.7)
                    bars[i].set_edgecolor('none')
                    bars[i].set_linewidth(0)
        
        # Update probability bars
        for i, t_star in enumerate(t_star_years):
            result = results[t_star]
            prob = result['probability']
            
            prob_bars[i].set_height(prob)
            prob_texts[i].set_position((t_star, prob))
            prob_texts[i].set_text(f"{prob:.3f}")
            
            # Highlight current T* if applicable
            if highlight and t_star == current_t_star:
                prob_bars[i].set_alpha(1.0)
                prob_bars[i].set_edgecolor('black')
                prob_bars[i].set_linewidth(2)
            else:
                prob_bars[i].set_alpha(0.7)
                prob_bars[i].set_edgecolor('none')
                prob_bars[i].set_linewidth(0)
        
        # Return all updated artists
        artists = [title_text] + list(t_star_lines.values()) + list(prob_texts)
        for bars in bar_containers.values():
            artists.extend(bars)
        artists.extend(prob_bars)
        
        return artists
    
    # Create animation
    print(f"[CONV] Generating animation with {len(t_star_years) + 1} frames")
    anim = animation.FuncAnimation(
        fig, update, frames=len(t_star_years) + 1,  # +1 for the "all together" frame
        interval=1000,  # 1 second per frame
        blit=True
    )
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    # Save animation
    print(f"[CONV] Saving animation to {output_file}")
    anim.save(output_file, writer='ffmpeg', fps=1, dpi=200)
    plt.close()
    
    print(f"[CONV] Animation saved to {output_file}")
    return output_file

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Animate how T* probability is driven by observed/simulated matches")
    p.add_argument("--patterns", default="rng", help="comma list of patterns")
    p.add_argument("--tstar-range", nargs=2, type=int, metavar=("MIN", "MAX"), required=True,
                   help="range of T* values to simulate")
    p.add_argument("--leak-lambda", type=float, default=0.2, help="λ leak strength (all patterns)")
    p.add_argument("--leak-tau0", type=float, default=20.0, help="τ0 decay time (years)")
    p.add_argument("--max-tau", type=int, default=50, help="simulate this many post years")
    p.add_argument("--seed", type=int, help="seed for random number generator")
    p.add_argument("--output", type=str, help="output filename for animation")
    return p

def main(argv=None):
    args = build_argparser().parse_args(argv)
    
    seed = getattr(args, "seed", None)
    t_star_min, t_star_max = args.tstar_range
    
    print(f"[CONV] Using SUPER-lenient constraints with natural variation")
    print(f"[CONV] Using base seed: {seed}")
    print(f"[CONV] Simulating for T* range {t_star_min}-{t_star_max}")
    
    # Generate timelines for each T*
    results = generate_timelines_for_tstar_range(
        args.patterns, 
        (t_star_min, t_star_max), 
        args.leak_lambda, 
        args.leak_tau0, 
        args.max_tau, 
        seed
    )
    
    if not results:
        print(f"[CONV] No valid timelines were generated.")
        return 1
    
    # Create the animation
    output_file = args.output if args.output else f"tstar_convergence_{t_star_min}_{t_star_max}.mp4"
    create_tstar_convergence_animation(results, output_file)
    
    # Also save a histogram of T* probabilities
    hist_file = f"tstar_convergence_hist_{t_star_min}_{t_star_max}.png"
    t_stars = sorted(results.keys())
    probs = [results[t]['probability'] for t in t_stars]
    
    plt.figure(figsize=(10, 6))
    plt.bar(t_stars, probs, color='green', alpha=0.7)
    plt.xlabel("T* (Singularity Year)")
    plt.ylabel("Probability")
    plt.title(f"T* Probability Distribution ({t_star_min}-{t_star_max})")
    plt.xticks(t_stars)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Add text labels with probability values
    for i, (t, p) in enumerate(zip(t_stars, probs)):
        plt.text(t, p, f"{p:.3f}", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(hist_file, dpi=200)
    plt.close()
    
    print(f"[CONV] Histogram saved to {hist_file}")
    return 0

if __name__ == "__main__":
    main() 