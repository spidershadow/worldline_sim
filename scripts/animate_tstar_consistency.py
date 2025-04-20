#!/usr/bin/env python3
"""Animate how T* probabilities converge across multiple simulations.

This script visualizes how different simulation parameters affect the T* probability
distribution and demonstrates that the ~2050 time period consistently has the
highest probability across diverse parameter settings.

Usage example:
-------------
python animate_tstar_consistency.py --tstar-range 2045 2055 --simulations 50 --seed 42
"""

import argparse
import itertools
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path to ensure imports work after reorganization
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# Import from worldline_sim and our consistent T* script
from worldline_sim.patterns import load_patterns
from worldline_sim.sim import Timeline
from worldline_sim.patterns.base import Pattern
from worldline_sim.patterns.rng import RngPattern

# Import functions from find_consistent_tstar.py which is now in the scripts directory
# Since we're in the same directory, we can import directly
from find_consistent_tstar import (
    lenient_constraint,
    _retro_kernel,
    _post_value,
    custom_backfill_timeline,
    run_simulation,
)

def generate_parameter_configs(args):
    """Generate diverse parameter configurations to test."""
    configs = []
    
    # Define parameter ranges to explore
    pattern_combinations = [
        "rng", 
        "rng,tech", 
        "rng,uap", 
        "rng,tech,uap"
    ]
    
    # Use custom patterns if provided
    if args.patterns:
        pattern_combinations = args.patterns.split(',')
    
    leak_lambdas = [0.1, 0.2, 0.3]
    leak_tau0s = [10, 20, 30]
    alphas = [5.0, 10.0, 15.0]
    correction_factors = [0.3, 0.5, 0.7]
    noise_ranges = [0.05, 0.1, 0.15]
    rng_tolerances = [5.0, 10.0, 15.0]
    tstar_bias_options = [True, False]
    
    # Generate all parameter combinations
    param_space = list(itertools.product(
        pattern_combinations,
        leak_lambdas,
        leak_tau0s,
        alphas,
        correction_factors,
        noise_ranges,
        rng_tolerances,
        tstar_bias_options
    ))
    
    # Sample the requested number of parameter combinations
    random.seed(args.seed)
    
    # Ensure roughly equal representation of each pattern combination
    balanced_params = []
    simulations_per_pattern = max(1, args.simulations // len(pattern_combinations))
    
    for pattern in pattern_combinations:
        # Filter parameter space for this pattern
        pattern_params = [p for p in param_space if p[0] == pattern]
        # Sample from this pattern's parameter space
        if len(pattern_params) > simulations_per_pattern:
            sampled = random.sample(pattern_params, simulations_per_pattern)
        else:
            sampled = pattern_params
        balanced_params.extend(sampled)
    
    # If we need more configurations to meet the requested simulation count
    if len(balanced_params) < args.simulations:
        remaining = args.simulations - len(balanced_params)
        remaining_params = [p for p in param_space if p not in balanced_params]
        if remaining_params and remaining > 0:
            balanced_params.extend(random.sample(remaining_params, min(remaining, len(remaining_params))))
    
    # Limit to the requested number of simulations
    param_space = balanced_params[:args.simulations]
    
    # Create configs
    t_star_range = (args.tstar_range[0], args.tstar_range[1])
    for run_id, params in enumerate(param_space):
        patterns_str, leak_lambda, leak_tau0, alpha, correction_factor, noise_range, rng_tolerance, apply_tstar_bias = params
        
        config = {
            'patterns': patterns_str,
            't_star_range': t_star_range,
            'leak_lambda': leak_lambda,
            'leak_tau0': leak_tau0,
            'max_tau': 50,  # Default max tau value
            'seed': args.seed + run_id,
            'alpha': alpha,
            'correction_factor': correction_factor,
            'noise_range': noise_range,
            'rng_tolerance': rng_tolerance,
            'apply_tstar_bias': apply_tstar_bias,
            'run_id': run_id
        }
        
        configs.append(config)
    
    print(f"[ANIM] Generated {len(configs)} parameter configurations for animation")
    return configs

def run_all_simulations(configs):
    """Run all simulations and collect results."""
    results = []
    
    for i, config in enumerate(configs):
        print(f"[ANIM] Running simulation {i+1}/{len(configs)}: {config['config_id'] if 'config_id' in config else ''}")
        result = run_simulation(config)
        results.append(result)
    
    return results

def create_probability_animation(results, t_star_range, output_file="tstar_convergence.mp4"):
    """Create an animation showing how T* probabilities converge across simulations."""
    t_star_min, t_star_max = t_star_range
    t_stars = list(range(t_star_min, t_star_max + 1))
    
    # Check if we have valid results
    if not results:
        print("[ANIM] No valid results to animate")
        return
    
    # Create output directories
    output_file = Path(output_file)
    animations_dir = output_file.parent
    images_dir = animations_dir.parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Get output file names
    final_image = images_dir / f"{output_file.stem}_final.png"
    pattern_comparison_image = images_dir / f"{output_file.stem}_pattern_comparison.png"
    
    # Determine unique pattern combinations
    pattern_combinations = sorted(list(set(r['patterns'] for r in results)))
    pattern_colors = plt.cm.tab10(np.linspace(0, 1, len(pattern_combinations)))
    pattern_to_color = {pattern: color for pattern, color in zip(pattern_combinations, pattern_colors)}
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("T* Probability Convergence Across Multiple Simulations", fontsize=16)
    
    # Create grid for subplots
    gs = plt.GridSpec(3, 3, height_ratios=[2, 1.5, 1.5], width_ratios=[2, 1, 1], hspace=0.35, wspace=0.25)
    
    # 1. Probability bar chart (main plot)
    ax_prob = fig.add_subplot(gs[0, 0])
    ax_prob.set_title("Mean T* Probability Distribution")
    ax_prob.set_xlabel("T* (Singularity Year)")
    ax_prob.set_ylabel("Probability")
    
    # 2. Running statistics plot
    ax_stats = fig.add_subplot(gs[1, 0])
    ax_stats.set_title("Running Statistics for T*=2050")
    ax_stats.set_xlabel("Simulation Count")
    ax_stats.set_ylabel("Probability")
    
    # 3. Heatmap of all simulations
    ax_heat = fig.add_subplot(gs[:2, 1])
    ax_heat.set_title("Simulation Results Heatmap")
    ax_heat.set_xlabel("T* (Singularity Year)")
    ax_heat.set_ylabel("Simulation #")
    
    # 4. Most frequent T* histogram
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_hist.set_title("Most Probable T* Frequency")
    ax_hist.set_xlabel("T* (Singularity Year)")
    ax_hist.set_ylabel("Count")
    
    # 5. Confidence interval plot
    ax_ci = fig.add_subplot(gs[2, 1])
    ax_ci.set_title("90% Confidence Interval")
    ax_ci.set_xlabel("Simulation Count")
    ax_ci.set_ylabel("T* Year")
    
    # 6. Pattern comparison (NEW)
    ax_pattern = fig.add_subplot(gs[0:2, 2])
    ax_pattern.set_title("Pattern Impact on T* Probability")
    ax_pattern.set_xlabel("T* (Singularity Year)")
    ax_pattern.set_ylabel("Probability")
    
    # 7. Pattern legend and statistics (NEW)
    ax_pattern_stats = fig.add_subplot(gs[2, 2])
    ax_pattern_stats.set_title("Pattern Statistics")
    ax_pattern_stats.set_axis_off()  # No axes needed for text
    
    # Initialize data structures for tracking
    all_probs = pd.DataFrame(index=t_stars, columns=[r['config_id'] for r in results])
    all_probs = all_probs.fillna(0.0).astype(float)  # Explicitly convert to float
    
    # Group results by pattern combination
    pattern_groups = {}
    for pattern in pattern_combinations:
        pattern_groups[pattern] = [r for r in results if r['patterns'] == pattern]
    
    # Track running statistics
    running_means = []
    running_medians = []
    confidence_intervals = []
    
    # Prepare the bar container for the main probability plot
    bars = ax_prob.bar(t_stars, [0] * len(t_stars), alpha=0.7)
    
    # Prepare line objects for running statistics
    mean_line, = ax_stats.plot([], [], 'b-', label='Mean Probability of T*=2050')
    median_line, = ax_stats.plot([], [], 'g-', label='Median Probability of T*=2050')
    ax_stats.legend()
    
    # Prepare histogram for most frequent T*
    hist_bars = ax_hist.bar(t_stars, [0] * len(t_stars), alpha=0.7)
    
    # Prepare confidence interval lines
    ci_lower, = ax_ci.plot([], [], 'r-', label='Lower Bound')
    ci_upper, = ax_ci.plot([], [], 'g-', label='Upper Bound')
    ax_ci.legend()
    
    # Prepare heatmap data
    heatmap_data = np.zeros((len(results), len(t_stars)))
    
    # Prepare pattern lines for pattern comparison
    pattern_lines = {}
    for pattern in pattern_combinations:
        line, = ax_pattern.plot([], [], '-', color=pattern_to_color[pattern], 
                               label=pattern, linewidth=2)
        pattern_lines[pattern] = line
    
    ax_pattern.legend()
    
    # Prepare heatmap with pattern coloring
    pattern_indices = np.zeros(len(results))
    for i, result in enumerate(results):
        pattern_idx = pattern_combinations.index(result['patterns'])
        pattern_indices[i] = pattern_idx
    
    # Create heatmap with simulation results
    heatmap = ax_heat.imshow(heatmap_data, aspect='auto', cmap='viridis', 
                            origin='lower', interpolation='none')
    plt.colorbar(heatmap, ax=ax_heat, label="Probability")
    
    # Add vertical lines on heatmap to separate pattern groups
    pattern_counts = {}
    current_idx = 0
    for pattern in pattern_combinations:
        count = sum(1 for r in results if r['patterns'] == pattern)
        pattern_counts[pattern] = count
        if current_idx > 0:
            ax_heat.axhline(y=current_idx - 0.5, color='white', linestyle='-', alpha=0.7)
        current_idx += count
    
    # Prepare pattern background coloring for heatmap
    alpha_val = 0.2  # Transparency for background
    current_row = 0
    for pattern, count in pattern_counts.items():
        color = pattern_to_color[pattern]
        ax_heat.add_patch(plt.Rectangle((0, current_row - 0.5), len(t_stars), count,
                                      facecolor=color, alpha=alpha_val, edgecolor=None))
        current_row += count
    
    # Set up axis limits and ticks
    if t_star_max - t_star_min > 20:
        # For wide ranges, use fewer ticks
        tick_step = max(1, (t_star_max - t_star_min) // 10)
        tick_positions = range(t_star_min, t_star_max + 1, tick_step)
        ax_prob.set_xticks(tick_positions)
        ax_hist.set_xticks(tick_positions)
        ax_pattern.set_xticks(tick_positions)
    else:
        # For narrower ranges, can use all ticks
        ax_prob.set_xticks(t_stars)
        ax_hist.set_xticks(t_stars)
        ax_pattern.set_xticks(t_stars)
    
    ax_prob.set_ylim(0, 1)
    ax_stats.set_xlim(1, len(results))
    ax_stats.set_ylim(0, 1)
    ax_hist.set_ylim(0, len(results))
    ax_ci.set_xlim(1, len(results))
    ax_ci.set_ylim(t_star_min - 0.5, t_star_max + 0.5)
    ax_pattern.set_xlim(t_star_min - 0.5, t_star_max + 0.5)
    ax_pattern.set_ylim(0, 1)
    
    # Set heatmap ticks
    if len(t_stars) > 20:
        tick_step = max(1, (t_star_max - t_star_min) // 10)
        ax_heat.set_xticks(np.arange(0, len(t_stars), tick_step))
        ax_heat.set_xticklabels([t_stars[i] for i in range(0, len(t_stars), tick_step)], rotation=90, fontsize=8)
    else:
        ax_heat.set_xticks(np.arange(len(t_stars)))
        ax_heat.set_xticklabels(t_stars, rotation=90, fontsize=8)
    
    ax_heat.set_yticks(np.arange(len(results)))
    ax_heat.set_yticklabels(range(1, len(results) + 1))
    
    # Title text for additional info
    sim_text = fig.text(0.5, 0.96, "", ha="center", fontsize=12)
    
    # Pattern statistics text
    pattern_stats_text = ax_pattern_stats.text(0.5, 0.95, "", ha="center", va="top", 
                                             fontsize=9, transform=ax_pattern_stats.transAxes)
    
    def update(frame):
        # Get data up to the current frame (simulation)
        current_results = results[:frame+1]
        
        # Update the simulation info text
        if frame < len(results):
            current_sim = results[frame]
            sim_text.set_text(f"Simulation {frame+1}/{len(results)}: " + 
                            f"Pattern={current_sim['patterns']}, " +
                            f"λ={current_sim['leak_lambda']}, " +
                            f"τ₀={current_sim['leak_tau0']}, " +
                            f"α={current_sim['alpha']}")
        
        # Fill the DataFrame with current data
        for i, result in enumerate(current_results):
            config_id = result['config_id']
            for t_star in t_stars:
                all_probs.loc[t_star, config_id] = result['probabilities'].get(t_star, 0.0)
        
        # Calculate statistics
        if frame > 0:
            current_df = all_probs.iloc[:, :frame+1]
            mean_probs = current_df.mean(axis=1)
            median_probs = current_df.median(axis=1)
            
            # Update the main probability bars
            for i, t_star in enumerate(t_stars):
                bars[i].set_height(mean_probs[t_star])
            
            # Update running statistics for T*=2050 (or middle of range if not present)
            target_year = 2050 if 2050 in mean_probs.index else mean_probs.index[len(mean_probs)//2]
            
            running_means.append(mean_probs[target_year])
            running_medians.append(median_probs[target_year])
            
            mean_line.set_data(range(1, len(running_means) + 1), running_means)
            median_line.set_data(range(1, len(running_medians) + 1), running_medians)
            
            # Update most probable T* histogram
            most_probable = current_df.idxmax(axis=0)
            t_star_counts = most_probable.value_counts()
            
            for i, t_star in enumerate(t_stars):
                hist_bars[i].set_height(t_star_counts.get(t_star, 0))
            
            # Calculate confidence interval
            if mean_probs.sum() > 0:
                cumulative_probs = mean_probs.sort_index().cumsum() / mean_probs.sum()
                
                lower_bound = None
                for t_star in sorted(t_stars):
                    if cumulative_probs[t_star] > 0.05:
                        lower_bound = t_star
                        break
                
                upper_bound = None
                for t_star in sorted(t_stars, reverse=True):
                    if cumulative_probs[t_star] < 0.95:
                        upper_bound = t_star
                        break
                
                if lower_bound is not None and upper_bound is not None:
                    confidence_intervals.append((lower_bound, upper_bound))
                else:
                    confidence_intervals.append((t_stars[0], t_stars[-1]))
            else:
                confidence_intervals.append((t_stars[0], t_stars[-1]))
            
            lower_bounds = [ci[0] for ci in confidence_intervals]
            upper_bounds = [ci[1] for ci in confidence_intervals]
            
            ci_lower.set_data(range(1, len(lower_bounds) + 1), lower_bounds)
            ci_upper.set_data(range(1, len(upper_bounds) + 1), upper_bounds)
            
            # Update pattern-specific plots
            pattern_stats = []
            for pattern in pattern_combinations:
                # Get results for this pattern
                pattern_results = [r for r in current_results if r['patterns'] == pattern]
                if pattern_results:
                    # Calculate pattern-specific probabilities
                    pattern_columns = [r['config_id'] for r in pattern_results]
                    pattern_df = current_df[pattern_columns]
                    pattern_means = pattern_df.mean(axis=1)
                    
                    # Update the pattern line
                    pattern_lines[pattern].set_data(t_stars, pattern_means)
                    
                    # Find most likely T* for this pattern
                    most_likely_t = pattern_means.idxmax()
                    max_prob = pattern_means.max()
                    
                    # Add to pattern statistics
                    pattern_stats.append(f"{pattern}: Peak T*={most_likely_t} (P={max_prob:.3f})")
            
            # Update pattern statistics text
            pattern_stats_text.set_text("\n".join(pattern_stats))
            
            # Update heatmap
            for i in range(frame + 1):
                result = results[i]
                for j, t_star in enumerate(t_stars):
                    heatmap_data[i, j] = result['probabilities'].get(t_star, 0.0)
            
            heatmap.set_array(heatmap_data[:frame+1])
        
        # Return all the artists that need to be updated
        artists = [sim_text, *bars, mean_line, median_line, *hist_bars, ci_lower, ci_upper, 
                  heatmap, pattern_stats_text]
        artists.extend(list(pattern_lines.values()))
        return artists
    
    # Create the animation
    print("[ANIM] Creating animation...")
    anim = animation.FuncAnimation(
        fig, update, frames=len(results),
        interval=500,  # milliseconds between frames
        blit=True,
        repeat=False
    )
    
    # Save the animation
    writer = animation.FFMpegWriter(fps=2, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(output_file, writer=writer)
    print(f"[ANIM] Animation saved to {output_file}")
    
    # Also create a final summary image
    update(len(results) - 1)  # Update with all simulations
    plt.savefig(final_image, dpi=200)
    print(f"[ANIM] Final frame saved as {final_image}")
    
    # Create an additional figure for pattern comparison
    create_pattern_comparison_figure(results, t_stars, pattern_combinations, pattern_to_color,
                                    str(pattern_comparison_image))
    
    plt.close()

def create_pattern_comparison_figure(results, t_stars, pattern_combinations, pattern_to_color, output_file):
    """Create a standalone figure comparing T* distributions across different pattern combinations."""
    # Group results by pattern combination
    pattern_groups = {pattern: [] for pattern in pattern_combinations}
    for result in results:
        pattern_groups[result['patterns']].append(result)
    
    # Calculate statistics for each pattern
    pattern_stats = {}
    for pattern, pattern_results in pattern_groups.items():
        if not pattern_results:
            continue
            
        # Create DataFrame for this pattern's probabilities
        pattern_df = pd.DataFrame(index=t_stars)
        for result in pattern_results:
            pattern_df[result['config_id']] = [result['probabilities'].get(t, 0.0) for t in t_stars]
        
        # Calculate statistics
        mean_probs = pattern_df.mean(axis=1)
        median_probs = pattern_df.median(axis=1)
        std_probs = pattern_df.std(axis=1)
        max_probs = pattern_df.max(axis=1)
        min_probs = pattern_df.min(axis=1)
        
        # Find most likely T*
        most_likely_t = mean_probs.idxmax()
        
        # Calculate 90% confidence interval
        if mean_probs.sum() > 0:
            cumulative_probs = mean_probs.sort_index().cumsum() / mean_probs.sum()
            lower_bound = None
            for t in sorted(t_stars):
                if cumulative_probs[t] > 0.05:
                    lower_bound = t
                    break
            
            upper_bound = None
            for t in sorted(t_stars, reverse=True):
                if cumulative_probs[t] < 0.95:
                    upper_bound = t
                    break
                    
            ci = (lower_bound, upper_bound)
        else:
            ci = (t_stars[0], t_stars[-1])
        
        # Store statistics
        pattern_stats[pattern] = {
            'mean': mean_probs,
            'median': median_probs,
            'std': std_probs,
            'max': max_probs,
            'min': min_probs,
            'most_likely': most_likely_t,
            'ci': ci,
            'count': len(pattern_results)
        }
    
    # Create a figure
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle("Impact of Pattern Combinations on T* Probability", fontsize=16)
    
    # Create a 2x2 grid of subplots
    gs = plt.GridSpec(2, 2, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)
    
    # 1. Mean probability with error bands
    ax_mean = fig.add_subplot(gs[0, 0])
    ax_mean.set_title("Mean T* Probability by Pattern Combination")
    ax_mean.set_xlabel("T* (Singularity Year)")
    ax_mean.set_ylabel("Probability")
    
    # 2. Boxplot comparing distributions
    ax_box = fig.add_subplot(gs[0, 1])
    ax_box.set_title("Distribution of Most Likely T* by Pattern")
    ax_box.set_xlabel("Pattern Combination")
    ax_box.set_ylabel("T* (Singularity Year)")
    
    # 3. Summary statistics table
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.set_title("Summary Statistics by Pattern Combination")
    ax_table.set_axis_off()
    
    # Plot mean probability with error bands
    for pattern, stats in pattern_stats.items():
        color = pattern_to_color[pattern]
        x = t_stars
        y = stats['mean']
        y_err = stats['std']
        
        # Plot line and error band
        ax_mean.plot(x, y, '-', color=color, linewidth=2, label=pattern)
        ax_mean.fill_between(x, y - y_err, y + y_err, color=color, alpha=0.2)
    
    ax_mean.legend()
    ax_mean.grid(True, linestyle='--', alpha=0.5)
    
    # Create boxplot data
    boxplot_data = []
    boxplot_labels = []
    most_likely_by_pattern = {}
    
    for pattern in pattern_combinations:
        if pattern not in pattern_stats:
            continue
            
        # Get all simulation results for this pattern
        pattern_results = pattern_groups[pattern]
        if not pattern_results:
            continue
            
        # Extract most likely T* from each simulation
        most_likely_tstars = []
        for result in pattern_results:
            probabilities = {t: result['probabilities'].get(t, 0.0) for t in t_stars}
            if probabilities:
                most_likely_t = max(probabilities.items(), key=lambda x: x[1])[0]
                most_likely_tstars.append(most_likely_t)
        
        if most_likely_tstars:
            boxplot_data.append(most_likely_tstars)
            boxplot_labels.append(pattern)
            most_likely_by_pattern[pattern] = most_likely_tstars
    
    # Create boxplot
    if boxplot_data:
        ax_box.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True,
                      boxprops=dict(alpha=0.7))
        
        # Add individual points
        for i, (pattern, points) in enumerate(most_likely_by_pattern.items()):
            # Add jitter to x position
            jitter = np.random.normal(0, 0.1, size=len(points))
            x = [i + 1 + j for j in jitter]
            ax_box.scatter(x, points, color=pattern_to_color[pattern], alpha=0.6)
    
    ax_box.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Create summary statistics table
    table_data = []
    table_columns = ['Pattern', 'Most Likely T*', '90% CI', 'Peak Probability', 'Simulation Count']
    
    for pattern in pattern_combinations:
        if pattern not in pattern_stats:
            continue
            
        stats = pattern_stats[pattern]
        most_likely = stats['most_likely']
        ci = stats['ci']
        peak_prob = stats['mean'].max()
        count = stats['count']
        
        table_data.append([pattern, most_likely, f"{ci[0]}-{ci[1]}", f"{peak_prob:.3f}", count])
    
    if table_data:
        table = ax_table.table(cellText=table_data, colLabels=table_columns, 
                              loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Color the rows by pattern
        for i, pattern in enumerate(p for p in pattern_combinations if p in pattern_stats):
            color = pattern_to_color[pattern]
            for j in range(len(table_columns)):
                table[(i+1, j)].set_facecolor((*color[:3], 0.2))
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file, dpi=200)
    print(f"[ANIM] Pattern comparison figure saved as {output_file}")
    plt.close()

def build_argparser():
    p = argparse.ArgumentParser(description="Animate T* probability convergence across simulations")
    p.add_argument("--tstar-range", nargs=2, type=int, metavar=("MIN", "MAX"), default=[2040, 2080],
                  help="range of T* values to simulate")
    p.add_argument("--simulations", type=int, default=30, 
                  help="number of simulations to run with different parameters")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    p.add_argument("--output", type=str, default="tstar_convergence.mp4", 
                  help="output filename for animation")
    p.add_argument("--patterns", type=str, help="comma-separated list of pattern combinations to use")
    return p

def main():
    args = build_argparser().parse_args()
    
    print(f"[ANIM] Animating T* convergence for range {args.tstar_range[0]}-{args.tstar_range[1]}")
    print(f"[ANIM] Will run {args.simulations} simulations with different parameters")
    
    # Generate parameter configurations
    configs = generate_parameter_configs(args)
    
    # Run all simulations
    results = run_all_simulations(configs)
    
    # Create animation
    # Update output path to use the new directory structure
    output_dir = Path(project_root) / "output" / "animations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output
    
    create_probability_animation(results, args.tstar_range, output_file=str(output_file))
    
    print("[ANIM] Animation process complete")
    return 0

if __name__ == "__main__":
    main() 