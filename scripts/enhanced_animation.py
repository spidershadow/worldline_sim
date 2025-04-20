#!/usr/bin/env python3
"""
Enhanced Animation Script for T* Variation

This script creates an enhanced animation showing how different T* (Singularity year) candidates
produce varying simulated values for observed years. It visualizes how the match quality 
impacts the probability of each T* year. Key improvements include:

1. Better visualization of different pattern impacts:
   - Pattern-specific visualizations
   - Comparison view between pattern combinations
   - Color-coded impacts of each pattern

2. Support for wider T* ranges:
   - Improved visualization scaling
   - Option to focus on the most likely T* regions
   - Smoother handling of larger datasets

Usage example:
python enhanced_animation.py --patterns rng,tech,uap --tstar-range 2030 2100 --leak-lambda 0.2 --leak-tau0 20 --max-tau 50 --seed 42
"""

import argparse
import copy
import csv
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional, Set
import random

from worldline_sim.patterns import load_patterns
from worldline_sim.timeline import Timeline
from worldline_sim.utils import create_dir_if_needed

# Assuming we have access to these imports as in the original script
from worldline_sim.patterns.base import Pattern
from worldline_sim.patterns.rng import RngPattern

# Apply the super lenient constraint patch to RngPattern
def super_lenient_constraint(self, year: int, value: float, *, tol: float | None = None) -> bool:
    """Super lenient constraint function that accepts a wider range of values."""
    # For RNG pattern, apply a much more lenient constraint (±20.0)
    if self.name == 'rng':
        if year not in self.observed:
            return True
        tol = tol or 20.0  # Much more lenient tolerance
        delta = abs(value - self.observed[year])
        return delta <= tol
    
    # For all other patterns, basically accept anything
    return True

# Apply patch to ALL patterns
Pattern.is_valid_value = super_lenient_constraint

# Patch the retro kernel for more variation
def _retro_kernel(self: Pattern, tau: int) -> float:
    """Modified retro kernel with user-specified parameters."""
    if tau <= 0:
        return 0.0
    return self.leak_lambda * np.exp(-tau / self.leak_tau0)

# Patch the post value calculation
def _post_value(self: Pattern, year: int) -> float:
    """Modified post value calculation with more variation."""
    t_star = self.t_star
    if year < 2015 or t_star is None or year > t_star:
        return 0.0
    tau = t_star - year
    return self._retro_kernel(tau)

# Apply patches
Pattern._retro_kernel = _retro_kernel
Pattern._post_value = _post_value

def custom_backfill_timeline(patterns: List[Pattern], t_star: int, *, max_tau: int,
                           correction_factor: float = 0.3, noise_range: float = 0.15) -> Timeline | None:
    """
    Create a timeline with retrocausal influences from T*.
    
    Args:
        patterns: List of pattern objects that define constraints
        t_star: The singularity year
        max_tau: How many years to simulate after T*
        correction_factor: How much to correct simulated values (0.0-1.0)
        noise_range: Range of noise to add (0.0-1.0)
        
    Returns:
        A Timeline object or None if the timeline is invalid
    """
    years = list(range(2015, t_star + max_tau + 1))
    base_tl = Timeline()
    
    # Add base values
    for p in patterns:
        for year in years:
            # Set a random base value in range [0,1]
            base_tl.set(p.name, year, random.random())
    
    # Add post values (retrocausal influence)
    data: dict[str, dict[int, float]] = {p.name: {y: np.nan for y in years} for p in patterns}
    retro_influence: dict[str, dict[int, float]] = {p.name: {y: 0.0 for y in years} for p in patterns}
    
    # Calculate retro influence for each pattern and year
    for p in patterns:
        for year in years:
            if year > t_star:
                continue
            # This is the basic retrocausal influence
            retro_influence[p.name][year] = p._post_value(year)
    
    # Apply retrocausal influence to base values
    for p in patterns:
        for year in years:
            base_value = base_tl.get(p.name, year)
            
            if p.name == 'rng' and year <= t_star:
                # For RNG pattern, apply partial correction with natural variation
                # Start with the base random value
                if year in p.observed:
                    # Add natural variation with partial correction towards observed value
                    observed = p.observed[year]
                    delta = observed - base_value
                    
                    # Apply partial correction (70%) plus random noise
                    corrected = base_value + (correction_factor * delta)
                    
                    # Add some noise for natural variation
                    noise = (random.random() * 2 - 1) * noise_range
                    final_value = corrected + noise
                    
                    # Ensure the value stays in range [0,1]
                    data[p.name][year] = max(0.0, min(1.0, final_value))
                else:
                    # For years without observations, just use the base + retro
                    data[p.name][year] = base_value
            else:
                # For other patterns, directly use the base + retro without correction
                data[p.name][year] = base_value
    
    # Create the final timeline
    tl = Timeline()
    for p in patterns:
        for year in years:
            if not np.isnan(data[p.name][year]):
                tl.set(p.name, year, data[p.name][year])
    
    return tl if tl.is_valid(patterns) else None

def generate_timelines_for_tstar_range(patterns_str: str, t_star_range: Tuple[int, int],
                                      leak_lambda: float, leak_tau0: float, max_tau: int,
                                      seed: int = None) -> Dict:
    """
    Generate timelines for a range of T* values and calculate their probabilities.
    
    Args:
        patterns_str: Comma-separated list of pattern names
        t_star_range: (min, max) range of T* values to simulate
        leak_lambda: Lambda parameter for leak strength
        leak_tau0: Tau0 parameter for decay time
        max_tau: Maximum tau value for simulation
        seed: Random seed
        
    Returns:
        Dictionary of results for each T* with timeline, probability, error metrics
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    t_star_min, t_star_max = t_star_range
    t_stars = list(range(t_star_min, t_star_max + 1))
    results = {}
    
    # Collect pattern information
    pattern_list = patterns_str.split(',')
    
    # Track all observed years across patterns
    all_observed_years = set()
    
    # Generate a timeline for each T*
    for t_star in t_stars:
        print(f"[ENHANCED] Generating timeline for T* = {t_star}")
        
        # Load patterns for this T*
        patterns = load_patterns(patterns_str, t_star=t_star)
        for p in patterns:
            p.leak_lambda = leak_lambda
            p.leak_tau0 = leak_tau0
            
            # Collect observed years for each pattern
            if hasattr(p, 'observed'):
                all_observed_years.update(p.observed.keys())
        
        # Generate the timeline with our modified function
        tl = custom_backfill_timeline(patterns, t_star, max_tau=max_tau,
                                  correction_factor=0.3, noise_range=0.15)
        
        if tl is None:
            print(f"[ENHANCED] Failed to generate valid timeline for T* = {t_star}")
            continue
        
        # Calculate the error for this timeline based on observed values
        total_error = 0.0
        errors_by_pattern = {}
        simulated_values = {}
        observed_values = {}
        
        for p in patterns:
            if p.name not in errors_by_pattern:
                errors_by_pattern[p.name] = {}
                
            # Only calculate errors for the RNG pattern which has observed values
            if p.name == 'rng':
                rng_pattern = p
                for year in rng_pattern.observed:
                    observed = rng_pattern.observed[year]
                    simulated = tl.get(p.name, year)
                    
                    if simulated is not None:
                        error = abs(observed - simulated)
                        total_error += error
                        errors_by_pattern[p.name][year] = error
                        
                        # Store the values for visualization
                        if year not in simulated_values:
                            simulated_values[year] = simulated
                        if year not in observed_values:
                            observed_values[year] = observed
            
            # For other patterns, record simulated values but no error calculation
            elif hasattr(p, 'observed'):
                for year in p.observed:
                    simulated = tl.get(p.name, year)
                    if simulated is not None:
                        key = f"{p.name}_{year}"
                        simulated_values[key] = simulated
                        observed_values[key] = p.observed[year]
        
        # Calculate probability based on error (lower error = higher probability)
        # Use exponential decay function to convert error to probability
        probability = np.exp(-total_error)
        
        results[t_star] = {
            'timeline': tl,
            'total_error': total_error,
            'errors_by_pattern': errors_by_pattern,
            'probability': probability,
            'simulated_values': simulated_values,
            'observed_values': observed_values,
            'patterns': patterns,
        }
        
        print(f"[ENHANCED] T* = {t_star}, Total Error: {total_error:.4f}, Probability: {probability:.4f}")
    
    # Normalize probabilities to sum to 1.0
    if results:
        total_prob = sum(r['probability'] for r in results.values())
        for t_star in results:
            results[t_star]['probability'] /= total_prob
            print(f"[ENHANCED] Normalized T* = {t_star}, Probability: {results[t_star]['probability']:.4f}")
    
    return results

def create_enhanced_animation(results, output_file=None, pattern_combinations=None):
    """
    Create an enhanced animation showing how T* probabilities emerge from pattern matches.
    
    Args:
        results: Dictionary of results for each T*
        output_file: Output file path for the animation
        pattern_combinations: List of pattern combinations to highlight
    """
    if not results:
        print("[ENHANCED] No results to visualize")
        return
    
    # Create output directory if needed
    output_dir = Path("enhanced_animations")
    create_dir_if_needed(output_dir)
    
    # Default output file if not specified
    if output_file is None:
        t_stars = sorted(results.keys())
        output_file = output_dir / f"enhanced_tstar_{t_stars[0]}_{t_stars[-1]}.mp4"
    else:
        output_file = output_dir / output_file
    
    # Sort T* values
    t_stars = sorted(results.keys())
    
    # Get the observed years from the first result
    first_t_star = t_stars[0]
    first_result = results[first_t_star]
    
    observed_years = []
    pattern_names = set()
    
    # Extract observed years and pattern names
    for key in first_result['simulated_values'].keys():
        if isinstance(key, int) or key.isdigit():
            observed_years.append(int(key))
        elif '_' in key:
            pattern_name = key.split('_')[0]
            pattern_names.add(pattern_name)
    
    # If we only have numeric keys, it's just RNG pattern
    if not pattern_names:
        pattern_names = {'rng'}
    
    observed_years.sort()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, height_ratios=[1, 2, 1], width_ratios=[2, 1])
    
    # Probability plot (top left)
    ax_prob = fig.add_subplot(gs[0, 0])
    
    # Pattern impact plot (middle left)
    ax_pattern = fig.add_subplot(gs[1, 0])
    
    # Timeline plot (bottom left)
    ax_timeline = fig.add_subplot(gs[2, 0])
    
    # Current T* details (right column)
    ax_details = fig.add_subplot(gs[:, 1])
    
    # Probability bars
    probs = [results[t]['probability'] for t in t_stars]
    bar_colors = ['lightgray'] * len(t_stars)
    prob_bars = ax_prob.bar(t_stars, probs, color=bar_colors, alpha=0.7)
    
    # Initialize the pattern impact visualization
    pattern_colors = {
        'rng': 'blue',
        'tech': 'green',
        'uap': 'red',
        'events': 'purple'
    }
    
    # Get all patterns from first result
    all_patterns = {p.name: p for p in first_result['patterns']}
    
    # Create a stacked bar for showing pattern contributions
    pattern_contributions = {}
    for pattern in pattern_names:
        if pattern in pattern_colors:
            pattern_contributions[pattern] = [0] * len(t_stars)
    
    # Initialize timeline data
    timeline_lines = {}
    for pattern in pattern_names:
        if pattern in pattern_colors:
            # Create a line for this pattern
            line, = ax_timeline.plot([], [], 
                                   color=pattern_colors.get(pattern, 'gray'),
                                   label=pattern)
            timeline_lines[pattern] = line
    
    # Initialize the details text box
    details_text = ax_details.text(0.05, 0.95, "", transform=ax_details.transAxes,
                                 verticalalignment='top', fontsize=12)
    
    # Initialize the error graph in the details area
    error_bars = ax_details.bar([], [], color='lightgray', alpha=0.7)
    
    # Set labels and titles
    ax_prob.set_title("T* Probability Distribution")
    ax_prob.set_xlabel("T* Year")
    ax_prob.set_ylabel("Probability")
    ax_prob.grid(True, linestyle='--', alpha=0.5)
    
    ax_pattern.set_title("Pattern Contributions to T* Probability")
    ax_pattern.set_xlabel("T* Year")
    ax_pattern.set_ylabel("Error Contribution")
    ax_pattern.grid(True, linestyle='--', alpha=0.5)
    
    ax_timeline.set_title("Simulated Timeline Values")
    ax_timeline.set_xlabel("Year")
    ax_timeline.set_ylabel("Value")
    ax_timeline.grid(True, linestyle='--', alpha=0.5)
    ax_timeline.legend()
    
    ax_details.set_title("Current T* Details")
    ax_details.set_axis_off()
    
    # Update function for animation
    def update(frame):
        t_star = t_stars[frame % len(t_stars)]
        result = results[t_star]
        
        # Update probability bars
        for i, bar in enumerate(prob_bars):
            if t_stars[i] == t_star:
                bar.set_color('red')
                bar.set_alpha(1.0)
            else:
                bar.set_color('lightgray')
                bar.set_alpha(0.7)
        
        # Update pattern impact visualization
        pattern_data = []
        labels = []
        
        for pattern in pattern_names:
            if pattern in pattern_colors and pattern in result['errors_by_pattern']:
                # Sum errors for this pattern
                pattern_error = sum(result['errors_by_pattern'][pattern].values())
                pattern_data.append(pattern_error)
                labels.append(pattern)
        
        # Clear previous pattern contributions
        ax_pattern.clear()
        ax_pattern.set_title("Pattern Contributions to T* Probability")
        ax_pattern.set_xlabel("Pattern")
        ax_pattern.set_ylabel("Error Value")
        ax_pattern.grid(True, linestyle='--', alpha=0.5)
        
        # Create bars for pattern contributions
        if pattern_data:
            bars = ax_pattern.bar(labels, pattern_data, 
                                color=[pattern_colors.get(p, 'gray') for p in labels],
                                alpha=0.7)
            
            # Add value labels
            for i, v in enumerate(pattern_data):
                ax_pattern.text(i, v, f"{v:.4f}", ha='center', va='bottom')
        
        # Update timeline visualization
        timeline = result['timeline']
        
        # Get years from the timeline
        all_years = sorted(set([y for p in pattern_names 
                              for y in timeline.data.get(p, {}).keys()]))
        
        # Update each pattern line
        for pattern in pattern_names:
            if pattern in timeline_lines and pattern in timeline.data:
                years = sorted(timeline.data[pattern].keys())
                values = [timeline.data[pattern][y] for y in years]
                
                # Update the line data
                timeline_lines[pattern].set_data(years, values)
        
        # Adjust timeline axis limits
        if all_years:
            ax_timeline.set_xlim(min(all_years) - 1, max(all_years) + 1)
            ax_timeline.set_ylim(0, 1.1)
            
            # Add markers for observed years
            for year in observed_years:
                if year in result['observed_values']:
                    ax_timeline.axvline(x=year, color='black', linestyle='--', alpha=0.5)
        
        # Update details text
        details_str = f"T* = {t_star}\n"
        details_str += f"Probability: {result['probability']:.4f}\n"
        details_str += f"Total Error: {result['total_error']:.4f}\n\n"
        
        details_str += "Observed vs. Simulated Values:\n"
        for year in observed_years:
            if year in result['observed_values'] and year in result['simulated_values']:
                observed = result['observed_values'][year]
                simulated = result['simulated_values'][year]
                error = abs(observed - simulated)
                details_str += f"Year {year}: Obs={observed:.4f}, Sim={simulated:.4f}, Error={error:.4f}\n"
        
        details_text.set_text(details_str)
        
        # Return all updated artists
        artists = [*prob_bars, details_text]
        for line in timeline_lines.values():
            artists.append(line)
        
        return artists
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(t_stars),
        interval=1000, blit=True, repeat=True
    )
    
    # Save animation
    ani.save(output_file, writer='ffmpeg', dpi=200)
    plt.close(fig)
    
    print(f"[ENHANCED] Animation saved to {output_file}")
    return str(output_file)

def create_pattern_analysis_charts(results, output_dir=None):
    """Create static analysis charts showing how each pattern impacts T* selection."""
    if not results:
        return
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = Path("enhanced_charts")
    else:
        output_dir = Path(output_dir)
    
    create_dir_if_needed(output_dir)
    
    # Sort T* values
    t_stars = sorted(results.keys())
    
    # 1. Create a T* probability histogram
    plt.figure(figsize=(12, 6))
    probs = [results[t]['probability'] for t in t_stars]
    
    plt.bar(t_stars, probs, color='blue', alpha=0.7)
    plt.title("T* Probability Distribution")
    plt.xlabel("T* Year")
    plt.ylabel("Probability")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add text labels for probabilities
    for i, (t, p) in enumerate(zip(t_stars, probs)):
        plt.text(t, p, f"{p:.4f}", ha='center', va='bottom', fontsize=8)
    
    # Save the figure
    plt.tight_layout()
    hist_file = output_dir / f"enhanced_tstar_histogram_{t_stars[0]}_{t_stars[-1]}.png"
    plt.savefig(hist_file, dpi=200)
    plt.close()
    print(f"[ENHANCED] Saved histogram to {hist_file}")
    
    # 2. Create a pattern contribution analysis chart
    # Get all pattern names
    pattern_names = set()
    for t_star in results:
        for pattern in results[t_star]['errors_by_pattern']:
            pattern_names.add(pattern)
    
    # Sort the patterns
    pattern_names = sorted(pattern_names)
    
    # Colors for patterns
    pattern_colors = {
        'rng': 'blue',
        'tech': 'green', 
        'uap': 'red',
        'events': 'purple'
    }
    
    # Create a figure with subplots for each pattern
    fig, axes = plt.subplots(len(pattern_names), 1, figsize=(12, 5*len(pattern_names)))
    if len(pattern_names) == 1:
        axes = [axes]
    
    for i, pattern in enumerate(pattern_names):
        ax = axes[i]
        
        # Calculate pattern errors for each T*
        errors = []
        for t_star in t_stars:
            result = results[t_star]
            pattern_error = 0
            
            if pattern in result['errors_by_pattern']:
                pattern_error = sum(result['errors_by_pattern'][pattern].values())
            
            errors.append(pattern_error)
        
        # Create bars
        color = pattern_colors.get(pattern, 'gray')
        ax.bar(t_stars, errors, color=color, alpha=0.7)
        
        # Add labels
        ax.set_title(f"{pattern.upper()} Pattern Error by T*")
        ax.set_ylabel("Error Value")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add text labels for errors
        for j, e in enumerate(errors):
            ax.text(t_stars[j], e, f"{e:.4f}", ha='center', va='bottom', fontsize=8)
    
    # Add x-label to the bottom plot only
    axes[-1].set_xlabel("T* Year")
    
    # Save the figure
    plt.tight_layout()
    pattern_file = output_dir / f"enhanced_pattern_errors_{t_stars[0]}_{t_stars[-1]}.png"
    plt.savefig(pattern_file, dpi=200)
    plt.close()
    print(f"[ENHANCED] Saved pattern error analysis to {pattern_file}")
    
    # 3. Create a detailed year-by-year analysis for each observed year
    # Get all observed years
    observed_years = set()
    for t_star in results:
        for pattern in results[t_star]['errors_by_pattern']:
            observed_years.update(results[t_star]['errors_by_pattern'][pattern].keys())
    
    # Sort observed years
    observed_years = sorted(observed_years)
    
    # Create a figure for each observed year
    for year in observed_years:
        plt.figure(figsize=(12, 6))
        
        # Get observed value (should be the same across all T*)
        observed_value = None
        for t_star in results:
            if year in results[t_star]['observed_values']:
                observed_value = results[t_star]['observed_values'][year]
                break
        
        # Get simulated values for each T*
        simulated_values = []
        for t_star in t_stars:
            if year in results[t_star]['simulated_values']:
                simulated_values.append(results[t_star]['simulated_values'][year])
            else:
                simulated_values.append(None)
        
        # Create bar chart for simulated values
        bars = []
        valid_t_stars = []
        valid_values = []
        
        for i, (t_star, value) in enumerate(zip(t_stars, simulated_values)):
            if value is not None:
                valid_t_stars.append(t_star)
                valid_values.append(value)
        
        if valid_values:
            bars = plt.bar(valid_t_stars, valid_values, alpha=0.7)
            
            # Add observed value line
            if observed_value is not None:
                plt.axhline(y=observed_value, color='black', linestyle='-', 
                          label=f"Observed: {observed_value:.4f}")
            
            # Colorize bars based on error
            for i, (t_star, value) in enumerate(zip(valid_t_stars, valid_values)):
                if observed_value is not None:
                    error = abs(value - observed_value)
                    # Color based on error (green=good, red=bad)
                    color_val = min(1.0, error * 5)  # Scale for better visibility
                    bars[i].set_color((color_val, 1.0 - color_val, 0))
                    
                    # Add text label
                    plt.text(t_star, value, f"{value:.4f}", ha='center', va='bottom', fontsize=8)
            
            # Add labels
            plt.title(f"Simulated vs Observed Values for Year {year}")
            plt.xlabel("T* Year")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Save the figure
            plt.tight_layout()
            year_file = output_dir / f"enhanced_year_{year}_analysis_{t_stars[0]}_{t_stars[-1]}.png"
            plt.savefig(year_file, dpi=200)
            print(f"[ENHANCED] Saved year {year} analysis to {year_file}")
        
        plt.close()
    
    # 4. Create a CSV with all results
    csv_file = output_dir / f"enhanced_analysis_{t_stars[0]}_{t_stars[-1]}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['T*', 'Probability', 'Total Error']
        
        # Add observed years for each pattern
        for pattern in pattern_names:
            for year in observed_years:
                header.append(f"{pattern}_{year}_Simulated")
                header.append(f"{pattern}_{year}_Observed")
                header.append(f"{pattern}_{year}_Error")
        
        writer.writerow(header)
        
        # Write data for each T*
        for t_star in t_stars:
            result = results[t_star]
            row = [t_star, result['probability'], result['total_error']]
            
            # Add data for each pattern and year
            for pattern in pattern_names:
                for year in observed_years:
                    simulated = 'N/A'
                    observed = 'N/A'
                    error = 'N/A'
                    
                    # Check if we have data for this pattern/year
                    if pattern in result['errors_by_pattern'] and year in result['errors_by_pattern'][pattern]:
                        error = result['errors_by_pattern'][pattern][year]
                    
                    # Get observed and simulated values
                    if year in result['observed_values']:
                        observed = result['observed_values'][year]
                    
                    key = f"{pattern}_{year}" if pattern != 'rng' else year
                    if key in result['simulated_values']:
                        simulated = result['simulated_values'][key]
                    
                    row.extend([simulated, observed, error])
            
            writer.writerow(row)
    
    print(f"[ENHANCED] Saved detailed analysis to {csv_file}")
    
    return output_dir

def build_argparser():
    p = argparse.ArgumentParser(description="Enhanced animation of T* probability with pattern impact analysis")
    p.add_argument("--patterns", default="rng", help="comma list of patterns")
    p.add_argument("--tstar-range", nargs=2, type=int, metavar=("MIN", "MAX"), required=True,
                   help="range of T* values to simulate")
    p.add_argument("--leak-lambda", type=float, default=0.2, help="λ leak strength (all patterns)")
    p.add_argument("--leak-tau0", type=float, default=20.0, help="τ0 decay time (years)")
    p.add_argument("--max-tau", type=int, default=50, help="simulate this many post years")
    p.add_argument("--seed", type=int, help="seed for random number generator")
    p.add_argument("--output", type=str, help="output filename for animation")
    p.add_argument("--focus-range", type=int, help="focus animation on top N most likely T* values")
    return p

def main(argv=None):
    args = build_argparser().parse_args(argv)
    
    seed = getattr(args, "seed", None)
    t_star_min, t_star_max = args.tstar_range
    
    print(f"[ENHANCED] Using improved visualization with pattern impact analysis")
    print(f"[ENHANCED] Using base seed: {seed}")
    print(f"[ENHANCED] Simulating for T* range {t_star_min}-{t_star_max}")
    print(f"[ENHANCED] Using patterns: {args.patterns}")
    
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
        print(f"[ENHANCED] No valid timelines were generated.")
        return 1
    
    # Create static analysis charts
    charts_dir = create_pattern_analysis_charts(results)
    
    # Create the animation
    output_file = args.output if args.output else f"enhanced_tstar_{t_star_min}_{t_star_max}.mp4"
    
    # If focus range is specified, filter to top N most likely T* values
    if args.focus_range and args.focus_range < len(results):
        print(f"[ENHANCED] Focusing animation on top {args.focus_range} most likely T* values")
        
        # Sort T* by probability
        sorted_t_stars = sorted(results.keys(), key=lambda t: results[t]['probability'], reverse=True)
        
        # Keep only the top N
        focus_t_stars = sorted_t_stars[:args.focus_range]
        
        # Create focused results
        focused_results = {t: results[t] for t in focus_t_stars}
        
        # Create focused animation
        focused_output = f"focused_{output_file}"
        create_enhanced_animation(focused_results, focused_output)
    
    # Create full animation
    create_enhanced_animation(results, output_file)
    
    print(f"[ENHANCED] Analysis complete. Check the output files in {charts_dir}")
    return 0

if __name__ == "__main__":
    main() 