#!/usr/bin/env python3
"""Run comparison analysis with enhanced RNG pattern using GCP methodology.

This script runs the window scoring analysis with a focus on the enhanced RNG pattern
that uses the comprehensive Global Consciousness Project data. It tests for potential
Singularity years around 2010-2020, as the previous analysis showed strong signals
in this range.

Example
-------
python scripts/run_enhanced_rng_comparison.py \
       --tstar-range 2010 2020 --runs 50 --window 15 --alpha 5.0 --seed 123
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup
from worldline_sim.patterns import load_patterns
from worldline_sim.patterns.base import Pattern


def run_window_scoring_simulation(patterns, tstar_range, runs, window, alpha, seed, output_prefix):
    """Run animate_window_scoring.py with the given parameters."""
    cmd = [
        "python", 
        "scripts/animate_window_scoring.py",
        "--patterns", patterns,
        "--tstar-range", str(tstar_range[0]), str(tstar_range[1]),
        "--runs", str(runs),
        "--window", str(window),
        "--alpha", str(alpha),
        "--seed", str(seed),
        "--output-prefix", output_prefix
    ]
    
    print(f"[ENHANCED] Running simulation with patterns: {patterns}")
    subprocess.run(cmd, check=True)
    print(f"[ENHANCED] Completed simulation with patterns: {patterns}")
    
    # Return the path to the results CSV
    results_path = PROJECT_ROOT / "data" / output_prefix / f"{output_prefix}_results.csv"
    return results_path


def analyze_gcp_data_symmetry(tstar_range):
    """Analyze the GCP data directly for symmetry around potential T* years.
    
    This function implements the cumulative variance analysis methodology from GCP
    to test if there's statistical evidence for symmetry around candidate T* years.
    """
    # Load the RNG data
    data_path = PROJECT_ROOT / "data" / "rng_avg_abs_z.csv"
    if not data_path.exists():
        print(f"[ENHANCED] GCP data file not found: {data_path}")
        return {}
    
    df = pd.read_csv(data_path)
    df['year'] = df.iloc[:, 0]
    df['avg_abs_z'] = df.iloc[:, 1]
    
    # Calculate baseline for deviation measurements
    baseline = 0.8  # Expected value under null hypothesis
    
    # Store symmetry scores for each candidate T*
    symmetry_scores = {}
    chi_square_values = {}
    
    for t_star in range(tstar_range[0], tstar_range[1] + 1):
        # Skip if T* is too recent for meaningful analysis
        if t_star > 2020:
            continue
            
        # Get years before and after T*
        pre_t_star = df[df['year'] < t_star]
        post_t_star = df[df['year'] > t_star]
        
        # Skip if insufficient data on either side
        if len(pre_t_star) < 3 or len(post_t_star) < 3:
            continue
        
        # Calculate deviations from baseline
        pre_t_star['deviation'] = (pre_t_star['avg_abs_z'] - baseline).abs()
        post_t_star['deviation'] = (post_t_star['avg_abs_z'] - baseline).abs()
        
        # Get average deviations
        pre_avg_dev = pre_t_star['deviation'].mean()
        post_avg_dev = post_t_star['deviation'].mean()
        
        # Calculate symmetry score (1.0 = perfect symmetry)
        if pre_avg_dev > 0 and post_avg_dev > 0:
            symmetry_score = min(pre_avg_dev/post_avg_dev, post_avg_dev/pre_avg_dev)
        else:
            symmetry_score = 0
            
        symmetry_scores[t_star] = symmetry_score
        
        # Calculate chi-square for entire dataset (cumulative variance)
        all_years = df.copy()
        all_years['squared_dev'] = (all_years['avg_abs_z'] - baseline) ** 2
        chi_square = all_years['squared_dev'].sum()
        chi_square_values[t_star] = chi_square
        
        # Print detailed analysis for this T*
        print(f"[ENHANCED] T* = {t_star} analysis:")
        print(f"  Pre-T* avg deviation:  {pre_avg_dev:.6f}")
        print(f"  Post-T* avg deviation: {post_avg_dev:.6f}")
        print(f"  Symmetry score:        {symmetry_score:.6f}")
        print(f"  Chi-square (cum var):  {chi_square:.6f}")
        
        # For the window around T*, show yearly breakdown
        window_size = 5
        window_years = all_years[(all_years['year'] >= t_star - window_size) & 
                                (all_years['year'] <= t_star + window_size)]
        if not window_years.empty:
            print(f"  Values in ±{window_size} year window around T*:")
            for _, row in window_years.iterrows():
                year = int(row['year'])
                z = row['avg_abs_z']
                dev = abs(z - baseline)
                diff_t = year - t_star
                print(f"    {year} (T*{diff_t:+d}): z={z:.6f}, dev={dev:.6f}")
    
    return {
        'symmetry_scores': symmetry_scores,
        'chi_square_values': chi_square_values
    }


def extract_final_probabilities(results_path, tstar_range):
    """Extract final probability distribution from results CSV."""
    if not results_path.exists():
        print(f"[ENHANCED] WARNING: Results file not found: {results_path}")
        return None
    
    # Read results
    df = pd.read_csv(results_path)
    
    # Get the final cumulative weights for each T*
    final_weights = df.groupby('t_star')['cumulative_weight'].max().reset_index()
    
    # Normalize to get probabilities
    total_weight = final_weights['cumulative_weight'].sum()
    if total_weight > 0:
        final_weights['probability'] = final_weights['cumulative_weight'] / total_weight
    else:
        final_weights['probability'] = 0
    
    # Ensure we have all years in range
    all_years = pd.DataFrame({'t_star': range(tstar_range[0], tstar_range[1] + 1)})
    final_probs = all_years.merge(final_weights[['t_star', 'probability']], on='t_star', how='left').fillna(0)
    
    return final_probs


def create_comparison_plot(results_dict, gcp_analysis, tstar_range, output_dir):
    """Create comparison plot of probability distributions from different pattern runs."""
    # Create figure with probability and GCP symmetry plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Colors for different patterns
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    # Plot 1: Probability distributions
    years = list(range(tstar_range[0], tstar_range[1] + 1))
    
    for i, (pattern_name, data) in enumerate(results_dict.items()):
        if data is not None:
            ax1.plot(data['t_star'], data['probability'], 'o-', 
                    label=pattern_name, color=colors[i % len(colors)])
    
    # Find and annotate most likely years
    for pattern_name, data in results_dict.items():
        if data is not None and data['probability'].max() > 0:
            most_likely_idx = data['probability'].idxmax()
            most_likely_year = data.loc[most_likely_idx, 't_star']
            most_likely_prob = data.loc[most_likely_idx, 'probability']
            ax1.annotate(f"{most_likely_year}", 
                        xy=(most_likely_year, most_likely_prob),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold')
            print(f"[ENHANCED] Pattern '{pattern_name}' -> Most likely T*: {most_likely_year} (p={most_likely_prob:.4f})")
    
    # Set axis properties
    ax1.set_ylabel('Posterior Probability')
    ax1.set_title('Probability Distribution of Potential Singularity Years (T*)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: GCP Symmetry Scores
    if gcp_analysis and 'symmetry_scores' in gcp_analysis:
        symmetry_years = list(gcp_analysis['symmetry_scores'].keys())
        symmetry_values = list(gcp_analysis['symmetry_scores'].values())
        
        ax2.bar(symmetry_years, symmetry_values, color='tab:gray', alpha=0.7, label='GCP Symmetry Score')
        
        # Find and annotate peak symmetry
        if symmetry_values:
            max_idx = np.argmax(symmetry_values)
            max_year = symmetry_years[max_idx]
            max_value = symmetry_values[max_idx]
            ax2.annotate(f"{max_year}", 
                        xy=(max_year, max_value),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold')
            print(f"[ENHANCED] GCP Analysis -> Highest symmetry at T*: {max_year} (score={max_value:.4f})")
    
    # Set axis properties
    ax2.set_xlabel('Candidate Singularity Year (T*)')
    ax2.set_ylabel('GCP Symmetry Score')
    ax2.set_title('GCP Data Symmetry Analysis (higher = more symmetry around year)')
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = output_dir / "enhanced_rng_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"[ENHANCED] Comparison plot saved to {output_path}")
    return output_path


def build_argparser():
    p = argparse.ArgumentParser(description="Run enhanced RNG pattern analysis")
    p.add_argument("--tstar-range", nargs=2, type=int, default=[2010, 2020], metavar=("MIN", "MAX"),
                   help="candidate T* range (inclusive)")
    p.add_argument("--runs", type=int, default=30, help="Monte‑Carlo trajectories per pattern")
    p.add_argument("--window", type=int, default=15, help="±years to include in symmetry window")
    p.add_argument("--alpha", type=float, default=5.0, help="weight sharpness: w = exp(-α⋅err)")
    p.add_argument("--seed", type=int, default=123, help="base RNG seed")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    
    # Set up timestamp and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"enhanced_rng_{timestamp}"
    output_dir = PROJECT_ROOT / "output" / output_base
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First, perform direct GCP data analysis
    print("\n[ENHANCED] Running direct GCP data analysis...")
    gcp_analysis = analyze_gcp_data_symmetry(args.tstar_range)
    
    # Define pattern combinations to test
    pattern_sets = [
        "enhanced_rng",                    # Enhanced RNG only
        "enhanced_rng,quantum_volume",     # Enhanced RNG + Quantum volume
        "enhanced_rng,uap",                # Enhanced RNG + UAP
    ]
    
    # Store results paths for each pattern set
    results_files = {}
    
    # Run simulations for each pattern set
    for i, patterns in enumerate(pattern_sets):
        pattern_prefix = f"{output_base}_patterns_{patterns.replace(',', '_')}"
        try:
            results_path = run_window_scoring_simulation(
                patterns=patterns,
                tstar_range=args.tstar_range,
                runs=args.runs,
                window=args.window,
                alpha=args.alpha,
                seed=args.seed + i,  # Use different seed for each pattern
                output_prefix=pattern_prefix
            )
            results_files[patterns] = results_path
        except Exception as e:
            print(f"[ENHANCED] Error running simulation for patterns '{patterns}': {e}")
            results_files[patterns] = None
    
    # Extract final probabilities
    results_dict = {}
    for patterns, results_path in results_files.items():
        if results_path is not None:
            probs = extract_final_probabilities(results_path, args.tstar_range)
            if probs is not None:
                results_dict[patterns] = probs
    
    # Generate comparison visualizations
    if results_dict:
        create_comparison_plot(results_dict, gcp_analysis, args.tstar_range, output_dir)
        
        # Write summary file
        with open(output_dir / "summary.txt", "w") as f:
            f.write(f"Enhanced RNG Analysis Summary\n")
            f.write(f"===========================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"T* range: {args.tstar_range[0]}-{args.tstar_range[1]}\n")
            f.write(f"Runs per pattern: {args.runs}\n")
            f.write(f"Window size: {args.window}\n")
            f.write(f"Alpha: {args.alpha}\n\n")
            
            f.write("Most Likely Singularity Years by Pattern:\n")
            for pattern_name, data in results_dict.items():
                if data is not None and data['probability'].max() > 0:
                    most_likely_idx = data['probability'].idxmax()
                    most_likely_year = data.loc[most_likely_idx, 't_star']
                    most_likely_prob = data.loc[most_likely_idx, 'probability']
                    f.write(f"* {pattern_name}: T* = {most_likely_year} (p = {most_likely_prob:.4f})\n")
            
            f.write("\nGCP Symmetry Analysis:\n")
            if gcp_analysis and 'symmetry_scores' in gcp_analysis:
                symmetry_years = list(gcp_analysis['symmetry_scores'].keys())
                symmetry_values = list(gcp_analysis['symmetry_scores'].values())
                
                if symmetry_values:
                    max_idx = np.argmax(symmetry_values)
                    max_year = symmetry_years[max_idx]
                    max_value = symmetry_values[max_idx]
                    f.write(f"* Highest symmetry at T* = {max_year} (score = {max_value:.4f})\n")
                    
                    f.write("\nTop 5 years by symmetry score:\n")
                    top_indices = np.argsort(symmetry_values)[-5:][::-1]
                    for i in top_indices:
                        year = symmetry_years[i]
                        score = symmetry_values[i]
                        f.write(f"* {year}: {score:.4f}\n")
        
        print("\n--- ENHANCED RNG ANALYSIS SUMMARY ---")
        print(f"Analyzed {len(results_dict)} pattern combinations with GCP methodology")
        print(f"Results saved to: {output_dir}")
        print("--------------------------------\n")
    else:
        print("[ENHANCED] No valid results to compare")


if __name__ == "__main__":  # pragma: no cover
    main() 