#!/usr/bin/env python3
"""Run pattern comparison analysis across different pattern combinations.

This script runs the animate_window_scoring.py script with different pattern combinations
(UAP, RNG, quantum_volume, etc.) and then collates and compares the results to identify
which patterns show the strongest signal for specific time windows.

Example
-------
python scripts/run_pattern_comparison.py \
       --tstar-range 2030 2100 --runs 50 --window 20 --alpha 5.0 --seed 123
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

def run_simulation(patterns, tstar_range, runs, window, alpha, seed, output_prefix):
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
    
    print(f"[COMPARE] Running simulation with patterns: {patterns}")
    subprocess.run(cmd, check=True)
    print(f"[COMPARE] Completed simulation with patterns: {patterns}")
    
    # Return the path to the results CSV
    results_path = PROJECT_ROOT / "data" / output_prefix / f"{output_prefix}_results.csv"
    return results_path

def extract_final_probabilities(results_path, tstar_range):
    """Extract final probability distribution from results CSV."""
    if not results_path.exists():
        print(f"[COMPARE] WARNING: Results file not found: {results_path}")
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

def create_comparison_plot(results_dict, tstar_range, output_dir):
    """Create comparison plot of probability distributions from different pattern runs."""
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up x-axis
    years = list(range(tstar_range[0], tstar_range[1] + 1))
    x = np.arange(len(years))
    width = 0.8 / len(results_dict)
    
    # Colors for different patterns
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    
    # Plot bars for each pattern set
    for i, (pattern_name, data) in enumerate(results_dict.items()):
        if data is not None:
            offset = i * width - 0.4 + width/2
            ax.bar(x + offset, data['probability'], width, label=pattern_name, color=colors[i % len(colors)])
    
    # Add most likely year annotations
    for pattern_name, data in results_dict.items():
        if data is not None and data['probability'].max() > 0:
            most_likely_idx = data['probability'].idxmax()
            most_likely_year = data.loc[most_likely_idx, 't_star']
            most_likely_prob = data.loc[most_likely_idx, 'probability']
            print(f"[COMPARE] Pattern '{pattern_name}' -> Most likely T*: {most_likely_year} (p={most_likely_prob:.4f})")
    
    # Set chart labels and properties
    ax.set_xlabel('Candidate Singularity Year (T*)')
    ax.set_ylabel('Posterior Probability')
    ax.set_title('Comparison of T* Distributions Across Different Pattern Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    output_path = output_dir / "pattern_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"[COMPARE] Comparison plot saved to {output_path}")
    return output_path

def create_most_likely_year_table(results_dict, output_dir):
    """Create and save a table of most likely T* years for each pattern combination."""
    data = []
    for pattern_name, df in results_dict.items():
        if df is not None and df['probability'].max() > 0:
            most_likely_idx = df['probability'].idxmax()
            most_likely_year = df.loc[most_likely_idx, 't_star']
            most_likely_prob = df.loc[most_likely_idx, 'probability']
            data.append({
                'Pattern': pattern_name,
                'Most Likely T*': most_likely_year,
                'Probability': most_likely_prob
            })
    
    if data:
        # Create and save DataFrame
        results_df = pd.DataFrame(data)
        results_df = results_df.sort_values('Probability', ascending=False)
        
        # Save to CSV
        csv_path = output_dir / "most_likely_years.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"[COMPARE] Most likely years table saved to {csv_path}")
        
        # Also generate a simple HTML table
        html_path = output_dir / "most_likely_years.html"
        with open(html_path, 'w') as f:
            f.write("<html><body>\n")
            f.write("<h2>Most Likely Singularity Years by Pattern</h2>\n")
            f.write(results_df.to_html(index=False))
            f.write("</body></html>")
        
        return results_df
    else:
        print("[COMPARE] No valid results to create a table")
        return None

def build_argparser():
    p = argparse.ArgumentParser(description="Run pattern comparison analysis")
    p.add_argument("--tstar-range", nargs=2, type=int, required=True, metavar=("MIN", "MAX"),
                   help="candidate T* range (inclusive)")
    p.add_argument("--runs", type=int, default=50, help="Monte‑Carlo trajectories per pattern")
    p.add_argument("--window", type=int, default=20, help="±years to include in symmetry window")
    p.add_argument("--alpha", type=float, default=5.0, help="weight sharpness: w = exp(-α⋅err)")
    p.add_argument("--seed", type=int, default=123, help="base RNG seed")
    return p

def main(argv=None):
    args = build_argparser().parse_args(argv)
    
    # Set up timestamp and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"pattern_comparison_{timestamp}"
    output_dir = PROJECT_ROOT / "output" / output_base
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define pattern combinations to test
    pattern_sets = [
        "uap",                      # UAP only (baseline)
        "rng",                      # RNG only
        "quantum_volume",           # Quantum volume only
        "uap,rng",                  # UAP + RNG
        "uap,quantum_volume",       # UAP + Quantum volume
        "rng,quantum_volume",       # RNG + Quantum volume
        "uap,rng,quantum_volume"    # All patterns
    ]
    
    # Store results paths for each pattern set
    results_files = {}
    
    # Run simulations for each pattern set
    for i, patterns in enumerate(pattern_sets):
        pattern_prefix = f"{output_base}_patterns_{patterns.replace(',', '_')}"
        try:
            results_path = run_simulation(
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
            print(f"[COMPARE] Error running simulation for patterns '{patterns}': {e}")
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
        create_comparison_plot(results_dict, args.tstar_range, output_dir)
        create_most_likely_year_table(results_dict, output_dir)
        
        # Write summary file
        with open(output_dir / "summary.txt", "w") as f:
            f.write(f"Pattern Comparison Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"T* range: {args.tstar_range[0]}-{args.tstar_range[1]}\n")
            f.write(f"Runs per pattern: {args.runs}\n")
            f.write(f"Window size: {args.window}\n")
            f.write(f"Alpha: {args.alpha}\n\n")
            
            f.write("Most Likely Singularity Years:\n")
            for pattern_name, data in results_dict.items():
                if data is not None and data['probability'].max() > 0:
                    most_likely_idx = data['probability'].idxmax()
                    most_likely_year = data.loc[most_likely_idx, 't_star']
                    most_likely_prob = data.loc[most_likely_idx, 'probability']
                    f.write(f"* {pattern_name}: T* = {most_likely_year} (p = {most_likely_prob:.4f})\n")
        
        print("\n--- PATTERN COMPARISON SUMMARY ---")
        print(f"Analyzed {len(results_dict)} pattern combinations")
        print(f"Results saved to: {output_dir}")
        print("--------------------------------\n")
    else:
        print("[COMPARE] No valid results to compare")

if __name__ == "__main__":  # pragma: no cover
    main() 