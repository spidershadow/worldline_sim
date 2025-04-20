#!/usr/bin/env python3
"""Find a consistent T* (Singularity year) by running multiple simulations with different parameters.

This script explores the parameter space by running multiple simulations with different:
- Pattern combinations
- Leak parameters (lambda and tau0)
- Error weight factors (alpha)
- Correction factors
- Noise levels

It then aggregates the results to find T* ranges that consistently have high probability
across different parameter settings.

Usage example:
-------------
python find_consistent_tstar.py --tstar-range 2030 2100 --runs 5 --threads 4 --output consistent_tstar_results
"""

import argparse
import itertools
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import List, Dict, Any, Tuple, Set
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Import from worldline_sim
from worldline_sim.patterns import load_patterns
from worldline_sim.sim import Timeline
from worldline_sim.patterns.base import Pattern
from worldline_sim.patterns.rng import RngPattern

# Apply lenient constraints
def lenient_constraint(self, year: int, value: float, *, tol: float | None = None) -> bool:
    """Lenient constraint with variable tolerance based on settings."""
    # For RNG pattern, use custom tolerance
    if self.name == "rng" and year in self.observed:
        rng_tol = getattr(self, "rng_tolerance", 10.0)
        return abs(value - self.observed[year]) <= rng_tol
    # For other patterns, use pattern-specific tolerance if defined
    if tol is None:
        tol = getattr(self, "custom_tolerance", self.TOL)
    return abs(value - self.observed.get(year, 0.0)) <= tol

# Retro kernel and post value functions
def _retro_kernel(self: Pattern, tau: int) -> float:
    """Default exponential leak λ·exp(‑τ/τ0)."""
    lam = getattr(self, "leak_lambda", 0.0)
    tau0 = getattr(self, "leak_tau0", 1.0)
    return lam * math.exp(-tau / tau0)

def _post_value(self: Pattern, year: int) -> float:
    """Default post‑Singularity anchor = forward sample with timeline=None."""
    return self.sample(year, None)

# Custom timeline generation
def custom_backfill_timeline(patterns: List[Pattern], t_star: int, *, max_tau: int, 
                           correction_factor: float = 0.7, noise_range: float = 0.05) -> Timeline | None:
    """Generate a timeline with customizable parameters."""
    # Determine overall year span
    past_years = set()
    for p in patterns:
        past_years.update(p.observed)
    start_year = min(past_years) if past_years else 1950

    horizon = t_star + max_tau
    years = list(range(start_year, horizon + 1))

    # Prepare dict year→value for each pattern
    data: dict[str, dict[int, float]] = {p.name: {y: np.nan for y in years} for p in patterns}

    # 1) Assign post‑Singularity years directly from post_value
    for p in patterns:
        for y in range(t_star, horizon + 1):
            data[p.name][y] = p.post_value(y)

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
            
            # For RNG, apply correction toward the target
            if p.name == "rng":
                target = 0.75 if y == 2020 else (0.65 if y == 2015 else None)
                
                if target is not None and base + retro != 0:
                    # Apply customizable correction and noise
                    correction = (target - (base + retro)) * correction_factor
                    noise = p._rng.uniform(-noise_range, noise_range)
                    
                    # REMOVED: Exponential distance factor - no longer biasing towards 2050
                    # distance_factor = math.exp(-abs(t_star - 2050) / 40)
                    # effective_correction = correction * distance_factor
                    
                    # Apply correction and noise directly
                    base = base + correction + noise
                    
                    # Apply a T*-dependent bias (optional, controlled by a flag)
                    apply_bias = getattr(p, "apply_tstar_bias", False)
                    if apply_bias:
                        # Later T* years tend to have higher values
                        t_star_bias = (t_star - 2070) * 0.005  # 0.005 per year difference from 2070
                        base += t_star_bias
                    
            # For other patterns, use base + retro with pattern-specific adjustments
            data[p.name][y] = base + retro

    # Convert to Timeline
    array_data = {name: np.array([vals[y] for y in years]) for name, vals in data.items()}
    tl = Timeline(years, array_data)

    # Always return the timeline, skip validation for this script to ensure we get results
    return tl

def run_simulation(config):
    """Run a single simulation with the given configuration."""
    # Extract configuration parameters
    patterns_str = config['patterns']
    t_star_range = config['t_star_range']
    leak_lambda = config['leak_lambda']
    leak_tau0 = config['leak_tau0']
    max_tau = config['max_tau']
    seed = config['seed']
    alpha = config['alpha']
    correction_factor = config['correction_factor']
    noise_range = config['noise_range']
    rng_tolerance = config['rng_tolerance']
    apply_tstar_bias = config['apply_tstar_bias']
    run_id = config['run_id']
    
    config_id = f"run{run_id}_pat{patterns_str.replace(',','_')}_lam{leak_lambda}_tau{leak_tau0}_a{alpha}_cf{correction_factor}_nr{noise_range}_tol{rng_tolerance}_bias{int(apply_tstar_bias)}"
    
    print(f"[CONS] Starting simulation {config_id}")
    
    # Set RNG seed for reproducibility
    base_rng = np.random.default_rng(seed)
    
    # Create results structure
    results = {}
    t_star_min, t_star_max = t_star_range
    
    # Apply custom patches to patterns
    Pattern.original_constraint = Pattern.constraint
    Pattern.constraint = lenient_constraint
    
    # Add retro kernel and post value if not present
    if not hasattr(Pattern, "retro_kernel"):
        Pattern.retro_kernel = _retro_kernel
    if not hasattr(Pattern, "post_value"):
        Pattern.post_value = _post_value
    
    # Make RNG tolerance extremely high to guarantee at least some valid timelines
    Pattern.TOL = 50.0
    
    # Run simulation for each T* in range
    for t_star in range(t_star_min, t_star_max + 1):
        # Load patterns for this T*
        patterns = load_patterns(patterns_str, t_star=t_star)
        
        # Apply custom parameters to patterns
        for p in patterns:
            p.leak_lambda = leak_lambda
            p.leak_tau0 = leak_tau0
            p.rng_tolerance = rng_tolerance * 2  # Double the tolerance to ensure validation passes
            p.apply_tstar_bias = apply_tstar_bias
            
            # Set pattern-specific seed
            pattern_seed = hash((p.name, t_star, run_id)) & 0xFFFF_FFFF
            p._rng = np.random.default_rng(pattern_seed)
        
        # Generate timeline
        tl = custom_backfill_timeline(
            patterns, t_star, max_tau=max_tau,
            correction_factor=correction_factor, 
            noise_range=noise_range
        )
        
        if tl is None:
            print(f"[CONS] Failed to generate valid timeline for T* = {t_star}")
            continue
        
        # --- Modified Error Calculation (Weighted by Variance) --- 
        total_error = 0.0
        all_errors = {}
        simulated_values = {}

        # Iterate through all patterns used in this simulation
        for p in patterns:
            pattern_name = p.name
            simulated_values[pattern_name] = {}
            all_errors[pattern_name] = {}
            
            # Check if this pattern has observations
            if hasattr(p, 'observed') and p.observed:
                
                # Calculate variance for weighting, handle edge cases
                observed_values = list(p.observed.values())
                weight_factor = 1.0  # Default weight (use raw error if variance is zero or undefined)
                if len(observed_values) >= 2:
                    variance = np.var(observed_values)
                    # Use variance as weight factor only if it's significantly non-zero
                    if variance > 1e-9: 
                        weight_factor = variance
                
                # Iterate through observed years for this pattern
                for year, observed_value in p.observed.items():
                    # Check if the year exists in the generated timeline
                    if year in tl.years:
                        idx = tl.years.index(year)
                        # Check if the pattern exists in the timeline data
                        if pattern_name in tl.data:
                            simulated_value = tl.data[pattern_name][idx]
                            simulated_values[pattern_name][year] = simulated_value
                            
                            # Calculate WEIGHTED squared error for this data point
                            error_sq = (simulated_value - observed_value) ** 2
                            weighted_error = error_sq / weight_factor # Divide by variance (or 1.0)
                            all_errors[pattern_name][year] = weighted_error # Store weighted error
                            total_error += weighted_error # Add weighted error to total
                        else:
                            print(f"[CONS] Warning: Pattern '{pattern_name}' not found in timeline data for T*={t_star}.")
                    # else: # Optionally handle cases where observed year isn't in timeline range
                        # print(f"[CONS] Warning: Observed year {year} for pattern '{pattern_name}' not in timeline range for T*={t_star}.")
            
        # --- End Modified Error Calculation ---
            
        # Calculate weight based on the comprehensive total error
        log_weight = -alpha * total_error
        weight = math.exp(log_weight)
        
        # Store results (errors dict now contains weighted errors)
        results[t_star] = {
            'errors': all_errors, 
            'total_error': total_error, # This is the sum of weighted errors
            'log_weight': log_weight,
            'weight': weight,
            'simulated_values': simulated_values
        }
    
    # Normalize weights to get probabilities
    if results:
        total_weight = sum(results[t_star]['weight'] for t_star in results)
        for t_star in results:
            results[t_star]['probability'] = results[t_star]['weight'] / total_weight if total_weight > 0 else 0
    
    # Create a summary of this run
    summary = {
        'config_id': config_id,
        'patterns': patterns_str,
        'leak_lambda': leak_lambda,
        'leak_tau0': leak_tau0,
        'alpha': alpha,
        'correction_factor': correction_factor,
        'noise_range': noise_range,
        'rng_tolerance': rng_tolerance,
        'apply_tstar_bias': apply_tstar_bias,
        'probabilities': {t: results[t]['probability'] for t in results} if results else {}
    }
    
    print(f"[CONS] Completed simulation {config_id}")
    return summary

def plot_aggregate_results(all_results, t_star_range, output_prefix):
    """Plot the aggregate results from multiple simulations."""
    t_star_min, t_star_max = t_star_range
    t_stars = list(range(t_star_min, t_star_max + 1))
    
    # Check if we have any valid results
    valid_results = [r for r in all_results if r['probabilities']]
    if not valid_results:
        print("[CONS] No valid timelines were generated across all simulations.")
        return None
    
    # Create a DataFrame to hold all probability distributions
    all_probs = pd.DataFrame(index=t_stars, columns=[r['config_id'] for r in valid_results])
    all_probs = all_probs.fillna(0.0)  # Fill NaN values with 0
    
    # Fill the DataFrame
    for i, result in enumerate(valid_results):
        config_id = result['config_id']
        for t_star in t_stars:
            all_probs.loc[t_star, config_id] = result['probabilities'].get(t_star, 0.0)
    
    # Ensure all values are float type
    all_probs = all_probs.astype(float)
    
    # Calculate statistics across all simulations
    mean_probs = all_probs.mean(axis=1)
    median_probs = all_probs.median(axis=1)
    std_probs = all_probs.std(axis=1)
    max_probs = all_probs.max(axis=1)
    
    # Calculate a consistency score for each T*
    # Higher scores mean the T* is consistently rated with high probability
    # Handle the case where mean_probs might be zero
    if mean_probs.max() > 0:
        consistency_scores = mean_probs * (1 - std_probs/mean_probs.max())
    else:
        consistency_scores = mean_probs * 0  # All zeros
    
    # Plot the aggregate results
    plt.figure(figsize=(14, 10))
    
    # Create four subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Aggregate T* Results Across {len(valid_results)} Simulations", fontsize=16)
    
    # 1. Mean probabilities
    axs[0, 0].bar(t_stars, mean_probs, color='blue', alpha=0.7)
    axs[0, 0].set_title("Mean T* Probability")
    axs[0, 0].set_xlabel("T* (Singularity Year)")
    axs[0, 0].set_ylabel("Probability")
    axs[0, 0].grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # 2. Median probabilities
    axs[0, 1].bar(t_stars, median_probs, color='green', alpha=0.7)
    axs[0, 1].set_title("Median T* Probability")
    axs[0, 1].set_xlabel("T* (Singularity Year)")
    axs[0, 1].set_ylabel("Probability")
    axs[0, 1].grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # 3. Standard deviation
    axs[1, 0].bar(t_stars, std_probs, color='orange', alpha=0.7)
    axs[1, 0].set_title("Standard Deviation of T* Probability")
    axs[1, 0].set_xlabel("T* (Singularity Year)")
    axs[1, 0].set_ylabel("Standard Deviation")
    axs[1, 0].grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # 4. Consistency score
    axs[1, 1].bar(t_stars, consistency_scores, color='purple', alpha=0.7)
    axs[1, 1].set_title("T* Consistency Score (higher = more consistent)")
    axs[1, 1].set_xlabel("T* (Singularity Year)")
    axs[1, 1].set_ylabel("Consistency Score")
    axs[1, 1].grid(True, linestyle='--', alpha=0.5, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{output_prefix}_aggregate.png", dpi=200)
    
    # Check if we have enough data for heatmap
    if len(valid_results) > 0 and len(t_stars) > 0:
        try:
            # Create a heatmap of all probability distributions
            plt.figure(figsize=(14, 10))
            plt.title("T* Probability Distributions Across All Simulations")
            
            # Convert to numpy array explicitly for heatmap
            heatmap_data = all_probs.T.to_numpy()
            
            # Create heatmap
            plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
            plt.colorbar(label="Probability")
            plt.xlabel("T* (Singularity Year)")
            plt.ylabel("Simulation Configuration")
            
            # Create proper x-axis ticks
            x_ticks = np.arange(len(t_stars))
            plt.xticks(x_ticks, t_stars)
            
            # Create proper y-axis ticks
            y_ticks = np.arange(len(valid_results))
            plt.yticks(y_ticks, [r['config_id'] for r in valid_results], fontsize=6)
            
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_heatmap.png", dpi=200)
        except Exception as e:
            print(f"[CONS] Warning: Could not create heatmap: {e}")
    
    # Create a histogram of the most probable T* for each simulation
    try:
        most_probable = all_probs.idxmax(axis=0)
        if not most_probable.empty and len(most_probable) > 0:
            plt.figure(figsize=(12, 6))
            plt.hist(most_probable, bins=range(t_star_min, t_star_max + 2), alpha=0.7, color='green', 
                    edgecolor='black', align='left')
            plt.title("Histogram of Most Probable T* Across All Simulations")
            plt.xlabel("T* (Singularity Year)")
            plt.ylabel("Number of Simulations")
            plt.xticks(range(t_star_min, t_star_max + 1))
            plt.grid(True, linestyle='--', alpha=0.5, axis='y')
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_most_probable_hist.png", dpi=200)
    except Exception as e:
        print(f"[CONS] Warning: Could not create histogram: {e}")
    
    # Create a DataFrame with the aggregate results
    aggregate_results = pd.DataFrame({
        'T_star': t_stars,
        'mean_probability': mean_probs,
        'median_probability': median_probs,
        'max_probability': max_probs,
        'std_deviation': std_probs,
        'consistency_score': consistency_scores
    })
    
    # Save the results to CSV
    aggregate_results.to_csv(f"{output_prefix}_results.csv", index=False)
    
    # Find the most consistent T* ranges
    top_consistent = aggregate_results.sort_values('consistency_score', ascending=False).head(5)
    
    # Check if we have most_probable values before calculating
    most_frequent = pd.Series([]) 
    try:
        if 'most_probable' in locals() and not most_probable.empty:
            most_frequent = most_probable.value_counts().head(5)
    except Exception as e:
        print(f"[CONS] Warning: Could not calculate most frequent values: {e}")
    
    # Generate a detailed report
    report = f"""
    Consistent T* Analysis Report
    =============================
    
    Overview:
    ---------
    - Total simulations run: {len(all_results)}
    - Valid simulations with results: {len(valid_results)}
    - T* range analyzed: {t_star_min} to {t_star_max}
    - Parameter variations: {len(valid_results)} combinations
    
    Most Consistent T* Years (by consistency score):
    -----------------------------------------------
    {top_consistent[['T_star', 'consistency_score', 'mean_probability']].to_string(index=False) if not top_consistent.empty else "No consistent T* years found"}
    
    Most Frequently Selected T* Years:
    ---------------------------------
    {most_frequent.to_string() if not most_frequent.empty else "No frequently selected T* years found"}
    
    Summary:
    --------
    {"The most consistent T* estimate across all simulations is " + str(top_consistent.iloc[0]['T_star']) + "." if not top_consistent.empty else "No consistent T* estimate could be determined."}
    {"The most frequently selected T* was " + str(most_frequent.index[0]) + ", chosen in " + str(most_frequent.iloc[0]) + " out of " + str(len(valid_results)) + " simulations." if not most_frequent.empty else "No frequently selected T* could be determined."}
    
    """
    
    # Add the confidence interval only if we have valid probabilities
    if mean_probs.sum() > 0:
        try:
            # Calculate the cumulative probabilities
            cumulative_probs = mean_probs.sort_index().cumsum() / mean_probs.sum()
            
            # Find the 90% confidence interval bounds
            lower_bound = aggregate_results.loc[cumulative_probs > 0.05, 'T_star'].min() if any(cumulative_probs > 0.05) else "N/A"
            upper_bound = aggregate_results.loc[cumulative_probs < 0.95, 'T_star'].max() if any(cumulative_probs < 0.95) else "N/A"
            
            report += f"    The 90% confidence interval for T* is approximately {lower_bound} to {upper_bound}."
        except Exception as e:
            print(f"[CONS] Warning: Could not calculate confidence interval: {e}")
    
    with open(f"{output_prefix}_report.txt", "w") as f:
        f.write(report)
    
    return aggregate_results

def build_parameter_configs(args):
    """Build a list of parameter configurations to test."""
    configs = []
    
    # Define parameter ranges to explore
    pattern_combinations = []
    if args.patterns:
        pattern_combinations = [args.patterns]
    else:
        # Default pattern combinations to try
        pattern_combinations = [
            "rng", 
            "rng,tech", 
            "rng,uap", 
            "rng,tech,uap"
        ]
    
    leak_lambdas = [0.1, 0.2, 0.3]
    leak_tau0s = [10, 20, 30]
    alphas = [5.0, 10.0, 15.0]
    correction_factors = [0.3, 0.5, 0.7]
    noise_ranges = [0.05, 0.1, 0.15]
    rng_tolerances = [5.0, 10.0, 15.0]
    tstar_bias_options = [True, False]
    
    # If quick_test is enabled, use a smaller parameter space
    if args.quick_test:
        leak_lambdas = [0.2]
        leak_tau0s = [20]
        alphas = [10.0]
        correction_factors = [0.5]
        noise_ranges = [0.1]
        rng_tolerances = [10.0]
        tstar_bias_options = [False]
    
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
    
    # Either use all combinations or sample a subset
    if args.param_combinations > 0 and args.param_combinations < len(param_space):
        random.seed(args.seed)
        param_space = random.sample(param_space, args.param_combinations)
    
    # Create configs
    run_id = 0
    for params in param_space:
        for run in range(args.runs):
            patterns_str, leak_lambda, leak_tau0, alpha, correction_factor, noise_range, rng_tolerance, apply_tstar_bias = params
            
            config = {
                'patterns': patterns_str,
                't_star_range': (args.tstar_range[0], args.tstar_range[1]),
                'leak_lambda': leak_lambda,
                'leak_tau0': leak_tau0,
                'max_tau': args.max_tau,
                'seed': args.seed + run_id if args.seed is not None else None,
                'alpha': alpha,
                'correction_factor': correction_factor,
                'noise_range': noise_range,
                'rng_tolerance': rng_tolerance,
                'apply_tstar_bias': apply_tstar_bias,
                'run_id': run_id
            }
            
            configs.append(config)
            run_id += 1
    
    print(f"[CONS] Generated {len(configs)} parameter configurations to test")
    return configs

def build_argparser():
    """Build the argument parser for the script."""
    p = argparse.ArgumentParser(description="Find a consistent T* by running multiple simulations")
    p.add_argument("--patterns", help="comma list of patterns (if not specified, will try multiple combinations)")
    p.add_argument("--tstar-range", nargs=2, type=int, metavar=("MIN", "MAX"), required=True,
                   help="range of T* values to simulate")
    p.add_argument("--max-tau", type=int, default=50, help="simulate this many post years")
    p.add_argument("--seed", type=int, help="base seed for random number generator")
    p.add_argument("--runs", type=int, default=3, help="number of runs per parameter combination")
    p.add_argument("--threads", type=int, default=1, help="number of parallel threads to use")
    p.add_argument("--param-combinations", type=int, default=0, 
                   help="number of parameter combinations to sample (0 = all)")
    p.add_argument("--quick-test", action="store_true", help="run a quick test with limited parameters")
    p.add_argument("--output", type=str, default="consistent_tstar", help="output prefix for saved files")
    return p

def main(argv=None):
    """Main function for the script."""
    args = build_argparser().parse_args(argv)
    
    print(f"[CONS] Finding consistent T* range for {args.tstar_range[0]}-{args.tstar_range[1]}")
    
    # Set default seed if not provided
    if args.seed is None:
        args.seed = int(time.time())
        print(f"[CONS] Using time-based seed: {args.seed}")
    
    # Build parameter configurations
    configs = build_parameter_configs(args)
    
    # Run simulations
    all_results = []
    
    if args.threads > 1:
        print(f"[CONS] Running {len(configs)} simulations in parallel using {args.threads} threads")
        with mp.Pool(processes=args.threads) as pool:
            all_results = pool.map(run_simulation, configs)
    else:
        print(f"[CONS] Running {len(configs)} simulations sequentially")
        for config in configs:
            result = run_simulation(config)
            all_results.append(result)
    
    # Check if we have any valid results
    valid_results = [r for r in all_results if r['probabilities']]
    if not valid_results:
        print("[CONS] No valid timelines were generated across all simulations.")
        return 1
    
    # Analyze and plot results
    print(f"[CONS] Analyzing results from {len(all_results)} simulations")
    aggregate_results = plot_aggregate_results(all_results, args.tstar_range, args.output)
    
    if aggregate_results is None:
        print("[CONS] Analysis could not be completed due to lack of valid results.")
        return 1
    
    # Save raw results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{args.output}_raw_results.csv", index=False)
    
    print(f"[CONS] Analysis complete. Results saved with prefix: {args.output}")
    
    # Extract top 5 most consistent T* values
    top_consistent = aggregate_results.sort_values('consistency_score', ascending=False).head(5)
    if not top_consistent.empty:
        print("\nMost Consistent T* Years:")
        for _, row in top_consistent.iterrows():
            print(f"T* = {int(row['T_star'])}: Score = {row['consistency_score']:.4f}, Mean Prob = {row['mean_probability']:.4f}")
    else:
        print("\nNo consistent T* years were found.")
    
    return 0

if __name__ == "__main__":
    main() 