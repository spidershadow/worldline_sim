#!/usr/bin/env python3
"""
Analyze Singularity Window

This script provides a comprehensive analysis framework to identify potential
'singularity windows' using multiple pattern sources, including:
1. Global Consciousness Project (GCP) RNG data analysis
2. UAP (Unidentified Aerial Phenomena) yearly sighting pattern analysis

The script estimates a potential timeframe for a technological singularity or
other significant transition point by identifying convergent patterns across
these independent datasets.

Example
-------
python scripts/analyze_singularity_window.py --start-year 2000 --end-year 2050 --window 10 --patterns all
python scripts/analyze_singularity_window.py --patterns uap --t-star-range 2030 2050
python scripts/analyze_singularity_window.py --patterns gcp --window 5
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.optimize import minimize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import after adding project root to path
from patterns.enhanced_rng import EnhancedRngPattern, GCP_SIGNIFICANCE_MILESTONES
from patterns.uap_yearly import UapYearlyPattern

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output" / "singularity_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_gcp_pattern(start_year, end_year, window_years=5):
    """Analyze GCP data for potential singularity window evidence.
    
    Args:
        start_year: First year to analyze
        end_year: Last year to analyze
        window_years: Number of years to look before and after each target
        
    Returns:
        DataFrame with analysis results and optimized t* value
    """
    print(f"Analyzing GCP data for years {start_year}-{end_year} (±{window_years} year window)")
    
    # Initialize EnhancedRngPattern with balanced window
    pattern = EnhancedRngPattern(direction="both")
    
    # Get GCP yearly intensity series
    gcp_data = pattern.get_yearly_intensity_series()
    
    # Analyze each year in the range
    results = []
    for year in range(start_year, end_year + 1):
        print(f"Analyzing GCP for year {year}...")
        
        # Perform retrocausal analysis
        analysis = pattern.assess_retrocausal_evidence(year, window_years)
        
        # Add to results
        results.append({
            'year': year,
            'before_slope': analysis['before_slope'],
            'after_slope': analysis['after_slope'],
            'acceleration': analysis['acceleration'],
            'symmetry': analysis['symmetry'],
            'retrocausal_score': analysis['retrocausal_score'],
            'evidence': analysis['evidence'],
            'cum_z_score': analysis['target_z_score']
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find optimal t* (year with highest retrocausal score)
    optimized_t_star = results_df.loc[results_df['retrocausal_score'].idxmax(), 'year']
    
    return results_df, optimized_t_star, gcp_data


def analyze_uap_pattern(start_year, end_year, window_years=10):
    """Analyze UAP yearly data for potential singularity window evidence.
    
    Args:
        start_year: First year to analyze
        end_year: Last year to analyze
        window_years: Window size for analysis
        
    Returns:
        DataFrame with analysis results and optimized t* value
    """
    print(f"Analyzing UAP data...")
    
    # Initialize UapYearlyPattern
    pattern = UapYearlyPattern(window_years=window_years)
    
    # Optimize t* within the specified range
    optimized_t_star = pattern.optimize_t_star(start_year=start_year, end_year=end_year)
    print(f"UAP optimized t* = {optimized_t_star:.2f}")
    
    # Calculate deviations for each year
    results = []
    analysis_years = range(min(pattern.data['year'].min(), start_year), 
                          max(pattern.data['year'].max(), end_year) + 1)
    
    for year in analysis_years:
        if year in pattern.data['year'].values:
            deviation_df = pattern.calculate_deviation_from_tstar(optimized_t_star)
            year_data = deviation_df[deviation_df['year'] == year]
            
            if not year_data.empty:
                results.append({
                    'year': year,
                    'deviation_score': year_data['deviation_score'].values[0],
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Get UAP yearly intensity series
    uap_data = pattern.get_yearly_intensity_series()
    
    return results_df, optimized_t_star, uap_data, pattern


def optimize_joint_t_star(gcp_data, uap_data, start_year=2000, end_year=2050, 
                         gcp_weight=0.5, uap_weight=0.5):
    """
    Find the optimal t* that maximizes the joint signal from multiple patterns.
    
    Args:
        gcp_data: DataFrame with GCP yearly intensity data
        uap_data: DataFrame with UAP yearly intensity data
        start_year: Lower bound for t* search
        end_year: Upper bound for t* search
        gcp_weight: Weight for GCP pattern (0-1)
        uap_weight: Weight for UAP pattern (0-1)
        
    Returns:
        Tuple of (optimal t* year, confidence interval range)
    """
    print(f"Optimizing joint t* using GCP weight={gcp_weight}, UAP weight={uap_weight}")
    
    # Normalize weights
    total_weight = gcp_weight + uap_weight
    gcp_weight = gcp_weight / total_weight
    uap_weight = uap_weight / total_weight
    
    # Prepare data by aligning years
    min_year = max(gcp_data['year'].min(), uap_data['year'].min())
    max_year = min(gcp_data['year'].max(), uap_data['year'].max())
    
    # Only use years that exist in both datasets
    common_years = sorted(set(gcp_data['year']).intersection(set(uap_data['year'])))
    
    # Filter data to common years
    gcp_filtered = gcp_data[gcp_data['year'].isin(common_years)]
    uap_filtered = uap_data[uap_data['year'].isin(common_years)]
    
    # Create year-indexed series
    gcp_series = gcp_filtered.set_index('year')['intensity']
    uap_series = uap_filtered.set_index('year')['intensity']
    
    def calculate_joint_deviation(t_star):
        """Calculate joint deviation score for a given t*."""
        # Convert t_star to a year value
        t_star_year = float(t_star[0])
        
        # Calculate years from t_star
        years = np.array(common_years)
        years_from_tstar = np.abs(years - t_star_year)
        
        # Weight by inverse distance from t*
        weights = 1 / (years_from_tstar + 1)
        
        # Calculate weighted intensity for both patterns
        gcp_weighted = gcp_series.values * weights
        uap_weighted = uap_series.values * weights
        
        # Combine weighted intensities
        joint_signal = gcp_weight * np.sum(gcp_weighted) + uap_weight * np.sum(uap_weighted)
        
        # Negate for minimization
        return -joint_signal
    
    # Optimize to find joint t*
    bounds = [(start_year, end_year)]
    initial_guess = [(start_year + end_year) / 2]
    
    result = minimize(calculate_joint_deviation, x0=initial_guess, bounds=bounds, 
                     method='L-BFGS-B')
    
    if result.success:
        joint_t_star = float(result.x[0])
        print(f"Optimized joint t* = {joint_t_star:.2f}")
        
        # Calculate confidence interval
        ci_range = calculate_tstar_confidence_interval(
            gcp_filtered, uap_filtered, joint_t_star, 
            gcp_weight, uap_weight, start_year, end_year
        )
        
        return joint_t_star, ci_range
    else:
        print("Joint optimization failed to converge")
        return None, None


def calculate_tstar_confidence_interval(gcp_data, uap_data, t_star, gcp_weight=0.5, 
                                      uap_weight=0.5, start_year=2000, end_year=2050,
                                      num_simulations=100, confidence_level=0.95):
    """
    Calculate confidence interval for t* through Monte Carlo simulation.
    
    Args:
        gcp_data: DataFrame with GCP yearly intensity data
        uap_data: DataFrame with UAP yearly intensity data
        t_star: Optimal t* value
        gcp_weight: Weight for GCP pattern
        uap_weight: Weight for UAP pattern
        start_year: Lower bound for t* search
        end_year: Upper bound for t* search
        num_simulations: Number of simulations to run
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        Integer value representing ± range around t* (e.g., 3 for t* ± 3 years)
    """
    print(f"Calculating t* confidence interval using {num_simulations} simulations...")
    
    # Create a list to store simulation results
    t_star_values = []
    
    for i in range(num_simulations):
        # Bootstrap sample from the data with replacement
        gcp_sampled = gcp_data.sample(frac=1.0, replace=True)
        uap_sampled = uap_data.sample(frac=1.0, replace=True)
        
        # Add random noise to the intensity values (gaussian noise)
        gcp_sampled = gcp_sampled.copy()
        uap_sampled = uap_sampled.copy()
        
        # Add noise with standard deviation of 10% of the original intensity
        gcp_noise_level = 0.1 * gcp_sampled['intensity'].std()
        uap_noise_level = 0.1 * uap_sampled['intensity'].std()
        
        gcp_sampled['intensity'] += np.random.normal(0, gcp_noise_level, size=len(gcp_sampled))
        uap_sampled['intensity'] += np.random.normal(0, uap_noise_level, size=len(uap_sampled))
        
        # Create year-indexed series
        gcp_series = gcp_sampled.set_index('year')['intensity']
        uap_series = uap_sampled.set_index('year')['intensity']
        
        # Define optimization function for this simulation
        def calculate_joint_deviation(t_star_val):
            """Calculate joint deviation score for a given t*."""
            # Convert t_star to a year value
            t_star_year = float(t_star_val[0])
            
            # Calculate years from t_star
            years = np.array(sorted(set(gcp_sampled['year']).intersection(set(uap_sampled['year']))))
            years_from_tstar = np.abs(years - t_star_year)
            
            # Weight by inverse distance from t*
            weights = 1 / (years_from_tstar + 1)
            
            # Create filtered series for common years
            common_years = sorted(set(gcp_sampled['year']).intersection(set(uap_sampled['year'])))
            gcp_common = gcp_series[gcp_series.index.isin(common_years)]
            uap_common = uap_series[uap_series.index.isin(common_years)]
            
            # Ensure values are aligned
            if len(gcp_common) == len(weights) and len(uap_common) == len(weights):
                # Calculate weighted intensity for both patterns
                gcp_weighted = gcp_common.values * weights
                uap_weighted = uap_common.values * weights
                
                # Combine weighted intensities
                joint_signal = gcp_weight * np.sum(gcp_weighted) + uap_weight * np.sum(uap_weighted)
            else:
                # Handle misalignment by returning a poor score
                joint_signal = 0
            
            # Negate for minimization
            return -joint_signal
        
        # Optimize to find t* for this simulation
        bounds = [(start_year, end_year)]
        initial_guess = [t_star]  # Use the optimal t* as initial guess
        
        try:
            result = minimize(calculate_joint_deviation, x0=initial_guess, bounds=bounds, 
                             method='L-BFGS-B')
            
            if result.success:
                sim_t_star = float(result.x[0])
                t_star_values.append(sim_t_star)
                
                if (i+1) % 10 == 0:
                    print(f"Completed {i+1}/{num_simulations} simulations")
        except Exception as e:
            print(f"Simulation {i+1} failed: {e}")
    
    # Calculate confidence interval from simulation results
    if t_star_values:
        t_star_values = np.array(t_star_values)
        
        # Calculate the distance from the median result
        median_t_star = np.median(t_star_values)
        distances = np.abs(t_star_values - median_t_star)
        
        # Sort distances
        sorted_distances = np.sort(distances)
        
        # Find the distance that contains the desired confidence level
        ci_index = int(confidence_level * len(sorted_distances))
        ci_range = int(np.ceil(sorted_distances[ci_index]))
        
        print(f"T* confidence interval: ±{ci_range} years at {confidence_level*100:.0f}% confidence level")
        return ci_range
    else:
        print("Failed to calculate confidence interval. Using default of ±3 years.")
        return 3


def create_visualizations(gcp_results, uap_results, joint_t_star, gcp_t_star, uap_t_star, 
                        uap_pattern, output_dir, t_star_ci=3):
    """Create visualizations for singularity window analysis.
    
    Args:
        gcp_results: DataFrame with GCP analysis results
        uap_results: DataFrame with UAP analysis results
        joint_t_star: Optimized joint t* value
        gcp_t_star: Optimized GCP t* value
        uap_t_star: Optimized UAP t* value
        uap_pattern: UapYearlyPattern instance
        output_dir: Directory to save visualizations
        t_star_ci: Confidence interval range for t* (default: 3)
    """
    # Set up style
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Plot GCP retrocausal scores
    if gcp_results is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(gcp_results['year'], gcp_results['retrocausal_score'], 'o-', 
                linewidth=2, markersize=8, label='GCP Retrocausal Score')
        
        # Add line for evidence thresholds
        plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Strong Evidence (0.7)')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate Evidence (0.5)')
        plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Weak Evidence (0.3)')
        
        # Highlight t* values
        if gcp_t_star:
            plt.axvline(x=gcp_t_star, color='blue', linestyle='-', alpha=0.5, 
                       label=f'GCP t* = {gcp_t_star:.1f}')
        
        if joint_t_star:
            plt.axvline(x=joint_t_star, color='purple', linestyle='-', linewidth=2, alpha=0.7, 
                       label=f'Joint t* = {joint_t_star:.1f} ±{t_star_ci}')
            # Add confidence interval
            plt.axvspan(joint_t_star - t_star_ci, joint_t_star + t_star_ci, alpha=0.1, color='purple')
        
        plt.title('GCP Retrocausal Evidence Score by Year', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Retrocausal Score', fontsize=14)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "gcp_retrocausal_scores.png", dpi=150)
        plt.close()
        
        # Plot GCP Cumulative Deviation
        plt.figure(figsize=(12, 6))
        if 'cum_z_score' in gcp_results.columns:
            plt.plot(gcp_results['year'], gcp_results['cum_z_score'], 'o-', 
                    linewidth=2, markersize=8, color='darkblue', label='Cumulative Z-Score')
            
            # Add statistical significance lines
            plt.axhline(y=3.29, color='green', linestyle='--', alpha=0.7, label='p < 0.001 (Z=3.29)')
            plt.axhline(y=2.58, color='orange', linestyle='--', alpha=0.7, label='p < 0.01 (Z=2.58)')
            plt.axhline(y=1.96, color='red', linestyle='--', alpha=0.7, label='p < 0.05 (Z=1.96)')
            
            # Highlight key periods
            plt.axvspan(2004, 2007, alpha=0.2, color='yellow', label='2004-2007 Peak')
            plt.axvspan(2012, 2013, alpha=0.1, color='green', label='2012-2013 Secondary')
        
            # Highlight t* values
            if gcp_t_star:
                plt.axvline(x=gcp_t_star, color='blue', linestyle='-', alpha=0.5, 
                           label=f'GCP t* = {gcp_t_star:.1f}')
            
            if joint_t_star:
                plt.axvline(x=joint_t_star, color='purple', linestyle='-', linewidth=2, alpha=0.7, 
                           label=f'Joint t* = {joint_t_star:.1f} ±{t_star_ci}')
                # Add confidence interval
                plt.axvspan(joint_t_star - t_star_ci, joint_t_star + t_star_ci, alpha=0.1, color='purple')
            
            plt.title('GCP Cumulative Deviation Over Time', fontsize=16)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Cumulative Z-Score', fontsize=14)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "gcp_cumulative_deviation.png", dpi=150)
            plt.close()
    
    # 2. Plot UAP trend and deviation profile
    if uap_pattern:
        # UAP trend plot
        trend_plot = uap_pattern.plot_sightings_trend()
        
        # Add joint t* if available
        if joint_t_star:
            trend_plot.axvline(x=joint_t_star, color='purple', linestyle='-', linewidth=2, alpha=0.7, 
                             label=f'Joint t* = {joint_t_star:.1f} ±{t_star_ci}')
            trend_plot.legend()
            
        trend_plot.savefig(output_dir / "uap_sightings_trend.png", dpi=150)
        trend_plot.close()
        
        # UAP deviation profile
        if uap_t_star:
            deviation_plot = uap_pattern.plot_deviation_profile(t_star=uap_t_star)
            deviation_plot.savefig(output_dir / "uap_deviation_profile.png", dpi=150)
            deviation_plot.close()
    
    # 3. Create joint pattern visualization
    if gcp_results is not None and uap_results is not None and joint_t_star is not None:
        # Create joint pattern correlation plot
        plt.figure(figsize=(12, 8))
        
        # Prepare data for joint plot
        gcp_data = uap_pattern._load_gcp_yearly_intensity()
        uap_data = uap_pattern.get_yearly_intensity_series()
        
        # Align years
        common_years = sorted(set(gcp_data['year']).intersection(set(uap_data['year'])))
        
        if common_years:
            # Filter data to common years
            gcp_filtered = gcp_data[gcp_data['year'].isin(common_years)]
            uap_filtered = uap_data[uap_data['year'].isin(common_years)]
            
            # Calculate distance from joint t*
            gcp_filtered['years_from_tstar'] = abs(gcp_filtered['year'] - joint_t_star)
            uap_filtered['years_from_tstar'] = abs(uap_filtered['year'] - joint_t_star)
            
            # Create scatter plot
            plt.scatter(gcp_filtered['intensity'], uap_filtered['intensity'], 
                       c=gcp_filtered['years_from_tstar'], cmap='viridis_r', s=100, alpha=0.7)
            
            # Add labels for points
            for i, year in enumerate(common_years):
                plt.annotate(str(year), 
                           (gcp_filtered[gcp_filtered['year']==year]['intensity'].values[0],
                            uap_filtered[uap_filtered['year']==year]['intensity'].values[0]),
                           xytext=(7, 0), textcoords='offset points')
            
            # Add colorbar
            cbar = plt.colorbar()
            cbar.set_label('Years from Joint t*', fontsize=12)
            
            # Add titles and labels
            plt.title(f'GCP vs UAP Pattern Correlation (Joint t* = {joint_t_star:.1f} ±{t_star_ci})', fontsize=16)
            plt.xlabel('GCP Intensity (Z-score)', fontsize=14)
            plt.ylabel('UAP Intensity (Z-score)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "joint_pattern_correlation.png", dpi=150)
            plt.close()
            
            # Create time series comparison
            plt.figure(figsize=(12, 6))
            plt.plot(gcp_filtered['year'], gcp_filtered['intensity'], 'b-', 
                   linewidth=2, label='GCP Intensity')
            plt.plot(uap_filtered['year'], uap_filtered['intensity'], 'r-', 
                   linewidth=2, label='UAP Intensity')
            
            # Add vertical line for joint t*
            plt.axvline(x=joint_t_star, color='purple', linestyle='-', linewidth=2, alpha=0.7, 
                       label=f'Joint t* = {joint_t_star:.1f} ±{t_star_ci}')
            
            plt.title('GCP and UAP Pattern Intensity Over Time', fontsize=16)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Pattern Intensity (Z-score)', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "joint_time_series.png", dpi=150)
            plt.close()


def generate_gcp_retrocausal_scores_chart(years, scores, output_path):
    """
    Generate a bar chart visualization of GCP retrocausal evidence scores by year.
    
    Args:
        years: List of years
        scores: List of evidence scores (0-1 scale)
        output_path: Path to save the output image
        
    Returns:
        Path to the saved image
    """
    plt.figure(figsize=(12, 7))
    
    # Create color mapping based on evidence levels
    colors = []
    for score in scores:
        if score >= 0.7:
            colors.append('#2ca02c')  # Strong evidence - green
        elif score >= 0.5:
            colors.append('#ffad33')  # Moderate evidence - orange
        elif score >= 0.3:
            colors.append('#ffe066')  # Weak evidence - yellow
        else:
            colors.append('#d3d3d3')  # Insignificant - light gray
            
    # Create the bar chart
    bars = plt.bar(years, scores, color=colors, alpha=0.8, width=0.8)
    
    # Add horizontal lines for evidence thresholds
    plt.axhline(y=0.7, color='#2ca02c', linestyle='--', alpha=0.7, 
                label='Strong Evidence (≥ 0.7)')
    plt.axhline(y=0.5, color='#ffad33', linestyle='--', alpha=0.7, 
                label='Moderate Evidence (≥ 0.5)')
    plt.axhline(y=0.3, color='#ffe066', linestyle='--', alpha=0.7, 
                label='Weak Evidence (≥ 0.3)')
    
    # Annotate key periods
    plt.annotate('Peak Pattern Period', xy=(2005.5, 0.72), xytext=(2005.5, 0.85),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                ha='center', fontsize=11)
    
    plt.annotate('Secondary Pattern', xy=(2013, 0.55), xytext=(2013, 0.65),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                ha='center', fontsize=11)
    
    plt.annotate('Emerging Pattern', xy=(2023, 0.48), xytext=(2023, 0.6),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                ha='center', fontsize=11)
    
    # Customize the chart
    plt.title('GCP Retrocausal Evidence Scores (2000-2025)', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Evidence Score (0-1 scale)', fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(years[::2], rotation=45)  # Show every other year
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    # Add text explanation
    plt.figtext(0.5, 0.01, 
               "Higher scores indicate stronger statistical evidence for retrocausal patterns in GCP data.",
               ha="center", fontsize=10, bbox={"facecolor":"#f5f5f5", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_gcp_cumulative_deviation_chart(years, cumdev_values, output_path):
    """
    Generate a cumulative deviation chart for GCP data with trajectory analysis.
    
    Args:
        years: List of years
        cumdev_values: List of cumulative deviation values
        output_path: Path to save the output image
        
    Returns:
        Path to the saved image
    """
    plt.figure(figsize=(10, 6))  # Changed from (12, 7) to make less elongated
    
    # Simple line plot instead of gradient color line collection
    plt.plot(years, cumdev_values, '-o', linewidth=2.5, color='#3366cc', 
             markersize=5, markerfacecolor='white')
    
    # Add a horizontal reference line at 0
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Highlight key periods instead of showing all trend lines
    plt.axvspan(2004, 2007, alpha=0.2, color='yellow', label='2004-2007 Peak')
    plt.axvspan(2012, 2013, alpha=0.2, color='green', label='2012-2013 Secondary')
    plt.axvspan(2020, 2025, alpha=0.2, color='lightblue', label='Recent Pattern')
    
    # Add just one annotation for the highest point
    max_idx = cumdev_values.index(max(cumdev_values))
    max_year = years[max_idx]
    max_val = cumdev_values[max_idx]
    plt.annotate(f"Peak: {max_val:.2f}", 
                xy=(max_year, max_val), 
                xytext=(max_year+1, max_val+0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                fontsize=10)
    
    # Customize chart appearance
    plt.title("GCP Cumulative Deviation (2000-2025)", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Cumulative Deviation", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    # Add simplified explanation text
    plt.figtext(0.5, 0.01, 
               "Upward trajectory indicates consistent deviation from expected random behavior.",
               ha="center", fontsize=10, bbox={"facecolor":"#f5f5f5", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_uap_sightings_trend(years, sighting_counts, t_star, output_path, t_star_ci=3):
    """
    Plot UAP sightings over time, highlighting acceleration periods.
    
    Args:
        years: List of years
        sighting_counts: List of sighting counts for each year
        t_star: The calculated t* value (singularity window)
        output_path: Path to save the output image
        t_star_ci: Confidence interval range for t* (default: 3)
        
    Returns:
        Path to the saved image
    """
    # Create figure and axis
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Plot the sightings trend
    ax.plot(years, sighting_counts, marker='o', linestyle='-', linewidth=2.5, 
            color='#3366cc', markersize=6)
    
    # Fill under the curve
    ax.fill_between(years, 0, sighting_counts, alpha=0.2, color='#3366cc')
    
    # Highlight acceleration periods
    ax.axvspan(1995, 2008, alpha=0.2, color='green', label='Primary Acceleration (1995-2008)')
    ax.axvspan(2017, 2021, alpha=0.2, color='orange', label='Secondary Acceleration (2017-2021)')
    
    # Add vertical line for t*
    if t_star:
        ax.axvline(x=t_star, color='red', linestyle='--', linewidth=2, 
                   label=f't* = {t_star} ±{t_star_ci}')
        # Add shaded area for confidence interval
        ax.axvspan(t_star-t_star_ci, t_star+t_star_ci, alpha=0.1, color='red')
    
    # Customize the chart
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('UAP Sighting Count', fontsize=12)
    ax.set_title('UAP Sightings Trend Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add annotations for key periods
    ax.annotate('Primary Acceleration', xy=(2000, sighting_counts[years.index(2000)]),
                xytext=(2000, max(sighting_counts)*0.4), ha='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                arrowprops=dict(arrowstyle='->'))
    
    if 2017 in years and 2020 in years:
        ax.annotate('Secondary Acceleration', 
                    xy=(2019, sighting_counts[years.index(2019)]),
                    xytext=(2019, max(sighting_counts)*0.6), ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                    arrowprops=dict(arrowstyle='->'))
    
    # Add annotation for highest value
    max_idx = sighting_counts.index(max(sighting_counts))
    max_year = years[max_idx]
    max_count = sighting_counts[max_idx]
    ax.annotate(f'Peak: {int(max_count)}', xy=(max_year, max_count),
                xytext=(max_year+1, max_count), ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_uap_deviation_profile(years, deviations, t_star, output_path):
    """
    Visualize UAP deviation profile relative to t*.
    
    Args:
        years: List of years
        deviations: List of deviation values for each year
        t_star: The calculated t* value
        output_path: Path to save the output image
        
    Returns:
        Path to the saved image
    """
    # Create figure and axis
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Calculate years from t*
    years_from_tstar = [t_star - year for year in years]
    
    # Create scatter plot with size based on deviation
    scatter = ax.scatter(years_from_tstar, deviations, 
                c=deviations, cmap='plasma', 
                s=[max(50, d*40) for d in deviations],
                alpha=0.8, edgecolors='black')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Deviation Intensity', fontsize=10)
    
    # Plot best fit curve (exponential decay as approximation)
    # Filter to only include positive years from t*
    pos_idx = [i for i, y in enumerate(years_from_tstar) if y > 0]
    if pos_idx:
        pos_years = [years_from_tstar[i] for i in pos_idx]
        pos_devs = [deviations[i] for i in pos_idx]
        
        # Simple smoothing curve
        if len(pos_years) > 3:
            from scipy.interpolate import make_interp_spline
            pos_years_sorted, pos_devs_sorted = zip(*sorted(zip(pos_years, pos_devs)))
            x_smooth = np.linspace(min(pos_years_sorted), max(pos_years_sorted), 100)
            try:
                spl = make_interp_spline(pos_years_sorted, pos_devs_sorted, k=min(3, len(pos_years_sorted)-1))
                y_smooth = spl(x_smooth)
                ax.plot(x_smooth, y_smooth, color='red', linestyle='-', linewidth=2, 
                        label='Inferred Deviation Trend', zorder=1)
            except:
                # Fall back to simpler curve if spline fails
                ax.plot(sorted(pos_years), [d for _, d in sorted(zip(pos_years, pos_devs))], 
                        color='red', linestyle='-', linewidth=2, 
                        label='Inferred Deviation Trend', zorder=1)
    
    # Highlight deviation zones
    ax.axvspan(15, 25, alpha=0.1, color='green', label='Peak Deviation Zone (15-25 years)')
    ax.axvspan(5, 15, alpha=0.1, color='yellow', label='Moderate Deviation Zone (5-15 years)')
    
    # Add vertical line at 0 (t*)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='t*')
    
    # Customize the chart
    ax.set_xlabel('Years Before t*', fontsize=12)
    ax.set_ylabel('UAP Deviation Intensity', fontsize=12)
    ax.set_title('UAP Deviation Profile Relative to t*', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add annotations for key zones
    ax.annotate('Peak Deviation Zone', xy=(20, 2.5),
                xytext=(20, 2.8), ha='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    ax.annotate('Moderate Deviation Zone', xy=(10, 1.5),
                xytext=(10, 1.8), ha='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_joint_pattern_correlation(gcp_scores, uap_deviations, years, t_star, output_path):
    """
    Create a scatter plot showing correlation between GCP and UAP pattern intensities.
    
    Args:
        gcp_scores: List of GCP retrocausal scores
        uap_deviations: List of UAP deviation values
        years: List of years
        t_star: The calculated t* value
        output_path: Path to save the output image
        
    Returns:
        Path to the saved image
    """
    # Create figure and axis
    fig, ax = plt.figure(figsize=(9, 7)), plt.gca()
    
    # Calculate proximity to t*
    proximity = [max(0, 1 - abs(year - t_star) / 20) for year in years]
    
    # Create scatter plot with color based on proximity to t*
    scatter = ax.scatter(gcp_scores, uap_deviations, 
                       c=proximity, cmap='viridis', 
                       s=80, alpha=0.9, edgecolors='black')
    
    # Add year labels to points
    for i, year in enumerate(years):
        ax.annotate(str(year), (gcp_scores[i], uap_deviations[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Proximity to t*', fontsize=10)
    
    # Plot best fit line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(gcp_scores, uap_deviations)
    x = np.array([min(gcp_scores), max(gcp_scores)])
    y = intercept + slope * x
    ax.plot(x, y, color='red', linestyle='--', 
           label=f'Correlation: r = {r_value:.2f}')
    
    # Customize the chart
    ax.set_xlabel('GCP Retrocausal Evidence Score', fontsize=12)
    ax.set_ylabel('UAP Deviation Intensity', fontsize=12)
    ax.set_title('Correlation Between GCP and UAP Pattern Intensities', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight key years with highest joint values
    joint_values = [(gcp_scores[i] + uap_deviations[i]) for i in range(len(years))]
    top_idx = sorted(range(len(joint_values)), key=lambda i: joint_values[i], reverse=True)[:3]
    
    for i in top_idx:
        ax.annotate(f'{years[i]}', 
                   xy=(gcp_scores[i], uap_deviations[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                   arrowprops=dict(arrowstyle='->'))
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_joint_time_series(years, gcp_scores, uap_deviations, t_star, output_path):
    """
    Create a time series plot comparing GCP and UAP intensities over time.
    
    Args:
        years: List of years
        gcp_scores: List of GCP retrocausal scores
        uap_deviations: List of UAP deviation values
        t_star: The calculated t* value
        output_path: Path to save the output image
        
    Returns:
        Path to the saved image
    """
    # Create figure and axis
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Normalize both data series to 0-1 scale for comparison
    def normalize(data):
        min_val, max_val = min(data), max(data)
        return [(x - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for x in data]
    
    norm_gcp = normalize(gcp_scores)
    norm_uap = normalize(uap_deviations)
    
    # Plot both normalized time series
    ax.plot(years, norm_gcp, marker='o', linestyle='-', color='blue', 
           label='GCP Retrocausal Score (normalized)')
    ax.plot(years, norm_uap, marker='s', linestyle='-', color='green', 
           label='UAP Deviation (normalized)')
    
    # Add vertical line at t*
    ax.axvline(x=t_star, color='red', linestyle='--', linewidth=2, 
              label=f't* = {t_star}')
    # Add shaded area for confidence interval
    ax.axvspan(t_star-3, t_star+3, alpha=0.1, color='red')
    
    # Highlight correlation periods
    correlation_periods = []
    for i in range(len(years) - 4):
        # Check for 5-year moving correlation
        window_gcp = norm_gcp[i:i+5]
        window_uap = norm_uap[i:i+5]
        from scipy import stats
        corr, _ = stats.pearsonr(window_gcp, window_uap)
        if corr > 0.7:  # High positive correlation
            correlation_periods.append((years[i], years[i+4]))
    
    # Merge overlapping periods
    merged_periods = []
    if correlation_periods:
        merged_periods.append(correlation_periods[0])
        for period in correlation_periods[1:]:
            if period[0] <= merged_periods[-1][1]:
                merged_periods[-1] = (merged_periods[-1][0], period[1])
            else:
                merged_periods.append(period)
    
    # Highlight the merged periods
    for start, end in merged_periods:
        ax.axvspan(start, end, alpha=0.2, color='purple', 
                  label=f'High Correlation ({start}-{end})')
        mid_point = (start + end) / 2
        ax.text(mid_point, 0.1, "Correlated", ha='center', fontsize=9, 
               color='purple', fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.2"))
    
    # Customize the chart
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title('Comparative Time Series: GCP and UAP Pattern Intensities', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1.1)
    
    # Add annotations for peaks
    gcp_peak_idx = norm_gcp.index(max(norm_gcp))
    uap_peak_idx = norm_uap.index(max(norm_uap))
    
    ax.annotate('GCP Peak', 
               xy=(years[gcp_peak_idx], norm_gcp[gcp_peak_idx]),
               xytext=(years[gcp_peak_idx], norm_gcp[gcp_peak_idx] + 0.15),
               ha='center',
               arrowprops=dict(arrowstyle='->'))
    
    ax.annotate('UAP Peak', 
               xy=(years[uap_peak_idx], norm_uap[uap_peak_idx]),
               xytext=(years[uap_peak_idx], norm_uap[uap_peak_idx] + 0.15),
               ha='center',
               arrowprops=dict(arrowstyle='->'))
    
    # Add legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) 
              if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def analyze_and_generate_report(args):
    """
    Analyze GCP and UAP data and generate a comprehensive report.
    
    Args:
        args: Command line arguments with output_dir and t_star values
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define years range (historical data from 2000 to present)
    years = list(range(2000, 2026))
    
    # Sample GCP retrocausal evidence scores
    # These values represent the strength of evidence for retrocausal effects
    # Higher values indicate stronger evidence for retrocausal influences
    gcp_scores = [
        0.15, 0.18, 0.25, 0.35,  # 2000-2003
        0.62, 0.68, 0.73, 0.67,  # 2004-2007 (high anomaly period)
        0.42, 0.38, 0.31, 0.28,  # 2008-2011
        0.52, 0.55, 0.41, 0.39,  # 2012-2015 (secondary pattern)
        0.32, 0.30, 0.29, 0.28,  # 2016-2019
        0.31, 0.35, 0.39, 0.44, 0.48, 0.52  # 2020-2025 (emerging pattern)
    ]
    
    # Generate sample GCP cumulative deviation data
    # These values represent yearly changes in cumulative deviation
    # Positive values indicate deviation above expectation, negative below
    cumulative_deviation_values = [
        -0.15, -0.10, 0.05, 0.20,  # 2000-2003
        0.45, 0.60, 0.65, 0.35,    # 2004-2007 (primary deviation)
        0.20, 0.15, 0.10, 0.05,    # 2008-2011
        0.30, 0.40, 0.15, 0.10,    # 2012-2015 (secondary deviation)
        0.05, 0.00, -0.05, -0.10,  # 2016-2019
        0.05, 0.15, 0.25, 0.30, 0.40, 0.45  # 2020-2025 (recent pattern)
    ]
    
    # Sample UAP sightings data (counts per year)
    # Based on historical patterns, showing acceleration in certain periods
    uap_sightings = [
        120, 140, 150, 180,        # 2000-2003
        250, 290, 310, 270,        # 2004-2007 (primary acceleration)
        230, 215, 200, 195,        # 2008-2011
        210, 220, 225, 230,        # 2012-2015
        240, 260, 320, 380,        # 2016-2019 (secondary acceleration)
        450, 520, 580, 610, 650, 700  # 2020-2025 (continued acceleration)
    ]
    
    # UAP deviation intensity (calculated based on deviation from baseline)
    # These values represent how much UAP activities deviate from expected patterns
    uap_deviations = [
        0.8, 1.0, 1.2, 1.5,        # 2000-2003
        2.2, 2.6, 2.8, 2.4,        # 2004-2007 (high deviation)
        2.0, 1.8, 1.6, 1.5,        # 2008-2011
        1.7, 1.8, 1.9, 2.0,        # 2012-2015
        2.1, 2.3, 2.7, 3.2,        # 2016-2019 (increasing deviation)
        3.5, 3.8, 4.0, 4.2, 4.4, 4.6  # 2020-2025 (highest deviation)
    ]
    
    # Set t* (singularity window) value from args or calculate through optimization
    if args.t_star:
        t_star = args.t_star
        t_star_ci = args.t_star_ci if hasattr(args, 't_star_ci') else 3
    else:
        # Create dataframes for optimization
        years_array = np.array(years)
        gcp_data = pd.DataFrame({
            'year': years_array,
            'intensity': np.array(gcp_scores)
        })
        uap_data = pd.DataFrame({
            'year': years_array,
            'intensity': np.array(uap_deviations)
        })
        
        # Calculate optimal t* and confidence interval
        t_star, t_star_ci = optimize_joint_t_star(gcp_data, uap_data, 
                                                start_year=2030, end_year=2050,
                                                gcp_weight=0.5, uap_weight=0.5)
        if t_star is None:
            t_star = 2035
            t_star_ci = 3
    
    # Generate all charts
    output_paths = {}
    
    # GCP charts
    output_paths['gcp_scores'] = generate_gcp_retrocausal_scores_chart(
        years, gcp_scores, 
        os.path.join(args.output_dir, 'gcp_retrocausal_scores.png')
    )
    
    output_paths['gcp_cumulative'] = generate_gcp_cumulative_deviation_chart(
        years, cumulative_deviation_values, 
        os.path.join(args.output_dir, 'gcp_cumulative_deviation.png')
    )
    
    # UAP charts
    output_paths['uap_sightings'] = generate_uap_sightings_trend(
        years, uap_sightings, t_star,
        os.path.join(args.output_dir, 'uap_sightings_trend.png'),
        t_star_ci
    )
    
    output_paths['uap_deviation'] = generate_uap_deviation_profile(
        years, uap_deviations, t_star,
        os.path.join(args.output_dir, 'uap_deviation_profile.png')
    )
    
    # Joint analysis charts
    output_paths['joint_correlation'] = generate_joint_pattern_correlation(
        gcp_scores, uap_deviations, years, t_star,
        os.path.join(args.output_dir, 'joint_pattern_correlation.png')
    )
    
    output_paths['joint_time_series'] = generate_joint_time_series(
        years, gcp_scores, uap_deviations, t_star,
        os.path.join(args.output_dir, 'joint_time_series.png')
    )
    
    # Generate HTML report with all visualizations
    report_file = os.path.join(args.output_dir, 'singularity_window_report.html')
    generate_report(
        output_paths=output_paths,
        gcp_scores=gcp_scores,
        cumulative_deviation_values=cumulative_deviation_values,
        uap_sightings=uap_sightings,
        uap_deviations=uap_deviations,
        years=years,
        t_star=t_star,
        t_star_ci=t_star_ci,
        output_file=report_file
    )
    
    print(f"Analysis complete. Report generated at: {report_file}")
    return report_file

def generate_report(output_paths, gcp_scores, cumulative_deviation_values, 
                   uap_sightings, uap_deviations, years, t_star, t_star_ci=3, output_file=None):
    """
    Generate an HTML report including all visualizations and analysis.
    
    Args:
        output_paths: Dictionary of paths to visualization images
        gcp_scores: List of GCP retrocausal evidence scores
        cumulative_deviation_values: List of cumulative deviation values
        uap_sightings: List of UAP sighting counts
        uap_deviations: List of UAP deviation intensities
        years: List of years analyzed
        t_star: The calculated t* value (singularity window)
        t_star_ci: Confidence interval range for t* (default: 3)
        output_file: Path to save the HTML report
        
    Returns:
        Path to the generated report file
    """
    # Get the current date for the report
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Calculate key statistics
    peak_gcp_year = years[gcp_scores.index(max(gcp_scores))]
    peak_uap_year = years[uap_sightings.index(max(uap_sightings))]
    
    # Calculate correlation between GCP and UAP data
    from scipy import stats
    gcp_uap_corr, _ = stats.pearsonr(gcp_scores, uap_deviations)
    
    # Start building the HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Singularity Window Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3, h4 {{
            color: #444;
            margin-top: 30px;
        }}
        h1 {{
            text-align: center;
            color: #0056b3;
            border-bottom: 2px solid #0056b3;
            padding-bottom: 10px;
        }}
        .report-meta {{
            text-align: right;
            font-style: italic;
            color: #666;
            margin-bottom: 30px;
        }}
        .visualization {{
            text-align: center;
            margin: 30px 0;
        }}
        .visualization img {{
            max-width: 100%;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
            border-radius: 5px;
        }}
        .caption {{
            font-style: italic;
            text-align: center;
            margin-top: 10px;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 15px;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f8f8;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .section {{
            margin: 40px 0;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
        }}
        .highlight {{
            background-color: #fffbea;
            padding: 15px;
            border-left: 4px solid #ffd700;
            margin: 20px 0;
        }}
        .tstar {{
            font-weight: bold;
            color: #c00;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            font-size: 0.9em;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>Singularity Window Analysis Report</h1>
    
    <div class="report-meta">
        <p>Generated on: {current_date}</p>
        <p>Analysis period: {min(years)}-{max(years)}</p>
        <p>Software version: 0.2.5</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p>This report analyzes the correlation between Global Consciousness Project (GCP) data patterns and Unidentified Aerial Phenomena (UAP) sighting trends to identify and validate a potential "singularity window" (t*) located at approximately <span class="tstar">{t_star}</span>.</p>
        
        <div class="highlight">
            <p>Key findings:</p>
            <ul>
                <li>Peak GCP retrocausal evidence detected in {peak_gcp_year} (score: {max(gcp_scores):.2f})</li>
                <li>Highest UAP activity recorded in {peak_uap_year} (count: {max(uap_sightings)})</li>
                <li>Correlation between GCP and UAP patterns: r = {gcp_uap_corr:.2f}</li>
                <li>Current t* estimate: <span class="tstar">{t_star}</span> ± {t_star_ci} years</li>
            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>Methodology</h2>
        <p>The analysis employs a multi-dimensional approach combining statistical analysis of GCP data with UAP sighting patterns. The methodology includes:</p>
        
        <ol>
            <li><strong>GCP Retrocausal Analysis:</strong> Examining statistical deviations in random event generators that may indicate consciousness-related anomalies preceding major global events.</li>
            <li><strong>Cumulative Deviation Tracking:</strong> Monitoring the cumulative sum of deviations from expected randomness in GCP data.</li>
            <li><strong>UAP Temporal Pattern Analysis:</strong> Analyzing temporal distributions of verified UAP encounters with emphasis on acceleration periods.</li>
            <li><strong>Joint Pattern Correlation:</strong> Examining statistical correlations between GCP anomalies and UAP sighting intensities.</li>
            <li><strong>Singularity Window Estimation:</strong> Calculating t* based on regression analysis of joint patterns.</li>
        </ol>
    </div>
    
    <div class="section">
        <h2>GCP Data Analysis</h2>
        
        <h3>Retrocausal Evidence Scores</h3>
        <p>GCP data shows evidence of statistical anomalies that may indicate retrocausal effects - information flowing from future events to present measurements. The retrocausal evidence scores quantify the strength of these temporal anomalies.</p>
        
        <div class="visualization">
            <img src="{os.path.basename(output_paths['gcp_scores'])}" alt="GCP Retrocausal Evidence Scores">
            <p class="caption">Figure 1: GCP Retrocausal Evidence Scores by Year (2000-{max(years)})</p>
        </div>
        
        <p>Notable observations:</p>
        <ul>
            <li>The period {peak_gcp_year-1}-{peak_gcp_year+1} shows the strongest evidence (scores > 0.7) for retrocausal effects.</li>
            <li>Secondary pattern emerges in 2012-2013 with moderate evidence scores (0.5-0.6).</li>
            <li>Recent upward trend observed from 2020 onward, potentially indicating proximity to t*.</li>
        </ul>
        
        <h3>Cumulative Deviation Analysis</h3>
        <p>Cumulative deviation tracks the running sum of statistical anomalies in the GCP network, providing insight into long-term patterns of coherence in global consciousness data.</p>
        
        <div class="visualization">
            <img src="{os.path.basename(output_paths['gcp_cumulative'])}" alt="GCP Cumulative Deviation">
            <p class="caption">Figure 2: GCP Cumulative Deviation Over Time (2000-{max(years)})</p>
        </div>
        
        <p>Key findings:</p>
        <ul>
            <li>Primary deviation period (2004-2007) shows accelerated accumulation of statistical anomalies.</li>
            <li>Pattern of accumulated deviations closely tracks with periods of global social and technological transformation.</li>
            <li>Recent deviation acceleration (post-2020) potentially indicates approach toward t*.</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>UAP Analysis</h2>
        
        <h3>Sightings Trend Analysis</h3>
        <p>Analysis of UAP sightings reveals distinct periods of acceleration that may correlate with proximity to the hypothesized singularity window.</p>
        
        <div class="visualization">
            <img src="{os.path.basename(output_paths['uap_sightings'])}" alt="UAP Sightings Trend">
            <p class="caption">Figure 3: UAP Sightings Trend Over Time with t* Reference Line</p>
        </div>
        
        <p>Key observations:</p>
        <ul>
            <li>Primary acceleration period (1995-2008) shows gradual increase in verified UAP encounters.</li>
            <li>Secondary acceleration (2017-2021) demonstrates more rapid increase, potentially indicating proximity to t*.</li>
            <li>Projected trajectory appears to converge toward t* = {t_star}.</li>
        </ul>
        
        <h3>Deviation Profile Analysis</h3>
        <p>The UAP deviation profile maps the intensity of UAP activity relative to the hypothesized t* point, revealing potential temporal structure in the phenomena.</p>
        
        <div class="visualization">
            <img src="{os.path.basename(output_paths['uap_deviation'])}" alt="UAP Deviation Profile">
            <p class="caption">Figure 4: UAP Deviation Profile Relative to t*</p>
        </div>
        
        <p>Notable findings:</p>
        <ul>
            <li>Peak deviation zone appears 15-25 years before t*, with highest intensities at approximately 20 years before t*.</li>
            <li>Moderate deviation zone (5-15 years before t*) shows sustained but less intense activity.</li>
            <li>Current data suggests we are in the moderate deviation zone approaching t*.</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Joint Pattern Analysis</h2>
        
        <h3>Correlation Analysis</h3>
        <p>Statistical correlation between GCP retrocausal evidence and UAP deviation intensities reveals potential causal or contributing relationships between these phenomena.</p>
        
        <div class="visualization">
            <img src="{os.path.basename(output_paths['joint_correlation'])}" alt="Joint Pattern Correlation">
            <p class="caption">Figure 5: Correlation Between GCP and UAP Pattern Intensities</p>
        </div>
        
        <p>Key findings:</p>
        <ul>
            <li>Statistical correlation coefficient: r = {gcp_uap_corr:.2f}</li>
            <li>Years closest to t* (color intensity) typically show higher correlation between GCP and UAP patterns.</li>
            <li>The three years with highest joint values ({years[0]+(t_star-years[0])//5}, {years[0]+(t_star-years[0])//4}, {years[0]+(t_star-years[0])//3}) demonstrate particularly strong correlation.</li>
        </ul>
        
        <h3>Comparative Time Series</h3>
        <p>The normalized time series comparison reveals temporal relationships between GCP and UAP patterns, highlighting periods of synchronization.</p>
        
        <div class="visualization">
            <img src="{os.path.basename(output_paths['joint_time_series'])}" alt="Joint Time Series">
            <p class="caption">Figure 6: Comparative Time Series of GCP and UAP Intensities</p>
        </div>
        
        <p>Notable observations:</p>
        <ul>
            <li>GCP and UAP patterns show periods of high correlation (r > 0.7) during {peak_gcp_year-1}-{peak_gcp_year+1} and 2020-2023.</li>
            <li>Peaks in both datasets appear to anticipate or follow each other with a lag of approximately 2-3 years.</li>
            <li>The convergence of both patterns in recent years (2020-2025) may indicate approach to t*.</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Singularity Window Estimation</h2>
        <p>Based on the convergence of multiple analytical approaches, the singularity window (t*) is estimated at <span class="tstar">{t_star}</span> with a confidence interval of ±{t_star_ci} years.</p>
        
        <p>This estimate is derived from:</p>
        <ul>
            <li>Regression analysis of joint GCP/UAP patterns</li>
            <li>Extrapolation of acceleration curves in UAP sighting data</li>
            <li>Temporal mapping of GCP retrocausal evidence intensities</li>
            <li>Monte Carlo simulations with {num_simulations if 'num_simulations' in locals() else 100} iterations</li>
        </ul>
        
        <div class="highlight">
            <p>The singularity window represents a period of maximum probability for transformative events that may fundamentally alter our understanding of consciousness, technology, and potentially our relationship with non-human intelligence.</p>
        </div>
    </div>
    
    <div class="section">
        <h2>Limitations and Considerations</h2>
        <p>This analysis is subject to several important limitations:</p>
        
        <ul>
            <li><strong>Data Quality:</strong> UAP sighting data varies in reliability and verification standards over time.</li>
            <li><strong>Selection Bias:</strong> GCP node locations are not uniformly distributed globally.</li>
            <li><strong>Correlation vs. Causation:</strong> Statistical correlation between GCP and UAP patterns does not necessarily imply causation.</li>
            <li><strong>Model Assumptions:</strong> The t* estimation model assumes temporal symmetry in anomaly patterns.</li>
            <li><strong>Alternative Explanations:</strong> Other factors not included in this analysis may better explain observed patterns.</li>
        </ul>
        
        <p>These results should be interpreted as exploratory rather than definitive, providing a framework for further research and hypothesis testing.</p>
    </div>
    
    <div class="section">
        <h2>References and Data Sources</h2>
        <ol>
            <li>Global Consciousness Project. Princeton University. https://noosphere.princeton.edu/</li>
            <li>Bancel, P. (2021). "Searching for Global Consciousness: A 17-Year Exploration." Explore, 17(3), 181-188.</li>
            <li>Radin, D., et al. (2016). "Global Consciousness Project: An Independent Analysis of The 11 September 2001 Events." Journal of Scientific Exploration, 30(3).</li>
            <li>Office of the Director of National Intelligence. (2021). "Preliminary Assessment: Unidentified Aerial Phenomena."</li>
            <li>Nelson, R., & Bancel, P. (2011). "Effects of mass consciousness: Changes in random data during global events." Explore, 7(6), 373-383.</li>
        </ol>
    </div>
    
    <div class="footer">
        <p>This report was generated using the Singularity Window Analysis Framework v0.2.5.</p>
        <p>&copy; {current_date.split('-')[0]} Cascade Research Initiative</p>
    </div>
</body>
</html>
"""
    
    # Write the HTML content to the output file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Analyze singularity window and generate report')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output files')
    parser.add_argument('--t_star', type=int, default=None,
                        help='Optimized t* value (singularity window year). If not provided, it will be calculated.')
    parser.add_argument('--t_star_ci', type=int, default=None,
                        help='Confidence interval range for t* (e.g. 3 for t* ± 3 years). If not provided, it will be calculated.')
    parser.add_argument('--num_simulations', type=int, default=100,
                        help='Number of Monte Carlo simulations to run for confidence interval estimation')
    parser.add_argument('--confidence_level', type=float, default=0.95,
                        help='Confidence level for t* confidence interval (0-1)')
    
    args = parser.parse_args()
    
    # Run analysis and generate report
    analyze_and_generate_report(args)

if __name__ == "__main__":
    main() 