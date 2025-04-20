"""Global Consciousness Project (GCP) data handling and analysis.

This module provides functionality for working with Global Consciousness Project data,
including historical Z-score milestones, statistical significance calculations,
and methods for analyzing RNG data in the context of global consciousness research.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats, interpolate


# Historical GCP cumulative significance milestones
# Source: Global Consciousness Project (GCP) reports and publications
GCP_SIGNIFICANCE_MILESTONES = {
    # Year: (cumulative_events, approximate_z_score, probability)
    2000: (75, 3.0, 0.001),  # ~3σ, 1 in 1,000
    2004: (180, 4.4, 1.0e-5),  # ~4.4σ, few hundred thousand to 1
    2006: (220, 5.0, 1.0e-6),  # >5σ, over 1 in 1,000,000
    2014: (480, 7.0, 1.0e-12),  # ~7σ, 1 in trillion
    2015: (500, 7.3, 1.3e-13),  # 7.3σ, 1 in 7.5 trillion
    2020: (550, 7.5, 6.3e-14),  # Estimate based on trend
}


def interpolate_z_scores(start_year: int = 2000, end_year: int = 2023) -> pd.DataFrame:
    """Interpolate GCP Z-scores for years not explicitly in the milestones.
    
    Args:
        start_year: First year to include
        end_year: Last year to include
        
    Returns:
        DataFrame with interpolated values for all years in range
    """
    years = list(GCP_SIGNIFICANCE_MILESTONES.keys())
    events = [GCP_SIGNIFICANCE_MILESTONES[y][0] for y in years]
    z_scores = [GCP_SIGNIFICANCE_MILESTONES[y][1] for y in years]
    probs = [GCP_SIGNIFICANCE_MILESTONES[y][2] for y in years]
    
    # Create interpolation functions
    events_interp = interpolate.interp1d(years, events, kind='linear', 
                                         fill_value='extrapolate')
    z_score_interp = interpolate.interp1d(years, z_scores, kind='linear', 
                                          fill_value='extrapolate')
    
    # Calculate for all years
    all_years = list(range(start_year, end_year + 1))
    all_events = events_interp(all_years)
    all_z_scores = z_score_interp(all_years)
    
    # Calculate probabilities from z-scores
    all_probs = [stats.norm.sf(abs(z)) for z in all_z_scores]
    
    # Create DataFrame
    df = pd.DataFrame({
        'year': all_years,
        'cumulative_events': [round(e) for e in all_events],
        'z_score': all_z_scores,
        'probability': all_probs
    })
    
    return df


def get_z_score_for_year(year: int) -> float:
    """Get the interpolated Z-score for a specific year.
    
    Args:
        year: The year to get Z-score for
        
    Returns:
        Interpolated Z-score for the specified year
    """
    if year in GCP_SIGNIFICANCE_MILESTONES:
        return GCP_SIGNIFICANCE_MILESTONES[year][1]
        
    df = interpolate_z_scores()
    year_data = df[df['year'] == year]
    
    if not year_data.empty:
        return year_data.iloc[0]['z_score']
    else:
        # Fallback for years outside our interpolation range
        min_year = min(GCP_SIGNIFICANCE_MILESTONES.keys())
        max_year = max(GCP_SIGNIFICANCE_MILESTONES.keys())
        
        if year < min_year:
            return GCP_SIGNIFICANCE_MILESTONES[min_year][1]
        else:  # year > max_year
            return GCP_SIGNIFICANCE_MILESTONES[max_year][1]


def calculate_event_significance(rng_data: pd.DataFrame, 
                                 event_time: Union[str, datetime],
                                 window_hours: float = 3.0,
                                 direction: str = 'bidirectional') -> Dict:
    """Calculate statistical significance of RNG deviations around an event time.
    
    Args:
        rng_data: DataFrame containing RNG data with timestamp and deviation columns
        event_time: Time of the event (datetime or string in ISO format)
        window_hours: Hours before and after event to analyze
        direction: 'before', 'after', or 'bidirectional'
        
    Returns:
        Dictionary containing significance metrics
    """
    # Ensure event_time is datetime
    if isinstance(event_time, str):
        event_time = datetime.fromisoformat(event_time)
    
    # Calculate window boundaries
    before_start = event_time - timedelta(hours=window_hours)
    after_end = event_time + timedelta(hours=window_hours)
    
    # Filter data within window
    if 'timestamp' in rng_data.columns:
        rng_data = rng_data.copy()
        if isinstance(rng_data['timestamp'].iloc[0], str):
            rng_data['timestamp'] = pd.to_datetime(rng_data['timestamp'])
        
        window_data = rng_data[(rng_data['timestamp'] >= before_start) & 
                               (rng_data['timestamp'] <= after_end)]
    else:
        # If no timestamp column, assume data is already filtered
        window_data = rng_data
    
    # Extract before and after data
    if 'timestamp' in window_data.columns:
        before_data = window_data[window_data['timestamp'] < event_time]
        after_data = window_data[window_data['timestamp'] >= event_time]
        
        if 'deviation' in window_data.columns:
            before_values = before_data['deviation'].values
            after_values = after_data['deviation'].values
        else:
            # Assume first numeric column is the data column
            numeric_cols = window_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data_col = numeric_cols[0]
                before_values = before_data[data_col].values
                after_values = after_data[data_col].values
            else:
                return {
                    'z_score': 0.0,
                    'p_value': 1.0,
                    'samples': 0,
                    'direction': direction,
                    'status': 'error',
                    'message': 'No numeric data column found'
                }
    else:
        # If no timestamp column, use the middle point as the event
        middle = len(window_data) // 2
        before_values = window_data.iloc[:middle].values.flatten()
        after_values = window_data.iloc[middle:].values.flatten()
    
    # Calculate statistics based on direction
    if direction == 'before':
        values = before_values
    elif direction == 'after':
        values = after_values
    else:  # bidirectional
        values = np.concatenate((before_values, after_values))
    
    # Calculate Z-score (assuming RNG data follows standard normal distribution)
    if len(values) > 0:
        mean = np.mean(values)
        sem = stats.sem(values) if len(values) > 1 else 1.0
        z_score = mean / sem if sem > 0 else 0
        p_value = 2 * stats.norm.sf(abs(z_score))  # Two-tailed test
    else:
        z_score = 0
        p_value = 1.0
    
    return {
        'z_score': z_score,
        'p_value': p_value,
        'samples': len(values),
        'mean': np.mean(values) if len(values) > 0 else 0,
        'std': np.std(values) if len(values) > 1 else 0,
        'direction': direction,
        'status': 'ok'
    }


def plot_gcp_significance_trend(output_path: str = None) -> plt.Figure:
    """Create a plot showing the trend of GCP significance over time.
    
    Args:
        output_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    df = interpolate_z_scores()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Z-scores
    ax1.plot(df['year'], df['z_score'], 'b-', linewidth=2)
    ax1.scatter(df['year'], df['z_score'], color='blue', s=30)
    
    # Mark the known milestone years
    milestone_years = list(GCP_SIGNIFICANCE_MILESTONES.keys())
    milestone_zscores = [GCP_SIGNIFICANCE_MILESTONES[y][1] for y in milestone_years]
    ax1.scatter(milestone_years, milestone_zscores, color='red', s=80, 
                label='Published Milestones')
    
    ax1.set_ylabel('Cumulative Z-score')
    ax1.set_title('GCP Cumulative Significance Over Time')
    ax1.grid(True)
    ax1.legend()
    
    # Plot probability (log scale)
    ax2.semilogy(df['year'], df['probability'], 'g-', linewidth=2)
    ax2.scatter(df['year'], df['probability'], color='green', s=30)
    
    milestone_probs = [GCP_SIGNIFICANCE_MILESTONES[y][2] for y in milestone_years]
    ax2.scatter(milestone_years, milestone_probs, color='red', s=80)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Probability (p-value)')
    ax2.set_title('Probability of GCP Results by Chance')
    ax2.grid(True)
    
    # Common threshold lines
    ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='p=0.05')
    ax2.axhline(y=0.01, color='r', linestyle=':', alpha=0.5, label='p=0.01')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def summarize_gcp_milestones(output_path: str = None) -> pd.DataFrame:
    """Create a summary table of GCP significance milestones.
    
    Args:
        output_path: Optional path to save the table as CSV
        
    Returns:
        DataFrame containing the milestones
    """
    data = []
    
    for year, (events, z_score, prob) in GCP_SIGNIFICANCE_MILESTONES.items():
        data.append({
            'Year': year,
            'Cumulative Events': events,
            'Z-score': z_score,
            'P-value': prob,
            'Odds Against Chance': f"1 in {int(1/prob) if prob > 0 else '∞'}",
            'Sigma Level': f"{z_score:.1f}σ",
        })
    
    df = pd.DataFrame(data)
    
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df 