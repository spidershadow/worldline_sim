"""Utility functions for pattern analysis modules.

This module provides common utility functions used by various pattern
analysis modules, particularly for the EnhancedRngPattern class.
"""

import os
import requests
import numpy as np
import pandas as pd
from pathlib import Path


def fetch_csv(url, local_path=None, force_download=False):
    """Fetch a CSV file from a URL or use local copy if available.
    
    Args:
        url: URL to fetch the CSV from
        local_path: Path to save/load the local copy
        force_download: Whether to download even if local copy exists
        
    Returns:
        Pandas DataFrame containing the CSV data
    """
    if local_path and os.path.exists(local_path) and not force_download:
        try:
            return pd.read_csv(local_path)
        except Exception as e:
            print(f"Error reading local CSV: {e}")
    
    # Download from URL
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(response.text))
        
        # Save local copy if path provided
        if local_path:
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            df.to_csv(local_path, index=False)
            
        return df
    except Exception as e:
        print(f"Error fetching CSV from URL: {e}")
        return None


def calculate_symmetry_score(before_values, after_values):
    """Calculate a symmetry score between two sequences.
    
    This function calculates how symmetrical two sequences are around a central point.
    Higher values indicate better symmetry.
    
    Args:
        before_values: Values before the central point
        after_values: Values after the central point
        
    Returns:
        Symmetry score between 0 and 1
    """
    if not before_values or not after_values:
        return 0
    
    # Ensure sequences are numpy arrays
    before = np.array(before_values)
    after = np.array(after_values)
    
    # If sequences have different lengths, trim to the shorter one
    min_len = min(len(before), len(after))
    before = before[-min_len:] if len(before) > min_len else before
    after = after[:min_len] if len(after) > min_len else after
    
    # Reverse the 'after' sequence for symmetry comparison
    after_rev = after[::-1]
    
    # Calculate correlation
    if len(before) > 1:
        corr = np.corrcoef(before, after_rev)[0, 1]
        # Convert from [-1, 1] to [0, 1]
        sym_score = (abs(corr) + 1) / 2
    else:
        # For single values, compare absolute difference
        diff = abs(before[0] - after_rev[0])
        max_val = max(abs(before[0]), abs(after_rev[0]))
        sym_score = 1 - (diff / max_val if max_val > 0 else 0)
    
    return float(sym_score)


def rolling_z_score(values, window=24):
    """Calculate rolling Z-scores for a sequence of values.
    
    Args:
        values: Sequence of values
        window: Window size for rolling calculations
        
    Returns:
        Array of rolling Z-scores
    """
    if len(values) < window:
        return np.zeros(len(values))
        
    # Convert to numpy array if needed
    vals = np.array(values)
    
    # Calculate rolling mean and std
    rolling_mean = np.zeros(len(vals))
    rolling_std = np.zeros(len(vals))
    
    for i in range(len(vals)):
        if i < window:
            # For the first window positions, use all available data
            window_vals = vals[:i+1]
        else:
            # For later positions, use the specified window
            window_vals = vals[i-window+1:i+1]
            
        rolling_mean[i] = np.mean(window_vals)
        rolling_std[i] = np.std(window_vals)
    
    # Calculate Z-scores, avoiding division by zero
    z_scores = np.zeros(len(vals))
    for i in range(len(vals)):
        if rolling_std[i] > 0:
            z_scores[i] = (vals[i] - rolling_mean[i]) / rolling_std[i]
    
    return z_scores


def detect_anomalies(values, threshold=2.0, window=24):
    """Detect anomalies in a sequence using rolling Z-scores.
    
    Args:
        values: Sequence of values
        threshold: Z-score threshold for anomaly detection
        window: Window size for rolling calculations
        
    Returns:
        Dictionary with anomaly indices and scores
    """
    # Calculate rolling Z-scores
    z_scores = rolling_z_score(values, window)
    
    # Find anomalies
    anomalies = np.where(np.abs(z_scores) > threshold)[0]
    
    # Create result dictionary
    result = {
        'anomaly_indices': anomalies.tolist(),
        'anomaly_values': [values[i] for i in anomalies],
        'anomaly_z_scores': [z_scores[i] for i in anomalies],
    }
    
    return result


def calculate_cumulative_deviation(values, expected=None):
    """Calculate cumulative deviation of values from their mean.
    
    Args:
        values: Sequence of values
        expected: Expected value (if None, uses mean of values)
        
    Returns:
        Array of cumulative deviations
    """
    if not values:
        return []
        
    # Convert to numpy array
    vals = np.array(values)
    
    # Calculate deviations from expected value or mean
    if expected is None:
        deviations = vals - np.mean(vals)
    else:
        deviations = vals - expected
    
    # Calculate cumulative sum
    cum_dev = np.cumsum(deviations)
    
    return cum_dev.tolist() if isinstance(cum_dev, np.ndarray) else cum_dev 