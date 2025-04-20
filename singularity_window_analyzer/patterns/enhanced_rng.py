"""Pattern implementation for analysis of RNG data based on Global Consciousness Project methodology.

This pattern implements statistical approaches to detect potential retrocausal effects
in Random Number Generator (RNG) data, building on methods from the Global Consciousness Project.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import requests
import gzip
import io

# Import from our utils module
from patterns.utils import fetch_csv, calculate_symmetry_score

# New import for direct RNG data access
from scripts.fetch_rng_direct import fetch_day_data, process_day_data

# Directory for storing data
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Direct URL for fetching RNG data from the Global Consciousness Project
GCP_DATA_URL = "https://noosphere.princeton.edu/cgi-bin/eggdatareq.pl"

# Historical GCP cumulative significance milestones with projection toward 2035
# Year: (Cumulative events, Approximate Z-score, Probability)
GCP_SIGNIFICANCE_MILESTONES = {
    2000: (63, 3.0, 1.0/1000),           # ~3σ (1 in 1,000)
    2004: (174, 4.4, 1.0/300000),        # ~4.4σ (1 in few hundred thousand)
    2006: (250, 5.0, 1.0/1000000),       # ~5σ (1 in million)
    2014: (450, 7.0, 1.0/1000000000000), # ~7σ (1 in trillion)
    2015: (500, 7.3, 1.33e-13),          # 7.3σ (1 in 7.5 trillion)
    # Projected values based on growth trend toward singularity at ~2032
    2020: (650, 8.1, 1.0e-16),           # Projection
    2025: (800, 9.2, 1.0e-20),           # Projection 
    2030: (1000, 11.0, 1.0e-28),         # Projection approaching singularity
    2035: (1250, 14.0, 1.0e-40)          # Projection past singularity
}

class EnhancedRngPattern:
    """Pattern detector based on RNG deviations around a specified time (T*).
    
    This pattern looks for statistical anomalies in random number generator data
    surrounding significant global events, using methods derived from the Global
    Consciousness Project but with enhancements for potential retrocausal effects.
    """
    
    def __init__(self, window_hours=24, direction="both", threshold=1.96):
        """Initialize the RNG pattern detector.
        
        Args:
            window_hours: Number of hours to analyze before and after the target time
            direction: Whether to analyze data "before", "after", or "both" relative to target time
            threshold: Z-score threshold for significance (default: 1.96, corresponding to p=0.05)
        """
        self.window_hours = window_hours
        self.direction = direction
        self.threshold = threshold
        self.rng_data = None
        self.daily_data = None
        self.symmetry_scores = []
        self.gcp_milestones = GCP_SIGNIFICANCE_MILESTONES
        
        # Load or download the RNG data
        self._load_rng_data()
        
    def _load_rng_data(self):
        """Load RNG data from CSV or download if not available."""
        csv_path = DATA_DIR / "rng_avg_abs_z.csv"
        daily_path = DATA_DIR / "rng_daily_data.csv"
        
        # Try to load the RNG yearly data
        if os.path.exists(csv_path):
            try:
                self.rng_data = pd.read_csv(csv_path)
                print(f"Loaded RNG data with {len(self.rng_data)} yearly records")
            except Exception as e:
                print(f"Error loading RNG data: {e}")
                self._use_fallback_data()
        else:
            # If file doesn't exist, use fallback data
            self._use_fallback_data()
            
        # Try to load the daily data if available
        if os.path.exists(daily_path):
            try:
                self.daily_data = pd.read_csv(daily_path)
                self.daily_data['date'] = pd.to_datetime(self.daily_data['date'])
                print(f"Loaded daily RNG data with {len(self.daily_data)} records")
            except Exception as e:
                print(f"Error loading daily RNG data: {e}")
                self.daily_data = None
                
        # Enhance the data with GCP significance milestones
        self._incorporate_gcp_significance()
    
    def _use_fallback_data(self):
        """Use fallback data when RNG data file is not available."""
        print("Using fallback RNG data")
        # The Global Consciousness Project started in 1998
        # This is simplified fallback data for demonstration
        years = list(range(1998, 2024))
        # These values approximate the avg_abs_z scores from GCP data
        # with a slight increasing trend
        base_values = np.linspace(0.98, 1.05, len(years))
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.02, len(years))
        values = base_values + noise
        
        self.rng_data = pd.DataFrame({
            'year': years,
            'value': values
        })
    
    def _incorporate_gcp_significance(self):
        """Incorporate GCP significance milestones into our model."""
        if self.rng_data is None:
            return
            
        # Add columns for cumulative z-score and significance
        if 'cum_z_score' not in self.rng_data.columns:
            self.rng_data['cum_z_score'] = np.nan
            self.rng_data['cum_p_value'] = np.nan
            self.rng_data['cum_events'] = np.nan
            
            # Fill in known milestone values
            for year, (events, z_score, p_value) in self.gcp_milestones.items():
                if year in self.rng_data['year'].values:
                    idx = self.rng_data[self.rng_data['year'] == year].index
                    self.rng_data.loc[idx, 'cum_z_score'] = z_score
                    self.rng_data.loc[idx, 'cum_p_value'] = p_value
                    self.rng_data.loc[idx, 'cum_events'] = events
            
            # Interpolate missing values between milestones
            self.rng_data['cum_z_score'] = self.rng_data['cum_z_score'].interpolate(method='linear')
            self.rng_data['cum_p_value'] = self.rng_data['cum_p_value'].interpolate(method='linear')
            self.rng_data['cum_events'] = self.rng_data['cum_events'].interpolate(method='linear').round()
            
            # Fill any missing values at the beginning or end
            self.rng_data['cum_z_score'] = self.rng_data['cum_z_score'].fillna(method='ffill').fillna(method='bfill')
            self.rng_data['cum_p_value'] = self.rng_data['cum_p_value'].fillna(method='ffill').fillna(method='bfill')
            self.rng_data['cum_events'] = self.rng_data['cum_events'].fillna(method='ffill').fillna(method='bfill')
    
    def _fetch_specific_day_data(self, target_date):
        """Fetch RNG data for a specific day directly from GCP website if needed."""
        # Check if we already have this date in our daily data
        if self.daily_data is not None:
            date_data = self.daily_data[self.daily_data['date'].dt.date == target_date.date()]
            if not date_data.empty:
                return date_data.iloc[0]['avg_abs_z']
        
        # If we don't have the data, try to fetch it directly from GCP
        print(f"Fetching RNG data for {target_date.date()} directly from GCP...")
        
        # First try using the existing fetch_day_data function
        raw_data = fetch_day_data(target_date.year, target_date.month, target_date.day)
        if raw_data:
            day_stats = process_day_data(raw_data, target_date.year, target_date.month, target_date.day)
            if day_stats:
                return day_stats['avg_abs_z']
        
        # If that fails, try using the direct URL approach
        try:
            params = {
                'z': 1,
                'year': target_date.year,
                'month': target_date.month,
                'day': target_date.day,
                'stime': '00:00:00',
                'etime': '23:59:59',
                'gzip': 'Yes',
                'idate': 'Yes'
            }
            
            response = requests.get(GCP_DATA_URL, params=params, timeout=10)
            if response.status_code == 200:
                # Decompress gzip content
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
                    data_str = f.read().decode('utf-8')
                
                # Process the data using the existing function
                day_stats = process_day_data(data_str, target_date.year, target_date.month, target_date.day)
                if day_stats:
                    return day_stats['avg_abs_z']
        
        except Exception as e:
            print(f"Error fetching RNG data directly from URL: {e}")
        
        # If direct fetch failed, estimate from yearly data
        print(f"Using estimated RNG data for {target_date.date()}")
        return self._estimate_from_yearly_data(target_date.year)
    
    def _estimate_from_yearly_data(self, year):
        """Estimate a daily RNG value based on yearly data."""
        if year < self.rng_data['year'].min():
            return self.rng_data['value'].iloc[0]
        if year > self.rng_data['year'].max():
            return self.rng_data['value'].iloc[-1]
            
        year_data = self.rng_data[self.rng_data['year'] == year]
        if not year_data.empty:
            return year_data.iloc[0]['value']
        else:
            # Interpolate between years if needed
            prev_year = self.rng_data[self.rng_data['year'] < year]['year'].max()
            next_year = self.rng_data[self.rng_data['year'] > year]['year'].min()
            
            prev_value = self.rng_data[self.rng_data['year'] == prev_year]['value'].iloc[0]
            next_value = self.rng_data[self.rng_data['year'] == next_year]['value'].iloc[0]
            
            # Linear interpolation
            weight = (year - prev_year) / (next_year - prev_year)
            return prev_value + weight * (next_value - prev_value)
    
    def get_cumulative_z_score(self, year):
        """Get the cumulative Z-score for a given year based on GCP milestones."""
        if self.rng_data is None or 'cum_z_score' not in self.rng_data.columns:
            # Use milestone data directly if rng_data is not available
            if year in self.gcp_milestones:
                return self.gcp_milestones[year][1]  # Z-score
            elif year < min(self.gcp_milestones.keys()):
                return 0.0  # Early years before milestones
            elif year > max(self.gcp_milestones.keys()):
                # Use latest milestone plus a small increment for future years
                latest_year = max(self.gcp_milestones.keys())
                latest_z = self.gcp_milestones[latest_year][1]
                years_diff = year - latest_year
                # Assume diminishing increases in Z-score over time
                return latest_z + 0.1 * np.log(1 + years_diff)
            else:
                # Interpolate between milestone years
                prev_year = max([y for y in self.gcp_milestones.keys() if y <= year])
                next_year = min([y for y in self.gcp_milestones.keys() if y >= year])
                if prev_year == next_year:
                    return self.gcp_milestones[prev_year][1]
                    
                prev_z = self.gcp_milestones[prev_year][1]
                next_z = self.gcp_milestones[next_year][1]
                weight = (year - prev_year) / (next_year - prev_year)
                return prev_z + weight * (next_z - prev_z)
        else:
            # Use processed rng_data if available
            if year < self.rng_data['year'].min():
                return self.rng_data['cum_z_score'].iloc[0]
            if year > self.rng_data['year'].max():
                return self.rng_data['cum_z_score'].iloc[-1]
                
            year_data = self.rng_data[self.rng_data['year'] == year]
            if not year_data.empty:
                return year_data.iloc[0]['cum_z_score']
            else:
                # Interpolate between years
                prev_year = self.rng_data[self.rng_data['year'] < year]['year'].max()
                next_year = self.rng_data[self.rng_data['year'] > year]['year'].min()
                
                prev_z = self.rng_data[self.rng_data['year'] == prev_year]['cum_z_score'].iloc[0]
                next_z = self.rng_data[self.rng_data['year'] == next_year]['cum_z_score'].iloc[0]
                
                weight = (year - prev_year) / (next_year - prev_year)
                return prev_z + weight * (next_z - prev_z)
    
    def calculate_deviations(self, target_time):
        """Calculate RNG deviations around the specified time (T*).
        
        Args:
            target_time: Datetime object representing the time of interest (T*)
            
        Returns:
            Dictionary with deviation statistics
        """
        if not isinstance(target_time, datetime):
            try:
                target_time = pd.to_datetime(target_time)
            except:
                raise ValueError("target_time must be a datetime object or convertible to one")
        
        # Create time windows based on direction parameter
        if self.direction == "before":
            start_time = target_time - timedelta(hours=self.window_hours)
            end_time = target_time
        elif self.direction == "after":
            start_time = target_time
            end_time = target_time + timedelta(hours=self.window_hours)
        else:  # "both"
            start_time = target_time - timedelta(hours=self.window_hours)
            end_time = target_time + timedelta(hours=self.window_hours)
        
        # Sample RNG values around the target time
        z_values = self._sample_rng_pattern(start_time, end_time, target_time)
        
        # Calculate cumulative deviation
        if len(z_values) > 0:
            cum_dev = np.cumsum(z_values - np.mean(z_values))
            max_dev = np.max(np.abs(cum_dev))
            
            # Calculate symmetry score if we're looking at both before and after
            symmetry_score = 0
            if self.direction == "both":
                mid_point = len(z_values) // 2
                before_values = z_values[:mid_point]
                after_values = z_values[mid_point:]
                
                # Only calculate if we have data on both sides
                if len(before_values) > 0 and len(after_values) > 0:
                    symmetry_score = calculate_symmetry_score(before_values, after_values)
                    self.symmetry_scores.append(symmetry_score)
            
            # Check if deviation exceeds threshold
            significance = max_dev > (self.threshold * np.sqrt(len(z_values)))
            
            # Get historical GCP significance for this year
            year_cum_z = self.get_cumulative_z_score(target_time.year)
            
            return {
                'target_time': target_time,
                'z_values': z_values,
                'cum_dev': cum_dev.tolist() if isinstance(cum_dev, np.ndarray) else cum_dev,
                'max_dev': max_dev,
                'symmetry_score': symmetry_score,
                'significant': significance,
                'sample_size': len(z_values),
                'historical_cum_z': year_cum_z
            }
        else:
            return {
                'target_time': target_time,
                'z_values': [],
                'cum_dev': [],
                'max_dev': 0,
                'symmetry_score': 0,
                'significant': False,
                'sample_size': 0,
                'historical_cum_z': self.get_cumulative_z_score(target_time.year)
            }
    
    def _sample_rng_pattern(self, start_time, end_time, target_time):
        """Sample RNG pattern around the target time.
        
        Args:
            start_time: Start of sampling window
            end_time: End of sampling window
            target_time: The target time (T*)
            
        Returns:
            List of z-values sampled from the time window
        """
        # Initialize empty list for z-values
        z_values = []
        
        # Loop through days in the window
        current_date = start_time.date()
        end_date = end_time.date()
        
        while current_date <= end_date:
            current_datetime = datetime.combine(current_date, datetime.min.time())
            
            # Get RNG z-value for this day
            daily_z = self._fetch_specific_day_data(current_datetime)
            
            # Apply constraints based on target time
            time_diff = (current_datetime - target_time).total_seconds() / 3600  # in hours
            
            # Apply weighting based on GCP significance
            # Years with higher GCP significance (higher Z-scores) get higher weight
            # This accounts for the observed increasing anomalies in the GCP data over time
            year_weight = 1.0 + 0.1 * self.get_cumulative_z_score(current_datetime.year) / 7.3  # Normalize by max Z-score
            
            # Also weight by time proximity - values closer to target time get higher weight
            time_weight = 1.0 / (1.0 + abs(time_diff) / 24.0)
            
            # Combine weights
            combined_weight = year_weight * time_weight
            
            # Add weighted z-value to list, repeating according to weight to simulate hourly data
            samples = max(1, int(combined_weight * 24))  # Up to 24 samples per day
            z_values.extend([daily_z] * samples)
        
            # Move to next day
            current_date += timedelta(days=1)
        
        return z_values
        
    def check_constraints(self, z_values, target_time):
        """Check various constraints on RNG values.
        
        Args:
            z_values: List of z-values to check
            target_time: The target time (T*)
            
        Returns:
            Dictionary of constraint check results
        """
        if not z_values:
            return {'constraints_met': False, 'reason': 'No data'}
        
        # Check for sustained deviation (the primary GCP metric)
        cum_z = np.cumsum(z_values - np.mean(z_values))
        sustained_dev = np.max(np.abs(cum_z)) > (1.96 * np.sqrt(len(z_values)))
        
        # Check for temporal symmetry around T*
        symmetry = 0
        if self.direction == "both":
            mid_point = len(z_values) // 2
            before_values = z_values[:mid_point]
            after_values = z_values[mid_point:]
            
            if len(before_values) > 0 and len(after_values) > 0:
                symmetry = calculate_symmetry_score(before_values, after_values)
        
        # Check for correlation with historical GCP patterns
        # Use the known GCP milestones to inform our assessment
        year = target_time.year
        year_cum_z = self.get_cumulative_z_score(year)
        
        # Calculate current z-score
        current_z = np.sum(z_values - np.mean(z_values)) / np.sqrt(len(z_values))
        
        # Higher correlation if our z-score is consistent with historical trend
        historical_correlation = abs(current_z - year_cum_z/10) < 0.5
        
        # Results dictionary
        results = {
            'sustained_deviation': sustained_dev,
            'temporal_symmetry': symmetry > 0.6,  # Threshold for good symmetry
            'historical_correlation': historical_correlation,
            'constraints_met': sustained_dev and (symmetry > 0.6 or self.direction != "both"),
            'gcp_cum_z': year_cum_z
        }
        
        return results
        
    def assess_retrocausal_evidence(self, target_time, window_years=5):
        """Assess evidence for retrocausal effects around a target time.
        
        This method analyzes GCP data before and after the target year to look for
        evidence of retrocausal influence based on the trends in cumulative Z-scores.
        
        Args:
            target_time: Datetime object or year representing the time of interest (T*)
            window_years: Number of years to look before and after the target
            
        Returns:
            Dictionary with retrocausal analysis results
        """
        # Convert to year if datetime is provided
        if isinstance(target_time, datetime):
            target_year = target_time.year
        elif isinstance(target_time, (int, float)):
            target_year = int(target_time)
        else:
            try:
                target_year = pd.to_datetime(target_time).year
            except:
                raise ValueError("target_time must be a datetime, year, or convertible to one")
        
        # Get years before and after the target
        before_years = range(target_year - window_years, target_year)
        after_years = range(target_year + 1, target_year + window_years + 1)
        
        # Get GCP cumulative Z-scores for these years
        before_z = [self.get_cumulative_z_score(y) for y in before_years]
        target_z = self.get_cumulative_z_score(target_year)
        after_z = [self.get_cumulative_z_score(y) for y in after_years]
        
        # Calculate rates of change
        if before_z:
            before_slope = (target_z - before_z[0]) / len(before_z) if len(before_z) > 0 else 0
        else:
            before_slope = 0
            
        if after_z:
            after_slope = (after_z[-1] - target_z) / len(after_z) if len(after_z) > 0 else 0
        else:
            after_slope = 0
        
        # Calculate acceleration at the target year
        # Positive acceleration indicates a potential influence on future/past
        acceleration = after_slope - before_slope
        
        # Calculate symmetry between before and after patterns
        # High symmetry might indicate retrocausal influence
        if before_z and after_z:
            # Normalize slopes for comparison
            norm_before = [(z - before_z[0]) / (target_z - before_z[0]) if target_z != before_z[0] else 0 for z in before_z]
            norm_after = [(z - target_z) / (after_z[-1] - target_z) if after_z[-1] != target_z else 0 for z in after_z]
            
            # Calculate correlation between normalized patterns
            if len(norm_before) == len(norm_after):
                symmetry = np.corrcoef(norm_before, norm_after)[0, 1] if len(norm_before) > 1 else 0
            else:
                # If lengths differ, use the shorter one
                min_len = min(len(norm_before), len(norm_after))
                symmetry = np.corrcoef(norm_before[:min_len], norm_after[:min_len])[0, 1] if min_len > 1 else 0
        else:
            symmetry = 0
        
        # Assess evidence for retrocausal influence
        # Higher scores indicate stronger evidence
        retrocausal_score = 0.4 * abs(acceleration) + 0.6 * abs(symmetry)
        if np.isnan(retrocausal_score):
            retrocausal_score = 0
            
        # Classify the result
        if retrocausal_score > 0.7:
            evidence = "Strong"
        elif retrocausal_score > 0.5:
            evidence = "Moderate"
        elif retrocausal_score > 0.3:
            evidence = "Weak"
        else:
            evidence = "None"
            
        # Return results
        return {
            'target_year': target_year,
            'before_years': list(before_years),
            'after_years': list(after_years),
            'before_z_scores': before_z,
            'target_z_score': target_z,
            'after_z_scores': after_z,
            'before_slope': before_slope,
            'after_slope': after_slope,
            'acceleration': acceleration,
            'symmetry': symmetry,
            'retrocausal_score': retrocausal_score,
            'evidence': evidence
        }
    
    def to_yearly_intensity(self):
        """Return DataFrame(year, gcp_intensity) using cum_z_score
        (or any scalar you like)."""
        years = list(range(int(self.rng_data['year'].min()), int(self.rng_data['year'].max()) + 1))
        intensities = [self.get_cumulative_z_score(year) for year in years]
        
        return pd.DataFrame({
            "year": years,
            "gcp_intensity": intensities
        }) 
    
    def get_yearly_intensity_series(self):
        """Return DataFrame with yearly intensity data.
        This is an alias for to_yearly_intensity() for compatibility."""
        return self.to_yearly_intensity() 