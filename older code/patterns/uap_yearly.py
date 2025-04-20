"""
UAP Yearly Pattern Analysis

This module implements analysis techniques for Unidentified Aerial Phenomena (UAP)
yearly sighting data to detect patterns and potential singularity windows.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

class UapYearlyPattern:
    """Analyzes yearly UAP sighting data to detect patterns and singularity windows."""
    
    def __init__(self, early_data_path="data/uap_early.csv", 
                 recent_data_path="data/uap_recent.csv",
                 window_years=10):
        """
        Initialize UAP pattern detector.
        
        Args:
            early_data_path: Path to CSV with early UAP sighting data
            recent_data_path: Path to CSV with recent UAP sighting data
            window_years: Size of the analysis window in years
        """
        self.early_data_path = early_data_path
        self.recent_data_path = recent_data_path
        self.window_years = window_years
        self.data = None
        self.optimized_t_star = None
        self._load_data()
        
    def _load_data(self):
        """Load and combine UAP sighting data from early and recent datasets."""
        try:
            early_df = pd.read_csv(self.early_data_path)
            recent_df = pd.read_csv(self.recent_data_path)
            
            # Combine the datasets
            self.data = pd.concat([early_df, recent_df]).sort_values('year')
            
            # Fill any gaps in the year sequence
            all_years = np.arange(self.data['year'].min(), self.data['year'].max() + 1)
            full_df = pd.DataFrame({'year': all_years})
            self.data = pd.merge(full_df, self.data, on='year', how='left')
            
            # Interpolate missing values
            self.data['reported_sightings'] = self.data['reported_sightings'].interpolate(method='linear')
            
            # Normalize the data
            self.data['normalized_sightings'] = stats.zscore(self.data['reported_sightings'])
            
        except Exception as e:
            print(f"Error loading UAP data: {e}")
            self._use_fallback_data()
    
    def _use_fallback_data(self):
        """Generate simplified fallback data if real data is unavailable."""
        years = np.arange(1947, 2024)
        # Generate synthetic data with an acceleration point around 2004-2008
        base = np.linspace(100, 500, len(years))
        inflection_point = np.where(years == 2005)[0][0]
        
        # Create increasing trend with more dramatic rise after the inflection point
        reported_sightings = np.concatenate([
            base[:inflection_point],
            base[inflection_point:] * np.linspace(1, 10, len(years) - inflection_point)
        ])
        
        # Add noise
        reported_sightings = reported_sightings * (1 + 0.2 * np.random.randn(len(years)))
        reported_sightings = np.round(reported_sightings).astype(int)
        
        # Create dataframe
        self.data = pd.DataFrame({
            'year': years,
            'reported_sightings': reported_sightings,
            'normalized_sightings': stats.zscore(reported_sightings)
        })

    def calculate_growth_rate(self):
        """Calculate yearly growth rates in UAP sightings."""
        self.data['growth_rate'] = self.data['reported_sightings'].pct_change() * 100
        return self.data[['year', 'growth_rate']].dropna()
    
    def find_acceleration_points(self, threshold=50):
        """
        Find years with significant acceleration in UAP sightings.
        
        Args:
            threshold: Percentage growth rate threshold to consider significant
            
        Returns:
            DataFrame with years and growth rates where growth exceeded threshold
        """
        growth_data = self.calculate_growth_rate()
        return growth_data[growth_data['growth_rate'] > threshold]
    
    def calculate_deviation_from_tstar(self, t_star):
        """
        Calculate deviation scores based on distance from a hypothetical t* year.
        
        Args:
            t_star: The hypothetical "singularity" year to measure deviations from
            
        Returns:
            DataFrame with years and deviation scores
        """
        # Calculate years from t_star
        years_from_tstar = abs(self.data['year'] - t_star)
        
        # Calculate deviation score (inverse relationship to distance from t_star)
        # Normalize sightings by distance from t_star
        deviation = self.data['normalized_sightings'] / (years_from_tstar + 1)
        
        result = pd.DataFrame({
            'year': self.data['year'],
            'deviation_score': deviation
        })
        
        return result
    
    def optimize_t_star(self, start_year=2000, end_year=2050):
        """
        Find the optimal t* that maximizes the pattern signal.
        
        Args:
            start_year: Lower bound for t* search
            end_year: Upper bound for t* search
            
        Returns:
            Optimal t* year
        """
        def objective(t_star):
            deviation_df = self.calculate_deviation_from_tstar(t_star[0])
            # Our goal is to maximize the sum of deviations
            return -np.sum(deviation_df['deviation_score'])
        
        # Optimize to find t_star
        result = minimize(objective, x0=[2030], bounds=[(start_year, end_year)])
        
        if result.success:
            self.optimized_t_star = float(result.x[0])
            return self.optimized_t_star
        else:
            print("Optimization failed to converge")
            return None
    
    def plot_sightings_trend(self):
        """Plot the historical trend of UAP sightings."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['year'], self.data['reported_sightings'], marker='o')
        plt.title('UAP Reported Sightings by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Reported Sightings')
        plt.grid(True, alpha=0.3)
        
        # Add vertical line at t* if it exists
        if self.optimized_t_star is not None:
            plt.axvline(x=self.optimized_t_star, color='r', linestyle='--', 
                      label=f'Estimated t* = {self.optimized_t_star:.1f}')
            plt.legend()
            
        plt.tight_layout()
        return plt
    
    def plot_deviation_profile(self, t_star=None):
        """
        Plot the deviation profile around a given t* year.
        
        Args:
            t_star: The t* year to analyze (uses optimized t* if None)
        """
        if t_star is None:
            if self.optimized_t_star is None:
                self.optimize_t_star()
            t_star = self.optimized_t_star
            
        deviation_df = self.calculate_deviation_from_tstar(t_star)
        
        plt.figure(figsize=(12, 6))
        plt.bar(deviation_df['year'], deviation_df['deviation_score'])
        plt.title(f'UAP Sightings Deviation Profile (t* = {t_star:.1f})')
        plt.xlabel('Year')
        plt.ylabel('Deviation Score')
        plt.axvline(x=t_star, color='r', linestyle='--', label=f't* = {t_star:.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
        
    def get_yearly_intensity_series(self):
        """
        Get a yearly intensity series for UAP sightings.
        
        Returns:
            DataFrame with years and intensity values (normalized sightings)
        """
        return self.data[['year', 'normalized_sightings']].rename(
            columns={'normalized_sightings': 'intensity'}) 