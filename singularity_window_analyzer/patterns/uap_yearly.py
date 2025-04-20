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
                 window_years=10,
                 default_t_star=2033,
                 default_k=0.23):
        """
        Initialize UAP pattern detector.
        
        Args:
            early_data_path: Path to CSV with early UAP sighting data
            recent_data_path: Path to CSV with recent UAP sighting data
            window_years: Size of the analysis window in years
            default_t_star: Default t* value (pivot year) from research
            default_k: Default logistic growth constant (yr^-1) from research
        """
        self.early_data_path = early_data_path
        self.recent_data_path = recent_data_path
        self.window_years = window_years
        self.data = None
        self.optimized_t_star = None
        self.default_t_star = default_t_star
        self.default_k = default_k
        self.logistic_params = None
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
            
            # Approximate per-billion people (normalize by world population)
            # This is an approximation since we don't have exact population data
            self.data['per_billion'] = self.data['reported_sightings'] / 7.0  # Rough approximation
            
        except Exception as e:
            print(f"Error loading UAP data: {e}")
            self._use_fallback_data()
    
    def _use_fallback_data(self):
        """Generate simplified fallback data if real data is unavailable."""
        years = np.arange(1947, 2024)
        
        # Generate synthetic data using logistic function with parameters from research
        # t* = 2033, k = 0.23, earliest significant reports in 1947
        reported_sightings = self.logistic(years, self.default_t_star, self.default_k, 5000, 100)
        
        # Add noise to make it more realistic
        reported_sightings = reported_sightings * (1 + 0.2 * np.random.randn(len(years)))
        reported_sightings = np.round(reported_sightings).astype(int)
        
        # Create dataframe
        self.data = pd.DataFrame({
            'year': years,
            'reported_sightings': reported_sightings,
            'normalized_sightings': stats.zscore(reported_sightings),
            'per_billion': reported_sightings / 7.0  # Rough approximation per billion people
        })

    @staticmethod
    def logistic(x, t, k, A, B=0):
        """
        Logistic function for modeling UAP reports over time.
        
        Args:
            x: Input year values
            t: t* parameter (inflection point / pivot year)
            k: Growth rate constant (steepness parameter)
            A: Amplitude (upper asymptote - lower asymptote)
            B: Baseline value (lower asymptote)
            
        Returns:
            Logistic curve values
        """
        return A / (1 + np.exp(-k * (x - t))) + B
    
    def fit_logistic(self, start_year=1947, end_year=2023, 
                    t_min=2025, t_max=2045,
                    k_min=0.1, k_max=0.5):
        """
        Fit a logistic curve to the UAP sighting data.
        
        Args:
            start_year: First year to include in the fit
            end_year: Last year to include in the fit
            t_min: Minimum allowed t* value
            t_max: Maximum allowed t* value
            k_min: Minimum allowed k value
            k_max: Maximum allowed k value
            
        Returns:
            Dict of fitted parameters
        """
        # Filter data to the specified range
        fit_data = self.data[(self.data['year'] >= start_year) & 
                            (self.data['year'] <= end_year)].copy()
        
        if fit_data.empty:
            print("No data in specified range for fitting")
            return {'t': self.default_t_star, 'k': self.default_k, 'A': 1.1, 'B': 0.1}
        
        # If per_billion column doesn't exist, use normalized_sightings
        if 'per_billion' not in fit_data.columns:
            fit_data['per_billion'] = fit_data['normalized_sightings']
        
        # Get x and y data
        x_data = fit_data['year'].values
        y_data = fit_data['per_billion'].values
        
        # Define error function to minimize
        def error_func(params):
            t, k, A, B = params
            y_pred = self.logistic(x_data, t, k, A, B)
            return np.sum((y_pred - y_data)**2)
        
        # Initial guess and bounds
        initial_guess = [self.default_t_star, self.default_k, np.max(y_data) - np.min(y_data), np.min(y_data)]
        bounds = [(t_min, t_max), (k_min, k_max), (0.1, 10.0), (0, 1.0)]
        
        # Minimize error function
        result = minimize(error_func, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            params = {'t': result.x[0], 'k': result.x[1], 'A': result.x[2], 'B': result.x[3]}
            self.logistic_params = params
            self.optimized_t_star = params['t']
            return params
        else:
            print("Logistic fit optimization failed to converge")
            default_params = {'t': self.default_t_star, 'k': self.default_k, 'A': 1.1, 'B': 0.1}
            self.logistic_params = default_params
            self.optimized_t_star = self.default_t_star
            return default_params
    
    def calculate_retrocausal_reach(self, year, earliest_significant_year=1947):
        """
        Calculate the maximum retrocausal reach for a given year.
        
        Based on the formula: R(t) proportional to exp(k*t)
        Where R is the "reach" in years and k is the logistic growth parameter.
        
        Args:
            year: The year to calculate reach for
            earliest_significant_year: The earliest year with significant UAP reports
            
        Returns:
            Maximum past-reach in years and the earliest visitable year
        """
        if self.logistic_params is None:
            self.fit_logistic()
            
        # Extract parameters
        t_star = self.logistic_params['t']
        k = self.logistic_params['k']
        
        # Calculate baseline reach at t*
        baseline_reach = t_star - earliest_significant_year
        
        # Calculate relative reach based on k parameter
        years_from_tstar = year - t_star
        relative_reach = np.exp(k * years_from_tstar)
        
        # Scale the reach
        max_reach = baseline_reach * relative_reach
        earliest_visitable = year - max_reach
        
        return {
            'year': year,
            'max_reach_years': max_reach,
            'earliest_visitable': earliest_visitable,
            't_star': t_star,
            'k': k
        }
        
    def get_retrocausal_timeline(self, start_year=2000, end_year=2050):
        """
        Generate a timeline of retrocausal reach capabilities.
        
        Args:
            start_year: First year to include
            end_year: Last year to include
            
        Returns:
            DataFrame with years and corresponding reach data
        """
        if self.logistic_params is None:
            self.fit_logistic()
            
        years = range(start_year, end_year + 1)
        
        reach_data = []
        for year in years:
            reach = self.calculate_retrocausal_reach(year)
            reach_data.append({
                'year': year,
                'max_reach_years': reach['max_reach_years'],
                'earliest_visitable': reach['earliest_visitable']
            })
            
        return pd.DataFrame(reach_data)

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
    
    def calculate_deviation_from_tstar(self, t_star=None):
        """
        Calculate deviation scores based on distance from a hypothetical t* year.
        
        Args:
            t_star: The hypothetical "singularity" year to measure deviations from
            
        Returns:
            DataFrame with years and deviation scores
        """
        if t_star is None:
            if self.optimized_t_star is None:
                if self.logistic_params is None:
                    self.fit_logistic()
                t_star = self.optimized_t_star
            else:
                t_star = self.optimized_t_star
                
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
    
    def optimize_t_star(self, start_year=2025, end_year=2045):
        """
        Find the optimal t* that maximizes the pattern signal.
        
        Args:
            start_year: Lower bound for t* search
            end_year: Upper bound for t* search
            
        Returns:
            Optimal t* year
        """
        # First try fitting the logistic model
        params = self.fit_logistic(t_min=start_year, t_max=end_year)
        t_star_logistic = params['t']
        
        # As a backup, also try the deviation approach
        def objective(t_star):
            deviation_df = self.calculate_deviation_from_tstar(t_star[0])
            # Our goal is to maximize the sum of deviations
            return -np.sum(deviation_df['deviation_score'])
        
        # Optimize to find t_star using deviation method
        result = minimize(objective, x0=[t_star_logistic], 
                         bounds=[(start_year, end_year)])
        
        if result.success:
            t_star_deviation = float(result.x[0])
            # Average the two approaches
            self.optimized_t_star = (t_star_logistic + t_star_deviation) / 2
            return self.optimized_t_star
        else:
            print("Optimization failed to converge, using logistic t*")
            self.optimized_t_star = t_star_logistic
            return self.optimized_t_star
    
    def plot_sightings_trend(self, include_logistic=True):
        """
        Plot the historical trend of UAP sightings.
        
        Args:
            include_logistic: Whether to overlay the fitted logistic curve
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['year'], self.data['reported_sightings'], marker='o')
        
        # Add fitted logistic curve if requested
        if include_logistic:
            if self.logistic_params is None:
                self.fit_logistic()
                
            # Create smooth curve for plotting
            years = np.linspace(self.data['year'].min(), self.data['year'].max() + 20, 300)
            logistic_curve = self.logistic(
                years,
                self.logistic_params['t'],
                self.logistic_params['k'],
                self.logistic_params['A'] * 7.0,  # Scale back to match raw counts
                self.logistic_params['B'] * 7.0
            )
            plt.plot(years, logistic_curve, 'r-', linewidth=2, 
                   label=f'Logistic Fit (t* = {self.logistic_params["t"]:.1f}, k = {self.logistic_params["k"]:.2f})')
        
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
    
    def plot_retrocausal_reach(self, start_year=2000, end_year=2050):
        """
        Plot the retrocausal reach timeline.
        
        Args:
            start_year: First year to include
            end_year: Last year to include
            
        Returns:
            Matplotlib figure
        """
        reach_df = self.get_retrocausal_timeline(start_year, end_year)
        
        plt.figure(figsize=(12, 6))
        
        # Plot max reach
        plt.plot(reach_df['year'], reach_df['max_reach_years'], 'b-', 
               linewidth=2, label='Maximum Retrocausal Reach (years)')
        
        # Plot earliest visitable year
        plt.plot(reach_df['year'], reach_df['earliest_visitable'], 'g-', 
               linewidth=2, label='Earliest Visitable Year')
        
        # Add vertical line at t*
        if self.logistic_params is not None:
            t_star = self.logistic_params['t']
            plt.axvline(x=t_star, color='r', linestyle='--', 
                      label=f'Singularity t* = {t_star:.1f}')
        
        # Add current year line
        current_year = 2024  # Could use datetime.now().year
        plt.axvline(x=current_year, color='k', linestyle=':', 
                  label=f'Current Year ({current_year})')
        
        plt.title('Retrocausal Reach Timeline')
        plt.xlabel('Year')
        plt.ylabel('Years / Calendar Year')
        plt.grid(True, alpha=0.3)
        plt.legend()
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
    
    def _load_gcp_yearly_intensity(self):
        """
        Helper method to load GCP yearly intensity data for joint visualization.
        This uses the EnhancedRngPattern class if available, otherwise returns mock data.
        
        Returns:
            DataFrame with years and GCP intensity values
        """
        try:
            from patterns.enhanced_rng import EnhancedRngPattern
            pattern = EnhancedRngPattern()
            return pattern.get_yearly_intensity_series()
        except Exception as e:
            print(f"Error loading GCP data: {e}")
            # Return mock data matching our year range
            years = self.data['year'].unique()
            mock_intensity = np.random.normal(0, 1, size=len(years))
            return pd.DataFrame({
                'year': years,
                'intensity': mock_intensity
            }) 