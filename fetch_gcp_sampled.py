import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import gzip
import io
import time
import random
import os
import json
import re

def fetch_gcp_data_for_day(year, month, day, base_url="https://noosphere.princeton.edu/cgi-bin/eggdatareq.pl"):
    """Fetch GCP data for a specific day."""
    params = {
        'z': 1,
        'year': year,
        'month': month,
        'day': day,
        'stime': '00:00:00',
        'etime': '23:59:59',
        'gzip': 'Yes',
        'idate': 'Yes'
    }
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            try:
                # Try to decompress as gzipped data
                content = gzip.decompress(response.content)
                data = content.decode('utf-8')
            except:
                # If not gzipped, just use the raw response
                data = response.text
            
            return data
        else:
            print(f"Error: Status code {response.status_code} for {year}-{month:02d}-{day:02d}")
            return None
    except Exception as e:
        print(f"Exception fetching {year}-{month:02d}-{day:02d}: {e}")
        return None

def extract_z_scores_from_data(data):
    """Extract z-scores from raw GCP data."""
    z_scores = []
    
    if not data:
        return z_scores
    
    # Print a sample of the data to debug
    print(f"Data sample (first 200 chars): {data[:200]}")
    
    # Try to parse different formats
    for record in data.split('\n'):
        # Handle data rows (usually start with '13,')
        if record.startswith('13,'):
            parts = record.split(',')
            if len(parts) >= 4:  # Need at least timestamp, date, and one value
                # Skip the first 3 parts (record type, timestamp, date)
                try:
                    values = []
                    for x in parts[3:]:
                        x = x.strip()
                        if x and x.replace('.', '', 1).replace('-', '', 1).isdigit():
                            values.append(float(x))
                    z_scores.extend(values)
                except ValueError as e:
                    print(f"Error parsing values: {e} - Record: {record[:50]}")
                    continue
    
    # If no z_scores found, try an alternative parsing approach
    if not z_scores:
        try:
            # Try to extract any numbers that might be RNG values
            # Find sequences of digits (potentially with decimal points)
            pattern = r'\b\d+\.?\d*\b'
            numbers = re.findall(pattern, data)
            
            # Skip header info - typically first values are metadata
            if len(numbers) > 10:
                z_scores = [float(num) for num in numbers[10:]]
        except Exception as e:
            print(f"Alternative parsing failed: {e}")
    
    print(f"Extracted {len(z_scores)} values")
    return z_scores

def fetch_gcp_yearly_data(start_year, end_year, output_dir, days_per_month=2, delay=0.2):
    """
    Fetch GCP data with sampling and calculate yearly averages.
    
    Args:
        start_year: First year to fetch data for
        end_year: Last year to fetch data for
        output_dir: Directory to save data and checkpoints
        days_per_month: Number of days to sample per month (reduced from 5 to 2)
        delay: Delay between requests in seconds (reduced from 1.0 to 0.2)
    """
    base_url = "https://noosphere.princeton.edu/cgi-bin/eggdatareq.pl"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File to save progress
    checkpoint_file = output_dir / "gcp_fetch_checkpoint.json"
    
    # Try to load checkpoint
    yearly_data = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                yearly_data = checkpoint.get('yearly_data', {})
                last_year = checkpoint.get('last_year')
                last_month = checkpoint.get('last_month')
                last_day = checkpoint.get('last_day')
                
                if last_year and last_month and last_day:
                    print(f"Resuming from checkpoint: {last_year}-{last_month:02d}-{last_day:02d}")
                    
                    # Convert string keys back to integers
                    yearly_data = {int(k): v for k, v in yearly_data.items()}
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            yearly_data = {}
            last_year, last_month, last_day = None, None, None
    else:
        last_year, last_month, last_day = None, None, None
    
    # Initialize or update the dataframe to track our progress
    total_days = 0
    successful_days = 0
    
    for year in range(start_year, end_year + 1):
        # Skip years we've already completed
        if str(year) in yearly_data and yearly_data[year].get('completed', False):
            print(f"Skipping year {year} (already completed)")
            continue
            
        # Initialize year data if not exists
        if year not in yearly_data:
            yearly_data[year] = {'z_scores': [], 'days_processed': 0, 'completed': False}
        
        print(f"Processing year {year}...")
        
        # Process each month
        for month in range(1, 13):
            # Skip months we've already processed
            if last_year and year < int(last_year):
                continue
            if last_year and year == int(last_year) and month < int(last_month):
                continue
                
            # Get days in this month
            try:
                _, days_in_month = calendar.monthrange(year, month)
            except:
                # Fallback to 28 days if calendar module not available
                days_in_month = 28
                
            # Select random days to sample - reduced from 5 to 2 days per month
            days_to_sample = min(days_per_month, days_in_month)
            sample_days = sorted(random.sample(range(1, days_in_month + 1), days_to_sample))
            
            for day in sample_days:
                # Skip days we've already processed
                if last_year and year == int(last_year) and month == int(last_month) and day <= int(last_day):
                    continue
                    
                total_days += 1
                print(f"  Fetching {year}-{month:02d}-{day:02d}...")
                
                # Fetch data for this day
                data = fetch_gcp_data_for_day(year, month, day, base_url)
                
                if data:
                    # Extract z-scores
                    z_scores = extract_z_scores_from_data(data)
                    
                    if z_scores:
                        # Add to yearly data
                        yearly_data[year]['z_scores'].extend(z_scores)
                        yearly_data[year]['days_processed'] += 1
                        successful_days += 1
                        
                        # Save checkpoint
                        checkpoint = {
                            'yearly_data': yearly_data,
                            'last_year': year,
                            'last_month': month,
                            'last_day': day
                        }
                        with open(checkpoint_file, 'w') as f:
                            json.dump(checkpoint, f)
                
                # Reduced delay between requests (0.2 seconds instead of 1.0)
                time.sleep(delay)
        
        # Mark year as completed
        yearly_data[year]['completed'] = True
        
        # Save checkpoint
        checkpoint = {
            'yearly_data': yearly_data,
            'last_year': year,
            'last_month': 12,
            'last_day': 31
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
    
    # Calculate yearly statistics
    yearly_stats = []
    for year, data in yearly_data.items():
        if data['z_scores']:  # Only process years with data
            z_scores = data['z_scores']
            yearly_stats.append({
                'year': year,
                'avg_abs_z': np.mean(np.abs(z_scores)),  # Average absolute z-score
                'std_z': np.std(z_scores),               # Standard deviation
                'count': len(z_scores),                  # Number of data points
                'days': data['days_processed'],          # Days processed
                'min_z': min(z_scores),                  # Minimum z-score
                'max_z': max(z_scores)                   # Maximum z-score
            })
    
    # Convert to DataFrame and return
    print(f"\nProcessed {successful_days}/{total_days} days successfully.")
    return pd.DataFrame(yearly_stats)

def save_gcp_data(df, output_dir):
    """Save the processed GCP data."""
    output_dir = Path(output_dir)
    
    # Save the complete data
    df.to_csv(output_dir / "rng_avg_abs_z_full.csv", index=False)
    
    # Save the simplified version required by the analysis script
    simplified_df = df[['year', 'avg_abs_z']].copy()
    simplified_df.to_csv(output_dir / "rng_avg_abs_z.csv", index=False)
    
    # Print summary statistics
    print("\nYearly Statistics Summary:")
    print(df[['year', 'avg_abs_z', 'std_z', 'count', 'days']].describe())
    
    # Print years with highest average absolute z-scores
    print("\nTop 5 years with highest average absolute z-scores:")
    print(df.nlargest(5, 'avg_abs_z')[['year', 'avg_abs_z', 'days']])

if __name__ == "__main__":
    import argparse
    import calendar
    
    parser = argparse.ArgumentParser(description='Fetch GCP data and calculate yearly averages')
    parser.add_argument('--start-year', type=int, default=2000, help='First year to analyze')
    parser.add_argument('--end-year', type=int, default=datetime.now().year, help='Last year to analyze')
    parser.add_argument('--days-per-month', type=int, default=2, help='Number of days to sample per month (default: 2)')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay between requests in seconds (default: 0.2)')
    parser.add_argument('--output-dir', type=str, default="singularity_window_analyzer/data", 
                        help='Directory to save data and checkpoints')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching GCP data from {args.start_year} to {args.end_year}...")
    print(f"Sampling {args.days_per_month} days per month with {args.delay}s delay between requests")
    
    # Fetch data
    yearly_df = fetch_gcp_yearly_data(
        args.start_year, 
        args.end_year, 
        output_dir,
        days_per_month=args.days_per_month,
        delay=args.delay
    )
    
    if not yearly_df.empty:
        print("\nSaving data...")
        save_gcp_data(yearly_df, output_dir)
        print("\nData processing complete!")
    else:
        print("\nNo data was processed successfully.") 