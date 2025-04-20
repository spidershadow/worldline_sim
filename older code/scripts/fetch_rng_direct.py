"""Fetches RNG data directly from the Global Consciousness Project (GCP) website.

This script retrieves raw RNG data from the GCP database using their data API 
and processes it to create a dataset that can be used by the EnhancedRngPattern.
It downloads daily data and calculates z-scores and statistical metrics
following GCP methodology.

Example URL format:
https://noosphere.princeton.edu/cgi-bin/eggdatareq.pl?z=1&year=2025&month=4&day=20&stime=00%3A00%3A00&etime=23%3A59%3A59&gzip=Yes&idate=Yes
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import gzip
import io
import os
from pathlib import Path
import time
import argparse

# Directory for storing data
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Output file paths
RNG_OUTPUT_FILE = DATA_DIR / "rng_avg_abs_z.csv"
RNG_DAILY_FILE = DATA_DIR / "rng_daily_data.csv"


def fetch_day_data(year, month, day):
    """Fetch RNG data for a specific day from the GCP website."""
    # Format the URL with the specified date
    url = (
        f"https://noosphere.princeton.edu/cgi-bin/eggdatareq.pl?"
        f"z=1&year={year}&month={month}&day={day}"
        f"&stime=00%3A00%3A00&etime=23%3A59%3A59&gzip=Yes&idate=Yes"
    )
    
    try:
        # Make the request with timeout
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Decompress gzip data
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            data = f.read().decode('utf-8')
        
        return data
    except Exception as e:
        print(f"Error fetching data for {year}-{month:02d}-{day:02d}: {e}")
        return None


def process_day_data(data_str, year, month, day):
    """Process raw data string for a day and calculate statistics."""
    if not data_str:
        return None
    
    lines = data_str.strip().split('\n')
    
    # Initialize data structures
    hourly_data = {}
    
    for line in lines:
        parts = line.split()
        if not parts or len(parts) < 4:
            continue
        
        try:
            # Extract timestamp and values
            timestamp = parts[0]
            if not timestamp.replace('.', '', 1).isdigit():
                continue
                
            # Extract RNG values
            values = [float(v) for v in parts[3:] if v and v.replace('.', '', 1).replace('-', '', 1).isdigit()]
            if not values:
                continue
                
            # Calculate statistics for this timestamp
            mean = np.mean(values)
            std_dev = np.std(values)
            z_score = (mean - 0.5) / (std_dev / np.sqrt(len(values))) if std_dev > 0 else 0
            abs_z = abs(z_score)
            
            # Extract hour from timestamp
            hour = int(float(timestamp)) % 24
            
            # Store in hourly data
            if hour not in hourly_data:
                hourly_data[hour] = []
            hourly_data[hour].append(abs_z)
                
        except Exception as e:
            continue
    
    # Calculate daily averages
    results = {}
    if hourly_data:
        all_abs_z = [z for hour_values in hourly_data.values() for z in hour_values]
        if all_abs_z:
            results = {
                'date': f"{year}-{month:02d}-{day:02d}",
                'avg_abs_z': np.mean(all_abs_z),
                'cum_var': np.var(all_abs_z),
                'n_samples': len(all_abs_z)
            }
    
    return results


def fetch_year_data(year, start_month=1, start_day=1, end_month=12, end_day=31, delay=1):
    """Fetch data for a specific year with optional date range and delay between requests."""
    print(f"Fetching data for year {year}...")
    
    start_date = datetime(year, start_month, start_day)
    end_date = datetime(year, end_month, end_day)
    
    # Adjust end date if in the future
    today = datetime.now()
    if end_date > today:
        end_date = today
        print(f"Adjusted end date to today: {end_date.strftime('%Y-%m-%d')}")
    
    # Initialize daily data list
    daily_data = []
    
    # Iterate through each day in the range
    current_date = start_date
    while current_date <= end_date:
        print(f"Fetching {current_date.strftime('%Y-%m-%d')}...", end="\r")
        
        # Fetch and process data for this day
        raw_data = fetch_day_data(
            current_date.year, 
            current_date.month, 
            current_date.day
        )
        
        if raw_data:
            day_stats = process_day_data(
                raw_data, 
                current_date.year, 
                current_date.month, 
                current_date.day
            )
            
            if day_stats:
                daily_data.append(day_stats)
        
        # Move to next day
        current_date += timedelta(days=1)
        
        # Add delay to avoid overloading the server
        if delay > 0:
            time.sleep(delay)
    
    print(f"\nCompleted fetching data for {year}: {len(daily_data)} days processed")
    return daily_data


def combine_and_save_data(all_daily_data):
    """Combine daily data and save to CSV files."""
    if not all_daily_data:
        print("No data to save")
        return
    
    # Create a DataFrame from the daily data
    daily_df = pd.DataFrame(all_daily_data)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values('date')
    
    # Save the daily data
    daily_df.to_csv(RNG_DAILY_FILE, index=False)
    print(f"Saved daily data to {RNG_DAILY_FILE}")
    
    # Calculate yearly averages
    yearly_df = daily_df.copy()
    yearly_df['year'] = yearly_df['date'].dt.year
    yearly_avg = yearly_df.groupby('year')['avg_abs_z'].mean().reset_index()
    yearly_avg.columns = ['year', 'value']
    
    # Save the yearly data
    yearly_avg.to_csv(RNG_OUTPUT_FILE, index=False)
    print(f"Saved yearly averages to {RNG_OUTPUT_FILE}")
    
    return daily_df, yearly_avg


def main():
    """Main function to parse arguments and execute the script."""
    parser = argparse.ArgumentParser(description="Fetch RNG data from the Global Consciousness Project")
    parser.add_argument("--start-year", type=int, default=1998, help="Year to start fetching data from")
    parser.add_argument("--end-year", type=int, default=datetime.now().year, help="Year to end fetching data at")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument("--update", action="store_true", help="Update existing data instead of fetching all")
    
    args = parser.parse_args()
    
    # If update flag is set and the daily file exists, only fetch new data
    start_year = args.start_year
    if args.update and os.path.exists(RNG_DAILY_FILE):
        try:
            existing_df = pd.read_csv(RNG_DAILY_FILE)
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            latest_date = existing_df['date'].max()
            
            # Start from the day after the latest date
            start_year = latest_date.year
            start_month = latest_date.month
            start_day = latest_date.day + 1
            
            # If we're already at the end of the month, move to next month
            if start_day > 28:  # Simple approach to handle month transitions
                start_month += 1
                start_day = 1
            
            # If we're already at the end of the year, move to next year
            if start_month > 12:
                start_year += 1
                start_month = 1
                
            print(f"Updating data from {start_year}-{start_month:02d}-{start_day:02d}")
            
            # Get existing data
            all_daily_data = existing_df.to_dict('records')
            
        except Exception as e:
            print(f"Error reading existing data: {e}")
            print("Falling back to fetching all data")
            all_daily_data = []
    else:
        all_daily_data = []
    
    # Fetch data year by year
    for year in range(start_year, args.end_year + 1):
        if year == start_year and 'start_month' in locals():
            # For the first year, use the calculated start date
            year_data = fetch_year_data(year, start_month, start_day, delay=args.delay)
        else:
            # For other years, fetch the entire year
            year_data = fetch_year_data(year, delay=args.delay)
            
        all_daily_data.extend(year_data)
    
    # Combine and save the data
    combine_and_save_data(all_daily_data)


if __name__ == "__main__":
    main() 