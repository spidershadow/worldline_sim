#!/usr/bin/env python3
"""
Analyze GCP data for retrocausal effects.

This script uses the enhanced RNG pattern class to analyze Global Consciousness Project
data for potential retrocausal effects, based on the cumulative deviation metrics
across years. It examines potential Singularity years (T*) and evaluates how different
years could show evidence of retrocausal influence.

Example
-------
python scripts/analyze_gcp_retrocausality.py --start-year 2000 --end-year 2023 --window 5
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

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import after adding project root to path
from patterns.enhanced_rng import EnhancedRngPattern, GCP_SIGNIFICANCE_MILESTONES

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output" / "gcp_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_year_range(start_year, end_year, window_years=5):
    """Analyze a range of years for potential retrocausal effects.
    
    Args:
        start_year: First year to analyze
        end_year: Last year to analyze
        window_years: Number of years to look before and after each target
        
    Returns:
        DataFrame with analysis results
    """
    print(f"Analyzing years {start_year}-{end_year} (±{window_years} year window)")
    
    # Initialize EnhancedRngPattern with balanced window
    pattern = EnhancedRngPattern(direction="both")
    
    # Analyze each year in the range
    results = []
    for year in range(start_year, end_year + 1):
        print(f"Analyzing year {year}...")
        
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
    df = pd.DataFrame(results)
    
    return df


def create_visualizations(results_df, output_dir):
    """Create visualizations from the analysis results.
    
    Args:
        results_df: DataFrame with analysis results
        output_dir: Directory to save visualizations
    """
    # Set up style
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Plot retrocausal scores over time
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['year'], results_df['retrocausal_score'], 'o-', linewidth=2, markersize=8)
    
    # Add line for evidence thresholds
    plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Strong Evidence (0.7)')
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate Evidence (0.5)')
    plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Weak Evidence (0.3)')
    
    # Highlight years with highest scores
    top_years = results_df.nlargest(3, 'retrocausal_score')
    for _, row in top_years.iterrows():
        plt.annotate(f"{int(row['year'])}", 
                    xy=(row['year'], row['retrocausal_score']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.title('Retrocausal Evidence Score by Year', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Retrocausal Score', fontsize=14)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "retrocausal_scores.png", dpi=150)
    plt.close()
    
    # 2. Plot cumulative Z-scores with retrocausal evidence
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot cumulative Z-score
    ax1.plot(results_df['year'], results_df['cum_z_score'], 'b-', linewidth=2, label='GCP Cumulative Z-Score')
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Cumulative Z-Score', color='b', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot retrocausal score on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(results_df['year'], results_df['retrocausal_score'], 'r--', linewidth=2, label='Retrocausal Score')
    ax2.set_ylabel('Retrocausal Score', color='r', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, 1.0)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('GCP Cumulative Z-Score vs. Retrocausal Evidence', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "zscore_vs_retrocausal.png", dpi=150)
    plt.close()
    
    # 3. Create heatmap of components
    components_df = results_df[['year', 'symmetry', 'acceleration', 'retrocausal_score']]
    
    # Pivot for heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(components_df.set_index('year').T, 
                cmap='viridis', 
                annot=True, 
                fmt=".2f",
                linewidths=0.5)
    plt.title('Components of Retrocausal Evidence by Year', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "retrocausal_components.png", dpi=150)
    plt.close()
    
    # 4. Create bar chart of evidence categories
    evidence_counts = results_df['evidence'].value_counts()
    plt.figure(figsize=(10, 6))
    colors = {'Strong': 'green', 'Moderate': 'orange', 'Weak': 'yellow', 'None': 'red'}
    evidence_counts.plot(kind='bar', color=[colors.get(x, 'gray') for x in evidence_counts.index])
    plt.title('Distribution of Retrocausal Evidence Categories', fontsize=16)
    plt.xlabel('Evidence Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "evidence_categories.png", dpi=150)
    plt.close()


def generate_report(results_df, output_dir, args):
    """Generate a summary report of the analysis.
    
    Args:
        results_df: DataFrame with analysis results
        output_dir: Directory to save the report
    """
    # Create HTML report
    html_output = output_dir / "gcp_retrocausal_report.html"
    
    # Find top candidates
    top_candidates = results_df.nlargest(5, 'retrocausal_score')
    
    # Calculate statistics
    evidence_counts = results_df['evidence'].value_counts()
    total_years = len(results_df)
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GCP Retrocausal Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .highlight {{ background-color: #ffffcc; }}
            .container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
            .image-container {{ width: 48%; margin-bottom: 20px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Global Consciousness Project (GCP) Retrocausal Analysis</h1>
        <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Analysis Period:</strong> {args.start_year} - {args.end_year}</p>
        <p><strong>Window Size:</strong> ±{args.window} years</p>
        
        <h2>Executive Summary</h2>
        <p>This analysis examined {total_years} years for evidence of retrocausal effects in GCP data. 
        The retrocausal score combines symmetry (correlation between pre- and post-patterns) and 
        acceleration (change in deviation rates) around each candidate year.</p>
        
        <p>Evidence levels found:</p>
        <ul>
    """
    
    # Add evidence counts
    for category, count in evidence_counts.items():
        percentage = (count / total_years) * 100
        html_content += f"<li><strong>{category}:</strong> {count} years ({percentage:.1f}%)</li>\n"
    
    html_content += """
        </ul>
        
        <h2>Top Candidate Years</h2>
        <p>The following years show the strongest evidence for potential retrocausal effects:</p>
        <table>
            <tr>
                <th>Year</th>
                <th>Retrocausal Score</th>
                <th>Evidence Level</th>
                <th>Symmetry</th>
                <th>Acceleration</th>
                <th>Cum. Z-Score</th>
            </tr>
    """
    
    # Add top candidate rows
    for _, row in top_candidates.iterrows():
        html_content += f"""
            <tr class="highlight">
                <td>{int(row['year'])}</td>
                <td>{row['retrocausal_score']:.3f}</td>
                <td>{row['evidence']}</td>
                <td>{row['symmetry']:.3f}</td>
                <td>{row['acceleration']:.3f}</td>
                <td>{row['cum_z_score']:.3f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Full Results</h2>
        <table>
            <tr>
                <th>Year</th>
                <th>Retrocausal Score</th>
                <th>Evidence Level</th>
                <th>Symmetry</th>
                <th>Acceleration</th>
                <th>Before Slope</th>
                <th>After Slope</th>
                <th>Cum. Z-Score</th>
            </tr>
    """
    
    # Add all results
    for _, row in results_df.sort_values('year').iterrows():
        html_content += f"""
            <tr>
                <td>{int(row['year'])}</td>
                <td>{row['retrocausal_score']:.3f}</td>
                <td>{row['evidence']}</td>
                <td>{row['symmetry']:.3f}</td>
                <td>{row['acceleration']:.3f}</td>
                <td>{row['before_slope']:.3f}</td>
                <td>{row['after_slope']:.3f}</td>
                <td>{row['cum_z_score']:.3f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Visualizations</h2>
        <div class="container">
            <div class="image-container">
                <h3>Retrocausal Evidence Scores</h3>
                <img src="retrocausal_scores.png" alt="Retrocausal Scores">
            </div>
            <div class="image-container">
                <h3>Z-Score vs. Retrocausal Evidence</h3>
                <img src="zscore_vs_retrocausal.png" alt="Z-Score vs Retrocausal">
            </div>
            <div class="image-container">
                <h3>Components of Retrocausal Evidence</h3>
                <img src="retrocausal_components.png" alt="Retrocausal Components">
            </div>
            <div class="image-container">
                <h3>Evidence Categories Distribution</h3>
                <img src="evidence_categories.png" alt="Evidence Categories">
            </div>
        </div>
        
        <h2>Methodology</h2>
        <p>This analysis uses the EnhancedRngPattern class to evaluate Global Consciousness Project data 
        for potential retrocausal effects. For each candidate year, we calculate:</p>
        <ul>
            <li><strong>Symmetry:</strong> Correlation between normalized deviation patterns before and after the year</li>
            <li><strong>Acceleration:</strong> Change in the rate of deviation growth around the candidate year</li>
            <li><strong>Retrocausal Score:</strong> Weighted combination of symmetry and acceleration (0.6 * |symmetry| + 0.4 * |acceleration|)</li>
        </ul>
        
        <p>The evidence levels are classified as:</p>
        <ul>
            <li><strong>Strong:</strong> Score > 0.7</li>
            <li><strong>Moderate:</strong> Score > 0.5</li>
            <li><strong>Weak:</strong> Score > 0.3</li>
            <li><strong>None:</strong> Score ≤ 0.3</li>
        </ul>
        
        <h2>References</h2>
        <p>Global Consciousness Project: <a href="https://noosphere.princeton.edu/">https://noosphere.princeton.edu/</a></p>
        <p>Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """
    
    # Write HTML report
    with open(html_output, 'w') as f:
        f.write(html_content)
    
    print(f"Report generated: {html_output}")
    
    # Also create a CSV export
    csv_output = output_dir / "gcp_retrocausal_results.csv"
    results_df.to_csv(csv_output, index=False)
    print(f"Results exported to CSV: {csv_output}")


def save_milestones_table(output_dir):
    """Save the GCP milestones as a reference table.
    
    Args:
        output_dir: Directory to save the table
    """
    milestones_df = pd.DataFrame([
        {'year': year, 'events': data[0], 'z_score': data[1], 'p_value': data[2]}
        for year, data in GCP_SIGNIFICANCE_MILESTONES.items()
    ])
    
    # Sort by year
    milestones_df = milestones_df.sort_values('year')
    
    # Save as CSV
    csv_output = output_dir / "gcp_milestones.csv"
    milestones_df.to_csv(csv_output, index=False)
    print(f"GCP milestones saved to: {csv_output}")
    
    # Create a visualization
    plt.figure(figsize=(10, 6))
    plt.plot(milestones_df['year'], milestones_df['z_score'], 'o-', linewidth=2, markersize=8)
    
    # Add labels
    for _, row in milestones_df.iterrows():
        plt.annotate(f"{int(row['year'])}\nZ={row['z_score']:.1f}", 
                    xy=(row['year'], row['z_score']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom')
    
    plt.title('GCP Cumulative Z-Score Milestones', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Cumulative Z-Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "gcp_milestones.png", dpi=150)
    plt.close()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze GCP data for retrocausal effects")
    parser.add_argument("--start-year", type=int, default=2000, help="First year to analyze")
    parser.add_argument("--end-year", type=int, default=2020, help="Last year to analyze")
    parser.add_argument("--window", type=int, default=5, help="Window size (years before/after) for analysis")
    return parser.parse_args()


def main():
    """Main function to run the analysis."""
    args = parse_args()
    
    # Validate input
    if args.start_year < 1998:
        print("Warning: GCP data starts in 1998. Setting start_year to 1998.")
        args.start_year = 1998
    
    current_year = datetime.now().year
    if args.end_year > current_year - args.window:
        end_year = current_year - args.window
        print(f"Warning: Need {args.window} years after end_year. Setting end_year to {end_year}.")
        args.end_year = end_year
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save GCP milestones
    save_milestones_table(output_dir)
    
    # Run analysis
    results_df = analyze_year_range(args.start_year, args.end_year, args.window)
    
    # Create visualizations
    create_visualizations(results_df, output_dir)
    
    # Generate report
    generate_report(results_df, output_dir, args)
    
    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 