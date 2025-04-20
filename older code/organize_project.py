#!/usr/bin/env python3
"""
Organize the worldline simulator project files into a cleaner directory structure.
This script creates appropriate folders and moves files to their logical locations.

Usage:
    python organize_project.py

The script will create the following structure:
- scripts/ - Animation and simulation scripts
- data/ - CSV and data files
- output/
  - images/ - PNG visualization files 
  - animations/ - MP4 animation files
  - reports/ - Text reports
"""

import os
import shutil
from pathlib import Path

# Define the folders to create
FOLDERS = [
    "scripts",
    "data",
    "output",
    "output/images",
    "output/animations", 
    "output/reports"
]

# Files to move to scripts/
SCRIPT_FILES = [
    "animate_tstar_consistency.py",
    "enhanced_animation.py",
    "find_consistent_tstar.py",
    "animate_tstar_variation.py",
    "animate_tstar_convergence.py",
    "animate_retrocausality.py"
]

# Files to move to data/
DATA_FILES = [
    "consistent_tstar_raw_results.csv",
    "consistent_tstar_results.csv",
    "forecast_rng_tech_uap_events_20runs.csv"
]

# Files to move to output/reports/
REPORT_FILES = [
    "consistent_tstar_report.txt"
]

# Files to move to output/animations/
ANIMATION_FILES = [
    "tstar_pattern_comparison.mp4",
    "tstar_convergence_demo.mp4",
    "tstar_variation_2080_2100.mp4",
    "tstar_variation_2030_2040.mp4",
    "tstar_variation_2090_2100.mp4",
    "tstar_convergence_2090_2100.mp4",
    "retrocausal_animation_t2100.mp4"
]

# Files to move to output/images/
def get_image_files():
    """Get all PNG image files that should be moved to output/images/"""
    image_files = []
    for file in os.listdir():
        if file.endswith(".png") and os.path.isfile(file):
            image_files.append(file)
    return image_files

def create_folders():
    """Create the folder structure if it doesn't exist"""
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}/")

def move_files():
    """Move files to their appropriate folders"""
    # Move script files
    for file in SCRIPT_FILES:
        if os.path.exists(file):
            shutil.move(file, os.path.join("scripts", file))
            print(f"Moved {file} to scripts/")
    
    # Move data files
    for file in DATA_FILES:
        if os.path.exists(file):
            shutil.move(file, os.path.join("data", file))
            print(f"Moved {file} to data/")
    
    # Move report files
    for file in REPORT_FILES:
        if os.path.exists(file):
            shutil.move(file, os.path.join("output", "reports", file))
            print(f"Moved {file} to output/reports/")
    
    # Move animation files
    for file in ANIMATION_FILES:
        if os.path.exists(file):
            shutil.move(file, os.path.join("output", "animations", file))
            print(f"Moved {file} to output/animations/")
    
    # Move image files
    for file in get_image_files():
        if os.path.exists(file):
            shutil.move(file, os.path.join("output", "images", file))
            print(f"Moved {file} to output/images/")

def create_readme():
    """Create a README.md file explaining the project structure"""
    readme_content = """# Worldline Simulator

A simulation framework for exploring T* (Singularity) probability distributions across multiple parameters.

## Project Structure

- `worldline_sim/` - Core simulation framework and modules
  - `patterns/` - Pattern implementations (RNG, tech, UAP)
  
- `scripts/` - Animation and simulation scripts
  - `animate_tstar_consistency.py` - Visualizes T* consistency across simulations
  - `enhanced_animation.py` - Enhanced animation framework
  - `find_consistent_tstar.py` - Core T* analysis simulation
  - `animate_tstar_variation.py` - T* variation visualization
  - `animate_tstar_convergence.py` - T* convergence visualization
  - `animate_retrocausality.py` - Retrocausality animations
  
- `data/` - Input and output data files
  - CSV files with simulation results
  
- `output/` - Generated outputs
  - `images/` - PNG visualizations
  - `animations/` - MP4 animations
  - `reports/` - Text reports and analysis

## Usage

Example usage for main scripts:

```bash
# Run T* consistency finder
python scripts/find_consistent_tstar.py --tstar-range 2030 2100 --runs 5 --output data/consistent_tstar_results

# Create animations
python scripts/animate_tstar_consistency.py --tstar-range 2045 2055 --simulations 50
```
"""
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("Created README.md with project structure documentation")

def update_import_paths():
    """Update import paths in scripts to reflect new directory structure"""
    # This would typically scan through script files and update import paths
    # For simplicity, we're just printing a warning that this might be needed
    print("\nWARNING: You may need to update import paths in scripts to reflect the new directory structure.")
    print("For example, you might need to add the project root to sys.path in scripts, or use relative imports.")

def main():
    print("Organizing project files into a cleaner structure...")
    create_folders()
    move_files()
    create_readme()
    update_import_paths()
    print("\nProject organization complete!")
    print("Directory structure created with scripts/, data/, and output/ folders.")
    print("You may need to adjust import paths in Python files if scripts can't find modules.")

if __name__ == "__main__":
    main() 