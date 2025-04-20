# Singularity Window Analyzer

A Python package for analyzing potential "singularity windows" using multiple pattern sources, including GCP (Global Consciousness Project) data and UAP (Unidentified Aerial Phenomena) sighting frequencies.

## Overview

This package implements a multi-resolution approach to detect and analyze patterns that might indicate a technological singularity or other significant transition timeframe. It uses independent data sources with different temporal resolutions to triangulate a potential "t*" (pivot year).

## Components

- **Enhanced RNG Pattern Analysis**: Examines retrocausal effects in RNG data from the Global Consciousness Project
- **UAP Yearly Pattern Analysis**: Analyzes temporal patterns in UAP sighting data
- **Joint Optimization**: Combines multiple pattern sources to find a common t* that maximizes the joint signal

## Usage

```bash
# Basic usage with all patterns
python scripts/analyze_singularity_window.py --start-year 2000 --end-year 2050 --window 10 --patterns all

# Use only UAP pattern with custom t* range
python scripts/analyze_singularity_window.py --patterns uap --t-star-range 2030 2050

# Use only GCP pattern with custom window
python scripts/analyze_singularity_window.py --patterns gcp --window 5
```

## Data Sources

- GCP data from the [Global Consciousness Project](https://noosphere.princeton.edu/)
- UAP sighting data from historical records

## Output

The analysis generates:
- Interactive visualizations
- HTML report with findings
- CSV exports of results
- Joint pattern analysis with optimized t* estimate
