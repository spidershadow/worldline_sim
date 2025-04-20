# Worldline Simulator

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
  - `misc/` - Miscellaneous output files

## Usage

Example usage for main scripts:

```bash
# Run T* consistency finder
python scripts/find_consistent_tstar.py --tstar-range 2030 2100 --runs 5 --output data/consistent_tstar_results

# Create animations
python scripts/animate_tstar_consistency.py --tstar-range 2045 2055 --simulations 50
```
