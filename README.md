# Lifetime Simulation

This project simulates the effect of interface defects on minority carrier lifetime in semiconductors. It calculates and visualises how various parameters affect the surface recombination and resulting carrier lifetime.

## Project Structure

The project is organised into the following directories:

- `src/`: Contains the source code files
  - `chargeanddit.py`: Core functions and parameters for the simulation
  - `figure_generation.py`: Functions for generating figures
  - `main_runner.py`: Main entry point for running the simulation
  - `simulate_uv_comparison.py`: Script to simulate and compare lifetime curves before and after UV exposure
- `figures/`: Contains the generated figures
  - `figure 1.png`: Effect of varying fixed charge (Qfix) and varying peak Dit (Ditmax)
  - `figure 2.png`: Effect of Gaussian Width and Defect Position
  - `figure 3.png`: Effect of Charged Capture Cross-Section and Correlation Energy
  - `figure_uv_comparison.png`: Comparison of simulated and experimental lifetime curves before and after UV exposure
- `reference/`: Contains reference figures for validation
- `validation/`: Contains validation scripts
  - `validate_results.py`: Script for validating the generated figures against references

## How to Run the Simulation

To run the main simulation and generate all standard figures:

```bash
python3 src/main_runner.py
```

This will:
1. Generate three figures showing the effect of various parameters on minority carrier lifetime
2. Save the figures to the `figures/` directory
3. Validate the figures against reference figures (if available)

To run the UV comparison simulation specifically:

```bash
python3 src/simulate_uv_comparison.py
```

This will:
1. Generate a figure comparing simulated and experimental lifetime curves before and after UV exposure
2. Save the figure as `figure_uv_comparison.png` in the `figures/` directory

## Validation

The project includes a validation system to ensure that the simulation results are consistent. The validation process:

1. Checks if the figures exist and have non-zero size
2. Compares the generated figures with reference figures (if available)
3. Creates reference figures if they don't exist

To run the validation separately:

```bash
python3 validation/validate_results.py
```

## Background

The simulation models the effect of interface defects on minority carrier lifetime in semiconductors. It considers:

- Fixed charge at the interface (Qfix)
- Density of interface traps (Dit)
- Gaussian distribution of defects
- Capture cross-sections
- Correlation energy

The results are visualised as plots of minority carrier lifetime vs. carrier density for different parameter values.
