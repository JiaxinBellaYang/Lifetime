#!/usr/bin/env python3
"""
Main runner script for the lifetime simulation.

This script serves as the entry point for running the simulation and generating figures.
It imports the necessary functions from chargeanddit.py and figure_generation.py.
"""

import numpy as np
from scipy.constants import e as elementary_charge
import matplotlib.pyplot as plt

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import from chargeanddit.py
from chargeanddit import (
    surfaceLifetime, ni_func, Ev, Ec, ENERGY_POINTS, T,
    Ndop_bulk, dop_type_bulk, Ndop_emitter, dop_type_emitter,
    GAUSS_E0, GAUSS_SIGMA, lw
)

# Import from figure_generation.py
from figure_generation import generate_figure1, generate_figure2, generate_figure3

def validate_results():
    """
    Validate the results by checking if the generated figures exist and have non-zero size.
    Also performs a comprehensive validation by comparing with reference figures.
    """
    import os
    import subprocess
    
    # First, perform basic validation
    figures_dir = "figures/"
    figures = ["figure 1.png", "figure 2.png", "figure 3.png"]
    all_valid = True
    
    print("\nPerforming basic validation...")
    for fig in figures:
        fig_path = os.path.join(figures_dir, fig)
        if not os.path.exists(fig_path):
            print(f"  ✗ {fig_path} is missing")
            all_valid = False
            continue
            
        if os.path.getsize(fig_path) == 0:
            print(f"  ✗ {fig_path} is empty (zero size)")
            all_valid = False
            continue
            
        print(f"  ✓ {fig_path} exists and has non-zero size")
    
    if not all_valid:
        print("Basic validation failed: Some figures are missing or empty.")
        return False
    
    print("Basic validation successful: All figures exist and have content.")
    
    # Now perform comprehensive validation using the validate_results.py script
    print("\nPerforming comprehensive validation...")
    try:
        # Run the validation script
        result = subprocess.run(['python3', 'validation/validate_results.py'], 
                               capture_output=True, text=True, check=False)
        
        # Print the output from the validation script
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"Validation script returned error code: {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            return False
        
        # Check if validation was successful based on output
        if "All figures match their references" in result.stdout:
            print("Comprehensive validation successful!")
            return True
        elif "Reference figure" in result.stdout and "does not exist" in result.stdout:
            print("Reference figures created. Run again to validate against these references.")
            return True
        else:
            print("Comprehensive validation failed: Figures differ from references.")
            return False
            
    except Exception as e:
        print(f"Error running validation script: {str(e)}")
        return False

def main():
    """
    Main function to run the simulation and generate figures.
    """
    print("Starting lifetime simulation...")
    
    # Generate Figure 1: Effect of varying fixed charge (Qfix) and varying peak Dit (Ditmax)
    generate_figure1(
        surfaceLifetime=surfaceLifetime,
        ni_func=ni_func,
        Ev=Ev,
        Ec=Ec,
        ENERGY_POINTS=ENERGY_POINTS,
        T=T,
        Ndop_bulk=Ndop_bulk,
        dop_type_bulk=dop_type_bulk,
        Ndop_emitter=Ndop_emitter,
        dop_type_emitter=dop_type_emitter,
        GAUSS_E0=GAUSS_E0,
        GAUSS_SIGMA=GAUSS_SIGMA,
        elementary_charge=elementary_charge,
        lw=lw
    )
    
    # Generate Figure 2: Effect of Gaussian Width and Defect Position
    generate_figure2(
        surfaceLifetime=surfaceLifetime,
        ni_func=ni_func,
        Ev=Ev,
        Ec=Ec,
        ENERGY_POINTS=ENERGY_POINTS,
        T=T,
        Ndop_bulk=Ndop_bulk,
        dop_type_bulk=dop_type_bulk,
        Ndop_emitter=Ndop_emitter,
        dop_type_emitter=dop_type_emitter,
        elementary_charge=elementary_charge,
        lw=lw
    )
    
    # Generate Figure 3: Effect of Charged Capture Cross-Section and Correlation Energy
    generate_figure3(
        surfaceLifetime=surfaceLifetime,
        ni_func=ni_func,
        Ev=Ev,
        Ec=Ec,
        ENERGY_POINTS=ENERGY_POINTS,
        T=T,
        Ndop_bulk=Ndop_bulk,
        dop_type_bulk=dop_type_bulk,
        Ndop_emitter=Ndop_emitter,
        dop_type_emitter=dop_type_emitter,
        elementary_charge=elementary_charge,
        lw=lw
    )
    
    print("All figures generated successfully.")
    
    # Validate the results
    validate_results()

if __name__ == "__main__":
    main()
