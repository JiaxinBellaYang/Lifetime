#!/usr/bin/env python3
"""
Script to simulate and plot lifetime curves comparing before and after UV exposure.

This script calculates the effective lifetime based on surface, intrinsic,
and bulk SRH recombination.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e as elementary_charge
from matplotlib.ticker import ScalarFormatter
import os
import sys
import pandas as pd

# Add the src directory to the Python path if not already there
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import necessary functions and constants from chargeanddit
try:
    from chargeanddit import (
        surfaceLifetime, intrinsicLifetime, Dig_func, ni_func, create_energy_array,
        Ev, Ec, T, dop_type_bulk, Ndop_emitter, dop_type_emitter,
        tau_SRH, ENERGY_POINTS, kB, NC, NV, GAUSS_U,
        s0n, s0p, splusn, sminusp, vth_n, vth_p,
        sigma_n1, sigma_p1, sigma_n2, sigma_p2,
        lw
    )
except ImportError as e:
    print(f"Error importing from chargeanddit: {e}")
    print("Ensure chargeanddit.py is in the same directory or Python path.")
    sys.exit(1)


Ndop_bulk = 1.5e15  # Bulk doping concentration (cm^-3)
W = 0.015  # Sample thickness (cm)

def simulate_and_plot_uv_comparison():
    print("Starting calculations for UV Comparison Figure...")

    # Define experimental data file paths
    data_folder = "data"
    before_uv_file = os.path.join(data_folder, "L2_pre_ini_G.xlsx")
    after_uv_file = os.path.join(data_folder, "L2_pre_UV20_G.xlsx")

    # Define simulation parameters
    params_before_uv = {
        "label": "Before UV (Simulated)",
        "Dit0_v": 1, "Ev_trap_sigma": 1.500e-02,
        "Dit0_c": 1, "Ec_trap_sigma": 1.000e-02,
        "Dit0_g1": 1.340e+12, "E0_g1": 3.716e-01, "sigma_g1": 1.724e-01,
        "Dit0_g2": 1.805e+12, "E0_g2": 8.107e-01, "sigma_g2": 1.186e-01,
        "Qfix": 1e12,
        "color": "blue"
    }

    params_after_uv = {
        "label": "After UV (Simulated)",
        "Dit0_v": 1, "Ev_trap_sigma": 1.448e-02,
        "Dit0_c": 1, "Ec_trap_sigma": 3.599e-02,
        "Dit0_g1": 8.806e+12, "E0_g1": 2.000e-01, "sigma_g1": 1.339e-01,
        "Dit0_g2": 3.111e+12, "E0_g2": 6.978e-01, "sigma_g2": 2.000e-01,
        "Qfix": 8.5e11,
        "color": "red"
    }

    parameter_sets = [params_before_uv, params_after_uv]

    # Load experimental data
    try:
        df_before_uv = pd.read_excel(before_uv_file, sheet_name="RawData")
        df_after_uv = pd.read_excel(after_uv_file, sheet_name="RawData")

        # Extract relevant columns
        time_before_uv = df_before_uv["Time (s)"].values
        tau_before_uv = df_before_uv["Tau (sec)"].values
        minority_carrier_density_before_uv = df_before_uv["Minority Carrier Density"].values

        time_after_uv = df_after_uv["Time (s)"].values
        tau_after_uv = df_after_uv["Tau (sec)"].values
        minority_carrier_density_after_uv = df_after_uv["Minority Carrier Density"].values

    except FileNotFoundError:
        print("Error: Experimental data files not found. Skipping experimental data plotting.")
        time_before_uv = []
        tau_before_uv = []
        minority_carrier_density_before_uv = []
        time_after_uv = []
        tau_after_uv = []
        minority_carrier_density_after_uv = []


    E_array = create_energy_array(Ev, Ec, ENERGY_POINTS)
    
    dn_array = np.logspace(14, 17, 100)

    ni_b = ni_func(T, Ndop_bulk, dop_type_bulk)
    
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('axes', labelsize=10)
    plt.rc('legend', fontsize=10)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['figure.dpi'] = 80
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['mathtext.fontset'] = "dejavuserif"
    
    fig, ax = plt.subplots(figsize=(7, 5))

    all_tau_eff = []
    for params in parameter_sets:
        print(f"  Calculating lifetime for: {params['label']}")
        
        # Single-Gaussian for valence and conduction band traps
        Dit_valence = Dig_func(E_array, Ev, params["Dit0_v"], params["Ev_trap_sigma"], 0, 0, 1)
        Dit_conduction = Dig_func(E_array, Ec, params["Dit0_c"], params["Ec_trap_sigma"], 0, 0, 1)
        
        # Double-Gaussian for midgap traps
        Dit_midgap = Dig_func(
            E_array,
            params["E0_g1"], params["Dit0_g1"], params["sigma_g1"],
            params["E0_g2"], params["Dit0_g2"], params["sigma_g2"]
        )

        Dit_tot = Dit_valence + Dit_conduction + Dit_midgap

        tau_eff_array = []

        if Ndop_bulk > 0:
            if dop_type_bulk == 1:
                n0 = Ndop_bulk
                p0 = ni_b**2/Ndop_bulk
            else:
                n0 = ni_b**2/Ndop_bulk
                p0 = Ndop_bulk
        else:
            n0 = ni_b
            p0 = ni_b

        for dn in dn_array:
            n = n0 + dn
            p = p0 + dn

            # Apply negative sign and convert to charge for Qfix
            qfix_charge = -params["Qfix"] * elementary_charge
            
            tau_surf = surfaceLifetime(
                n0, p0, n, p, dn, qfix_charge, T,
                Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk,
                dn, Dit_tot
            )
            tau_intr = intrinsicLifetime(n0, p0, n, p, dn)

            inv_tau_srh = 1.0 / tau_SRH if tau_SRH > 0 else 0
            inv_tau_intr = 1.0 / tau_intr if tau_intr > 0 and np.isfinite(tau_intr) else 0
            inv_tau_surf = 1.0 / tau_surf if tau_surf > 0 and np.isfinite(tau_surf) else 0
            inv_tau_eff_sum = inv_tau_srh + inv_tau_intr + inv_tau_surf

            if inv_tau_eff_sum > 1e-20:
                 tau_eff = 1.0 / inv_tau_eff_sum
            else:
                 tau_eff = np.inf

            tau_eff_array.append(tau_eff)
        
        current_tau_eff = np.array(tau_eff_array)
        ax.loglog(dn_array, current_tau_eff * 1e3, label=params["label"], color=params["color"], linewidth=lw)
        all_tau_eff.extend(current_tau_eff * 1e3)
        print(f"  Finished calculation for: {params['label']}")

    # Plot experimental data
    if len(time_before_uv) > 0:
        ax.loglog(minority_carrier_density_before_uv, tau_before_uv * 1e3, 'bo', label="Before UV (Exp)", markersize=5)
    if len(time_after_uv) > 0:
        ax.loglog(minority_carrier_density_after_uv, tau_after_uv * 1e3, 'ro', label="After UV (Exp)", markersize=5)

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.set_xlim([1e14, 1.1e17])
    ax.set_ylim([0.01, 100])

    ax.set_xlabel('Minority Carrier Density (cm$^{-3}$)')
    ax.set_ylabel('Effective Lifetime (ms)')
    ax.set_title('Simulated and Experimental Lifetime: Before vs After UV')
    ax.legend(frameon=False)
    ax.tick_params(axis='both', which='major', direction='in', length=8)
    ax.tick_params(axis='both', which='minor', direction='in', length=8)
    # ax.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    
    project_root = os.path.dirname(script_dir)
    figures_dir = os.path.join(project_root, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    output_filename = os.path.join(figures_dir, 'figure_uv_comparison.png')
    fig.savefig(output_filename)
    print(f"\nUV Comparison calculations complete. Saving figure...")
    print(f"Figure saved as '{output_filename}'")

    print("Debug: sigma_n1 values:", sigma_n1)  # Debugging capture cross section data
    print("Debug: sigma_p1 values:", sigma_p1)  # Debugging capture cross section data
    print("Debug: sigma_n2 values:", sigma_n2)  # Debugging capture cross section data
    print("Debug: sigma_p2 values:", sigma_p2)  # Debugging capture cross section data

if __name__ == "__main__":
    simulate_and_plot_uv_comparison()
