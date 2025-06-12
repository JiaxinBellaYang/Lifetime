import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as boltzmann_k, e as elementary_charge, epsilon_0
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pylab as pl

# Function to create energy arrays
def create_energy_array(start, end, points):
    """Create a linearly spaced energy array with the specified parameters."""
    return np.linspace(start, end, points)

# Function to log progress
def log_progress(current, total, item_name, value=None):
    """Log progress of an iteration."""
    if value is not None:
        print(f"  Processing {item_name} {current+1}/{total}: {value:.2e}")
    else:
        print(f"Processing {item_name} {current+1}/{total}")

# Function to calculate Gaussian distribution for Dit
def Dig_func(E, *params):
    """
    Calculates the Gaussian distribution for the Density of Interface Traps (Dit).

    Args:
        E (np.ndarray): Energy array (eV).
        *params: Tuple containing:
            E0 (float): Center energy of the Gaussian distribution (eV).
            Dit_0g (float): Peak value of the Dit distribution (cm^-2 eV^-1).
            sigma (float): Standard deviation (width) of the Gaussian distribution (eV).

    Returns:
        np.ndarray: Dit distribution as a function of energy (cm^-2 eV^-1).
    """
    E0_gauss = params[0]
    Dit_0g = params[1]
    sigma_gauss = params[2]
    Dit_g = Dit_0g * np.exp(-((E - E0_gauss) / sigma_gauss)**2 / 2)
    return Dit_g

def generate_figure1(surfaceLifetime, ni_func, Ev, Ec, ENERGY_POINTS, T, Ndop_bulk, dop_type_bulk, 
                     Ndop_emitter, dop_type_emitter, GAUSS_E0, GAUSS_SIGMA, elementary_charge, lw=2):
    """
    Generate Figure 1: Effect of varying fixed charge (Qfix) and varying peak Dit (Ditmax)
    
    Args:
        surfaceLifetime: Function to calculate surface lifetime
        ni_func: Function to calculate intrinsic carrier concentration
        Ev, Ec: Valence and conduction band energies
        ENERGY_POINTS: Number of points for energy arrays
        T: Temperature in Kelvin
        Ndop_bulk, dop_type_bulk: Bulk doping parameters
        Ndop_emitter, dop_type_emitter: Emitter doping parameters
        GAUSS_E0, GAUSS_SIGMA: Gaussian distribution parameters
        elementary_charge: Elementary charge constant
        lw: Line width for plotting
        
    Returns:
        None (saves figure to file)
    """
    print("Starting calculations for Figure 1...")
    
    # Create energy array
    E = create_energy_array(Ev, Ec, ENERGY_POINTS)
    
    # Carrier density array for calculations
    dn_array = np.logspace(14, 17)
    
    # Calculate intrinsic carrier concentration
    ni_b = ni_func(T, Ndop_bulk, dop_type_bulk)
    
    # Set up colors for plotting
    colors = pl.cm.hsv(np.linspace(0, 1, 12))
    
    # Create figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    
    # --- Simulation Loop 1: Varying Fixed Charge (Qfix) ---
    print("Starting calculations for varying Qfix...")
    # Define parameters held constant in this loop
    Ditmax_q_loop = 1E10  # Peak Dit (cm^-2 eV^-1)
    params_g_q_loop = (GAUSS_E0, Ditmax_q_loop, GAUSS_SIGMA)
    Dit_g_q_loop = Dig_func(E, *params_g_q_loop)  # Dit distribution for this loop
    
    # Loop over different fixed charge values (convert cm^-2 to C/cm^2)
    qfix_cm2_array = np.logspace(10, 11.5, 10)  # Charge density in cm^-2
    for i, qfix_cm2 in enumerate(qfix_cm2_array):
        Qfix_coulomb = -qfix_cm2 * elementary_charge  # Fixed charge in C/cm^2
        log_progress(i, len(qfix_cm2_array), "Qfix (cm^-2)", qfix_cm2)
        tau_surface_array_q = []  # Store results for this Qfix value
        
        # Loop over injection densities
        for count, dn in enumerate(dn_array):
            Delta_n = dn
            Ndop = Ndop_bulk
            dop_type = dop_type_bulk
            
            # Calculate equilibrium concentrations
            if Ndop > 0:
                n0 = Ndop * (1 - dop_type) + ni_b**2 / Ndop * dop_type
                p0 = Ndop * dop_type + ni_b**2 / Ndop * (1 - dop_type)
            else:
                n0 = ni_b
                p0 = ni_b
                
            # Calculate non-equilibrium concentrations
            n = Delta_n + n0
            p = Delta_n + p0
            
            # Calculate surface lifetime for this dn and Qfix
            tau_surface = surfaceLifetime(n0, p0, n, p, Delta_n, Qfix_coulomb, T, 
                                         Ndop_emitter, Ndop_bulk, dop_type_emitter, 
                                         dop_type_bulk, dn, Dit_g_q_loop)
            tau_surface_array_q.append(tau_surface * 1E3)  # Convert to ms
            
        # Plot results for this Qfix value
        ax[0].loglog(dn_array, tau_surface_array_q, color=colors[i], linewidth=lw)
    
    # Add annotation for Qfix plot
    arrow_q = patches.FancyArrowPatch(
        (1E15, 1.5), (1E15, 100), arrowstyle='->', mutation_scale=15, color='black')
    ax[0].text(-0.1, 1.05, '(a)', transform=ax[0].transAxes, fontsize=12, va='bottom', ha='left')
    ax[0].add_patch(arrow_q)
    ax[0].text(1.5e15, 100, 'Higher charge', fontsize=10, color='black')
    
    # --- Simulation Loop 2: Varying Peak Dit (Ditmax) ---
    print("\nStarting calculations for varying Ditmax...")
    # Define parameters held constant in this loop
    Qfix_dit_loop_cm2 = 1e10  # Fixed charge density (cm^-2)
    Qfix_dit_loop_coulomb = -Qfix_dit_loop_cm2 * elementary_charge  # Convert to C/cm^2
    
    # Loop over different peak Dit values
    ditmax_array = np.logspace(10, 12, 10)  # Peak Dit (cm^-2 eV^-1)
    for i, ditmax_val in enumerate(ditmax_array):
        log_progress(i, len(ditmax_array), "Ditmax (cm^-2 eV^-1)", ditmax_val)
        
        # Calculate Dit distribution for this Ditmax value
        params_g_dit_loop = (GAUSS_E0, ditmax_val, GAUSS_SIGMA)
        Dit_g_dit_loop = Dig_func(E, *params_g_dit_loop)
        tau_surface_array_dit = []  # Store results for this Ditmax
        
        # Loop over injection densities
        for count, dn in enumerate(dn_array):
            Delta_n = dn
            Ndop = Ndop_bulk
            dop_type = dop_type_bulk
            
            # Calculate equilibrium concentrations
            if Ndop > 0:
                n0 = Ndop * (1 - dop_type) + ni_b**2 / Ndop * dop_type
                p0 = Ndop * dop_type + ni_b**2 / Ndop * (1 - dop_type)
            else:
                n0 = ni_b
                p0 = ni_b
                
            # Calculate non-equilibrium concentrations
            n = Delta_n + n0
            p = Delta_n + p0
            
            # Calculate surface lifetime for this dn and Ditmax
            tau_surface = surfaceLifetime(n0, p0, n, p, Delta_n, Qfix_dit_loop_coulomb, T, 
                                         Ndop_emitter, Ndop_bulk, dop_type_emitter, 
                                         dop_type_bulk, dn, Dit_g_dit_loop)
            tau_surface_array_dit.append(tau_surface * 1E3)  # Convert to ms
            
        # Plot results for this Ditmax value
        ax[1].loglog(dn_array, tau_surface_array_dit, color=colors[i], linewidth=lw)
    
    # Add annotation for Dit plot
    arrow_dit = patches.FancyArrowPatch(
        (1E15, 0.01), (1E15, 10), arrowstyle='<-', mutation_scale=15, color='black')
    ax[1].text(-0.1, 1.05, '(b)', transform=ax[1].transAxes, fontsize=12, va='bottom', ha='left')
    ax[1].add_patch(arrow_dit)
    ax[1].text(1.4e15, 0.02, 'Higher Dit', fontsize=10, color='black')
    
    # --- Final Plot Formatting ---
    for axis in ax:
        axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        axis.set_xlim([1e14, 1.1e16])
        axis.set_xlabel('Carrier density $(cm^{-3})$')
        axis.set_ylabel('Minority carrier lifetime $(ms)$')
        axis.tick_params(axis='both', which='major', direction='in', length=8)
        axis.tick_params(axis='both', which='minor', direction='in', length=8)
    
    plt.tight_layout()
    print("\nCalculations complete. Saving figure...")
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/figure 1.png')
    print("Figure saved as 'figures/figure 1.png'")

def generate_figure2(surfaceLifetime, ni_func, Ev, Ec, ENERGY_POINTS, T, Ndop_bulk, dop_type_bulk, 
                     Ndop_emitter, dop_type_emitter, elementary_charge, lw=2):
    """
    Generate Figure 2: Effect of Gaussian Width and Defect Position
    
    Args:
        surfaceLifetime: Function to calculate surface lifetime
        ni_func: Function to calculate intrinsic carrier concentration
        Ev, Ec: Valence and conduction band energies
        ENERGY_POINTS: Number of points for energy arrays
        T: Temperature in Kelvin
        Ndop_bulk, dop_type_bulk: Bulk doping parameters
        Ndop_emitter, dop_type_emitter: Emitter doping parameters
        elementary_charge: Elementary charge constant
        lw: Line width for plotting
        
    Returns:
        None (saves figure to file)
    """
    print("\nStarting calculations for Figure 2...")
    
    # Create energy array
    E = create_energy_array(Ev, Ec, ENERGY_POINTS)
    
    # Carrier density array for calculations
    dn_array = np.logspace(14, 17)
    
    # Calculate intrinsic carrier concentration
    ni_b = ni_func(T, Ndop_bulk, dop_type_bulk)
    
    # Set up colors for plotting
    colors = pl.cm.hsv(np.linspace(0, 1, 12))
    
    # Create figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    
    # --- Simulation Loop 1: Varying Gaussian Width ---
    print("Starting calculations for varying Gaussian Width...")
    for i, gaussianWidth in enumerate(np.linspace(0.1, 0.5, 10)):
        U = 0.15 
        Ditmax = 1E10
        Ditposition = 0.56
        Qtot = -5e10 * elementary_charge
        params_g = (Ditposition, Ditmax, gaussianWidth)
        Dit_g = Dig_func(E, *params_g)
        tau_surface_array = []
        
        for count, dn in enumerate(dn_array):
            Delta_n = dn
            Ndop = Ndop_bulk
            dop_type = dop_type_bulk
            n0 = Ndop*(1-dop_type) + ni_b**2/Ndop*dop_type
            p0 = Ndop*dop_type + ni_b**2/Ndop*(1 - dop_type)
            n = Delta_n + n0
            p = Delta_n + p0
            tau_surface = surfaceLifetime(n0, p0, n, p, Delta_n, Qtot, T, Ndop_emitter, 
                                         Ndop_bulk, dop_type_emitter, dop_type_bulk, dn, Dit_g)
            tau_surface_array.append(tau_surface*1E3)
            
        ax[0].loglog(dn_array, tau_surface_array, color=colors[i], linewidth=lw)
    
    arrow = patches.FancyArrowPatch(
        (1.5E14, 3), (1.5E14, 10), arrowstyle='<-', mutation_scale=15, color='black')
    ax[0].text(-0.1, 1.05, '(a)', transform=ax[0].transAxes, fontsize=12, va='bottom', ha='left')
    ax[0].add_patch(arrow)
    ax[0].text(1.6e14, 2.5, 'Wider Gaussian distribution', fontsize=10, color='black')
    
    # --- Simulation Loop 2: Varying Defect Position ---
    print("Starting calculations for varying Defect Position...")
    for i, Ditposition in enumerate(np.linspace(0.1, 0.55, 10)):
        U = 0.15 
        gaussianWidth = 0.18
        Ditmax = 1E10
        Qtot = -5e10 * elementary_charge
        params_g = (Ditposition, Ditmax, gaussianWidth)
        Dit_g = Dig_func(E, *params_g)
        tau_surface_array = []
        
        for count, dn in enumerate(dn_array):
            Delta_n = dn
            Ndop = Ndop_bulk
            dop_type = dop_type_bulk
            n0 = Ndop*(1-dop_type) + ni_b**2/Ndop*dop_type
            p0 = Ndop*dop_type + ni_b**2/Ndop*(1 - dop_type)
            n = Delta_n + n0
            p = Delta_n + p0
            tau_surface = surfaceLifetime(n0, p0, n, p, Delta_n, Qtot, T, Ndop_emitter, 
                                         Ndop_bulk, dop_type_emitter, dop_type_bulk, dn, Dit_g)
            tau_surface_array.append(tau_surface*1E3)
            
        ax[1].loglog(dn_array, tau_surface_array, color=colors[i], linewidth=lw)
    
    arrow = patches.FancyArrowPatch(
        (1.5E14, 5), (1.5E14, 10), arrowstyle='<-', mutation_scale=15, color='black')
    ax[1].text(-0.1, 1.05, '(b)', transform=ax[1].transAxes, fontsize=12, va='bottom', ha='left')
    ax[1].add_patch(arrow)
    ax[1].text(1.6e14, 4, 'Deeper defect', fontsize=10, color='black')
    
    # --- Final Plot Formatting for Figure 2 ---
    for axis in ax:
        axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        axis.set_xlim([1e14, 1.1e16])
        axis.set_ylim([0.9, 12])
        axis.set_xlabel('Carrier density $(cm^{-3})$')
        axis.set_ylabel('Minority carrier lifetime $(ms)$')
        axis.tick_params(axis='both', which='major', direction='in', length=8)
        axis.tick_params(axis='both', which='minor', direction='in', length=8)
    
    plt.tight_layout()
    print("\nFigure 2 calculations complete. Saving figure...")
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/figure 2.png')
    print("Figure saved as 'figures/figure 2.png'")

def generate_figure3(surfaceLifetime, ni_func, Ev, Ec, ENERGY_POINTS, T, Ndop_bulk, dop_type_bulk, 
                     Ndop_emitter, dop_type_emitter, elementary_charge, lw=2):
    """
    Generate Figure 3: Effect of Charged Capture Cross-Section and Correlation Energy
    
    Args:
        surfaceLifetime: Function to calculate surface lifetime
        ni_func: Function to calculate intrinsic carrier concentration
        Ev, Ec: Valence and conduction band energies
        ENERGY_POINTS: Number of points for energy arrays
        T: Temperature in Kelvin
        Ndop_bulk, dop_type_bulk: Bulk doping parameters
        Ndop_emitter, dop_type_emitter: Emitter doping parameters
        elementary_charge: Elementary charge constant
        lw: Line width for plotting
        
    Returns:
        None (saves figure to file)
    """
    print("\nStarting calculations for Figure 3...")
    
    # Create energy array
    E = create_energy_array(Ev, Ec, ENERGY_POINTS)
    
    # Carrier density array for calculations
    dn_array = np.logspace(14, 17)
    
    # Calculate intrinsic carrier concentration
    ni_b = ni_func(T, Ndop_bulk, dop_type_bulk)
    
    # Set up colors for plotting
    colors = pl.cm.hsv(np.linspace(0, 1, 12))
    
    # Create figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    
    # Set capture cross-sections for neutral states
    s0n = 1E-17
    s0p = 1E-17
    
    # --- Simulation Loop 1: Varying Charged Capture Cross-Section ---
    print("Starting calculations for varying Charged Capture Cross-Section...")
    for i, scharged in enumerate(np.logspace(-17, -15, 10)):
        splusn = scharged  # Charged electron capture cross-section
        sminusp = scharged  # Charged hole capture cross-section
        U = 0.15 
        Ditmax = 1E10
        gaussianWidth = 0.18
        Ditposition = 0.56
        Qtot = -5e10 * elementary_charge
        params_g = (Ditposition, Ditmax, gaussianWidth)
        Dit_g = Dig_func(E, *params_g)
        tau_surface_array = []
        
        for count, dn in enumerate(dn_array):
            Delta_n = dn
            Ndop = Ndop_bulk
            dop_type = dop_type_bulk
            n0 = Ndop*(1-dop_type) + ni_b**2/Ndop*dop_type
            p0 = Ndop*dop_type + ni_b**2/Ndop*(1 - dop_type)
            n = Delta_n + n0
            p = Delta_n + p0
            tau_surface = surfaceLifetime(n0, p0, n, p, Delta_n, Qtot, T, Ndop_emitter, 
                                         Ndop_bulk, dop_type_emitter, dop_type_bulk, dn, Dit_g)
            tau_surface_array.append(tau_surface*1E3)
            
        ax[0].loglog(dn_array, tau_surface_array, color=colors[i], linewidth=lw)
    
    arrow = patches.FancyArrowPatch(
        (1.5E14, 0.5), (1.5E14, 50), arrowstyle='<-', mutation_scale=15, color='black')
    ax[0].text(-0.1, 1.05, '(a)', transform=ax[0].transAxes, fontsize=12, va='bottom', ha='left')
    ax[0].add_patch(arrow)
    ax[0].text(1.6e14, 0.55, 'Larger charge capture cross section', fontsize=10, color='black')
    ax[0].set_xlim([1e14, 1.1e16])
    ax[0].set_xlabel('Carrier density $(cm^{-3})$')
    ax[0].set_ylabel('Minority carrier lifetime $(ms)$')
    ax[0].tick_params(axis='both', which='major', direction='in', length=8)
    ax[0].tick_params(axis='both', which='minor', direction='in', length=8)
    
    # --- Simulation Loop 2: Varying Correlation Energy ---
    print("Starting calculations for varying Correlation Energy...")
    for i, U in enumerate(np.linspace(0.05, 0.7, 10)):
        Ditmax = 1E10
        gaussianWidth = 0.18
        Ditposition = 0.56
        Qtot = -5e10 * elementary_charge
        params_g = (Ditposition, Ditmax, gaussianWidth)
        Dit_g = Dig_func(E, *params_g)
        tau_surface_array = []
        
        for count, dn in enumerate(dn_array):
            Delta_n = dn
            Ndop = Ndop_bulk
            dop_type = dop_type_bulk
            n0 = Ndop*(1-dop_type) + ni_b**2/Ndop*dop_type
            p0 = Ndop*dop_type + ni_b**2/Ndop*(1 - dop_type)
            n = Delta_n + n0
            p = Delta_n + p0
            tau_surface = surfaceLifetime(n0, p0, n, p, Delta_n, Qtot, T, Ndop_emitter, 
                                         Ndop_bulk, dop_type_emitter, dop_type_bulk, dn, Dit_g)
            tau_surface_array.append(tau_surface*1E3)
            
        ax[1].loglog(dn_array, tau_surface_array, color=colors[i], linewidth=lw)
    
    arrow = patches.FancyArrowPatch(
        (1.5E14, 0.5), (1.5E14, 15), arrowstyle='->', mutation_scale=15, color='black')
    ax[1].text(-0.1, 1.05, '(b)', transform=ax[1].transAxes, fontsize=12, va='bottom', ha='left')
    ax[1].add_patch(arrow)
    ax[1].text(1.6e14, 0.55, 'Larger correlation energy', fontsize=10, color='black')
    
    # --- Final Plot Formatting for Figure 3 ---
    for axis in ax:
        axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        axis.set_xlim([1e14, 1.1e16])
        axis.set_xlabel('Carrier density $(cm^{-3})$')
        axis.set_ylabel('Minority carrier lifetime $(ms)$')
        axis.tick_params(axis='both', which='major', direction='in', length=8)
        axis.tick_params(axis='both', which='minor', direction='in', length=8)
    
    plt.tight_layout()
    print("\nFigure 3 calculations complete. Saving figure...")
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/figure 3.png')
    print("Figure saved as 'figures/figure 3.png'")

