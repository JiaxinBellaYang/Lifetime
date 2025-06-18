import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as boltzmann_k, e as elementary_charge, epsilon_0 # Import constants directly
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as colors
import matplotlib.patches as patches
import datetime # Used for potential file saving, keep for now
from scipy.optimize import fsolve
import matplotlib.pylab as pl
import pandas as pd  # For loading experimental data
import os
# Import figure generation functions
# from figure_generation import generate_figure1, generate_figure2, generate_figure3

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Matplotlib configuration
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=10)
plt.rc('legend', fontsize=10)
plt.rcParams["font.family"] = "serif"
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['figure.dpi'] = 120 # Increased for better viewing
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['mathtext.fontset'] = "dejavuserif"

# Array dimensions
ENERGY_POINTS = 10000  # Number of points for energy arrays

# Physical constants
T = 300.0  # Temperature in Kelvin
Ec = 1.12  # Conduction band energy (eV)
Ev = 0.0   # Valence band energy (eV)
Eg = Ec - Ev  # Band gap (eV)
kT = (boltzmann_k / elementary_charge) * T # Thermal energy (eV)
kB = boltzmann_k / elementary_charge # Boltzmann constant in eV/K
W = 0.014  # Sample thickness (cm)

# Semiconductor parameters
NC = 2.86E+19  # Effective density of states in conduction band (cm^-3)
NV = 3.11E+19  # Effective density of states in valence band (cm^-3)
Ndop_bulk = 1.55e15  # Bulk doping concentration (cm^-3)
Ndop_emitter = 0.0  # Emitter doping concentration (cm^-3)
dop_type_bulk = 1.0  # Bulk doping type (1 for n-type, 0 for p-type)
dop_type_emitter = 1.0  # Emitter doping type (1 for n-type, 0 for p-type)
eps_rel = 11.68  # Relative permittivity
eps_Si = eps_rel * epsilon_0 * 1e-2  # Silicon permittivity in F/cm

# Thermal velocities
vth_n = 2.046e7 * (T/300.0)**0.5  # Electron thermal velocity (cm/s)
vth_p = 1.688e7 * (T/300.0)**0.5  # Hole thermal velocity (cm/s)

#Add new parameters for the Gaussian cross-section model
# --- Gaussian Capture Cross Section Parameters ---
# For electrons (sigma_n)
SIGMA0_N = 4.1e-14  # cm^-2
A_N = 80.0          # eV^-2
E0_N = 0.04         # eV (relative to mid-gap)
# For holes (sigma_p)
SIGMA0_P = 4.1e-16  # cm^-2
A_P = 110.0         # eV^-2
E0_P = -0.15        # eV (relative to mid-gap)
###Debug parameters
print(f"DEBUG STEP 1: Using SIGMA0_N = {SIGMA0_N:.2e} and SIGMA0_P = {SIGMA0_P:.2e}")

# s0n = 1E-17
# s0p = 1E-17
# splusn = 1E-16
# sminusp = 1E-16
# SP0_DEFAULT = 4.5E-18
# SN0_DEFAULT = 100*SP0_DEFAULT
# SPMINUS_DEFAULT = 10 * SP0_DEFAULT
# SNPLUS_DEFAULT = 10 * SN0_DEFAULT

# Gaussian Defect Distribution Parameters (for Dit)
GAUSS_E0 = 0.56  # Centre of Gaussian Dit distribution (eV)
GAUSS_SIGMA = 0.18  # Width of Gaussian Dit distribution (eV)
GAUSS_U = 0.15  # Correlation energy for amphoteric defects (eV)

# Hardcoded Dit parameters (used in J0sv2_func, consider making configurable)
VALENCE_DIT_E0 = 0.0
VALENCE_DIT_MAX = 4E10 # cm^-2 eV^-1
VALENCE_DIT_SIGMA = 0.18
CONDUCTION_DIT_E0 = 1.68 # eV - Note: This is outside the typical Si bandgap
CONDUCTION_DIT_MAX = 8E12 # cm^-2 eV^-1
CONDUCTION_DIT_SIGMA = 0.195

# Capture Cross Sections (used in J0sv2_func and main script section)
# Note: These are currently set but some are overwritten later. Clarify intended usage.
SP0_DEFAULT = 1E-20 # cm^2
SN0_DEFAULT = 100*SP0_DEFAULT # cm^2
SPMINUS_DEFAULT = 10 * SP0_DEFAULT # cm^2
SNPLUS_DEFAULT = 10 * SN0_DEFAULT # cm^2

# Intrinsic Lifetime Parameters (Richter model - used in intrinsicLifetime)
INTR_RMIN = 0.0
INTR_RMAX = 0.2
INTR_SMIN = 1e7
INTR_SMAX = 1.5e18
INTR_WMAX = 4.0e-18
INTR_WMIN = 1e19
INTR_B2 = 0.54
INTR_B4 = 1.25
INTR_R1 = 320.0
INTR_R2 = 2.5
INTR_S1 = 550.0
INTR_S2 = 3.0
INTR_W1 = 365.0
INTR_W2 = 3.54
INTR_N0EEH = 3.3e17 # cm^-3
INTR_N0EHH = 7.0e17 # cm^-3
INTR_GEEH_FACTOR = 13.0
INTR_GEHH_FACTOR = 7.5
INTR_GEEH_EXP = 0.6
INTR_GEHH_EXP = 0.63
INTR_AUGER_EEH = 2.5e-31 # cm^6/s
INTR_AUGER_EHH = 8.5e-32 # cm^6/s
INTR_AUGER_XXX_COEFF = 3.0e-29 # cm^(6-3*0.92)/s ? Units seem unusual
INTR_AUGER_XXX_EXP = 0.92

# a-Si like states
snaSi = 7E-16  # Electron capture cross section (cm^2)
spaSi = 7E-16  # Hole capture cross section (cm^2)

# Bulk lifetime
tau_SRH = 25E-3  # Shockley-Read-Hall lifetime (s)

# Optimization parameters
maxiterations = 140  # Max number of iterations in optimization routine
pointToSkip = 8  # Point to skip in injection dependent lifetime curve
Dit_tot_err = 0  # Error in Dit

# Plotting parameters
lw = 2  # Line width
ms = 8  # Marker size

# File to analyse
filenameArray = ['IP/PostLaser', 'IP/Initial']
label = ['After Treatment', 'Before Treatment']
colors_line = ['#FF6666', '#FFB366', '#FFFF66', '#66FF66', '#66FFFF', '#6666FF', '#B366FF', '#FF66FF', '#FF66B2', '#FF9999']
symbols = ['^', 'o', 's', 'd', 'v', 'p', 'h']

def make_color_darker(color, factor):
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    
    new_r = max(0, r - factor)
    new_g = max(0, g - factor)
    new_b = max(0, b - factor)
    
    new_color = "#{:02X}{:02X}{:02X}".format(int(new_r), int(new_g), int(new_b))
    return new_color

colors_marker = [make_color_darker(color, factor=70) for color in colors_line]

# Initialize arrays for results
tausamples = np.zeros(len(filenameArray))
qsamples = np.zeros(len(filenameArray))
ditsamples = np.zeros(len(filenameArray))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# The new function to calculate Gaussian cross-sections ###
def calculate_gaussian_sigma(E, sigma0, A, E0):
    """
    Calculates energy-dependent capture cross-section based on a Gaussian model.
    The energy E should be relative to the same reference as E0 (e.g., mid-gap).
    
    Args:
        E (np.ndarray): Energy array (eV), typically from Ev to Ec.
        sigma0 (float): Peak capture cross-section (cm^-2).
        A (float): Gaussian width parameter (eV^-2).
        E0 (float): Center energy of the Gaussian peak (eV), relative to mid-gap.
        
    Returns:
        np.ndarray: Energy-dependent capture cross-section (cm^-2).
    """
    # The energy array E is defined from Ev to Ec. We need it relative to mid-gap
    # for the formula to be correct, as E0 is given relative to mid-gap.
    E_midgap = (Ec + Ev) / 2
    E_relative_to_midgap = E - E_midgap
    return sigma0 * np.exp(-A * (E_relative_to_midgap - E0)**2)

def create_energy_array(start=Ev, end=Ec, points=ENERGY_POINTS):
    """Create a linearly spaced energy array with the specified parameters."""
    return np.linspace(start, end, points)

# Dig_func, ns_zero_func
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

def ns_zero_func(ns, *params):
    Q, T, Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk, Δn = params
    Ndop_surface_n = dop_type_emitter * Ndop_emitter + dop_type_bulk * Ndop_bulk
    Ndop_surface_p = (1-dop_type_emitter)*Ndop_emitter + (1-dop_type_bulk)*Ndop_bulk
    Ndop_surface = np.abs(Ndop_surface_n - Ndop_surface_p)
    Eth = kB*T
    ni_b = ni_func(T, Ndop_bulk, dop_type_bulk)    
    ni_e = ni_func(T, Ndop_surface, dop_type_emitter)  
    nd0 = dop_type_bulk*Ndop_bulk + (1-dop_type_bulk)*ni_b**2/Ndop_bulk
    pd0 = (1-dop_type_bulk)*Ndop_bulk + dop_type_bulk*ni_b**2/Ndop_bulk
    pd = pd0 + Δn
    nd = nd0 + Δn
    ps = (ni_e**2/ni_b**2)*pd*nd/ns    
    Phi_s = -Eth*np.log((ns*ni_b)/(nd*ni_e))
    fzero = ps - pd + ns - nd + Ndop_surface * Phi_s / Eth - Q**2 / (2 * elementary_charge * eps_Si * Eth)
    return fzero

def lookup(Y, yval, X):
    Y = np.array(Y)
    index = np.argmin(abs(Y-yval))
    if len(X) == 0:
        return index
    else:
        return X[index]


def J0sv2_func(X, *params):
    ns_in, T, Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk, dn, Nit_err = X
    Dit_E = params[0] 
    sigma_n = params[1] # This will now be an array from the Gaussian model
    sigma_p = params[2] # This will now be an array from the Gaussian model
    
    Eth = kB * T
    ni_b = ni_func(T, Ndop_bulk, dop_type_bulk)
    Ndop_surface_n = dop_type_emitter*Ndop_emitter + dop_type_bulk*Ndop_bulk
    Ndop_surface_p = (1-dop_type_emitter)*Ndop_emitter + (1-dop_type_bulk)*Ndop_bulk
    Ndop_surface = np.abs(Ndop_surface_n - Ndop_surface_p)  
    ni_e = ni_func(T, Ndop_surface, dop_type_emitter)

    nd0 = dop_type_bulk*Ndop_bulk + (1-dop_type_bulk)*ni_b**2/Ndop_bulk
    pd0 = (1-dop_type_bulk)*Ndop_bulk + dop_type_bulk*ni_b**2/Ndop_bulk

    pd = pd0 + dn
    nd = nd0 + dn
    ps = (ni_e**2 / ni_b**2) * pd * nd / ns_in
    
    n = nd # using bulk concentration for alpha calculation
    p = pd # using bulk concentration for alpha calculation
    
    E = create_energy_array()
    dE = E[1:] - E[:-1]

    cn = sigma_n * vth_n
    cp = sigma_p * vth_p
    
    # SRH parameters n1, p1
    n11 = NC * np.exp(-(Ec - E) / kT) # Corrected n1 definition
    p11 = NV * np.exp(-(E - Ev) / kT) # Corrected p1 definition

    # Standard SRH recombination formula
    R_denominator = (ns_in + n11) / cp + (ps + p11) / cn
    R = np.divide(Dit_E * (ps * ns_in - ni_e**2), R_denominator, out=np.zeros_like(R_denominator), where=R_denominator!=0)

    UofE = R
    Uint = np.sum((UofE[1:] + UofE[:-1]) / 2 * dE)

    # Occupancy factor calculation
    k = np.divide(sigma_n, sigma_p, out=np.zeros_like(sigma_n), where=sigma_p != 0)
    alpha_denominator = (k * n + p11)
    alpha = np.divide((k * n11 + p), alpha_denominator, out=np.zeros_like(alpha_denominator), where=alpha_denominator!=0)

    # Defect state populations
    denominator = alpha + 1
    Nsplus = np.divide(Dit_E, denominator, out=np.zeros_like(Dit_E), where=denominator!=0)
    Ns = alpha * Nsplus
    N = Nsplus + Ns
    
    Sn0_tem = N * sigma_n * vth_n
    Sp0_tem = N * sigma_p * vth_p

    denominator1 = np.divide(1, Sn0_tem, out=np.full_like(Sn0_tem, np.inf), where=Sn0_tem!=0) + np.divide(1, Sp0_tem, out=np.full_like(Sp0_tem, np.inf), where=Sp0_tem!=0)
    J0s_tem = np.divide(elementary_charge * ni_e**2, denominator1, out=np.zeros_like(denominator1), where=denominator1!=0) # This formula seems simplified, check SRH J0 definition

    J0s = np.sum((J0s_tem[1:] + J0s_tem[:-1]) / 2 * dE)
    J0s_err = 0.0

    Sn0 = np.sum((Sn0_tem[1:] + Sn0_tem[:-1]) / 2 * dE)
    Sp0 = np.sum((Sp0_tem[1:] + Sp0_tem[:-1]) / 2 * dE)
    Phi_s = -Eth * np.log((ns_in * ni_b) / (nd * ni_e))

    return J0s, Uint, ps, ns_in, Phi_s, Sn0, Sp0, J0s_err

def ni_func(T, Ndop, dop_type):
    k = 8.617e-5
    Eth = k*T
    Egi = 1.206 - 2.73e-4*T
    ni = 1.541e15*T**1.712*np.exp(-Egi/(2*Eth))
    if Ndop > 1e13: # Avoid log of small numbers
      Delta_Eg = 4.2e-5*np.log(Ndop/1e14)**3
      BGN = np.exp(Delta_Eg/Eth)
      ni_eff = ni*np.sqrt(BGN)
    else:
      ni_eff = ni
    return ni_eff

def surfaceLifetime(n0, p0, n, p, Delta_n, Qfixi, T, Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk, dn, Dit_tot,sigma_n, sigma_p):
    params_ns_zero = (Qfixi, T, Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk, dn)
    ns_search_range = np.logspace(16, 20, 10000)
    fzero_values = ns_zero_func(ns_search_range, *params_ns_zero)
    fzero_abs = np.abs(fzero_values)
    ns_guess = lookup(fzero_abs, fzero_abs.min(), ns_search_range)
    ns_solved = fsolve(ns_zero_func, ns_guess, args=params_ns_zero)[0]

    X_j0s = (ns_solved, T, Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk, dn, 0) # Dit_tot_err is 0
    
    #Correct the parameters passed to J0sv2_func ###
    #Pass the globally defined, energy-dependent sigma_n and sigma_p arrays.
    params_j0s = (Dit_tot, sigma_n, sigma_p) # Corrected tuple

    J0s_results = J0sv2_func(X_j0s, *params_j0s)
    Uint = J0s_results[1]

    if Delta_n > 1e-10:
        S = Uint / Delta_n
    else:
        S = 0

    if S > 1e-10:
        tau_surface = W / (2 * S)
    else:
        tau_surface = np.inf

    return tau_surface


def intrinsicLifetime(n0, p0, n, p, Delta_n):
    """
    Calculates the intrinsic lifetime limit in silicon based on Auger and
    radiative recombination using the Richter model (parameterization).

    Args:
        n0, p0 (float): Equilibrium electron/hole concentrations (cm^-3).
        n, p (float): Non-equilibrium electron/hole concentrations (cm^-3).
        Delta_n (float): Excess carrier concentration (cm^-3).

    Returns:
        float: Intrinsic lifetime (s).
    """
    # Richter model parameters (moved from global scope for clarity)
    bmax = 1.0 # Unitless

    # Temperature dependent parameters using the constants defined above
    bmin = INTR_RMAX + (INTR_RMIN - INTR_RMAX) / (1 + (T / INTR_R1)**INTR_R2)
    b1 = INTR_SMAX + (INTR_SMIN - INTR_SMAX) / (1 + (T / INTR_S1)**INTR_S2)
    b3 = INTR_WMAX + (INTR_WMIN - INTR_WMAX) / (1 + (T / INTR_W1)**INTR_W2)

    # Low injection radiative recombination coefficient (temperature dependent)
    Blow = 10**(-9.6514 - 8.0525e-2 * T + 6.0269e-4 * T**2 - 2.294e-6 * T**3 + 4.3193e-9 * T**4 - 3.16154e-12 * T**5) # cm^3/s

    # Injection dependent radiative recombination coefficient
    Brel = bmin + (bmax - bmin) / (1 + ((n + p) / b1)**INTR_B2 + ((n + p) / b3)**INTR_B4)
    Brad = Brel * Blow # Effective radiative coefficient (cm^3/s)

    # Auger enhancement factors
    geeh = 1 + INTR_GEEH_FACTOR * (1 - np.tanh(n0 / INTR_N0EEH)**INTR_GEEH_EXP)
    gehh = 1 + INTR_GEHH_FACTOR * (1 - np.tanh(p0 / INTR_N0EHH)**INTR_GEHH_EXP)

    # Total intrinsic recombination rate (Auger + Radiative)
    # Note: The Delta_n^0.92 term's origin/units should be verified.
    Uintr = (n * p - ni_b**2) * (INTR_AUGER_EEH * geeh * n0 + INTR_AUGER_EHH * gehh * p0 + INTR_AUGER_XXX_COEFF * Delta_n**INTR_AUGER_XXX_EXP + Brad) # cm^-3 s^-1

    # Calculate intrinsic lifetime
    # Avoid division by zero
    if Uintr > 1e-10:
        tau_intr = Delta_n / Uintr
    else:
        tau_intr = np.inf

    return tau_intr


def taueff(dn_array, tau_array, Dit_0instance, Dit_bandedgeinstance, Qfixi, export=False):
    """
    Calculates the effective lifetime based on bulk (SRH, Intrinsic) and surface
    recombination mechanisms, comparing it to experimental data.
    NOTE: This function seems designed for fitting experimental data but is not
          called in the main script flow generating Figure 1.
          Dit_bandedgeinstance is unused.

    Args:
        dn_array (np.ndarray): Array of experimental excess carrier densities (cm^-3).
        tau_array (np.ndarray): Array of experimental lifetimes (s).
        Dit_0instance (float): Peak value of the Gaussian Dit distribution (cm^-2 eV^-1).
        Dit_bandedgeinstance: Unused parameter.
        Qfixi (float): Fixed interface charge density (Coulombs/cm^2).
        export (bool): If True, plots the results and saves data.

    Returns:
        tuple: Contains lists of squared errors, absolute differences, and differences
               between calculated and experimental log(lifetime).
    """
    # Energy array for Dit calculation (high resolution)
    E_fit = np.linspace(Ev, Ec, 100000) # Consider if 100k points are needed

    # Calculate the Gaussian Dit distribution based on the instance parameters
    params_g = (GAUSS_E0, Dit_0instance, GAUSS_SIGMA) # Using global constants
    Dit_g = Dig_func(E_fit, *params_g)

    # In this function context, Dit_tot is just the single Gaussian Dit
    Dit_tot_fit = Dit_g
    tau_eff_array = []
    tau_surface_array = []
    tau_intr_array = []
    tau_SRH_array = []
    squares = []
    absolutedifference = []
    difference = []
    # Loop through experimental injection levels
    for count, dn_exp_val in enumerate(dn_array):
        Delta_n = dn_exp_val
        Ndop = Ndop_bulk # Assuming bulk doping applies
        dop_type = dop_type_bulk # Assuming bulk doping type

        # Calculate equilibrium concentrations
        # Ensure Ndop is not zero if used in division
        if Ndop > 0:
            n0 = Ndop * (1 - dop_type) + ni_b**2 / Ndop * dop_type
            p0 = Ndop * dop_type + ni_b**2 / Ndop * (1 - dop_type)
        else:
            n0 = ni_b # Intrinsic case if Ndop is zero
            p0 = ni_b

        # Calculate non-equilibrium concentrations
        n = Delta_n + n0
        p = Delta_n + p0

        # Calculate lifetime components
        tau_surface = surfaceLifetime(n0, p0, n, p, Delta_n, Qfixi, T, Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk, dn_exp_val, Dit_tot_fit)
        tau_intr = intrinsicLifetime(n0, p0, n, p, Delta_n)

        # Calculate effective lifetime using Mathiessen's rule
        # Handle potential division by zero if any lifetime component is zero or infinite
        inv_tau_srh = 1 / tau_SRH if tau_SRH > 0 else 0
        inv_tau_intr = 1 / tau_intr if tau_intr > 0 else 0
        inv_tau_surf = 1 / tau_surface if tau_surface > 0 else 0
        inv_tau_eff_sum = inv_tau_srh + inv_tau_intr + inv_tau_surf

        if inv_tau_eff_sum > 1e-20: # Avoid division by zero
             tau_eff = 1 / inv_tau_eff_sum
        else:
             tau_eff = np.inf # Infinite lifetime if all components are zero

        # Store results
        tau_SRH_array.append(tau_SRH) # Constant bulk SRH lifetime
        tau_intr_array.append(tau_intr)
        tau_surface_array.append(tau_surface)
        tau_eff_array.append(tau_eff)
        squares.append((np.log(tau_eff)-np.log(tau_array[count]))**2)
        absolutedifference.append(np.abs((np.log(tau_eff)-np.log(tau_array[count]))))
        difference.append((np.log(tau_eff)-np.log(tau_array[count])))

    if export==True:
        # Create figure and axes for plotting
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Define plotting parameters
        currentSample = 0  # Default sample index
        symbols = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']
        colors_line = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors_marker = ['darkblue', 'darkred', 'darkgreen', 'darkorange', 'darkviolet', 'maroon', 'deeppink', 'dimgray', 'olive', 'darkcyan']
        label = ['Experimental Data', 'Fitted Data']
        
        # Plot experimental and fitted data
        ax.loglog(dn_array, np.multiply(tau_array, 1E3), symbols[currentSample], 
                 markersize=10, markerfacecolor=colors_line[currentSample], 
                 markeredgewidth=1.5, markeredgecolor=colors_marker[currentSample], 
                 label=label[currentSample])
        ax.loglog(dn_array, np.multiply(tau_eff_array, 1E3), colors_line[currentSample], linewidth=3.5)
        
        # Set plot properties
        ax.set_xlim([1e14, 1.1e16])
        ax.set_ylim(np.min(tau_array)*0.7*1E3, np.max(tau_array)*1.3*1E3)
        ax.set_xlabel('Carrier density $(cm^{-3})$')
        ax.set_ylabel('Minority carrier lifetime $(ms)$')
        ax.tick_params(axis='both', which='major', direction='in', length=8)
        ax.tick_params(axis='both', which='minor', direction='in', length=8)
        ax.legend(frameon=False, ncol=1)
        #ax[1].plot([currentSample],-qsamples[currentSample],symbols[currentSample], markersize=10, markerfacecolor=colors_line[currentSample], markeredgewidth=1.5, markeredgecolor=colors_marker[currentSample])
        #ax[1].tick_params(axis='both', which='major', direction='in', length=8)
        #ax[1].set_ylim([np.min(qsamples[currentSample])/3,np.max(qsamples[currentSample])*3])
        #ax[1].set_ylabel('Charge  $(cm^{-2})$')
        #ax[1].set_yscale('log')
        #ax[1].set_xlabel('Time (weeks)')
        #plt.tight_layout()
    #plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=4, title="Dit = "+"{:2e}".format(Dit_0instance)+"cm-2\nQ = "+"{:2e}".format(Qfixi/constants.e)+"cm-2",frameon=False)
    #plt.show()
    '''
    if export==True:
        fig1.savefig(filename+'_lifetime{}.png'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S')))
        fig2 = plt.figure()
        fig2.set_size_inches(6, 6)
        plt.semilogy(E,Dit_tot,'k',linewidth=2,label = 'D_{it}')
        plt.xlim([0, 1.12])
        plt.xlabel('Energy $(eV)$')
        plt.ylabel('Dit  $(cm^{-2})$')
        plt.tight_layout()
        plt.tick_params(axis='both', which='major', direction='in', length=8)
        plt.show()
        fig2.savefig(filename+'_Dit{}.png'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S')))
        df = pd.DataFrame(data=[dn_array,tau_array, tau_eff_array, tau_SRH_array, tau_surface_array, tau_intr_array]).T
        df.to_csv(filename+'fit'+'_{}.csv'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S')))
    '''
    return squares, absolutedifference, difference

def calculate_mean_square_error(params):
    Qfixinit, Dit_0g, Dit_bandedge = params
    Nq = 1
    ns_guess = np.zeros(Nq)
    ns = np.zeros_like(ns_guess)      
    Qfix = -Qfixinit * elementary_charge
    results = taueff(dn_exp,tau_exp,Dit_0g,Dit_bandedge,Qfix,export=False)
    squares = results[0]
    rmse = (sum(squares)/len(squares))**0.5
    print('The RMSE is ' + str(rmse))
    return rmse

def calculate_mean_absolute_error(params):
    Qfixi, Dit_0g, Dit_bandedge = params
    Nq = 1
    ns_guess = np.zeros(Nq)
    ns = np.zeros_like(ns_guess)      
    Qfix = -Qfixi * elementary_charge
    results = taueff(dn_exp,tau_exp,Dit_0g,Dit_bandedge,Qfix,export=False)
    absolutedifference = results[1]
    mae = (sum(absolutedifference)/len(absolutedifference))
    print('The MAE is ' + str(mae))
    return mae

def calculate_mean_bias_error(params):
    Qfixi, Dit_0g, Dit_bandedge = params
    Nq = 1
    ns_guess = np.zeros(Nq)
    ns = np.zeros_like(ns_guess)      
    Qfix = -Qfixi * elementary_charge
    results = taueff(dn_exp,tau_exp,Dit_0g,Dit_bandedge,Qfix,export=False)
    difference = results[2]
    mbe = (sum(difference)/len(difference))
    print('The MBE is ' + str(mbe))
    return mbe

# ============================================================================
# MAIN SCRIPT EXECUTION
# ============================================================================

# Calculate derived values
Eth = kB*T
ni_b = ni_func(T, Ndop_bulk, dop_type_bulk) 
Ef = (Ec+Ev)/2 - Eth*np.log(Ndop_bulk/ni_b)

# Create energy array for calculations
E_sigma = create_energy_array()

#Replace constant cross-section with Gaussian model ###
# --- Define Capture Cross-Sections using the new Gaussian Model ---
# We call our new function to generate the energy-dependent arrays.
print("Calculating energy-dependent capture cross-sections using Gaussian model...")
sigma_n = calculate_gaussian_sigma(E_sigma, SIGMA0_N, A_N, E0_N)
sigma_p = calculate_gaussian_sigma(E_sigma, SIGMA0_P, A_P, E0_P)

###Debug
print(f"DEBUG STEP 2: Peak of calculated sigma_n array: {np.max(sigma_n):.2e}")
print(f"DEBUG STEP 2: Peak of calculated sigma_p array: {np.max(sigma_p):.2e}")

# Carrier density array for calculations
dn_array = np.logspace(14, 17)
Qtotarray = np.logspace(6, 13, 2)

# Load experimental data
try:
    exp_data = pd.read_excel('data/L2_pre_ini_G.xlsx', sheet_name='RawData')
    dn_exp = exp_data["Minority Carrier Density"].to_numpy()
    tau_exp = exp_data["Tau (sec)"].to_numpy()
    valid_mask = (dn_exp > 0) & (tau_exp > 0) & np.isfinite(dn_exp) & np.isfinite(tau_exp)
    dn_exp = dn_exp[valid_mask]
    tau_exp = tau_exp[valid_mask]
    print(f"Loaded experimental data: {len(dn_exp)} data points")
except (FileNotFoundError, KeyError) as e:
    print(f"Warning: Could not load experimental data: {e}. Using default arrays.")
    dn_exp = np.logspace(14, 16, 10)
    tau_exp = np.ones_like(dn_exp) * 1e-3

colors = pl.cm.hsv(np.linspace(0,1,12))

# This module contains the core functions and parameters for the lifetime simulation.

#Add a verification plot to show the new cross-sections
def plot_gaussian_cross_sections():
    """
    Generates a plot to visualize the implemented Gaussian capture cross-sections.
    """
    print("\nGenerating plot for Gaussian Capture Cross-Sections...")
    # Set Chinese font for the plot
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Energy relative to mid-gap for plotting
    E_midgap = (Ec + Ev) / 2
    E_plot = E_sigma - E_midgap

    # Plotting the data
    ax.plot(E_plot, sigma_n, color='blue', linewidth=2, label=r'$\sigma_n$ electron cross section')
    ax.plot(E_plot, sigma_p, color='red', linewidth=2, label=r'$\sigma_p$ hole cross section')
    ax.set_yscale('log')
    ax.set_ylim(1e-22, 1e-13) 
    ax.set_xlim(-0.6, 1.2) 
    
    # Formatting
    ax.set_title('Gaussian fitting', fontsize=16)
    ax.set_xlabel('E - E_midgap [eV]', fontsize=12)
    ax.set_ylabel('capture cross section [cm²]', fontsize=12)
    ax.legend(frameon=False)
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', direction='in', length=6)
    ax.tick_params(axis='both', which='minor', direction='in', length=4)
    # Set x-axis limits for better visualization
    ax.set_xlim([E_plot.min(), E_plot.max()])
    plt.tight_layout()
    
    data_folder = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(data_folder, 'capture cross section.png'))
    #plt.show()

# --- Run the verification plot ---
plot_gaussian_cross_sections()