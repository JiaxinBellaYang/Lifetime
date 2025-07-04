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
# Import figure generation functions
from figure_generation import generate_figure1, generate_figure2, generate_figure3

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
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['mathtext.fontset'] = "dejavuserif"

# Array dimensions
ENERGY_POINTS = 10000  # Number of points for energy arrays

# Physical constants
T = 300.0  # Temperature in Kelvin
Ec = 1.12  # Conduction band energy (eV)
Ev = 0.0   # Valence band energy (eV)
Eg = 1.1246  # Band gap (eV)
kT = 0.025692223  # Thermal energy (eV)
# kB = constants.k/constants.e  # Boltzmann constant in eV/K - Use imported constants
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
# eps_Si = eps_rel * constants.epsilon_0 * 1e-2  # Silicon permittivity in F/cm - Use imported constant
eps_Si = eps_rel * epsilon_0 * 1e-2  # Silicon permittivity in F/cm

# Thermal velocities
vth_n = 2.046e7 * (T/300.0)**0.5  # Electron thermal velocity (cm/s)
vth_p = 1.688e7 * (T/300.0)**0.5  # Hole thermal velocity (cm/s)

# Dangling bond parameters
s0n = 1E-17  # Neutral electron capture cross section (cm^2)
s0p = 1E-17  # Neutral hole capture cross section (cm^2)
splusn = 1E-16  # Positive electron capture cross section (cm^2)
sminusp = 1E-16  # Negative hole capture cross section (cm^2)
# Gaussian Defect Distribution Parameters (used in Dig_func and J0sv2_func)
GAUSS_E0 = 0.56  # Centre of Gaussian Dit distribution (eV)
GAUSS_SIGMA = 0.18  # Width of Gaussian Dit distribution (eV)
GAUSS_U = 0.15  # Correlation energy for amphoteric defects (eV) - Note: U is defined but not explicitly used in the main calculation loops.

# Hardcoded Dit parameters (used in J0sv2_func, consider making configurable)
VALENCE_DIT_E0 = 0.0
VALENCE_DIT_MAX = 4E10 # cm^-2 eV^-1
VALENCE_DIT_SIGMA = 0.18
CONDUCTION_DIT_E0 = 1.68 # eV - Note: This is outside the typical Si bandgap
CONDUCTION_DIT_MAX = 8E12 # cm^-2 eV^-1
CONDUCTION_DIT_SIGMA = 0.195

# Capture Cross Sections (used in J0sv2_func and main script section)
# Note: These are currently set but some are overwritten later. Clarify intended usage.
SP0_DEFAULT = 4.5E-18 # cm^2
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
tau_SRH = 2E-3  # Shockley-Read-Hall lifetime (s)

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

def create_energy_array(start=Ev, end=Ec, points=ENERGY_POINTS):
    """Create a linearly spaced energy array with the specified parameters."""
    return np.linspace(start, end, points)

def log_progress(current, total, item_name, value=None):
    """Log progress of an iteration."""
    if value is not None:
        print(f"  Processing {item_name} {current+1}/{total}: {value:.2e}")
    else:
        print(f"Processing {item_name} {current+1}/{total}")

def find_lifetime(dnarray, lifetimearray, valueDeltaN):
    dnarray = np.asarray(dnarray)
    idx = (np.abs(dnarray - valueDeltaN)).argmin()
    return lifetimearray[idx]

def Dig_func(E, E0_1, Dit_0g1, sigma_1, E0_2, Dit_0g2, sigma_2):
    """
    Calculates the double Gaussian distribution for the Density of Interface Traps (Dit).

    Args:
        E (np.ndarray): Energy array (eV).
        E0_1 (float): Center energy of the first Gaussian distribution (eV).
        Dit_0g1 (float): Peak value of the first Dit distribution (cm^-2 eV^-1).
        sigma_1 (float): Standard deviation (width) of the first Gaussian distribution (eV).
        E0_2 (float): Center energy of the second Gaussian distribution (eV).
        Dit_0g2 (float): Peak value of the second Dit distribution (cm^-2 eV^-1).
        sigma_2 (float): Standard deviation (width) of the second Gaussian distribution (eV).

    Returns:
        np.ndarray: Combined Dit distribution as a function of energy (cm^-2 eV^-1).
    """
    # Calculate the two Gaussian distributions
    Dit_g1 = Dit_0g1 * np.exp(-((E - E0_1) / sigma_1)**2 / 2)
    Dit_g2 = Dit_0g2 * np.exp(-((E - E0_2) / sigma_2)**2 / 2)

    # Combine the two distributions
    Dit_g = Dit_g1 + Dit_g2

    return Dit_g

def ns_zero_func(ns, *params):
    """
    Zero-finding function to determine the surface electron concentration (ns).
    This function represents the charge neutrality condition at the surface.
    It needs to be solved for ns such that fzero = 0.

    Args:
        ns (float): Surface electron concentration guess (cm^-3).
        *params: Tuple containing:
            Q (float): Fixed surface charge density (Coulombs/cm^2).
            T (float): Temperature (K).
            Ndop_emitter (float): Emitter doping concentration (cm^-3).
            Ndop_bulk (float): Bulk doping concentration (cm^-3).
            dop_type_emitter (float): Emitter doping type (1=n, 0=p).
            dop_type_bulk (float): Bulk doping type (1=n, 0=p).
            Δn (float): Excess carrier concentration in the bulk (cm^-3).

    Returns:
        float: Value of the charge neutrality equation. Should be zero at the correct ns.
    """
    Q, T, Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk, Δn = params

    # Calculate effective surface doping concentration
    Ndop_surface_n = dop_type_emitter * Ndop_emitter + dop_type_bulk * Ndop_bulk
    Ndop_surface_p = (1-dop_type_emitter)*Ndop_emitter + (1-dop_type_bulk)*Ndop_bulk
    Ndop_surface = np.abs(Ndop_surface_n - Ndop_surface_p)

    Eth = kB*T
    # equilibrium coincentrations in the bulk(_d) and the the surface (_s)  
    ni_b = ni_func(T, Ndop_bulk, dop_type_bulk)    
    ni_e = ni_func(T, Ndop_surface, dop_type_emitter)  
    
    nd0 = dop_type_bulk*Ndop_bulk + (1-dop_type_bulk)*ni_b**2/Ndop_bulk
    pd0 = (1-dop_type_bulk)*Ndop_bulk + dop_type_bulk*ni_b**2/Ndop_bulk

    # carrier concentrations in the bulk
    pd = pd0 + Δn
    nd = nd0 + Δn
     
    ps = (ni_e**2/ni_b**2)*pd*nd/ns    
    Phi_s = -Eth*np.log((ns*ni_b)/(nd*ni_e))
    
    # Charge neutrality equation:
    # (Surface holes - Bulk holes) + (Surface electrons - Bulk electrons)
    # + (Charge due to surface doping * Surface Potential / Thermal Voltage)
    # - (Charge induced by fixed charge Q) = 0
    # Note: The Q^2 term seems unusual for charge neutrality. Typically, it involves Q directly.
    # Check the derivation of this term. It might represent something else, like field effect.
    fzero = ps - pd + ns - nd + Ndop_surface * Phi_s / Eth - Q**2 / (2 * elementary_charge * eps_Si * Eth) # Use imported constant
    return fzero

def lookup2(Y, yval, X=[]): # Note: Using mutable default argument X=[] is discouraged.

  Y=np.array(Y)
  index = np.argmin(abs(Y-yval))
  if X==[]:
      return index
  else:
      return X[index]

def lookup(Y, yval, X):
    Y = np.array(Y)
    index = np.argmin(abs(Y-yval))
    if len(X) == 0:  # Check if X is empty
        return index
    else:
        return X[index]

def J0sv2_func(X, *params):
    """
    Calculates surface recombination parameters based on the amphoteric defect model
    and potentially hardcoded band tail states.

    Args:
        X (tuple): Input parameters containing:
            ns (float): Surface electron concentration (cm^-3).
            T (float): Temperature (K).
            Ndop_emitter (float): Emitter doping concentration (cm^-3).
            Ndop_bulk (float): Bulk doping concentration (cm^-3).
            dop_type_emitter (float): Emitter doping type (1=n, 0=p).
            dop_type_bulk (float): Bulk doping type (1=n, 0=p).
            dn (float): Excess carrier concentration (cm^-3).
            Nit_err: Error in Dit (currently unused effectively).
        *params: Tuple containing:
            Dit_E (np.ndarray): Interface defect density distribution vs Energy (cm^-2 eV^-1).
                                This seems to be the primary Dit input used in calculations.
            sigma_n1 (np.ndarray): Electron capture cross-section for state 1 vs E (cm^2).
            sigma_p1 (np.ndarray): Hole capture cross-section for state 1 vs E (cm^2).
            sigma_n2 (np.ndarray): Electron capture cross-section for state 2 vs E (cm^2).
            sigma_p2 (np.ndarray): Hole capture cross-section for state 2 vs E (cm^2).

    Returns:
        tuple: Contains calculated surface recombination parameters:
            J0s (float): Surface saturation current density (A/cm^2).
            Uint (float): Integrated surface recombination rate (cm^-2 s^-1).
            ps (float): Surface hole concentration (cm^-3).
            ns_out (float): Surface electron concentration (same as input ns) (cm^-3).
            Phi_s (float): Surface potential (V).
            Sn0 (float): Effective electron surface recombination velocity parameter (cm/s * cm^2?). Units need check.
            Sp0 (float): Effective hole surface recombination velocity parameter (cm/s * cm^2?). Units need check.
            J0s_err (float): Error in J0s calculation (currently seems incorrect).
    """
    ns_in, T, Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk, dn, Nit_err = X
    Dit_E = params[0] # Renamed from Nit for clarity - represents Dit(E) distribution
    sigma_n1 = params[1]
    sigma_p1 = params[2]
    sigma_n2 = params[3]
    sigma_p2 = params[4]

    Eth = kB * T
    local_vth_n = 2.046e7*np.sqrt(T/300)
    local_vth_p = 1.688e7*np.sqrt(T/300)        
    
    Ndop_surface_n = dop_type_emitter*Ndop_emitter + dop_type_bulk*Ndop_bulk
    Ndop_surface_p = (1-dop_type_emitter)*Ndop_emitter + (1-dop_type_bulk)*Ndop_bulk
    Ndop_surface = np.abs(Ndop_surface_n - Ndop_surface_p)  
    
    # equilibrium coincentrations in the bulk(_d) and the the surface (_s)    
    ni_b = ni_func(T, Ndop_bulk, dop_type_bulk)    
    ni_e = ni_func(T, Ndop_surface, dop_type_emitter)    

    nd0 = dop_type_bulk*Ndop_bulk + (1-dop_type_bulk)*ni_b**2/Ndop_bulk
    pd0 = (1-dop_type_bulk)*Ndop_bulk + dop_type_bulk*ni_b**2/Ndop_bulk  

    pd = pd0 + dn
    nd = nd0 + dn
    ps = (ni_e**2 / ni_b**2) * pd * nd / ns_in # Corrected variable name ns -> ns_in

###### energy range Ev to Ec
    n = nd0 + dn
    p = pd0 + dn
    
    # Use helper function to create energy arrays
    E = create_energy_array()
    dE = E[1:] - E[:-1]

    # --- Use Input Capture Cross-Sections for SRH Model ---
    # Calculate capture rates using the input sigma parameters
    cn1 = sigma_n1 * vth_n  # Electron capture rate for state 1
    cp1 = sigma_p1 * vth_p  # Hole capture rate for state 1
    cn2 = sigma_n2 * vth_n  # Electron capture rate for state 2
    cp2 = sigma_p2 * vth_p  # Hole capture rate for state 2

    # SRH parameters n1, p1, etc.
    n11 = NC * np.exp(-E / kT)
    p11 = NV * np.exp(-(Eg - E) / kT)

    n12 = NC * np.exp(-(E + GAUSS_U) / kT) # Using GAUSS_U constant
    p12 = NV * np.exp(-(Eg - (E + GAUSS_U)) / kT) # Using GAUSS_U constant

    # Calculate SRH recombination rate using input capture cross sections
    # Standard SRH recombination formula for two charge states
    R1 = Dit_E * (ps * ns_in - ni_e**2) / ((ns_in + n11) / cp1 + (ps + p11) / cn1)
    R2 = Dit_E * (ps * ns_in - ni_e**2) / ((ns_in + n12) / cp2 + (ps + p12) / cn2)
    
    # Total recombination rate per unit energy
    UofE = R1 + R2
    
    # Integrate recombination rate over energy (Trapezoidal rule)
    Uint = np.sum((UofE[1:] + UofE[:-1]) / 2 * dE)

    # Capture cross-section ratios
    k1 = sigma_n1 / sigma_p1
    k2 = sigma_n2 / sigma_p2

    # Occupancy factors (alpha) and defect state populations (Nsplus1, Ns, Nsplus2)
    # These calculations seem related to the amphoteric model and use Dit_E (Nit).
    alpha1 = (k1 * n11 + p) / (k1 * n + p11)
    alpha2 = (k2 * n12 + p) / (k2 * n + p12)

    # Avoid division by zero or handle cases where denominators are zero
    denominator = alpha1 + 1 + 1 / alpha2
    Nsplus1 = np.divide(Dit_E, denominator, out=np.zeros_like(Dit_E), where=denominator!=0)
    # Nsplus1 = Dit_E / (alpha1 + 1 + 1 / alpha2) # Original, potential division by zero

    Ns = alpha1 * Nsplus1
    Nsplus2 = Nsplus1 / alpha2 # Potential division by zero if alpha2 is zero

    # Total defects in state 1 and state 2 configurations
    N1 = Nsplus1 + Ns
    N2 = Nsplus1 + Nsplus2

    # Effective SRV parameters per unit energy (check units)
    Sn0_tem1 = N1 * sigma_n1 * vth_n
    Sp0_tem1 = N1 * sigma_p1 * vth_p

    Sn0_tem2 = N2 * sigma_n2 * vth_n
    Sp0_tem2 = N2 * sigma_p2 * vth_p

    # Saturation current density per unit energy
    denominator1 = (ps + p11) / Sn0_tem1 + (ns_in + n11) / Sp0_tem1
    J0s_tem1 = np.divide(elementary_charge * ni_e**2, denominator1, out=np.zeros_like(denominator1), where=denominator1!=0) # Use imported constant
    # J0s_tem1 = constants.e * ni_e**2 / ((ps + p11) / Sn0_tem1 + (ns_in + n11) / Sp0_tem1) # Original

    denominator2 = (ps + p12) / Sn0_tem2 + (ns_in + n12) / Sp0_tem2
    J0s_tem2 = np.divide(elementary_charge * ni_e**2, denominator2, out=np.zeros_like(denominator2), where=denominator2!=0) # Use imported constant
    # J0s_tem2 = constants.e * ni_e**2 / ((ps + p12) / Sn0_tem2 + (ns_in + n12) / Sp0_tem2) # Original
    J0s_tem = J0s_tem1 + J0s_tem2

    # Integrate terms over energy (Trapezoidal rule)
    J0s = np.sum((J0s_tem[1:] + J0s_tem[:-1]) / 2 * dE) # Integrated saturation current
    J0s_err = 0.0 # Removed incorrect error calculation

    # Integrate effective SRV parameters (check units)
    Sn0 = np.sum((Sn0_tem1[1:] + Sn0_tem1[:-1]) / 2 * dE)
    Sp0 = np.sum((Sp0_tem1[1:] + Sp0_tem1[:-1]) / 2 * dE)

    # Calculate surface potential
    Phi_s = -Eth * np.log((ns_in * ni_b) / (nd * ni_e))

    return J0s, Uint, ps, ns_in, Phi_s, Sn0, Sp0, J0s_err # Return ns_in for consistency

def ni_func(T, Ndop, dop_type):
    """
    Calculates the effective intrinsic carrier concentration (ni_eff) in silicon,
    considering band gap narrowing (BGN) effects.

    Args:
        T (float): Temperature (K).
        Ndop (float): Doping concentration (cm^-3).
        dop_type (float): Doping type (1=n, 0=p) - currently unused in calculation.

    Returns:
        float: Effective intrinsic carrier concentration (cm^-3).
    """
    k = 8.617e-5
    Eth = k*T
    Egi = 1.206 - 2.73e-4*T
    mdcm0 = -4.609e-10*T**3 + 6.753e-7*T**2 -1.312e-5*T + 1.094
    mdvm0 = 2.525e-9*T**3 - 4.689e-6*T**2 + 3.376e-3*T + 4.326e-1
    ni = 1.541e15*T**1.712*np.exp(-Egi/(2*Eth))
    Delta_Eg = 4.2e-5*np.log(Ndop/1e14)**3
    BGN = np.exp(Delta_Eg/Eth)
    ni_eff = ni*np.sqrt(BGN)
    
    return ni_eff

def surfaceLifetime(n0, p0, n, p, Delta_n, Qfixi, T, Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk, dn, Dit_tot):
    """
    Calculates the effective surface lifetime limited by interface recombination.

    Args:
        n0, p0 (float): Equilibrium electron/hole concentrations in bulk (cm^-3).
        n, p (float): Non-equilibrium electron/hole concentrations in bulk (cm^-3).
        Delta_n (float): Excess carrier concentration (cm^-3).
        Qfixi (float): Fixed interface charge density (Coulombs/cm^2).
        T (float): Temperature (K).
        Ndop_emitter, Ndop_bulk (float): Doping concentrations (cm^-3).
        dop_type_emitter, dop_type_bulk (float): Doping types (1=n, 0=p).
        dn (float): Excess carrier concentration (redundant, same as Delta_n).
        Dit_tot (np.ndarray): Interface defect density distribution vs Energy (cm^-2 eV^-1).

    Returns:
        float: Effective surface lifetime (s).
    """
    # --- 1. Find surface electron concentration (ns) using charge neutrality ---
    params_ns_zero = (Qfixi, T, Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk, dn)
    # Define search range for ns. Adjust if necessary.
    ns_search_range = np.logspace(16, 20, 10000) # Search range for ns
    fzero_values = ns_zero_func(ns_search_range, *params_ns_zero)
    fzero_abs = np.abs(fzero_values)

    # Find the ns value in the search range that minimizes the function (initial guess)
    ns_guess = lookup(fzero_abs, fzero_abs.min(), ns_search_range)

    # Use fsolve to find the precise root (ns)
    ns_solved = fsolve(ns_zero_func, ns_guess, args=params_ns_zero)[0] # fsolve returns array

    # --- 2. Calculate surface recombination rate (Uint) using J0sv2_func ---
    # Prepare inputs for J0sv2_func
    # Note: Dit_tot_err is currently 0, making J0s_err calculation potentially meaningless
    X_j0s = (ns_solved, T, Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk, dn, Dit_tot_err)
    # Pass the Dit distribution and capture cross-sections
    params_j0s = (Dit_tot, sigma_n1, sigma_p1, sigma_n2, sigma_p2)

    # Call J0sv2_func and get the integrated recombination rate (Uint)
    J0s_results = J0sv2_func(X_j0s, *params_j0s)
    Uint = J0s_results[1] # Index 1 corresponds to Uint

    # --- 3. Calculate effective surface recombination velocity (S) and lifetime ---
    # Avoid division by zero if Delta_n is very small
    if Delta_n > 1e-10: # Threshold to avoid division by zero
        S = Uint / Delta_n # Effective surface recombination velocity (cm/s)
    else:
        S = 0 # Or handle as appropriate (e.g., infinity if Uint is non-zero)

    # Calculate surface lifetime using the standard formula for symmetric surfaces
    # Avoid division by zero if S is very small
    if S > 1e-10:
        tau_surface = W / (2 * S)
    else:
        tau_surface = np.inf # Effectively infinite lifetime if S is zero

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
        ax[0].loglog(dn_array,np.multiply(tau_array,1E3),symbols[currentSample], markersize=10, markerfacecolor=colors_line[currentSample], markeredgewidth=1.5, markeredgecolor=colors_marker[currentSample], label = label[currentSample])
        ax[0].loglog(dn_array,np.multiply(tau_eff_array,1E3),colors_line[currentSample],linewidth=3.5)
        #plt.semilogx(dn_array,np.multiply(tau_intr_array,1E3),'plum',linestyle=':',linewidth=2,label = 'Intrinsic')
        #plt.semilogx(dn_array,np.multiply(tau_surface_array,1E3),'salmon',linestyle='--',linewidth=2,label = 'Interface')
        #plt.semilogx(dn_array,np.multiply(tau_SRH_array,1E3),'pink',label = r' $\tau_{SRH}$')
        ax[0].set_xlim([1e14, 1.1e16])
        ax[0].set_ylim(np.min(tau_array)*0.7*1E3,np.max(tau_array)*1.3*1E3)
        ax[0].set_xlabel('Carrier density $(cm^{-3})$')
        ax[0].set_ylabel('Minority carrier lifetime $(ms)$')
        ax[0].tick_params(axis='both', which='major', direction='in', length=8)
        ax[0].tick_params(axis='both', which='minor', direction='in', length=8)
        ax[0].legend(frameon=False, ncol=1)
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
    Qfix = -Qfixinit * constants.e
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
    Qfix = -Qfixi * constants.e
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
    Qfix = -Qfixi * constants.e
    results = taueff(dn_exp,tau_exp,Dit_0g,Dit_bandedge,Qfix,export=False)
    difference = results[2]
    mbe = (sum(difference)/len(difference))
    print('The MBE is ' + str(mbe))
    return mbe


# Calculate derived values
Eth = kB*T
ni_b = ni_func(T, Ndop_bulk, dop_type_bulk) 
Ef = (Ec+Ev)/2 - Eth*np.log(Ndop_bulk/ni_b)

# Create energy arrays
E = create_energy_array()
E_sigma = create_energy_array()

# --- Define Capture Cross-Sections for the Simulation ---
# These overwrite the defaults defined earlier. Clarify which set is intended.
sp0_sim = SP0_DEFAULT # 3E-15 cm^2
sn0_sim = SN0_DEFAULT # sp0_sim
spminus_sim = SPMINUS_DEFAULT # 10 * sp0_sim
snplus_sim = SNPLUS_DEFAULT # 10 * sn0_sim

# Define the energy-dependent sigma arrays used in J0sv2_func
# Currently assumes constant cross-sections across energy.
sigma_p1 = E_sigma * 0 + sp0_sim    # Hole capture for state 1 (+)
sigma_n1 = E_sigma * 0 + snplus_sim # Electron capture for state 1 (0) -> Should this be sn0_sim? Check model.
sigma_p2 = E_sigma * 0 + spminus_sim# Hole capture for state 2 (0) -> Should this be sp0_sim? Check model.
sigma_n2 = E_sigma * 0 + sn0_sim    # Electron capture for state 2 (-)
# Verify the mapping of sp0, sn0, spminus, snplus to sigma_p1, sigma_n1, sigma_p2, sigma_n2 based on the amphoteric model being used.

# Carrier density array for calculations
dn_array = np.logspace(14, 17)
Qtotarray = np.logspace(6, 13, 2)

# Load experimental data for fitting functions
try:
    # Load experimental lifetime data
    exp_data = pd.read_excel('data/L2_pre_ini_G.xlsx', sheet_name='RawData')
    dn_exp = exp_data["Minority Carrier Density"].to_numpy()
    tau_exp = exp_data["Tau (sec)"].to_numpy()
    
    # Remove any invalid data points
    valid_mask = (dn_exp > 0) & (tau_exp > 0) & np.isfinite(dn_exp) & np.isfinite(tau_exp)
    dn_exp = dn_exp[valid_mask]
    tau_exp = tau_exp[valid_mask]
    
    print(f"Loaded experimental data: {len(dn_exp)} data points")
    print(f"dn_exp range: {dn_exp.min():.2e} - {dn_exp.max():.2e} cm^-3")
    print(f"tau_exp range: {tau_exp.min():.2e} - {tau_exp.max():.2e} s")
    
except (FileNotFoundError, KeyError) as e:
    print(f"Warning: Could not load experimental data: {e}")
    print("Using default arrays for dn_exp and tau_exp")
    # Provide default values if file loading fails
    dn_exp = np.logspace(14, 16, 10)  # Default carrier density array
    tau_exp = np.ones_like(dn_exp) * 1e-3  # Default lifetime array (1 ms)

colors = pl.cm.hsv(np.linspace(0,1,12))

# This module contains the core functions and parameters for the lifetime simulation.
# The figure generation has been moved to figure_generation.py and can be run from main_runner.py
