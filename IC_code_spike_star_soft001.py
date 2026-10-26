import os
import h5py
import numpy as np
import numpy.random as rd

# NumPy 2.x removed the long-standing np.trapz alias (renamed to np.trapezoid).
# The cluster venv ships NumPy 2.x, so provide a back-compat shim to keep this
# script working on both 1.x and 2.x without touching every call site.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad, cumulative_trapezoid
import conversions as co
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.concentration_models import ConcentrationDiemerJoyce

util = LensCosmo()
concentration_model = ConcentrationDiemerJoyce(util.cosmo.astropy, scatter=False)

import time
import warnings
from scipy.integrate import IntegrationWarning
warnings.simplefilter('ignore', IntegrationWarning)
import matplotlib
matplotlib.use("Agg")  # headless-safe: this script must also run unattended on a supercomputer
import matplotlib.pyplot as plt


###################
# Dark-matter "spike" + central black hole initial conditions.
#
# CENTRAL-STAR VARIANT: the BH is written as a separate "star" (Step#0 star::* attributes),
# NOT as particle 0, for the central-star nbody propagator (BH handled outside the tree via a
# direct softened force -- see SPHEXA_NBODY.md sec 9). Requires a sphexa built with
# -DSPH_EXA_WITH_DISKS=ON. NON-SOFTCUT spike (inner cutoff at the ISCO).
#
# SOFTENING EXPERIMENT (soft001): the softening is the mean interparticle separation (MIS) of
# the particles INSIDE 0.01 pc, instead of the default MIS averaged over the 1 pc sphere. This
# resolves the inner cusp at its local spacing -- ~19x smaller softening (and inner_size) than
# the default, hence ~80x smaller timestep (dt ~ inner_size^1.5) and ~80x more steps for the
# same physical time. The Eddington DF is rebuilt self-consistently with this softening
# (softening_BH_DM_pair feeds the softened PsiBH), so the IC stays an equilibrium. Note the
# outer halo is now UNDER-softened relative to its local spacing (more two-body noise there).
# Distinct output_file so it cannot clobber the default star IC.
#
# Density model: a Gondolo & Silk (1999) adiabatic spike. For an NFW host with inner
# cusp slope gamma=1, adiabatic growth of the BH sharpens the cusp to slope
# gamma_sp = (9 - 2*gamma) / (4 - gamma) = 7/3 inside a break radius, matching back onto
# the ambient gamma=1 (rho ~ r^-1) NFW cusp further out, then an exponential cutoff.
# The spike's normalization is fixed by requiring its enclosed mass at 1 pc to equal the
# enclosed mass of the underlying (unmodified) reference NFW halo at 1 pc.

N_1pc = 1e7          # number of DM particles inside 1 pc (mass-normalization anchor)
Ngas = 0

# Particles are sampled and written to disk in chunks of this size rather than as one
# monolithic array, so peak memory stays bounded regardless of the total particle count
# (needed once N_1pc gets large enough that all sampled arrays together would no longer
# fit on a single node -- e.g. at N_1pc=1e10, N_halo~2.3e10, and a single float64 array
# of that length alone is ~183 GB). 2e8 keeps each chunk's handful of float64 arrays at
# a few GB each, comfortably below any single node's memory.
CHUNK_SIZE = int(1e7)

# Multiplies the mean-interparticle-separation softening below (diagnostic knob for
# testing how the settled post-relaxation energy offset depends on softening length;
# 1.0 reproduces the plain mean-separation formula).
softening_multiplier = 1.0

co.G = 43007.10573170632  #  kpc (km/s)^2 / Msun * 10**10 Msun

gas_frac = 0.0
gas_R0   = 0.1
Lambda   = 0.0    # nuclear DM spikes are modeled as dispersion-supported/isotropic, not
                   # rotating like a cosmological halo -- the MMW spin-parameter formalism
                   # below is calibrated against a halo's r200/M200 and is not meaningful at
                   # the ~pc scale of the spike, so no net rotation is imposed by default.
S_omega  = 1.0     # 0.0 for rigid body rotation (only relevant if Lambda != 0)

# Sampling
seed     = 42

pc_to_kpc = 1.0e-3

# ---- Reference NFW halo (unchanged from the pure-DM merger run) ----
NFW_M200 = 100                  # 10^10 Msun
NFW_c    = 10.0
ref_rhos, ref_rs, ref_r200 = util.NFW_params_physical(NFW_M200*1e10, NFW_c, 0)  # Msun/kpc^3, kpc, kpc

def mass_NFW_enclosed(rs, rhos, rmax):
    """Enclosed NFW mass, safe against catastrophic cancellation at rmax << rs."""
    x = rmax / rs
    if x < 1e-4:
        f = 0.5*x**2 - (2.0/3.0)*x**3 + 0.75*x**4   # Taylor expansion of log1p(x) - x/(1+x)
    else:
        f = np.log1p(x) - x/(1.0 + x)
    return 4*np.pi*rhos*rs**3*f

# ---- Black hole ----
M_BH    = 1e6 / 1e10             # 10^10 Msun
c_light = 299792.458             # km/s
r_isco  = 6.0 * co.G * M_BH / c_light**2   # Schwarzschild ISCO; don't sample DM inside this

# ---- Spike break radii (Gondolo-Silk) ----
gamma_sp      = 7.0 / 3.0         # adiabatic spike slope for an NFW (gamma=1) host
eta_spike     = 4.0                # transition sharpness between gamma_sp and the gamma=1 cusp
r_spike_break = 0.1 * pc_to_kpc    # 0.1 pc: spike -> NFW-like cusp transition
r_nfw_break   = 1.0 * pc_to_kpc    # 1.0 pc: mass-matching radius / onset of exponential decay
r_decay_scale = 0.4 * pc_to_kpc    # e-folding length of the outer exponential cutoff
cutoff_width  = 1.0 * pc_to_kpc    # smoothing width of the decay's onset (see below)
# r_decay_scale/cutoff_width are not free/cosmetic choices: the profile's ambient outer
# slope is exactly rho~r^-1, which in this BH-dominated (~Keplerian) potential is the
# *marginal* case of Eddington's convexity requirement (d2rho/dPsi2 = rho''(Psi) >= 0,
# needed for a non-negative isotropic DF to exist at all -- for a pure power law
# rho~r^-n this reduces to n(n-1)>=0, so n=1 gives exactly zero curvature to spare).
# Turning on the exponential decay too abruptly -- however smoothly, any cutoff_width
# comparable to or smaller than r_decay_scale -- drives d2rho/dPsi2 negative right in the
# transition, i.e. no non-negative isotropic DF exists for a sharper cutoff here. This
# was verified numerically (finite-difference cross-checked) while developing this
# script. This (0.4 pc, 1.0 pc) choice keeps d2rho/dPsi2 positive everywhere (checked on
# a fine grid) while still suppressing the density to <1% of its 1pc value by 3 pc and
# <0.1% by 4 pc.

NFW_M1pc = mass_NFW_enclosed(ref_rs, ref_rhos, r_nfw_break) / 1e10  # 10^10 Msun

def _smooth_excess(r):
    """Smooth (C-infinity) approximation to max(r - r_nfw_break, 0), via a softplus of
    width cutoff_width. A hard max(...,0) onset (as in exp(-max(r-r_nfw_break,0)/decay))
    keeps the density C0 but gives it a genuine kink (jump in d(ln rho)/dr) exactly at
    r_nfw_break -- harmless for the original file's finite-difference Eddington DF, but
    for the much steeper spike profile here it produces a real, large, localized jump in
    d2(rho)/dPsi2 right where the profile enters its decay, corrupting the DF. The
    softplus reduces to the exact same asymptotics (no decay well inside r_nfw_break, the
    requested exponential decay well outside it) while being smooth everywhere, and (see
    the note above) wide enough that the transition itself doesn't violate the Eddington
    convexity requirement."""
    z = (r - r_nfw_break) / cutoff_width
    return cutoff_width * np.logaddexp(0.0, z)

def DmShape(r):
    """Unnormalized radial shape: r^-gamma_sp inside r_spike_break, smoothly blending to
    r^-1 (the ambient NFW cusp slope) by r_nfw_break, then an exponential cutoff beyond
    r_nfw_break. This is the standard Zhao (1996) double-power-law form, chosen (rather
    than a discontinuous 3-piece law) so density and its derivatives stay continuous
    across the break -- required for a numerically stable Eddington inversion below."""
    x = r / r_spike_break
    x_eta = x**eta_spike
    shape = x**(-gamma_sp) * (1.0 + x_eta)**((gamma_sp - 1.0)/eta_spike)
    cutoff = np.exp(-_smooth_excess(r) / r_decay_scale)
    return shape * cutoff

_shape_norm = 2.0**((gamma_sp - 1.0)/eta_spike)   # DmShape(r_spike_break) == 1 before any cutoff

def DmShapeNormalized(r):
    return DmShape(r) / _shape_norm

shape_mass_1pc = quad(lambda rp: 4.0*np.pi*rp**2*DmShapeNormalized(rp), 0.0, r_nfw_break,
                       epsabs=0.0, epsrel=1e-10)[0]
rho_ref = NFW_M1pc / shape_mass_1pc

print("r_isco [pc]:", r_isco/pc_to_kpc)
print("r_spike_break [pc]:", r_spike_break/pc_to_kpc)
print("r_nfw_break [pc]:", r_nfw_break/pc_to_kpc)
print("NFW mass inside 1 pc:", NFW_M1pc)
print("rho_ref:", rho_ref)

# Sampling domain, expressed relative to the mass-matching break radius exactly as the
# original file expressed R_min/R_max/R_zero relative to r200. R_max/R_zero are widened
# from the original file's 1.5/5.0 to match the wider decay transition above (density is
# already <0.1% of its 1pc value by 4 pc, so little mass/particle budget is spent here).
R_min  = r_isco / r_nfw_break
R_max  = 4.0      # sample out to 4 pc
R_bins = 1000
R_zero = 10.0     # set the potential's zero-point at 10 pc
Rcut = 1.5  # only for gas

# Output: written directly in SPH-EXA's native "/Step#0" format (see the writer at the
# end of this file) -- no separate IC_to_SPH_EXA.ipynb conversion step needed.
#output_file = "/capstor/scratch/cscs/jgarlichs/spikes/IC_spikes/N1e10_star/relax/IC_spike_SPHEXA_N1e10_star_soft001.h5"
output_file = "ic_files/IC_spike_SPHEXA_N1e7_star_soft001.h5"

############################################################################################################
############################################################################################################

# Repurposed anchor variables: keep the original file's variable names for the (inert,
# gas_frac=0) gas branch and the softening formula, now pointing at the spike's own scales
# rather than the halo's r200/rhos/rs.
NFW_r200 = r_nfw_break
NFW_rhos = rho_ref
NFW_rs   = r_spike_break

decay_r   = 1.0
decay_eps = 1/decay_r - (1+3*NFW_c)/(1+NFW_c)   # only feeds the inert GasOuterRho below

NFW_delta = (200.0/3.0) * NFW_c**3.0/(np.log(1.0+NFW_c)-(NFW_c/(1.0 + NFW_c)))

MassCorrection = 1.0
MassScale = 4*np.pi*NFW_rhos*NFW_rs**3.0   # only used by the inert Gas*Mass functions below

if (Ngas > 0):
    gas_mass = NFW_M200*gas_frac/Ngas
else:
    gas_mass = 0.0

####### softening: mean interparticle separation (MIS) of the particles INSIDE 0.01 pc.
####### d = (V/N)^(1/3) with V the volume of a 0.01 pc sphere and N the number of particles
####### ACTUALLY enclosed there (from the profile's enclosed-mass fraction), scaled by
####### softening_multiplier. Resolves the inner cusp at its local spacing rather than the
####### 1-pc-averaged spacing -- ~19x smaller softening -> ~80x smaller timestep. The MIS
####### radius (0.01 pc) is < r_spike_break, so it sits inside the gamma_sp=7/3 spike.
r_mis_ref = 0.01 * r_nfw_break   # 0.01 pc in internal units (r_nfw_break = 1 pc)
shape_mass_mis = quad(lambda rp: 4.0*np.pi*rp**2*DmShapeNormalized(rp), 0.0, r_mis_ref,
                      epsabs=0.0, epsrel=1e-10)[0]
N_inside_mis = N_1pc * shape_mass_mis / shape_mass_1pc   # particles enclosed within 0.01 pc
soft_val = softening_multiplier * (4.0/3.0*np.pi*r_mis_ref**3 / N_inside_mis)**(1.0/3.0)
_soft_default = (4.0/3.0*np.pi*r_nfw_break**3 / N_1pc)**(1.0/3.0)   # old MIS-at-1pc, for comparison
print(f"particles inside 0.01 pc: {N_inside_mis:.3e}")
print("softening (MIS inside 0.01 pc) [pc]:", soft_val/pc_to_kpc)
print("  default MIS-at-1pc softening [pc]:", _soft_default/pc_to_kpc,
      f"  (ratio {soft_val/_soft_default:.3f}, ~{(soft_val/_soft_default)**1.5:.4f}x the timestep)")

# SPH-EXA's ryoanji P2P gravity kernel uses pair softening h_ij = h_i + h_j and replaces
# R^2 by max(R^2, h_ij^2) (verified directly against ryoanji/nbody/kernel.hpp) -- the
# softened BH potential below must match this exactly so the Eddington DF is built
# against the same force law SPH-EXA will actually integrate.
softening_DM  = soft_val
softening_BH  = soft_val
softening_BH_DM_pair = softening_BH + softening_DM

# Interpolation parameters
INTERPOL_BINS   = R_bins
INTERPOL_R_MIN  = NFW_r200*R_min
INTERPOL_R_MAX  = NFW_r200*R_max
INTERPOL_R_ZERO = NFW_r200*R_zero
Rcut *= NFW_r200


############################################################################################################
############################################################################################################

# HELPER FUNCTIONS

def interpolate(x,y):
    nan_mask = np.logical_or(np.isnan(x),np.isnan(y))
    inf_mak = np.logical_or(np.isinf(x),np.isinf(y))
    general_mask = np.logical_or(nan_mask,inf_mak)

    x = x[~general_mask]
    y = y[~general_mask]

    x, unique_mask = np.unique(x,True)
    y = y[unique_mask]

    return PchipInterpolator(x,y, extrapolate=True)

############################################################################

print("\n-DISTRIBUTION FUNCTION-\n")

print("Generating spike density/mass/potential functions...")
PROFILE_INTERPOL_BINS = INTERPOL_BINS * 20

interpolate_radial_bins = np.logspace(np.log10(INTERPOL_R_MIN),np.log10(INTERPOL_R_ZERO),PROFILE_INTERPOL_BINS)
log_radial_bins = np.log10(interpolate_radial_bins)

############################################################################

# Density
print(" Generating Density Profile...")
begin = time.time()

def DmRho(r):
    return (1 - gas_frac) * rho_ref * DmShapeNormalized(r)

def GasInnerRho(r):
    x0 = gas_R0 / NFW_rs
    x  = r      / NFW_rs
    return ( NFW_rhos
           / ((x + x0)*(1 + x)**2) )

def GasOuterRho(r):
    s0       = gas_R0 / NFW_r200
    s        = r      / NFW_r200
    rho_norm = GasInnerRho(NFW_r200)
    return rho_norm * s**decay_eps * np.exp(-(s - 1)/decay_r)

def GasRho(r):
    raw =  GasInnerRho(r) if r <= NFW_r200 else GasOuterRho(r)
    return raw *gas_frac

def Rho(r):
    return GasRho(r)+DmRho(r)

vecDmRho = np.vectorize(DmRho)
vecGasRho = np.vectorize(GasRho)
vecRho = np.vectorize(Rho)

# Interpolating Density
density_bins = vecRho(interpolate_radial_bins)
non_negative_density_mask = density_bins > 0
if not non_negative_density_mask.all():
    print(" !!Negative Density Warning")
    print(f" r = {interpolate_radial_bins[density_bins <= 0]}, rho = {density_bins[density_bins <= 0]}")
log_density_bins = np.log10(density_bins[non_negative_density_mask])
log_density_interpolate = interpolate(log_radial_bins[non_negative_density_mask],log_density_bins)
def Rho_Interpolate(r):
    return 10**log_density_interpolate(np.log10(r))

#interpolating DM density
dm_density_bins = vecDmRho(interpolate_radial_bins)
dm_non_negative_density_mask = dm_density_bins > 0
if not dm_non_negative_density_mask.all():
    print(" !!Negative Density Warning")
    print(f" r = {interpolate_radial_bins[dm_density_bins <= 0]}, rho = {dm_density_bins[dm_density_bins <= 0]}")
log_dm_density_bins = np.log10(dm_density_bins[dm_non_negative_density_mask])
log_dm_density_interpolate = interpolate(log_radial_bins[dm_non_negative_density_mask],log_dm_density_bins)
def Dm_Rho_Interpolate(r):
    return 10**log_dm_density_interpolate(np.log10(r))

#interpolating Gas Density
if Ngas >0:
    gas_density_bins = vecGasRho(interpolate_radial_bins)
    gas_non_negative_density_mask = gas_density_bins > 0
    if not gas_non_negative_density_mask.all():
        print(" !!Negative Density Warning")
        print(f" r = {interpolate_radial_bins[gas_density_bins <= 0]}, rho = {gas_density_bins[gas_density_bins <= 0]}")
    log_gas_density_bins = np.log10(gas_density_bins[gas_non_negative_density_mask])
    log_gas_density_interpolate = interpolate(log_radial_bins[gas_non_negative_density_mask],log_gas_density_bins)

    def Gas_Rho_Interpolate(r):
        return 10**log_gas_density_interpolate(np.log10(r))

print(f" -- {round(time.time()-begin,1)}s --")


############################################################################

# Analytic density derivatives (needed for the Eddington DF below). No closed form
# exists for the blended spike/cusp/decay shape, so d(ln rho)/dr and its derivative are
# built from the shape function's log-slope analytically (avoids the finite-difference
# noise that a much steeper, more structured profile than plain NFW would otherwise
# introduce into the DF).
def DmLogSlopeAndDerivative(r):
    x = r / r_spike_break
    x_eta = x**eta_spike
    log_slope_shape = -gamma_sp + (gamma_sp - 1.0)*x_eta/(1.0 + x_eta)
    slope_deriv_shape = (gamma_sp - 1.0)*eta_spike*x_eta/(1.0 + x_eta)**2  # d(log_slope)/d(ln r)

    L_shape = log_slope_shape / r
    Lp_shape = (slope_deriv_shape - log_slope_shape) / r**2

    # d(ln cutoff)/dr = -sigmoid(z)/r_decay_scale, with z=(r-r_nfw_break)/cutoff_width;
    # its derivative is -sigmoid(z)(1-sigmoid(z))/(cutoff_width*r_decay_scale). Both are
    # smooth and nonzero only within a few cutoff_width of r_nfw_break -- unlike a hard
    # max(...,0) onset, whose second-derivative contribution is a Dirac delta there.
    z = (r - r_nfw_break) / cutoff_width
    sig = 1.0/(1.0 + np.exp(-z))
    dlncutoff_dr = -sig / r_decay_scale
    d2lncutoff_dr2 = -sig*(1.0 - sig) / (cutoff_width*r_decay_scale)

    L = L_shape + dlncutoff_dr
    Lp = Lp_shape + d2lncutoff_dr2
    return L, Lp

def DmRhoRadialDerivatives(r):
    """Return rho, d(rho)/dr, d2(rho)/dr2."""
    rho = DmRho(r)
    L, Lp = DmLogSlopeAndDerivative(r)
    rho_prime = rho*L
    rho_second = rho*(L**2 + Lp)
    return rho, rho_prime, rho_second


############################################################################

# Mass
print(" Generating Mass Profile...")
begin = time.time()

def GasInnerMass(r):
    x0 = gas_R0/NFW_rs
    x = r/NFW_rs
    return MassCorrection*MassScale*(x*(x0-1)/(1+x)+x0*x0*np.log(1+x/x0)+(1-2*x0)*np.log(1+x))/(1-x0)**2.0

if Ngas<0:
    EdgeMass_gas = GasInnerMass(NFW_r200)

def GasOuterMass(r):
    integral_gas = quad(lambda r:4*np.pi*r**2*GasOuterRho(r), NFW_r200, r)[0]
    return EdgeMass_gas + integral_gas

def GasMass(r):
    raw = GasInnerMass(r) if r <= NFW_r200 else GasOuterMass(r)
    return gas_frac * raw

# No closed form exists for the blended DM density, so tabulate the enclosed mass by
# cumulative trapezoidal integration of 4*pi*r^2*rho(r) instead of an analytic formula.
dm_mass_offset = quad(lambda rp: 4.0*np.pi*rp**2*DmRho(rp), 0.0, INTERPOL_R_MIN,
                      epsabs=0.0, epsrel=1e-10)[0]
dm_mass_integrand_bins = 4.0*np.pi*interpolate_radial_bins**2*dm_density_bins
dm_mass_profile_bins = dm_mass_offset + cumulative_trapezoid(
    dm_mass_integrand_bins, interpolate_radial_bins, initial=0.0,
)
log_dm_mass_profile_interpolate = interpolate(log_radial_bins, np.log10(dm_mass_profile_bins))

def HaloMass(r):
    return 10**log_dm_mass_profile_interpolate(np.log10(r))

def Mass(r):
    if Ngas > 0:
        return GasMass(r) + HaloMass(r)
    else:
        return HaloMass(r)

vecMass = np.vectorize(Mass)
vecHaloMass = np.vectorize(HaloMass)

mass_bins = vecMass(interpolate_radial_bins)
halo_mass_bins = vecHaloMass(interpolate_radial_bins)

log_mass_bins= np.log10(mass_bins)
log_halo_mass_bins = np.log10(halo_mass_bins)

log_mass_interpolate = interpolate(log_radial_bins, log_mass_bins)
log_halo_mass_interpolate = interpolate(log_radial_bins, log_halo_mass_bins)

if Ngas>0:
    vecGasMass = np.vectorize(GasMass)
    gas_mass_bins = vecGasMass(interpolate_radial_bins)
    log_gas_mass_bins = np.log10(gas_mass_bins)
    log_gas_mass_interpolate = interpolate(log_radial_bins, log_gas_mass_bins)

def Mass_Interpolate(r):
    return 10**log_mass_interpolate(np.log10(r))

def Halo_Mass_Interpolate(r):
    return 10**log_halo_mass_interpolate(np.log10(r))

def Gas_Mass_Interpolate(r):
    return 10**log_gas_mass_interpolate(np.log10(r))

spike_mass_ref = Halo_Mass_Interpolate(r_nfw_break)   # M(<1pc); should equal NFW_M1pc by construction
halo_mass = spike_mass_ref / N_1pc
print("Mass inside 1 pc (spike):", spike_mass_ref)
print("Mass inside 1 pc (reference NFW):", NFW_M1pc)
print("Particle mass:", halo_mass)

print(f" -- {round(time.time()-begin,1)}s --")


###########################################################################

def Omega(r):
    if (S_omega == 0):
        return 1.0
    else:
        return (Mass_Interpolate(r)/spike_mass_ref)**S_omega/r**2.0

def Sigma_Integrand(r):
    return co.G*Mass_Interpolate(r)*Rho_Interpolate(r)/r**2.0

# --- Parameters ---
r0 = NFW_r200
sigma_tail_alpha = 0.1
sigma_tail_rcut  = 0.1 * r0

def Sigma_Numeric(r):
    r = np.atleast_1d(r)
    out = np.zeros_like(r, dtype=np.float64)
    for i, ri in enumerate(r):
        r_safe = ri * (1 - 1e-6) if np.isclose(ri, r0) else ri
        val = quad(Sigma_Integrand, r_safe, r0, epsrel=1e-3)[0] / Rho_Interpolate(r_safe)
        out[i] = np.sqrt(val) if val > 0 else 0.0
    return out if out.size > 1 else out[0]

sigma0 = Sigma_Numeric(r0)

def sigma_tail(r, NFW_r200, sigma200, alpha=0.5, rcut_frac=0.2):
    rcut = rcut_frac * NFW_r200
    s = r / NFW_r200
    exponent = np.clip(-(r - NFW_r200) / rcut, -700, 700)
    return sigma200 * s**(-alpha) * np.exp(exponent)

def Sigma(r):
    r = np.asarray(r)
    sigma = np.zeros_like(r)

    inner_mask = r <= NFW_r200
    outer_mask = r > NFW_r200

    sigma[inner_mask] = np.array([Sigma_Numeric(ri) for ri in r[inner_mask]])

    if np.any(outer_mask):
        sigma200 = Sigma_Numeric(NFW_r200 * (1 - 1e-5))
        sigma_tail_vals = sigma_tail(r[outer_mask], NFW_r200, sigma200)
        w = 0.5 * (1 + np.tanh((r[outer_mask]/NFW_r200 - 1)/0.05))
        sigma[outer_mask] = (1 - w) * sigma200 + w * sigma_tail_vals

    return sigma

vecOmega = np.vectorize(Omega)
vecSigma = np.vectorize(Sigma)

# angular momentum (MMW); Lambda=0 by default (see note above), rJ=0 in that case.
FC = (2.0/3.0)+(NFW_c/21.5)**0.7
HaloEnergy = -(co.G*spike_mass_ref**2.0*FC)/(2*NFW_r200)
rJ = Lambda * (co.G*spike_mass_ref**2.5) / np.sqrt(np.abs(HaloEnergy))

sigma_bins = vecSigma(interpolate_radial_bins)
valid_sigma_mask = (sigma_bins > 0) & (~np.isnan(sigma_bins)) & (~np.isinf(sigma_bins))

log_sigma_bins = np.log10(sigma_bins[valid_sigma_mask])
log_radial_bins_sigma = log_radial_bins[valid_sigma_mask]
log_InterpolSigma = interpolate(log_radial_bins_sigma, log_sigma_bins)

omega_bins = vecOmega(interpolate_radial_bins)
valid_omega_mask = (omega_bins > 0) & (~np.isnan(omega_bins)) & (~np.isinf(omega_bins))

log_omega_bins = np.log10(omega_bins[valid_omega_mask])
log_radial_bins_omega = log_radial_bins[valid_omega_mask]
log_InterpolOmega = interpolate(log_radial_bins_omega, log_omega_bins)


############################################################################

# Potential
print(" Generating Potential Profile...")
begin = time.time()

def OuterPotential(r):
    # Shell-theorem integral; fully generic in Mass_Interpolate, needs no closed form.
    integrand = lambda rp: -co.G * Mass_Interpolate(rp) / rp**2
    return -quad(integrand, r, INTERPOL_R_ZERO)[0]

def PsiBH(r):
    """Central BH potential, softened to match SPH-EXA's hard-capped R_eff = max(R, h_ij)
    pairwise force law exactly (see softening_BH_DM_pair above): this is the antiderivative
    of a(r) = G*M_BH*r/h^3 for r<h (i.e. the force law SPH-EXA actually applies below the
    pair-softening length), matched continuously (C1) to -G*M_BH/r at r=h."""
    h = softening_BH_DM_pair
    if r >= h:
        return co.G * M_BH * (1.0/r - 1.0/INTERPOL_R_ZERO)
    else:
        return co.G * M_BH * (1.5/h - 0.5*r**2/h**3 - 1.0/INTERPOL_R_ZERO)

def Potential(r):
    return PsiBH(r) + OuterPotential(r)

vecPotential = np.vectorize(Potential)
potential_bins = vecPotential(interpolate_radial_bins)
log_potential_bins = np.log10(potential_bins)
log_potential_interpolate = interpolate(log_radial_bins,log_potential_bins)

def Potential_Interpolate(r):
    return 10**log_potential_interpolate(np.log10(r))

radius_from_potential = interpolate(potential_bins, interpolate_radial_bins)

def PotentialRadialDerivatives(r):
    """dPsi/dr, d2Psi/dr2, from Poisson's equation for the DM part
    (Psi' = -G*M(r)/r^2, Psi'' = 2*G*M(r)/r^3 - 4*pi*G*rho(r)) plus the analytic
    derivative of the softened BH term above."""
    rho = DmRho(r)
    Mr = Mass_Interpolate(r)
    psi_prime_dm = -co.G*Mr/r**2
    psi_second_dm = 2.0*co.G*Mr/r**3 - 4.0*np.pi*co.G*rho

    h = softening_BH_DM_pair
    softened = r < h
    psi_prime_bh = np.where(softened, -co.G*M_BH*r/h**3, -co.G*M_BH/r**2)
    psi_second_bh = np.where(softened, -co.G*M_BH/h**3, 2.0*co.G*M_BH/r**3)

    return psi_prime_dm + psi_prime_bh, psi_second_dm + psi_second_bh

def DmDensitySecondPotential(phi):
    """d2(rho)/dPsi2 via the chain rule (rho''*Psi' - rho'*Psi'') / Psi'^3, evaluated by
    mapping phi back to radius. Replaces finite-differencing a tabulated rho(Psi) curve,
    which is not robust enough for a profile this structured (steep 7/3 cusp, a break,
    and a BH-softening scale ~13 orders of magnitude below the halo scale)."""
    phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    result = np.zeros_like(phi)
    valid = (phi > 0.0) & (phi <= potential_bins.max())
    if np.any(valid):
        r = radius_from_potential(phi[valid])
        _, rho_prime, rho_second = DmRhoRadialDerivatives(r)
        psi_prime, psi_second = PotentialRadialDerivatives(r)
        result[valid] = (rho_second*psi_prime - rho_prime*psi_second) / psi_prime**3
    return result

print(f" -- {round(time.time()-begin,1)}s --")
pot_vals_t = Potential_Interpolate(interpolate_radial_bins)
mass_val= Mass_Interpolate(interpolate_radial_bins)


############################################################################################################

# Eddington DF
print("Generating Eddington Distribution...")
begin = time.time()

potential_max = Potential_Interpolate(INTERPOL_R_MIN)
potential_floor = potential_max * 1e-12
Eddington_potential_bins = np.concatenate((
    [0.0],
    np.geomspace(potential_floor, potential_max, PROFILE_INTERPOL_BINS - 1),
    [Potential_Interpolate(softening_BH_DM_pair), Potential_Interpolate(r_spike_break),
     Potential_Interpolate(r_nfw_break)],
))
Eddington_potential_bins = np.unique(np.sort(Eddington_potential_bins))
Dm_Eddington_derivative_bins = DmDensitySecondPotential(Eddington_potential_bins)

if not np.all(np.isfinite(Dm_Eddington_derivative_bins)):
    invalid = np.where(~np.isfinite(Dm_Eddington_derivative_bins))[0]
    raise RuntimeError(
        f"Non-finite d2rho/dPsi2 in {invalid.size} Eddington bins; "
        f"phi range=({Eddington_potential_bins[invalid].min()}, {Eddington_potential_bins[invalid].max()})"
    )

Dm_Eddington_derivative_interpolate = PchipInterpolator(
    Eddington_potential_bins, Dm_Eddington_derivative_bins, extrapolate=False,
)

def Dm_Eddington_Derivative_Interpolate(phi):
    phi_arr = np.asarray(phi)
    result = np.zeros_like(phi_arr, dtype=np.float64)
    valid = (phi_arr >= 0.0) & (phi_arr <= potential_max)
    result[valid] = Dm_Eddington_derivative_interpolate(phi_arr[valid])
    return result

def Dm_Eddington_DF(epsilon):
    """f(epsilon) = 1/(sqrt(8) pi^2) * int_0^epsilon d2rho/dPsi2 / sqrt(epsilon-phi) dphi,
    using the substitution phi = epsilon*(1-t^2) to remove the 1/sqrt(epsilon-phi)
    endpoint singularity."""
    if epsilon <= 0.0:
        return 0.0
    integral = quad(
        lambda t: Dm_Eddington_Derivative_Interpolate(np.array([epsilon*(1.0 - t*t)]))[0],
        0.0, 1.0, epsabs=0.0, epsrel=1e-6, limit=200,
    )[0]
    return 2.0*np.sqrt(epsilon)*integral / (np.sqrt(8.0)*np.pi**2)

Dm_vecEddington_DF = np.vectorize(Dm_Eddington_DF)

epsilon_bins = np.geomspace(potential_floor, potential_max, PROFILE_INTERPOL_BINS)
Dm_Eddington_DF_bins = Dm_vecEddington_DF(epsilon_bins)
if not np.all(np.isfinite(Dm_Eddington_DF_bins)):
    raise RuntimeError("Non-finite values in Eddington distribution function")
if np.any(Dm_Eddington_DF_bins <= 0.0):
    invalid = np.where(Dm_Eddington_DF_bins <= 0.0)[0]
    raise RuntimeError(
        f"Eddington distribution is non-positive in {invalid.size} energy bins "
        f"(first invalid epsilon={epsilon_bins[invalid[0]]}); "
        "the requested density has no non-negative isotropic DF."
    )
Dm_log_Eddington_DF_bins = np.log10(Dm_Eddington_DF_bins)
Dm_log_Eddington_DF_interpolate = interpolate(epsilon_bins,Dm_log_Eddington_DF_bins)
def Dm_Eddington_DF_Interpolate(epsilon):
    epsilon_arr = np.asarray(epsilon)
    out = np.zeros_like(epsilon_arr, dtype=np.float64)
    valid = epsilon_arr > epsilon_bins[0]
    out[valid] = 10**Dm_log_Eddington_DF_interpolate(epsilon_arr[valid])
    return out.item() if np.ndim(epsilon) == 0 else out


def ValidateDmDistributionFunction():
    """Reconstruct rho(r) and the radial velocity dispersion from f(E) and compare to the
    target density / Jeans-equation dispersion. Standard due-diligence check for an
    Eddington-inversion IC that is going to be scaled to 1e10 particles and published:
    fails loudly here rather than silently producing a bad realization at scale."""
    validation_radii = np.geomspace(
        max(3.0*softening_BH_DM_pair, INTERPOL_R_MIN), INTERPOL_R_MAX*(1.0 - 1e-4), 48,
    )
    density_target = DmRho(validation_radii)
    density_reconstructed = np.zeros_like(validation_radii)
    sigma_df = np.zeros_like(validation_radii)

    for i, radius in enumerate(validation_radii):
        psi = Potential_Interpolate(radius)
        v_escape = np.sqrt(2.0*psi)
        density_reconstructed[i] = 4.0*np.pi*v_escape**3*quad(
            lambda q: Dm_Eddington_DF_Interpolate(psi*(1.0 - q*q))*q*q,
            0.0, 1.0, epsabs=0.0, epsrel=2e-5, limit=200,
        )[0]
        radial_moment = 4.0*np.pi/3.0*v_escape**5*quad(
            lambda q: Dm_Eddington_DF_Interpolate(psi*(1.0 - q*q))*q**4,
            0.0, 1.0, epsabs=0.0, epsrel=2e-5, limit=200,
        )[0]
        sigma_df[i] = radial_moment / density_reconstructed[i]

    # Integrated out to INTERPOL_R_ZERO (not just INTERPOL_R_MAX): rho*sigma_r^2 -> 0 is
    # only a good boundary condition where the density is truly negligible, and this
    # profile is still not exactly zero right at the sampling cutoff R_MAX (it just isn't
    # sampled beyond there) -- stopping the integral exactly at R_MAX understates
    # sigma_jeans for validation radii close to R_MAX.
    jeans_grid = np.geomspace(validation_radii.min(), INTERPOL_R_ZERO, 20000)
    jeans_density = DmRho(jeans_grid)
    psi_prime, _ = PotentialRadialDerivatives(jeans_grid)
    jeans_integrand = jeans_density * (-psi_prime)
    jeans_integral = -cumulative_trapezoid(jeans_integrand[::-1], jeans_grid[::-1], initial=0.0)[::-1]
    sigma_jeans = np.interp(
        validation_radii, jeans_grid, jeans_integral / np.maximum(jeans_density, np.finfo(float).tiny)
    )

    density_error = np.abs(density_reconstructed/density_target - 1.0)
    jeans_error = np.abs(sigma_df/sigma_jeans - 1.0)
    print("Maximum DF density reconstruction error:", density_error.max())
    print("Maximum DF/Jeans dispersion error:", jeans_error.max())
    if density_error.max() > 0.01 or jeans_error.max() > 0.01:
        raise RuntimeError("Eddington validation failed: density and Jeans errors must both be <= 1%")

ValidateDmDistributionFunction()

print(f" -- {round(time.time()-begin,1)}s --")


############################################################################

def Dm_Eddington_Velocity_Distribution(phi,rho,v):
    return Dm_Eddington_DF_Interpolate(phi-v**2/2) * 4*np.pi * v**2 / rho

print("done.")

rd.seed(seed)

print("\n-SAMPLING-\n")

spike_mass_sampled = Halo_Mass_Interpolate(INTERPOL_R_MAX)
N_halo = int(np.rint(N_1pc * spike_mass_sampled / spike_mass_ref))
print("Mass inside sampling radius:", spike_mass_sampled)
print("Number of particles inside 1 pc:", int(N_1pc))
print("Total sampled DM particles:", N_halo)

# The chunked sampling/writing loop below keeps peak memory bounded to O(CHUNK_SIZE)
# regardless of N_halo (needed for N_1pc up to 1e10 on a single node -- see CHUNK_SIZE
# above), but it only implements the two paths this script actually exercises: no gas,
# and no net rotation (Lambda=0, the physically-motivated default for a dispersion-
# supported spike -- see the note where Lambda is set). Both would need a second global
# pass over all particles (gas density-weighted binning; angular-momentum matching),
# which isn't compatible with bounded per-chunk memory as implemented here.
if Ngas > 0:
    raise NotImplementedError(
        "Chunked generation only supports Ngas=0 (gas was never exercised by this "
        "script's physics and isn't wired into the chunked sampling/writing loop)."
    )
N_gas = 0

if Lambda != 0.0:
    raise NotImplementedError(
        "Chunked generation only supports the default Lambda=0 (isotropic, "
        "non-rotating spike). Matching a nonzero net angular momentum needs a global "
        "correction pass over all particles, incompatible with bounded per-chunk memory."
    )

radial_bins_halo = np.logspace(np.log10(INTERPOL_R_MIN),np.log10(INTERPOL_R_MAX),INTERPOL_BINS)
mass_bins_halo = Halo_Mass_Interpolate(radial_bins_halo)
HaloRadius = interpolate(mass_bins_halo,radial_bins_halo)
mass_min_halo = mass_bins_halo.min()
mass_max_halo = mass_bins_halo.max()

# ---- Per-radial-bin velocity CDFs ----
# Built once from the fixed INTERPOL_BINS grid (independent of N_halo) and reused by
# every chunk sampled below.
print("Building per-radial-bin velocity CDFs...")
begin = time.time()

radius_bins_center = (radial_bins_halo[1:]+radial_bins_halo[:-1])/2
potential_bins_center = Potential_Interpolate(radius_bins_center)
dm_rho_bins_center = Dm_Rho_Interpolate(radius_bins_center)
vmax_bins = np.sqrt(2*potential_bins_center)

N_VEL_BINS = PROFILE_INTERPOL_BINS
velocity_grid_bins = []
velocity_cdf_bins = []

for i in range(INTERPOL_BINS-1):
    if vmax_bins[i] <= 0 or np.isnan(vmax_bins[i]) or not np.isfinite(potential_bins_center[i]):
        velocity_grid_bins.append(None)
        velocity_cdf_bins.append(None)
        continue

    v_array_full = np.linspace(0.0, vmax_bins[i], N_VEL_BINS)
    v_distribution_full = Dm_Eddington_Velocity_Distribution(potential_bins_center[i], dm_rho_bins_center[i], v_array_full)
    v_distribution_full[~np.isfinite(v_distribution_full)] = 0.0
    v_distribution_full[v_distribution_full < 0.0] = 0.0

    velocity_cdf = np.zeros_like(v_array_full)
    velocity_cdf[1:] = np.cumsum(
        0.5 * (v_distribution_full[1:] + v_distribution_full[:-1]) * np.diff(v_array_full)
    )

    if velocity_cdf[-1] <= 0.0 or not np.isfinite(velocity_cdf[-1]):
        velocity_grid_bins.append(None)
        velocity_cdf_bins.append(None)
        continue

    velocity_cdf /= velocity_cdf[-1]
    velocity_cdf_unique, unique_indices = np.unique(velocity_cdf, return_index=True)
    velocity_grid_unique = v_array_full[unique_indices]

    velocity_grid_bins.append(velocity_grid_unique)
    velocity_cdf_bins.append(velocity_cdf_unique)

print(f" -- {round(time.time()-begin,1)}s --")

# Diagnostic: sampled velocity histogram vs. the analytic Eddington-DF PDF at a few
# radii spanning the BH-softening core, the spike break, and the outer cusp. Bounded
# per-bin accumulation (DIAG_CAP samples) below keeps this independent of N_halo.
diagnostic_radii = np.geomspace(max(3.0*softening_BH_DM_pair, INTERPOL_R_MIN*3.0), r_nfw_break, 6)
diagnostic_ibin = [int(np.argmin(np.abs(radius_bins_center - rc))) for rc in diagnostic_radii]
DIAG_CAP = 20000
diagnostic_samples = [[] for _ in diagnostic_radii]
diagnostic_counts = [0 for _ in diagnostic_radii]

# ---- Rescale to pc / Msun / km/s for the SPH-EXA file ----
# (native units above are kpc / 1e10 Msun / km/s, as in the postdoc's original file)
length_rescale = 1e3      # kpc -> pc
mass_rescale   = 1e10     # 1e10 Msun -> Msun
G_rescale      = length_rescale / mass_rescale
G_sphexa = co.G * G_rescale
M_BH_sphexa = M_BH * mass_rescale
# The BH is a separate "star" object (not a tree particle). Fold the unsampled DM interior to
# the inner cutoff (dm_mass_offset; ~0.008 Msun for the non-softcut ISCO cutoff) onto the star's
# mass, so the run's central force field matches the potential the Eddington DF was built against.
M_BH_particle = M_BH + dm_mass_offset
M_BH_particle_sphexa = M_BH_particle * mass_rescale
softening_BH_DM_pair_sphexa = softening_BH_DM_pair * length_rescale
halo_mass_sphexa = halo_mass * mass_rescale
softening_DM_sphexa = softening_DM * length_rescale
softening_BH_sphexa = softening_BH * length_rescale

# ---- Initial timestep bootstrap ----
# Matches SPH-EXA's own acceleration-based adaptive timestep criterion
# (dt = etaAcc * sqrt(h / |a|), see sph/include/sph/ts_global.hpp) so the very first
# leapfrog step (x_m1/y_m1/z_m1 below) is already close to what SPH-EXA would choose on
# its own, rather than reusing an arbitrary halo-scale value tuned for a completely
# different (~500 kpc, no BH) system. a_max is the largest acceleration any particle can
# feel: the softened BH force is capped at r = softening_BH_DM_pair (see PsiBH above),
# so a_max = G*M_BH / softening_BH_DM_pair^2. Softening is uniform across every DM
# particle (softening_DM == softening_BH by construction above), so its minimum is just
# that scalar -- no need to touch a per-particle array to find it.
etaAcc = 0.2
a_max_sphexa = G_sphexa * M_BH_sphexa / softening_BH_DM_pair_sphexa**2
# Extra 1e-3 safety factor on top of the etaAcc criterion: the global (non-block)
# timestep is only guaranteed accurate for whichever particle currently has the worst
# h/|a|, so eccentric-orbit particles that plunge into the softened region *between*
# snapshots can still be under-resolved even when the reported dt looks converged.
# Shrinking the bootstrap this much reduces that one-time under-resolution risk at step 1.
minDt_safety_factor = 1e-3
minDt = etaAcc * np.sqrt(min(softening_DM_sphexa, softening_BH_sphexa) / a_max_sphexa) * minDt_safety_factor
minDt_m1 = minDt
print("Initial timestep bootstrap (minDt) [pc/(km/s)]:", minDt)

numParticles = N_halo   # BH is a separate "star" object (star::* attrs), NOT a particle
print(f"Final counts: Ngas={Ngas}, Ndm={N_halo} (BH is a star, not a particle) = {numParticles}")

def write_step_attributes(initStep):
    """Write all Step#0 metadata attributes: the ones SPH-EXA requires to start
    (numParticlesGlobal, time, minDt, gravConstant, box, ...) plus provenance
    metadata for the paper. Factored out so it can be written either at the end of a
    full generation run or, in patch mode, into an already-written particle file whose
    attribute step was interrupted. Reads module-level globals computed in setup."""
    initStep.attrs["numParticlesGlobal"] = numParticles
    initStep.attrs["time"] = np.float64(0.0)
    initStep.attrs["minDt"] = np.float64(minDt)
    initStep.attrs["minDt_m1"] = np.float64(minDt_m1)
    initStep.attrs["iteration"] = np.uint64(1)
    initStep.attrs["gravConstant"] = np.float64(G_sphexa)
    initStep.attrs["etaAcc"] = np.float64(etaAcc)

    initStep.attrs["gamma_sp"] = np.float64(gamma_sp)
    initStep.attrs["r_spike_break"] = np.float64(r_spike_break * length_rescale)
    initStep.attrs["r_nfw_break"] = np.float64(r_nfw_break * length_rescale)
    initStep.attrs["r_decay_scale"] = np.float64(r_decay_scale * length_rescale)
    initStep.attrs["r_trunc"] = np.float64(INTERPOL_R_MAX * length_rescale)
    initStep.attrs["M_BH"] = np.float64(M_BH_sphexa)                    # true physical BH mass
    initStep.attrs["M_BH_particle"] = np.float64(M_BH_particle_sphexa)  # star mass in the run (incl. interior DM)
    initStep.attrs["r_isco"] = np.float64(r_isco * length_rescale)

    # ---- central "star" (BH) for the central-star nbody propagator (disk::StarData) ----
    # The BH is read from these attributes -- it is NOT a particle. inner_size softens the
    # central force (dist2 = max(inner_size^2, r^2)) to match the DF's softened PsiBH exactly.
    initStep.attrs["star::x"] = np.float64(0.0)
    initStep.attrs["star::y"] = np.float64(0.0)
    initStep.attrs["star::z"] = np.float64(0.0)
    initStep.attrs["star::x_m1"] = np.float64(0.0)   # BH at rest -> zero previous-step drift
    initStep.attrs["star::y_m1"] = np.float64(0.0)
    initStep.attrs["star::z_m1"] = np.float64(0.0)
    initStep.attrs["star::m"] = np.float64(M_BH_particle_sphexa)                  # Msun
    initStep.attrs["star::inner_size"] = np.float64(softening_BH_DM_pair_sphexa)  # pc
    initStep.attrs["star::fixed_star"] = np.int32(1)      # 1 = pinned at origin (relaxation); 0 = free (merger)
    initStep.attrs["star::potentialType"] = np.int32(0)  # 0 = newtonian
    initStep.attrs["M_NFW_inside_1pc"] = np.float64(NFW_M1pc * mass_rescale)
    initStep.attrs["M_spike_inside_1pc"] = np.float64(spike_mass_ref * mass_rescale)
    initStep.attrs["M_spike_sampled"] = np.float64(spike_mass_sampled * mass_rescale)
    initStep.attrs["N_particles_inside_1pc"] = np.float64(N_1pc)
    initStep.attrs["N_particles_sampled"] = np.uint64(N_halo)
    initStep.attrs["softening_DM"] = np.float64(softening_DM * length_rescale)
    initStep.attrs["softening_BH"] = np.float64(softening_BH * length_rescale)
    initStep.attrs["softening_BH_DM_pair"] = np.float64(softening_BH_DM_pair_sphexa)
    initStep.attrs["softening_multiplier"] = np.float64(softening_multiplier)

    initStep.attrs["unitLength"] = f"{length_rescale}*kpc"
    initStep.attrs["unitMass"] = f"{mass_rescale}*1e10Msun"
    initStep.attrs["unitVelocity"] = "km/s"
    initStep.attrs["lengthrescale"] = np.float64(length_rescale)
    initStep.attrs["massrescale"] = np.float64(mass_rescale)

    initStep.attrs["boundaryType"] = np.array([0, 0, 0], dtype=np.uint8)
    box_half_width = 10.0 * INTERPOL_R_MAX * length_rescale   # 10x R_max, in pc
    initStep.attrs["box"] = np.array([-1, 1, -1, 1, -1, 1], dtype=np.float64) * box_half_width


# Patch mode: an earlier run wrote all particle data but was interrupted before the
# Step#0 attributes were written (e.g. the diagnostic-plot crash on NumPy 2.x). Re-open
# the existing file in place and write only the attributes -- no re-sampling of the
# multi-TB particle datasets. Enable with:  IC_PATCH_ATTRS_ONLY=1 python IC_code_spike.py
if os.environ.get("IC_PATCH_ATTRS_ONLY", "0") == "1":
    print(f"\n[patch mode] Writing Step#0 attributes into existing file: {output_file}")
    with h5py.File(output_file, "r+") as f:
        write_step_attributes(f["Step#0"])
    print("[patch mode] Attributes written. Done.")
    raise SystemExit(0)

print("\nSampling & writing DM particles in chunks of", CHUNK_SIZE, "...")
begin = time.time()

with h5py.File(output_file, "w") as f:
    initStep = f.create_group("/Step#0")

    dset = {}
    for name in ["x","y","z","vx","vy","vz","x_m1","y_m1","z_m1","m","h"]:
        dset[name] = initStep.create_dataset(name, shape=(numParticles,), dtype=np.float64, chunks=True)
    dset["id"] = initStep.create_dataset("id", shape=(numParticles,), dtype=np.uint64, chunks=True)

    # No BH particle: the BH is a separate "star" object (see the star::* Step#0 attributes),
    # handled outside the tree by the central-star nbody propagator's computeCentralForce. So
    # all numParticles=N_halo entries are DM, written from index 0.

    kinetic_sum = 0.0
    virial_sum = 0.0
    L_sum = 0.0

    write_pos = 0
    remaining = N_halo
    while remaining > 0:
        n_chunk = min(CHUNK_SIZE, remaining)

        u = mass_min_halo + rd.random_sample(n_chunk) * (mass_max_halo - mass_min_halo)
        radius_chunk = HaloRadius(u)
        phi_chunk = 2.0*np.pi*rd.random_sample(n_chunk)
        theta_chunk = np.arcsin(2.0*rd.random_sample(n_chunk)-1.0)

        x_chunk = radius_chunk*np.cos(theta_chunk)*np.cos(phi_chunk)
        y_chunk = radius_chunk*np.cos(theta_chunk)*np.sin(phi_chunk)
        z_chunk = radius_chunk*np.sin(theta_chunk)

        # Velocity sampling: inverse-CDF per radial bin. Rejection sampling (as used for
        # the smooth NFW profile) is not robust here -- the velocity distribution near
        # the BH-dominated cusp is far more sharply peaked than for a plain NFW halo, and
        # a Gaussian-envelope proposal tuned per bin can have arbitrarily low acceptance
        # in the steepest bins, a real risk of an unbounded loop at 1e10 particles on a
        # supercomputer. Inverse-CDF sampling is O(1) per particle and cannot stall.
        bins_idx_chunk = np.clip(np.digitize(radius_chunk, radial_bins_halo) - 1, 0, INTERPOL_BINS-2)
        velocity_chunk = np.zeros(n_chunk)
        for i in range(INTERPOL_BINS-1):
            mask = bins_idx_chunk == i
            n_sample = np.sum(mask)
            if n_sample == 0:
                continue
            velocity_grid = velocity_grid_bins[i]
            velocity_cdf = velocity_cdf_bins[i]
            if velocity_grid is None or velocity_cdf is None:
                continue
            velocity_chunk[mask] = np.interp(rd.random_sample(n_sample), velocity_cdf, velocity_grid)

        phi_vel_chunk = 2.0*np.pi*rd.random_sample(n_chunk)
        theta_vel_chunk = np.arcsin(2.0*rd.random_sample(n_chunk)-1.0)
        # No streaming term: Lambda=0 is enforced above, so the full velocity is exactly
        # the isotropic Eddington-DF dispersion velocity (V_stream would be identically 0).
        vx_chunk = velocity_chunk*np.cos(theta_vel_chunk)*np.cos(phi_vel_chunk)
        vy_chunk = velocity_chunk*np.cos(theta_vel_chunk)*np.sin(phi_vel_chunk)
        vz_chunk = velocity_chunk*np.sin(theta_vel_chunk)

        accel_chunk = -PotentialRadialDerivatives(radius_chunk)[0]
        kinetic_sum += 0.5*halo_mass*np.sum(velocity_chunk**2)
        virial_sum  += halo_mass*np.sum(radius_chunk*accel_chunk)
        L_sum += halo_mass*np.sum(x_chunk*vy_chunk - y_chunk*vx_chunk)

        for k in range(len(diagnostic_radii)):
            if diagnostic_counts[k] >= DIAG_CAP:
                continue
            mask = bins_idx_chunk == diagnostic_ibin[k]
            if not np.any(mask):
                continue
            take = velocity_chunk[mask][:DIAG_CAP - diagnostic_counts[k]]
            diagnostic_samples[k].append(take)
            diagnostic_counts[k] += take.size

        end = write_pos + n_chunk
        dset["x"][write_pos:end] = x_chunk * length_rescale
        dset["y"][write_pos:end] = y_chunk * length_rescale
        dset["z"][write_pos:end] = z_chunk * length_rescale
        dset["vx"][write_pos:end] = vx_chunk
        dset["vy"][write_pos:end] = vy_chunk
        dset["vz"][write_pos:end] = vz_chunk
        dset["x_m1"][write_pos:end] = vx_chunk * minDt
        dset["y_m1"][write_pos:end] = vy_chunk * minDt
        dset["z_m1"][write_pos:end] = vz_chunk * minDt
        dset["m"][write_pos:end] = np.full(n_chunk, halo_mass_sphexa, dtype=np.float64)
        dset["h"][write_pos:end] = np.full(n_chunk, softening_DM_sphexa, dtype=np.float64)
        dset["id"][write_pos:end] = np.arange(write_pos, end, dtype=np.uint64)

        write_pos = end
        remaining -= n_chunk
        print(f"  ... {write_pos - 1}/{N_halo} DM particles written")

    print(f" -- {round(time.time()-begin,1)}s --")
    print("done.\n")

    sample_virial_ratio = 2.0 * kinetic_sum / np.abs(virial_sum)
    print("Sampled softened virial ratio 2K/|W|:", sample_virial_ratio)
    # This is a finite-N Monte Carlo estimate, not the exact continuum DF -- expect a few
    # percent of statistical scatter at small N (e.g. ~1.3% at N_1pc=2e3 in testing), shrinking
    # towards the continuum value as N grows. At the production N_1pc (~1e6-1e7) this should
    # converge comfortably inside the 1% tolerance below.
    if np.abs(sample_virial_ratio - 1.0) > 0.01:
        raise RuntimeError(f"Sampled softened virial ratio {sample_virial_ratio} is outside [0.99, 1.01]")

    print("Desired angular momentum: ", rJ)
    print("Achieved angular momentum:", L_sum)
    print("Relative error:", 0.0 if rJ == 0.0 else abs(L_sum - rJ)/rJ)

    # Write the Step#0 attributes NOW -- immediately after the physics checks pass and
    # before the (optional, purely cosmetic) diagnostic plots. This guarantees the file
    # is complete and SPH-EXA-loadable the instant the particle data is valid; a failure
    # in the plotting below can no longer leave a multi-TB file without its 'box' attr.
    write_step_attributes(initStep)

    # Diagnostic velocity-distribution plots. Non-essential: wrapped so any failure here
    # (e.g. a plotting/library issue) is logged but never invalidates the finished IC file.
    try:
        for k, rc in enumerate(diagnostic_radii):
            if diagnostic_counts[k] == 0:
                continue
            i_bin = diagnostic_ibin[k]
            phi_i = potential_bins_center[i_bin]
            rho_i = dm_rho_bins_center[i_bin]
            v_esc = np.sqrt(2*phi_i)

            v_grid = np.linspace(0, v_esc, 200)
            f_v = Dm_Eddington_Velocity_Distribution(phi_i, rho_i, v_grid)
            f_v[-1] = 0.0
            f_v /= np.trapz(f_v, v_grid)

            v_samp = np.concatenate(diagnostic_samples[k])

            plt.figure()
            plt.hist(v_samp, bins=50, density=True, alpha=0.6, label='sampled')
            plt.plot(v_grid, f_v, '-k', lw=2, label='Eddington-DF')
            plt.xlabel(r'$v$ [km s$^{-1}$]'); plt.ylabel(r'$P(v\,|\,r)$ [(km s$^{-1}$)$^{-1}$]'); plt.legend()
            plt.title(f'Bin at r={rc/pc_to_kpc:.3e} pc')
            plt.savefig(f"vel_dist_{N_1pc:.0e}_r{rc/pc_to_kpc:.3e}pc.png", dpi=150, bbox_inches="tight")
            plt.close()

        # Combined 2x2 panel (publication figure): four of the diagnostic radii in a single
        # frame, sharing a single legend. Each panel keeps its own velocity axis in km/s,
        # since v_esc varies strongly with radius. The per-radius plots above are unchanged.
        populated = [k for k in range(len(diagnostic_radii)) if diagnostic_counts[k] > 0]
        if len(populated) >= 4:
            picks = sorted({populated[j] for j in
                            np.linspace(0, len(populated) - 1, 4).round().astype(int)})
            fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))
            for ax, k in zip(axes.flat, picks):
                i_bin = diagnostic_ibin[k]
                phi_i = potential_bins_center[i_bin]
                rho_i = dm_rho_bins_center[i_bin]
                v_esc = np.sqrt(2*phi_i)

                v_grid = np.linspace(0, v_esc, 200)
                f_v = Dm_Eddington_Velocity_Distribution(phi_i, rho_i, v_grid)
                f_v[-1] = 0.0
                f_v /= np.trapz(f_v, v_grid)

                v_samp = np.concatenate(diagnostic_samples[k])
                ax.hist(v_samp, bins=40, density=True, histtype='stepfilled',
                        alpha=0.45, color='tab:blue', edgecolor='tab:blue', label='sampled')
                ax.plot(v_grid, f_v, '-k', lw=1.8, label='Eddington DF')
                ax.set_xlim(0.0, v_esc)
                ax.text(0.04, 0.93, f"$r = {diagnostic_radii[k]/pc_to_kpc:.3g}$ pc",
                        transform=ax.transAxes, ha='left', va='top')
            for ax in axes.flat[len(picks):]:
                ax.set_visible(False)
            for ax in axes[-1, :]:
                ax.set_xlabel(r'$v$ [km s$^{-1}$]')
            for ax in axes[:, 0]:
                ax.set_ylabel(r'$P(v\,|\,r)$ [(km s$^{-1}$)$^{-1}$]')
            axes.flat[0].legend(loc='upper right', frameon=False)
            fig.tight_layout()
            fig.savefig(f"vel_dist_panel_{N_1pc:.0e}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    except Exception as e:
        print(f"WARNING: diagnostic velocity plots failed ({e!r}); IC file is unaffected.")

print(f"Wrote SPH-EXA-ready HDF5 IC: {output_file}")


print("\n-FINISHED-\n")

print("r_spike_break [pc] = ", r_spike_break/pc_to_kpc)
print("r_nfw_break [pc]   = ", r_nfw_break/pc_to_kpc)
print("gamma_sp = ", gamma_sp)
print("M_BH [1e10 Msun] = ", M_BH)
print("m_part = ", halo_mass)
print("N_1pc  = ", N_1pc)
print("N_halo (DM, excl. BH) = ", N_halo)
print("N_gas = ", Ngas)
print("gas_mass = ", gas_mass)
