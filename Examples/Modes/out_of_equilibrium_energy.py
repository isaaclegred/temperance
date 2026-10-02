# When a star is perturbed from equilibrium, the energy
# of the system reaches a new minimum at some displacement
# which is determined by the tidal deformability.
#
# If the out-of-equilibrium configuration is at frozen
# composition, then it is straightforward to compute the
# change in energy from returning the system to equilibrium
#
# It looks like
# \begin{equation}
#     \delta E = -\int \left[ n Y^\beta_Q E'_{\rm lep}(nY_Q^{\beta}) + (1-Y_{Q})^2\frac{L}{3}\right] \nabla \cdot \vec \xi \, dV
# \end{equation}
#
# where $\vec \xi$ is the perturbation displacement, $n$ is the number density
# $Y^{\beta}_Q$ is the equilibrium charge fraction (electron fraction, if electrons are the only charged lepton)
# and E_{\lep} is the energy due to leptons (again, only leptons),
# ' is the derivative, and L is the slope of the symmetry energy.
#
# This can be computed for an ell=2 perturbation using the
# utilities in temperance
# in particular, the solving module, to compute the out-of-equilibrium
# energy, as well as compute the mode profile.

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import integrate


from temperance.solving.analytic_eos import polytropic_eos, analytic_composition_dependent_eos
from temperance.solving.stellar_models import NewtonianStellarModel
from temperance.solving.solve_oscillations import NewtonianModes

# ---------------------------------------------------------------------------
# Unit-conversion constants
# ---------------------------------------------------------------------------
RHO_SAT_GEOM = 0.00045       # nuclear saturation density [geometric units]
RHO_SAT_CGS  = 2.8e14        # nuclear saturation density [g/cm^3]
GEOM_TO_CGS  = RHO_SAT_CGS / RHO_SAT_GEOM   # geometric -> g/cm^3
KM_PER_MSUN  = 1.477          # 1 Msun_geom = 1.477 km
FM_PER_KM    = 1.0e18         # 1 km = 10^18 fm

# 1 MeV/fm^3 in g/cm^3: 1 MeV = 1.602e-6 erg, 1 fm^3 = 1e-39 cm^3
# => 1 MeV/fm^3 = 1.602e-6 / 1e-39 erg/cm^3 / c^2 = 1.602e-6 / 1e-39 / (3e10)^2 g/cm^3
MEV_FM3_TO_CGS = 1.602176634e-6 / 1.0e-39 / (2.99792458e10)**2   # ~1.782e12 g/cm^3
CGS_TO_MEV_FM3 = 1.0 / MEV_FM3_TO_CGS

# Nuclear matter parameters (standard empirical values)
N0   = 0.16    # nuclear saturation density [fm^-3]
E0   = -16.0   # symmetric-matter energy/nucleon at saturation [MeV]
K0   = 220.0   # nuclear incompressibility [MeV]
S0   = 32.0    # symmetry energy at saturation [MeV]
L    = 60.0    # slope of symmetry energy at saturation [MeV]
HBAR = 197.3   # hbar*c [MeV*fm]
ME   = 0.511   # electron mass [MeV]

# ---------------------------------------------------------------------------
# Helper: relativistic electron chemical potential
# ---------------------------------------------------------------------------

def electron_chemical_potential(n_fm3, Ye):
    """Relativistic electron chemical potential mu_e [MeV]."""
    ne = n_fm3 * Ye
    if ne <= 0.0:
        return ME
    pF = HBAR * (3.0 * np.pi**2 * ne)**(1.0 / 3.0)
    return float(np.sqrt(pF**2 + ME**2))


# ---------------------------------------------------------------------------
# Build EoS: polytrope (geometric units) wrapped for nuclear-unit interface,
# then extended to arbitrary composition via analytic_composition_dependent_eos.
# ---------------------------------------------------------------------------
eos   = polytropic_eos(K=123.6, Gamma=2.0)
rho_c = 2.0 * RHO_SAT_GEOM

class _GeomToNuclearEos:
    """Thin wrapper: accepts n_B [fm^-3], returns e [MeV/fm^3]."""
    def e_of_rho(self, n_fm3):
        rho_cgs  = np.atleast_1d(n_fm3) * (1.6749e-24 / 1.0e-39)
        rho_geom = rho_cgs / GEOM_TO_CGS
        e_geom   = eos.e_of_rho(rho_geom)
        return e_geom * GEOM_TO_CGS / MEV_FM3_TO_CGS

eos_comp = analytic_composition_dependent_eos(
    _GeomToNuclearEos(),
    saturation_params={"EB": E0, "nsat": N0, "K0": K0},
    symmetry_params={"S": S0, "L": L, "Ksym": 0.0, "Ybeta": 0.04},
    nuclear_masses={"mp": 938.272, "mn": 939.565, "mN": 931.494, "me": ME},
    density_range=np.linspace(0.05, 0.8, 40),
)

# ---------------------------------------------------------------------------
# TODO (1): Given a cold EoS and a central density, construct the TOV solution
# ---------------------------------------------------------------------------
model = NewtonianStellarModel(eos, rho_c=rho_c)

M_msun = model.M
R_km   = model.R * KM_PER_MSUN
print(f"Stellar model:  M = {M_msun:.3e} Msun,  R = {R_km:.2f} km")

# ---------------------------------------------------------------------------
# TODO (2): Compute the equilibrium electron fraction Y_e(r) along the profile
# ---------------------------------------------------------------------------

# Baryon number density in fm^-3: n = (rho [g/cm^3]) / (m_n [g]) / (1 fm^-3 in cm^-3)
rho_cgs = model.rho * GEOM_TO_CGS   # [g/cm^3]
n_fm3   = rho_cgs / (1.6749e-24) * 1.0e-39   # fm^-3

Ye    = np.clip(eos_comp._Ybeta_of_n(n_fm3), 0.0, 0.5)
mu_e  = np.array([electron_chemical_potential(float(n), float(y)) for n, y in zip(n_fm3, Ye)])

print(f"Equilibrium Y_e: centre = {Ye[0]:.4f},  surface ~ {Ye[-5]:.4f}")
print(f"mu_e at centre  = {mu_e[0]:.2f} MeV")

# ---------------------------------------------------------------------------
# TODO (3): Solve for the ell=2 f-mode; assign a perturbation of strength epsilon
#
# The ell=2 radial displacement eigenfunction xi_r(r) is normalised to
# max|xi_r| = 1.  The physical amplitude is epsilon * R * xi_r  (with xi_r
# being the normalised profile stored in mode.xi_r).
# ---------------------------------------------------------------------------
ell    = 2
solver = NewtonianModes(model, cowling=True)
mode   = solver.compute_nonradial_mode(ell=ell, m=0, n=0)

SPEED_OF_LIGHT = 299792.458  # km/s
OMEGA_TO_HZ    = SPEED_OF_LIGHT / (2.0 * np.pi * KM_PER_MSUN)
f0_hz = mode.omega.real * OMEGA_TO_HZ
print(f"\nell=2 f-mode frequency: {f0_hz:.4e} Hz")

# Scale the normalised eigenfunction by epsilon and R
epsilon = 1.0e-4            # dimensionless tidal amplitude
r_grid  = mode.radial_grid  # [geometric length units] on the mode's adaptive grid
xi_r    = epsilon * mode.xi_r * model.R   # physical displacement [geometric length units]

# Divergence of the displacement field (radial contribution):
#   (nabla . xi)_radial = d(xi_r)/dr + 2 xi_r / r
#                       = (1/r^2) d(r^2 xi_r)/dr
dxi_r  = np.gradient(xi_r, r_grid)
div_xi = dxi_r + 2.0 * xi_r / r_grid   # [dimensionless]

# ---------------------------------------------------------------------------
# TODO (4): Compute the out-of-equilibrium energy change
#
#   delta_E = -int [n Y_Q mu_e(n Y_Q) + (1-Y_Q)^2 L/3] * (nabla . xi) dV
#
# The integrand is in MeV/fm^3; the volume element r^2 dr is converted to fm^3.
# We interpolate the background (n, Y_e, mu_e) onto the mode's radial grid.
# ---------------------------------------------------------------------------

# Interpolate background quantities onto the mode grid
rho_on_mode = interp1d(model.r, model.rho, kind="linear", fill_value="extrapolate")(r_grid)
n_on_mode   = interp1d(model.r, n_fm3, kind="linear", fill_value="extrapolate")(r_grid)
Ye_on_mode  = interp1d(model.r, Ye,    kind="linear", fill_value="extrapolate")(r_grid)
mue_on_mode = interp1d(model.r, mu_e,  kind="linear", fill_value="extrapolate")(r_grid)

# Exclude the crust: only integrate over the core (rho >= 0.5 * rho_sat)
RHO_CRUST_CORE = 0.5 * RHO_SAT_GEOM   # crust-core boundary [geometric units]
print("density is", rho_on_mode)
core_mask = rho_on_mode >= RHO_CRUST_CORE

r_core  = r_grid[core_mask]
print(r_grid)
r_cc_km = r_core[-1] * KM_PER_MSUN     # crust-core interface radius [km]
print(f"\nCrust-core boundary at r = {r_cc_km:.2f} km  "
      f"(rho >= {RHO_CRUST_CORE/RHO_SAT_GEOM:.1f} rho_sat)")

# Integrand: [MeV/fm^3] * [1] (div_xi is dimensionless)
integrand = (
    n_on_mode * Ye_on_mode * mue_on_mode    # n Y_e mu_e  [MeV/fm^3]
    + (1.0 - Ye_on_mode)**2 * L / 3.0       # (1-Y_e)^2 L/3  [MeV/fm^3 -- dimensionless * MeV approx]
) * div_xi

# Volume element in fm^3: dV = 4 pi r^2 dr with r converted to fm
r_fm = r_grid * KM_PER_MSUN * FM_PER_KM   # [fm]

# Integrate over core only
delta_E_mev = -4.0 * np.pi * integrate.simpson(
    (integrand * r_fm**2)[core_mask], r_fm[core_mask]
)

print(f"\nOut-of-equilibrium energy change (epsilon = {epsilon:.1e}):")
print(f"  delta_E = {delta_E_mev:.4e}  MeV")
print(f"  delta_E = {delta_E_mev * 1.602176634e-13:.4e}  J")

# ---------------------------------------------------------------------------
# Plots  (crust excluded: rho < 0.5 rho_sat)
# ---------------------------------------------------------------------------
r_km = model.r * KM_PER_MSUN
r_mode_km = r_grid * KM_PER_MSUN

# Core mask on the model's background grid
core_mask_model = model.rho >= RHO_CRUST_CORE

fig, axes = plt.subplots(1, 4, figsize=(15, 4))

axes[0].plot(r_km[core_mask_model], Ye[core_mask_model], lw=1.8)
axes[0].set_xlabel("r  [km]")
axes[0].set_ylabel(r"$Y_e^{\beta}(r)$")
axes[0].set_title("Equilibrium electron fraction")

axes[1].plot(r_mode_km[core_mask], mode.xi_r[core_mask], lw=1.8)
axes[1].axhline(0, color="k", lw=0.6, ls=":")
axes[1].set_xlabel("r  [km]")
axes[1].set_ylabel(r"$\xi_r / \max|\xi_r|$")
axes[1].set_title(fr"$\ell=2$ f-mode  ($f = {f0_hz:.3e}$ Hz)")

axes[2].plot(r_mode_km[core_mask], (integrand * r_fm**2)[core_mask], lw=1.8)
axes[2].axhline(0, color="k", lw=0.6, ls=":")
axes[2].set_xlabel("r  [km]")
axes[2].set_ylabel(r"$\mathcal{I}(r)\,r^2$  [MeV]")
axes[2].set_title(r"Energy integrand $\times\,r^2$")

axes[3].plot(r_km[core_mask_model], model.rho[core_mask_model] / RHO_SAT_GEOM, lw=1.8)
axes[3].set_xlabel("r  [km]")
axes[3].set_ylabel(r"$\rho / \rho_{\rm sat}$")
axes[3].set_title(r"Density profile (core)")

fig.suptitle(
    fr"Out-of-equilibrium energy  ($\epsilon={epsilon:.0e}$, "
    fr"$M={M_msun:.2e}\,M_\odot$, $R={R_km:.1f}$ km)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig("oo_equilibrium_energy.pdf", bbox_inches="tight")
print("\nSaved: oo_equilibrium_energy.pdf")
plt.show()
