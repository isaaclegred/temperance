"""
CompOSE QMC-RMF3 EoS: download, stellar structure, tidal deformability,
f-mode, and out-of-equilibrium energy.

Demonstrates:
  1. Downloading a CompOSE 3-D table via download_compose_3d_files.
  2. Building a 2-D (nB, Yq) EoS with
     interpolated_eos_density_and_charge_fraction.from_3d_compose_table.
  3. Extracting the cold beta-equilibrium 1-D EoS via
     find_1d_eos_by_taking_T_equal_Tmin_then_optimizing_ye (order=2).
  4. Sweeping central density to produce an M-R-Lambda family (TOV + tidal).
  5. Computing the ell=2 f-mode eigenfunction for a reference star.
  6. Computing the out-of-equilibrium energy δE for a tidal perturbation,
     with Yq_beta(n) and L(n) read from the CompOSE table.

Run
---
    python Examples/Compose/compose_eos_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import integrate

import temperance.external.read_3d_compose_table as tmcomp
from temperance.solving.analytic_eos import (
    interpolated_eos,
    interpolated_eos_density_and_charge_fraction,
)
from temperance.solving.stellar_models import NewtonianStellarModel
from temperance.solving.solve_relativistic import get_tov_family
from temperance.solving.solve_oscillations import NewtonianModes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
URL   = "https://compose.obspm.fr/download//3D/Brodie/Qmc_rmf_3/"
LABEL = "qmc_rmf3"

RHO_SAT_GEOM   = 0.00045            # nuclear saturation density [geometric]
RHO_SAT_CGS    = 2.8e14             # [g/cm^3]
GEOM_TO_CGS    = RHO_SAT_CGS / RHO_SAT_GEOM
KM_PER_MSUN    = 1.477
FM_PER_KM      = 1.0e18
N0             = 0.1546              # [fm^-3]
M_N            = 939           # neutron mass [MeV]
HBAR           = 197.3              # hbar c [MeV fm]
ME             = 0.511              # electron mass [MeV]
OMEGA_TO_HZ    = 2.99792458e5 / (2.0 * np.pi * KM_PER_MSUN)

# ---------------------------------------------------------------------------
# 1. Download & read CompOSE files
# ---------------------------------------------------------------------------
print("Downloading QMC-RMF3 from CompOSE ...")
thermo_path, nb_path, t_path, yq_path = tmcomp.download_compose_3d_files(
    URL, LABEL, dest_dir="."
)

thermo, nb, t, Yq = tmcomp.read_compose_3d_table(thermo_path, nb_path, t_path, yq_path)
nb_arr = np.asarray(nb["nb"])   # fm^-3
Yq_arr = np.asarray(Yq["Yq"])  # dimensionless

print(f"  nB : {len(nb_arr)} pts  [{nb_arr.min():.3f}, {nb_arr.max():.3f}] fm^-3")
print(f"  T  : {len(t)} pts")
print(f"  Yq : {len(Yq_arr)} pts  [{Yq_arr.min():.3f}, {Yq_arr.max():.3f}]")

# ---------------------------------------------------------------------------
# 2. Build EoS objects
#
# eos_2d  — 2-D (nB, Yq) EoS at min T; used for composition-dependent
#            quantities (Yq_beta, symmetry energy).
# eos_1d  — 1-D cold beta-equilibrium EoS wrapped in interpolated_eos for
#            TOV + mode solvers (which need the log-enthalpy interface).
# ---------------------------------------------------------------------------
eos_2d = interpolated_eos_density_and_charge_fraction.from_3d_compose_table(
    thermo, nb, t, Yq
)
print("\nBuilt 2-D EoS (nB, Yq) at min T.")

# Cold beta-equilibrium 1-D EoS: take T=T_min, optimize Yq at each density.
thermo_cold_1d = tmcomp.find_1d_eos_by_taking_T_equal_Tmin_then_optimizing_ye(
    thermo, nb, t, Yq, order=2
)
# thermo_cold_1d has columns "nb" (fm^-3), "Yq_opt", and all thermo quantities.
cgs_eos = tmcomp.cold_eos_to_cgs_standard(thermo_cold_1d, thermo_cold_1d[["nb"]])
eos_1d  = interpolated_eos(cgs_eos)
print("Built 1-D cold beta-equilibrium EoS.")

# ---------------------------------------------------------------------------
# 3. Composition profile: Yq_beta(n) and mu_e(n) from the beta-eq table
# ---------------------------------------------------------------------------
nb_beq = thermo_cold_1d["nb"].values       # fm^-3
Yq_beq = thermo_cold_1d["Yq_opt"].values


def _mu_e(n_fm3, Ye):
    """Relativistic electron chemical potential [MeV]."""
    ne = float(n_fm3) * float(Ye)
    if ne <= 0.0:
        return ME
    return float(np.sqrt((HBAR * (3.0 * np.pi**2 * ne) ** (1.0 / 3.0))**2 + ME**2))


mu_e_beq   = np.array([_mu_e(n, y) for n, y in zip(nb_beq, Yq_beq)])
Yq_beta_fn = interp1d(nb_beq, Yq_beq,   kind="cubic", fill_value="extrapolate")
mu_e_fn    = interp1d(nb_beq, mu_e_beq, kind="cubic", fill_value="extrapolate")

# ---------------------------------------------------------------------------
# 4. Symmetry energy S(n) and slope L(n) from the cold 2-D table
#
# At each density, fit e_per_baryon vs (1-2Yq)^2 with a line; the slope
# is S(n) in the parabolic approximation.  L = 3 n dS/dn.
# ---------------------------------------------------------------------------
thermo_cold_2d = tmcomp.find_2d_eos_by_taking_T_equal_minT(thermo, nb, t)

_S_nb, _S_vals = [], []
for inb_val in sorted(thermo_cold_2d["inb"].unique().astype(int)):
    rows = thermo_cold_2d[thermo_cold_2d["inb"] == inb_val]
    if len(rows) < 3:
        continue
    Yq_i  = Yq_arr[rows["iYq"].values.astype(int) - 1]
    e_i   = (rows["e_per_nbmn_minus_1"].values) * M_N  # MeV / baryon
    delta = (1.0 - 2.0 * Yq_i) ** 2
    if np.ptp(delta) < 1e-6:
        continue
    S_poly = np.polynomial.Polynomial.fit(delta, e_i, 1)
    _S_nb.append(nb_arr[inb_val - 1])
    _S_vals.append(float(S_poly.coef[0]))

S_nb   = np.asarray(_S_nb)
S_vals = np.asarray(_S_vals)
L_arr  = 3.0 * S_nb * np.gradient(S_vals, S_nb)

S_fn = interp1d(S_nb, S_vals, kind="cubic", fill_value="extrapolate")
L_fn = interp1d(S_nb, L_arr,  kind="cubic", fill_value="extrapolate")

S0_eos = float(S_fn(N0))
L_eos  = float(L_fn(N0))
print(f"\nQMC-RMF3 nuclear-matter properties at n0 = {N0} fm^-3:")
print(f"  S0 = {S0_eos:.1f} MeV,  L = {L_eos:.1f} MeV")

# ---------------------------------------------------------------------------
# 5. M-R-Lambda family via TOV + tidal deformability
# ---------------------------------------------------------------------------
print("\nComputing M-R-Lambda family ...")
rho_c_sweep = np.linspace(1 * RHO_SAT_GEOM, 8 * RHO_SAT_GEOM, 60)
family = get_tov_family(eos_1d, rho_c_sweep)

M_max = family["M"].max()
print(f"  Maximum mass: {M_max:.3f} Msun")

idx_14   = (family["M"] - 1.4).abs().argmin()
Lambda14 = family["Lambda"].iloc[idx_14]
print(f"  Lambda(1.4 Msun) = {Lambda14:.0f}")

# ---------------------------------------------------------------------------
# 6. Reference stellar model + ell=2 f-mode (Newtonian, Cowling)
# ---------------------------------------------------------------------------
rho_c_ref = 2.0 * RHO_SAT_GEOM
model     = NewtonianStellarModel(eos_1d, rho_c=rho_c_ref)
M_msun    = model.M
R_km      = model.R * KM_PER_MSUN
print(f"\nReference star (Newtonian): M = {M_msun:.3e} Msun,  R = {R_km:.2f} km")

mode  = NewtonianModes(model, cowling=True).compute_nonradial_mode(ell=2, m=0, n=0)
f_hz  = mode.omega.real * OMEGA_TO_HZ
print(f"ell=2 f-mode: {f_hz:.4e} Hz")

# ---------------------------------------------------------------------------
# 7. Out-of-equilibrium energy δE
#
# δE = -∫ [ n Yq_β μ_e + (1-Yq_β)^2 L(n)/3 ] ∇·ξ dV
#
# Yq_beta and L are read from the CompOSE table (sections 3 & 4).
# The integral is over the core only (rho >= 0.5 rho_sat).
# ---------------------------------------------------------------------------
epsilon = 1.0e-2
r_grid  = mode.radial_grid
xi_r    = epsilon * mode.xi_r * model.R
div_xi  = np.gradient(xi_r, r_grid) + 2.0 * xi_r / r_grid

rho_mode = interp1d(model.r, model.rho, fill_value="extrapolate")(r_grid)
n_mode   = rho_mode * GEOM_TO_CGS / (1.6749e-24) * 1.0e-39   # fm^-3
Ye_mode  = np.clip(Yq_beta_fn(n_mode), 0.0, 0.5)
mue_mode = mu_e_fn(n_mode)
L_mode   = L_fn(n_mode)

RHO_CC   = 0.00001 * RHO_SAT_GEOM
core     = rho_mode >= RHO_CC

integrand = (n_mode * Ye_mode * mue_mode
             + (1.0 - Ye_mode)**2 * L_mode / 3.0) * div_xi
r_fm      = r_grid * KM_PER_MSUN * FM_PER_KM   # fm

delta_E = -4.0 * np.pi * integrate.simpson(
    (integrand * r_fm**2)[core], r_fm[core]
)
print(f"\nδE (epsilon = {epsilon:.0e}): {delta_E:.4e} MeV  =  {delta_E * 1.602e-13:.4e} J")

# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------
RHO_CC_M = model.rho >= RHO_CC
r_km_m   = model.r * KM_PER_MSUN
n_m      = model.rho * GEOM_TO_CGS / (1.6749e-24) * 1.0e-39
Ye_m     = np.clip(Yq_beta_fn(n_m), 0.0, 0.5)
r_km_mode = r_grid * KM_PER_MSUN

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0, 0].plot(family["R"], family["M"], lw=1.8)
axes[0, 0].set_xlabel("R  [km]")
axes[0, 0].set_ylabel(r"$M\;[M_\odot]$")
axes[0, 0].set_title("Mass–radius relation")

axes[0, 1].semilogy(family["M"], family["Lambda"], lw=1.8)
axes[0, 1].set_xlabel(r"$M\;[M_\odot]$")
axes[0, 1].set_ylabel(r"$\Lambda$")
axes[0, 1].set_title("Tidal deformability")
axes[0, 1].grid(which="both", ls=":", lw=0.5)

axes[0, 2].plot(S_nb / N0, S_vals, lw=1.8)
axes[0, 2].axvline(1.0, color="grey", ls=":", lw=0.8)
axes[0, 2].set_xlabel(r"$n / n_0$")
axes[0, 2].set_ylabel("S(n)  [MeV]")
axes[0, 2].set_title(fr"Symmetry energy  ($S_0={S0_eos:.0f}$, $L={L_eos:.0f}$ MeV)")

axes[1, 0].plot(r_km_m[RHO_CC_M], Ye_m[RHO_CC_M], lw=1.8)
axes[1, 0].set_xlabel("r  [km]")
axes[1, 0].set_ylabel(r"$Y_e^\beta(r)$")
axes[1, 0].set_title("Equilibrium electron fraction")

axes[1, 1].plot(r_km_mode[core], mode.xi_r[core], lw=1.8)
axes[1, 1].axhline(0, color="k", lw=0.6, ls=":")
axes[1, 1].set_xlabel("r  [km]")
axes[1, 1].set_ylabel(r"$\xi_r / \max|\xi_r|$")
axes[1, 1].set_title(fr"$\ell=2$ f-mode  ($f = {f_hz:.3e}$ Hz)")

axes[1, 2].plot(r_km_mode[core], (integrand * r_fm**2)[core], lw=1.8)
axes[1, 2].axhline(0, color="k", lw=0.6, ls=":")
axes[1, 2].set_xlabel("r  [km]")
axes[1, 2].set_ylabel(r"$\mathcal{I}(r)\,r^2$  [MeV]")
axes[1, 2].set_title(r"Energy integrand $\times\,r^2$")

fig.suptitle(
    fr"QMC-RMF3 (CompOSE) — "
    fr"$M_\mathrm{{ref}} = {M_msun:.2e}\,M_\odot$, "
    fr"$R_\mathrm{{ref}} = {R_km:.1f}$ km, "
    fr"$\delta E = {delta_E:.2e}$ MeV",
    fontsize=10,
)
fig.tight_layout()
out_pdf = f"{LABEL}_analysis.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
print(f"\nSaved: {out_pdf}")
plt.show()
