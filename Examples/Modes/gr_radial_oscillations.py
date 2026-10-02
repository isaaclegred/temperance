"""
GR radial oscillation modes of a neutron star.

Demonstrates:
  1. Building a RelativisticStellarModel from a polytropic EOS.
  2. Computing the first three radial modes (n=0,1,2) via the Chandrasekhar
     equations (PRD 108, 103035) and comparing to the Newtonian result.
  3. Plotting normalised eigenfunctions xi(r) for each GR mode.
  4. Sweeping central density to produce fundamental-mode frequency vs mass and
     checking the Gondek-Piotrowska (1997) stability criterion: at the
     mass maximum dM/drho_c = 0, the n=0 frequency passes through zero.

Units
-----
All internal quantities are in geometric units (G = c = 1).
  1 solar mass  = 1.477 km  =>  R [km] = R_geom * 1.477
  Frequencies: omega_geom [1/km_geom].  To convert to Hz multiply by
               c / (2 pi * 1.477 km) ~ 32261 Hz.

Run
---
    python Examples/gr_radial_oscillations.py
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "temperance"))

from solving.analytic_eos import polytropic_eos
from solving.stellar_models import NewtonianStellarModel, RelativisticStellarModel
from solving.solve_oscillations import NewtonianModes, RelativisticModes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RHO_SAT        = 0.00045        # nuclear saturation density (geometric units)
KM_PER_MSUN    = 1.477          # 1 Msun = 1.477 km
SPEED_OF_LIGHT = 299792.458     # km/s
OMEGA_TO_HZ    = SPEED_OF_LIGHT / (2.0 * np.pi * KM_PER_MSUN)  # geom omega -> Hz
OMEGA2_TO_KHZ2 = (OMEGA_TO_HZ / 1e3) ** 2                       # geom omega^2 -> kHz^2

# ---------------------------------------------------------------------------
# EOS: Gamma=2 polytrope (K=123.6 in geometric units)
# ---------------------------------------------------------------------------
eos = polytropic_eos(K=123.6, Gamma=2.0)

# ---------------------------------------------------------------------------
# 1.  Single star at 3 x nuclear saturation density
# ---------------------------------------------------------------------------
rho_c = 3.0 * RHO_SAT

gr_model   = RelativisticStellarModel(eos, rho_c=rho_c)
newt_model = NewtonianStellarModel(eos, rho_c=rho_c)

M_gr   = gr_model.M    # already in Msun
R_gr   = gr_model.R * KM_PER_MSUN
M_newt = newt_model.M 
R_newt = newt_model.R * KM_PER_MSUN

print(f"GR model:        M = {M_gr:.3f} Msun,  R = {R_gr:.2f} km,  C = {gr_model.C:.4f}")
print(f"Newtonian model: M = {M_newt:.3f} Msun,  R = {R_newt:.2f} km")
print()

# ---------------------------------------------------------------------------
# 2.  First three radial modes — GR vs Newtonian comparison
# ---------------------------------------------------------------------------
gr_solver   = RelativisticModes(gr_model)
newt_solver = NewtonianModes(newt_model, cowling=False)

print(f"{'n':>2}  {'f_GR [Hz]':>12}  {'f_Newt [Hz]':>12}")
print("-" * 32)

gr_modes, newt_modes = [], []
for n in range(3):
    mg = gr_solver.compute_radial_mode(n)
    mn = newt_solver.compute_radial_mode(n)
    fg = mg.omega.real * OMEGA_TO_HZ
    fn = mn.omega.real * OMEGA_TO_HZ
    print(f"{n:>2}  {fg:>12.1f}  {fn:>12.1f}")
    gr_modes.append(mg)
    newt_modes.append(mn)

# ---------------------------------------------------------------------------
# 3.  Plot normalised eigenfunctions xi(r)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

for n, (mg, mn) in enumerate(zip(gr_modes, newt_modes)):
    ax = axes[n]
    r_gr   = mg.radial_grid * KM_PER_MSUN
    r_newt = mn.radial_grid * KM_PER_MSUN
    ax.plot(r_gr,   mg.xi_r * r_gr, color="C0", lw=1.8, label="GR")
    ax.plot(r_newt, mn.xi_r, color="C1", lw=1.8, ls="--", label="Newtonian")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("r  [km]")
    if n == 0:
        ax.set_ylabel(r"$\xi / \max|\xi|$")
    fg = mg.omega.real * OMEGA_TO_HZ
    fn = mn.omega.real * OMEGA_TO_HZ
    ax.set_title(f"n={n}   $f_{{\\rm GR}}$ = {fg:.0f} Hz\n$f_{{\\rm Newt}}$ = {fn:.0f} Hz")
    ax.legend(fontsize=8)

fig.suptitle(
    f"GR vs Newtonian radial eigenfunctions  "
    f"($M_{{\\rm GR}}={M_gr:.2f}\\,M_\\odot$, $R_{{\\rm GR}}={R_gr:.1f}$ km, polytrope $\\Gamma=2$)"
)
fig.tight_layout()
fig.savefig("gr_radial_eigenfunctions.pdf", bbox_inches="tight")
print("\nSaved: gr_radial_eigenfunctions.pdf")

# ---------------------------------------------------------------------------
# 4.  omega^2 vs mass — density sweep (captures instability as omega^2 < 0)
#
# ---------------------------------------------------------------------------
rho_min = 0.5 * RHO_SAT
rho_max = 10.0 * RHO_SAT    # well past the mass maximum for Gamma=2

densities = np.linspace(rho_min, rho_max, 30)


def find_n0_omega2(solver: RelativisticModes) -> float:
    """
    Return the n=0 eigenvalue omega^2 (geometric units), allowing negative
    values for unstable stars.  Scans from -3*omega_char^2 to find the
    first sign change in the surface boundary condition.
    """
    m = solver.model
    omega_char2 = m.M / m.R**3

    def bc(omega2):
        _, _, y2s = solver._shoot_radial(omega2)
        return y2s

    omega2_lo = -3.0 * omega_char2
    omega2_hi = (4.0 * np.pi) ** 2 * omega_char2   # generous upper bound for n=0
    omega2_scan = np.linspace(omega2_lo, omega2_hi, 300)
    bc_vals = np.array([bc(o2) for o2 in omega2_scan])

    for i in range(len(bc_vals) - 1):
        if np.isfinite(bc_vals[i]) and np.isfinite(bc_vals[i + 1]):
            if bc_vals[i] * bc_vals[i + 1] < 0.0:
                return scipy.optimize.brentq(
                    bc, omega2_scan[i], omega2_scan[i + 1], xtol=1e-16, rtol=1e-10
                )
    raise ValueError("No bracket found for n=0 mode")


Ms_gr, omega2_gr = [], []

print("\nBuilding GR fundamental-mode family (omega^2, includes instability) ...")
for rho_c_i in densities:
    try:
        gm_i = RelativisticStellarModel(eos, rho_c=rho_c_i)
        gs_i = RelativisticModes(gm_i)
        omega2_i = find_n0_omega2(gs_i)
        Ms_gr.append(gm_i.M)
        omega2_gr.append(omega2_i)
        stability = "stable" if omega2_i > 0 else "UNSTABLE"
        print(f"  rho_c={rho_c_i/RHO_SAT:.2f} rho_sat  M={Ms_gr[-1]:.3f} Msun  "
              f"omega2={omega2_i * OMEGA2_TO_KHZ2:+.4f} kHz^2  [{stability}]")
    except Exception as exc:
        print(f"  rho_c={rho_c_i/RHO_SAT:.2f} rho_sat  skipped ({exc})")

omega2_plot = [o2 * OMEGA2_TO_KHZ2 for o2 in omega2_gr]
i_max = int(np.argmax(Ms_gr))

fig2, (ax_m, ax_f) = plt.subplots(1, 2, figsize=(11, 4))

# M vs rho_c
rho_plot = [d / RHO_SAT for d in densities[:len(Ms_gr)]]
ax_m.plot(rho_plot, Ms_gr, "o-", color="C0", lw=1.8)
ax_m.set_xlabel(r"$\rho_c / \rho_{\rm sat}$")
ax_m.set_ylabel(r"$M \; [M_\odot]$")
ax_m.set_title("Mass-central density curve")
ax_m.axvline(rho_plot[i_max], color="k", ls=":", lw=0.8)
ax_m.annotate(
    f"$M_{{\\rm max}}={Ms_gr[i_max]:.2f}\\,M_\\odot$",
    xy=(rho_plot[i_max], Ms_gr[i_max]),
    xytext=(rho_plot[i_max] + 0.3, Ms_gr[i_max] - 0.05),
    fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8),
)

# omega^2 vs M  (negative = unstable)
stable_M   = [M for M, o2 in zip(Ms_gr, omega2_plot) if o2 >= 0]
stable_o2  = [o2 for o2 in omega2_plot if o2 >= 0]
unstable_M  = [M for M, o2 in zip(Ms_gr, omega2_plot) if o2 < 0]
unstable_o2 = [o2 for o2 in omega2_plot if o2 < 0]

ax_f.plot(stable_M,   stable_o2,   "s-",  color="C1", lw=1.8, label="stable")
ax_f.plot(unstable_M, unstable_o2, "s--", color="C3", lw=1.8, label="unstable")
ax_f.axhline(0, color="k", lw=0.8, ls=":")
ax_f.axvline(Ms_gr[i_max], color="k", ls=":", lw=0.8,
             label=f"$M_{{\\rm max}}={Ms_gr[i_max]:.2f}\\,M_\\odot$")
ax_f.set_xlabel(r"$M \; [M_\odot]$")
ax_f.set_ylabel(r"$\omega_0^2 \; [\mathrm{kHz}^2]$")
ax_f.set_title(r"Radial stability: $\omega^2$ changes sign at $M_{\rm max}$")
ax_f.legend(fontsize=8)

fig2.tight_layout()
fig2.savefig("gr_omega2_vs_mass.pdf", bbox_inches="tight")
print("Saved: gr_omega2_vs_mass.pdf")
