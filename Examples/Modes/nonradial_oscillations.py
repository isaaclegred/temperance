"""
Newtonian nonradial oscillation modes of a neutron star.

Demonstrates:
  1. Building a stellar model from a polytropic EOS.
  2. Computing the f-mode and first two p-modes (n=0,1,2) for ell=2
     in the Cowling approximation and the full (self-gravity) calculation.
  3. Plotting the normalised eigenfunctions xi_r(r) for each mode.
  4. Sweeping central density to produce f-mode frequency vs mass curves
     for several angular degrees (ell=2,3,4).

Units
-----
All internal quantities are in geometric units (G = c = 1).
  1 solar mass  = 1.477 km  =>  R [km] = R_geom * 1.477
  Frequencies: omega_geom [1/km_geom].  To convert to kHz multiply by
               c / (2 pi * 1.477 km) ~ 32.3 kHz.

Run
---
    python Examples/Modes/nonradial_oscillations.py
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "temperance"))

from solving.analytic_eos import polytropic_eos
from solving.stellar_models import NewtonianStellarModel
from solving.solve_oscillations import NewtonianModes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RHO_SAT        = 0.00045          # nuclear saturation density [geometric units]
KM_PER_MSUN    = 1.477            # 1 Msun = 1.477 km
SPEED_OF_LIGHT = 299792.458       # km/s
OMEGA_TO_HZ    = SPEED_OF_LIGHT / (2.0 * np.pi * KM_PER_MSUN)  # geom omega -> Hz
OMEGA_TO_KHZ   = OMEGA_TO_HZ / 1e3  # kept for axis labels; prints use Hz

# ---------------------------------------------------------------------------
# EOS: Gamma=2 polytrope
# ---------------------------------------------------------------------------
eos = polytropic_eos(K=123.6, Gamma=2.0)

# ---------------------------------------------------------------------------
# 1.  Single star at 3 x nuclear saturation density
# ---------------------------------------------------------------------------
rho_c = 1e-3 * RHO_SAT
model = NewtonianStellarModel(eos, rho_c=rho_c)

M_msun = model.M
R_km   = model.R * KM_PER_MSUN

print(f"Stellar model:  M = {M_msun:.3e} Msun,  R = {R_km:.2f} km,  C = {model.M/model.R:.4f}")
print()

# ---------------------------------------------------------------------------
# 2.  n = 0, 1, 2 modes for ell=2 — Cowling vs full
# ---------------------------------------------------------------------------
solver_cowling = NewtonianModes(model, cowling=True)
solver_full    = NewtonianModes(model, cowling=False)

print(f"{'n':>2}  {'Cowling f [Hz]':>16}  {'Full f [Hz]':>13}  {'Δf/f':>7}")
print("-" * 46)

cowling_modes, full_modes = [], []
for n in range(3):
    mc = solver_cowling.compute_nonradial_mode(ell=2, m=0, n=n)
    mf = solver_full.compute_nonradial_mode(ell=2, m=0, n=n)
    fc = mc.omega.real * OMEGA_TO_HZ
    ff = mf.omega.real * OMEGA_TO_HZ
    print(f"{n:>2}  {fc:>16.4e}  {ff:>13.4e}  {(ff-fc)/fc:>+7.3f}")
    cowling_modes.append(mc)
    full_modes.append(mf)

# ---------------------------------------------------------------------------
# 3.  Normalised eigenfunctions xi_r(r) for each mode
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

for n, (mc, mf) in enumerate(zip(cowling_modes, full_modes)):
    ax = axes[n]
    r_c = mc.radial_grid * KM_PER_MSUN
    r_f = mf.radial_grid * KM_PER_MSUN
    ax.plot(r_c, mc.xi_r, color="C0", lw=1.8, label="Cowling")
    ax.plot(r_f, mf.xi_r, color="C1", lw=1.8, ls="--", label="Full")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("r  [km]")
    if n == 0:
        ax.set_ylabel(r"$\xi_r / \max|\xi_r|$")
    fc = mc.omega.real * OMEGA_TO_HZ
    ff = mf.omega.real * OMEGA_TO_HZ
    label = "f" if n == 0 else f"p{n}"
    ax.set_title(
        f"{label}-mode  ($\\ell=2$)\n"
        f"$f_{{\\rm Cowl}}={fc:.3e}$ Hz   $f_{{\\rm Full}}={ff:.3e}$ Hz"
    )
    ax.legend(fontsize=8)

fig.suptitle(
    f"Newtonian nonradial eigenfunctions  "
    f"($M={M_msun:.3e}\\,M_\\odot$, $R={R_km:.1f}$ km, polytrope $\\Gamma=2$)"
)
fig.tight_layout()
fig.savefig("nonradial_eigenfunctions.pdf", bbox_inches="tight")
print("\nSaved: nonradial_eigenfunctions.pdf")

# ---------------------------------------------------------------------------
# 4.  f-mode frequency vs gravitational mass for ell = 2, 3, 4
# ---------------------------------------------------------------------------
densities = np.linspace(1e-3 * RHO_SAT, 1e-2 * RHO_SAT, 5)
ells = [2]

fig2, ax2 = plt.subplots(figsize=(7, 5))

print("\nBuilding f-mode family vs mass ...")
for ell in ells:
    Ms, freqs, freqs_fund = [], [], []
    for rho_c_i in densities:
        try:
            print(f"  ell={ell}  rho_c={rho_c_i/RHO_SAT:.2f} rho_sat ... ", end="")
            m_i = NewtonianStellarModel(eos, rho_c=rho_c_i)
            s_i = NewtonianModes(m_i, cowling=False)
            mode_i = s_i.compute_nonradial_mode(ell=ell, m=0, n=0)
            Ms.append(m_i.M)
            freqs.append(mode_i.omega.real * OMEGA_TO_HZ)
            freqs_fund.append( m_i.M / (2 * np.pi * R_km**3) * OMEGA_TO_HZ)  # fundamental mode estimate
        except Exception as exc:
            print(f"  ell={ell}  rho_c={rho_c_i/RHO_SAT:.2f} rho_sat  skipped ({exc})")
    ax2.plot(Ms, freqs, "o-", lw=1.8, label=fr"$\ell={ell}$")
    ax2.plot(Ms, freqs_fund, "x--", lw=1.8, label=fr"$\ell={ell}$  fundamental estimate")
    print(f"  ell={ell}: {len(Ms)} points computed")

ax2.set_xlabel(r"$M \; [M_\odot]$")
ax2.set_ylabel(r"$f_{\rm f} \; [\mathrm{Hz}]$")
ax2.set_title(r"Nonradial f-mode frequency vs mass  (Full, polytrope $\Gamma=2$)")
ax2.legend()
fig2.tight_layout()
fig2.savefig("nonradial_fmode_vs_mass.pdf", bbox_inches="tight")
print("Saved: nonradial_fmode_vs_mass.pdf")

plt.show()
