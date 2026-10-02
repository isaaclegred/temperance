"""
Newtonian radial oscillation modes of a neutron star.

Demonstrates:
  1. Building a stellar model from a polytropic EOS.
  2. Computing the first three radial modes (n=0,1,2) in both the Cowling
     approximation and the full (self-gravity) calculation.
  3. Plotting the normalised eigenfunctions xi_r(r) for each mode.
  4. Sweeping central density to produce an f-mode frequency vs mass curve.

Units
-----
All internal quantities are in geometric units (G = c = 1).
  1 solar mass  = 1.477 km  =>  R [km] = R_geom * 1.477
  Frequencies: omega_geom [1/km_geom].  To convert to kHz multiply by
               c / (2 pi * 1.477 km) ~ 32.3 kHz.

Run
---
    python Examples/radial_oscillations.py
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "temperance"))

from solving.analytic_eos import polytropic_eos, css_eos, interpolated_eos
from solving.stellar_models import NewtonianStellarModel
from solving.solve_oscillations import NewtonianModes

# Conversion factors
RHO_SAT = 0.00045          # nuclear saturation density in geometric units
KM_PER_MSUN = 1.477        # 1 Msun = 1.477 km (geometric unit conversion)
SPEED_OF_LIGHT = 299792.458  # km/s, for converting frequencies to kHz
OMEGA_TO_HZ = 1.0 / (2.0 * np.pi * KM_PER_MSUN) * SPEED_OF_LIGHT # geom omega -> Hz

# ---------------------------------------------------------------------------
# EOS: SLy-like polytrope (K=123.6, Gamma=2)
# ---------------------------------------------------------------------------
eos = interpolated_eos(pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "chandra_wd_eq.csv"))) #polytropic_eos(K=123.6, Gamma=2.0)

# ---------------------------------------------------------------------------
# 1.  Single star at 3 x nuclear saturation density
# ---------------------------------------------------------------------------
rho_c = 1e-2 * RHO_SAT
model = NewtonianStellarModel(eos, rho_c=rho_c)

M_msun = model.M  # solar masses  (M_geom already in Msun units)
R_km   = model.R * KM_PER_MSUN

print(f"Stellar model:  M = {M_msun:.3f} Msun,  R = {R_km:.2f} km,  C = {model.M/model.R:.4f}")
print()

# ---------------------------------------------------------------------------
# 2.  Compute n = 0, 1, 2 modes in both approximations
# ---------------------------------------------------------------------------
modes_cowling = NewtonianModes(model, cowling=True)
modes_full    = NewtonianModes(model, cowling=False)

print(f"{'n':>2}  {'Cowling f [kHz]':>16}  {'Full f [kHz]':>13}  {'Δf/f':>7}")
print("-" * 46)

cowling_modes, full_modes = [], []
for n in range(3):
    mc = modes_cowling.compute_radial_mode(n)
    mf = modes_full.compute_radial_mode(n)
    fc = mc.omega.real * OMEGA_TO_HZ
    ff = mf.omega.real * OMEGA_TO_HZ
    print(f"{n:>2}  {fc:>16.4f}  {ff:>13.4f}  {(ff-fc)/fc:>+7.3f}")
    cowling_modes.append(mc)
    full_modes.append(mf)

# ---------------------------------------------------------------------------
# 3.  Plot normalised eigenfunctions for each mode
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

colors = {"Cowling": "C0", "Full": "C1"}
for n, (mc, mf) in enumerate(zip(cowling_modes, full_modes)):
    ax = axes[n]
    r_c = mc.radial_grid * KM_PER_MSUN
    r_f = mf.radial_grid * KM_PER_MSUN
    ax.plot(r_c, mc.xi_r, color=colors["Cowling"], label="Cowling", lw=1.8)
    ax.plot(r_f, mf.xi_r, color=colors["Full"],    label="Full",    lw=1.8, ls="--")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("r  [km]")
    ax.set_ylabel(r"$\xi_r / \max|\xi_r|$") if n == 0 else None
    fc = mc.omega.real * OMEGA_TO_HZ
    ff = mf.omega.real * OMEGA_TO_HZ
    ax.set_title(f"n={n}   $f_{{\\rm Cowl}}$ = {fc:.2f} Hz\n$f_{{\\rm Full}}$ = {ff:.2f} Hz")
    ax.legend(fontsize=8)

fig.suptitle(
    f"Newtonian radial eigenfunctions  "
    f"($M={M_msun:.2f}\\,M_\\odot$, $R={R_km:.1f}$ km, polytrope $\\Gamma=2$)"
)
fig.tight_layout()
fig.savefig("radial_eigenfunctions.pdf", bbox_inches="tight")
print("\nSaved: radial_eigenfunctions.pdf")

# ---------------------------------------------------------------------------
# 4.  f-mode frequency vs gravitational mass (both approximations)
# ---------------------------------------------------------------------------
densities = np.linspace(0.5 * rho_c, 5 * rho_c, 20)

Ms, Rs, f0_cowling, f0_full = [], [], [], []

print("\nBuilding fundamental-mode family ...")
for rho_c_i in densities:
    m_i = NewtonianStellarModel(eos, rho_c=rho_c_i)
    mc_i = NewtonianModes(m_i, cowling=True).compute_radial_mode(0)
    mf_i = NewtonianModes(m_i, cowling=False).compute_radial_mode(0)
    Ms.append(m_i.M)
    Rs.append(m_i.R)    
    f0_cowling.append(mc_i.omega.real * OMEGA_TO_HZ)
    f0_full.append(mf_i.omega.real * OMEGA_TO_HZ)
    print(f"  M={Ms[-1]:.3f} Msun  f0_Cowl={f0_cowling[-1]:.3f} Hz  f0_Full={f0_full[-1]:.3f} Hz Omega_fund={np.sqrt(m_i.M * KM_PER_MSUN/m_i.R**3) * OMEGA_TO_HZ:.4f}")

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(Ms, f0_cowling, "o-", color="C0", label="Cowling")
ax2.plot(Ms, f0_full,    "s--", color="C1", label="Full")
ax2.plot(Ms, (np.array(Ms)/np.array(Rs)**3)**0.5 * OMEGA_TO_HZ, "k:", label=r"$\sqrt{M/R^3}$")
ax2.set_xlabel(r"$M \; [M_\odot]$")
ax2.set_ylabel(r"$f_0 \; [\mathrm{Hz}]$")
ax2.set_title(r"Fundamental radial mode frequency vs mass")
ax2.legend()
fig2.tight_layout()
fig2.savefig("f0_vs_mass.pdf", bbox_inches="tight")
print("Saved: f0_vs_mass.pdf")
