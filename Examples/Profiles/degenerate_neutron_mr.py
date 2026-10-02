"""
M-R curve for a degenerate nonrelativistic neutron gas (GR/TOV).

The equation of state is a Gamma=5/3 polytrope whose coefficient K is derived
from the nonrelativistic Fermi-gas formula

    P = K_NR * rho^(5/3),  K_NR = (3 pi^2)^(2/3) hbar^2 / (5 m_n^(8/3))

converted to the geometric units (G = c = 1) used by the solvers.

Run
---
    python Examples/Profiles/degenerate_neutron_mr.py
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "temperance"))
from solving.analytic_eos import polytropic_eos
from solving.solve_relativistic import get_tov_family

import matplotlib as mpl
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["mathpazo"],
})

# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------
HBAR   = 1.054571817e-27   # erg s
M_N    = 1.674927471e-24   # g (neutron mass)
C      = 2.99792458e10     # cm / s

# Geometric-unit conventions used throughout the codebase
RHO_SAT_GEOM = 0.00045     # nuclear saturation density [geometric]
RHO_SAT_CGS  = 2.8e14      # nuclear saturation density [g/cm^3]
GEOM_TO_CGS  = RHO_SAT_CGS / RHO_SAT_GEOM

# ---------------------------------------------------------------------------
# Derive K for the nonrelativistic degenerate neutron Fermi gas
#
#   P_cgs = K_NR * rho_cgs^(5/3)     [dyn/cm^2]
#
# To convert to geometric units where P and rho share the same unit:
#   rho_geom = rho_cgs / GEOM_TO_CGS
#   P_geom   = P_cgs   / (C^2 * GEOM_TO_CGS)
#
# => P_geom = (K_NR / C^2 * GEOM_TO_CGS^(2/3)) * rho_geom^(5/3)
#           = K_GEOM * rho_geom^(5/3)
# ---------------------------------------------------------------------------
K_NR   = (3.0 * np.pi**2)**(2.0/3.0) * HBAR**2 / (5.0 * M_N**(8.0/3.0))
K_GEOM = K_NR / C**2 * GEOM_TO_CGS**(2.0/3.0)
GAMMA  = 5.0 / 3.0

print(f"Nonrelativistic neutron Fermi gas polytrope")
print(f"  K_NR   = {K_NR:.4e}  g^(-2/3) cm^4 s^-2")
print(f"  K_geom = {K_GEOM:.4f}  (geometric units, rho_sat = {RHO_SAT_GEOM})")

eos   = polytropic_eos(K=K_GEOM, Gamma=GAMMA)
rho_c = np.geomspace(0.1* RHO_SAT_GEOM, 3e4 * RHO_SAT_GEOM, 60)

print("\nSolving TOV equations ...")
gr      = get_tov_family(eos, rho_c)
idx_max = gr["M"].idxmax()
print(f"  M_max = {gr['M'].max():.3f} M_sun  at R = {gr['R'].iloc[idx_max]:.1f} km")

# Phenomenological EoS: stiffer polytrope
K_PHENO, GAMMA_PHENO = 1e5, 3.0
eos_pheno  = polytropic_eos(K=K_PHENO, Gamma=GAMMA_PHENO)
gr_pheno   = get_tov_family(eos_pheno, rho_c)
idx_max_ph = gr_pheno["M"].idxmax()
print(f"\nPhenomenological EoS (K={K_PHENO}, Gamma={GAMMA_PHENO})")
print(f"  M_max = {gr_pheno['M'].max():.3f} M_sun  at R = {gr_pheno['R'].iloc[idx_max_ph]:.1f} km")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 4.0))

ax.plot(gr["R"],       gr["M"],       lw=2.0, color="k",
        label=fr"$\Gamma={GAMMA:.2f},\ K={K_GEOM:.2f}$ (free neutron gas)")
ax.scatter(gr["R"].iloc[idx_max], gr["M"].max(),
           s=80, zorder=5, color="k",
           label=fr"$M_{{\max}} = {gr['M'].max():.2f}\,M_\odot$")

ax.plot(gr_pheno["R"], gr_pheno["M"], lw=2.0, color="C0", ls="--",
        label=fr"$\Gamma={GAMMA_PHENO},\ K={K_PHENO/10**(int(np.log10(K_PHENO))) } \times 10^{{{int(np.log10(K_PHENO))}}}$ (phenomenological)")
ax.scatter(gr_pheno["R"].iloc[idx_max_ph], gr_pheno["M"].max(),
           s=80, zorder=5, color="C0",
           label=fr"$M_{{\max}} = {gr_pheno['M'].max():.2f}\,M_\odot$")

ax.set_xlabel("$R  [\mathrm{km}]$")
ax.set_ylabel(r"$M\;[M_\odot]$")
ax.set_title(r"Polytropic neutron star M-R curves")
ax.legend(fontsize=8)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

fig.tight_layout()
out = "degenerate_neutron_mr.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.show()
