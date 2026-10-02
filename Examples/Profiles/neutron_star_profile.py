"""
Neutron star interior profile plots.

Demonstrates neutron_star_profile, which plots radial profiles of density,
pressure, sound speed, mass integrand, and other quantities for one or
more stellar models.

Three figures are produced:

  1. Default four-panel profile (rho, p, cs, dm_dr) for a single GR star.
  2. Newtonian vs GR comparison for the same EoS and central density,
     showing profiles side-by-side and highlighting which quantities are
     model-type specific (e for GR, phi for Newtonian).
  3. Central-density sweep: five GR stars overlaid on the same axes,
     illustrating how the profiles evolve with compactness.

Units
-----
Geometric units (G = c = 1) throughout.
  rho_sat = 0.00045  (nuclear saturation density)
  1 solar mass = 1.477 km

Run
---
    python Examples/neutron_star_profile.py
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "temperance"))

from solving.analytic_eos import polytropic_eos
from solving.stellar_models import NewtonianStellarModel, RelativisticStellarModel
from plotting.neutron_star import neutron_star_profile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RHO_SAT    = 0.00045      # nuclear saturation density [geometric units]
KM_PER_MSUN = 1.477       # 1 Msun = 1.477 km

eos = polytropic_eos(K=0.0195, Gamma=2.0)

# ---------------------------------------------------------------------------
# Figure 1: single GR star, default four-panel profile
# ---------------------------------------------------------------------------
rho_c   = 4.5 * RHO_SAT
model_gr = RelativisticStellarModel(eos, rho_c=rho_c)
M = model_gr.M
R = model_gr.R * KM_PER_MSUN
print(f"GR star: M = {M:.3f} Msun,  R = {R:.2f} km")

fig1, axes1 = neutron_star_profile(model_gr)
fig1.suptitle(
    r"GR interior profiles — polytrope $\Gamma=2$, "
    f"$M={M:.2f}\\,M_\\odot$, $R={R:.1f}$ km",
    fontsize=11,
)
fig1.savefig("ns_profile_single.pdf", bbox_inches="tight")
print("Saved: ns_profile_single.pdf")

# ---------------------------------------------------------------------------
# Figure 2: Newtonian vs GR — full quantity set including model-specific ones
# ---------------------------------------------------------------------------
model_newt = NewtonianStellarModel(eos, rho_c=rho_c)
M_n = model_newt.M
R_n = model_newt.R * KM_PER_MSUN
print(f"Newtonian star: M = {M_n:.3f} Msun,  R = {R_n:.2f} km")

# 'e'   is GR-only  -> silently absent on the Newtonian line
# 'phi' is Newt-only -> silently absent on the GR line
quantities_compare = ["rho", "p", "cs", "dm_dr", "e", "phi"]

fig2, axes2 = neutron_star_profile(
    [model_gr, model_newt],
    quantities=quantities_compare,
    model_labels=["GR", "Newtonian"],
    ncols=3,
)
fig2.suptitle(
    r"Newtonian vs GR profiles — polytrope $\Gamma=2$, "
    fr"$\rho_c = {rho_c/RHO_SAT:.1f}\,\rho_{{\rm sat}}$",
    fontsize=11,
)
fig2.savefig("ns_profile_newt_vs_gr.pdf", bbox_inches="tight")
print("Saved: ns_profile_newt_vs_gr.pdf")

# ---------------------------------------------------------------------------
# Figure 3: central-density sweep — five GR stars overlaid
# ---------------------------------------------------------------------------
rho_c_values = np.array([1.5, 2.5, 3.5, 4.5, 6.0]) * RHO_SAT

models_sweep = [RelativisticStellarModel(eos, rho_c=rc) for rc in rho_c_values]
labels_sweep  = [
    fr"$\rho_c = {rc/RHO_SAT:.1f}\,\rho_{{\rm sat}}$"
    for rc in rho_c_values
]

fig3, axes3 = neutron_star_profile(
    models_sweep,
    quantities=["rho", "p", "cs", "dm_dr"],
    model_labels=labels_sweep,
    linestyles=["-", "--", "-.", ":", (0, (3, 1, 1, 1))],
)
fig3.suptitle(
    r"GR profiles — central-density sweep, polytrope $\Gamma=2$",
    fontsize=11,
)
fig3.savefig("ns_profile_density_sweep.pdf", bbox_inches="tight")
print("Saved: ns_profile_density_sweep.pdf")

plt.show()
