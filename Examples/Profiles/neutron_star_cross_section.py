"""
Neutron star interior cross-section visualisation.

Demonstrates neutron_star_cross_section, which fills concentric annular
regions of a neutron star's cross-section according to user-specified
density or pressure level sets.

Three figures are produced:

  1. Single star with density-based region labels (crust / outer core /
     inner core) for both Newtonian and relativistic models side-by-side.
  2. Four-panel sweep across central density: same EoS, growing inner core
     as rho_c increases.
  3. Pressure-based level sets on the same star, showing how the shell
     boundaries change when switching from density to pressure cuts.

Units
-----
Geometric units (G = c = 1) throughout.
  rho_sat = 0.00045  (nuclear saturation density)
  1 solar mass = 1.477 km

Run
---
    python Examples/neutron_star_cross_section.py
"""

import sys
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "temperance"))

from temperance.plotting import envelope
import matplotlib as mpl
envelope.get_defaults(mpl, fontsize=18)

from solving.analytic_eos import polytropic_eos
from solving.analytic_eos import interpolated_eos

from solving.stellar_models import NewtonianStellarModel, RelativisticStellarModel
from plotting.neutron_star import neutron_star_cross_section

# ---------------------------------------------------------------------------
# EoS and conversion constants
# ---------------------------------------------------------------------------
RHO_SAT = 0.00045           # nuclear saturation density [geometric units]
KM_PER_MSUN = 1.477         # 1 Msun = 1.477 km

# SLy-like polytrope (n=1, Gamma=2)

eos  = interpolated_eos(pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "Compose", "QMC-RMF3", "qmc_rmf3_cgs.csv")))

# Density boundaries that loosely match crust / outer core / inner core
RHO_CRUST_CORE  = 0.5  * RHO_SAT   # crust–outer-core boundary
RHO_OUTER_INNER = 2.0  * RHO_SAT   # outer-core–inner-core boundary

# ---------------------------------------------------------------------------
# Figure 1: Newtonian vs relativistic side-by-side, density level sets
# ---------------------------------------------------------------------------
rho_c = 4.5 * RHO_SAT

model_newt = NewtonianStellarModel(eos, rho_c=rho_c)
model_gr   = RelativisticStellarModel(eos, rho_c=rho_c)

M_newt = model_newt.M
R_newt = model_newt.R * KM_PER_MSUN
M_gr   = model_gr.M
R_gr   = model_gr.R * KM_PER_MSUN

print(f"Newtonian:    M = {M_newt:.3f} Msun,  R = {R_newt:.2f} km")
print(f"Relativistic: M = {M_gr:.3f} Msun,  R = {R_gr:.2f} km")

density_levels = [RHO_CRUST_CORE, RHO_OUTER_INNER]
region_labels  = ["Crust", "Outer core", "Inner core"]
region_colors  = ["#d6c9aa", "#6e9ab5", "#1f4e79"]

fig1, (ax_newt, ax_gr) = plt.subplots(1, 2, figsize=(10, 5))

neutron_star_cross_section(
    model_newt,
    density_levels,
    quantity="rho",
    colors=region_colors,
    labels=region_labels,
    ax=ax_newt,
)
ax_newt.set_title(
    f"Newtonian\n$M={M_newt:.2f}\\,M_\\odot$, $R={R_newt:.1f}$ km",
    fontsize=11,
)

neutron_star_cross_section(
    model_gr,
    density_levels,
    quantity="rho",
    colors=region_colors,
    labels=region_labels,
    ax=ax_gr,
)
ax_gr.set_title(
    f"Relativistic (GR)\n$M={M_gr:.2f}\\,M_\\odot$, $R={R_gr:.1f}$ km",
    fontsize=11,
)

fig1.suptitle(
    r"Neutron star cross-section — density level sets"
    f"\n$\\rho_c = {rho_c/RHO_SAT:.1f}\\,\\rho_{{\\rm sat}}$,  "
    r"polytrope $\Gamma=2$",
    fontsize=12,
)
fig1.tight_layout()
fig1.savefig("ns_cross_section_newt_vs_gr.pdf", bbox_inches="tight")
print("Saved: ns_cross_section_newt_vs_gr.pdf")

# ---------------------------------------------------------------------------
# Figure 2: four-panel density sweep (relativistic), same level sets
# ---------------------------------------------------------------------------
rho_c_values = np.array([ 2.3, 2.7, 5.0]) * RHO_SAT

fig2, axes2 = plt.subplots(1, len(rho_c_values), figsize=(len(rho_c_values) * 4, 4))

for ax, rho_c_i in zip(axes2, rho_c_values):
    model_i = RelativisticStellarModel(eos, rho_c=rho_c_i)
    M_i = model_i.M
    R_i = model_i.R * KM_PER_MSUN

    neutron_star_cross_section(
        model_i,
        density_levels,
        quantity="rho",
        colors=region_colors,
        labels=region_labels if ax is axes2[0] else None,
        ax=ax,
        cmap="magma_r",
        show_mass_fractions=True,
    )
    ax.set_title(
        f"$\\rho_c = {rho_c_i/RHO_SAT:.1f}\\,\\rho_{{\\rm sat}}$\n"
        f"$M={M_i:.2f}\\,M_\\odot$, $R={R_i:.1f}$ km",
        fontsize=14,
    )


fig2.tight_layout()
fig2.savefig("ns_cross_section_density_sweep.pdf", bbox_inches="tight")
print("Saved: ns_cross_section_density_sweep.pdf")

# ---------------------------------------------------------------------------
# Figure 3: pressure-based level sets vs density-based, same star
# ---------------------------------------------------------------------------
model_ref = RelativisticStellarModel(eos, rho_c=4.5 * RHO_SAT)

# Choose pressure boundaries matching the same density cuts via the EoS
p_crust_core  = float(eos.p_of_rho(np.array([RHO_CRUST_CORE]))[0])
p_outer_inner = float(eos.p_of_rho(np.array([RHO_OUTER_INNER]))[0])
pressure_levels = [p_crust_core, p_outer_inner]

fig3, (ax_rho, ax_p) = plt.subplots(1, 2, figsize=(10, 5))

neutron_star_cross_section(
    model_ref,
    density_levels,
    quantity="rho",
    colors=region_colors,
    labels=region_labels,
    ax=ax_rho,
    show_mass_fractions=True
)
ax_rho.set_title("Density level sets\n" r"$\rho_{\rm cut}/\rho_{\rm sat} = 0.5,\;2.0$")

neutron_star_cross_section(
    model_ref,
    pressure_levels,
    quantity="p",
    colors=region_colors,
    labels=region_labels,
    ax=ax_p,
    show_mass_fractions=True
)
ax_p.set_title("Equivalent pressure level sets\n"
               r"$p_{\rm cut}$ from EoS at same $\rho$")

fig3.suptitle(
    r"Density vs pressure cuts — GR, $\rho_c = 4.5\,\rho_{\rm sat}$",
    fontsize=12,
)
fig3.tight_layout()
fig3.savefig("ns_cross_section_rho_vs_p.pdf", bbox_inches="tight")
print("Saved: ns_cross_section_rho_vs_p.pdf")

plt.show()
