"""
Speed of sound vs baryon density from a CompOSE 3-D EoS table.

Steps:
  1. Download the QMC-RMF3 table from CompOSE.
  2. Extract the cold (T = T_min) beta-equilibrium 1-D slice.
  3. Compute cs^2 = dP/d(epsilon) numerically.
  4. Plot cs^2 / c^2 vs n_B.

All thermodynamic quantities are kept in nuclear units throughout
(MeV, fm^{-3}) — no CGS or geometric conversion needed.

Run
---
    python Examples/Compose/cs2_vs_nb.py
"""

import sys
import os
import types

import numpy as np
import matplotlib.pyplot as plt

# Stub the top-level temperance package so its __init__.py (which requires
# optional heavy dependencies) is not executed.
_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _repo)
_stub = types.ModuleType("temperance")
_stub.__path__ = [os.path.join(_repo, "temperance")]
_stub.__package__ = "temperance"
sys.modules.setdefault("temperance", _stub)

import temperance.external.read_3d_compose_table as tmcomp

# ---------------------------------------------------------------------------
# 1. Download (cached after first run)
# ---------------------------------------------------------------------------
URL   = "https://compose.obspm.fr/download//3D/Hempel_SchaffnerBielich/hs_dd2_compose/with_electrons/"
LABEL = "hs(dd2)"

thermo_path, nb_path, t_path, yq_path = tmcomp.download_compose_3d_files(
    URL, LABEL, dest_dir="."
)
thermo, nb, t, yq = tmcomp.read_compose_3d_table(
    thermo_path, nb_path, t_path, yq_path
)

# ---------------------------------------------------------------------------
# 2. Cold beta-equilibrium 1-D slice
#
# Returns a DataFrame with columns: nb [fm^-3], Yq_opt, p_per_nb [MeV],
# e_per_nbmn_minus_1 [dimensionless], and other thermo quantities.
# ---------------------------------------------------------------------------
eos_1d = tmcomp.find_1d_eos_by_taking_T_equal_Tmin_then_optimizing_ye(
    thermo, nb, t, yq, order=2
)

nb_arr = eos_1d["nb"].values                  # fm^-3
M_N    = 939.565                              # neutron mass [MeV]

P   = eos_1d["p_per_nb"].values * nb_arr      # pressure            [MeV/fm^3]
eps = (eos_1d["e_per_nbmn_minus_1"].values + 1.0) * M_N * nb_arr  # energy density [MeV/fm^3]

# ---------------------------------------------------------------------------
# 3. Speed of sound:  cs^2/c^2 = dP/d(epsilon)
# ---------------------------------------------------------------------------
cs2 = np.gradient(P, eps)

print(f"n_B range : {nb_arr.min():.3f} – {nb_arr.max():.3f}  fm^-3")
print(f"cs^2/c^2  : {cs2.min():.3f} – {cs2.max():.3f}")

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(nb_arr, cs2, lw=2.0, color="k")
ax.axhline(1.0 / 3.0, color="grey", ls=":", lw=1.0, label=r"Conformal limit  $c_s^2 = c^2/3$")
ax.axvline(0.16, color="grey", ls="--", lw=0.8, label=r"$n_0 = 0.16\ \mathrm{fm}^{-3}$")

ax.set_xlabel(r"$n_B\ [\mathrm{fm}^{-3}]$")
ax.set_ylabel(r"$c_s^2 / c^2$")
ax.set_title("QMC-RMF3 (CompOSE) — speed of sound")
ax.set_xlim(nb_arr.min(), nb_arr.max())
ax.set_ylim(bottom=0.0)
ax.legend(fontsize=9)

fig.tight_layout()
out = "cs2_vs_nb.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.show()
