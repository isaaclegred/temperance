"""
Circular null orbits (photon orbits) of a static, spherical star.

Circular null orbits are extrema of the optical potential

    U(r) = e^{2 nu(r)} / r^2          (= b^{-2} for impact parameter b),

and dU/dr = 2 U F / [r (r - 2m)] has the sign of

    F(r) = 3 m(r) + 4 pi r^3 p(r) - r,

so circular orbits are roots of F.  At a root, by the TOV equation,
F'(r_c) = 8 pi r_c^2 (e + p) - 1; F' > 0 is a minimum of U (a stable orbit),
F' < 0 a maximum (unstable).

Outside the star F = 3M - r, so the exterior photon sphere r = 3M exists iff
R < 3M.  Since F(0+) = -r < 0 and F(R) = 3M - R, the number of interior roots
is odd for R < 3M (a stable orbit inside, its unstable partner at 3M outside,
plus possibly further pairs) and even for R > 3M (stable and unstable
partner both inside).  Null rays with b^{-2} between the minimum and the
bounding maximum of U are trapped between turning points.

Every function takes a relativistic model exposing r, R, M and the
accessors mass(r), pressure(r), energy_density(r) and nu_of_r(r) (see
stellar_models.RelativisticProfile).  Lengths are in the model's units
(geometric, Msun = 1 for temperance's models).

Moved here from TrappedNullGeos (trapgeo/nullorbits.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.optimize


@dataclass
class NullOrbit:
    """One circular null orbit."""
    r: float
    stable: bool
    location: str          # "interior" or "exterior"
    Fprime: float          # dF/dr at the orbit, 8 pi r^2 (e + p) - 1
    U: float               # optical potential e^{2 nu} / r^2 at the orbit


@dataclass
class NullOrbitSummary:
    """
    Whether a star has a stable null orbit, and the trapping well around it.

    The well is bounded by the outermost unstable orbit outside the
    innermost (deepest) stable one; with several stable/unstable pairs the
    deepest combination is reported.
    """
    has_stable_orbit: bool         # any stable circular null orbit
    n_stable: int
    n_unstable: int
    partner: str | None            # location of the bounding unstable orbit, "interior" / "exterior"
    r_stable: float = np.nan       # radius of the stable orbit at the well minimum
    r_unstable: float = np.nan     # radius of the unstable orbit bounding the well
    well_depth: float = np.nan     # (U_max - U_min) / U_max
    barrier_U: float = np.nan      # U at the bounding unstable orbit (b_c^{-2})
    well_U: float = np.nan         # U at the well minimum
    trap_r_min: float = np.nan     # inner turning point of a ray with b^{-2} = barrier_U
    trap_r_max: float = np.nan     # outer edge of the trapping region (= r_unstable)
    marginal: bool = False         # F nearly touches zero without crossing (a near-tangency)
    surface_flag: bool = False     # R within 1e-6 R of 3M
    R_over_M: float = np.nan

    def as_dict(self):
        return asdict(self)


def F_null(model, r):
    """F(r) = 3 m(r) + 4 pi r^3 p(r) - r, whose roots are circular null orbits."""
    r = np.asarray(r, dtype=float)
    return 3.0 * model.mass(r) + 4.0 * np.pi * r**3 * model.pressure(r) - r


def optical_potential(model, r):
    """U(r) = e^{2 nu(r)} / r^2."""
    r = np.asarray(r, dtype=float)
    return np.exp(2.0 * model.nu_of_r(r)) / r**2


def find_circular_null_orbits(model, n_scan=4000, tangency_tol=1e-6):
    """
    All circular null orbits of ``model``: the roots of F on (0, R], found by
    scanning and refining with Brent's method, plus the exterior photon
    sphere r = 3M when R < 3M.

    The scan uses the model's own radial grid plus ``n_scan`` uniform points,
    so close stable/unstable pairs are resolved down to the profile
    resolution.  A local maximum of F within ``tangency_tol * R`` of zero
    that does not cross it is reported as marginal rather than as a pair.

    Returns
    -------
    orbits : list of NullOrbit, interior ones in increasing r, then the
        exterior one
    marginal : bool
    """
    R, M = model.R, model.M
    grid = np.asarray(model.r, dtype=float)
    r_scan = np.union1d(grid[grid > 0], np.linspace(R * 1e-4, R, n_scan))
    F = F_null(model, r_scan)
    orbits = []
    for i in np.nonzero(np.sign(F[:-1]) * np.sign(F[1:]) < 0)[0]:
        r_c = scipy.optimize.brentq(lambda x: float(F_null(model, x)), r_scan[i], r_scan[i + 1],
                                    xtol=1e-12 * R)
        e, p = float(model.energy_density(r_c)), float(model.pressure(r_c))
        orbits.append(NullOrbit(r_c, bool(F[i + 1] > F[i]), "interior",
                                8.0 * np.pi * r_c**2 * (e + p) - 1.0,
                                float(optical_potential(model, r_c))))
    local_max = np.nonzero((F[1:-1] > F[:-2]) & (F[1:-1] > F[2:]))[0] + 1
    marginal = bool(np.any(np.abs(F[local_max]) < tangency_tol * R))
    if R < 3.0 * M:
        orbits.append(NullOrbit(3.0 * M, False, "exterior", -1.0, 1.0 / (27.0 * M**2)))
    return orbits, marginal


def classify_null_orbits(model, **kwargs) -> NullOrbitSummary:
    """
    Whether ``model`` has a stable circular null orbit, and the trapping
    well it sits in.  Extra kwargs go to find_circular_null_orbits.
    """
    orbits, marginal = find_circular_null_orbits(model, **kwargs)
    stable = [orbit for orbit in orbits if orbit.stable]
    unstable = [orbit for orbit in orbits if not orbit.stable]
    summary = NullOrbitSummary(has_stable_orbit=bool(stable), n_stable=len(stable),
                               n_unstable=len(unstable), partner=None, marginal=marginal,
                               R_over_M=model.R / model.M,
                               surface_flag=bool(abs(model.R - 3.0 * model.M) < 1e-6 * model.R))
    if not stable:
        return summary
    well = min(stable, key=lambda orbit: orbit.U)
    barrier = max((orbit for orbit in unstable if orbit.r > well.r), key=lambda orbit: orbit.U)
    summary.partner = barrier.location
    summary.r_stable, summary.r_unstable = well.r, barrier.r
    summary.well_U, summary.barrier_U = well.U, barrier.U
    summary.well_depth = (barrier.U - well.U) / barrier.U
    # inner turning point of a ray at the barrier top (the wall from the centrifugal term)
    r_in = np.linspace(model.R * 1e-4, well.r, 2000)
    below = np.nonzero(optical_potential(model, r_in) <= barrier.U)[0]
    summary.trap_r_min = float(r_in[below[0]]) if below.size else np.nan
    summary.trap_r_max = barrier.r
    return summary
