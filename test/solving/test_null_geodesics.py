"""
Tests for temperance.solving.null_geodesics.
"""

import numpy as np
import pytest

from temperance.solving.analytic_eos import piecewise_eos
from temperance.solving.null_geodesics import (F_null, classify_null_orbits, find_circular_null_orbits,
                                               optical_potential)
from temperance.solving.stellar_models import (RelativisticStellarModel, TolmanVIIStellarModel,
                                               UniformDensityStellarModel)


def test_uniform_density_has_stable_orbit_iff_R_lt_3M():
    compact = classify_null_orbits(UniformDensityStellarModel(2.9))
    assert compact.has_stable_orbit and compact.partner == "exterior"
    assert compact.r_unstable == pytest.approx(3.0)
    assert compact.barrier_U == pytest.approx(1 / 27)
    assert not classify_null_orbits(UniformDensityStellarModel(3.05)).has_stable_orbit


def test_tolman_vii_traps_beyond_3M_up_to_3_20():
    summary = classify_null_orbits(TolmanVIIStellarModel(3.1))
    assert summary.has_stable_orbit and summary.partner == "interior"
    assert summary.r_stable < summary.r_unstable < 3.1
    assert classify_null_orbits(TolmanVIIStellarModel(3.19)).has_stable_orbit
    assert not classify_null_orbits(TolmanVIIStellarModel(3.21)).has_stable_orbit
    assert not classify_null_orbits(TolmanVIIStellarModel(3.4)).has_stable_orbit


def test_orbits_are_extrema_of_the_optical_potential():
    model = TolmanVIIStellarModel(3.1)
    orbits, marginal = find_circular_null_orbits(model)
    assert not marginal and len(orbits) == 2
    for orbit in orbits:
        assert F_null(model, orbit.r) == pytest.approx(0.0, abs=1e-9)
        # stability from the sign change agrees with the analytic F'
        assert orbit.stable == (orbit.Fprime > 0)
        h = 1e-4 * orbit.r
        U0, U_minus, U_plus = optical_potential(model, [orbit.r, orbit.r - h, orbit.r + h])
        assert (U_minus > U0 and U_plus > U0) if orbit.stable else (U_minus < U0 and U_plus < U0)


def test_trapping_region():
    summary = classify_null_orbits(TolmanVIIStellarModel(3.1))
    assert 0 < summary.trap_r_min < summary.r_stable < summary.trap_r_max == summary.r_unstable
    assert 0 < summary.well_depth < 1
    assert summary.well_depth == pytest.approx((summary.barrier_U - summary.well_U) / summary.barrier_U)


def test_compact_tov_star_with_causal_core():
    eos = piecewise_eos([{"type": "rest_mass_polytrope", "Gamma": 2.0, "K": 100.0, "rho_max": 2 * 0.00045},
                         {"type": "css", "cs2": 1.0, "e_max": 0.2}])
    star = RelativisticStellarModel.from_logenthalpy(eos, 1.0, n_points=3000)
    assert star.R < 3 * star.M
    summary = classify_null_orbits(star)
    assert summary.has_stable_orbit and summary.partner == "exterior"
    assert summary.r_stable < star.R
    less_compact = RelativisticStellarModel.from_logenthalpy(eos, 0.6, n_points=3000)
    assert not classify_null_orbits(less_compact).has_stable_orbit


def test_matches_trapped_null_geos():
    """Same orbits as trapgeo.nullorbits on the same Tolman VII star (when available)."""
    trapgeo_background = pytest.importorskip("trapgeo.background")
    from trapgeo.nullorbits import classify
    theirs = classify(trapgeo_background.tolman_vii(3.1))
    ours = classify_null_orbits(TolmanVIIStellarModel(3.1))
    assert ours.r_stable == pytest.approx(theirs.r_stable, rel=1e-6)
    assert ours.r_unstable == pytest.approx(theirs.r_unstable, rel=1e-6)
    assert ours.well_depth == pytest.approx(theirs.well_depth, rel=1e-5)
    assert ours.trap_r_min == pytest.approx(theirs.trap_r_min, rel=1e-6)
