"""
Tests for the axial (odd-parity) quasinormal modes in
temperance.solving.solve_oscillations.RelativisticModes.
"""

import numpy as np
import pytest

from temperance.solving.solve_oscillations import RelativisticModes
from temperance.solving.stellar_models import TolmanVIIStellarModel, UniformDensityStellarModel


@pytest.fixture(scope="module")
def ultracompact():
    """Uniform density, R/M = 2.3: an exterior-barrier well with two trapped l = 2 modes."""
    modes = RelativisticModes(UniformDensityStellarModel(2.3))
    well, trapped = modes.trapped_axial_modes(2, n_scan=80)
    return modes, well, trapped


def test_exterior_potential_is_regge_wheeler():
    modes = RelativisticModes(UniformDensityStellarModel(2.5))
    r = np.array([5.0, 10.0, 40.0])
    f = 1 - 2 / r
    np.testing.assert_allclose(modes.axial_potential(r, 2), f * (6 / r**2 - 6 / r**3), rtol=1e-10)
    np.testing.assert_allclose(modes.axial_potential(r, 3), f * (12 / r**2 - 6 / r**3), rtol=1e-10)


def test_tortoise_coordinate():
    modes = RelativisticModes(UniformDensityStellarModel(2.5))
    assert modes.tortoise(0.0)[0] == pytest.approx(0.0)
    r = np.linspace(0.1, 20, 400)
    rs = modes.tortoise(r)
    assert np.all(np.diff(rs) > 0)
    outside = r > 3.0
    # dr*/dr = 1 / (1 - 2M/r) outside
    np.testing.assert_allclose(np.gradient(rs, r)[outside][1:-1], (1 / (1 - 2 / r))[outside][1:-1], rtol=1e-3)


def test_potential_well(ultracompact):
    modes, well, _ = ultracompact
    assert well.exists and well.barrier_location == "exterior"
    assert well.r_min < modes.model.R < well.r_max
    assert well.V_min < well.V_max
    # the exterior barrier of Regge-Wheeler l = 2 peaks near r = 3.28 M
    assert well.r_max == pytest.approx(3.28, rel=1e-2)


def test_uniform_density_trapped_l2_spectrum(ultracompact):
    modes, well, trapped = ultracompact
    assert len(trapped) == 2 and all(mode.trapped for mode in trapped)
    fundamental, overtone = trapped
    assert fundamental.quality_factor == pytest.approx(9.5e3, rel=0.02)
    assert fundamental.quality_factor > overtone.quality_factor > 100
    assert [fundamental.n, overtone.n] == [0, 1]
    for mode in trapped:
        assert mode.omega.imag < 0
        assert well.V_min < mode.omega.real**2 < well.V_max
        assert abs(modes.axial_wronskian(mode.omega, 2)) < 1e-7
        assert mode.damping_time == pytest.approx(1 / abs(mode.omega.imag))
        assert mode.quality_factor == pytest.approx(mode.omega.real / (-2 * mode.omega.imag))
        assert mode.mode_type == "w" and mode.solver_info["parity"] == "odd"


def test_wkb_agrees_with_solver_for_fundamental(ultracompact):
    _, _, (fundamental, _) = ultracompact
    assert fundamental.solver_info["wkb_Q"] == pytest.approx(fundamental.quality_factor, rel=0.3)


def test_trapped_modes_are_cached_copies(ultracompact):
    modes, _, trapped = ultracompact
    again = modes.trapped_axial_modes(2, n_scan=80)[1]
    assert [mode.omega for mode in again] == [mode.omega for mode in trapped]
    again[0].solver_info["changed"] = True
    again[0].m = 2
    fresh = modes.trapped_axial_modes(2, n_scan=80)[1][0]
    assert "changed" not in fresh.solver_info and fresh.m == 0


def test_nonradial_dispatch(ultracompact):
    modes, _, trapped = ultracompact
    # (the default scan resolution also finds both modes)
    w_modes = modes.compute_w_modes(2, n_max=5)
    assert [mode.omega for mode in w_modes] == pytest.approx([mode.omega for mode in trapped])
    assert modes.compute_nonradial_mode(2, 1, 0, mode_family="w", parity="odd").omega == pytest.approx(trapped[0].omega)
    with pytest.raises(ValueError, match="not found"):
        modes.compute_nonradial_mode(2, 0, 2, mode_family="w", parity="odd")
    with pytest.raises(ValueError, match="trivial"):
        modes.compute_nonradial_mode(2, 0, 0, mode_family="f", parity="odd")
    with pytest.raises(NotImplementedError):
        modes.compute_nonradial_mode(2, 0, 0, mode_family="f", parity="even")


def test_less_compact_star_has_no_trapped_l2_modes():
    modes = RelativisticModes(UniformDensityStellarModel(3.5))
    assert modes.compute_w_modes(2, n_max=3) == []
    assert modes.well_phase(2) < 0.25


def test_minimal_trapping_ell():
    assert RelativisticModes(UniformDensityStellarModel(2.3)).minimal_trapping_ell() == 2
    # values from TrappedNullGeos (trapgeo.axial.minimal_trapping_ell)
    assert RelativisticModes(TolmanVIIStellarModel(3.1)).minimal_trapping_ell() == 16
    assert RelativisticModes(UniformDensityStellarModel(2.9)).minimal_trapping_ell() == 126


def test_matches_trapped_null_geos(ultracompact):
    """Same trapped spectrum as trapgeo.axial on the same star (when available)."""
    trapgeo_background = pytest.importorskip("trapgeo.background")
    from trapgeo.axial import AxialSolver
    _, theirs = AxialSolver(trapgeo_background.uniform_density(2.3), ell=2).trapped_modes(n_scan=80)
    _, _, ours = ultracompact
    # trapgeo's accessors return the exterior e = 0 at r = R exactly, smearing the
    # surface density jump over one grid cell; that shifts the modes slightly
    for mine, other in zip(ours, theirs):
        assert mine.omega.real == pytest.approx(other.omega.real, rel=1e-5)
        assert mine.quality_factor == pytest.approx(other.Q, rel=1e-3)
