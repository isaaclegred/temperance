"""
Tests for the TOV and Newtonian stellar structure solvers.

All tests use a simple polytropic EOS (p = K rho^Gamma) with
K = 123.6, Gamma = 2 at a central density of 3 x nuclear saturation,
which produces a ~1.4 Msun neutron star in GR.

Units: geometric (G = c = 1).  1 Msun = 1.477 km in these units.
"""

import numpy as np
import pytest

from solving.analytic_eos import polytropic_eos, interpolated_eos
from solving.solve_tov import (
    lindblom_solver,
    solve_tov,
    construct_star,
    get_tov_family,
    eta_to_love_number,
)
from solving.solve_newtonian_structure import (
    newtonian_solver,
    solve_newtonian,
    construct_newtonian_star,
    get_newtonian_family,
    eta_to_love_number_newtonian,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Nuclear saturation density in geometric units
RHO_SAT = 0.00045

# Polytropic EOS used by all tests
POLY_K = 123.6
POLY_GAMMA = 2.0

@pytest.fixture(scope="module")
def eos():
    return polytropic_eos(K=POLY_K, Gamma=POLY_GAMMA)


@pytest.fixture(scope="module")
def rho_c():
    """Central density: 3 x nuclear saturation."""
    return 3.0 * RHO_SAT


@pytest.fixture(scope="module")
def tov_solution(eos, rho_c):
    """Cached TOV integration result."""
    return solve_tov(eos, rho_c)


@pytest.fixture(scope="module")
def newt_solution(eos, rho_c):
    """Cached Newtonian integration result."""
    return solve_newtonian(eos, rho_c)


# ---------------------------------------------------------------------------
# EOS tests
# ---------------------------------------------------------------------------

class TestPolytropicEOS:
    def test_pressure_positive(self, eos):
        rhos = np.geomspace(0.1 * RHO_SAT, 10 * RHO_SAT, 50)
        assert np.all(eos.p_of_rho(rhos) > 0)

    def test_energy_exceeds_rest_mass(self, eos):
        """Total energy density must be >= baryon density."""
        rhos = np.geomspace(0.1 * RHO_SAT, 10 * RHO_SAT, 50)
        assert np.all(eos.e_of_rho(rhos) >= rhos)

    def test_logenthalpy_roundtrip(self, eos):
        """logenthalpy_of_rho and rho_of_logenthalpy are mutual inverses."""
        rhos = np.geomspace(0.5 * RHO_SAT, 5 * RHO_SAT, 20)
        lnhs = eos.logenthalpy_of_rho(rhos)
        rhos_rt = eos.rho_of_logenthalpy(lnhs)
        np.testing.assert_allclose(rhos_rt, rhos, rtol=1e-10)

    def test_sound_speed_causal(self, eos):
        """cs2 must be in (0, 1) everywhere (subluminal)."""
        rhos = np.geomspace(0.1 * RHO_SAT, 10 * RHO_SAT, 50)
        lnhs = eos.logenthalpy_of_rho(rhos)
        cs2 = eos.cs2_of_logenthalpy(lnhs)
        assert np.all(cs2 > 0)
        assert np.all(cs2 < 1)

    def test_pressure_increases_with_density(self, eos):
        rhos = np.geomspace(0.1 * RHO_SAT, 10 * RHO_SAT, 50)
        p = eos.p_of_rho(rhos)
        assert np.all(np.diff(p) > 0)


# ---------------------------------------------------------------------------
# TOV solver tests
# ---------------------------------------------------------------------------

class TestTOVSolver:
    def test_solution_shape(self, tov_solution):
        lnhs, sols = tov_solution
        assert lnhs.ndim == 1
        assert sols.shape == (len(lnhs), 4)

    def test_radial_coordinate_increases(self, tov_solution):
        lnhs, sols = tov_solution
        r = np.sqrt(sols[:, 0])
        assert np.all(np.diff(r) > 0), "r must increase from centre to surface"

    def test_enclosed_mass_increases(self, tov_solution):
        lnhs, sols = tov_solution
        r = np.sqrt(sols[:, 0])
        m = sols[:, 1] * r
        assert np.all(np.diff(m) > 0), "m(r) must be non-decreasing"

    def test_mass_in_physical_range(self, tov_solution):
        """Gravitational mass should be between 0.5 and 3 solar masses."""
        lnhs, sols = tov_solution
        r_surf = np.sqrt(sols[-1, 0])
        M = r_surf * sols[-1, 1]   # in geometric units (solar masses)
        assert 0.5 < M < 3.0, f"Mass {M:.3f} outside expected range"

    def test_radius_in_physical_range(self, tov_solution):
        """Circumferential radius should be between 5 and 20 km."""
        lnhs, sols = tov_solution
        R_km = np.sqrt(sols[-1, 0]) * 1.477
        assert 5 < R_km < 20, f"Radius {R_km:.2f} km outside expected range"

    def test_compactness_below_buchdahl(self, tov_solution):
        """Compactness M/R must be < 4/9 (Buchdahl limit)."""
        lnhs, sols = tov_solution
        C = sols[-1, 1]   # v at surface = M/R
        assert C < 4.0 / 9.0, f"Compactness {C:.4f} exceeds Buchdahl limit"

    def test_tidal_deformability_positive(self, tov_solution):
        lnhs, sols = tov_solution
        C = float(sols[-1, 1])
        eta_R = float(sols[-1, 2])
        Lambda = eta_to_love_number(eta_R, C)
        assert Lambda > 0, f"Tidal deformability {Lambda:.1f} is non-positive"

    def test_construct_star_dataframe(self, eos, rho_c):
        df = construct_star(eos, rho_c)
        assert set(df.columns) >= {"r", "m", "baryon_density", "m_baryon"}
        assert len(df) > 10
        assert np.all(df["r"] > 0)
        assert np.all(df["m"] > 0)

    def test_get_tov_family_columns(self, eos):
        densities = np.linspace(2 * RHO_SAT, 5 * RHO_SAT, 5)
        family = get_tov_family(eos, densities)
        assert set(family.columns) >= {"M", "R", "Lambda", "M_baryon"}
        assert np.all(family["M"] > 0)
        assert np.all(family["R"] > 0)
        assert np.all(family["Lambda"] > 0)

    def test_mass_increases_with_central_density(self, eos):
        """For a stiff polytrope at low central densities, M should increase."""
        densities = np.linspace(1.5 * RHO_SAT, 4 * RHO_SAT, 6)
        family = get_tov_family(eos, densities)
        assert family["M"].iloc[-1] > family["M"].iloc[0]


# ---------------------------------------------------------------------------
# Newtonian solver tests
# ---------------------------------------------------------------------------

class TestNewtonianSolver:
    def test_solution_shape(self, newt_solution):
        lnhs, sols = newt_solution
        assert lnhs.ndim == 1
        assert sols.shape == (len(lnhs), 4)

    def test_radial_coordinate_increases(self, newt_solution):
        lnhs, sols = newt_solution
        r = np.sqrt(sols[:, 0])
        assert np.all(np.diff(r) > 0)

    def test_enclosed_mass_increases(self, newt_solution):
        lnhs, sols = newt_solution
        r = np.sqrt(sols[:, 0])
        m = sols[:, 1] * r
        assert np.all(np.diff(m) > 0)

    def test_newtonian_radius_larger_than_gr(self, eos, rho_c, tov_solution, newt_solution):
        """Newtonian stars are less compact than GR stars at the same rho_c."""
        _, gr_sols = tov_solution
        _, newt_sols = newt_solution
        R_gr   = np.sqrt(gr_sols[-1, 0])
        R_newt = np.sqrt(newt_sols[-1, 0])
        assert R_newt > R_gr, (
            f"Expected R_Newtonian ({R_newt:.4f}) > R_GR ({R_gr:.4f})"
        )

    def test_newtonian_love_number_positive(self, newt_solution):
        lnhs, sols = newt_solution
        eta_R = float(sols[-1, 2])
        k2 = eta_to_love_number_newtonian(eta_R)
        assert k2 > 0, f"Newtonian k2 = {k2:.4f} is non-positive"

    def test_construct_newtonian_star_dataframe(self, eos, rho_c):
        df = construct_newtonian_star(eos, rho_c)
        assert set(df.columns) >= {"r", "m", "baryon_density", "m_baryon"}
        assert np.all(df["r"] > 0)
        assert np.all(df["m"] > 0)

    def test_get_newtonian_family_columns(self, eos):
        densities = np.linspace(2 * RHO_SAT, 5 * RHO_SAT, 5)
        family = get_newtonian_family(eos, densities)
        assert set(family.columns) >= {"M", "R", "k2", "M_baryon"}
        assert np.all(family["M"] > 0)
        assert np.all(family["R"] > 0)
