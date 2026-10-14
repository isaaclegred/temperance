"""
Tests for the relativistic stellar models in temperance.solving.stellar_models.
"""

import numpy as np
import pytest

from temperance.solving.analytic_eos import polytropic_eos
from temperance.solving.stellar_models import (PrescribedDensityStellarModel, RelativisticStellarModel,
                                               TolmanVIIStellarModel, UniformDensityStellarModel)

EOS = polytropic_eos(K=123.6, Gamma=2.0)
RHO_C = 3 * 0.00045


@pytest.fixture(scope="module")
def star():
    return RelativisticStellarModel(EOS, RHO_C, n_points=2000)


def tov_dnu_dr(model, r):
    m, p = model.mass(r), model.pressure(r)
    return (m + 4 * np.pi * r**3 * p) / (r * (r - 2 * m))


def tov_dp_dr(model, r):
    e, p = model.energy_density(r), model.pressure(r)
    return -(e + p) * tov_dnu_dr(model, r)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", [
    RelativisticStellarModel(EOS, RHO_C), UniformDensityStellarModel(2.6), TolmanVIIStellarModel(3.2)],
    ids=["tov", "uniform", "tolman"])
def test_exterior_is_schwarzschild(model):
    r = model.R * np.array([1.01, 2.0, 10.0])
    np.testing.assert_allclose(model.mass(r), model.M)
    np.testing.assert_allclose(model.pressure(r), 0.0)
    np.testing.assert_allclose(model.energy_density(r), 0.0)
    np.testing.assert_allclose(model.nu_of_r(r), 0.5 * np.log(1 - 2 * model.M / r))
    np.testing.assert_allclose(model.lam_of_r(r), -0.5 * np.log(1 - 2 * model.M / r))
    # continuous at the surface
    below, above = model.R * (1 - 1e-9), model.R * (1 + 1e-9)
    for accessor in (model.mass, model.nu_of_r, model.lam_of_r):
        assert accessor(below) == pytest.approx(accessor(above), rel=1e-6)


def test_accessors_match_profile_inside(star):
    inside = slice(5, -5, 37)
    for accessor, array in [(star.mass, star.m), (star.pressure, star.p),
                            (star.energy_density, star.e), (star.nu_of_r, star.nu), (star.lam_of_r, star.lam)]:
        np.testing.assert_allclose(accessor(star.r[inside]), array[inside], rtol=1e-10)
    # defined down to the centre
    assert star.mass(0.0) == 0.0
    assert star.pressure(0.0) == pytest.approx(star.p[0])


def test_interpolate_is_exterior_aware(star):
    r = 2 * star.R
    assert star.interpolate("m")(r) == pytest.approx(star.M)
    assert star.interpolate("nu")(r) == pytest.approx(0.5 * np.log(1 - star.M / star.R))
    assert star.interpolate("p")(r) == 0.0
    assert star.interpolate("rho")(r) == 0.0
    assert star.interpolate("rho")(star.r[100]) == pytest.approx(star.rho[100])


# ---------------------------------------------------------------------------
# Metric potential and constructors
# ---------------------------------------------------------------------------

def test_nu_is_exact_function_of_enthalpy(star):
    np.testing.assert_allclose(star.nu - star.nu[-1], -(star.lnh - star.lnh[-1]), rtol=1e-12)
    assert star.nu[-1] == pytest.approx(0.5 * np.log(1 - 2 * star.M / star.R))
    # and it satisfies the TOV relation d nu / dr = (m + 4 pi r^3 p) / (r (r - 2m))
    r = np.linspace(0.1, 0.95, 30) * star.R
    np.testing.assert_allclose(np.gradient(star.nu_of_r(r), r)[1:-1], tov_dnu_dr(star, r)[1:-1], rtol=2e-3)


def test_constructors_agree(star):
    by_lnh = RelativisticStellarModel.from_logenthalpy(EOS, star.lnh_c, n_points=2000)
    by_pressure = RelativisticStellarModel.from_central_pressure(EOS, star.central_pressure, n_points=2000)
    for model in (by_lnh, by_pressure):
        assert model.M == pytest.approx(star.M, rel=1e-8)
        assert model.R == pytest.approx(star.R, rel=1e-8)
    assert by_lnh.rho_c == pytest.approx(RHO_C)
    with pytest.raises(ValueError, match="exactly one"):
        RelativisticStellarModel(EOS, RHO_C, lnh_c=0.2)


def test_custom_lnh_grid(star):
    grid = np.geomspace(star.lnh_c, 1e-10, 800)
    model = RelativisticStellarModel(EOS, RHO_C, lnh_grid=grid)
    np.testing.assert_allclose(model.lnh, grid)
    assert model.M == pytest.approx(star.M, rel=1e-6)
    assert model.R == pytest.approx(star.R, rel=1e-6)
    # the crust is resolved: many points within 1% of the surface
    assert np.sum(model.r > 0.99 * model.R) > 100
    with pytest.raises(ValueError, match="decrease"):
        RelativisticStellarModel(EOS, RHO_C, lnh_grid=grid[::-1])


# ---------------------------------------------------------------------------
# Analytic and prescribed-density models
# ---------------------------------------------------------------------------

def test_uniform_density_closed_form():
    model = UniformDensityStellarModel(2.5, M=1.4)
    assert model.R == pytest.approx(3.5) and model.M == pytest.approx(1.4) and model.C == pytest.approx(0.4)
    b = np.sqrt(1 - 2 * 0.4)
    e0 = 3 * 1.4 / (4 * np.pi * 3.5**3)
    assert model.central_pressure == pytest.approx(e0 * (1 - b) / (3 * b - 1))
    r = np.linspace(0.05, 0.95, 400) * model.R
    np.testing.assert_allclose(np.gradient(model.pressure(r), r)[1:-1], tov_dp_dr(model, r)[1:-1], rtol=5e-3)
    with pytest.raises(ValueError, match="9/4"):
        UniformDensityStellarModel(2.2)


def test_prescribed_uniform_density_reproduces_closed_form():
    exact = UniformDensityStellarModel(3.0)
    e0 = exact.e[0]
    shot = PrescribedDensityStellarModel(lambda r: e0, exact.R)
    assert shot.M == pytest.approx(exact.M, rel=1e-8)
    assert shot.central_pressure == pytest.approx(exact.central_pressure, rel=1e-6)
    np.testing.assert_allclose(shot.nu, exact.nu, atol=1e-6)


def test_tolman_vii():
    model = TolmanVIIStellarModel(3.0, M=1.2)
    assert model.R / model.M == pytest.approx(3.0, rel=1e-6)
    assert model.pressure(model.R) == pytest.approx(0.0, abs=1e-12)
    r = np.linspace(0.05, 0.95, 400) * model.R
    np.testing.assert_allclose(np.gradient(model.pressure(r), r)[1:-1], tov_dp_dr(model, r)[1:-1], rtol=5e-3)


def test_radial_modes_still_work():
    from temperance.solving.solve_oscillations import RelativisticModes
    mode = RelativisticModes(RelativisticStellarModel(EOS, RHO_C)).compute_radial_mode(0, n_scan=40)
    assert mode.solver_info["omega2"] > 0       # a stable star well below M_max
