"""
Tests for the equations of state in temperance.solving.analytic_eos.
"""

import numpy as np
import pandas as pd
import pytest

from temperance.solving.analytic_eos import css_eos, interpolated_eos, polytropic_eos
from temperance.solving.solve_relativistic import lindblom_solver
from temperance.utilities.units import CGS_DENSITY_TO_GEOM

E0 = 1e-3


@pytest.fixture(params=[0.2, 1.0 / 3.0, 1.0])
def css(request):
    return css_eos(request.param, E0)


def test_css_surface(css):
    assert css.e_of_logenthalpy(np.array([0.0]))[0] == pytest.approx(E0)
    assert css.rho_of_logenthalpy(np.array([0.0]))[0] == pytest.approx(E0)
    assert css.p_of_logenthalpy(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-18)
    # rho = e (p = 0) below e0, continuous at e0
    assert css.rho_of_e(np.array([0.5 * E0, E0])) == pytest.approx([0.5 * E0, E0])


def test_css_round_trips(css):
    lnh = np.linspace(0.0, 0.8, 9)
    rho = css.rho_of_logenthalpy(lnh)
    np.testing.assert_allclose(css.logenthalpy_of_rho(rho), lnh, atol=1e-12)
    np.testing.assert_allclose(css.e_of_rho(rho), css.e_of_logenthalpy(lnh), rtol=1e-12)
    e = css.e_of_logenthalpy(lnh)
    np.testing.assert_allclose(css.rho_of_e(e), rho, rtol=1e-12)
    np.testing.assert_allclose(css.p_of_rho(rho), css.p_of_logenthalpy(lnh), rtol=1e-12, atol=1e-18)


def test_css_thermodynamic_consistency(css):
    e = np.linspace(1.01 * E0, 5 * E0, 400)
    p, rho = css.p_of_e(e), css.rho_of_e(e)
    # first law, integrated: ln rho(e) - ln rho(e0) = int_e0^e de' / (e' + p(e'))
    from scipy.integrate import quad
    for e_i, rho_i in zip(e[::50], rho[::50]):
        integral, _ = quad(lambda x: 1 / (x + css.p_of_e(x)), E0, e_i, epsabs=0, epsrel=1e-12)
        assert np.log(rho_i / E0) == pytest.approx(integral, rel=1e-9)
    # h = (e + p) / rho = exp(lnh)
    lnh = css.logenthalpy_of_rho(rho)
    np.testing.assert_allclose((e + p) / rho, np.exp(lnh), rtol=1e-12)
    np.testing.assert_allclose(np.gradient(p, e), css.cs2, rtol=1e-10)
    np.testing.assert_allclose(css.cs2_of_logenthalpy(lnh), css.cs2)


def test_css_self_bound_star_solves():
    eos = css_eos(1.0 / 3.0, E0)
    lnhs, sol = lindblom_solver(0.3, eos)
    r, v = np.sqrt(sol[-1, 0]), sol[-1, 1]
    assert np.isfinite(r) and r > 0 and 0 < v < 0.5


def polytrope_table(n=40):
    eos = polytropic_eos(K=123.6, Gamma=2.0)
    rho = np.geomspace(1e-6, 4e-3, n)
    return eos, pd.DataFrame({"baryon_density": rho / CGS_DENSITY_TO_GEOM,
                              "pressurec2": eos.p_of_rho(rho) / CGS_DENSITY_TO_GEOM,
                              "energy_densityc2": eos.e_of_rho(rho) / CGS_DENSITY_TO_GEOM})


def test_interpolated_eos_default_is_linear():
    _, table = polytrope_table()
    eos = interpolated_eos(table)
    rho = np.array(table["baryon_density"]) * CGS_DENSITY_TO_GEOM
    p = np.array(table["pressurec2"]) * CGS_DENSITY_TO_GEOM
    mid = np.sqrt(rho[10:20] * rho[11:21])
    np.testing.assert_allclose(eos.p_of_rho(mid), np.interp(mid, rho, p), rtol=1e-12)


def test_interpolated_eos_cubic_is_honoured_and_more_accurate():
    exact, table = polytrope_table()
    linear, cubic = interpolated_eos(table), interpolated_eos(table, method="cubic")
    rho = np.array(table["baryon_density"]) * CGS_DENSITY_TO_GEOM
    mid = np.sqrt(rho[10:30] * rho[11:31])
    err_linear = np.abs(linear.p_of_rho(mid) / exact.p_of_rho(mid) - 1)
    err_cubic = np.abs(cubic.p_of_rho(mid) / exact.p_of_rho(mid) - 1)
    assert err_cubic.max() < 0.1 * err_linear.max()
