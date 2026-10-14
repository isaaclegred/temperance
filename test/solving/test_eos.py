"""
Tests for the equations of state in temperance.solving.analytic_eos.
"""

import numpy as np
import pandas as pd
import pytest

from temperance.solving.analytic_eos import (css_eos, energy_density_polytrope, interpolated_eos,
                                              piecewise_eos, polytropic_eos)
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


def all_eos():
    _, table = polytrope_table(200)
    return {
        "polytropic": polytropic_eos(K=123.6, Gamma=2.0),
        "energy_density_polytrope": energy_density_polytrope(K=50.0, Gamma=2.0, e_max=5e-3),
        "css": css_eos(1.0 / 3.0, E0),
        "interpolated": interpolated_eos(table),
        "piecewise": piecewise_eos([
            {"type": "polytrope", "Gamma": 2.0, "K": 50.0, "e_max": 1e-3},
            {"type": "css", "cs2": 0.5, "e_max": 5e-3},
        ]),
    }


@pytest.mark.parametrize("name", list(all_eos()))
def test_logenthalpy_of_p_inverts_p_of_logenthalpy(name):
    eos = all_eos()[name]
    lnh = np.linspace(0.05, 0.3, 6)
    p = eos.p_of_logenthalpy(lnh)
    np.testing.assert_allclose(eos.logenthalpy_of_p(p), lnh, rtol=1e-3)


def test_interpolated_eos_table_limits():
    _, table = polytrope_table()
    eos = interpolated_eos(table)
    assert eos.p_max == pytest.approx(table["pressurec2"].iloc[-1] * CGS_DENSITY_TO_GEOM)
    assert eos.rho_max == pytest.approx(table["baryon_density"].iloc[-1] * CGS_DENSITY_TO_GEOM)
    assert eos.e_max == pytest.approx(table["energy_densityc2"].iloc[-1] * CGS_DENSITY_TO_GEOM)
    assert eos.lnh_max == pytest.approx(float(eos.logenthalpy_of_rho(np.array([eos.rho_max]))[0]))
    assert eos.lnh_max == pytest.approx(float(eos.logenthalpy_of_p(np.array([eos.p_max]))[0]))


def test_interpolated_eos_uses_tabulated_cs2():
    exact, table = polytrope_table(200)
    rho = np.array(table["baryon_density"]) * CGS_DENSITY_TO_GEOM
    lnh = exact.logenthalpy_of_rho(rho)
    table["cs2c2"] = exact.cs2_of_logenthalpy(lnh)
    tabulated = interpolated_eos(table)
    gradient = interpolated_eos(table, cs2_column=None)
    rows = slice(50, 150, 10)
    # at the table's own nodes (its lnh grid starts at 0 at the lowest pressure)
    nodes = tabulated.logenthalpy_of_rho(rho[rows])
    np.testing.assert_allclose(tabulated.cs2_of_logenthalpy(nodes), table["cs2c2"].iloc[rows], rtol=1e-12)
    # the finite-difference estimate is close to, but not exactly, the tabulated value
    estimate = gradient.cs2_of_logenthalpy(nodes)
    assert not np.allclose(estimate, table["cs2c2"].iloc[rows], rtol=1e-6, atol=0)
    np.testing.assert_allclose(estimate, table["cs2c2"].iloc[rows], rtol=2e-2)


def test_tabulated_cs2_reaches_the_tov_solver():
    """The Love-number ODE sees the tabulated sound speed."""
    exact, table = polytrope_table(200)
    lnh = exact.logenthalpy_of_rho(np.array(table["baryon_density"]) * CGS_DENSITY_TO_GEOM)
    table["cs2c2"] = 0.5 * exact.cs2_of_logenthalpy(lnh)   # deliberately wrong by a factor 2
    eta_tabulated = lindblom_solver(0.2, interpolated_eos(table))[1][-1, 2]
    eta_gradient = lindblom_solver(0.2, interpolated_eos(table, cs2_column=None))[1][-1, 2]
    assert abs(eta_tabulated - eta_gradient) > 1e-3


def test_piecewise_eos_is_accurate_next_to_segment_boundaries():
    """
    Each segment has its own spline: e(lnh) on the CSS side of a polytrope ->
    CSS boundary (where cs2 jumps from ~0.1 to 0.6) matches the closed form.
    A single spline over the whole table was off by up to 4% here.
    """
    eos = piecewise_eos([{"type": "polytrope", "Gamma": 4 / 3, "K": 0.01, "e_max": 1e-4},
                         {"type": "polytrope", "Gamma": 2.5, "e_max": 5e-4},
                         {"type": "css", "cs2": 0.6, "e_max": 2e-3}])
    K1, G1, e1 = 0.01, 4 / 3, 1e-4
    p1 = K1 * e1**G1
    lnh1 = G1 / (G1 - 1) * np.log(1 + K1 * e1 ** (G1 - 1))
    G2, e2, cs2 = 2.5, 5e-4, 0.6
    K2 = p1 / e1**G2
    p2 = K2 * e2**G2
    lnh2 = lnh1 + G2 / (G2 - 1) * np.log((1 + K2 * e2 ** (G2 - 1)) / (1 + K2 * e1 ** (G2 - 1)))
    lnh = lnh2 + np.linspace(1e-4, 1e-2, 50)
    # on the CSS segment: e + p = (e2 + p2) exp((1 + cs2) / cs2 (lnh - lnh2))
    sum_ep = (e2 + p2) * np.exp((1 + cs2) / cs2 * (lnh - lnh2))
    e_exact = (sum_ep - p2 + cs2 * e2) / (1 + cs2)
    np.testing.assert_allclose(eos.e_of_logenthalpy(lnh), e_exact, rtol=1e-8)
    np.testing.assert_allclose(eos.cs2_of_logenthalpy(lnh), cs2, rtol=1e-12)
    # and on the polytrope side the sound speed is the polytropic one
    below = lnh2 - 1e-4
    e_b = eos.e_of_logenthalpy(np.array([below]))[0]
    assert eos.cs2_of_logenthalpy(np.array([below]))[0] == pytest.approx(G2 * K2 * e_b**G2 / (e_b + K2 * e_b**G2), rel=1e-6)


# ---------------------------------------------------------------------------
# piecewise_eos: rest-mass polytropes and first-order transitions
# ---------------------------------------------------------------------------

RHO_T = 2 * 0.00045


def test_rest_mass_polytrope_segment_matches_polytropic_eos():
    exact = polytropic_eos(K=100.0, Gamma=2.0)
    eos = piecewise_eos([{"type": "rest_mass_polytrope", "Gamma": 2.0, "K": 100.0, "rho_max": RHO_T},
                         {"type": "css", "cs2": 0.5, "e_max": 0.01}])
    rho = np.geomspace(1e-6, 0.9 * RHO_T, 7)
    lnh = exact.logenthalpy_of_rho(rho)
    np.testing.assert_allclose(eos.p_of_logenthalpy(lnh), exact.p_of_logenthalpy(lnh), rtol=1e-6)
    np.testing.assert_allclose(eos.e_of_logenthalpy(lnh), exact.e_of_logenthalpy(lnh), rtol=1e-6)
    np.testing.assert_allclose(eos.logenthalpy_of_rho(rho), lnh, rtol=1e-6)


def test_causal_core_matches_closed_form():
    """polytrope + c_s^2 = 1 core, as in TrappedNullGeos' causal_core_table."""
    K, Gamma, cs2 = 100.0, 2.0, 1.0
    eos = piecewise_eos([{"type": "rest_mass_polytrope", "Gamma": Gamma, "K": K, "rho_max": RHO_T},
                         {"type": "css", "cs2": cs2, "e_max": 0.05}])
    p_t = K * RHO_T**Gamma
    e_t = RHO_T + p_t / (Gamma - 1)
    A = e_t * (1 + cs2) + p_t - cs2 * e_t
    rho = np.geomspace(1.01 * RHO_T, 10 * RHO_T, 6)
    e = (A * (rho / RHO_T) ** (1 + cs2) - p_t + cs2 * e_t) / (1 + cs2)
    np.testing.assert_allclose(eos.e_of_rho(rho), e, rtol=1e-5)
    np.testing.assert_allclose(eos.p_of_rho(rho), p_t + cs2 * (e - e_t), rtol=1e-5)


def test_rest_mass_polytrope_after_another_segment_is_continuous():
    eos = piecewise_eos([{"type": "rest_mass_polytrope", "Gamma": 1.4, "K": 0.5, "rho_max": 1e-4},
                         {"type": "rest_mass_polytrope", "Gamma": 3.0, "e_max": 3e-3}])
    rho_b = 1e-4
    below, above = rho_b * (1 - 1e-7), rho_b * (1 + 1e-7)
    for function in (eos.p_of_rho, eos.e_of_rho, eos.logenthalpy_of_rho):
        assert function(np.array([above]))[0] == pytest.approx(function(np.array([below]))[0], rel=1e-5)
    lnh = np.linspace(0.01, 0.3, 9)
    e, p, rho = eos.e_of_logenthalpy(lnh), eos.p_of_logenthalpy(lnh), eos.rho_of_logenthalpy(lnh)
    np.testing.assert_allclose((e + p) / rho, np.exp(lnh), rtol=1e-6)
    assert eos.e_of_logenthalpy(np.array([0.3]))[0] < 3e-3 * 1.01


@pytest.fixture
def fopt():
    return piecewise_eos([{"type": "rest_mass_polytrope", "Gamma": 2.0, "K": 100.0, "rho_max": RHO_T},
                          {"type": "jump", "delta_e": 1e-3},
                          {"type": "css", "cs2": 1.0, "e_max": 0.05}])


def test_jump_is_recorded(fopt):
    (jump,) = fopt.jumps
    p_t = 100.0 * RHO_T**2
    e_t = RHO_T + p_t
    assert jump["p"] == pytest.approx(p_t)
    assert jump["e_below"] == pytest.approx(e_t) and jump["e_above"] == pytest.approx(e_t + 1e-3)
    assert jump["rho_below"] == pytest.approx(RHO_T)
    assert jump["rho_above"] / jump["rho_below"] == pytest.approx((e_t + 1e-3 + p_t) / (e_t + p_t))
    assert jump["lnh"] == pytest.approx(float(np.log(1 + 2 * 100.0 * RHO_T)))


def test_jump_in_logenthalpy(fopt):
    (jump,) = fopt.jumps
    eps = 1e-9
    lnh = np.array([jump["lnh"] - eps, jump["lnh"], jump["lnh"] + eps])
    e, p, rho = fopt.e_of_logenthalpy(lnh), fopt.p_of_logenthalpy(lnh), fopt.rho_of_logenthalpy(lnh)
    # p is continuous; e and rho jump, and the transition point itself is the dense side
    np.testing.assert_allclose(p, jump["p"], rtol=1e-6)
    assert e[0] == pytest.approx(jump["e_below"], rel=1e-6)
    assert e[1] == pytest.approx(jump["e_above"], rel=1e-6) and e[2] == pytest.approx(jump["e_above"], rel=1e-6)
    assert rho[0] == pytest.approx(jump["rho_below"], rel=1e-6)
    assert rho[2] == pytest.approx(jump["rho_above"], rel=1e-6)
    assert fopt.logenthalpy_of_p(np.array([jump["p"]]))[0] == pytest.approx(jump["lnh"], rel=1e-9)


def test_coexistence_interval_in_density(fopt):
    (jump,) = fopt.jumps
    rho = np.linspace(jump["rho_below"], jump["rho_above"], 11)
    np.testing.assert_allclose(fopt.p_of_rho(rho), jump["p"], rtol=1e-12)
    np.testing.assert_allclose(fopt.logenthalpy_of_rho(rho), jump["lnh"], rtol=1e-12)
    e = fopt.e_of_rho(rho)
    np.testing.assert_allclose(e, (jump["e_below"] + jump["p"]) * rho / jump["rho_below"] - jump["p"], rtol=1e-12)
    # first law inside the mixed phase: d ln rho = de / (e + p)
    np.testing.assert_allclose(np.diff(np.log(rho)), np.diff(np.log(e + jump["p"])), rtol=1e-10)


def test_hybrid_star_matches_quasi_maxwell_ramp(fopt):
    """A true jump and a c_s^2 = 1e-4 ramp over the same delta_e give the same star."""
    (jump,) = fopt.jumps
    ramp = piecewise_eos([{"type": "rest_mass_polytrope", "Gamma": 2.0, "K": 100.0, "rho_max": RHO_T},
                          {"type": "css", "cs2": 1e-4, "e_max": jump["e_above"]},
                          {"type": "css", "cs2": 1.0, "e_max": 0.05}], n_table=2000)
    stars = []
    for eos in (fopt, ramp):
        lnhc = float(eos.logenthalpy_of_p(np.array([4 * jump["p"]]))[0])
        _, sol = lindblom_solver(lnhc, eos, points_to_solve_for=4000)
        r = np.sqrt(sol[-1, 0])
        stars.append((r * sol[-1, 1], r))
    (M_jump, R_jump), (M_ramp, R_ramp) = stars
    assert M_jump == pytest.approx(M_ramp, rel=5e-3)
    assert R_jump == pytest.approx(R_ramp, rel=5e-3)


@pytest.mark.parametrize("segments, message", [
    ([{"type": "jump", "delta_e": 1e-3}], "first or last"),
    ([{"type": "rest_mass_polytrope", "Gamma": 2.0, "K": 100.0, "rho_max": RHO_T},
      {"type": "jump", "delta_e": 1e-3}], "first or last"),
    ([{"type": "rest_mass_polytrope", "Gamma": 2.0, "K": 100.0, "rho_max": RHO_T},
      {"type": "jump", "delta_e": -1e-3}, {"type": "css", "cs2": 1.0, "e_max": 0.05}], "positive"),
    ([{"type": "rest_mass_polytrope", "Gamma": 2.0, "rho_max": RHO_T}], "must specify K"),
    ([{"type": "rest_mass_polytrope", "Gamma": 2.0, "K": 100.0, "rho_max": RHO_T},
      {"type": "jump", "delta_e": 1e-3}, {"type": "jump", "delta_e": 1e-3},
      {"type": "css", "cs2": 1.0, "e_max": 0.05}], "consecutive"),
])
def test_piecewise_segment_errors(segments, message):
    with pytest.raises(ValueError, match=message):
        piecewise_eos(segments)
