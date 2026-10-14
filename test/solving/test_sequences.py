"""
Tests for temperance.solving.sequences.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import pytest

from temperance.solving.analytic_eos import polytropic_eos
from temperance.solving.sequences import refine_boundary, sequence_window_from_macro, solve_sequence
from temperance.solving.solve_relativistic import get_tov_family
from temperance.utilities.units import CGS_DENSITY_TO_GEOM, KM_PER_MSUN

EOS = polytropic_eos(K=100.0, Gamma=2.0)
LNH = np.linspace(0.1, 1.2, 12)      # M_max is near lnh_c = 0.5


@pytest.fixture(scope="module")
def sequence():
    return solve_sequence(EOS, lnh_c=LNH, star_properties={"compactness": lambda model: model.M / model.R})


def test_matches_get_tov_family():
    rho_c = np.array([1e-4, 5e-4, 1e-3])
    stars = solve_sequence(EOS, rho_c=rho_c)
    family = get_tov_family(EOS, rho_c)
    np.testing.assert_allclose(stars["M"], family["M"], rtol=1e-8)
    np.testing.assert_allclose(stars["R_km"], family["R"], rtol=1e-8)
    np.testing.assert_allclose(stars["R_km"], stars["R"] * KM_PER_MSUN)


def test_parameterisations_agree(sequence):
    by_pressure = solve_sequence(EOS, p_c=sequence["p_c"])
    by_density = solve_sequence(EOS, rho_c=sequence["rho_c"])
    for other in (by_pressure, by_density):
        np.testing.assert_allclose(other["M"], sequence["M"], rtol=1e-7)
        np.testing.assert_allclose(other["lnh_c"], sequence["lnh_c"], rtol=1e-7)
    with pytest.raises(ValueError, match="exactly one"):
        solve_sequence(EOS, lnh_c=LNH, p_c=sequence["p_c"])


def test_stability_and_only_stable(sequence):
    i_max = int(np.argmax(sequence["M"]))
    assert sequence["stable"].iloc[:i_max].all()
    assert not sequence["stable"].iloc[i_max + 1:].any()
    stable = sequence["stable"].to_numpy()
    np.testing.assert_allclose(sequence.loc[stable, "compactness"].astype(float), sequence.loc[stable, "C"])
    assert sequence.loc[~stable, "compactness"].isna().all()
    every = solve_sequence(EOS, lnh_c=LNH, star_properties={"C2": lambda model: model.C}, only_stable=False)
    assert every["C2"].notna().all()


def test_stability_options():
    assert solve_sequence(EOS, lnh_c=LNH, stability=None)["stable"].all()
    custom = solve_sequence(EOS, lnh_c=LNH, stability=lambda model: model.M > 1.5)
    np.testing.assert_array_equal(custom["stable"], custom["M"] > 1.5)
    with pytest.raises(ValueError, match="stability"):
        solve_sequence(EOS, lnh_c=LNH, stability="radial_mode")


@dataclass
class Summary:
    """Stands in for a per-star summary object such as NullOrbitSummary."""
    ultracompact: bool
    compactness: float

    def as_dict(self):
        return asdict(self)


def test_dict_and_summary_properties_become_columns():
    stars = solve_sequence(EOS, lnh_c=[0.3, 0.4], star_properties={
        "both": lambda model: {"M": model.M, "R": model.R},
        "summary": lambda model: Summary(model.R < 3 * model.M, model.C)})
    np.testing.assert_allclose(stars["both_M"].astype(float), stars["M"])
    assert "summary_ultracompact" in stars and not stars["summary_ultracompact"].any()
    np.testing.assert_allclose(stars["summary_compactness"].astype(float), stars["C"])


def test_errors_are_recorded_or_raised():
    def fails_for_heavy(model):
        if model.M > 1.4:
            raise RuntimeError("too heavy")
        return 1.0
    stars = solve_sequence(EOS, lnh_c=[0.1, 0.3], star_properties={"x": fails_for_heavy})
    assert stars["error"].iloc[0] is None and "too heavy" in stars["error"].iloc[1]
    with pytest.raises(RuntimeError, match="too heavy"):
        solve_sequence(EOS, lnh_c=[0.1, 0.3], star_properties={"x": fails_for_heavy}, on_error="raise")


def test_refine_boundary():
    value, model = refine_boundary(EOS, lambda model: model.C > 0.2, 0.1, 0.5, n_bisect=30)
    assert model.C == pytest.approx(0.2, abs=1e-6)
    assert model.lnh_c == pytest.approx(value)
    pressure, at_pressure = refine_boundary(EOS, lambda model: model.C > 0.2, model.central_pressure / 2,
                                            model.central_pressure * 2, var="p_c")
    assert at_pressure.C == pytest.approx(0.2, abs=1e-5)
    with pytest.raises(ValueError, match="same value"):
        refine_boundary(EOS, lambda model: True, 0.1, 0.5)


def macro_table():
    """A macro table (cgs pressure, M [Msun], R [km]) with a maximum and a spiral."""
    pc = np.geomspace(1e34, 1e37, 12) * 1e-21          # g/cm^3, arbitrary scale
    M = np.array([0.5, 1.0, 1.5, 1.9, 2.1, 2.2, 2.15, 2.0, 1.95, 1.97, 1.99, 1.96])
    R = np.array([13, 12.8, 12.5, 12, 11.5, 11, 10.5, 10, 9.8, 9.9, 10.0, 9.9])
    return pd.DataFrame({"central_pressurec2": pc, "M": M, "R": R})


def test_sequence_window_from_macro():
    macro = macro_table()
    C = macro["M"] / (macro["R"] / KM_PER_MSUN)
    window = sequence_window_from_macro(macro, C_min=0.25, densify=3)
    stable_selected = np.flatnonzero((C >= 0.25).to_numpy() & (np.arange(12) <= 5))
    lo, hi = stable_selected[0] - 1, 6             # one neighbour on each side
    expected = np.exp(np.unique(np.concatenate(
        [np.linspace(a, b, 4)[:-1] for a, b in zip(np.log(macro["central_pressurec2"][lo:hi + 1])[:-1],
                                                   np.log(macro["central_pressurec2"][lo:hi + 1])[1:])]
        + [[np.log(macro["central_pressurec2"][hi])]]))) * CGS_DENSITY_TO_GEOM
    np.testing.assert_allclose(window, expected, rtol=1e-12)
    # the spiral's rising segment (rows 8-10) is not stable, so it is not in the window
    assert window.max() <= macro["central_pressurec2"][6] * CGS_DENSITY_TO_GEOM * (1 + 1e-12)
    assert len(sequence_window_from_macro(macro, C_min=0.4)) == 0
    clipped = sequence_window_from_macro(macro, C_min=0.25, densify=3, p_max=window[3])
    np.testing.assert_allclose(clipped, window[:4])
