"""
Tests for Extraction("extremize", ...) in EoSPriorDistribution.get_property.
"""

import h5py
import numpy as np
import pandas as pd
import pytest

from temperance.sampling.eos_prior import EoSPriorH5, EoSPriorSet, Extraction, extremum_columns
from temperance.utilities.units import KM_PER_MSUN

# rises to M_max = 2.2, falls, then a spiral up to 2.15: R turns (its minimum,
# 10.3) before M does, so R is increasing at the mass minimum
M_SPIRAL = np.array([0.6, 1.2, 1.8, 2.1, 2.2, 2.0, 1.9, 2.0, 2.15, 2.1])
R_SPIRAL = np.array([12.8, 12.6, 12.3, 11.9, 11.4, 10.3, 10.5, 10.7, 10.9, 10.8])
# a twin branch: second rising segment with R still decreasing, reaching 2.3
M_TWIN = np.array([0.6, 1.2, 1.8, 2.0, 1.9, 1.8, 2.0, 2.3, 2.2, 2.1])
R_TWIN = np.array([12.8, 12.6, 12.3, 12.0, 11.0, 10.0, 9.6, 9.3, 9.1, 9.0])
TABLES = {0: (M_SPIRAL, R_SPIRAL), 1: (M_TWIN, R_TWIN)}


@pytest.fixture
def prior_dir(tmp_path):
    subdir = tmp_path / "DRAWmod1000-000000"
    subdir.mkdir()
    for draw, (M, R) in TABLES.items():
        pd.DataFrame({"rhoc": np.geomspace(1e14, 3e15, len(M)), "M": M, "R": R}).to_csv(
            subdir / f"macro-draw-{draw:06d}.csv", index=False)
    return EoSPriorSet(eos_dir=str(tmp_path), eos_column="eos", eos_per_dir=1000, macro_dir=str(tmp_path))


def compactness(M, R):
    return M / (R / KM_PER_MSUN)


def stable_rows(draw):
    # (the sampled twin minimum, row 5, has dM > 0 by central differences)
    # (the sampled spiral peak, row 4, has dM < 0 by central differences, so it
    # is excluded: the extremum is over *sampled* stable stars)
    return {0: [0, 1, 2, 3], 1: [0, 1, 2, 3, 5, 6, 7]}[draw]


def test_fixture_stability():
    from temperance.sampling.stability import stable_mask
    for draw, (M, R) in TABLES.items():
        np.testing.assert_array_equal(np.flatnonzero(stable_mask(M, R)), stable_rows(draw))


def test_column_names():
    assert extremum_columns(["M", "C"], "max") == ["max(M)", "max(C)"]
    assert extremum_columns(["R"], "min", "M") == ["R(min(M))"]


def test_maximum_mass_and_compactness_over_stable_stars(prior_dir):
    out = prior_dir.get_property([0, 1], ["M", "C"], Extraction("extremize", True))
    assert list(out.columns) == ["max(M)", "max(C)"]
    # spiral excluded: the largest stable sampled mass, not the spiral's 2.15
    assert out["max(M)"].tolist() == pytest.approx([2.1, 2.3])
    for draw, (M, R) in TABLES.items():
        rows = stable_rows(draw)
        assert out["max(C)"].iloc[draw] == pytest.approx(compactness(M, R)[rows].max())


def test_spiral_counts_without_branches(prior_dir):
    out = prior_dir.get_property([0], ["C"], Extraction("extremize", True, branched=False))
    assert out["max(C)"].iloc[0] == pytest.approx(compactness(M_SPIRAL, R_SPIRAL).max())


def test_mask(prior_dir):
    heavy = Extraction("extremize", True, mask=lambda table: table["M"] >= 1.9)
    out = prior_dir.get_property([0, 1], ["R"], Extraction("extremize", True, extremum="min",
                                                          mask=lambda table: table["M"] >= 1.9))
    assert out["min(R)"].tolist() == pytest.approx([11.9, 9.3])
    assert prior_dir.get_property([1], ["M"], heavy)["max(M)"].iloc[0] == pytest.approx(2.3)


def test_values_at_the_extremum(prior_dir):
    out = prior_dir.get_property([0, 1], ["R", "C"], Extraction("extremize", True, independent_variable="M"))
    assert list(out.columns) == ["R(max(M))", "C(max(M))"]
    assert out["R(max(M))"].tolist() == pytest.approx([11.9, 9.3])
    assert out["C(max(M))"].iloc[1] == pytest.approx(compactness(2.3, 9.3))


def test_custom_derived_column(prior_dir):
    extraction = Extraction("extremize", True, derived={"MR": lambda table: table["M"] * table["R"]})
    out = prior_dir.get_property([1], ["MR"], extraction)
    assert out["max(MR)"].iloc[0] == pytest.approx(max(M_TWIN[i] * R_TWIN[i] for i in stable_rows(1)))


def test_no_qualifying_star(prior_dir):
    extraction = Extraction("extremize", True, mask=lambda table: table["M"] > 5)
    assert np.isnan(prior_dir.get_property([0], ["M"], extraction)["max(M)"].iloc[0])
    filled = prior_dir.get_property([0], ["M"], extraction, default_values=[0.0])
    assert filled["max(M)"].iloc[0] == 0.0


def test_h5_prior_gives_the_same(prior_dir, tmp_path):
    path = tmp_path / "set.h5"
    with h5py.File(path, "w") as f:
        for draw in TABLES:
            f.create_dataset(f"Macro/{draw:05d}", data=prior_dir.get_macro(draw).to_records(index=False))
            f.create_dataset(f"EoS/{draw:05d}", data=prior_dir.get_macro(draw).to_records(index=False))
    extraction = Extraction("extremize", True, independent_variable="C")
    with EoSPriorH5(path, "EoS", "Macro", index_to_key=lambda i: f"{i:05d}") as prior:
        pd.testing.assert_frame_equal(prior.get_property([0, 1], ["M"], extraction),
                                      prior_dir.get_property([0, 1], ["M"], extraction))


def test_bad_extremum(prior_dir):
    with pytest.raises(ValueError, match="extremum"):
        prior_dir.get_property([0], ["M"], Extraction("extremize", True, extremum="median"))
