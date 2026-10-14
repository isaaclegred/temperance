"""
Tests for temperance.sampling.stability (turning-point radial stability).
"""

import numpy as np
import pandas as pd
import pytest

from temperance.sampling import branched_interpolator
from temperance.sampling.eos_prior import EoSPriorSet, Extraction, find_stable_branches
from temperance.sampling.stability import stable_branches, stable_mask

PC = np.geomspace(1, 100, 9)
# M rises to a maximum, falls, then rises again with R increasing: the spiral
SPIRAL_M = np.array([1, 2, 3, 2.5, 2, 2.1, 2.2, 2.15, 2.1])
SPIRAL_R = np.array([12, 11, 10, 9, 8, 8.2, 8.4, 8.3, 8.2])
# M rises, falls, then rises again with R still decreasing: a twin branch
TWIN_M = np.array([1, 2, 2.2, 2.0, 1.9, 2.0, 2.1, 2.05, 2.0])
TWIN_R = np.array([13, 12.5, 12, 11, 10, 9.5, 9.0, 8.8, 8.7])


def legacy_stable(mass):
    """The rule temperance used before: every run of non-decreasing mass."""
    steps = np.diff(mass) >= 0
    return np.concatenate([[steps[0]], steps])


def test_spiral_is_rejected():
    mask = stable_mask(SPIRAL_M, SPIRAL_R)
    assert mask[:3].all() and not mask[3:].any()


def test_twin_branch_is_kept():
    mask = stable_mask(TWIN_M, TWIN_R)
    # (the sampled peak at index 2 has dM = 0 by central differences, so it is excluded)
    np.testing.assert_array_equal(mask, [1, 1, 0, 0, 0, 1, 1, 0, 0])
    assert len(stable_branches(TWIN_M, TWIN_R)) == 2


def test_maximum_with_radius_increasing_regains_a_mode():
    # e.g. the minimum-mass end of the neutron-star branch, read backwards
    M = np.array([1.0, 1.2, 1.1, 1.0, 1.3, 1.5])
    R = np.array([10, 11, 12, 11, 10, 9.0])
    # max with R increasing regains (count stays 0); min with R decreasing regains
    np.testing.assert_array_equal(stable_mask(M, R), [1, 1, 0, 1, 1, 1])


def test_without_radius_matches_legacy_rule():
    rng = np.random.default_rng(0)
    for _ in range(50):
        mass = np.cumsum(rng.normal(0.1, 0.3, 30))
        np.testing.assert_array_equal(stable_mask(mass), legacy_stable(mass))
    # the spiral cannot be told from a twin branch without radii
    np.testing.assert_array_equal(stable_mask(SPIRAL_M), legacy_stable(SPIRAL_M))


def test_central_variable_orders_the_sequence():
    shuffle = np.random.default_rng(1).permutation(len(PC))
    mask = stable_mask(SPIRAL_M[shuffle], SPIRAL_R[shuffle], central_variable=PC[shuffle])
    np.testing.assert_array_equal(mask, stable_mask(SPIRAL_M, SPIRAL_R, central_variable=PC)[shuffle])
    branches = stable_branches(TWIN_M[shuffle], TWIN_R[shuffle], central_variable=PC[shuffle])
    expected = stable_branches(TWIN_M, TWIN_R, central_variable=PC)
    assert len(branches) == len(expected) == 2
    for branch, reference in zip(branches, expected):
        np.testing.assert_array_equal(branch, reference[shuffle])


def test_min_mass_and_mask():
    assert not stable_mask(TWIN_M, TWIN_R, min_mass=1.5)[0]
    assert stable_mask(TWIN_M, TWIN_R, min_mass=1.5)[1]
    custom = np.ones(9, bool)
    custom[5] = False
    np.testing.assert_array_equal(stable_mask(TWIN_M, TWIN_R, mask=custom), [1, 1, 0, 0, 0, 0, 1, 0, 0])


def test_find_stable_branches_uses_radius():
    assert len(find_stable_branches(SPIRAL_M)) == 2            # legacy: spiral counted
    assert len(find_stable_branches(SPIRAL_M, SPIRAL_R)) == 1


def test_get_branches_rejects_spiral():
    table = pd.DataFrame({"M": SPIRAL_M, "R": SPIRAL_R, "Lambda": np.ones(9), "I": np.ones(9)})
    branches = branched_interpolator.get_branches(table)
    assert len(branches) == 1
    assert list(branches[0].index) == [0, 1, 2]
    twins = branched_interpolator.get_branches(
        pd.DataFrame({"M": TWIN_M, "R": TWIN_R, "Lambda": np.ones(9), "I": np.ones(9)}))
    assert [list(b.index) for b in twins] == [[0, 1], [5, 6]]


def test_branched_extraction_ignores_spiral(tmp_path):
    rhoc = np.linspace(1e14, 3e15, 9)
    subdir = tmp_path / "DRAWmod1000-000000"
    subdir.mkdir()
    pd.DataFrame({"rhoc": rhoc, "M": SPIRAL_M, "R": SPIRAL_R}).to_csv(subdir / "macro-draw-000000.csv", index=False)
    prior = EoSPriorSet(eos_dir=str(tmp_path), eos_column="eos", eos_per_dir=1000, macro_dir=str(tmp_path))
    # 2.15 Msun is on the spiral's rising segment (and on the stable branch); only the stable one counts
    result = prior.get_property([0], ["R"], Extraction("interpolation", True, "M", [2.15], selection_rule="max"))
    assert result["R(M=2.15)"].iloc[0] == pytest.approx(np.interp(2.15, SPIRAL_M[:3], SPIRAL_R[:3]))
    # without the radius column the spiral's rising segment is accepted and supplies the smaller R
    legacy = prior.get_property([0], ["R"], Extraction("interpolation", True, "M", [2.15], selection_rule="min"),
                                radius_column=None)
    assert legacy["R(M=2.15)"].iloc[0] == pytest.approx(8.3)


def test_matches_trapped_null_geos_rule():
    """Same answer as trapgeo.stability.stable_mask on random sequences (when available)."""
    trapgeo_stability = pytest.importorskip("trapgeo.stability")
    rng = np.random.default_rng(2)
    for _ in range(100):
        pc = np.sort(rng.uniform(1, 100, 40))
        M = np.cumsum(rng.normal(0.05, 0.3, 40))
        R = np.cumsum(rng.normal(-0.1, 0.3, 40)) + 20
        np.testing.assert_array_equal(stable_mask(M, R, central_variable=pc),
                                      trapgeo_stability.stable_mask(pc, M, R))


def test_repeated_central_values_are_merged():
    """Pressures one ulp apart have equal logs; that must not produce NaN gradients."""
    pc = PC.copy()
    M, R = np.insert(TWIN_M, 6, TWIN_M[6]), np.insert(TWIN_R, 6, TWIN_R[6])
    pc = np.insert(pc, 6, np.nextafter(pc[6], np.inf))
    with np.errstate(all="raise"):
        mask = stable_mask(M, R, central_variable=pc)
    expected = stable_mask(TWIN_M, TWIN_R, central_variable=PC)
    np.testing.assert_array_equal(np.delete(mask, 6), expected)
    assert mask[6] == expected[6]
