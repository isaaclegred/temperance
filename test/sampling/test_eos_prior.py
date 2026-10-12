"""
Tests for the EoS prior classes in temperance.sampling.eos_prior.

The regression tests write a small set of EoS draws to disk and check that
UniversalityEoSPriorSet.get_property returns exactly what the pre-split implementation
did, i.e. universality's process2samples called with the same arguments
(skipped if universality is not installed).  The remaining tests cover the
universality-free path shared by EoSPriorSet and EoSPriorH5.
"""

import h5py
import numpy as np
import pandas as pd
import pytest

from sampling import branched_interpolator, eos_prior
from sampling.eos_prior import (BaseEoSPriorSet, EoSPriorDistribution, EoSPriorH5,
                                UniversalityEoSPriorSet, Extraction, EoSPriorSet,
                                find_stable_branches, output_columns)

EOS_PER_DIR = 2
NUM_EOS = 5


@pytest.fixture
def usamples(monkeypatch):
    samples = pytest.importorskip("universality.properties.samples")
    # universality's nearest_neighbor rule still uses np.infty, which numpy 2 removed
    monkeypatch.setattr(np, "infty", np.inf, raising=False)
    return samples


@pytest.fixture
def eos_dir(tmp_path):
    """
    Macro files with monotonic M(rhoc), R(rhoc), and a branches file per
    EoS marking a single stable branch that excludes the high-rhoc end.
    """
    rhoc = np.linspace(1e14, 3e15, 60)
    for draw in range(NUM_EOS):
        subdir = tmp_path / f"DRAWmod{EOS_PER_DIR}-{draw // EOS_PER_DIR:06d}"
        subdir.mkdir(exist_ok=True)
        m_max = 2.0 + 0.1 * draw
        mass = m_max * (1 - np.exp(-rhoc / 8e14))
        radius = 13.0 - 0.5 * draw - 2.0 * rhoc / 3e15
        pd.DataFrame({"rhoc": rhoc, "M": mass, "R": radius}).to_csv(
            subdir / f"macro-draw-{draw:06d}.csv", index=False)
        pd.DataFrame({"baryon_density": rhoc, "pressurec2": (1 + draw) * rhoc**2 / 1e30}).to_csv(
            subdir / f"eos-draw-{draw:06d}.csv", index=False)
        pd.DataFrame({"start_baryon_density": [rhoc[0]],
                      "end_baryon_density": [rhoc[45]]}).to_csv(
            subdir / f"macro-draw-{draw:06d}-branches.csv", index=False)
    return tmp_path


@pytest.fixture
def eos_h5(eos_dir):
    """The same draws as `eos_dir`, in the layout EoSPriorH5 reads."""
    native = make_prior_set(EoSPriorSet, eos_dir)
    path = eos_dir / "draws.h5"
    with h5py.File(path, "w") as f:
        for draw in range(NUM_EOS):
            f.create_dataset(f"eos/eos_{draw}", data=native.get_eos(draw).to_records(index=False))
            f.create_dataset(f"ns/eos_{draw}", data=native.get_macro(draw).to_records(index=False))
    prior = EoSPriorH5(path, index_to_key=lambda x: f"eos_{int(x)}")
    yield prior
    prior.h5_file.close()


def make_prior_set(cls, eos_dir):
    return cls(
        eos_dir=str(eos_dir), eos_column="eos", eos_per_dir=EOS_PER_DIR,
        macro_dir=str(eos_dir),
        branches_data=(
            str(eos_dir / "DRAWmod2-%(moddraw)06d" / "macro-draw-%(draw)06d-branches.csv"),
            "rhoc", "start_baryon_density", "end_baryon_density"))


def old_get_property(prior_set, eos_indices, dependent_variables, extraction, **kwargs):
    """The pre-split UniversalityEoSPriorSet.get_property, verbatim apart from the print."""
    from universality.properties import samples
    if extraction.dynamic is not None:
        dynamic_kwargs = {"dynamic_x_test": np.array(extraction.dynamic)}
    else:
        dynamic_kwargs = {}
    if extraction.branched:
        branches_kwargs = {"branches_mapping": prior_set.branches_data}
    else:
        branches_kwargs = {}
    data = samples.process2samples(
        np.array(eos_indices),
        prior_set.get_path_template(macro=extraction.macro),
        prior_set.eos_per_dir,
        extraction.independent_variable,
        dependent_variables,
        static_x_test=extraction.values,
        **dynamic_kwargs,
        **branches_kwargs,
        **kwargs)
    ref_columns = extraction.dynamic.columns if extraction.dynamic is not None else []
    columns = samples.outputcolumns(dependent_variables,
                                    extraction.independent_variable,
                                    reference_values=[] if extraction.values is None else extraction.values,
                                    reference_columns=ref_columns)
    return pd.DataFrame(data, columns=columns)


EXTRACTION_CASES = {
    "static": (Extraction("interpolation", True, "M", [1.2, 1.4, 1.8], branched=False), {}),
    "static_two_columns": (Extraction("interpolation", True, "rhoc", [5e14, 1e15], branched=False), {}),
    "dynamic": (Extraction("interpolation", True, "M", None, branched=False,
                           dynamic=pd.DataFrame({"m_a": np.linspace(1.0, 1.5, NUM_EOS),
                                                 "m_b": np.linspace(1.6, 1.9, NUM_EOS)})), {}),
    "static_and_dynamic": (Extraction("interpolation", True, "M", [1.3], branched=False,
                                      dynamic=pd.DataFrame({"m_a": np.linspace(1.0, 1.5, NUM_EOS)})), {}),
    # 2.05 Msun is only reached past the end of the stable branch for most draws
    "branched": (Extraction("interpolation", True, "M", [1.4, 2.05]),
                 {"default_values": [np.nan, np.nan]}),
    "nearest_neighbor": (Extraction("interpolation", True, "M", [1.4], branched=False),
                         {"selection_rule": "nearest_neighbor"}),
}
DEPENDENT = {"static_two_columns": ["M", "R"], "branched": ["R", "rhoc"]}


# ---------------------------------------------------------------------------
# Structure (no universality needed)
# ---------------------------------------------------------------------------

def test_eos_prior_imports_without_universality():
    assert "universality" not in eos_prior.__dict__


def test_get_default_returns_calling_class():
    assert type(UniversalityEoSPriorSet.get_default()) is UniversalityEoSPriorSet
    assert type(EoSPriorSet.get_default()) is EoSPriorSet


def test_output_columns_order():
    assert output_columns(["R", "Lambda"], "M", [1.4, 2.0], ["m1"]) == [
        "R(M=1.4)", "R(M=2.0)", "R(M@m1)", "Lambda(M=1.4)", "Lambda(M=2.0)", "Lambda(M@m1)"]


def test_class_hierarchy():
    for cls in (UniversalityEoSPriorSet, EoSPriorSet):
        assert issubclass(cls, BaseEoSPriorSet)
    for cls in (EoSPriorSet, EoSPriorH5):
        assert issubclass(cls, EoSPriorDistribution)
    # the universality-backed set keeps its own process2samples-based get_property
    assert not issubclass(UniversalityEoSPriorSet, EoSPriorDistribution)


@pytest.mark.parametrize("cls", [UniversalityEoSPriorSet, EoSPriorSet])
def test_prior_set_get_eos_and_get_macro(eos_dir, cls):
    prior_set = make_prior_set(cls, eos_dir)
    subdir = eos_dir / "DRAWmod2-000001"
    pd.testing.assert_frame_equal(prior_set.get_macro(3), pd.read_csv(subdir / "macro-draw-000003.csv"))
    pd.testing.assert_frame_equal(prior_set.get_eos(3), pd.read_csv(subdir / "eos-draw-000003.csv"))


def test_h5_get_eos_and_get_macro_match_prior_set(eos_dir, eos_h5):
    native = make_prior_set(EoSPriorSet, eos_dir)
    for draw in range(NUM_EOS):
        pd.testing.assert_frame_equal(pd.DataFrame(eos_h5.get_macro(draw)[()]), native.get_macro(draw))
        pd.testing.assert_frame_equal(pd.DataFrame(eos_h5.get_eos(draw)[()]), native.get_eos(draw))


def with_fake_per_eos(cls):
    """
    Subclass whose get_property_per_eos reads the real table (via get_macro or
    get_eos), then returns y at the first row, offset by x so the x_test each
    EoS receives is visible in the output.
    """
    class Fake(cls):
        def get_property_per_eos(self, eos_index, dependent_variables, extraction, x_test, **kwargs):
            table = self.get_macro(eos_index) if extraction.macro else self.get_eos(eos_index)
            table = pd.DataFrame(np.asarray(table)) if not isinstance(table, pd.DataFrame) else table
            return np.array([table[y].iloc[0] + x for y in dependent_variables for x in x_test])
    return Fake


@pytest.mark.parametrize("case", EXTRACTION_CASES)
def test_shared_get_property_layout(eos_dir, eos_h5, case):
    """
    The shared get_property hands each EoS its static then dynamic x values,
    and lays the results out exactly as UniversalityEoSPriorSet does (output_columns).
    Native and H5 priors give identical results.
    """
    extraction, _ = EXTRACTION_CASES[case]
    dependent = DEPENDENT.get(case, ["R"])
    static = list(extraction.values or [])
    dynamic = (np.array(extraction.dynamic) if extraction.dynamic is not None
               else np.empty((NUM_EOS, 0)))
    ref_columns = [] if extraction.dynamic is None else list(extraction.dynamic.columns)

    native = make_prior_set(with_fake_per_eos(EoSPriorSet), eos_dir)
    result = native.get_property(range(NUM_EOS), dependent, extraction)

    expected = pd.DataFrame(
        [[native.get_macro(draw)[y].iloc[0] + x
          for y in dependent for x in static + list(dynamic[draw])]
         for draw in range(NUM_EOS)],
        columns=output_columns(dependent, extraction.independent_variable, static, ref_columns))
    pd.testing.assert_frame_equal(result, expected)

    h5 = with_fake_per_eos(EoSPriorH5)(eos_h5.h5_path, index_to_key=eos_h5.index_to_key)
    try:
        pd.testing.assert_frame_equal(h5.get_property(range(NUM_EOS), dependent, extraction), expected)
    finally:
        h5.h5_file.close()


def test_shared_get_property_uses_eos_table_when_not_macro(eos_dir):
    native = make_prior_set(with_fake_per_eos(EoSPriorSet), eos_dir)
    extraction = Extraction("interpolation", False, "baryon_density", [0.0], branched=False)
    result = native.get_property(range(NUM_EOS), ["pressurec2"], extraction)
    np.testing.assert_allclose(result["pressurec2(baryon_density=0.0)"],
                               [native.get_eos(d)["pressurec2"].iloc[0] for d in range(NUM_EOS)])


def test_shared_get_property_requires_x_values(eos_dir):
    with pytest.raises(ValueError, match="values"):
        make_prior_set(EoSPriorSet, eos_dir).get_property(
            range(NUM_EOS), ["R"], Extraction("interpolation", True, "M"))


def test_unknown_strategy_raises(eos_dir):
    with pytest.raises(ValueError, match="interpolation"):
        make_prior_set(EoSPriorSet, eos_dir).get_property(
            range(NUM_EOS), ["R"], Extraction("bogus", True))


# ---------------------------------------------------------------------------
# Regression against the pre-split implementation (needs universality)
# ---------------------------------------------------------------------------

def test_output_columns_matches_universality(usamples):
    args = (["R", "Lambda"], "M", [1.4, 2.0], ["m1", "m2"])
    assert output_columns(*args) == usamples.outputcolumns(*args)


@pytest.mark.parametrize("case", EXTRACTION_CASES)
def test_get_property_matches_pre_split(usamples, eos_dir, case):
    extraction, kwargs = EXTRACTION_CASES[case]
    prior_set = make_prior_set(UniversalityEoSPriorSet, eos_dir)
    indices = np.arange(NUM_EOS)
    dependent = DEPENDENT.get(case, ["R"])
    new = prior_set.get_property(indices, dependent, extraction, **kwargs)
    old = old_get_property(prior_set, indices, dependent, extraction, **kwargs)
    pd.testing.assert_frame_equal(new, old)
    assert not new.isna().all().all()


# ---------------------------------------------------------------------------
# Native extraction (get_property_per_eos)
# ---------------------------------------------------------------------------

def twin_star_mass(rhoc, draw):
    """
    Piecewise-linear M(rhoc): rises to 1.8, falls to 1.6, rises to 2.1+0.02*draw,
    then falls.  Two stable branches, overlapping for 1.6 <= M <= 1.8.
    """
    knots_rho = [1e14, 1e15, 1.4e15, 2.2e15, 3e15]
    knots_m = [0.5, 1.8, 1.6, 2.1 + 0.02 * draw, 1.9]
    return np.interp(rhoc, knots_rho, knots_m)


@pytest.fixture
def twin_dir(tmp_path):
    """
    Twin-star macro files, with branches files marking exactly the rising
    segments, i.e. what find_stable_branches should detect on the fly.
    """
    rhoc = np.linspace(1e14, 3e15, 59)
    for draw in range(NUM_EOS):
        subdir = tmp_path / f"DRAWmod{EOS_PER_DIR}-{draw // EOS_PER_DIR:06d}"
        subdir.mkdir(exist_ok=True)
        mass = twin_star_mass(rhoc, draw)
        radius = 14.0 - 0.3 * draw - 3.0 * rhoc / 3e15
        pd.DataFrame({"rhoc": rhoc, "M": mass, "R": radius,
                      "Lambda": 1e3 * (1.4 / mass)**6}).to_csv(
            subdir / f"macro-draw-{draw:06d}.csv", index=False)
        branches = find_stable_branches(mass)
        pd.DataFrame({"start_baryon_density": [rhoc[b][0] for b in branches],
                      "end_baryon_density": [rhoc[b][-1] for b in branches]}).to_csv(
            subdir / f"macro-draw-{draw:06d}-branches.csv", index=False)
    return tmp_path


def test_find_stable_branches_twin_star():
    rhoc = np.linspace(1e14, 3e15, 59)
    mass = twin_star_mass(rhoc, 0)
    branches = find_stable_branches(mass)
    assert len(branches) == 2
    for branch in branches:
        assert np.all(np.diff(mass[branch]) >= 0)
        rows = np.flatnonzero(branch)
        assert np.all(np.diff(rows) == 1)  # contiguous
    # everything not on a branch is reached by a decreasing step
    unstable = ~np.logical_or(*branches)
    assert np.all(mass[unstable] < mass[np.flatnonzero(unstable) - 1])
    assert mass[branches[0]].max() == pytest.approx(1.8)


def test_find_stable_branches_matches_branched_interpolator():
    rhoc = np.linspace(1e14, 3e15, 59)
    table = pd.DataFrame({"M": twin_star_mass(rhoc, 2), "R": np.linspace(14, 11, 59),
                          "Lambda": np.ones(59), "I": np.ones(59)})
    expected = [list(branch.index) for branch in branched_interpolator.get_branches(table)]
    assert [list(np.flatnonzero(b)) for b in find_stable_branches(table["M"])] == expected


def test_find_stable_branches_monotonic_and_tiny():
    assert len(find_stable_branches(np.linspace(1, 2, 10))) == 1
    assert find_stable_branches(np.linspace(1, 2, 10))[0].all()
    assert len(find_stable_branches([1.0])) == 1
    assert find_stable_branches(np.linspace(2, 1, 10)) == []


def test_native_static_interpolation_exact(eos_dir):
    native = make_prior_set(EoSPriorSet, eos_dir)
    extraction = Extraction("interpolation", True, "rhoc", [5e14, 1e15], branched=False)
    result = native.get_property(range(NUM_EOS), ["R"], extraction)
    for draw in range(NUM_EOS):
        expected = 13.0 - 0.5 * draw - 2.0 * np.array([5e14, 1e15]) / 3e15
        np.testing.assert_allclose(result.iloc[draw], expected, rtol=1e-12)


def test_native_branches_on_the_fly(twin_dir):
    native = make_prior_set(EoSPriorSet, twin_dir)
    masses = [1.4, 1.7, 2.0, 2.5]
    low = native.get_property(range(NUM_EOS), ["R"],
                              Extraction("interpolation", True, "M", masses, selection_rule="max"))
    high = native.get_property(range(NUM_EOS), ["R"],
                               Extraction("interpolation", True, "M", masses, selection_rule="min"))
    # 1.4 and 2.0 are each on one branch; 1.7 is on both (larger R on the first)
    np.testing.assert_allclose(low["R(M=1.4)"], high["R(M=1.4)"])
    np.testing.assert_allclose(low["R(M=2.0)"], high["R(M=2.0)"])
    assert np.all(low["R(M=1.7)"] > high["R(M=1.7)"])
    # 2.5 is above every maximum mass
    assert low["R(M=2.5)"].isna().all() and high["R(M=2.5)"].isna().all()


def test_native_random_picks_a_branch_reproducibly(twin_dir):
    native = make_prior_set(EoSPriorSet, twin_dir)
    extraction = Extraction("interpolation", True, "M", [1.7] * 40, branched=True)
    picks = native.get_property_per_eos(0, ["R"], extraction, [1.7] * 40,
                                        rng=np.random.default_rng(0))
    lo = native.get_property_per_eos(0, ["R"], extraction, [1.7], selection_rule="min")
    hi = native.get_property_per_eos(0, ["R"], extraction, [1.7], selection_rule="max")
    assert set(np.round(picks, 12)) == {np.round(lo[0], 12), np.round(hi[0], 12)}
    again = native.get_property_per_eos(0, ["R"], extraction, [1.7] * 40,
                                        rng=np.random.default_rng(0))
    np.testing.assert_array_equal(picks, again)


def test_native_random_keeps_columns_on_same_branch(twin_dir):
    native = make_prior_set(EoSPriorSet, twin_dir)
    extraction = Extraction("interpolation", True, "M", None)
    out = native.get_property_per_eos(0, ["R", "rhoc"], extraction, [1.7] * 30,
                                      rng=np.random.default_rng(1)).reshape(2, 30)
    # larger R always comes with smaller rhoc (first branch)
    first = out[0] > out[0].min() + 1e-9
    assert first.any() and (~first).any()
    assert np.all(out[1][first] < out[1][~first].min())


def test_native_default_values(twin_dir):
    native = make_prior_set(EoSPriorSet, twin_dir)
    extraction = Extraction("interpolation", True, "M", [3.0])
    out = native.get_property(range(2), ["R", "Lambda"], extraction, default_values=[-1.0, 0.0])
    np.testing.assert_array_equal(out.to_numpy(), [[-1.0, 0.0], [-1.0, 0.0]])


def test_native_unbranched_nonmonotonic_raises(twin_dir):
    native = make_prior_set(EoSPriorSet, twin_dir)
    with pytest.raises(RuntimeError, match="monotonic"):
        native.get_property(range(1), ["R"], Extraction("interpolation", True, "M", [1.7], branched=False))


@pytest.mark.parametrize("rule, dependent, error", [
    ("min", ["R", "Lambda"], "one dependent"),
    ("bogus", ["R"], "not understood"),
])
def test_native_bad_selection_rule(twin_dir, rule, dependent, error):
    native = make_prior_set(EoSPriorSet, twin_dir)
    with pytest.raises(ValueError, match=error):
        native.get_property(range(1), dependent,
                            Extraction("interpolation", True, "M", [1.7], selection_rule=rule))


def test_native_and_h5_agree(eos_dir, eos_h5):
    native = make_prior_set(EoSPriorSet, eos_dir)
    extraction = Extraction("interpolation", True, "M", [1.2, 1.5, 1.9])
    pd.testing.assert_frame_equal(native.get_property(range(NUM_EOS), ["R", "rhoc"], extraction),
                                  eos_h5.get_property(range(NUM_EOS), ["R", "rhoc"], extraction))


# ---------------------------------------------------------------------------
# Native extraction vs universality (needs universality)
# ---------------------------------------------------------------------------

UNBRANCHED_CASES = ["static", "static_two_columns", "dynamic", "static_and_dynamic", "nearest_neighbor"]


@pytest.mark.parametrize("case", UNBRANCHED_CASES)
def test_native_matches_universality_unbranched(usamples, eos_dir, case):
    extraction, kwargs = EXTRACTION_CASES[case]
    dependent = DEPENDENT.get(case, ["R"])
    expected = make_prior_set(UniversalityEoSPriorSet, eos_dir).get_property(range(NUM_EOS), dependent, extraction, **kwargs)
    result = make_prior_set(EoSPriorSet, eos_dir).get_property(range(NUM_EOS), dependent, extraction, **kwargs)
    pd.testing.assert_frame_equal(result, expected, rtol=1e-12)


TWIN_CASES = {
    # (values, dependent, selection rule); "random" only where each value is on one branch
    "min": ([1.0, 1.4, 1.65, 1.7, 1.8, 2.0, 2.5], ["R"], "min"),
    "max": ([1.0, 1.4, 1.65, 1.7, 1.8, 2.0, 2.5], ["Lambda"], "max"),
    "random_single_valued": ([0.9, 1.4, 1.55, 1.95, 2.05, 2.5], ["R", "rhoc", "Lambda"], "random"),
    "nearest_neighbor": ([1.4, 1.7, 2.0], ["R", "rhoc"], "nearest_neighbor"),
}


@pytest.mark.parametrize("case", TWIN_CASES)
def test_native_matches_universality_branched(usamples, twin_dir, case):
    values, dependent, rule = TWIN_CASES[case]
    extraction = Extraction("interpolation", True, "M", values, selection_rule=rule)
    kwargs = {"selection_rule": rule, "default_values": [np.nan] * len(dependent)}
    expected = make_prior_set(UniversalityEoSPriorSet, twin_dir).get_property(range(NUM_EOS), dependent, extraction, **kwargs)
    result = make_prior_set(EoSPriorSet, twin_dir).get_property(range(NUM_EOS), dependent, extraction, **kwargs)
    pd.testing.assert_frame_equal(result, expected, rtol=1e-12)
    assert result.notna().any().any()


# ---------------------------------------------------------------------------
# Quantiles (EoSPriorDistribution.get_quantiles)
# ---------------------------------------------------------------------------

def test_native_quantiles_known_answer(eos_dir):
    native = make_prior_set(EoSPriorSet, eos_dir)
    # R(rhoc) is linear in rhoc and draw: 13 - 0.5 * draw - 2 rhoc / 3e15
    qs, median = native.get_quantiles(range(NUM_EOS), "rhoc", "R", [3e14, 1.5e15],
                                      [0.0, 0.5, 1.0], macro=True)
    for i, rho in enumerate([3e14, 1.5e15]):
        r = np.sort(13.0 - 0.5 * np.arange(NUM_EOS) - 2 * rho / 3e15)
        # quantiles interpolate the empirical CDF, which is i/N at the i-th sorted value
        cdf = np.arange(1, NUM_EOS + 1) / NUM_EOS
        np.testing.assert_allclose(qs[:, i], np.interp([0.0, 0.5, 1.0], cdf, r))
        assert median[i] == pytest.approx(np.interp(0.5, cdf, r))
        assert qs[0, i] == r.min() and qs[2, i] == r.max()


def test_native_quantiles_weights_and_coverage(twin_dir):
    from core.weights import quantile
    native = make_prior_set(EoSPriorSet, twin_dir)
    weights = np.array([5.0, 1.0, 0.5, 3.0, 2.0])
    q = [0.1, 0.5, 0.9]
    qs, median = native.get_quantiles(range(NUM_EOS), "M", "R", [1.4, 2.5], q,
                                      weights=weights, macro=True, selection_rule="min")
    extraction = Extraction("interpolation", True, "M")
    r = np.array([native.get_property_per_eos(d, ["R"], extraction, [1.4])[0] for d in range(NUM_EOS)])
    np.testing.assert_allclose(qs[:, 0], quantile(r, q, weights=weights))
    assert median[0] == pytest.approx(quantile(r, [0.5], weights=weights)[0])
    # 2.5 Msun is above every maximum mass
    assert np.isnan(qs[:, 1]).all() and np.isnan(median[1])


def test_native_quantiles_h5_matches_prior_set(eos_dir, eos_h5):
    args = (range(NUM_EOS), "M", "R", [1.2, 1.6], [0.05, 0.5, 0.95])
    kwargs = {"weights": np.arange(1.0, NUM_EOS + 1), "macro": True}
    for a, b in zip(make_prior_set(EoSPriorSet, eos_dir).get_quantiles(*args, **kwargs),
                    eos_h5.get_quantiles(*args, **kwargs)):
        np.testing.assert_array_equal(a, b)


QUANTILE_CASES = {
    # (fixture, x column, y column, x values, selection rule, branched)
    "unbranched_M": ("eos_dir", "M", "R", [1.2, 1.4, 1.8], "random", False),
    "unbranched_rhoc": ("eos_dir", "rhoc", "M", [3e14, 1e15, 2e15], "random", False),
    "branched_min": ("twin_dir", "M", "R", [1.0, 1.4, 1.7, 1.75, 2.0], "min", True),
    "branched_max": ("twin_dir", "M", "Lambda", [1.0, 1.4, 1.7, 1.75, 2.0], "max", True),
    "branched_nearest": ("twin_dir", "M", "R", [1.4, 1.7, 2.0], "nearest_neighbor", True),
}


@pytest.mark.parametrize("case", QUANTILE_CASES)
def test_native_quantiles_match_universality(usamples, request, case):
    fixture, xcol, ycol, x_test, rule, branched = QUANTILE_CASES[case]
    root = request.getfixturevalue(fixture)
    prior_set = make_prior_set(UniversalityEoSPriorSet, root)
    quantiles = np.linspace(0, 1, 21)
    weights = np.random.default_rng(7).exponential(size=NUM_EOS)
    expected_qs, expected_med = usamples.process2quantiles(
        np.arange(NUM_EOS), prior_set.get_path_template(macro=True), EOS_PER_DIR,
        xcol, ycol, x_test, quantiles, weights=weights, selection_rule=rule,
        branches_mapping=prior_set.branches_data if branched else None)
    qs, med = make_prior_set(EoSPriorSet, root).get_quantiles(
        np.arange(NUM_EOS), xcol, ycol, x_test, quantiles, weights=weights,
        macro=True, branched=branched, selection_rule=rule)
    np.testing.assert_allclose(qs, expected_qs, rtol=1e-12)
    np.testing.assert_allclose(med, expected_med, rtol=1e-12)


def test_plotting_get_quantiles_dispatch(usamples, twin_dir):
    """plotting.get_quantiles gives the same table through either prior class."""
    pytest.importorskip("universality")
    from temperance.core.result import EoSPosterior, WeightColumn
    from temperance.plotting import get_quantiles as gq
    posterior = EoSPosterior(pd.DataFrame({"eos": np.arange(NUM_EOS),
                                           "logweight": np.linspace(0, 1, NUM_EOS)}))
    kwargs = dict(weight_columns=[WeightColumn("logweight")], variables=("M", "R"),
                  x_points=np.array([1.2, 1.4, 1.75]), max_num_samples=NUM_EOS,
                  quantiles=np.linspace(0, 1, 11), use_macro=True, selection_rule="min")
    results = []
    for cls in (UniversalityEoSPriorSet, EoSPriorSet):
        np.random.seed(3)
        results.append(gq.get_quantiles(posterior, eos_data=make_prior_set(cls, twin_dir), **kwargs))
    pd.testing.assert_frame_equal(results[1], results[0], rtol=1e-12)


def test_native_eos_prior_set_alias():
    assert eos_prior.NativeEoSPriorSet is EoSPriorSet


def test_get_property_verbose(eos_dir, capsys):
    prior = make_prior_set(EoSPriorSet, eos_dir)
    quiet = prior.get_property(range(NUM_EOS), ["R"], EXTRACTION_CASES["static"][0])
    loud = prior.get_property(range(NUM_EOS), ["R"], EXTRACTION_CASES["static"][0], verbose=True)
    pd.testing.assert_frame_equal(quiet, loud)
    assert f"{NUM_EOS}/{NUM_EOS}" in capsys.readouterr().out
