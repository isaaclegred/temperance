"""
Tests for temperance.weighing.single_event_distributions.
"""

import numpy as np
import pandas as pd
import pytest

from temperance.core.result import EoSPosterior
from temperance.sampling.eos_prior import EoSPriorSet
from temperance.utilities.units import KM_PER_MSUN
from temperance.weighing import weigh_by_density_estimate as wbde
from temperance.weighing.single_event_distributions import (generate_gw_samples, generate_mr_samples,
                                                            sample_neutron_star_binary,
                                                            sample_single_neutron_star,
                                                            uniform_in_components_mass_prior)

N = 300


class RandomMassPrior:
    """Uniform masses from numpy's global RNG, so runs can be repeated with a seed."""
    def __init__(self, m_min=1.0, m_max=2.0):
        self.m_min, self.m_max = m_min, m_max

    def sample(self, n):
        return np.random.uniform(self.m_min, self.m_max, n)


@pytest.fixture
def stars(tmp_path):
    """One EoS with a single stable branch; Mmax set above it so some masses have no stable star."""
    rhoc = np.geomspace(1e14, 3e15, 60)
    M = 2.1 * (1 - np.exp(-rhoc / 6e14))
    table = pd.DataFrame({"rhoc": rhoc, "M": M, "R": 13 - rhoc / 1e15, "Lambda": 500 * (1.4 / M) ** 6})
    subdir = tmp_path / "DRAWmod1000-000000"
    subdir.mkdir()
    table.to_csv(subdir / "macro-draw-000000.csv", index=False)
    prior = EoSPriorSet(eos_dir=str(tmp_path), eos_column="eos", eos_per_dir=1000, macro_dir=str(tmp_path))
    posterior = EoSPosterior(pd.DataFrame({"eos": [0], "Mmax": [2.2]}))
    return prior, posterior, table


def test_single_star_radius(stars):
    prior, posterior, table = stars
    np.random.seed(0)
    out = sample_single_neutron_star(posterior, prior, RandomMassPrior, N)
    assert list(out.columns) == ["eos", "Mmax", "M", "R"] and len(out) == N
    assert (out["Mmax"] == 2.2).all() and out["M"].between(1.0, 2.2).all()
    on_branch = out["M"] <= table["M"].max()
    np.testing.assert_allclose(out.loc[on_branch, "R"], np.interp(out.loc[on_branch, "M"], table["M"], table["R"]))
    # no stable star: the Schwarzschild radius
    assert (~on_branch).any()
    np.testing.assert_allclose(out.loc[~on_branch, "R"], 2 * out.loc[~on_branch, "M"] * KM_PER_MSUN)


def test_single_star_several_observables(stars):
    prior, posterior, table = stars
    np.random.seed(0)
    radius_only = sample_single_neutron_star(posterior, prior, RandomMassPrior, N)
    np.random.seed(0)
    out = sample_single_neutron_star(posterior, prior, RandomMassPrior, N, observables=["R", "Lambda"])
    assert list(out.columns) == ["eos", "Mmax", "M", "R", "Lambda"]
    pd.testing.assert_frame_equal(out[radius_only.columns], radius_only)
    on_branch = out["M"] <= table["M"].max()
    np.testing.assert_allclose(out.loc[on_branch, "Lambda"],
                               np.interp(out.loc[on_branch, "M"], table["M"], table["Lambda"]))
    assert (out.loc[~on_branch, "Lambda"] == 0).all()


def test_black_hole_values(stars):
    prior, posterior, _ = stars
    with pytest.raises(ValueError, match="black-hole value"):
        sample_single_neutron_star(posterior, prior, RandomMassPrior, N, observables=["rhoc"])
    np.random.seed(0)
    out = sample_single_neutron_star(posterior, prior, RandomMassPrior, N, observables=["rhoc"],
                                     black_hole_values={"rhoc": lambda m: np.full_like(np.asarray(m, float), -1.0)})
    assert (out.loc[out["M"] > 2.1, "rhoc"] == -1.0).all()


def test_generate_mr_samples_is_the_radius_case(stars):
    prior, posterior, _ = stars
    np.random.seed(3)
    expected = sample_single_neutron_star(posterior, prior, RandomMassPrior, N, observables=["R"])
    for function in (generate_mr_samples, wbde.generate_mr_samples):
        np.random.seed(3)
        pd.testing.assert_frame_equal(function(posterior, prior, RandomMassPrior, N), expected)




# ---------------------------------------------------------------------------
# How many samples: eos_distribution / num_samples_per_eos / max_total_samples
# ---------------------------------------------------------------------------

@pytest.fixture
def two_eos(stars):
    prior, _, _ = stars
    return prior, EoSPosterior(pd.DataFrame({"eos": [0, 0], "Mmax": [2.2, 2.0]}))


def test_masses_only_without_an_eos_distribution():
    np.random.seed(1)
    out = sample_single_neutron_star(mass_prior_class=RandomMassPrior, max_total_samples=50)
    assert list(out.columns) == ["M"] and len(out) == 50
    np.random.seed(1)
    np.testing.assert_array_equal(out["M"], RandomMassPrior().sample(50))
    with pytest.raises(ValueError, match="max_total_samples"):
        sample_single_neutron_star(mass_prior_class=RandomMassPrior)
    with pytest.raises(ValueError, match="need an eos_distribution"):
        sample_single_neutron_star(mass_prior_class=RandomMassPrior, num_samples_per_eos=10, max_total_samples=50)


def test_max_total_samples_is_divided_among_the_eoss(two_eos, capsys):
    prior, distribution = two_eos
    np.random.seed(2)
    out = sample_single_neutron_star(eos_distribution=distribution, eos_prior_set=prior,
                                     mass_prior_class=RandomMassPrior, max_total_samples=301)
    assert "Drawing 300 of at most 301 samples (150 for each of 2 EoSs)" in capsys.readouterr().out
    assert len(out) == 300 and (out["Mmax"].to_numpy() == np.repeat([2.2, 2.0], 150)).all()
    np.random.seed(2)
    per_eos = sample_single_neutron_star(distribution, prior, RandomMassPrior, 150)
    pd.testing.assert_frame_equal(out, per_eos)
    with pytest.raises(ValueError, match="fewer than one sample"):
        sample_single_neutron_star(distribution, prior, RandomMassPrior, max_total_samples=1)


def test_num_samples_per_eos_within_max_total_samples(two_eos):
    prior, distribution = two_eos
    out = sample_single_neutron_star(distribution, prior, RandomMassPrior, 100, max_total_samples=250)
    assert len(out) == 200
    assert len(sample_single_neutron_star(distribution, prior, RandomMassPrior, 100, max_total_samples=200)) == 200
    with pytest.raises(ValueError, match="more than max_total_samples=150"):
        sample_single_neutron_star(distribution, prior, RandomMassPrior, 100, max_total_samples=150)


def test_argument_errors(two_eos):
    prior, distribution = two_eos
    with pytest.raises(TypeError, match="mass_prior_class"):
        sample_single_neutron_star(distribution, prior, num_samples_per_eos=10)
    with pytest.raises(ValueError, match="eos_prior_set"):
        sample_single_neutron_star(distribution, mass_prior_class=RandomMassPrior, num_samples_per_eos=10)
    with pytest.raises(ValueError, match="num_samples_per_eos, max_total_samples, or both"):
        sample_single_neutron_star(distribution, prior, RandomMassPrior)


# ---------------------------------------------------------------------------
# Binaries
# ---------------------------------------------------------------------------

def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


@pytest.fixture
def bilby_seed():
    from bilby.core.utils import random
    def seed(value):
        random.seed(value)
        np.random.seed(value)      # (choose_macro_per_m picks twin branches with numpy's RNG)
    return seed


def test_uniform_in_components_mass_prior(bilby_seed):
    import scipy.stats
    bilby_seed(1)
    prior = uniform_in_components_mass_prior(chirp_mass=(1.15, 1.25), mass_ratio=(0.6, 1.0))
    out = sample_neutron_star_binary(mass_prior=prior, max_total_samples=5000)
    assert list(out.columns) == ["mass_1", "mass_2"] and len(out) == 5000
    assert (out["mass_1"] >= out["mass_2"]).all()
    assert chirp_mass(out["mass_1"], out["mass_2"]).between(1.15, 1.25).all()
    assert (out["mass_2"] / out["mass_1"]).between(0.6, 1.0).all()
    # uniform in the components: compare with uniform (m1, m2) kept if inside the same bounds
    rng = np.random.default_rng(2)
    m = rng.uniform(1.0, 2.0, (2, 400_000))
    m1, m2 = m.max(axis=0), m.min(axis=0)
    keep = (chirp_mass(m1, m2) > 1.15) & (chirp_mass(m1, m2) < 1.25) & (m2 / m1 > 0.6)
    assert scipy.stats.ks_2samp(out["mass_1"], m1[keep]).pvalue > 1e-3
    assert scipy.stats.ks_2samp(out["mass_2"], m2[keep]).pvalue > 1e-3
    # component-mass bounds are applied as constraints
    bounded = uniform_in_components_mass_prior((1.15, 1.25), (0.6, 1.0), mass_1=(1.0, 1.5))
    assert (sample_neutron_star_binary(mass_prior=bounded, max_total_samples=500)["mass_1"] <= 1.5).all()


def test_source_frame_names(bilby_seed):
    from bilby.core.prior import PriorDict
    prior = uniform_in_components_mass_prior(chirp_mass=(1.15, 1.25), mass_ratio=(0.6, 1.0))
    source_named = PriorDict({"chirp_mass_source": prior["chirp_mass"], "mass_ratio": prior["mass_ratio"]})
    bilby_seed(3)
    plain = sample_neutron_star_binary(mass_prior=prior, max_total_samples=20)
    bilby_seed(3)
    pd.testing.assert_frame_equal(sample_neutron_star_binary(mass_prior=source_named, max_total_samples=20), plain)
    mixed = PriorDict({"chirp_mass": prior["chirp_mass"], "mass_1_source": prior["chirp_mass"],
                       "mass_ratio": prior["mass_ratio"]})
    with pytest.raises(ValueError, match="mixes plain mass names"):
        sample_neutron_star_binary(mass_prior=mixed, max_total_samples=5)


def test_binary_observables(stars, bilby_seed):
    eos_prior, distribution, table = stars
    bilby_seed(4)
    wide = uniform_in_components_mass_prior(chirp_mass=(0.9, 1.9), mass_ratio=(0.5, 1.0))
    out = sample_neutron_star_binary(distribution, eos_prior, wide, N, observables=["Lambda", "R"],
                                     objects="compact_objects")
    assert list(out.columns) == ["eos", "Mmax", "mass_1", "mass_2", "lambda_1", "lambda_2", "R_1", "R_2"]
    assert len(out) == N and (out["mass_1"] >= out["mass_2"]).all()
    for k in (1, 2):
        mass = out[f"mass_{k}"]
        neutron_star = mass <= table["M"].max()
        np.testing.assert_allclose(out.loc[neutron_star, f"lambda_{k}"],
                                   np.interp(mass[neutron_star], table["M"], table["Lambda"]))
        np.testing.assert_allclose(out.loc[neutron_star, f"R_{k}"],
                                   np.interp(mass[neutron_star], table["M"], table["R"]))
        # no stable star: a black hole
        assert (out.loc[~neutron_star, f"lambda_{k}"] == 0).all()
        np.testing.assert_allclose(out.loc[~neutron_star, f"R_{k}"], 2 * mass[~neutron_star] * KM_PER_MSUN)
    assert (out["mass_1"] > table["M"].max()).any()


def test_components_are_relabelled_heavier_first(stars, bilby_seed):
    from bilby.core.prior import PriorDict, Uniform
    eos_prior, distribution, table = stars
    independent = PriorDict({"mass_1": Uniform(1.0, 2.0), "mass_2": Uniform(1.0, 2.0)})
    bilby_seed(5)
    out = sample_neutron_star_binary(distribution, eos_prior, independent, N)
    assert (out["mass_1"] >= out["mass_2"]).all()
    neutron_stars = out["mass_1"] <= table["M"].max()
    np.testing.assert_allclose(out.loc[neutron_stars, "lambda_1"],
                               np.interp(out.loc[neutron_stars, "mass_1"], table["M"], table["Lambda"]))


def test_prior_as_a_function_of_maximum_mass(stars, bilby_seed):
    eos_prior, _, table = stars
    # (the fixtures' Mmax columns are deliberately above the table's maximum; use the true one here)
    distribution = EoSPosterior(pd.DataFrame({"eos": [0, 0], "Mmax": [table["M"].max(), 1.8]}))
    def below_m_max(m_max):
        return uniform_in_components_mass_prior((0.9, 1.9), (0.5, 1.0), mass_1=(0.5, m_max))
    bilby_seed(6)
    out = sample_neutron_star_binary(distribution, eos_prior, below_m_max, 100)
    assert (out["mass_1"] <= out["Mmax"]).all() and (out["lambda_1"] > 0).all()
    with pytest.raises(ValueError, match="no maximum mass"):
        sample_neutron_star_binary(mass_prior=below_m_max, max_total_samples=10)


def test_binary_sample_counts_and_reproducibility(two_eos, bilby_seed, capsys):
    eos_prior, distribution = two_eos
    prior = uniform_in_components_mass_prior((0.9, 1.9), (0.5, 1.0))
    bilby_seed(7)
    first = sample_neutron_star_binary(distribution, eos_prior, prior, max_total_samples=51)
    assert len(first) == 50 and "Drawing 50 of at most 51 samples" in capsys.readouterr().out
    bilby_seed(7)
    pd.testing.assert_frame_equal(sample_neutron_star_binary(distribution, eos_prior, prior, 25), first)
    with pytest.raises(ValueError, match="more than max_total_samples"):
        sample_neutron_star_binary(distribution, eos_prior, prior, 25, max_total_samples=49)
    with pytest.raises(TypeError, match="mass_prior"):
        sample_neutron_star_binary(distribution, eos_prior, num_samples_per_eos=10)


# ---------------------------------------------------------------------------
# Neutron stars (below each EoS's maximum mass) or compact objects
# ---------------------------------------------------------------------------

def test_single_star_neutron_stars_or_compact_objects(stars):
    eos_prior, distribution, table = stars           # Mmax column: 2.2
    wide = {"m_min": 1.0, "m_max": 3.0}
    np.random.seed(8)
    neutron_stars = sample_single_neutron_star(distribution, eos_prior, RandomMassPrior, N, mass_prior_kwargs=wide)
    assert neutron_stars["M"].max() <= 2.2           # capped at min(3.0, Mmax)
    np.random.seed(8)
    compact = sample_single_neutron_star(distribution, eos_prior, RandomMassPrior, N, mass_prior_kwargs=wide,
                                         objects="compact_objects")
    above = compact["M"] > 2.2
    assert above.any() and compact["M"].max() <= 3.0
    np.testing.assert_allclose(compact.loc[above, "R"], 2 * compact.loc[above, "M"] * KM_PER_MSUN)
    with pytest.raises(ValueError, match="objects must be one of"):
        sample_single_neutron_star(distribution, eos_prior, RandomMassPrior, N, objects="stars")


def test_generate_mr_samples_keeps_an_explicit_m_max(stars):
    eos_prior, distribution, _ = stars
    kwargs = {"m_min": 1.0, "m_max": 3.0}
    np.random.seed(9)
    legacy = generate_mr_samples(distribution, eos_prior, RandomMassPrior, N, mass_prior_kwargs=kwargs)
    assert legacy["M"].max() > 2.2                   # as before: an explicit m_max is not capped
    np.random.seed(9)
    pd.testing.assert_frame_equal(legacy, sample_single_neutron_star(
        distribution, eos_prior, RandomMassPrior, N, mass_prior_kwargs=kwargs, objects="compact_objects"))


def test_binary_neutron_stars_are_the_prior_conditioned_below_m_max(two_eos, bilby_seed):
    import scipy.stats
    eos_prior, distribution = two_eos                # Mmax columns 2.2 and 2.0
    wide = uniform_in_components_mass_prior(chirp_mass=(0.9, 2.2), mass_ratio=(0.5, 1.0))
    bilby_seed(10)
    neutron_stars = sample_neutron_star_binary(distribution, eos_prior, wide, 2000)
    assert (neutron_stars["mass_1"] <= neutron_stars["Mmax"]).all()
    bilby_seed(11)
    compact = sample_neutron_star_binary(distribution, eos_prior, wide, 4000, objects="compact_objects")
    assert (compact["mass_1"] > compact["Mmax"]).any()
    # neutron stars = compact objects restricted to mass_1 <= Mmax, for each EoS
    for m_max in (2.2, 2.0):
        ns = neutron_stars[neutron_stars["Mmax"] == m_max]
        co = compact[(compact["Mmax"] == m_max) & (compact["mass_1"] <= m_max)]
        assert scipy.stats.ks_2samp(ns["mass_1"], co["mass_1"]).pvalue > 1e-3
        assert scipy.stats.ks_2samp(ns["mass_2"], co["mass_2"]).pvalue > 1e-3


def test_binary_neutron_stars_need_support_below_m_max(two_eos, bilby_seed):
    eos_prior, distribution = two_eos
    too_heavy = uniform_in_components_mass_prior(chirp_mass=(2.5, 3.0), mass_ratio=(0.8, 1.0))
    bilby_seed(12)
    with pytest.raises(ValueError, match="no support"):
        sample_neutron_star_binary(distribution, eos_prior, too_heavy, 5)
    out = sample_neutron_star_binary(distribution, eos_prior, too_heavy, 5, objects="compact_objects")
    assert (out["lambda_1"] == 0).all() and (out["lambda_2"] == 0).all()   # all black holes
    with pytest.raises(ValueError, match="objects must be one of"):
        sample_neutron_star_binary(distribution, eos_prior, too_heavy, 5, objects="black_holes")


# ---------------------------------------------------------------------------
# generate_gw_samples (legacy wrapper)
# ---------------------------------------------------------------------------

def test_generate_gw_samples(stars):
    eos_prior, distribution, table = stars
    kwargs = {"m_min": 1.0, "m_max": 3.0}
    np.random.seed(13)
    out = generate_gw_samples(distribution, eos_prior, RandomMassPrior, N, mass_prior_kwargs=kwargs)
    assert list(out.columns) == ["eos", "Mmax", "mass_1", "mass_2", "lambda_1", "lambda_2"] and len(out) == N
    # the two masses are independent draws from the 1D prior, relabelled heavier-first
    np.random.seed(13)
    first, second = RandomMassPrior(**kwargs).sample(N), RandomMassPrior(**kwargs).sample(N)
    np.testing.assert_array_equal(out["mass_1"], np.maximum(first, second))
    np.testing.assert_array_equal(out["mass_2"], np.minimum(first, second))
    # compact objects: the prior is not truncated, and black holes have Lambda = 0
    for k in (1, 2):
        mass = out[f"mass_{k}"]
        neutron_star = mass <= table["M"].max()
        np.testing.assert_allclose(out.loc[neutron_star, f"lambda_{k}"],
                                   np.interp(mass[neutron_star], table["M"], table["Lambda"]))
        assert (out.loc[~neutron_star, f"lambda_{k}"] == 0).all()
    assert (out["mass_1"] > out["Mmax"]).any()
    # still importable from its old home
    np.random.seed(13)
    pd.testing.assert_frame_equal(wbde.generate_gw_samples(distribution, eos_prior, RandomMassPrior, N,
                                                           mass_prior_kwargs=kwargs), out)
