"""
Tests for temperance.core.populations (bilby priors as mass populations) and
the population option of weighing.weigh_by_pulsar.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from bilby.core.prior import Gaussian, PowerLaw, TruncatedGaussian, Uniform

from temperance.core.populations import (mixture, population_fraction, sample_truncated, truncated_cdf,
                                         truncated_normalisation, truncated_pdf)
from temperance.core.result import EoSPosterior
from scipy.integrate import trapezoid

POPULATIONS = {
    "uniform": Uniform(1.0, 2.5),
    "truncated_gaussian": TruncatedGaussian(1.33, 0.09, minimum=1.0, maximum=3.0),
    "power_law": PowerLaw(-2.5, minimum=1.0, maximum=3.0),
    "bimodal": mixture([TruncatedGaussian(1.393, 0.064, 1.0, 3.0), TruncatedGaussian(1.807, 0.177, 1.0, 3.0)],
                       [0.65, 0.35]),
}


@pytest.mark.parametrize("name", POPULATIONS)
@pytest.mark.parametrize("m_max", [None, 1.9])
def test_truncated_population_is_normalised(name, m_max):
    population = POPULATIONS[name]
    hi = population.maximum if m_max is None else min(m_max, population.maximum)
    m = np.linspace(population.minimum, hi, 20001)      # the density jumps at the bounds
    assert trapezoid(truncated_pdf(population, m, m_max=m_max), m) == pytest.approx(1.0, rel=1e-4)
    assert truncated_cdf(population, population.minimum, m_max) == pytest.approx(0.0, abs=1e-12)
    assert truncated_cdf(population, hi, m_max) == pytest.approx(1.0)
    assert np.all(np.diff(truncated_cdf(population, m, m_max)) >= -1e-12)
    outside = np.array([population.minimum - 0.1, hi + 0.1])
    np.testing.assert_array_equal(truncated_pdf(population, outside, m_max=m_max), [0.0, 0.0])


def test_truncation_matches_scipy():
    population = TruncatedGaussian(1.4, 0.2, minimum=1.1, maximum=3.0)
    a, b = (1.1 - 1.4) / 0.2, (1.8 - 1.4) / 0.2
    m = np.linspace(1.15, 1.75, 7)
    np.testing.assert_allclose(truncated_pdf(population, m, m_max=1.8), scipy.stats.truncnorm(a, b, 1.4, 0.2).pdf(m))
    np.testing.assert_allclose(truncated_cdf(population, m, m_max=1.8), scipy.stats.truncnorm(a, b, 1.4, 0.2).cdf(m))
    assert truncated_normalisation(population, 1.8) == pytest.approx(
        population.cdf(1.8), rel=1e-12)


def test_power_law_truncation():
    population = PowerLaw(-2.5, minimum=1.0, maximum=3.0)
    assert truncated_cdf(population, 2.0) == pytest.approx((1 - 2.0**-1.5) / (1 - 3.0**-1.5))
    assert truncated_cdf(population, 1.5, m_max=2.0) == pytest.approx((1 - 1.5**-1.5) / (1 - 2.0**-1.5))


def test_unbounded_population_is_rejected():
    with pytest.raises(ValueError, match="finite minimum"):
        truncated_pdf(Gaussian(1.33, 0.09), [1.3], m_max=2.0)


def test_mixture_is_the_weighted_sum():
    low, high = TruncatedGaussian(1.393, 0.064, 1.0, 3.0), TruncatedGaussian(1.807, 0.177, 1.0, 3.0)
    bimodal = POPULATIONS["bimodal"]
    m = np.linspace(1.1, 2.4, 9)
    np.testing.assert_allclose(bimodal.prob(m), 0.65 * low.prob(m) + 0.35 * high.prob(m), rtol=1e-3)
    assert truncated_cdf(bimodal, 1.6, m_max=1.6) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="finite range"):
        mixture([Gaussian(1.4, 0.1), Gaussian(1.8, 0.2)], [0.5, 0.5])
    with pytest.raises(ValueError, match="one weight"):
        mixture([low, high], [1.0])


@pytest.mark.parametrize("name", POPULATIONS)
def test_samples_follow_the_truncated_distribution(name):
    population = POPULATIONS[name]
    samples = sample_truncated(population, 4000, m_max=2.0, rng=3)
    assert samples.min() >= population.minimum and samples.max() <= 2.0
    assert scipy.stats.kstest(samples, lambda m: truncated_cdf(population, m, m_max=2.0)).pvalue > 1e-3


def test_population_fraction_closed_form():
    M = np.linspace(0.5, 2.0, 3001)
    population = Uniform(1.0, 5.0)
    assert population_fraction(M, M >= 1.5, population) == pytest.approx(0.5, abs=1e-3)
    assert population_fraction(M, np.zeros_like(M, bool), population) == 0.0
    assert np.isnan(population_fraction([], [], population))


def twin_sequence():
    """Branch 0 reaches 2.0; branch 1 covers 1.8 - 2.2, and only branch 1 has the property."""
    M0, M1 = np.linspace(0.5, 2.0, 151), np.linspace(1.8, 2.2, 41)
    return (np.r_[M0, M1], np.r_[np.zeros(151, bool), np.ones(41, bool)],
            np.r_[np.zeros(151, int), np.ones(41, int)])


def test_population_fraction_branch_rules():
    M, has, branch = twin_sequence()
    population = Uniform(1.0, 5.0)         # truncated at M_max = 2.2
    any_rule = population_fraction(M, has, population, branch=branch, branch_rule="any")
    all_rule = population_fraction(M, has, population, branch=branch, branch_rule="all")
    first = population_fraction(M, has, population, branch=branch, branch_rule="first")
    assert any_rule == pytest.approx((2.2 - 1.8) / 1.2, abs=3e-3)       # anywhere on branch 1
    assert all_rule == pytest.approx((2.2 - 2.0) / 1.2, abs=3e-3)       # only above branch 0
    assert first == pytest.approx(all_rule, abs=3e-3)                   # branch 0 wins the overlap
    with pytest.raises(ValueError, match="branch_rule"):
        population_fraction(M, has, population, branch=branch, branch_rule="majority")


def test_population_fraction_matches_trapped_null_geos():
    trapgeo_population = pytest.importorskip("trapgeo.population")
    rng = np.random.default_rng(8)
    for target, ours in [("uniform_gw", Uniform(1.0, 10.0)),
                         ("galactic_dns", TruncatedGaussian(1.33, 0.09, minimum=1.0, maximum=10.0))]:
        M = np.sort(rng.uniform(0.3, 2.3, 200))
        has = rng.uniform(size=200) < 0.3
        assert population_fraction(M, has, ours) == pytest.approx(
            trapgeo_population.population_fraction(M, has, target), abs=2e-3)


def test_pulsar_weights_with_a_uniform_population_match_the_old_interface():
    from temperance.weighing.weigh_by_pulsar import weigh_EoSs_by_mass_measurement
    m_max = np.array([1.8, 2.0, 2.3, 0.9])
    mass_likelihood = lambda m: scipy.stats.norm(2.01, 0.04).pdf(m)
    integrated = lambda M: scipy.stats.norm(2.01, 0.04).cdf(M) - scipy.stats.norm(2.01, 0.04).cdf(1.0)
    weights = {}
    for label, kwargs in [("old", dict(likelihood=integrated)),
                          ("population", dict(population=Uniform(1.0, 5.0), mass_likelihood=mass_likelihood))]:
        posterior = EoSPosterior(pd.DataFrame({"eos": np.arange(4), "Mmax": m_max}))
        column = weigh_EoSs_by_mass_measurement(posterior, weight_tag=label, log_output_weight=False, **kwargs)
        weights[label] = posterior.samples[column.name].to_numpy()
    np.testing.assert_allclose(weights["population"], weights["old"], rtol=1e-5, atol=1e-12)
    assert weights["population"][-1] == 0.0
    with pytest.raises(ValueError, match="mass_likelihood"):
        weigh_EoSs_by_mass_measurement(EoSPosterior(pd.DataFrame({"eos": [0], "Mmax": [2.0]})),
                                       population=Uniform(1.0, 5.0))


def test_pulsar_weights_with_a_gaussian_population():
    from temperance.weighing.weigh_by_pulsar import weigh_EoSs_by_mass_measurement
    population = TruncatedGaussian(1.5, 0.3, minimum=1.0, maximum=4.0)
    likelihood = scipy.stats.norm(2.0, 0.05)
    posterior = EoSPosterior(pd.DataFrame({"eos": [0, 1], "Mmax": [2.1, 2.5]}))
    column = weigh_EoSs_by_mass_measurement(posterior, population=population, mass_likelihood=likelihood.pdf,
                                            log_output_weight=False)
    m = np.linspace(1.0, 2.1, 200001)
    expected = trapezoid(likelihood.pdf(m) * truncated_pdf(population, m, m_max=2.1), m)
    assert posterior.samples[column.name].iloc[0] == pytest.approx(expected, rel=1e-5)
