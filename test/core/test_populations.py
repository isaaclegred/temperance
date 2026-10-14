"""
Tests for temperance.core.populations and the population option of
weighing.weigh_by_pulsar.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from temperance.core.populations import Gaussian, Mixture, PowerLaw, Uniform, population_fraction
from temperance.core.result import EoSPosterior

POPULATIONS = {
    "uniform": Uniform(1.0, 2.5),
    "gaussian": Gaussian(1.33, 0.09, m_min=1.0),
    "power_law": PowerLaw(-2.5, m_min=1.0, m_max=3.0),
    "bimodal": Mixture([Gaussian(1.393, 0.064), Gaussian(1.807, 0.177)], [0.65, 0.35], m_min=1.0),
}


@pytest.mark.parametrize("name", POPULATIONS)
@pytest.mark.parametrize("m_max", [None, 1.9])
def test_normalised(name, m_max):
    population = POPULATIONS[name]
    hi = min(4.0 if m_max is None else m_max, population.m_max)
    m = np.linspace(population.m_min, hi, 20001)      # the density jumps at the bounds
    assert np.trapezoid(population.pdf(m, m_max=m_max), m) == pytest.approx(1.0, rel=1e-6)
    assert population.cdf(population.m_min, m_max) == pytest.approx(0.0, abs=1e-12)
    assert population.cdf(hi, m_max) == pytest.approx(1.0)
    assert np.all(np.diff(population.cdf(m, m_max)) >= -1e-15)
    assert population.pdf(np.array([population.m_min - 0.1, hi + 0.1]), m_max=m_max) == pytest.approx([0.0, 0.0])


def test_truncated_gaussian_matches_scipy():
    population = Gaussian(1.4, 0.2, m_min=1.1)
    a, b = (1.1 - 1.4) / 0.2, (1.8 - 1.4) / 0.2
    m = np.linspace(1.15, 1.75, 7)
    np.testing.assert_allclose(population.pdf(m, m_max=1.8), scipy.stats.truncnorm(a, b, 1.4, 0.2).pdf(m))


def test_mixture_weights_and_mass_fraction():
    bimodal = POPULATIONS["bimodal"]
    low, high = bimodal.components
    m = np.linspace(1.2, 2.2, 5)
    # before truncation (besides m_min) the mixture is the weighted sum of its parts
    expected = (0.65 * low.pdf(m) + 0.35 * high.pdf(m)) / (0.65 * (1 - low.cdf(1.0)) + 0.35 * (1 - high.cdf(1.0)))
    np.testing.assert_allclose(bimodal.pdf(m) * (0.65 + 0.35), expected * (0.65 + 0.35), rtol=1e-6)
    assert bimodal.mass_fraction(1.0, 1.6) > 0.6
    assert bimodal.mass_fraction(1.0, 1.6, m_max=1.6) == pytest.approx(1.0)


def test_power_law_cdf():
    population = PowerLaw(-2.5, m_min=1.0, m_max=3.0)
    exact = (1 - 2.0**-1.5) / (1 - 3.0**-1.5)
    assert population.cdf(2.0) == pytest.approx(exact)
    with pytest.raises(ValueError):
        PowerLaw(1.0, m_min=1.0)


@pytest.mark.parametrize("name", POPULATIONS)
def test_samples_follow_the_distribution(name):
    population = POPULATIONS[name]
    samples = population.sample(4000, m_max=2.0, rng=3)
    assert samples.min() >= population.m_min and samples.max() <= 2.0
    assert scipy.stats.kstest(samples, lambda m: population.cdf(m, m_max=2.0)).pvalue > 1e-3


def test_population_fraction_closed_form():
    M = np.linspace(0.5, 2.0, 3001)
    f = population_fraction(M, M >= 1.5, Uniform(1.0))
    assert f == pytest.approx(0.5, abs=1e-3)
    assert population_fraction(M, np.zeros_like(M, bool), Uniform(1.0)) == 0.0
    assert np.isnan(population_fraction([], [], Uniform(1.0)))


def twin_sequence():
    """Branch 0 reaches 2.0; branch 1 covers 1.8 - 2.2, and only branch 1 has the property."""
    M0, M1 = np.linspace(0.5, 2.0, 151), np.linspace(1.8, 2.2, 41)
    return (np.r_[M0, M1], np.r_[np.zeros(151, bool), np.ones(41, bool)],
            np.r_[np.zeros(151, int), np.ones(41, int)])


def test_population_fraction_branch_rules():
    M, has, branch = twin_sequence()
    population = Uniform(1.0)              # truncated at M_max = 2.2
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
    for target, ours in [("uniform_gw", Uniform(1.0)), ("galactic_dns", Gaussian(1.33, 0.09, m_min=1.0))]:
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
                          ("population", dict(population=Uniform(1.0), mass_likelihood=mass_likelihood))]:
        posterior = EoSPosterior(pd.DataFrame({"eos": np.arange(4), "Mmax": m_max}))
        column = weigh_EoSs_by_mass_measurement(posterior, weight_tag=label, log_output_weight=False, **kwargs)
        weights[label] = posterior.samples[column.name].to_numpy()
    np.testing.assert_allclose(weights["population"], weights["old"], rtol=1e-5, atol=1e-12)
    assert weights["population"][-1] == 0.0
    with pytest.raises(ValueError, match="mass_likelihood"):
        weigh_EoSs_by_mass_measurement(EoSPosterior(pd.DataFrame({"eos": [0], "Mmax": [2.0]})),
                                       population=Uniform(1.0))


def test_pulsar_weights_with_a_gaussian_population():
    from temperance.weighing.weigh_by_pulsar import weigh_EoSs_by_mass_measurement
    population = Gaussian(1.5, 0.3, m_min=1.0)
    likelihood = scipy.stats.norm(2.0, 0.05)
    posterior = EoSPosterior(pd.DataFrame({"eos": [0, 1], "Mmax": [2.1, 2.5]}))
    column = weigh_EoSs_by_mass_measurement(posterior, population=population, mass_likelihood=likelihood.pdf,
                                            log_output_weight=False)
    m = np.linspace(1.0, 2.1, 200001)
    expected = np.trapezoid(likelihood.pdf(m) * population.pdf(m, m_max=2.1), m)
    assert posterior.samples[column.name].iloc[0] == pytest.approx(expected, rel=1e-5)
