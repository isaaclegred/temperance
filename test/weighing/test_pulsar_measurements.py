import numpy as np
import pandas as pd
import pytest
import scipy.stats
from bilby.core.prior import TruncatedGaussian
from scipy import integrate, optimize
from scipy.integrate import trapezoid

from temperance import datasets
from temperance.core.populations import truncated_pdf
from temperance.core.result import EoSPosterior
from temperance.weighing.pulsar_measurements import (ONE_SIGMA, DirectMassLikelihood, MassRatioLikelihood,
                                                     PulsarMeasurement, TotalMassLikelihood)


# J1750-37A (Freire et al. 2008), J1518+4904 (Janssen et al. 2008), J1816+4510 (Kaplan et al. 2013),
# B1957+20 (van Kerkwijk et al. 2011, as in Farr & Chatziioannou 2020)
J1750 = dict(mass_function=0.0518649, total_mass=1.97, total_mass_minus=0.15)
J1518 = dict(mass_function=0.115988, total_mass=2.7183, total_mass_minus=0.0007)
J1816 = dict(mass_function=0.0017607, mass_ratio=9.54, mass_ratio_minus=0.21)
B1957 = dict(mass_function=5.0e-6, mass_ratio=69.2, mass_ratio_minus=0.8)


def test_kind_and_symmetric_errors():
    measurement = PulsarMeasurement(mass=2.0, mass_minus=0.1)
    assert measurement.kind == "mass" and measurement.mass_plus == 0.1
    assert isinstance(measurement.get_mass_likelihood(), DirectMassLikelihood)
    assert isinstance(PulsarMeasurement(**J1750).get_mass_likelihood(), TotalMassLikelihood)
    assert isinstance(PulsarMeasurement(**J1816).get_mass_likelihood(), MassRatioLikelihood)


@pytest.mark.parametrize("quantities, message", [
    (dict(mass_function=0.1), "mass function with a total mass"),
    (dict(mass=1.4, mass_minus=0.1, **J1750), "mass function with a total mass"),
    (dict(total_mass=2.6, total_mass_minus=0.1, **J1816), "mass function with a total mass"),
    (dict(mass=1.4), "needs an error"),
])
def test_incomplete_or_ambiguous_measurements_are_refused(quantities, message):
    with pytest.raises(ValueError, match=message):
        PulsarMeasurement(**quantities)


@pytest.mark.parametrize("credible_level", [ONE_SIGMA, 0.9])
def test_quoted_interval_holds_the_credible_level(credible_level):
    likelihood = PulsarMeasurement(mass=1.47, mass_minus=0.06, mass_plus=0.07,
                                   credible_level=credible_level).get_mass_likelihood()
    assert likelihood.integrate() == pytest.approx(1.0, rel=1e-10)
    assert likelihood.integrate(m_min=1.41, m_max=1.54) == pytest.approx(credible_level, rel=1e-10)


def test_total_mass_likelihood_matches_alsing_eq3():
    f, mu, sigma = J1750["mass_function"], J1750["total_mass"], J1750["total_mass_minus"]
    likelihood = PulsarMeasurement(**J1750).get_mass_likelihood()

    def alsing(m):
        # Eq. 3 of Alsing et al. 2018 as written, integrating the 1/cos i edge singularity directly
        def integrand(M):
            sin2 = f ** (2 / 3) * M ** (4 / 3) / (M - m) ** 2
            if sin2 >= 1:
                return 0.0
            return (scipy.stats.norm.pdf(M, mu, sigma) * M ** (4 / 3)
                    / (3 * (M - m) ** 2 * f ** (1 / 3) * np.sqrt(1 - sin2)))
        edge = optimize.brentq(lambda M: (M - m) ** 3 - f * M ** 2, m + 1e-9, 10)
        return integrate.quad(integrand, edge, mu + 10 * sigma, limit=400)[0]

    for m in (0.8, 1.2, 1.4, 1.6, 1.75):
        assert likelihood(m) == pytest.approx(alsing(m), rel=1e-5)


def _monte_carlo_z_scores(likelihood, masses, weights, n, bins):
    """
    Weighted histogram of simulated masses against the likelihood averaged
    over each bin, in units of the Monte Carlo standard error.
    """
    s1, edges = np.histogram(masses, bins=bins, weights=weights)
    s2, _ = np.histogram(masses, bins=bins, weights=weights ** 2)
    width = np.diff(edges)
    estimate = s1 / (n * width)
    error = np.sqrt(s2 / n - (s1 / n) ** 2) / np.sqrt(n) / width
    exact = np.array([likelihood.integrate(m_min=a, m_max=b) / (b - a) for a, b in zip(edges[:-1], edges[1:])])
    well_sampled = estimate > 0.05 * estimate.max()
    return (estimate[well_sampled] - exact[well_sampled]) / error[well_sampled]


@pytest.mark.parametrize("quantities", [J1750, J1518], ids=["wide", "narrow"])
def test_total_mass_likelihood_matches_simulated_binaries(quantities):
    # draw true total masses and isotropic inclinations and solve the mass function for the pulsar
    # mass: L(m) is the density of those masses weighted by 1 / |df/dm| = m_c / 3f
    f, mu, sigma = quantities["mass_function"], quantities["total_mass"], quantities["total_mass_minus"]
    rng, n = np.random.default_rng(3), 2_000_000
    M = rng.normal(mu, sigma, n)
    sin_i = np.sqrt(1 - rng.uniform(0, 1, n) ** 2)
    m = M - np.cbrt(f * M ** 2) / sin_i
    weights = np.where(m > 0, (M - m) / (3 * f), 0.0)
    likelihood = PulsarMeasurement(**quantities).get_mass_likelihood()
    z = _monte_carlo_z_scores(likelihood, m, weights, n, np.linspace(0.2, likelihood.support[1], 30))
    assert np.abs(z).max() < 4 and abs(z.mean()) < 0.5


@pytest.mark.parametrize("quantities", [J1816, B1957], ids=["J1816+4510", "B1957+20"])
def test_mass_ratio_likelihood_matches_simulated_binaries(quantities):
    # as above, with m = f q (1 + q)^2 / sin^3 i and weights 1 / |df/dm| = m / f
    f, mu, sigma = quantities["mass_function"], quantities["mass_ratio"], quantities["mass_ratio_minus"]
    rng, n = np.random.default_rng(4), 2_000_000
    q = rng.normal(mu, sigma, n)
    m = f * q * (1 + q) ** 2 / (1 - rng.uniform(0, 1, n) ** 2) ** 1.5
    likelihood = PulsarMeasurement(**quantities).get_mass_likelihood()
    z = _monte_carlo_z_scores(likelihood, m, m / f, n, np.linspace(likelihood.support[0], 3.0, 30))
    assert np.abs(z).max() < 4 and abs(z.mean()) < 0.5


def test_mass_ratio_is_pulsar_over_companion():
    # with q = m_p / m_c the pulsar mass is at least f q (1 + q)^2: Kaplan et al. 2013 quote
    # m_p sin^3 i = 1.84 +/- 0.11 for J1816+4510, and van Kerkwijk et al. 2011 m_p > 1.66 for B1957+20.
    # Reading q as m_c / m_p instead would put the minimum near 1e-4 Msun.
    for quantities, lower_limit in ((J1816, 1.84 - 2 * 0.11), (B1957, 1.66)):
        likelihood = PulsarMeasurement(**quantities).get_mass_likelihood()
        assert likelihood.support[0] > 1.0
        below = likelihood.integrate(m_max=lower_limit) / likelihood.integrate(m_max=3.0)
        assert below < 0.05
        grid = np.linspace(1.0, 3.0, 2001)
        assert grid[np.argmax(likelihood(grid))] > lower_limit


def test_every_dataset_row_has_a_likelihood():
    table = datasets.load_table("pulsar_masses")
    measurements = PulsarMeasurement.from_table(table)
    for measurement in measurements:
        value = measurement.get_mass_likelihood().integrate(m_min=0.0, m_max=6.0)
        assert np.isfinite(value) and value > 0, (measurement.name, measurement.analysis)
    fan = [m for m in measurements if "fan2024" in m.metadata["compilations"].split()]
    kinds = [m.kind for m in fan]
    assert (kinds.count("mass"), kinds.count("total_mass"), kinds.count("mass_ratio")) == (106, 17, 6)


def test_from_dataset():
    measurement = PulsarMeasurement.from_dataset("J0740+6620", "fonseca2021")
    assert (measurement.mass, measurement.mass_minus, measurement.credible_level) == (2.08, 0.07, 0.683)
    assert measurement.metadata["doi"] == "10.3847/2041-8213/ac03b8"
    companion = PulsarMeasurement.from_dataset("J0453+1559", component="companion")
    assert companion.mass == 1.174
    with pytest.raises(ValueError, match="analyses"):
        PulsarMeasurement.from_dataset("J0740+6620")


def test_samples_follow_the_likelihood():
    for likelihood, m_max in ((PulsarMeasurement(mass=1.47, mass_minus=0.06, mass_plus=0.12).get_mass_likelihood(),
                               None),
                              (PulsarMeasurement(**J1816).get_mass_likelihood(), 3.0)):
        samples = likelihood.sample(20_000, m_max=m_max, rng=5)
        lo, hi = likelihood.support[0], likelihood._interval(None, m_max)[1]
        assert samples.min() >= lo and samples.max() <= hi
        # cumulative integral on a grid five times finer than the one sample() inverts
        grid = np.linspace(lo, hi, 20_001)
        cumulative = np.concatenate([[0.0], np.cumsum(np.diff(grid) * 0.5 * (likelihood(grid[1:])
                                                                               + likelihood(grid[:-1])))])
        assert scipy.stats.kstest(samples, lambda x: np.interp(x, grid, cumulative / cumulative[-1])).pvalue > 0.01
    with pytest.raises(ValueError, match="m_max"):
        PulsarMeasurement(**J1816).get_mass_likelihood().sample(10)


def test_population_weight_resolves_a_narrow_measurement():
    # the double pulsar's masses are known to ~1e-5 Msun, far below the default 2000-point grid spacing
    from temperance.weighing.weigh_by_pulsar import weigh_EoSs_by_mass_measurement
    measurement = PulsarMeasurement.from_dataset("J0737-3039A", "kramer2021_eos_free")
    population = TruncatedGaussian(1.35, 0.1, minimum=1.0, maximum=3.0)
    posterior = EoSPosterior(pd.DataFrame({"eos": [0, 1], "Mmax": [2.0, 1.2]}))
    column = weigh_EoSs_by_mass_measurement(posterior, population=population, log_output_weight=False,
                                            mass_likelihood=measurement.get_mass_likelihood())
    weights = posterior.samples[column.name].to_numpy()
    # reference: a grid that resolves the measurement
    likelihood = measurement.get_mass_likelihood()
    m = np.linspace(*likelihood.support, 200_001)
    expected = trapezoid(likelihood(m) * truncated_pdf(population, m, m_max=2.0), m)
    assert weights[0] == pytest.approx(expected, rel=1e-8)
    assert weights[1] == 0.0
