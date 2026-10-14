"""
Tests for the probability / averaging statistics in temperance.core.weights
and their EoSPosterior wrappers.
"""

import numpy as np
import pandas as pd
import pytest

from temperance.core.result import EoSPosterior, WeightColumn, average_over_posteriors
from temperance.core.weights import (conditional_probability, estimate_stratified, kish_neff, model_average,
                                     neff, weighted_histogram, weighted_probability)

N = 4000


@pytest.fixture
def draws():
    rng = np.random.default_rng(5)
    compactness = rng.uniform(0.1, 0.36, N)
    weights = rng.exponential(size=N) * np.exp(-((compactness - 0.25) / 0.05) ** 2)
    weights[17] = np.nan
    return compactness, weights


@pytest.fixture
def posterior(draws):
    compactness, weights = draws
    samples = pd.DataFrame({"eos": np.arange(N), "C": compactness, "weight_astro": np.nan_to_num(weights)})
    return EoSPosterior(samples, weight_columns=[WeightColumn("weight_astro", is_log=False)])


def test_effective_sample_sizes():
    assert kish_neff(np.ones(50)) == pytest.approx(50)
    assert neff(np.ones(50)) == pytest.approx(50)
    w = np.r_[np.ones(10), 0.0, np.nan]
    assert kish_neff(w) == pytest.approx(10)
    # for unequal weights the two definitions differ
    w = np.array([1.0, 2.0, 4.0, 8.0])
    assert kish_neff(w) == pytest.approx(15**2 / 85)
    assert kish_neff(w) != pytest.approx(neff(w))


def test_zero_count_upper_bound():
    """From TrappedNullGeos: 1000 equal weights, nothing seen -> 2.3e-3 at 90%."""
    out = weighted_probability(np.zeros(1000, bool), np.ones(1000))
    assert out["P"] == 0 and out["n_true"] == 0
    assert out["upper_if_zero"] == pytest.approx(2.3e-3, abs=2e-4)


def test_weighted_probability(draws):
    compactness, weights = draws
    indicator = compactness > 0.3
    out = weighted_probability(indicator, weights)
    good = np.isfinite(weights)
    assert out["P"] == pytest.approx(np.sum(weights[good] * indicator[good]) / np.sum(weights[good]))
    assert out["n_true"] == int(indicator[good].sum())
    # delta-method sigma agrees with a bootstrap of the ratio estimator
    rng = np.random.default_rng(0)
    w, x = weights[good], indicator[good].astype(float)
    boots = []
    for _ in range(1000):
        pick = rng.integers(0, len(w), len(w))
        boots.append(np.sum(w[pick] * x[pick]) / np.sum(w[pick]))
    assert out["sigma"] == pytest.approx(np.std(boots), rel=0.1)


def test_conditional_probability():
    a = np.array([1, 1, 0, 0, 1], bool)
    b = np.array([1, 1, 1, 0, 0], bool)
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert conditional_probability(a, b, w)["P"] == pytest.approx(3 / 6)
    assert np.isnan(conditional_probability(a, np.zeros(5, bool), w)["P"])


def test_posterior_probability_accepts_columns_arrays_callables(posterior, draws):
    compactness, weights = draws
    expected = weighted_probability(compactness > 0.3, np.nan_to_num(weights))["P"]
    posterior.samples["compact"] = posterior.samples["C"] > 0.3
    assert posterior.probability("compact")["P"] == pytest.approx(expected)
    assert posterior.probability(compactness > 0.3)["P"] == pytest.approx(expected)
    assert posterior.probability(lambda samples: samples["C"] > 0.3)["P"] == pytest.approx(expected)
    both = posterior.conditional_probability(lambda s: s["C"] > 0.33, lambda s: s["C"] > 0.3)
    assert 0 < both["P"] < 1
    with pytest.raises(ValueError, match="one entry per sample"):
        posterior.probability(np.ones(3, bool))


def test_stratified_estimate_equals_direct_when_whole_stratum_evaluated(draws):
    compactness, weights = draws
    stratum = compactness > 0.28
    values = (compactness > 0.32).astype(float)       # zero outside the stratum
    out = estimate_stratified(weights, stratum, stratum, values, n_boot=200, rng=1)
    assert out["P"] == pytest.approx(weighted_probability(values, np.nan_to_num(weights))["P"])
    good = np.isfinite(weights)
    assert out["stratum_mass"] == pytest.approx(np.sum(weights[good & stratum]) / np.sum(weights[good]))


def test_stratified_estimate_from_a_subsample(draws):
    compactness, weights = draws
    stratum = compactness > 0.28
    values = (compactness > 0.32).astype(float)
    exact = weighted_probability(values, np.nan_to_num(weights))["P"]
    estimates, sigmas = [], []
    for seed in range(100, 140):     # (not the fixture's seed, whose uniforms define C)
        rng = np.random.default_rng(seed)
        evaluated = stratum & (rng.uniform(size=N) < 0.25)
        out = estimate_stratified(weights, stratum, evaluated, values, n_boot=200, rng=seed)
        estimates.append(out["P"])
        sigmas.append(out["sigma"])
    # unbiased (to within the scatter of 40 repeats) and sigma describes the scatter
    assert np.mean(estimates) == pytest.approx(exact, abs=3 * np.std(estimates) / np.sqrt(40))
    assert np.mean(sigmas) == pytest.approx(np.std(estimates), rel=0.35)
    with pytest.raises(ValueError, match="inside the stratum"):
        estimate_stratified(weights, stratum, ~stratum, values)
    # nothing evaluated: no estimate
    assert np.isnan(estimate_stratified(weights, stratum, np.zeros(N, bool), values)["P"])


def test_posterior_estimate_stratified(posterior, draws):
    compactness, weights = draws
    stratum = compactness > 0.28
    chosen = np.flatnonzero(stratum)[::3]
    values = pd.Series((compactness[chosen] > 0.32).astype(float), index=chosen)
    out = posterior.estimate_stratified(lambda s: s["C"] > 0.28, values, n_boot=100, rng=2)
    evaluated = np.zeros(N, bool)
    evaluated[chosen] = True
    direct = estimate_stratified(np.nan_to_num(weights), stratum, evaluated,
                                 (compactness > 0.32).astype(float), n_boot=100, rng=2)
    assert out == pytest.approx(direct)
    assert out["n_evaluated"] == len(chosen)


def test_model_average():
    value, weights = model_average([0.1, 0.2, 0.4], np.log([1.0, 1.0, 2.0]))
    assert weights == pytest.approx([0.25, 0.25, 0.5])
    assert value == pytest.approx(0.25 * 0.1 + 0.25 * 0.2 + 0.5 * 0.4)


def test_average_over_posteriors(draws):
    compactness, weights = draws
    posteriors = []
    for scale in (1.0, 3.0):
        samples = pd.DataFrame({"eos": np.arange(N), "C": compactness, "weight_x": scale * np.nan_to_num(weights)})
        posteriors.append(EoSPosterior(samples, weight_columns=[WeightColumn("weight_x", is_log=False)]))
    value, evidence_weights = average_over_posteriors(posteriors, [0.0, 1.0])
    assert evidence_weights == pytest.approx([0.25, 0.75])
    assert value == pytest.approx(0.75)
    by_callable, _ = average_over_posteriors(posteriors, lambda post: post.probability(lambda s: s["C"] > 0.3)["P"])
    assert by_callable == pytest.approx(posteriors[0].probability(lambda s: s["C"] > 0.3)["P"])


def test_estimate_evidence_is_mean_weight(posterior, draws):
    _, weights = draws
    evidence, _ = posterior.estimate_evidence(posterior.weight_columns_available)
    assert evidence == pytest.approx(np.mean(np.nan_to_num(weights)))


def test_weighted_histogram(posterior, draws):
    compactness, weights = draws
    counts, edges = weighted_histogram(compactness, weights, bins=8, range=(0.1, 0.36))
    good = np.isfinite(weights)
    expected, _ = np.histogram(compactness[good], bins=8, range=(0.1, 0.36), weights=weights[good])
    np.testing.assert_allclose(counts, expected)
    via_posterior, _ = posterior.histogram("C", bins=8, range=(0.1, 0.36), density=True)
    assert np.sum(via_posterior * np.diff(edges)) == pytest.approx(1.0)


def test_matches_trapped_null_geos(draws):
    """Same numbers as trapgeo.probabilities (when available)."""
    trapgeo_probabilities = pytest.importorskip("trapgeo.probabilities")
    compactness, weights = draws
    for ours, theirs in [
        (weighted_probability(compactness > 0.3, weights), trapgeo_probabilities.weighted_probability(compactness > 0.3, weights)),
        (conditional_probability(compactness > 0.33, compactness > 0.3, np.nan_to_num(weights)),
         trapgeo_probabilities.conditional_probability(compactness > 0.33, compactness > 0.3, np.nan_to_num(weights))),
    ]:
        assert ours == pytest.approx(theirs)
    assert kish_neff(weights) == pytest.approx(trapgeo_probabilities.kish_neff(weights))
