"""
Tests for temperance.core.weights, including regression tests against the
universality functions they replace (skipped if universality is not installed).
"""

import numpy as np
import pandas as pd
import pytest

from temperance.core import result, weights


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def universality_stats():
    return pytest.importorskip("universality.stats")


WEIGHT_CASES = {
    "uniform": lambda rng: np.ones(500),
    "exponential": lambda rng: rng.exponential(size=500),
    "with_zeros": lambda rng: np.where(rng.uniform(size=500) < 0.3, 0.0, rng.uniform(size=500)),
    "lognormal": lambda rng: np.exp(3 * rng.normal(size=500)),
}


# ---------------------------------------------------------------------------
# Closed-form checks
# ---------------------------------------------------------------------------

def test_neff_equal_weights_is_count():
    assert weights.neff(np.full(37, 0.2)) == pytest.approx(37)


def test_neff_ignores_zero_weights():
    assert weights.neff([1.0, 1.0, 0.0, 0.0]) == pytest.approx(2)


def test_neff_does_not_modify_input():
    w = np.array([1.0, 3.0])
    weights.neff(w)
    np.testing.assert_array_equal(w, [1.0, 3.0])


def test_quantile_unweighted_matches_percentile(rng):
    x = rng.normal(size=1000)
    q = np.linspace(0, 1, 11)
    np.testing.assert_allclose(weights.quantile(x, q), np.percentile(x, q * 100))


def test_quantile_weighted_median():
    # cdf: 0.1, 0.2, 1.0 -> median interpolates between x=2 and x=3
    assert weights.quantile(np.array([3.0, 1.0, 2.0]), 0.5, weights=np.array([8.0, 1.0, 1.0])) == pytest.approx(2.375)


# ---------------------------------------------------------------------------
# Regression against universality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", WEIGHT_CASES)
def test_neff_matches_universality(universality_stats, rng, case):
    w = WEIGHT_CASES[case](rng)
    assert weights.neff(w) == pytest.approx(universality_stats.neff(w.copy()), rel=1e-12)


@pytest.mark.parametrize("case", WEIGHT_CASES)
@pytest.mark.parametrize("base", [2.0, np.e])
def test_entropy_matches_universality(universality_stats, rng, case, base):
    w = WEIGHT_CASES[case](rng)
    assert weights.entropy(w, base=base) == pytest.approx(
        universality_stats.entropy(w.copy(), base=base), rel=1e-12)


@pytest.mark.parametrize("case", [None, *WEIGHT_CASES])
def test_quantile_matches_universality(universality_stats, rng, case):
    x = rng.normal(size=500)
    w = None if case is None else WEIGHT_CASES[case](rng)
    q = np.linspace(0, 1, 101)
    np.testing.assert_allclose(weights.quantile(x, q, weights=w),
                               universality_stats.quantile(x.copy(), q, weights=w),
                               rtol=1e-12)


def test_compute_neff_matches_universality(universality_stats, rng):
    logw = rng.normal(size=200)
    posterior = result.EoSPosterior(pd.DataFrame({"eos": np.arange(200), "logweight": logw}))
    column = result.WeightColumn("logweight")
    assert posterior.compute_neff([column]) == pytest.approx(
        universality_stats.neff(np.exp(logw)), rel=1e-12)
