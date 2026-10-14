"""
Tests for temperance.core.kde, including regression tests against the
universality functions they replace (skipped if universality is not installed).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from temperance.core import kde, result
from temperance.core.stats import SamplesColumn


@pytest.fixture
def rng():
    return np.random.default_rng(4321)


@pytest.fixture
def ukde():
    return pytest.importorskip("universality.kde")


@pytest.fixture
def ustats(monkeypatch):
    stats = pytest.importorskip("universality.stats")
    # universality still uses np.infty, which numpy 2 removed
    monkeypatch.setattr(np, "infty", np.inf, raising=False)
    return stats


def make_kde_function(rng, n=300):
    samples = pd.DataFrame({
        "M": rng.normal(1.4, 0.1, size=n),
        "R": rng.normal(12.0, 0.8, size=n),
        "logweight": rng.normal(size=n),
    })
    columns = [SamplesColumn("M", label="M", bandwidth=0.01),
               SamplesColumn("R", label="R", bandwidth=0.3)]
    return kde.KDEFunction(samples, [result.WeightColumn("logweight")], columns)


# ---------------------------------------------------------------------------
# Closed-form checks
# ---------------------------------------------------------------------------

def test_logkde_single_point_is_gaussian():
    variances = np.array([0.5, 2.0])
    points = np.array([[0.0, 0.0], [1.0, -1.0], [0.3, 2.0]])
    expected = multivariate_normal(mean=[0.2, -0.1], cov=np.diag(variances)).logpdf(points)
    np.testing.assert_allclose(kde.logkde(points, np.array([[0.2, -0.1]]), variances), expected)


def test_logkde_1d_input():
    out = kde.logkde(np.array([0.0, 1.0]), np.array([0.0]), np.array([1.0]))
    np.testing.assert_allclose(out, -0.5 * np.log(2 * np.pi) - 0.5 * np.array([0.0, 1.0]))


def test_logkde_stable_far_from_data():
    # naive exp() would underflow to log(0) = -inf here
    out = kde.logkde(np.array([[100.0]]), np.array([[0.0]]), np.array([0.01]))
    assert np.isfinite(out).all()


def test_silverman_bandwidth_unweighted():
    data = np.arange(10.0)
    assert kde.silverman_bandwidth(data) == pytest.approx(0.9 * np.std(data) * 10**-0.2)


def test_kde_function_argmax_near_mode(rng):
    function = make_kde_function(rng)
    vects = {"M": np.linspace(1.0, 1.8, 81), "R": np.linspace(9.0, 15.0, 61)}
    m, r = function.argmax(vects)
    assert m == pytest.approx(1.4, abs=0.1)
    assert r == pytest.approx(12.0, abs=0.8)


# ---------------------------------------------------------------------------
# Regression against universality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("weighted", [False, True])
def test_silverman_bandwidth_matches_universality(ukde, rng, weighted):
    data = rng.normal(size=1000)
    w = rng.exponential(size=1000) if weighted else None
    assert kde.silverman_bandwidth(data, weights=w) == pytest.approx(
        ukde.silverman_bandwidth(data, weights=w), rel=1e-12)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_logkde_matches_universality(ukde, rng, weighted, ndim):
    data = rng.normal(size=(400, ndim)).squeeze()
    samples = rng.normal(scale=2.0, size=(50, ndim)).squeeze()
    variances = rng.uniform(0.05, 0.5, size=ndim)
    w = rng.exponential(size=400) if weighted else None
    if w is not None:
        w /= w.sum()
    np.testing.assert_allclose(kde.logkde(samples, data, variances, weights=w),
                               ukde.logkde(samples, data, variances, weights=w, num_proc=1),
                               rtol=1e-12)


def test_vects2flatgrid_matches_universality(ukde):
    vects = [np.linspace(0, 1, 3), np.linspace(5, 6, 4)]
    np.testing.assert_array_equal(kde.vects2flatgrid(*vects), ukde.vects2flatgrid(*vects))


def test_kde_function_matches_universality(ukde, rng):
    function = make_kde_function(rng)
    points = pd.DataFrame({"M": rng.normal(1.4, 0.2, size=40), "R": rng.normal(12.0, 1.0, size=40)})
    expected = ukde.logkde(np.array(points), function.data, function.variances,
                           weights=function.weights, num_proc=1)
    np.testing.assert_allclose(function(points), expected, rtol=1e-12)


def test_kde_function_grid_methods_match_universality(ukde, ustats, rng):
    function = make_kde_function(rng)
    vects = {"M": np.linspace(1.0, 1.8, 41), "R": np.linspace(9.0, 15.0, 31)}
    grid = [vects["M"], vects["R"]]
    log_density = ukde.logkde(ukde.vects2flatgrid(*grid), function.data, function.variances,
                              weights=function.weights, num_proc=1)
    shaped = log_density.reshape(41, 31)
    np.testing.assert_allclose(function.argmax(vects), ustats.logkde2argmax(grid, shaped))
    assert function.entropy(vects) == pytest.approx(ustats.logkde2entropy(grid, log_density), rel=1e-12)
    assert function.information(vects) == pytest.approx(
        ustats.logkde2information(grid, log_density), rel=1e-12)


def test_legacy_kde_function_alias():
    from temperance.core.kde import KDEFunction
    from temperance.external.universality_density_estimate import kde_function
    assert kde_function is KDEFunction
