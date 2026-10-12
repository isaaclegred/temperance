"""
Statistics of weighted samples.

These reproduce the corresponding functions in universality.stats so that
temperance does not need universality installed.
"""
import numpy as np


def entropy(weights, base=2.):
    """
    The entropy of the discrete distribution defined by `weights`
    (which need not be normalized); zero weights contribute nothing.
    """
    weights = np.array(weights, dtype=float)
    weights /= np.sum(weights)
    nonzero = weights > 0
    return -np.sum(weights[nonzero] * np.log(weights[nonzero])) / np.log(base)


def neff(weights):
    """
    The effective number of samples for a set of weights, exp(entropy).
    Equal weights give len(weights).
    """
    return np.exp(entropy(weights, base=np.e))


def quantile(x, quantiles, weights=None):
    """
    Weighted quantiles of the samples `x`, linearly interpolating the
    empirical CDF.  With no weights this is np.percentile.
    """
    x = np.asarray(x)
    if weights is None:
        return np.percentile(x, np.array(quantiles) * 100)
    order = x.argsort()
    cdf = np.cumsum(np.asarray(weights, dtype=float)[order])
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, x[order])
