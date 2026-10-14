"""
Statistics of weighted samples.

entropy, neff and quantile reproduce the corresponding functions in
universality.stats so that temperance does not need universality installed.

Two effective sample sizes are in use:

    neff(w)       = exp(entropy of w / sum w)   (universality's definition)
    kish_neff(w)  = (sum w)^2 / sum w^2         (Kish)

Both equal N for N equal weights.  The Kish number is the one that sets
the variance of a self-normalised importance-sampling estimate, so it is
used for Monte Carlo uncertainties (probability, the zero-count upper
bound); the entropy number is what universality reports.

The remaining functions estimate posterior probabilities from prior draws
with likelihood weights, P = sum_i w_i 1_i / sum_i w_i.  An importance
sampling estimate can only see properties that occur in the *prior* draws:
if none of N draws has a property, the bound reflects N, not the data.
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
    Equal weights give len(weights).  See kish_neff for the variance-based
    alternative.
    """
    return np.exp(entropy(weights, base=np.e))


def kish_neff(weights):
    """
    Kish's effective sample size (sum w)^2 / sum w^2, ignoring non-finite
    weights.  Equal weights give len(weights).
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights)]
    total_squared = np.sum(weights**2)
    return float(np.sum(weights) ** 2 / total_squared) if total_squared > 0 else 0.0


def weighted_probability(indicator, weights, cl=0.9):
    """
    Posterior probability of a boolean property from weighted draws.

    Returns a dict with
        P             sum w 1 / sum w
        sigma         delta-method standard error of that ratio
        n_eff         Kish effective sample size of the weights
        n_true        number of draws with the property
        upper_if_zero one-sided ``cl`` upper bound on P when n_true == 0,
                      1 - (1 - cl)^(1 / n_eff) (the "rule of three" with
                      n_eff in place of N)
    Draws with a non-finite indicator or weight are ignored.
    """
    x = np.asarray(indicator, dtype=float)
    w = np.asarray(weights, dtype=float)
    good = np.isfinite(x) & np.isfinite(w)
    x, w = x[good], w[good]
    total = np.sum(w)
    if total <= 0:
        return {"P": np.nan, "sigma": np.nan, "n_eff": 0.0, "n_true": int(np.sum(x)), "upper_if_zero": np.nan}
    P = float(np.sum(w * x) / total)
    sigma = float(np.sqrt(np.sum(w**2 * (x - P) ** 2)) / total)
    n_eff = kish_neff(w)
    upper = 1.0 - (1.0 - cl) ** (1.0 / n_eff) if n_eff > 0 else 1.0
    return {"P": P, "sigma": sigma, "n_eff": n_eff, "n_true": int(np.sum(x)), "upper_if_zero": upper}


def conditional_probability(a, b, weights, cl=0.9):
    """P(a | b) = P(a and b) / P(b), as weighted_probability over the draws with b."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    w = np.asarray(weights, dtype=float)
    if not np.any(b & (w > 0)):
        return {"P": np.nan, "sigma": np.nan, "n_eff": 0.0, "n_true": 0, "upper_if_zero": np.nan}
    return weighted_probability(a[b], w[b], cl=cl)


def estimate_stratified(weights, stratum, evaluated, values, n_boot=2000, rng=None, outside_value=0.0):
    """
    Posterior mean of a per-draw quantity that is expensive to compute, from
    a random subsample of a stratum.

    A cheap screen defines the stratum S (``stratum``) inside all draws A;
    only a random subset s of S (``evaluated``) is computed; outside S the
    quantity is taken to be ``outside_value`` (e.g. 0 for an indicator the
    screen rules out).  The ratio estimator

        E = [W_S * (sum_s w v / sum_s w) + (W_A - W_S) * outside_value] / W_A

    (W = sum of weights) is returned with its bootstrap standard error over
    the evaluated draws.

    Parameters
    ----------
    weights : (N,) weights of all draws
    stratum, evaluated : (N,) bool, evaluated must lie inside stratum
    values : (N,) the quantity (used where evaluated)

    Returns dict(P, sigma, stratum_mass = W_S / W_A, n_evaluated).
    """
    w = np.asarray(weights, dtype=float)
    stratum = np.asarray(stratum, dtype=bool)
    evaluated = np.asarray(evaluated, dtype=bool)
    if np.any(evaluated & ~stratum):
        raise ValueError("evaluated draws must lie inside the stratum")
    w = np.where(np.isfinite(w), w, 0.0)
    W_A, W_S = np.sum(w), np.sum(w[stratum])
    v = np.asarray(values, dtype=float)[evaluated]
    w_s = w[evaluated]
    if W_A <= 0 or len(v) == 0 or np.sum(w_s) <= 0:
        return {"P": np.nan, "sigma": np.nan, "stratum_mass": W_S / W_A if W_A > 0 else np.nan,
                "n_evaluated": int(len(v))}

    def estimate(values_s, weights_s):
        return (W_S * np.sum(weights_s * values_s) / np.sum(weights_s) + (W_A - W_S) * outside_value) / W_A

    rng = np.random.default_rng(rng)
    n = len(v)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        if np.sum(w_s[pick]) > 0:
            boots.append(estimate(v[pick], w_s[pick]))
    return {"P": float(estimate(v, w_s)), "sigma": float(np.std(boots)) if boots else np.nan,
            "stratum_mass": float(W_S / W_A), "n_evaluated": int(n)}


def model_average(values, log_evidences):
    """
    Evidence-weighted average over discrete models (e.g. EoS priors with
    different hyperparameters): sum_k Z_k v_k / sum_k Z_k.  Returns
    (average, normalised weights).  Whether this is legitimate depends on
    whether the models are treated as draws of a hyperparameter with a
    prior, or as a sensitivity axis.
    """
    log_z = np.asarray(log_evidences, dtype=float)
    z = np.exp(log_z - np.max(log_z))
    z /= np.sum(z)
    return float(np.sum(z * np.asarray(values, dtype=float))), z


def weighted_histogram(values, weights=None, bins=10, range=None, density=False):
    """
    np.histogram of a per-draw scalar with weights, ignoring draws whose
    value or weight is not finite.  Returns (counts, bin_edges).
    """
    values = np.asarray(values, dtype=float)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights)
    return np.histogram(values[good], bins=bins, range=range, weights=weights[good], density=density)


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
