"""
Neutron-star mass populations, truncated at the maximum mass of an EoS, and
the fraction of a population with some property.

A population is any one-dimensional ``bilby.core.prior.Prior``, e.g.

    from bilby.core.prior import PowerLaw, TruncatedGaussian, Uniform
    galactic_dns = TruncatedGaussian(1.33, 0.09, minimum=1.0, maximum=3.0)
    gw_like = Uniform(1.0, 3.0)
    pulsars = mixture([TruncatedGaussian(1.393, 0.064, 1.0, 3.0),
                       TruncatedGaussian(1.807, 0.177, 1.0, 3.0)], [0.65, 0.35])

The functions here truncate such a prior at an EoS's maximum mass using
only its own ``prob``, ``cdf`` and ``rescale``, so every bilby prior works:

    truncated_pdf(population, m, m_max=2.1)      # normalised on [minimum, 2.1]
    sample_truncated(population, 1000, m_max=2.1)
    population_fraction(M_stable, has_property, population)

The support must have a finite lower bound (``minimum``); use e.g.
TruncatedGaussian rather than Gaussian.  The parameters of target
populations belong in the analysis configuration, not here.

Moved from TrappedNullGeos (trapgeo/population.py).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid


def _bounds(population, m_max=None):
    lo = float(population.minimum)
    hi = float(population.maximum) if m_max is None else min(float(population.maximum), float(m_max))
    if not np.isfinite(lo):
        raise ValueError(f"{type(population).__name__} has no finite minimum; use a bounded prior "
                         "(e.g. TruncatedGaussian instead of Gaussian)")
    return lo, hi


def truncated_normalisation(population, m_max=None):
    """Probability mass of ``population`` on [minimum, min(maximum, m_max)]."""
    lo, hi = _bounds(population, m_max)
    if not hi > lo:
        return 0.0
    return float(population.cdf(hi) - population.cdf(lo))


def truncated_pdf(population, m, m_max=None):
    """
    Density of ``population`` truncated to masses <= m_max (and renormalised);
    0 outside [minimum, min(maximum, m_max)].
    """
    m = np.asarray(m, dtype=float)
    lo, hi = _bounds(population, m_max)
    norm = truncated_normalisation(population, m_max)
    if norm <= 0:
        return np.zeros_like(m)
    inside = (m >= lo) & (m <= hi)
    out = np.zeros_like(m)
    out[inside] = np.asarray(population.prob(m[inside]), dtype=float) / norm
    return out


def truncated_cdf(population, m, m_max=None):
    """Cumulative distribution of ``population`` truncated to masses <= m_max."""
    m = np.asarray(m, dtype=float)
    lo, hi = _bounds(population, m_max)
    norm = truncated_normalisation(population, m_max)
    if norm <= 0:
        return np.zeros_like(m)
    return np.clip((np.asarray(population.cdf(np.clip(m, lo, hi)), dtype=float)
                    - population.cdf(lo)) / norm, 0.0, 1.0)


def sample_truncated(population, n, m_max=None, rng=None):
    """n masses from ``population`` truncated to masses <= m_max (via its inverse CDF, rescale)."""
    lo, hi = _bounds(population, m_max)
    c_lo = float(population.cdf(lo))
    norm = truncated_normalisation(population, m_max)
    if norm <= 0:
        raise ValueError("the truncated population has no support")
    u = np.random.default_rng(rng).uniform(size=n)
    return np.clip(np.asarray(population.rescale(c_lo + u * norm), dtype=float), lo, hi)


def mixture(components, weights, minimum=None, maximum=None, n_grid=4000, name=None):
    """
    A weighted mixture of 1D bilby priors (e.g. a bimodal pair of
    TruncatedGaussians), as a ``bilby.core.prior.Interped`` prior on
    [minimum, maximum] (default: the components' combined support, which
    must be finite).  ``weights`` are the mixture fractions of the
    components, each normalised on its own support.
    """
    from bilby.core.prior import Interped
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(components):
        raise ValueError("need one weight per component")
    lo = min(c.minimum for c in components) if minimum is None else minimum
    hi = max(c.maximum for c in components) if maximum is None else maximum
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("a mixture needs a finite range; pass minimum and maximum")
    m = np.linspace(lo, hi, n_grid)
    density = sum(w * np.asarray(c.prob(m), dtype=float) for w, c in zip(weights / weights.sum(), components))
    return Interped(m, density, minimum=lo, maximum=hi, name=name)


_BRANCH_RULES = ("any", "all", "first")


def population_fraction(M_stable, indicator, population, branch=None, branch_rule="any", n_grid=4000):
    """
    Fraction of a mass population, truncated at the EoS's maximum stable
    mass, that has a property:

        f = int dM p(M | M <= M_max) Q(M),

    where Q(M) is whether a star of mass M has the property.

    Parameters
    ----------
    M_stable : array
        Masses of the stable stars of the EoS (e.g. along a sequence).
    indicator : array of bool
        Whether each of those stars has the property.
    population : bilby.core.prior.Prior
        One-dimensional, with a finite minimum; truncated at max(M_stable).
    branch : array of int, optional
        Stable-branch label of each star, increasing with central density
        (e.g. from sampling.stability.stable_branches).  Needed for twin
        stars, where one mass is realised on several branches.
    branch_rule : "any", "all" or "first"
        How a mass realised on several branches qualifies: on any branch,
        on all of them, or on the first (lowest-density) branch, i.e. if
        stars form there.  The population does not say how stars split
        between branches, so this is a modelling choice.
    n_grid : int
        Mass grid for the integral.  A branch realises the masses between
        its lightest and heaviest stars; there a grid mass takes the
        indicator of the nearest star (so pass the whole stable sequence,
        with False where the property was not evaluated).

    Returns nan if there are no stable stars.
    """
    if branch_rule not in _BRANCH_RULES:
        raise ValueError(f"branch_rule must be one of {_BRANCH_RULES}")
    M = np.asarray(M_stable, dtype=float)
    has = np.asarray(indicator, dtype=bool)
    if M.size == 0:
        return np.nan
    branch = np.zeros(M.size, dtype=int) if branch is None else np.asarray(branch)
    labels = np.unique(branch)
    M_max = float(M.max())
    lo, _ = _bounds(population)
    if M_max <= lo:
        return 0.0
    grid = np.linspace(lo, M_max, n_grid)
    covered = np.zeros((len(labels), n_grid), dtype=bool)
    qualifies = np.zeros((len(labels), n_grid), dtype=bool)
    for k, label in enumerate(labels):
        on_branch = branch == label
        Mb, hb = M[on_branch], has[on_branch]
        order = np.argsort(Mb)
        Mb, hb = Mb[order], hb[order]
        inside = (grid >= Mb[0]) & (grid <= Mb[-1])     # masses this branch realises
        nearest = np.clip(np.searchsorted(Mb, grid), 1, len(Mb) - 1) if len(Mb) > 1 else np.zeros(n_grid, int)
        if len(Mb) > 1:
            left = nearest - 1
            nearest = np.where(np.abs(grid - Mb[left]) <= np.abs(grid - Mb[nearest]), left, nearest)
        covered[k] = inside
        qualifies[k] = inside & hb[nearest]
    if branch_rule == "any":
        Q = qualifies.any(axis=0)
    elif branch_rule == "all":
        Q = covered.any(axis=0) & np.all(qualifies | ~covered, axis=0)
    else:
        first = np.argmax(covered, axis=0)
        Q = covered.any(axis=0) & qualifies[first, np.arange(n_grid)]
    p = truncated_pdf(population, grid, m_max=M_max)
    norm = trapezoid(p, grid)
    return float(trapezoid(p * Q, grid) / norm) if norm > 0 else np.nan
