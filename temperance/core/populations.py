"""
One-dimensional neutron-star mass populations, truncatable at the maximum
mass of an EoS, and the fraction of a population with some property.

    population = Mixture([Gaussian(1.393, 0.064), Gaussian(1.807, 0.177)], [0.65, 0.35],
                         m_min=1.0)
    population.pdf(m, m_max=2.1)          # normalised on [m_min, 2.1]
    population.sample(1000, m_max=2.1)
    population_fraction(M_stable, has_property, population)

Every population has a lower bound m_min and an upper bound m_max (default
infinity); passing m_max=M_max(EoS) to pdf / sample / mass_fraction further
truncates it at that EoS's maximum mass.  The parameters of target
populations belong in the analysis configuration, not here.

Moved from TrappedNullGeos (trapgeo/population.py).
"""

from __future__ import annotations

import numpy as np
import scipy.special


class MassPopulation:
    """
    Base class: subclasses define the untruncated density ``_density`` and
    its cumulative ``_cdf`` (any normalisation); truncation to
    [m_min, min(m_max, M_max)] and normalisation are handled here.
    """

    m_min: float = 0.0
    m_max: float = np.inf

    def _density(self, m):
        raise NotImplementedError

    def _cdf(self, m):
        raise NotImplementedError

    def _bounds(self, m_max=None):
        hi = self.m_max if m_max is None else min(self.m_max, m_max)
        return self.m_min, hi

    def normalisation(self, m_max=None):
        """Integral of the untruncated density over the (truncated) support."""
        lo, hi = self._bounds(m_max)
        return float(self._cdf(hi) - self._cdf(lo)) if hi > lo else 0.0

    def pdf(self, m, m_max=None):
        """Probability density, normalised on [m_min, min(self.m_max, m_max)]; 0 outside."""
        m = np.asarray(m, dtype=float)
        lo, hi = self._bounds(m_max)
        norm = self.normalisation(m_max)
        if norm <= 0:
            return np.zeros_like(m)
        return np.where((m >= lo) & (m <= hi), self._density(m) / norm, 0.0)

    def cdf(self, m, m_max=None):
        """Cumulative distribution of the truncated population."""
        m = np.asarray(m, dtype=float)
        lo, hi = self._bounds(m_max)
        norm = self.normalisation(m_max)
        if norm <= 0:
            return np.zeros_like(m)
        return np.clip((self._cdf(np.clip(m, lo, hi)) - self._cdf(lo)) / norm, 0.0, 1.0)

    def mass_fraction(self, lo, hi, m_max=None):
        """Probability that a mass of the (truncated) population lies in [lo, hi]."""
        return float(self.cdf(hi, m_max) - self.cdf(lo, m_max))

    def sample(self, n, m_max=None, rng=None):
        """n masses from the truncated population (inverse CDF on a fine grid)."""
        lo, hi = self._bounds(m_max)
        if not np.isfinite(hi):
            hi = self._finite_upper()
        grid = np.linspace(lo, hi, 4097)
        cdf = self.cdf(grid, m_max)
        cdf, unique = np.unique(cdf, return_index=True)
        rng = np.random.default_rng(rng)
        return np.interp(rng.uniform(size=n), cdf, grid[unique])

    def _finite_upper(self):
        raise ValueError("this population needs a finite m_max to be sampled")


class Uniform(MassPopulation):
    """Flat in mass on [m_min, m_max] (the usual gravitational-wave population assumption)."""

    def __init__(self, m_min=1.0, m_max=np.inf):
        self.m_min, self.m_max = float(m_min), float(m_max)

    def _density(self, m):
        return np.ones_like(np.asarray(m, dtype=float))

    def _cdf(self, m):
        return np.asarray(m, dtype=float)

    def _finite_upper(self):
        raise ValueError("a uniform population needs a finite upper bound (m_max) to be sampled")


class Gaussian(MassPopulation):
    """Normal(mu, sigma), restricted to [m_min, m_max]."""

    def __init__(self, mu, sigma, m_min=0.0, m_max=np.inf):
        self.mu, self.sigma = float(mu), float(sigma)
        self.m_min, self.m_max = float(m_min), float(m_max)

    def _density(self, m):
        z = (np.asarray(m, dtype=float) - self.mu) / self.sigma
        return np.exp(-0.5 * z**2) / (self.sigma * np.sqrt(2 * np.pi))

    def _cdf(self, m):
        return 0.5 * (1 + scipy.special.erf((np.asarray(m, dtype=float) - self.mu) / (self.sigma * np.sqrt(2))))

    def _finite_upper(self):
        return self.mu + 10 * self.sigma


class PowerLaw(MassPopulation):
    """Density proportional to m^alpha on [m_min, m_max] (m_min > 0)."""

    def __init__(self, alpha, m_min=1.0, m_max=np.inf):
        if m_min <= 0:
            raise ValueError("a power law needs m_min > 0")
        if not np.isfinite(m_max) and alpha >= -1:
            raise ValueError("a power law with alpha >= -1 needs a finite m_max")
        self.alpha, self.m_min, self.m_max = float(alpha), float(m_min), float(m_max)

    def _density(self, m):
        return np.asarray(m, dtype=float) ** self.alpha

    def _cdf(self, m):
        m = np.asarray(m, dtype=float)
        if self.alpha == -1:
            return np.log(m)
        return m ** (self.alpha + 1) / (self.alpha + 1)

    def _finite_upper(self):
        return self.m_min * 1e3


class Mixture(MassPopulation):
    """
    A weighted mixture of populations, e.g. a bimodal pair of Gaussians.
    Each component keeps its own normalisation, so ``weights`` are the
    mixture fractions before any truncation; the overall support is
    [m_min, m_max] intersected with the components'.
    """

    def __init__(self, components, weights, m_min=None, m_max=None):
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(components):
            raise ValueError("need one weight per component")
        self.components = list(components)
        self.weights = weights / weights.sum()
        self.m_min = float(min(c.m_min for c in components) if m_min is None else m_min)
        self.m_max = float(max(c.m_max for c in components) if m_max is None else m_max)

    def _component_terms(self, m, function):
        m = np.asarray(m, dtype=float)
        total = np.zeros_like(m)
        for weight, component in zip(self.weights, self.components):
            norm = component.normalisation()
            if norm > 0:
                total = total + weight * function(component, m) / norm
        return total

    def _density(self, m):
        return self._component_terms(m, lambda c, x: np.where((x >= c.m_min) & (x <= c.m_max),
                                                              c._density(x), 0.0))

    def _cdf(self, m):
        return self._component_terms(m, lambda c, x: c._cdf(np.clip(x, c.m_min, c.m_max)) - c._cdf(c.m_min))

    def _finite_upper(self):
        return max(c._finite_upper() if not np.isfinite(c.m_max) else c.m_max for c in self.components)


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
    population : MassPopulation
        Truncated at max(M_stable).
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
    lo = population.m_min
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
    p = population.pdf(grid, m_max=M_max)
    norm = np.trapezoid(p, grid)
    return float(np.trapezoid(p * Q, grid) / norm) if norm > 0 else np.nan
