"""
Radial stability of a one-parameter sequence of cold, non-rotating stars.

Bardeen-Thorne-Meltzer turning-point rule (rapid, i.e. equilibrium, phase
conversion): along a sequence ordered by central density (or pressure, or
log-enthalpy), radial stability changes only at extrema of M.  Starting
from zero unstable modes on the low-density branch, at each extremum one
radial mode changes stability.  In the M-R plane it is

    * lost if the curve bends counter-clockwise as the central density
      increases: a maximum of M with R decreasing, or a minimum with R
      increasing;
    * regained if it bends clockwise: a minimum of M with R decreasing
      (a twin-star branch), or a maximum with R increasing.

A star is stable if it has no unstable modes and dM/d(central density) >= 0.
This keeps genuine twin-star branches and rejects the high-density spiral,
whose rising segments start from a minimum with R increasing.

Without radii the rule cannot tell a twin branch from the spiral; every
rising segment is then counted as stable (the behaviour temperance had
before this rule was implemented).

AMBIGUITY: with *slow* conversion at a sharp phase interface, hybrid stars
past the mass maximum can be radially stable (Pereira, Flores & Lugones
2018); that needs an explicit radial-mode calculation, not this rule.
"""

import numpy as np


def stable_mask(mass, radius=None, central_variable=None, min_mass=None, mask=None):
    """
    Boolean mask of radially stable stars along a sequence.

    Parameters
    ----------
    mass : array-like
        Gravitational masses along the sequence.
    radius : array-like, optional
        Radii (any units).  Needed to apply the turning-point rule; without
        it every segment of non-decreasing mass is counted as stable.
    central_variable : array-like, optional
        Central density, pressure or log-enthalpy of each star.  If given the
        sequence is ordered by it (and dM is taken with respect to its log,
        if positive); otherwise the input order is taken to be the order of
        increasing central density, with uniform spacing.
    min_mass : float, optional
        Also exclude stars lighter than this (e.g. pathological low-mass,
        self-bound branches).
    mask : array-like of bool, optional
        Also exclude stars where this is False.

    Returns
    -------
    ndarray of bool, in the input order.
    """
    mass = np.asarray(mass, dtype=float)
    n = len(mass)
    order = (np.arange(n) if central_variable is None
             else np.argsort(np.asarray(central_variable, dtype=float), kind="stable"))
    M = mass[order]

    if n < 2:
        stable = np.ones(n, dtype=bool)
    elif radius is None:
        stable = _rising(M)
    else:
        R = np.asarray(radius, dtype=float)[order]
        if central_variable is None:
            x = np.arange(n, dtype=float)
        else:
            x = np.asarray(central_variable, dtype=float)[order]
            x = np.log(x) if np.all(x > 0) else x
        stable = _turning_point(M, R, x)

    result = np.empty(n, dtype=bool)
    result[order] = stable
    if min_mass is not None:
        result &= mass >= min_mass
    if mask is not None:
        result &= np.asarray(mask, dtype=bool)
    return result


def _rising(M):
    """Every run of non-decreasing mass (the first star follows the first step)."""
    rising_step = np.diff(M) >= 0
    return np.concatenate([[rising_step[0]], rising_step])


def _turning_point(M, R, x):
    """
    The turning-point count, discretised as in TrappedNullGeos
    (trapgeo.stability.stable_mask): dM/dx by central differences, an
    extremum where its sign changes between stars i-1 and i, and the
    direction of R from the step R[i] - R[i-1] leaving it.
    """
    dM = np.gradient(M, x)
    n_unstable = np.zeros(len(M), dtype=int)
    count = 0
    for i in range(1, len(M)):
        if np.sign(dM[i]) != np.sign(dM[i - 1]) and dM[i - 1] != 0:
            is_max = dM[i - 1] > 0
            curvature = -1.0 if is_max else 1.0
            # counter-clockwise (curvature * dR > 0) loses a mode
            count = max(count + (1 if curvature * (R[i] - R[i - 1]) > 0 else -1), 0)
        n_unstable[i] = count
    return (n_unstable == 0) & (dM > 0)


def stable_branches(mass, radius=None, central_variable=None, min_mass=None, mask=None):
    """
    Boolean masks (in the input order), one for each contiguous run of
    stable stars along the sequence; see stable_mask for the arguments.
    """
    mass = np.asarray(mass, dtype=float)
    n = len(mass)
    order = (np.arange(n) if central_variable is None
             else np.argsort(np.asarray(central_variable, dtype=float), kind="stable"))
    stable = stable_mask(mass, radius, central_variable, min_mass, mask)[order]
    starts = np.flatnonzero(stable & ~np.concatenate([[False], stable[:-1]]))
    stops = np.flatnonzero(stable & ~np.concatenate([stable[1:], [False]]))
    branches = []
    for start, stop in zip(starts, stops):
        branch = np.zeros(n, dtype=bool)
        branch[order[start:stop + 1]] = True
        branches.append(branch)
    return branches
