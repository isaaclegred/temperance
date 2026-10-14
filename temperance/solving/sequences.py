"""
One-parameter sequences of relativistic stars.

solve_sequence           solve a sequence of stars (parameterised by central
                         log-enthalpy, pressure or density), decide radial
                         stability, and evaluate arbitrary per-star
                         properties on each model
refine_boundary          bisect for where a boolean star property flips
sequence_window_from_macro
                         central pressures near the top of the stable branch,
                         chosen from a precomputed macro table

These generalise get_tov_family (fixed outputs, parameterised by rho_c only)
and replace the per-project drivers of TrappedNullGeos (trapgeo/indicators.py).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ..sampling.stability import stable_mask
from ..utilities.units import CGS_DENSITY_TO_GEOM, KM_PER_MSUN
from .stellar_models import RelativisticStellarModel


def _central_values(lnh_c, p_c, rho_c):
    given = [(name, values) for name, values in (("lnh_c", lnh_c), ("p_c", p_c), ("rho_c", rho_c))
             if values is not None]
    if len(given) != 1:
        raise ValueError("give exactly one of lnh_c, p_c and rho_c")
    name, values = given[0]
    return name, np.atleast_1d(np.asarray(values, dtype=float))


def _make_model(eos, name, value, model_kwargs):
    if name == "lnh_c":
        return RelativisticStellarModel.from_logenthalpy(eos, value, **model_kwargs)
    if name == "p_c":
        return RelativisticStellarModel.from_central_pressure(eos, value, **model_kwargs)
    return RelativisticStellarModel(eos, value, **model_kwargs)


def _add(row, name, value):
    """Store a star property, expanding dicts (and objects with as_dict) into columns."""
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    if isinstance(value, dict):
        for key, item in value.items():
            row[f"{name}_{key}"] = item
    else:
        row[name] = value


def solve_sequence(eos, *, lnh_c=None, p_c=None, rho_c=None,
                   star_properties: dict[str, Callable[[RelativisticStellarModel], Any]] | None = None,
                   stability: str | Callable[[RelativisticStellarModel], bool] | None = "turning_point",
                   only_stable: bool = True, on_error: str = "record",
                   model_kwargs: dict | None = None) -> pd.DataFrame:
    """
    Solve a sequence of stars and evaluate properties of each.

    Parameters
    ----------
    eos : EOS object
    lnh_c, p_c, rho_c : array-like
        The central log-enthalpies, pressures or baryon densities (geometric
        units) of the stars; give exactly one.
    star_properties : dict name -> callable(model)
        Evaluated on each (stable, if only_stable) star.  A callable may
        return a scalar, a dict, or an object with ``as_dict()`` (e.g. a
        dataclass summary); dicts become columns ``name_key``.  Example:
        ``{"C": lambda m: m.C, "surface": lambda m: {"p": m.p[-1], "e": m.e[-1]}}``.
    stability : "turning_point", None or callable(model) -> bool
        "turning_point": the Bardeen-Thorne-Meltzer rule over the whole
        sequence (sampling.stability), so it should span the turning
        points that matter.  None: every star counts as stable.  A
        callable decides star by star.
    only_stable : bool
        Evaluate star_properties only on stable stars (others get NaN).
    on_error : "record" or "raise"
        "record" keeps the scan going when a star fails, writing the
        exception to an ``error`` column.
    model_kwargs : dict
        Passed to RelativisticStellarModel (e.g. n_points, lnh_grid).

    Returns
    -------
    pandas.DataFrame, one row per star in the order given, with columns
    lnh_c, p_c, e_c, rho_c, M, R (geometric), R_km, C, stable, error and
    the star properties.
    """
    if on_error not in ("record", "raise"):
        raise ValueError("on_error must be 'record' or 'raise'")
    if not (stability is None or stability == "turning_point" or callable(stability)):
        raise ValueError("stability must be 'turning_point', None or a callable")
    name, values = _central_values(lnh_c, p_c, rho_c)
    model_kwargs = dict(model_kwargs or {})
    star_properties = star_properties or {}

    rows, models = [], []
    for value in values:
        row = {name: float(value)}
        try:
            model = _make_model(eos, name, float(value), model_kwargs)
        except Exception as error:   # noqa: BLE001 - keep the scan going
            if on_error == "raise":
                raise
            row["error"] = repr(error)
            rows.append(row)
            models.append(None)
            continue
        row.update(lnh_c=model.lnh_c, p_c=model.central_pressure, e_c=model.central_energy_density,
                   rho_c=model.rho_c, M=model.M, R=model.R, R_km=model.R * KM_PER_MSUN, C=model.C)
        rows.append(row)
        models.append(model)
    stars = pd.DataFrame(rows)
    for column in ("lnh_c", "p_c", "e_c", "rho_c", "M", "R", "R_km", "C", "error"):
        if column not in stars:
            stars[column] = np.nan if column != "error" else None

    solved = np.array([model is not None for model in models])
    stable = np.zeros(len(stars), dtype=bool)
    if stability is None:
        stable[solved] = True
    elif stability == "turning_point":
        if solved.any():
            stable[solved] = stable_mask(stars.loc[solved, "M"], stars.loc[solved, "R"],
                                         central_variable=stars.loc[solved, "p_c"])
    else:
        stable = np.array([bool(stability(model)) if model is not None else False for model in models])
    stars["stable"] = stable

    for i, model in enumerate(models):
        if model is None or (only_stable and not stable[i]):
            continue
        properties = {}
        try:
            for key, function in star_properties.items():
                _add(properties, key, function(model))
        except Exception as error:   # noqa: BLE001
            if on_error == "raise":
                raise
            stars.loc[i, "error"] = repr(error)
            continue
        for key, item in properties.items():
            if key not in stars:
                stars[key] = pd.Series([np.nan] * len(stars), dtype=object)
            stars.at[i, key] = item
    return stars


def refine_boundary(eos, predicate: Callable[[RelativisticStellarModel], bool], lo: float, hi: float,
                    *, var: str = "lnh_c", n_bisect: int = 25, model_kwargs: dict | None = None):
    """
    Bisect in a central variable for where ``predicate(model)`` changes value
    between ``lo`` and ``hi`` (e.g. the onset of a stable null orbit, or of
    trapped modes).

    Parameters
    ----------
    var : "lnh_c", "p_c" or "rho_c"
        The central variable that lo and hi are values of.

    Returns
    -------
    (value, model): the central value at the boundary (within
    (hi - lo) / 2**n_bisect) and the star there.
    """
    if var not in ("lnh_c", "p_c", "rho_c"):
        raise ValueError("var must be 'lnh_c', 'p_c' or 'rho_c'")
    model_kwargs = dict(model_kwargs or {})
    value_lo = bool(predicate(_make_model(eos, var, lo, model_kwargs)))
    if bool(predicate(_make_model(eos, var, hi, model_kwargs))) == value_lo:
        raise ValueError("predicate has the same value at lo and hi")
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if bool(predicate(_make_model(eos, var, mid, model_kwargs))) == value_lo:
            lo = mid
        else:
            hi = mid
    value = 0.5 * (lo + hi)
    return value, _make_model(eos, var, value, model_kwargs)


def sequence_window_from_macro(macro: pd.DataFrame, stable=None, *, C_min: float | None = None,
                               M_min: float | None = None, densify: int = 2,
                               pressure_column: str = "central_pressurec2",
                               conversion_factor: float = CGS_DENSITY_TO_GEOM,
                               p_max: float | None = None) -> np.ndarray:
    """
    Central pressures (geometric units, for solve_sequence(p_c=...)) covering
    the stable stars of a precomputed macro table that pass the cuts, plus
    one neighbour on each side so the turning points are bracketed,
    densified ``densify``-fold in log pressure.

    Parameters
    ----------
    macro : DataFrame
        Needs ``pressure_column`` (cgs, as in the Zenodo Macro tables), M
        [Msun] and R [km].
    stable : array of bool, optional
        Stability of each row; by default the turning-point rule over the
        whole table (a window alone cannot see earlier turning points).
    C_min, M_min : float, optional
        Keep only stars with compactness M / R >= C_min and mass >= M_min.
    p_max : float, optional
        Drop pressures above this (geometric), e.g. the top of the EOS table.

    Returns
    -------
    Sorted array; empty if no star passes.
    """
    pressure = np.asarray(macro[pressure_column], dtype=float)
    keep = pressure > 0
    order = np.argsort(pressure[keep], kind="stable")
    pc = pressure[keep][order]
    M = np.asarray(macro["M"], dtype=float)[keep][order]
    R = np.asarray(macro["R"], dtype=float)[keep][order]
    pc, unique = np.unique(pc, return_index=True)
    M, R = M[unique], R[unique]
    if stable is None:
        stable = stable_mask(M, R, central_variable=pc)
    else:
        stable = np.asarray(stable, dtype=bool)[keep][order][unique]
    C = M / (R / KM_PER_MSUN)
    selected = stable.copy()
    if C_min is not None:
        selected &= C >= C_min
    if M_min is not None:
        selected &= M >= M_min
    if not selected.any():
        return np.array([])
    rows = np.flatnonzero(selected)
    lo, hi = max(rows[0] - 1, 0), min(rows[-1] + 1, len(pc) - 1)
    log_p = np.log(pc[lo:hi + 1])
    fine = [np.linspace(a, b, densify + 1)[:-1] for a, b in zip(log_p[:-1], log_p[1:])]
    window = np.exp(np.unique(np.concatenate(fine + [log_p[-1:]]))) * conversion_factor
    if p_max is not None:
        window = window[window <= p_max]
    return window
