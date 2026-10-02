"""Utilities for visualizing neutron stars."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle


# ---------------------------------------------------------------------------
# Quantity registry for neutron_star_profile
# ---------------------------------------------------------------------------
# Each entry: name -> (y-axis label, extractor(model) -> ndarray | None)
# Extractors that return None signal the quantity is absent for that model type
# and will be silently skipped.

def _dm_dr(m):
    """RHS of dM/dr = 4 pi r^2 e, using energy density (GR) or rho (Newtonian)."""
    e = getattr(m, "e", None) if getattr(m, "e", None) is not None else m.rho
    return 4.0 * np.pi * m.r**2 * e


_QUANTITY_REGISTRY = {
    "rho":    (r"$\rho$",              lambda m: m.rho),
    "p":      (r"$p$",                 lambda m: m.p),
    "cs2":    (r"$c_s^2 / c^2$",      lambda m: m.cs2),
    "cs":     (r"$c_s / c$",           lambda m: np.sqrt(np.abs(m.cs2))),
    "m":      (r"$m(r)$",              lambda m: m.m),
    "e":      (r"$e$",                 lambda m: getattr(m, "e", None)),
    "dm_dr":  (r"$4\pi r^2 e$",        _dm_dr),
    "Gamma1": (r"$\Gamma_1$",          lambda m: m.Gamma1),
    "phi":    (r"$\Phi$",              lambda m: getattr(m, "phi", None)),
    "nu":     (r"$\nu$",               lambda m: getattr(m, "nu", None)),
    "lam":    (r"$\lambda$",           lambda m: getattr(m, "lam", None)),
}

_DEFAULT_QUANTITIES = ["rho", "p", "cs", "dm_dr"]


def neutron_star_cross_section(
    eos_or_model,
    level_sets: Sequence[float],
    *,
    rho_c: Optional[float] = None,
    quantity: str = "rho",
    colors=None,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "magma_r",
    normalize_radius: bool = True,
    show_boundaries: bool = True,
    boundary_linewidth: float = 0.8,
    boundary_color: str = "k",
    figsize: tuple = (5, 5),
    show_mass_fractions: bool = False,
) -> plt.Axes:
    """
    Draw a circular cross-section of a neutron star with regions filled
    by user-specified density or pressure level sets.

    Each pair of adjacent level-set values defines an annular shell.  Regions
    are painted from the outermost shell inward, so each inner circle
    overpaints the previous one, producing a clean onion-layer appearance.

    Parameters
    ----------
    eos_or_model : EoS object or stellar model
        Either a solved ``NewtonianStellarModel`` / ``RelativisticStellarModel``
        or a raw EoS object (``polytropic_eos``, ``css_eos``,
        ``interpolated_eos``, or a pandas DataFrame).  When passing an EoS,
        ``rho_c`` must also be supplied; a ``RelativisticStellarModel`` will be
        built internally.
    level_sets : sequence of float
        Values of ``quantity`` at which region boundaries are drawn.  Will be
        sorted ascending internally.  Values outside the stellar profile range
        are silently clipped to the boundary.
    rho_c : float, optional
        Central density (geometric units), required when ``eos_or_model`` is an
        EoS rather than a pre-built stellar model.
    quantity : {"rho", "p"}
        Which profile quantity the level sets refer to.  Both are
        monotonically decreasing with radius for a normal neutron star.
    colors : sequence, optional
        One color per region (``len(level_sets) + 1`` total, ordered from
        surface inward).  If omitted, sampled uniformly from ``cmap``.
    labels : sequence of str, optional
        Legend labels for each region, same length as ``colors``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Created fresh if not provided.
    cmap : str
        Matplotlib colormap name used when ``colors`` is not specified.
    normalize_radius : bool
        If True (default), the x/y axes are in units of R (stellar radius)
        so the star always fills a unit circle.  If False, physical units of
        the stellar model are used.
    show_boundaries : bool
        If True, draw a thin circle at each level-set radius to make the
        shell boundaries visible.
    boundary_linewidth : float
        Line width of the boundary circles.
    boundary_color : str
        Color of the boundary circles.
    figsize : tuple
        Figure size passed to ``plt.subplots`` when ``ax`` is None.
    show_mass_fractions : bool
        If True, print the enclosed mass fraction of each shell to stdout
        and annotate the fraction inside each shell on the plot.  The mass
        at each boundary is read from ``model.m`` (enclosed gravitational
        mass).  Shells are reported from surface inward.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the cross-section.

    Examples
    --------
    >>> from temperance.solving.analytic_eos import polytropic_eos
    >>> from temperance.solving.stellar_models import RelativisticStellarModel
    >>> eos = polytropic_eos(K=0.0195, Gamma=2.0)
    >>> model = RelativisticStellarModel(eos, rho_c=5e-4)
    >>> ax = neutron_star_cross_section(
    ...     model,
    ...     level_sets=[1e-4, 3e-4],   # two density boundaries
    ...     quantity="rho",
    ...     labels=["crust", "outer core", "inner core"],
    ... )
    >>> plt.show()
    """
    # ------------------------------------------------------------------
    # Resolve model
    # ------------------------------------------------------------------
    try:
        from ..solving.stellar_models import NewtonianStellarModel, RelativisticStellarModel
    except ImportError:
        from solving.stellar_models import NewtonianStellarModel, RelativisticStellarModel

    if isinstance(eos_or_model, (NewtonianStellarModel, RelativisticStellarModel)):
        model = eos_or_model
    else:
        if rho_c is None:
            raise ValueError(
                "rho_c must be provided when eos_or_model is an EoS, not a stellar model."
            )
        model = RelativisticStellarModel(eos_or_model, rho_c)

    # ------------------------------------------------------------------
    # Extract radial profile
    # ------------------------------------------------------------------
    r_profile = model.r
    R = model.R
    q_profile = getattr(model, quantity, None)
    if q_profile is None:
        raise AttributeError(
            f"Stellar model has no attribute '{quantity}'. "
            "Use 'rho' or 'p'."
        )

    # ------------------------------------------------------------------
    # Invert q(r) -> r(q)
    # Both rho and p decrease monotonically from centre (index 0) to surface.
    # Flip so the interpolation grid is ascending.
    # ------------------------------------------------------------------
    q_asc = q_profile[::-1]   # ascending (surface ~ 0 to centre ~ q_c)
    r_desc = r_profile[::-1]  # corresponding r, descending

    level_sets_arr = np.sort(np.asarray(level_sets, dtype=float))

    q_min, q_max = q_asc[0], q_asc[-1]
    out_of_range = (level_sets_arr < q_min) | (level_sets_arr > q_max)
    if np.any(out_of_range):
        warnings.warn(
            f"Some level_sets lie outside the profile range "
            f"[{q_min:.3g}, {q_max:.3g}] for '{quantity}' and will be clipped.",
            stacklevel=2,
        )
    level_sets_arr = np.clip(level_sets_arr, q_min, q_max)

    # r at each level-set boundary (descending: outermost first)
    r_levels = np.interp(level_sets_arr, q_asc, r_desc)

    # ------------------------------------------------------------------
    # Mass fractions (computed here regardless; used below if requested)
    # model.m is ascending (centre->surface); r_levels is descending.
    # Shell boundaries in descending order: [R, r_levels[0], ..., r_levels[-1], 0]
    # ------------------------------------------------------------------
    M_total = model.M
    m_at_levels = np.interp(r_levels, model.r, model.m)  # m(r) at each cut
    mass_shell = np.diff(                                  # mass in each shell
        np.concatenate([[M_total], m_at_levels, [0.0]]),
        ) * -1                                             # boundaries are descending
    mass_fractions = mass_shell / M_total

    # ------------------------------------------------------------------
    # Scale for plotting
    # ------------------------------------------------------------------
    scale = R if normalize_radius else 1.0
    R_plot = R / scale
    r_levels_plot = r_levels / scale

    # ------------------------------------------------------------------
    # Colors: one per region, ordered surface -> core
    # ------------------------------------------------------------------
    n_regions = len(level_sets_arr) + 1
    if colors is None:
        cm = plt.get_cmap(cmap)
        colors = [cm(i / max(n_regions - 1, 1)) for i in range(n_regions)]
    if len(colors) != n_regions:
        raise ValueError(
            f"len(colors)={len(colors)} does not match the number of regions "
            f"({n_regions} = len(level_sets) + 1)."
        )
    if labels is not None and len(labels) != n_regions:
        raise ValueError(
            f"len(labels)={len(labels)} does not match the number of regions "
            f"({n_regions})."
        )

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Paint from outermost to innermost.  Each circle overwrites the interior
    # of the previous one, leaving the annular region between them visible.
    draw_radii = np.concatenate([[R_plot], r_levels_plot])  # descending
    for radius, color, i in zip(draw_radii, colors, range(n_regions)):
        lbl = labels[i] if labels is not None else None
        ax.add_patch(
            Circle(
                (0.0, 0.0), radius,
                facecolor=color,
                edgecolor="none",
                label=lbl,
                zorder=i,
            )
        )

    # Boundary circles (optional thin outlines at each level-set radius)
    if show_boundaries:
        for rb in r_levels_plot:
            ax.add_patch(
                Circle(
                    (0.0, 0.0), rb,
                    fill=False,
                    edgecolor=boundary_color,
                    linewidth=boundary_linewidth,
                    zorder=n_regions,
                )
            )

    # Outer stellar boundary
    ax.add_patch(
        Circle(
            (0.0, 0.0), R_plot,
            fill=False,
            edgecolor=boundary_color,
            linewidth=boundary_linewidth * 1.5,
            zorder=n_regions,
        )
    )

    # ------------------------------------------------------------------
    # Mass fractions: print + annotate
    # ------------------------------------------------------------------
    if show_mass_fractions:
        # Stdout table
        shell_names = (labels if labels is not None
                       else [f"Shell {i}" for i in range(n_regions)])
        print(f"{'Shell':<18}  {'Mass fraction':>14}")
        print("-" * 34)
        for name, frac in zip(shell_names, mass_fractions):
            print(f"{str(name):<18}  {frac:>13.4f}")
        print("-" * 34)
        print(f"{'Total':<18}  {mass_fractions.sum():>13.4f}")

        # Annotation: place fraction text at the midpoint radius of each shell
        shell_radii = np.concatenate([[R_plot], r_levels_plot, [0.0]])
        text_zorder = n_regions + 1
        for i, frac in enumerate(mass_fractions):
            r_mid = (shell_radii[i] + shell_radii[i + 1]) / 2.0
            ax.text(
                0.0, r_mid,
                f"{frac:.1%}",
                ha="center", va="center",
                fontsize=7,
                color=boundary_color,
                zorder=text_zorder,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
            )

    # Legend via proxy patches so the overlapping circles don't confuse it
    if labels is not None:
        proxies = [
            mpatches.Patch(facecolor=colors[i], edgecolor=boundary_color,
                           linewidth=0.5, label=labels[i])
            for i in range(n_regions)
        ]
        ax.legend(handles=proxies, loc="upper right", framealpha=0.9)

    pad = 0.12 * R_plot
    ax.set_xlim(-R_plot - pad, R_plot + pad)
    ax.set_ylim(-R_plot - pad, R_plot + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("$r/R$" if normalize_radius else "$r$")
    ax.set_ylabel("$r/R$" if normalize_radius else "$r$")

    return ax


def neutron_star_profile(
    models: Union[object, Sequence],
    quantities: Optional[Sequence[str]] = None,
    *,
    normalize_radius: bool = True,
    axes: Optional[np.ndarray] = None,
    figsize: Optional[tuple] = None,
    model_labels: Optional[Sequence[str]] = None,
    colors=None,
    linestyles=None,
    ncols: int = 2,
    **plot_kwargs,
) -> tuple:
    """
    Plot radial profiles of neutron star interior quantities.

    Supports any combination of the quantities listed below and accepts either
    a single stellar model or a list of models (overlaid on the same axes).

    Built-in quantities
    -------------------
    ``"rho"``    baryon (mass) density
    ``"p"``      pressure
    ``"cs"``     adiabatic sound speed  sqrt(cs2)
    ``"cs2"``    adiabatic sound speed squared
    ``"m"``      enclosed gravitational mass
    ``"e"``      total energy density  (GR models only; skipped for Newtonian)
    ``"dm_dr"``  4 pi r^2 * e  (RHS of TOV mass equation; falls back to rho
                 for Newtonian models where e is not stored separately)
    ``"Gamma1"`` adiabatic index
    ``"phi"``    Newtonian gravitational potential  (Newtonian only)
    ``"nu"``     GR metric potential nu  (GR only)
    ``"lam"``    GR metric potential lambda  (GR only)

    Parameters
    ----------
    models : stellar model or list of stellar models
        ``NewtonianStellarModel`` or ``RelativisticStellarModel`` instances.
        A single model is wrapped in a list automatically.
    quantities : list of str, optional
        Quantities to plot, each on its own panel.  Defaults to
        ``["rho", "p", "cs", "dm_dr"]``.  Any quantity not available for a
        given model (e.g. ``"e"`` for Newtonian) is silently skipped for that
        model.
    normalize_radius : bool
        If True (default), the horizontal axis is ``r/R``; otherwise the raw
        radius in the model's native units is used.
    axes : array-like of Axes, optional
        Pre-existing axes to draw on, one per quantity.  Must have at least
        ``len(quantities)`` elements.  If omitted a new figure is created.
    figsize : tuple, optional
        Passed to ``plt.subplots`` when creating a new figure.  Defaults to
        ``(ncols * 4, nrows * 3)``.
    model_labels : list of str, optional
        Legend labels, one per model.  If omitted and only one model is
        given, no legend is drawn.
    colors : list, optional
        Line colors, one per model.  Defaults to the matplotlib prop cycle.
    linestyles : list, optional
        Line styles, one per model.  Defaults to solid lines.
    ncols : int
        Number of subplot columns when creating a new figure.
    **plot_kwargs
        Passed directly to ``ax.plot``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes
        Shape ``(n_quantities,)``.

    Examples
    --------
    >>> from temperance.solving.analytic_eos import polytropic_eos
    >>> from temperance.solving.stellar_models import RelativisticStellarModel
    >>> eos = polytropic_eos(K=0.0195, Gamma=2.0)
    >>> models = [RelativisticStellarModel(eos, rho_c=rc) for rc in [3e-4, 5e-4, 8e-4]]
    >>> fig, axes = neutron_star_profile(
    ...     models,
    ...     quantities=["rho", "p", "cs", "dm_dr"],
    ...     model_labels=[r"$3\\rho_0$", r"$5\\rho_0$", r"$8\\rho_0$"],
    ... )
    >>> plt.show()
    """
    if not isinstance(models, (list, tuple)):
        models = [models]

    if quantities is None:
        quantities = list(_DEFAULT_QUANTITIES)

    n_q = len(quantities)
    n_m = len(models)

    # ------------------------------------------------------------------
    # Colors and linestyles
    # ------------------------------------------------------------------
    if colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color",
                    [f"C{i}" for i in range(10)])
        colors = [cycle[i % len(cycle)] for i in range(n_m)]
    if linestyles is None:
        linestyles = ["-"] * n_m

    # ------------------------------------------------------------------
    # Axes layout
    # ------------------------------------------------------------------
    if axes is not None:
        axes_flat = np.asarray(axes).flatten()
        fig = axes_flat[0].get_figure()
    else:
        _ncols = min(ncols, n_q)
        _nrows = (n_q + _ncols - 1) // _ncols
        if figsize is None:
            figsize = (_ncols * 4, _nrows * 3)
        fig, axes_arr = plt.subplots(_nrows, _ncols, figsize=figsize, squeeze=False)
        axes_flat = axes_arr.flatten()
        for j in range(n_q, len(axes_flat)):
            axes_flat[j].set_visible(False)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    for j, qty in enumerate(quantities):
        if qty not in _QUANTITY_REGISTRY:
            warnings.warn(f"Unknown quantity '{qty}'; skipping.", stacklevel=2)
            continue

        ylabel, extract = _QUANTITY_REGISTRY[qty]
        ax_j = axes_flat[j]

        for k, model in enumerate(models):
            values = extract(model)
            if values is None:
                continue  # quantity not present for this model type

            r = model.r
            R = model.R
            x = r / R if normalize_radius else r
            lbl = (model_labels[k] if model_labels is not None
                   else (f"Model {k}" if n_m > 1 else None))
            ax_j.plot(x, values, color=colors[k], linestyle=linestyles[k],
                      label=lbl, **plot_kwargs)

        ax_j.set_ylabel(ylabel)
        ax_j.set_xlabel(r"$r/R$" if normalize_radius else r"$r$")
        ax_j.set_xlim(left=0)
        ax_j.set_title(qty, fontsize=9)

        if n_m > 1 and model_labels is not None:
            ax_j.legend(fontsize=8)

    fig.tight_layout()
    return fig, axes_flat[:n_q]
