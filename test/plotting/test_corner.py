"""
Tests for temperance.plotting.corner, which overlays corner plots from the
`corner` package by default (universality's kde_corner is optional).
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from temperance.core.result import EoSPosterior, WeightColumn
from temperance.plotting import corner as tcorner
from temperance.plotting.corner import (PlottableColumn, PlottableEoSSamples,
                                        PlottableSamples)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def rng():
    return np.random.default_rng(11)


def make_samples(rng, label, shift=0.0, color="C0", n=2000, **kwargs):
    frame = pd.DataFrame({"M": rng.normal(1.4 + shift, 0.1, n),
                          "R": rng.normal(12.0 + shift, 0.5, n),
                          "logweight": rng.normal(0, 0.3, n)})
    return PlottableSamples(label, frame, [WeightColumn("logweight")], color=color, **kwargs)


COLUMNS = [PlottableColumn("M", "$M$", plot_range=(0.9, 2.0)),
           PlottableColumn("R", "$R$", plot_range=(9.0, 15.0))]


def diagonal_heights(ax):
    """The bar heights of the 1D histogram drawn on a diagonal axis."""
    vertices = ax.patches[-1].get_path().vertices
    return np.unique(np.round(vertices[:, 1], 10))


def test_corner_samples_overlay(rng):
    samples = [make_samples(rng, "first"), make_samples(rng, "second", shift=0.2, color="C1")]
    fig = tcorner.corner_samples(samples, columns_to_plot=COLUMNS)
    axes = np.array(fig.axes).reshape(2, 2)
    # shared ranges from plot_range
    assert axes[1, 0].get_xlim() == pytest.approx((0.9, 2.0))
    assert axes[1, 0].get_ylim() == pytest.approx((9.0, 15.0))
    # one 1D histogram per dataset on each diagonal axis
    assert len(axes[0, 0].patches) == 2 and len(axes[1, 1].patches) == 2
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["first", "second"]


def test_overlaid_histograms_all_fit(rng):
    """Later, broader histograms must not clip earlier, taller ones."""
    narrow = make_samples(rng, "narrow")
    narrow.samples["M"] = rng.normal(1.4, 0.02, len(narrow.samples))
    fig = tcorner.corner_samples([narrow, make_samples(rng, "broad", color="C1")],
                                 columns_to_plot=COLUMNS)
    ax = fig.axes[0]
    tallest = max(patch.get_path().vertices[:, 1].max() for patch in ax.patches)
    assert ax.get_ylim()[1] >= tallest


def test_filled_contours_leave_background_clear(rng):
    samples = make_samples(rng, "filled", alpha=0.5)
    samples.filled = True
    fig = tcorner.corner_samples([samples], columns_to_plot=COLUMNS, legend=False)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())
    ax = fig.axes[2]
    # a pixel just inside the lower-left corner of the 2D panel is far outside every contour
    x0, y0 = ax.transAxes.transform((0.03, 0.03))
    pixel = image[int(image.shape[0] - y0), int(x0), :3]
    assert np.all(pixel > 240), pixel


def test_corner_samples_histogram_uses_weights(rng):
    samples = make_samples(rng, "only")
    fig = tcorner.corner_samples([samples], columns_to_plot=COLUMNS, legend=False, bins=20)
    weights = np.exp(samples.samples["logweight"])
    expected, _ = np.histogram(samples.samples["M"], bins=20, range=(0.9, 2.0),
                               weights=weights, density=True)
    unweighted, _ = np.histogram(samples.samples["M"], bins=20, range=(0.9, 2.0), density=True)
    heights = diagonal_heights(fig.axes[0])
    assert heights.max() == pytest.approx(expected.max())
    assert heights.max() != pytest.approx(unweighted.max())


def test_missing_ranges_cover_weighted_data_only(rng):
    first = make_samples(rng, "first")
    # a cluster with zero weight far away should not stretch the axes
    far = first.samples.copy()
    far["M"] += 10
    far["logweight"] = -np.inf
    first.samples = pd.concat([first.samples, far], ignore_index=True)
    columns = [PlottableColumn("M", "$M$"), PlottableColumn("R", "$R$")]
    fig = tcorner.corner_samples([first, make_samples(rng, "second", shift=0.3)],
                                 columns_to_plot=columns)
    low, high = np.array(fig.axes).reshape(2, 2)[1, 0].get_xlim()
    assert low < 1.2 and 1.8 < high < 3.0


def test_nonfinite_rows_are_dropped(rng):
    samples = make_samples(rng, "with nans")
    samples.samples.loc[:10, "R"] = np.nan
    tcorner.corner_samples([samples], columns_to_plot=COLUMNS)


def test_truths_are_drawn(rng):
    columns = [PlottableColumn("M", "$M$", plot_range=(0.9, 2.0), true_value=1.4),
               PlottableColumn("R", "$R$", plot_range=(9.0, 15.0), true_value=12.0)]
    fig = tcorner.corner_samples([make_samples(rng, "s")], columns_to_plot=columns)
    vertical = [line.get_xdata()[0] for line in fig.axes[0].lines]
    assert 1.4 in vertical


def test_kwargs_override_defaults(rng):
    fig = tcorner.corner_samples([make_samples(rng, "s")], columns_to_plot=COLUMNS,
                                 legend=False, plot_datapoints=True)
    # datapoints are drawn as a line of markers on the 2D axis
    assert any(len(line.get_xdata()) > 100 for line in fig.axes[2].lines)


def test_universality_only_options_warn(rng):
    columns = [PlottableColumn("M", "$M$", plot_range=(0.9, 2.0), bandwidth=0.01),
               PlottableColumn("R", "$R$", plot_range=(9.0, 15.0))]
    with pytest.warns(UserWarning, match="bandwidth"):
        tcorner.corner_samples([make_samples(rng, "s")], columns_to_plot=columns)


@pytest.fixture
def eos_samples(rng):
    n = 500
    posterior = EoSPosterior(pd.DataFrame({"eos": np.arange(n),
                                           "logweight_astro": rng.normal(0, 0.5, n),
                                           "Mmax": rng.normal(2.2, 0.15, n)}))
    properties = pd.DataFrame({"eos": np.arange(n)[::-1],
                               "R(M=1.4)": rng.normal(12.0, 0.6, n)})
    return posterior, properties


def test_corner_eos_default_columns(eos_samples):
    posterior, properties = eos_samples
    plottable = PlottableEoSSamples("posterior", posterior, [WeightColumn("logweight_astro")],
                                    properties, color="C3")
    prior = plottable.get_prior(color="k")
    fig = tcorner.corner_eos([plottable, prior])
    axes = np.array(fig.axes).reshape(2, 2)
    assert axes[1, 0].get_xlabel() == "$M_{\\max}$"
    assert len(axes[0, 0].patches) == 2
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["posterior", "prior(posterior)"]


def test_corner_eos_weights_follow_eos(eos_samples):
    """Weights are matched to rows by eos, even when properties are reordered."""
    posterior, properties = eos_samples
    # all weight on EoSs with large R: the weighted R histogram must sit above 12
    heavy = properties.loc[properties["R(M=1.4)"] > 12.5, "eos"]
    posterior.samples["logweight_astro"] = np.where(posterior.samples["eos"].isin(heavy), 0.0, -np.inf)
    plottable = PlottableEoSSamples("posterior", posterior, [WeightColumn("logweight_astro")],
                                    properties, color="C3")
    columns = [PlottableColumn("Mmax", "Mmax", plot_range=(1.5, 3.0)),
               PlottableColumn("R(M=1.4)", "R", plot_range=(9.0, 15.0))]
    fig = tcorner.corner_eos([plottable], columns_to_plot=columns, legend=False, bins=30)
    vertices = fig.axes[3].patches[-1].get_path().vertices
    occupied = vertices[vertices[:, 1] > 0, 0]
    assert occupied.min() >= 12.5 - (6.0 / 30)


def test_use_universality_still_works(rng):
    pytest.importorskip("universality.plot")
    columns = [PlottableColumn("M", "$M$", plot_range=(0.9, 2.0), bandwidth=0.01),
               PlottableColumn("R", "$R$", plot_range=(9.0, 15.0), bandwidth=0.1)]
    fig = tcorner.corner_samples([make_samples(rng, "s", n=200)], use_universality=True,
                                 columns_to_plot=columns)
    assert len(fig.axes) == 3  # kde_corner draws only the lower triangle


def test_corner_eos_into_given_figure_quietly(eos_samples, capsys):
    """The call pattern used in Examples/EoSPopulation.ipynb."""
    posterior, properties = eos_samples
    plottable = PlottableEoSSamples("posterior", posterior, [WeightColumn("logweight_astro")],
                                    properties, color="b")
    columns = [PlottableColumn("Mmax", "Mmax", plot_range=(1.5, 3.0)),
               PlottableColumn("R(M=1.4)", "R", plot_range=(9.0, 15.0))]
    figure = plt.figure(figsize=(9, 9))
    fig = tcorner.corner_eos([plottable, plottable.get_prior(color="k")], columns_to_plot=columns,
                             levels=np.array([.9, .5]), smooth=1.0, fig=figure)
    assert fig is figure and len(fig.axes) == 4
    assert tuple(fig.get_size_inches()) == (9, 9)
    assert capsys.readouterr().out == ""
