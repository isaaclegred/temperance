"""
Gaussian kernel density estimates with per-dimension (diagonal) variances.

These reproduce the corresponding functions in universality.kde and
universality.stats so that temperance does not need universality installed.
"""
import numpy as np

from . import result
from .weights import neff


def silverman_bandwidth(data, weights=None):
    """
    Silverman's rule-of-thumb bandwidth for 1D (optionally weighted) data,
    0.9 * std * neff**(-1/5).
    """
    data = np.asarray(data, dtype=float)
    if weights is None:
        std = np.std(data)
        num = len(data)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)
        std = (np.sum(weights * data**2) - np.sum(weights * data)**2)**0.5
        num = neff(weights)
    return 0.9 * std * num**(-0.2)


def logkde(samples, data, variances, weights=None):
    """
    log of the weighted Gaussian KDE built on `data`, evaluated at `samples`.

    samples: (Nsamp, Ndim) or (Nsamp,) points to evaluate at
    data: (Ndata, Ndim) or (Ndata,) points the KDE is built from
    variances: (Ndim,) kernel variance in each dimension
    weights: (Ndata,) weights for each data point, default uniform.  These
      are not renormalized, so they should sum to 1 for a normalized KDE.
    """
    samples = np.asarray(samples, dtype=float)
    data = np.asarray(data, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape((-1, 1))
        data = data.reshape((-1, 1))
    elif samples.ndim != 2:
        raise ValueError("bad shape for samples")
    variances = np.asarray(variances, dtype=float)
    if weights is None:
        weights = np.full(len(data), 1.0 / len(data))
    weights = np.asarray(weights, dtype=float)

    ndim = samples.shape[1]
    norm = -0.5 * ndim * np.log(2 * np.pi) - 0.5 * np.sum(np.log(variances))
    out = np.empty(len(samples))
    # Loop over samples rather than broadcasting so memory stays O(Ndata)
    for i, sample in enumerate(samples):
        z = -0.5 * np.sum((data - sample)**2 / variances, axis=1)
        m = np.max(z)
        out[i] = np.log(np.sum(weights * np.exp(z - m))) + m
    return out + norm


def vects2flatgrid(*vects):
    """
    The points of the grid spanned by the 1D arrays `vects`, as an
    (N1*N2*..., Ndim) array in 'ij' order.
    """
    return np.transpose([g.flatten() for g in np.meshgrid(*vects, indexing="ij")])


def vects2vol(vects):
    """
    The volume of one cell of the (assumed uniform) grid spanned by `vects`.
    """
    return np.prod([vect[1] - vect[0] for vect in vects])


class KDEFunction:
    """
    A weighted KDE over some columns of a set of samples, for use as the
    `density_function` of temperance.core.stats.DensityEstimate.
    """
    def __init__(self, samples, weight_columns, sample_columns):
        """
        samples: DataFrame-like; samples[column_name] gives one column
        weight_columns: iterable of temperance.core.result.WeightColumn,
          weights to use in the KDE
        sample_columns: iterable of temperance.core.stats.SamplesColumn,
          columns to build the KDE over.  Each column's `bandwidth` is used
          as the kernel *variance* in that dimension.
        """
        self.data = np.array(samples[[column.name for column in sample_columns]])
        self.variances = np.array([column.bandwidth for column in sample_columns])
        self.weights = result.get_total_weight(samples, weight_columns,
                                               weights_as_array=True)
        self.samples = samples
        self.weight_columns = weight_columns
        self.sample_columns = sample_columns

    @staticmethod
    def samples_to_input(samples):
        return np.transpose(np.array([samples[column] for column in samples]))

    def __call__(self, samples):
        """
        log of the KDE at each row of `samples` (DataFrame-like, one column
        per KDE dimension, in the order of `sample_columns`).
        """
        return logkde(self.samples_to_input(samples), self.data,
                      self.variances, weights=self.weights)

    def _grid_logkde(self, vects):
        vects = [np.asarray(vects[column]) for column in vects]
        return vects, logkde(vects2flatgrid(*vects), self.data, self.variances,
                             weights=self.weights)

    # The methods below take `vects`, a dict-like mapping each column to a
    # 1D uniform grid; the KDE is evaluated on the grid they span.
    def argmax(self, vects):
        """The grid point at which the KDE is largest."""
        vects, log_density = self._grid_logkde(vects)
        return vects2flatgrid(*vects)[np.argmax(log_density)]

    def entropy(self, vects):
        """Differential entropy of the KDE, approximated on the grid."""
        return _grid_entropy(*self._grid_logkde(vects))

    def information(self, vects):
        """log(number of grid points * cell volume) minus the entropy, as in universality."""
        vects, log_density = self._grid_logkde(vects)
        return (np.log(len(log_density) * vects2vol(vects))
                - _grid_entropy(vects, log_density))


def _grid_entropy(vects, log_density):
    finite = log_density > -np.inf
    return -vects2vol(vects) * np.sum(np.exp(log_density[finite]) * log_density[finite])
