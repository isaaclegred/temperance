"""
Neutron-star mass measurements and the likelihood of each as a function of
the neutron-star mass.  This is necessary because pulsar mass measurements come 
in diverse forms and we would like to be able to reproduce e.g. 
Alsing, Silva & Berti 2018 (MNRAS 478, 1377, Eqs. 3-4).  For NICER
and GW likelihoods, we always rely on direct posterior samples.

PulsarMeasurement       one measurement (e.g. a row of the pulsar_masses
                        dataset); get_mass_likelihood() builds its likelihood
DirectMassLikelihood    a quoted mass with (possibly asymmetric) errors
TotalMassLikelihood     a mass function and a total mass
MassRatioLikelihood     a mass function and a mass ratio

A measurement constrains the neutron-star mass m_p in one of three ways:

  mass                          m_p itself, e.g. from the Shapiro delay
  mass_function, total_mass     f = (m_c sin i)^3 / (m_p + m_c)^2 and
                                M_T = m_p + m_c, e.g. from the advance of
                                periastron
  mass_function, mass_ratio     f and q = m_p / m_c, e.g. from the radial
                                velocities of the companion

A quoted mass becomes a split normal: one normal on each side of the quoted
value, with the quoted errors as their widths, so the quoted interval holds
the quoted probability.  For the other two, the mass function is known to
negligible precision, the total mass or mass ratio has a (split) normal
measurement error, and following Alsing, Silva & Berti 2018 (MNRAS 478,
1377, Eqs. 3-4) the likelihood is marginalised over an isotropic inclination
and over the true total mass or mass ratio with a flat prior:

    L(m_p) = int dx N(x | measured x) int_0^1 dcos(i) delta(f - f(m_p, x, i)).

The inclination integral is done exactly by the delta function and leaves
1 / cos(i), which diverges where the orbit would be edge-on; substituting x
= x_edge-on -/+ t^2 removes the divergence, so a fixed Gauss-Legendre rule in
t is accurate however narrow the measurement.  The mass ratio here is the
pulsar mass over the companion mass (e.g. 69 for PSR B1957+20), as in the
pulsar_masses table; the minimum pulsar mass is then f q (1 + q)^2.

A likelihood is a function of the mass (vectorized), with ``support`` (the
masses outside which it vanishes), ``integrate(density, m_min, m_max)`` for
integrals against a mass density using nodes placed on its own support
(needed when the likelihood is far narrower than any shared mass grid), and
``sample``.  It can be passed as ``mass_likelihood`` to
weigh_by_pulsar.weigh_EoSs_by_mass_measurement.

    from temperance.weighing.pulsar_measurements import PulsarMeasurement
    measurement = PulsarMeasurement.from_dataset("J1816+4510", "kaplan2013")
    likelihood = measurement.get_mass_likelihood()
    likelihood([1.6, 1.9, 2.2])
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import numpy as np
from scipy import stats

# probability within one standard deviation of the mean
ONE_SIGMA = float(stats.norm.cdf(1) - stats.norm.cdf(-1))

# measurement errors are integrated out to this many widths
_WIDTHS = 8.0

_FLOAT_FIELDS = ("mass", "mass_minus", "mass_plus", "mass_function", "total_mass", "total_mass_minus",
                 "total_mass_plus", "mass_ratio", "mass_ratio_minus", "mass_ratio_plus", "credible_level")


def _split_normal_pdf(x, centre, sigma_minus, sigma_plus):
    """Normal of width sigma_minus below centre and sigma_plus above, continuous at centre."""
    x = np.asarray(x, dtype=float)
    sigma = np.where(x < centre, sigma_minus, sigma_plus)
    return np.sqrt(2 / np.pi) / (sigma_minus + sigma_plus) * np.exp(-0.5 * ((x - centre) / sigma) ** 2)


def _widths(minus, plus, credible_level):
    """
    Split-normal widths whose interval [-minus, +plus] holds credible_level:
    each half holds its share of the same central probability, so both
    errors scale by the same normal quantile.
    """
    z = stats.norm.ppf(0.5 + credible_level / 2)
    return minus / z, plus / z


def _gauss_legendre(n):
    """Nodes and weights on (0, 1)."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1), 0.5 * w


def _edge_on_total_mass(m_p, f):
    """
    The total mass at which a pulsar of mass m_p would have mass function f
    edge-on: (M - m_p)^3 = f M^2, i.e. the smallest total mass allowed.
    """
    # Newton's method in u = log(M - m_p): G(u) = 3u - log f - 2 log(e^u + m_p)
    # increases with slope in [1, 3] and is concave, so after the first step
    # the iterates rise monotonically to the root; starting below the root
    # (M - m_p >= f and >= (f m_p^2)^(1/3)) keeps every iterate below it.
    u = np.log(np.maximum(f, np.cbrt(f * m_p ** 2)))
    for _ in range(100):
        x = np.exp(u)
        step = (3 * u - np.log(f) - 2 * np.log(x + m_p)) / (3 - 2 * x / (x + m_p))
        u = u - step
        if np.all(np.abs(step) < 1e-14):
            break
    return m_p + np.exp(u)


def _edge_on_mass_ratio(m_p, f):
    """
    The mass ratio q = m_p / m_c at which a pulsar of mass m_p would have mass
    function f edge-on: q (1 + q)^2 = m_p / f, i.e. the largest q allowed.
    """
    # Newton's method in u = log q: H(u) = u + 2 log(1 + e^u) - log(m_p / f)
    # increases with slope in [1, 3] and is convex, so starting above the root
    # (q <= m_p / f and q <= (m_p / f)^(1/3)) the iterates fall monotonically to it.
    k = m_p / f
    u = np.log(np.minimum(k, np.cbrt(k)))
    for _ in range(100):
        q = np.exp(u)
        step = (u + 2 * np.log1p(q) - np.log(k)) / (1 + 2 * q / (1 + q))
        u = u - step
        if np.all(np.abs(step) < 1e-14):
            break
    return np.exp(u)


class PulsarMassLikelihood:
    """
    Likelihood of a measurement as a function of the neutron-star mass, up to
    a constant.  Subclasses define ``_likelihood`` (on masses inside the
    support) and ``support``.
    """

    support: tuple[float, float]

    def __call__(self, m):
        m = np.asarray(m, dtype=float)
        flat = np.atleast_1d(m).ravel()
        out = np.zeros(flat.shape)
        lo, hi = self.support
        inside = (flat > lo) & (flat < hi)
        if inside.any():
            out[inside] = self._likelihood(flat[inside])
        return out.reshape(m.shape) if m.ndim else float(out[0])

    def log_likelihood(self, m):
        with np.errstate(divide="ignore"):
            return np.log(self(m))

    def _likelihood(self, m):
        raise NotImplementedError

    def _interval(self, m_min, m_max):
        lo = self.support[0] if m_min is None else max(self.support[0], float(m_min))
        hi = self.support[1] if m_max is None else min(self.support[1], float(m_max))
        if not np.isfinite(hi):
            raise ValueError(f"{type(self).__name__} has a tail to infinite mass; give m_max")
        return lo, hi

    def _breakpoints(self, lo, hi):
        """Points inside (lo, hi) where the likelihood is not smooth."""
        return []

    def integrate(self, density=None, m_min=None, m_max=None, n_nodes=256):
        """
        int dm L(m) density(m) over the support of L within [m_min, m_max], by
        Gauss-Legendre quadrature on that interval (split where L has a kink),
        so it resolves L however narrow it is.  ``density`` is a vectorised
        function of mass (default 1); it should be smooth on the interval,
        e.g. a population truncated at m_max.
        """
        lo, hi = self._interval(m_min, m_max)
        if not hi > lo:
            return 0.0
        edges = [lo, *sorted(b for b in self._breakpoints(lo, hi) if lo < b < hi), hi]
        x, w = _gauss_legendre(n_nodes)
        total = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            m = a + (b - a) * x
            values = self(m) if density is None else self(m) * np.asarray(density(m), dtype=float)
            total += (b - a) * float(np.sum(w * values))
        return total

    def sample(self, n, m_min=None, m_max=None, rng=None, n_grid=4001):
        """
        n masses drawn from the likelihood times a flat prior on [m_min,
        m_max] (by default the support), by inverting the cumulative
        integral on a grid.  This is the representation used by
        density-estimate likelihoods, e.g. Fan et al. 2024.
        """
        lo, hi = self._interval(m_min, m_max)
        if not hi > lo:
            raise ValueError("the likelihood vanishes on [m_min, m_max]")
        edges = [lo, *sorted(b for b in self._breakpoints(lo, hi) if lo < b < hi), hi]
        grid = np.unique(np.concatenate([np.linspace(a, b, n_grid) for a, b in zip(edges[:-1], edges[1:])]))
        values = self(grid)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(grid))])
        if not cdf[-1] > 0:
            raise ValueError("the likelihood vanishes on [m_min, m_max]")
        u = np.random.default_rng(rng).uniform(0, cdf[-1], size=n)
        return np.interp(u, cdf, grid)


class DirectMassLikelihood(PulsarMassLikelihood):
    """
    A quoted mass with errors -mass_minus/+mass_plus holding credible_level,
    as a split normal in the mass (a normalised density).
    """

    def __init__(self, mass, mass_minus, mass_plus, credible_level=ONE_SIGMA):
        self.mass = float(mass)
        self.sigma_minus, self.sigma_plus = _widths(float(mass_minus), float(mass_plus), credible_level)
        self.support = (max(self.mass - 10 * self.sigma_minus, 0.0), self.mass + 10 * self.sigma_plus)

    def _likelihood(self, m):
        return _split_normal_pdf(m, self.mass, self.sigma_minus, self.sigma_plus)

    def _breakpoints(self, lo, hi):
        return [self.mass]


class TotalMassLikelihood(PulsarMassLikelihood):
    """
    A mass function and a total mass M_T with errors -total_mass_minus/
    +total_mass_plus holding credible_level (Alsing et al. 2018, Eq. 3).  The
    pulsar mass is at most M_T - (f M_T^2)^(1/3), reached edge-on.
    """

    def __init__(self, mass_function, total_mass, total_mass_minus, total_mass_plus,
                 credible_level=ONE_SIGMA, n_nodes=96):
        self.f = float(mass_function)
        self.total_mass = float(total_mass)
        self.sigma_minus, self.sigma_plus = _widths(float(total_mass_minus), float(total_mass_plus),
                                                    credible_level)
        self.nodes = _gauss_legendre(n_nodes)
        top = self.total_mass + _WIDTHS * self.sigma_plus
        self.support = (0.0, top - np.cbrt(self.f * top ** 2))

    def _likelihood(self, m):
        f = self.f
        m_edge = _edge_on_total_mass(m, f)[:, None]
        x, w = self.nodes
        total = np.zeros(len(m))
        # one panel either side of the measured value, where the split normal has a kink
        for a, b in ((self.total_mass - _WIDTHS * self.sigma_minus, self.total_mass),
                     (self.total_mass, self.total_mass + _WIDTHS * self.sigma_plus)):
            # total masses below the edge-on one are not allowed; write M = M_edge + t^2
            t_a = np.sqrt(np.maximum(a - m_edge, 0.0))
            t_b = np.sqrt(np.maximum(b - m_edge, 0.0))
            t = t_a + (t_b - t_a) * x
            M = m_edge + t ** 2
            m_c = M - m[:, None]
            sin_i = np.cbrt(f * M ** 2) / m_c
            cos_i = np.sqrt(np.maximum(1 - sin_i ** 2, 0.0))
            # int dcos(i) delta(f - f(m_p, M, i)) = M^(4/3) / (3 m_c^2 f^(1/3) cos i), times dM/dt = 2t
            with np.errstate(divide="ignore", invalid="ignore"):
                integrand = (2 * t * M ** (4 / 3) / (3 * m_c ** 2 * np.cbrt(f) * cos_i)
                             * _split_normal_pdf(M, self.total_mass, self.sigma_minus, self.sigma_plus))
            integrand = np.where(t_b > t_a, integrand, 0.0)
            total += np.sum((t_b - t_a) * w * integrand, axis=1)
        return total


class MassRatioLikelihood(PulsarMassLikelihood):
    """
    A mass function and a mass ratio q = m_p / m_c with errors
    -mass_ratio_minus/+mass_ratio_plus holding credible_level (Alsing et al.
    2018, Eq. 4, with q the pulsar over the companion mass).  The pulsar mass
    is at least f q (1 + q)^2, reached edge-on, and has a tail to infinite
    mass from nearly face-on orbits.
    """

    def __init__(self, mass_function, mass_ratio, mass_ratio_minus, mass_ratio_plus,
                 credible_level=ONE_SIGMA, n_nodes=96):
        self.f = float(mass_function)
        self.mass_ratio = float(mass_ratio)
        self.sigma_minus, self.sigma_plus = _widths(float(mass_ratio_minus), float(mass_ratio_plus),
                                                    credible_level)
        self.nodes = _gauss_legendre(n_nodes)
        q_low = max(self.mass_ratio - _WIDTHS * self.sigma_minus, 0.0)
        self.support = (self.f * q_low * (1 + q_low) ** 2, np.inf)

    def _likelihood(self, m):
        f = self.f
        q_edge = _edge_on_mass_ratio(m, f)[:, None]
        x, w = self.nodes
        total = np.zeros(len(m))
        for a, b in ((self.mass_ratio - _WIDTHS * self.sigma_minus, self.mass_ratio),
                     (self.mass_ratio, self.mass_ratio + _WIDTHS * self.sigma_plus)):
            # mass ratios above the edge-on one are not allowed; write q = q_edge - t^2
            a = np.clip(a, 0.0, q_edge)
            b = np.clip(b, 0.0, q_edge)
            t_a = np.sqrt(q_edge - b)
            t_b = np.sqrt(q_edge - a)
            t = t_a + (t_b - t_a) * x
            q = q_edge - t ** 2
            sin_i = np.cbrt(f * q * (1 + q) ** 2 / m[:, None])
            cos_i = np.sqrt(np.maximum(1 - sin_i ** 2, 0.0))
            # int dcos(i) delta(f - f(m_p, q, i)) = q^(2/3) (1 + q)^(4/3) / (3 f^(1/3) m_p^(2/3) cos i)
            with np.errstate(divide="ignore", invalid="ignore"):
                integrand = (2 * t * q ** (2 / 3) * (1 + q) ** (4 / 3)
                             / (3 * np.cbrt(f) * m[:, None] ** (2 / 3) * cos_i)
                             * _split_normal_pdf(q, self.mass_ratio, self.sigma_minus, self.sigma_plus))
            integrand = np.where(t_b > t_a, integrand, 0.0)
            total += np.sum((t_b - t_a) * w * integrand, axis=1)
        return total


@dataclasses.dataclass
class PulsarMeasurement:
    """
    One measurement of a neutron-star mass, in the columns of the
    pulsar_masses dataset: a ``mass``, or a ``mass_function`` with a
    ``total_mass`` or a ``mass_ratio`` (pulsar over companion mass).  Errors
    are -minus/+plus around the quoted value and hold ``credible_level``;
    give one of them for symmetric errors.  Columns of a dataset row that are
    not fields (reference links, data URLs, ...) go to ``metadata``.
    """

    name: str = ""
    analysis: str = ""
    component: str = "neutron_star"
    system: str = ""
    mass: float | None = None
    mass_minus: float | None = None
    mass_plus: float | None = None
    mass_function: float | None = None
    total_mass: float | None = None
    total_mass_minus: float | None = None
    total_mass_plus: float | None = None
    mass_ratio: float | None = None
    mass_ratio_minus: float | None = None
    mass_ratio_plus: float | None = None
    credible_level: float = ONE_SIGMA
    method: str = ""
    reference: str = ""
    notes: str = ""
    metadata: dict = dataclasses.field(default_factory=dict, repr=False)

    def __post_init__(self):
        for quantity in ("mass", "total_mass", "mass_ratio"):
            minus, plus = f"{quantity}_minus", f"{quantity}_plus"
            if getattr(self, minus) is None:
                setattr(self, minus, getattr(self, plus))
            if getattr(self, plus) is None:
                setattr(self, plus, getattr(self, minus))
            if getattr(self, quantity) is not None and getattr(self, minus) is None:
                raise ValueError(f"{self._label()}: {quantity} needs an error ({minus} or {plus})")
        _ = self.kind  # raises unless the quantities define one kind of measurement

    def _label(self):
        return " ".join(part for part in (self.name, self.analysis) if part) or "measurement"

    @property
    def kind(self):
        """"mass", "total_mass" or "mass_ratio": which quantities constrain the mass."""
        given = {q for q in ("mass", "mass_function", "total_mass", "mass_ratio") if getattr(self, q) is not None}
        if given == {"mass"}:
            return "mass"
        if given == {"mass_function", "total_mass"}:
            return "total_mass"
        if given == {"mass_function", "mass_ratio"}:
            return "mass_ratio"
        raise ValueError(f"{self._label()}: give a mass, or a mass function with a total mass or a mass "
                         f"ratio; got {sorted(given) or 'nothing'}")

    def get_mass_likelihood(self, **kwargs):
        """
        The likelihood of this measurement as a function of the neutron-star
        mass: a DirectMassLikelihood, TotalMassLikelihood or
        MassRatioLikelihood (keyword arguments, e.g. n_nodes, go to it).
        """
        if self.kind == "mass":
            return DirectMassLikelihood(self.mass, self.mass_minus, self.mass_plus, self.credible_level)
        if self.kind == "total_mass":
            return TotalMassLikelihood(self.mass_function, self.total_mass, self.total_mass_minus,
                                       self.total_mass_plus, self.credible_level, **kwargs)
        return MassRatioLikelihood(self.mass_function, self.mass_ratio, self.mass_ratio_minus,
                                   self.mass_ratio_plus, self.credible_level, **kwargs)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]):
        """A measurement from a row of the pulsar_masses dataset (a Series or dict)."""
        fields = {f.name for f in dataclasses.fields(cls)} - {"metadata"}
        values, metadata = {}, {}
        for key, value in dict(row).items():
            missing = value is None or (isinstance(value, float) and np.isnan(value))
            if key in _FLOAT_FIELDS:
                values[key] = None if missing else float(value)
            elif key in fields:
                values[key] = "" if missing else str(value)
            else:
                metadata[key] = value
        if values.get("credible_level") is None:
            values.pop("credible_level", None)
        return cls(**values, metadata=metadata)

    @classmethod
    def from_table(cls, frame):
        """Measurements from the rows of a pulsar_masses table (a DataFrame)."""
        return [cls.from_row(row) for _, row in frame.iterrows()]

    @classmethod
    def from_dataset(cls, name, analysis=None, component="neutron_star"):
        """
        The measurement of ``name`` by ``analysis`` in the pulsar_masses
        dataset (temperance.datasets); analysis may be omitted when there is
        only one.
        """
        from temperance import datasets
        table = datasets.load_table("pulsar_masses")
        rows = table[(table["name"] == name) & (table["component"] == component)]
        if analysis is not None:
            rows = rows[rows["analysis"] == analysis]
        if len(rows) != 1:
            choices = sorted(table.loc[(table["name"] == name) & (table["component"] == component), "analysis"])
            raise ValueError(f"{len(rows)} rows for {name} ({component}) analysis={analysis!r}; "
                             f"analyses: {choices}")
        return cls.from_row(rows.iloc[0])
