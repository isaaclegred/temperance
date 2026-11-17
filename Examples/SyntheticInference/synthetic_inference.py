"""
EoS inference from synthetic data, end to end.

A toy family of equations of state (a Gamma = 2 polytrope with a
constant-speed-of-sound core, three parameters) provides both the "true" EoS
and the set of EoS draws to infer from.  Neutron-star masses follow a fixed
population: a double Gaussian like the Galactic neutron-star mass
distribution, with an upper cutoff M_max equal to each EoS's maximum (TOV)
mass, so that its normalisation depends (weakly) on the EoS.  Every
observation is drawn from the true EoS and this population:

  heavy pulsars   N_PULSARS masses are drawn; the N_HEAVIEST heaviest are
                  "observed" with Gaussian errors, as radio timing would.
  NICER           one more star, observed in mass and radius with correlated
                  Gaussian errors, released (as real NICER analyses are) as
                  posterior samples.
  GW events       binary neutron stars with both masses from the population,
                  paired with probability proportional to q^PAIRING_POWER
                  (q = m2 / m1, so equal masses are favoured; the power is
                  known in the inference), observed in chirp mass, mass
                  ratio and tidal deformability
                  Lambda-tilde with Gaussian errors, and released (as LVK
                  analyses are) as posterior samples under a prior flat in
                  (m1, m2, Lambda1, Lambda2).

Each observation then weighs the EoS draws, using the same population:

  heavy pulsars   PulsarMeasurement(...).get_mass_likelihood() with
                  weigh_by_pulsar.weigh_EoSs_by_mass_measurement, i.e.
                  int dm L(m) p(m | M_TOV).
  NICER           sample_single_neutron_star draws masses from the population
                  (truncated at each EoS's M_TOV) and reads off radii; a
                  normalizing flow fit to the posterior samples (weigh_by_
                  density_estimate.get_normalizing_flow_mr_likelihood_estimate)
                  is averaged over them.
  GW events       a normalizing flow fit to the posterior samples (weigh_by_
                  density_estimate.get_normalizing_flow_gw_likelihood_estimate)
                  is evaluated on binaries that sample_neutron_star_binary
                  draws from a fixed prior around the measured chirp mass,
                  reweighted by population / prior and averaged.

Selecting the heaviest pulsars: analyses of real heavy pulsars treat each as
an independent draw from the population, but here they are the heaviest of
N_PULSARS.  Each lighter pulsar is known only to lie below the lightest kept
one, m_k, with probability P(m < m_k) / Z(M_TOV), where Z(M_TOV) is the
population's probability below the cutoff; so the selection multiplies the
weight by Z(M_TOV)^-(N_PULSARS - N_HEAVIEST), up to an EoS-independent
factor.  For a uniform population this factor pins M_TOV just above the
heaviest pulsars.  For the double Gaussian almost all of the population lies
well below any M_TOV that supports the pulsars, so Z is close to 1 and the
factor matters only for M_TOV close to the observed masses; the heaviest
pulsars sit in the population's tail (around 2.1 Msun for 50 pulsars, well
below the true M_TOV) and bound M_TOV mainly from below.  The summary shows the posterior with and
without this factor.

The normalizing flows need torch: pip install -e ".[ml]".

Run:
    python synthetic_inference.py                       # 300 EoSs, about 10 minutes
    python synthetic_inference.py --n-eos 1000 --n-procs 8 --plot synthetic_inference.pdf   # about 10 minutes
"""

import argparse
import functools
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from temperance.core.populations import mixture, sample_truncated, truncated_normalisation
from temperance.core.result import EoSPosterior, WeightColumn
from temperance.core.weights import kish_neff, quantile
from temperance.solving.analytic_eos import piecewise_eos
from temperance.solving.solve_relativistic import get_tov_family
from temperance.utilities.parallel import map_over_eos
from temperance.utilities.units import RHO_SAT_GEOM
from temperance.weighing import weigh_by_density_estimate
from temperance.weighing.pulsar_measurements import PulsarMeasurement
from temperance.weighing.single_event_distributions import (
    sample_neutron_star_binary, sample_single_neutron_star, uniform_in_components_mass_prior)
from temperance.weighing.weigh_by_pulsar import weigh_EoSs_by_mass_measurement

# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

# Toy EoS: p = K rho^2 up to rho = x_t * rho_sat, then p rises with constant
# sound speed cs2 (geometric units, Msun = 1)
TRUE_EOS = {"K": 60.0, "x_t": 2.0, "cs2": 0.8}   # Gamma = 2 up to 2 rho_sat, then cs2 = 0.8
EOS_PRIOR = {"K": (40.0, 100.0), "x_t": (1.5, 3.0), "cs2": (0.3, 1.0)}
# EoS draws must support this mass, so that R and Lambda at 1.4 Msun exist
MINIMUM_M_TOV = 1.5

# The population of every neutron star: a double Gaussian similar to fits of Galactic neutron-star
# masses, on [M_MIN, M_CEILING], with an upper cutoff M_max = M_TOV (each EoS's maximum mass)
POPULATION_COMPONENTS = [(1.393, 0.064), (1.807, 0.177)]   # (mean, standard deviation) in Msun
POPULATION_FRACTIONS = [0.65, 0.35]
M_MIN = 1.0
M_CEILING = 3.5   # above every M_TOV in the EoS prior, so the cutoff is always M_TOV
# GW binaries: both masses from the population, paired with probability proportional to
# q^PAIRING_POWER (q = m2 / m1 <= 1), which favours equal masses; known in the inference
PAIRING_POWER = 4.0

# Heavy pulsars
N_PULSARS = 50
N_HEAVIEST = 3
PULSAR_MASS_ERROR = 0.05

# NICER: errors in M and R, their correlation, and how many posterior samples are released
# (and fit with a normalizing flow)
NICER_MASS_ERROR = 0.10
NICER_RADIUS_ERROR = 0.5   # km
NICER_CORRELATION = 0.5
N_NICER_SAMPLES = 4000

# GW events: errors in source-frame chirp mass, mass ratio m2/m1 and Lambda-tilde, and how many
# posterior samples are released (and fit with a normalizing flow)
N_GW_EVENTS = 2
CHIRP_MASS_ERROR = 0.002
MASS_RATIO_ERROR = 0.1
LAMBDA_TILDE_ERROR = 100.0
N_GW_SAMPLES = 20000
# the GW prior: q = m2 / m1 >= MINIMUM_MASS_RATIO (the population reaches M_MIN / M_CEILING), and
# each Lambda uniform on [0, GW_LAMBDA_MAX]
MINIMUM_MASS_RATIO = 0.3
GW_LAMBDA_MAX = 5000.0


def toy_eos(K, x_t, cs2):
    return piecewise_eos([
        {"type": "rest_mass_polytrope", "Gamma": 2.0, "K": K, "rho_max": x_t * RHO_SAT_GEOM},
        {"type": "css", "cs2": cs2, "e_max": 0.05},
    ])


def macro_table(parameters, n_stars=30, max_mass_step=0.04, max_rounds=8):
    """
    M [Msun], R [km], Lambda along a sequence of central densities spanning
    the maximum mass.  The samplers interpolate R and Lambda linearly in M
    between the solved stars, and the mass rises steeply just above the
    transition, so the sequence is refined (central densities inserted
    between neighbours) until consecutive stars differ by at most
    max_mass_step in mass: Lambda ~ M^-6 then interpolates to well under 1%.
    """
    eos = toy_eos(**parameters)

    def solve(densities):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return get_tov_family(eos, densities)[["central_baryon_density", "M", "R", "Lambda"]]

    densities = np.geomspace(0.8 * RHO_SAT_GEOM, 12 * RHO_SAT_GEOM, n_stars)
    stars = solve(densities)
    for _ in range(max_rounds):
        gaps = np.abs(np.diff(stars["M"].to_numpy())) > max_mass_step
        if not gaps.any():
            break
        midpoints = np.sqrt(densities[:-1][gaps] * densities[1:][gaps])
        densities = np.sort(np.concatenate([densities, midpoints]))
        stars = pd.concat([stars, solve(midpoints)]).sort_values("central_baryon_density", ignore_index=True)
    return stars


def macro_summary(table):
    """M_TOV, and R and Lambda at 1.4 Msun on the stable branch (one branch for this family)."""
    top = int(table["M"].idxmax())
    stable = table.iloc[:top + 1]
    return {"Mmax": float(stable["M"].iloc[-1]),
            "R(M=1.4)": float(np.interp(1.4, stable["M"], stable["R"])),
            "Lambda(M=1.4)": float(np.interp(1.4, stable["M"], stable["Lambda"])),
            "M_lightest": float(stable["M"].iloc[0]),
            "top_of_grid": top == len(table) - 1}


@functools.lru_cache(maxsize=None)
def population():
    """The neutron-star mass population before the cutoff at M_TOV, a bilby prior on [M_MIN, M_CEILING]."""
    from bilby.core.prior import TruncatedGaussian
    return mixture([TruncatedGaussian(mean, sigma, minimum=M_MIN, maximum=M_CEILING)
                    for mean, sigma in POPULATION_COMPONENTS], POPULATION_FRACTIONS)


class PopulationMasses:
    """Masses from the population cut off at m_max (a mass prior for sample_single_neutron_star)."""

    def __init__(self, m_max=np.inf, rng=None):
        self.m_max, self.rng = m_max, rng

    def sample(self, n):
        return sample_truncated(population(), n, m_max=self.m_max, rng=self.rng)


class PairedBinaries:
    """
    Binaries from the population cut off at m_max, paired with probability
    proportional to q^PAIRING_POWER: two independent masses, kept with
    probability q^PAIRING_POWER (rejection sampling).  A mass prior for
    sample_neutron_star_binary, which passes it each EoS's m_max.
    """

    def __init__(self, m_max, rng=None):
        self.m_max, self.rng = m_max, np.random.default_rng(rng)

    def sample(self, n):
        mass_1, mass_2 = np.empty(0), np.empty(0)
        while len(mass_1) < n:
            a, b = (sample_truncated(population(), 4 * n, m_max=self.m_max, rng=self.rng) for _ in range(2))
            heavier, lighter = np.maximum(a, b), np.minimum(a, b)
            kept = self.rng.uniform(size=len(a)) < (lighter / heavier) ** PAIRING_POWER
            mass_1, mass_2 = np.concatenate([mass_1, heavier[kept]]), np.concatenate([mass_2, lighter[kept]])
        return {"mass_1": mass_1[:n], "mass_2": mass_2[:n]}


@functools.lru_cache(maxsize=None)
def _pairing_normalisation_table(n_grid=4001):
    m = np.linspace(M_MIN, M_CEILING, n_grid)
    p = np.asarray(population().prob(m), dtype=float)
    # int_{M_MIN}^{m1} dm2 p(m2) (m2 / m1)^PAIRING_POWER, for each m1 on the grid
    paired = cumulative_trapezoid(p * m ** PAIRING_POWER, m, initial=0) / m ** PAIRING_POWER
    return m, cumulative_trapezoid(p * paired, m, initial=0)


def binary_population_density(mass_1, mass_2, m_max):
    """
    Density of ordered pairs (mass_1 >= mass_2) in the population cut off at
    m_max (one value per pair):
        p(m1) p(m2) (m2 / m1)^PAIRING_POWER / N(m_max),
    with N(m_max) the integral of the numerator over M_MIN <= m2 <= m1 <= m_max.
    """
    density = (np.asarray(population().prob(mass_1), dtype=float) * np.asarray(population().prob(mass_2), dtype=float)
               * (mass_2 / mass_1) ** PAIRING_POWER)
    grid, normalisation = _pairing_normalisation_table()
    inside = (mass_2 >= M_MIN) & (mass_2 <= mass_1) & (mass_1 <= m_max)
    return np.where(inside, density / np.interp(m_max, grid, normalisation), 0.0)


def gw_observables(mass_1, mass_2, lambda_1, lambda_2):
    """Source-frame chirp mass, q = m2 / m1 and Lambda-tilde."""
    from bilby.gw.conversion import component_masses_to_chirp_mass, lambda_1_lambda_2_to_lambda_tilde
    return (component_masses_to_chirp_mass(mass_1, mass_2), mass_2 / mass_1,
            lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2))


def gw_log_likelihood(theta, observed):
    """The GW likelihood: Gaussian in chirp mass, q and Lambda-tilde, for rows (m1, m2, Lambda1, Lambda2)."""
    values = gw_observables(*np.asarray(theta, dtype=float).T)
    errors = (CHIRP_MASS_ERROR, MASS_RATIO_ERROR, LAMBDA_TILDE_ERROR)
    measured = (observed["chirp_mass"], observed["mass_ratio"], observed["lambda_tilde"])
    return -0.5 * sum(((v - m) / e) ** 2 for v, m, e in zip(values, measured, errors))


def gw_posterior_samples(observed, rng, n, chunk=1_000_000):
    """
    n posterior samples (m1, m2, Lambda1, Lambda2) of a GW event, under a
    prior flat in all four with q >= MINIMUM_MASS_RATIO and Lambda <=
    GW_LAMBDA_MAX, as an LVK release would provide.  By importance sampling:
    proposals normal in chirp mass and uniform in q and the Lambdas, in chunks
    until the weights' effective sample size is 5 n, then resampled.
    """
    from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
    from scipy import stats
    proposal = stats.norm(observed["chirp_mass"], 2 * CHIRP_MASS_ERROR)
    thetas, log_weights, ess = [], [], 0.0
    while ess < 5 * n:
        chirp_mass = proposal.rvs(chunk, random_state=rng)
        mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass,
                                                                        rng.uniform(MINIMUM_MASS_RATIO, 1.0, chunk))
        theta = np.column_stack([mass_1, mass_2, rng.uniform(0, GW_LAMBDA_MAX, (chunk, 2))])
        # a prior flat in (m1, m2) has density m1^2 / chirp mass in (chirp mass, q)
        log_weight = gw_log_likelihood(theta, observed) + np.log(mass_1 ** 2 / chirp_mass) - proposal.logpdf(chirp_mass)
        keep = log_weight > -30   # the largest log weights are of order 1; the rest are negligible
        thetas.append(theta[keep])
        log_weights.append(log_weight[keep])
        weights = np.exp(np.concatenate(log_weights))
        ess = weights.sum() ** 2 / np.sum(weights ** 2)
    theta = np.concatenate(thetas)
    samples = pd.DataFrame(theta[rng.choice(len(theta), n, p=weights / weights.sum())],
                           columns=["mass_1", "mass_2", "lambda_1", "lambda_2"])
    samples["Prior"] = 0.0   # log prior density in (m1, m2, Lambda1, Lambda2): flat
    return samples


# ---------------------------------------------------------------------------
# The EoS draws
# ---------------------------------------------------------------------------

def draw_eos_set(n_eos, rng, n_procs=1, n_stars=30):
    """
    n_eos EoSs with parameters uniform in EOS_PRIOR, keeping those with M_TOV
    >= MINIMUM_M_TOV.  Returns an EoSPosterior (columns eos, the parameters,
    Mmax, R(M=1.4), Lambda(M=1.4)) and {eos: macro table}.
    """
    rows, macros = [], {}
    while len(rows) < n_eos:
        draws = [{name: rng.uniform(*bounds) for name, bounds in EOS_PRIOR.items()}
                 for _ in range(int(1.3 * (n_eos - len(rows))) + 1)]
        tables = map_over_eos(functools.partial(_macro_for, draws, n_stars), range(len(draws)), n_procs=n_procs)
        for i, table in tables.items():
            if not isinstance(table, pd.DataFrame):
                continue   # a failed solve (map_over_eos records the error)
            summary = macro_summary(table)
            if (summary["Mmax"] < MINIMUM_M_TOV or summary["M_lightest"] > M_MIN
                    or summary["top_of_grid"] or len(rows) == n_eos):
                continue
            macros[len(rows)] = table
            rows.append({"eos": len(rows), **draws[i], **{k: summary[k] for k in ("Mmax", "R(M=1.4)", "Lambda(M=1.4)")}})
    return EoSPosterior(pd.DataFrame(rows), label="prior"), macros


def _macro_for(draws, n_stars, i):
    return macro_table(draws[i], n_stars)


# ---------------------------------------------------------------------------
# Synthetic observations, drawn from the true EoS and the population
# ---------------------------------------------------------------------------

def simulate_observations(true_macro, rng, n_gw_samples=N_GW_SAMPLES):
    """Heavy pulsars, a NICER posterior and GW posteriors from the true EoS (see the module docstring)."""
    import bilby
    bilby.core.utils.random.seed(int(rng.integers(2**31)))
    m_tov = macro_summary(true_macro)["Mmax"]
    # the samplers, run on a one-EoS distribution, draw the true stars
    truth = EoSPosterior(pd.DataFrame({"eos": [0], "Mmax": [m_tov]}), label="truth")
    get_true_macro = lambda eos: true_macro

    stars = sample_single_neutron_star(truth, mass_prior_class=functools.partial(PopulationMasses, rng=rng),
                                       mass_prior_kwargs={}, num_samples_per_eos=N_PULSARS + 1,
                                       observables=("R",), external_get_macro=get_true_macro)
    nicer_star, pulsars = stars.iloc[0], np.sort(stars["M"].to_numpy()[1:])[::-1][:N_HEAVIEST]

    heavy_pulsars = [PulsarMeasurement(name=f"heavy pulsar {k + 1}", mass=m + rng.normal(0, PULSAR_MASS_ERROR),
                                       mass_minus=PULSAR_MASS_ERROR)
                     for k, m in enumerate(pulsars)]

    covariance = np.array([[NICER_MASS_ERROR ** 2, NICER_CORRELATION * NICER_MASS_ERROR * NICER_RADIUS_ERROR],
                           [NICER_CORRELATION * NICER_MASS_ERROR * NICER_RADIUS_ERROR, NICER_RADIUS_ERROR ** 2]])
    observed = rng.multivariate_normal([nicer_star["M"], nicer_star["R"]], covariance)
    # a flat prior, so the released posterior samples are the likelihood around the observed point;
    # "Prior" is the log prior density that the likelihood estimate divides out
    nicer_samples = pd.DataFrame(rng.multivariate_normal(observed, covariance, N_NICER_SAMPLES), columns=["M", "R"])
    nicer_samples["Prior"] = 0.0

    binaries = sample_neutron_star_binary(truth, mass_prior=functools.partial(PairedBinaries, rng=rng),
                                          num_samples_per_eos=N_GW_EVENTS, observables=["Lambda"],
                                          external_get_macro=get_true_macro)
    chirp_mass, mass_ratio, lambda_tilde = gw_observables(
        *(binaries[c].to_numpy() for c in ("mass_1", "mass_2", "lambda_1", "lambda_2")))
    gw_events = []
    for k in range(N_GW_EVENTS):
        event = {"name": f"GW event {k + 1}",
                 "true": {"chirp_mass": chirp_mass[k], "mass_ratio": mass_ratio[k], "lambda_tilde": lambda_tilde[k],
                          "mass_1": binaries["mass_1"].iloc[k], "mass_2": binaries["mass_2"].iloc[k]},
                 "chirp_mass": chirp_mass[k] + rng.normal(0, CHIRP_MASS_ERROR),
                 "mass_ratio": mass_ratio[k] + rng.normal(0, MASS_RATIO_ERROR),
                 "lambda_tilde": lambda_tilde[k] + rng.normal(0, LAMBDA_TILDE_ERROR)}
        event["samples"] = gw_posterior_samples(event, rng, n_gw_samples)
        gw_events.append(event)

    return {"m_tov": m_tov, "pulsar_masses": pulsars, "heavy_pulsars": heavy_pulsars,
            "nicer_star": {"M": nicer_star["M"], "R": nicer_star["R"]}, "nicer_observed": observed,
            "nicer_samples": nicer_samples, "gw_events": gw_events}


# ---------------------------------------------------------------------------
# Weighing the EoS draws
# ---------------------------------------------------------------------------

def _per_eos_log_mean(samples, values, name, eos_column="eos"):
    """log of the mean of ``values`` over each EoS's Monte Carlo samples."""
    frame = pd.DataFrame({eos_column: samples[eos_column].astype(int).to_numpy(), "value": values})
    with np.errstate(divide="ignore"):
        logs = np.log(frame.groupby(eos_column)["value"].mean())
    return pd.DataFrame({eos_column: logs.index, name: logs.to_numpy()})


def _add_log_weight(posterior, name, per_eos):
    column = WeightColumn(name, is_log=True)
    posterior.add_weight_column(column, per_eos)
    return column


def weigh_heavy_pulsars(posterior, measurements):
    """int dm L(m) p(m | M_TOV) for each pulsar (weigh_EoSs_by_mass_measurement cuts the population off at Mmax)."""
    return [weigh_EoSs_by_mass_measurement(posterior, population=population(), weight_tag=f"pulsar_{k + 1}",
                                           mass_likelihood=measurement.get_mass_likelihood())
            for k, measurement in enumerate(measurements)]


def weigh_pulsar_selection(posterior, n_drawn, n_kept):
    """
    The heaviest n_kept of n_drawn pulsars: each of the n_drawn - n_kept
    lighter ones contributes P(m < m_k) / Z(M_TOV), where m_k is the lightest
    kept mass and Z(M_TOV) the population's probability below the cutoff;
    only Z depends on the EoS.
    """
    z = np.array([truncated_normalisation(population(), m_tov) for m_tov in posterior.samples["Mmax"]])
    name = "logweight_pulsar_selection"
    per_eos = pd.DataFrame({"eos": posterior.samples["eos"].to_numpy(), name: -(n_drawn - n_kept) * np.log(z)})
    return _add_log_weight(posterior, name, per_eos)


def weigh_nicer(posterior, macros, nicer_samples, n_mc, rng, seed=0, flow_kwargs=None):
    """
    The NICER likelihood, a normalizing flow fit to the posterior samples
    reweighted by 1 / prior, averaged over each EoS's stars, with masses from
    the population: sampling the population truncated at M_TOV makes the
    average the population-marginalised likelihood directly.
    """
    import torch
    from temperance.weighing import flow   # (seeds torch and numpy's global RNG on import)
    torch.manual_seed(seed)
    likelihood = weigh_by_density_estimate.get_normalizing_flow_mr_likelihood_estimate(
        nicer_samples, **{"print_every": 0, **(flow_kwargs or {})})
    stars = sample_single_neutron_star(posterior, mass_prior_class=functools.partial(PopulationMasses, rng=rng),
                                       mass_prior_kwargs={}, num_samples_per_eos=n_mc,
                                       observables=("R",), external_get_macro=macros.__getitem__)
    values = likelihood(stars[["M", "R"]].to_numpy())
    return _add_log_weight(posterior, "logweight_nicer", _per_eos_log_mean(stars, values, "logweight_nicer"))


def weigh_gw_event(posterior, macros, gw_samples, n_mc, label, seed=0, flow_kwargs=None):
    """
    The GW likelihood is a normalizing flow fit to the posterior samples
    (reweighted by 1 / prior).  Binaries are drawn from a fixed prior g,
    uniform in the component masses within +/- 6 standard deviations of the
    posterior's chirp mass (so the Monte Carlo resolves it); as compact
    objects, so g does not depend on the EoS.  Each likelihood is then
    reweighted by population / g, with the paired population cut off at
    M_TOV (binary_population_density; g is constant, so it cancels between
    EoSs).
    """
    import torch
    from temperance.weighing import flow   # (seeds torch and numpy's global RNG on import)
    torch.manual_seed(seed)
    likelihood = weigh_by_density_estimate.get_normalizing_flow_gw_likelihood_estimate(
        gw_samples, **{"print_every": 0, **(flow_kwargs or {})})
    chirp_mass = gw_observables(*(gw_samples[c].to_numpy() for c in ("mass_1", "mass_2", "lambda_1", "lambda_2")))[0]
    centre, width = np.median(chirp_mass), np.std(chirp_mass)
    prior = uniform_in_components_mass_prior(chirp_mass=(centre - 6 * width, centre + 6 * width),
                                             mass_ratio=(MINIMUM_MASS_RATIO, 1.0))
    binaries = sample_neutron_star_binary(posterior, mass_prior=prior, num_samples_per_eos=n_mc,
                                          observables=["Lambda"], external_get_macro=macros.__getitem__,
                                          objects="compact_objects")
    likelihood = likelihood(binaries[["mass_1", "mass_2", "lambda_1", "lambda_2"]].to_numpy())
    weights = likelihood * binary_population_density(binaries["mass_1"].to_numpy(), binaries["mass_2"].to_numpy(),
                                                     binaries["Mmax"].to_numpy())
    name = f"logweight_{label}"
    return _add_log_weight(posterior, name, _per_eos_log_mean(binaries, weights, name))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

QUANTITIES = ["Mmax", "R(M=1.4)", "Lambda(M=1.4)"]


def summarize(posterior, weight_columns, truth):
    """Median and 90% interval of each quantity, and the effective number of EoSs."""
    weights = posterior.get_total_weight(weight_columns)["total_weight"].to_numpy()
    row = {"n_eff": kish_neff(weights)}
    for quantity in QUANTITIES:
        lo, median, hi = quantile(posterior.samples[quantity].to_numpy(), [0.05, 0.5, 0.95], weights=weights)
        row[quantity] = f"{median:.3g} [{lo:.3g}, {hi:.3g}]"
    row["truth inside 90%"] = all(
        quantile(posterior.samples[q].to_numpy(), [0.05], weights=weights)[0] <= truth[q]
        <= quantile(posterior.samples[q].to_numpy(), [0.95], weights=weights)[0] for q in QUANTITIES)
    return row


def plot(posterior, weight_columns, truth, path):
    """Corner plot of M_TOV, R(1.4) and log10 Lambda(1.4): prior, posterior and truth."""
    import corner
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.lines import Line2D

    def columns(values):
        return np.column_stack([values["Mmax"], values["R(M=1.4)"], np.log10(values["Lambda(M=1.4)"])])

    data = columns(posterior.samples)
    weights = posterior.get_total_weight(weight_columns)["total_weight"].to_numpy()
    colors = {"prior": "#7f7f7f", "posterior": "#1f77b4", "truth": "#d62728"}
    settings = dict(bins=40, smooth=1.0, plot_datapoints=False, plot_density=False, levels=(0.5, 0.9),
                    range=[tuple(np.percentile(data[:, i], [1, 99])) for i in range(data.shape[1])])
    # (a fresh hist_kwargs per call: corner writes its colour into the dict)
    figure = corner.corner(data, color=colors["prior"], hist_kwargs={"density": True}, **settings,
                           labels=[r"$M_\mathrm{TOV}\ [M_\odot]$", r"$R_{1.4}$ [km]", r"$\log_{10}\Lambda_{1.4}$"])
    corner.corner(data, weights=weights, fig=figure, color=colors["posterior"], hist_kwargs={"density": True},
                  truths=list(columns({q: np.array([truth[q]]) for q in QUANTITIES})[0]), truth_color=colors["truth"],
                  **settings)
    figure.legend(handles=[Line2D([], [], color=color, label=label) for label, color in colors.items()],
                  loc="upper right")
    figure.savefig(path, bbox_inches="tight")


def run(n_eos=300, n_mc=300, n_stars=30, n_procs=1, seed=1, plot_path=None, verbose=True, flow_kwargs=None,
        n_gw_samples=N_GW_SAMPLES):
    """
    Draw the data and the EoS set, weigh the EoSs, and return
    {"posterior", "observations", "truth", "weight_columns", "summary"}.
    flow_kwargs go to the training of the NICER and GW flows
    (flow.generate_improved_flow_density_estimate).
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)   # choose_macro_per_m picks among twin branches with numpy's global RNG
    true_macro = macro_table(TRUE_EOS, n_stars)
    truth = macro_summary(true_macro)
    observations = simulate_observations(true_macro, rng, n_gw_samples)
    posterior, macros = draw_eos_set(n_eos, rng, n_procs=n_procs, n_stars=n_stars)

    columns = {"heavy pulsars, as random draws": weigh_heavy_pulsars(posterior, observations["heavy_pulsars"]),
               f"heaviest {N_HEAVIEST} of {N_PULSARS} (selection)": [weigh_pulsar_selection(posterior, N_PULSARS,
                                                                                             N_HEAVIEST)],
               "NICER": [weigh_nicer(posterior, macros, observations["nicer_samples"], n_mc, rng, seed, flow_kwargs)],
               "GW events": [weigh_gw_event(posterior, macros, event["samples"], n_mc, f"gw_{k + 1}", seed + k + 1,
                                            flow_kwargs)
                             for k, event in enumerate(observations["gw_events"])]}

    stages, used = {"prior": []}, []
    for name, cols in columns.items():
        used = used + cols
        stages[f"+ {name}"] = list(used)
    weight_columns = used
    stages["all data, ignoring the selection"] = [c for c in used if c.name != "logweight_pulsar_selection"]
    summary = pd.DataFrame({stage: summarize(posterior, cols, truth) for stage, cols in stages.items()}).T

    if verbose:
        print(f"True EoS {TRUE_EOS}: M_TOV = {truth['Mmax']:.3f}, R(1.4) = {truth['R(M=1.4)']:.2f} km, "
              f"Lambda(1.4) = {truth['Lambda(M=1.4)']:.0f}")
        print(f"Heaviest {N_HEAVIEST} of {N_PULSARS} pulsars: true "
              + ", ".join(f"{m:.3f}" for m in observations["pulsar_masses"]) + "; observed "
              + ", ".join(f"{p.mass:.3f}" for p in observations["heavy_pulsars"]) + f" (+/- {PULSAR_MASS_ERROR})")
        star = observations["nicer_star"]
        print(f"NICER star: true M = {star['M']:.3f}, R = {star['R']:.2f} km; observed "
              f"M = {observations['nicer_observed'][0]:.3f}, R = {observations['nicer_observed'][1]:.2f} km")
        for event in observations["gw_events"]:
            print(f"{event['name']}: true m1 = {event['true']['mass_1']:.3f}, m2 = {event['true']['mass_2']:.3f}, "
                  f"Lambda-tilde = {event['true']['lambda_tilde']:.0f}; observed chirp mass = {event['chirp_mass']:.4f}, "
                  f"q = {event['mass_ratio']:.2f}, Lambda-tilde = {event['lambda_tilde']:.0f}")
        print(f"\n{len(posterior.samples)} EoSs; median [90% interval], adding each kind of data in turn:")
        print(summary.to_string())
    if plot_path is not None:
        plot(posterior, weight_columns, truth, plot_path)
        if verbose:
            print(f"\nwrote {plot_path}")
    return {"posterior": posterior, "observations": observations, "truth": truth,
            "weight_columns": weight_columns, "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-eos", type=int, default=300, help="number of EoS draws to weigh")
    parser.add_argument("--n-mc", type=int, default=300, help="Monte Carlo stars or binaries per EoS per event")
    parser.add_argument("--n-procs", type=int, default=1, help="processes for solving the EoS draws")
    parser.add_argument("--n-gw-samples", type=int, default=N_GW_SAMPLES,
                        help="posterior samples released per GW event")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--plot", default=None, help="write a corner plot of prior and posterior here")
    arguments = parser.parse_args()
    run(n_eos=arguments.n_eos, n_mc=arguments.n_mc, n_procs=arguments.n_procs, seed=arguments.seed,
        plot_path=arguments.plot, n_gw_samples=arguments.n_gw_samples)
