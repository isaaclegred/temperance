import temperance as tmpy
import temperance.core.result as result
from temperance.core.result import EoSPosterior
from temperance.core import kde
# moved to single_event_distributions; still importable from here for existing callers
from temperance.weighing.single_event_distributions import generate_mr_samples, generate_gw_samples  # noqa: F401

from temperance.core.stats import SamplesColumn

# torch (via temperance.weighing.flow) is optional, so it is
# imported inside the functions that need it.


import numpy as np
import pandas as pd
import bilby
import h5py





def weigh_samples_by_likelihood(
    samples,
    likelihood,
    weight_tag,
    additional_samples=None,
    log_output_weight=True,
    auxiliary_dependent_additional_factor=None,
):
    """
    Weigh the EoSs in eos_posterior  according to the NS mass measurement
    with likelihood given by a function `likelihood`.  Assumes a uniform population of NS
    masses
    Arguments:
      samples: Samples to be weighed by the likelihood
      likelihood: The likelihood to evaluate on samples (may be given by, e.g.,
      a density estimate)
      weight_tag: the label of the output weight column, should not contain
      "weight" or "logweight"
      additional_samples: dataframe auxiliary data that may contribute to the
      likelihood evaluation (e.g. EoS maximum mass needed to compute occam factors)
      log_output_weight: whether to log the output weight, will modify the
      weight column name
      auxiliary_dependent_additional_factor: A function which takes any additional samples
      and assigns an additional multiplicative weight to the likelihood
    Returns:
      weight_column: The  WeightColumn which was added to the EoS Posterior
      representing the weights
    """
    weight_column_name = (
        f"{'logweight' if log_output_weight else 'weight'}_{weight_tag}"
    )
    weight_column = result.WeightColumn(weight_column_name, is_log=log_output_weight)
    weights = likelihood(samples)
    if auxiliary_dependent_additional_factor is not None:
        weights *= auxiliary_dependent_additional_factor(additional_samples)
    samples[weight_column_name] = np.log(weights) if log_output_weight else weights
    return samples, weight_column


# gw_mass_prior = bilby.gw.prior.BNSPriorDict()


  


def weigh_mr_samples(
    mr_samples,
    nicer_data_samples=None,
    prior_column=result.WeightColumn("Prior"),
    density_estimate=None,
    bandwidth=None,
    bandwidth_factor=1/20
):
    """
    Weigh a set of mass-radius samples by a density estimate for the likelihood.
    Arguments:
      mr_samples: The samples which have been drawn to evaluate the likelihood on.
      nicer_data_samples:  The samples whose density should be estimated to construct the likelihood
      prior_column: The column which should be read from the `nicer_data_samples` column to divide out,
      i.e. assuming the nicer_data_samples are drawn from the posterior on the event.
      density_estimate: If a likelihood can be represented in some other way, it can be passed as an argument
      here and `nicer_data_samples` will not be used
      bandwidth: the bandwidth to be used in density estimation, if it is known.
      bandwidth_factor:  if not None, then multiply this factor into the bandwidth used by
      the kde, whether it is specified or computed internally. Default of 1/20 seen to work
      for existing NICER targets.  
    """
    if bandwidth is None:
        # Use separate bandwidths for the mass and radius bandwidths, because in all the cases we look at they are
        # very different.
 
        r_bandwidth = kde.silverman_bandwidth(
            nicer_data_samples["R"].to_numpy(),
            weights=np.exp(-np.array(nicer_data_samples[prior_column.name])),
        )
        m_bandwidth = kde.silverman_bandwidth(
            nicer_data_samples["M"].to_numpy(),
            weights=np.exp(-np.array(nicer_data_samples[prior_column.name])),
        )
    if bandwidth_factor is None:
      bandwidth_factor = 1.0
    print("m bandiwidth is", .5 * m_bandwidth * bandwidth_factor)
    print("r bandiwidth is", r_bandwidth * bandwidth_factor)
     
    if density_estimate is None:
        density_estimate = kde.KDEFunction(
            nicer_data_samples,
            weight_columns=[prior_column.get_inverse()],
            sample_columns=[
                SamplesColumn("M", label="M", bandwidth=0.5 * m_bandwidth * bandwidth_factor),
                SamplesColumn("R", label="R", bandwidth=r_bandwidth * bandwidth_factor),
            ],
        )
    return density_estimate(mr_samples)



def get_normalizing_flow_mr_likelihood_estimate(
    nicer_data_samples,
    prior_distribution=None,
    prior_column=result.WeightColumn("Prior"),
    **flow_kwargs,
):
  """
  Get a likelihood estimate for the mass-radius samples using a normalizing flow.
  Arguments:
  nicer_data_samples: The samples whose density should be estimated to construct the likelihood
  prior_distribution: If the prior can be represented in some other way (an object with a torch
  log_prob), it can be passed as an argument here and is divided out of a flow fit to the posterior
  samples; otherwise the samples are reweighted by `prior_column`
  prior_column: The column which should be read from the `nicer_data_samples` column to divide out,
  i.e. assuming the nicer_data_samples are drawn from the posterior on the event.
  flow_kwargs: passed to flow.generate_improved_flow_density_estimate (e.g. epochs)
Returns:
  A function which takes an array of [M, R] rows and returns the likelihood of each
  """
  import torch
  from temperance.weighing import flow as tmflow
  safe_exp = tmflow.safe_exp
  if prior_distribution is not None:
    posterior_density_estimate = tmflow.generate_improved_flow_density_estimate(
      np.array(nicer_data_samples[["M", "R"]]), weights=None, **flow_kwargs
    )
    return lambda sample: safe_exp(posterior_density_estimate.log_prob(torch.tensor(sample, dtype=torch.float)) -
                                   prior_distribution.log_prob(torch.tensor(sample, dtype=torch.float)))
  else:
    # posterior samples reweighted by 1 / prior are distributed as the
    # likelihood, so a flow fit to them is the likelihood (up to a constant),
    # as for the KDE in weigh_mr_samples
    likelihood_density_estimate = tmflow.generate_improved_flow_density_estimate(
      np.array(nicer_data_samples[["M", "R"]]),
      weights=result.get_total_weight(
        nicer_data_samples,
        weight_columns=[result.WeightColumn.get_inverse(prior_column)],
      )["total_weight"].to_numpy(),
      **flow_kwargs,
    )
    return lambda sample: safe_exp(likelihood_density_estimate.log_prob(torch.tensor(sample, dtype=torch.float)))


def gw_flow_coordinates(samples):
  """
  The coordinates a GW likelihood flow is fit in, and the log Jacobian back
  to the original ones.

  samples: array of rows (mass_1, mass_2, lambda_1, lambda_2), mass_1 >= mass_2
  Returns:
    (phi, log_jacobian): phi has rows (chirp mass, logit q, log lambda_1,
    log lambda_2), with q = mass_2 / mass_1; log_jacobian is log |d phi / d
    samples|, so a density in phi times exp(log_jacobian) is a density in the
    original coordinates.

  The chirp mass is measured far better than either mass, which makes the
  posterior a thin ridge in (mass_1, mass_2) but not in (chirp mass, q); and
  logit q and log lambda send the prior boundaries (q <= 1, lambda >= 0),
  where posteriors often pile up, to infinity, where a flow handles them as
  smooth tails rather than smearing a sharp edge.
  """
  from bilby.gw.conversion import component_masses_to_chirp_mass
  samples = np.asarray(samples, dtype=float)
  mass_1, mass_2, lambda_1, lambda_2 = samples.T
  q = mass_2 / mass_1
  chirp_mass = component_masses_to_chirp_mass(mass_1, mass_2)
  phi = np.column_stack([chirp_mass, np.log(q / (1 - q)), np.log(lambda_1), np.log(lambda_2)])
  # |d(chirp mass, q) / d(mass_1, mass_2)| = chirp mass / mass_1^2; d logit(q) / dq = 1 / (q (1 - q))
  log_jacobian = (np.log(chirp_mass / mass_1 ** 2) - np.log(q * (1 - q))
                  - np.log(lambda_1) - np.log(lambda_2))
  return phi, log_jacobian


def get_normalizing_flow_gw_likelihood_estimate(
    gw_data_samples,
    prior_distribution=None,
    prior_column=result.WeightColumn("Prior"),
    columns=("mass_1", "mass_2", "lambda_1", "lambda_2"),
    **flow_kwargs,
):
  """
  Get a likelihood estimate for binary-neutron-star masses and tidal
  deformabilities using a normalizing flow fit to GW posterior samples.

  The flow is fit in (chirp mass, logit q, log lambda_1, log lambda_2) (see
  gw_flow_coordinates) and turned back into a function of (mass_1, mass_2,
  lambda_1, lambda_2) with the Jacobian.
  Arguments:
  gw_data_samples: DataFrame of posterior samples of the event
  prior_distribution: optionally, an object whose log_prob(torch tensor of rows (mass_1, mass_2,
  lambda_1, lambda_2)) is the log prior density of the samples; it is divided out of a flow fit to the
  posterior.  Otherwise the samples are reweighted by `prior_column`.
  prior_column: The column of `gw_data_samples` holding the (log, by default) prior density of
  each sample in (mass_1, mass_2, lambda_1, lambda_2), to divide out.  Zero for a prior flat in these.
  columns: the columns of gw_data_samples holding mass_1, mass_2, lambda_1, lambda_2.  Use the
  source-frame masses (e.g. ("mass_1_source", "mass_2_source", "lambda_1", "lambda_2") for LVK
  releases), which is what EoS calculations give.
  flow_kwargs: passed to flow.generate_improved_flow_density_estimate (e.g. epochs)
Returns:
  A function which takes an array of rows (mass_1, mass_2, lambda_1, lambda_2) and returns the
  likelihood of each (up to a constant).  The masses of a row are reordered if mass_2 > mass_1.  Points
  beyond the sampled region in q or lambda (e.g. lambda = 0 for a black hole) are moved to its edge,
  i.e. the posterior is taken to be flat beyond the samples there, as it is at a prior boundary.
  """
  import torch
  from temperance.weighing import flow as tmflow
  data = np.array(gw_data_samples[list(columns)], dtype=float)
  phi, log_jacobian = gw_flow_coordinates(data)
  if prior_distribution is not None:
    weights = None
  else:
    # posterior samples reweighted by 1 / prior are distributed as the likelihood
    weights = result.get_total_weight(
      gw_data_samples, weight_columns=[result.WeightColumn.get_inverse(prior_column)])["total_weight"].to_numpy()
  density_estimate = tmflow.generate_improved_flow_density_estimate(phi, weights=weights, **flow_kwargs)
  # the sampled region in q and lambda: points beyond it are clamped to its edge
  q_max = np.max(data[:, 1] / data[:, 0])
  lambda_min = np.min(data[:, 2:], axis=0)

  def likelihood(sample):
    sample = np.array(sample, dtype=float, ndmin=2)
    swap = sample[:, 1] > sample[:, 0]
    sample[swap] = sample[swap][:, [1, 0, 3, 2]]
    mass_2 = np.minimum(sample[:, 1], q_max * sample[:, 0])
    lambdas = np.maximum(sample[:, 2:], lambda_min)
    clamped = np.column_stack([sample[:, 0], mass_2, lambdas])
    phi, log_jacobian = gw_flow_coordinates(clamped)
    with torch.no_grad():
      log_density = density_estimate.log_prob(torch.tensor(phi, dtype=torch.float)).numpy()
    if prior_distribution is not None:
      with torch.no_grad():
        log_density = log_density - prior_distribution.log_prob(torch.tensor(clamped, dtype=torch.float)).numpy()
    return np.exp(log_density + log_jacobian)

  return likelihood
