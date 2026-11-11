import temperance as tmpy
import temperance.core.result as result
from temperance.core.result import EoSPosterior
from temperance.core import kde
# moved to single_event_distributions; still importable from here for existing callers
from temperance.weighing.single_event_distributions import generate_mr_samples, generate_gw_samples  # noqa: F401

from temperance.core.stats import SamplesColumn

# torch and pyro (via temperance.weighing.flow) are optional, so they are
# imported inside the functions that need them.


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
):
  """
  Get a likelihood estimate for the mass-radius samples using a normalizing flow.
  Arguments:
  nicer_data_samples: The samples whose density should be estimated to construct the likelihood
  prior_distribution: If the prior can be represented in some other way, it can be passed as an argument
  here and `nicer_data_samples` will not be used
  prior_column: The column which should be read from the `nicer_data_samples` column to divide out,
  i.e. assuming the nicer_data_samples are drawn from the posterior on the event.
Returns:
  A function which takes a sample and returns the likelihood of that sample
  """
  import torch
  from temperance.weighing import flow as tmflow
  safe_exp = tmflow.safe_exp
  if prior_distribution is not None:
    posterior_density_estimate = tmflow.generate_improved_flow_density_estimate(
      np.array(nicer_data_samples[["M", "R"]]), weights=None
    )
    return lambda sample: safe_exp(posterior_density_estimate.log_prob(torch.tensor(sample, dtype=torch.float)) - 
                                   prior_distribution.log_prob(torch.tensor(sample, dtype=torch.float)))
  else:
    posterior_density_estimate = tmflow.generate_improved_flow_density_estimate(
      np.array(nicer_data_samples[["M", "R"]]), weights=None
    )
    prior_density_estimate = tmflow.generate_improved_flow_density_estimate(
      np.array(nicer_data_samples[["M", "R"]]),
      weights=result.get_total_weight(
        nicer_data_samples,
        weight_columns=[result.WeightColumn.get_inverse(prior_column)],
      )["total_weight"],
    )
    return lambda sample: safe_exp(posterior_density_estimate.log_prob(
      torch.tensor(sample, dtype=torch.float)
    ) - prior_density_estimate.log_prob(torch.tensor(sample, dtype=torch.float)))


def get_normalizing_flow_gw_likelihood_estimate(
    gw_data_samples,
    prior_distribution=None,
    prior_column=result.WeightColumn("Prior"),
):
  """
  Get a likelihood estimate for the mass-radius samples using a normalizing flow.
  Arguments:
  gw_data_samples: The samples whose density should be estimated to construct the likelihood
  prior_distribution: If the prior can be represented in some other way, it can be passed as an argument
  here and `gw_data_samples` will not be used
  prior_column: The column which should be read from the `gw_data_samples` column to divide out,
  i.e. assuming the gw_data_samples are drawn from the posterior on the event.
Returns:
  A function which takes a sample and returns the likelihood of that sample
  """
  import torch
  from temperance.weighing import flow as tmflow
  safe_exp = tmflow.safe_exp
  if prior_distribution is not None:
    posterior_density_estimate = tmflow.generate_flow_density_estimate(
      np.array(gw_data_samples[["mass_1", "mass_2", "lambda_1", "lambda_2"]])
    )
    return lambda sample: safe_exp(posterior_density_estimate.log_prob(torch.tensor(sample, dtype=torch.float)) - prior_distribution.log_prob(
      torch.tensor(sample, dtype=torch.float)
    ))
  else:
    posterior_density_estimate = tmflow.generate_flow_density_estimate(
      np.array(gw_data_samples[["mass_1", "mass_2", "lambda_1", "lambda_2"]])
    )
    prior_density_estimate = tmflow.generate_flow_density_estimate(
      np.array(gw_data_samples[["mass_1", "mass_2", "lambda_1", "lambda_2"]]),
      weights=result.get_total_weight(
        gw_data_samples,
        weight_columns=[result.WeightColumn.get_inverse(prior_column)],
      )["total_weight"],
    )
    return lambda sample: safe_exp(posterior_density_estimate.log_prob(
      torch.tensor(sample, dtype=torch.float)
    ) - prior_density_estimate.log_prob(torch.tensor(sample, dtype=torch.float)))