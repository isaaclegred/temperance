import temperance as tmpy
import temperance.core.result as result
from temperance.core.result import EoSPosterior
from temperance.sampling.eos_prior import EoSPriorSet
from temperance.sampling.eos_prior import EoSPriorH5
import temperance.external.universality_density_estimate as ude
try:
  import universality.kde as kde
except ImportError:
  class kde:
    import scipy.stats.gaussian_kde as gaussian_kde
    @staticmethod
    def silverman_bandwidth(data, weights):
      return gaussian_kde(data, weights=weights)

import temperance.sampling.branched_interpolator as b_interp
from temperance.core.stats import SamplesColumn


import numpy as np
import pandas as pd
import bilby
import copy
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


  


def generate_mr_samples(
    eos_posterior,
    eos_prior_set,
    mass_prior_class,
    num_samples_per_eos,
    mass_prior_kwargs={"m_min": 1.0},
):
    """
    A helper function that probably doesn't belong here.  Generate m-r samples
    for use in evaluating a mass-radius likelihood.
    Arguments:
      eos_posterior: An EoSPosterior
      eos_prior_set: The EoSPriorSet corresponding to the eos_posterior
      mass_prior_class: A constructor for the mass prior which will be sampled
      to get the mass prior.  This class must implement a `sample` method
      num_samples_per_eos: the number of monte carlo samples used to resolve
      each eos, for most application should be large enough that
      num_samples_per_eos * (width of mass measurement)/ (maximum mass - 1) >> 1
      mass_prior_kwargs: additional kwargs to pass to the mass_prior_class
      constructor
    Returns:
      mr_samples: a DataFrame of samples with eos, eos maximum tov mass,  mass, radius.
      There will generically be many m-r samples for each eos.

    """
    def get_macro(eos_index):
        if isinstance(eos_prior_set, EoSPriorH5):
            return eos_prior_set.get_macro(int(eos_index))
        elif isinstance(eos_prior_set, EoSPriorSet):
            return pd.read_csv(eos_prior_set.get_macro_path(int(eos_index)))
          
    # EoS samples
    eoss_to_use = eos_posterior.samples[[eos_posterior.eos_column, "Mmax"]]
    # columns are "eos-mmax(eos)-mass-radius"
    columns_to_copy = [eos_posterior.eos_column, "Mmax"]
    output_data = np.empty(
        (len(eoss_to_use) * num_samples_per_eos, 2 + len(columns_to_copy))
    )

    
    for i, eos_index in enumerate(eoss_to_use[eos_posterior.eos_column]):
        mass_prior_kwargs_local = copy.deepcopy(mass_prior_kwargs)
        if "m_max" not in mass_prior_kwargs_local.keys():
           mass_prior_kwargs_local["m_max"] = eoss_to_use["Mmax"].iloc[i]
        mass_prior = mass_prior_class(**mass_prior_kwargs_local)
        eos_table = get_macro(int(eos_index))
        mass_samples = mass_prior.sample(num_samples_per_eos)
        samples = b_interp.choose_macro_per_m(
            mass_samples,
            eos_table,
            black_hole_values={"R": lambda m: 1.477 * 2 * m},
            only_lambda=False,
        )
        output_data[i * num_samples_per_eos : (i + 1) * num_samples_per_eos, 0] = (
            eos_index
        )
        output_data[i * num_samples_per_eos : (i + 1) * num_samples_per_eos, 1] = eoss_to_use["Mmax"].iloc[i]
        output_data[i * num_samples_per_eos : (i + 1) * num_samples_per_eos, 2] = (
            samples["m"]
        )
        output_data[i * num_samples_per_eos : (i + 1) * num_samples_per_eos, 3] = (
            samples["R"]
        )
    return pd.DataFrame(
        data=output_data, columns=[eos_posterior.eos_column, "Mmax", "M", "R"]
    )


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
        density_estimate = ude.kde_function(
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
  if prior_distribution is not None:
    posterior_density_estimate = tmflow.generate_flow_density_estimate(
      np.array(nicer_data_samples[["M", "R"]])
    )
    return lambda sample: posterior_density_estimate(sample) / prior_distribution(
      sample
    )
  else:
    posterior_density_estimate = tmflow.generate_flow_density_estimate(
      np.array(nicer_data_samples[["M", "R"]])
    )
    prior_density_estimate = tmflow.generate_flow_density_estimate(
      np.array(nicer_data_samples[["M", "R"]]),
      weights=result.get_total_weight(
        nicer_data_samples,
        weight_columns=[result.WeightColumn.get_inverse(prior_column)],
      )["total_weight"],
    )
    return lambda sample: posterior_density_estimate(
      sample
    ) / prior_density_estimate(sample)
