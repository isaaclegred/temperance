import temperance as tmpy
import temperance.core.result as result
from  temperance.core.result import EoSPosterior
import bilby
import universality.kde as kde
import temperance.external.universality_density_estimate as ude
import temperance.sampling.branched_interpolator as b_interp
from temperance.core.stats import SamplesColumn

import numpy as np
import pandas as pd 

def weigh_samples_by_likelihood(samples, likelihood,  weight_tag, additional_samples=None, log_output_weight=True, auxiliary_dependent_additional_factor=None,):
    """
    Weigh the EoSs in eos_posterior  according to the NS mass measurement 
    with likelihood given by a function `likelihood`.  Assumes a uniform population of NS 
    masses
    Arguments:
      samples: Samples to be weighed by the likelihood
      likelihood: The likelihood to evaluate on samples (may be given by, e.g.,
      a density estimate)
      additional_samples: dataframe auxiliary data that may contribute to the
      likelihood evaluation (e.g. EoS maximum mass needed to compute occam factors)
      m_max_column: the name of the m_max column in either the EoS posterior 
      or the additional_samples
      minimum_mass: the minimum mass of a neutron star in the population, 
      used to compute the occam factor
      weight_tag: the label of the output weight column, should not contain 
      "weight" or "logweight"
      log_output_weight: whether to log the output weight, will modify the 
      weight column name
    Returns:
      weight_column: The  WeightColumn which was added to the EoS Posterior 
      representing the weights
    """
    weight_column_name = f"{'logweight' if log_output_weight else 'weight'}_{weight_tag}"
    weight_column = result.WeightColumn(weight_column_name, is_log=log_output_weight)
    weights = likelihood(samples) * auxiliary_dependent_additional_factor(additional_samples)
    samples[weight_column_name] = np.log(weights) if log_output_weight else weights
    return samples, weight_column

#gw_mass_prior = bilby.gw.prior.BNSPriorDict()


def generate_mr_samples(eos_posterior, eos_prior_set,  mass_prior_class,
                        num_samples_per_eos, mass_prior_kwargs={"m_min":1.0}):
    # EoS samples
    eoss_to_use = eos_posterior.samples[[eos_posterior.eos_column, "Mmax"]]
    # columns are "eos-mmax(eos)-mass-radius"
    columns_to_copy = [eos_posterior.eos_column, "Mmax"]
    output_data = np.empty((len(eoss_to_use) * num_samples_per_eos, 2 + len(columns_to_copy)))
    for i, eos_index in enumerate(eoss_to_use[eos_posterior.eos_column]):
        print(eoss_to_use["Mmax"])
        m_max = eoss_to_use["Mmax"].iloc[i]
        mass_prior = mass_prior_class(**mass_prior_kwargs, m_max=m_max)
        eos_table = pd.read_csv(eos_prior_set.get_macro_path(int(eos_index)))
        mass_samples = mass_prior.sample(num_samples_per_eos)
        samples = b_interp.choose_macro_per_m(
            mass_samples,
            eos_table,
            black_hole_values={"R" : lambda m : 1.477 * 2 * m}, only_lambda=False)
        output_data[i*num_samples_per_eos: (i + 1) * num_samples_per_eos, 0] = eos_index
        output_data[i*num_samples_per_eos: (i + 1) * num_samples_per_eos, 1] = m_max
        output_data[i*num_samples_per_eos: (i + 1) * num_samples_per_eos, 2] = samples["m"]
        output_data[i*num_samples_per_eos: (i + 1) * num_samples_per_eos, 3] = samples["R"]
    return pd.DataFrame(data=output_data,
                        columns=[eos_posterior.eos_column, "Mmax", "M", "R"])
            

def weigh_mr_samples(mr_samples, nicer_data_samples, prior_column=result.WeightColumn("Prior"), density_estimate=None,
                     bandwidth=None):
    if bandwidth is None:
        print("Finding the bandwidth...")
        r_bandwidth = kde.silverman_bandwidth(nicer_data_samples["R"].to_numpy(), weights=np.exp(-np.array(nicer_data_samples["Prior"])))
        m_bandwidth = kde.silverman_bandwidth(nicer_data_samples["M"].to_numpy(), weights=np.exp(-np.array(nicer_data_samples["Prior"])))
        print("Bandwidth found:", m_bandwidth, r_bandwidth)

    if density_estimate is None:
        density_estimate = ude.kde_function(
            nicer_data_samples,
            weight_columns=[prior_column],
            sample_columns=[SamplesColumn("M", label="M", bandwidth=m_bandwidth),
                            SamplesColumn("R", label="R", bandwidth=r_bandwidth)])
    return density_estimate(mr_samples)
    
    
    
