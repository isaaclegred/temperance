import temperance as tmpy
import temperance.core.result as result
from  temperance.core.result import EoSPosterior

import numpy as np

def weigh_EoSs_by_mass_measurement(eos_posterior, likelihood, additional_samples=None, m_max_column="Mmax", minimum_mass=1.0, weight_tag="pulsar", log_output_weight=True ):
    """
    Weigh the EoSs in eos_posterior  according to the NS mass measurement 
    with likelihood given by a function `likelihood`.  Assumes a uniform population of NS 
    masses
    Arguments:
      eos_posterior: EoSPosterior containing EoSs to be weighed, will be 
      modified to add weight column of pulsar
      likelihood: a function which shoul map Mmax -> weight (note not log 
      weight), based only on data and not occam factor
      additional_samples: dataframe containing m_max_column, used as m_max if 
      not None
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
    m_max = (additional_samples[[eos_posterior.eos_column, m_max_column]]
             if additional_samples is not None else
             eos_posterior.samples[[eos_posterior.eos_column, m_max_column]])
    weight_column_name = f"{'logweight' if log_output_weight else 'weight'}_{weight_tag}"
    weight_column = result.WeightColumn(weight_column_name, is_log=log_output_weight)
    weights= np.where(m_max[m_max_column] > minimum_mass, likelihood(m_max[m_max_column])/(m_max[m_max_column] - minimum_mass), 0.0)
    m_max[weight_column.name] = np.log(weights) if log_output_weight else weights
    eos_posterior.add_weight_column(weight_column, m_max[[eos_posterior.eos_column, weight_column.name]])
    return weight_column
