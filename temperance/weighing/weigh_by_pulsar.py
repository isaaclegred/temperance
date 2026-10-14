import temperance as tmpy
import temperance.core.result as result
from  temperance.core.result import EoSPosterior

import numpy as np

def weigh_EoSs_by_mass_measurement(eos_posterior, likelihood=None, additional_samples=None, m_max_column="Mmax", minimum_mass=1.0, weight_tag="pulsar", log_output_weight=True,
                                   population=None, mass_likelihood=None, n_grid=2000):
    """
    Weigh the EoSs in eos_posterior  according to the NS mass measurement 
    with likelihood given by a function `likelihood`.  By default assumes a
    uniform population of NS masses on [minimum_mass, M_max]; with
    ``population`` (a core.populations.MassPopulation) and
    ``mass_likelihood`` the weight is instead

        int dm L(m) p(m | m <= M_max),

    so the same population object can drive both this likelihood and
    population fractions.
    Arguments:
      eos_posterior: EoSPosterior containing EoSs to be weighed, will be 
      modified to add weight column of pulsar
      likelihood: a function which shoul map Mmax -> weight (note not log 
      weight), based only on data and not occam factor; i.e. the integral of
      the mass likelihood from minimum_mass to Mmax.  Not used with
      ``population``.
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
      population: a MassPopulation, truncated at each EoS's Mmax
      mass_likelihood: a function m -> likelihood of the measurement given a
      mass m (needed with population)
      n_grid: mass grid for the population integral
    Returns:
      weight_column: The  WeightColumn which was added to the EoS Posterior 
      representing the weights
    """
    m_max = (additional_samples[[eos_posterior.eos_column, m_max_column]]
             if additional_samples is not None else
             eos_posterior.samples[[eos_posterior.eos_column, m_max_column]])
    weight_column_name = f"{'logweight' if log_output_weight else 'weight'}_{weight_tag}"
    weight_column = result.WeightColumn(weight_column_name, is_log=log_output_weight)
    if population is None:
        weights= np.where(m_max[m_max_column] > minimum_mass, likelihood(m_max[m_max_column])/(m_max[m_max_column] - minimum_mass), 0.0)
    else:
        if mass_likelihood is None:
            raise ValueError("a population needs mass_likelihood, the likelihood as a function of mass")
        weights = np.array([_population_weight(population, mass_likelihood, float(m), n_grid)
                            for m in m_max[m_max_column]])
    m_max[weight_column.name] = np.log(weights) if log_output_weight else weights
    eos_posterior.add_weight_column(weight_column, m_max[[eos_posterior.eos_column, weight_column.name]])
    return weight_column


def _population_weight(population, mass_likelihood, m_max, n_grid):
    """int dm L(m) p(m | m <= m_max) for one EoS."""
    if not m_max > population.m_min:
        return 0.0
    import scipy.integrate
    # the truncated density is smooth on [m_min, M_max], so Simpson's rule is accurate
    grid = np.linspace(population.m_min, min(m_max, population.m_max), n_grid + 1)
    return float(scipy.integrate.simpson(mass_likelihood(grid) * population.pdf(grid, m_max=m_max), x=grid))
