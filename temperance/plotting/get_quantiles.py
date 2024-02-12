import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser



### non-standard libraries
from temperance.core import result
from  temperance.core.result import EoSPosterior
import temperance.sampling.eos_prior as eos_prior

from universality.utils import (utils, io)
from universality.properties import samples as usamples
from universality import plot

# This is the eos set we use most often
default_eos_prior= eos_prior.EoSPriorSet.get_default()
default_eos_prior.eos_dir="/home/isaac.legred/local_mrgagn_big_with_cs2c2"
default_eos_prior.macro_dir="/home/philippe.landry/nseos/eos/gp/mrgagn/"

DEFAULT_MAX_NUM_SAMPLES = 10000 # arbitrary

def get_quantiles(eos_posterior,  weight_columns=None,
                  variables=("baryon_density", "pressurec2"),
                  x_points=np.linspace(2.8e13, 2.8e15, 100),
                  max_num_samples=DEFAULT_MAX_NUM_SAMPLES,
                  quantiles =np.linspace(0, 1, 101),
                  eos_data=default_eos_prior, selection_rule="random",
                  use_macro=False, save_path=None, weight_threshold=-np.inf,
                  **kwargs):
    """
    Currently a thin wrapper around universality's 
    properties.samples.process2samples.
    Given an EoS posterior, compute credible regions for 
    one variable at given values of another variable,
    
    samples.process2quantiles(data,
                            tmp,
                              mod,
                              xcolumn,
                              ycolumn,
                              x_test,
                              quantiles,
                              quantile_type='sym',
                              x_multiplier=1.,
                              y_multiplier=1.,
                              weights=None,
                              selection_rule=DEFAULT_SELECTION_RULE,
                              branches_mapping=None,
                              default_y_value=None,
                              verbose=False,)
    Parameters
    ----------
    eos_posterior : Required, EoSPosterior
      The  object storing EoS samples
    weight_columns : Optional, List[Weight Columns]
      A collection of WeightColumn's which will be used to weigh EoSs
      in constructing the quantiles.
    variables : Optional, tuple[str]
      The (x, y) variables to be used in constructing quantiles,
      we produce quantiles for y at each given value of x
    x_points : Optional, np.ndarrray[float]
      The x-points we produce quantiles at, see above
    max_num_samples: Optional, int
      The maximum number of EoS samples to use in the analysis,
      must be less than the number of samples in the EoSPosterior
    eos_data: Optional, dictionary
      A dictionary storing all needed data to find the 
      EoS samples and interpret them, 
    selection_rule: Optional, str
      See universality documentation, strategy for ambiguous branch-value
    use_macro : Optional, str,
      Use macroscopic EoS quantities
    
    """
    # TODO This is not right, but it's fast, figure out how to correctly
    # "subtract 1" from the weight for having been sampling EoS from the
    # weights as well
    if (eos_posterior.compute_neff(weight_columns) > max_num_samples):
        print("warning, you might not be using enough samples to resolve the posterior")
    samples = eos_posterior.sample(size=max_num_samples,
                                   weight_columns=[])
    weights = np.array(result.get_total_weight(samples, weight_columns)["total_weight"])
    truth = weights > weight_threshold * max(weights)
    data = np.array(samples[eos_posterior.eos_column])
    branches_data=None if not(use_macro) else eos_data.branches_data
    template_base = (eos_data.eos_path_template
                     if not(use_macro) else
                     eos_data.macro_path_template)
    print( f"for vars, {variables}, template_base is {template_base}")
    template = eos_data.get_path_template(base=template_base, eos_dir=eos_data.macro_dir if use_macro else eos_data.eos_dir)
    print(f"template is, {template}")
    raw_quantiles, med =  usamples.process2quantiles(
        data=data[truth], tmp=template,
        mod=eos_data.eos_per_dir, xcolumn=variables[0],
        ycolumn=variables[1],
        quantiles=quantiles,
        x_test=x_points,
        weights=weights[truth], selection_rule=selection_rule,
        branches_mapping=branches_data,  **kwargs)
    quantiles = pd.DataFrame(
        raw_quantiles,
        columns = [f"{variables[1]}(variables[0]={x_val})" for x_val in x_points])
    if save_path is not None:
        quantiles.to_csv(save_path, index=False)
    return quantiles


# Convenience Wrappers
def get_p_of_rho_quantiles(eos_posterior,  weight_columns=None,
                  variables=("baryon_density", "pressurec2"),
                        x_points=np.linspace(2.8e13, 2.8e15, 100), **kwargs):
    """
    Default call to get p-rho quantiles
    """
    return get_quantiles(eos_posterior, weight_columns=weight_columns,
                         variables=variables, x_points=x_points,
                         selection_rule="nearest_neighbor", **kwargs)
def get_p_of_eps_quantiles(eos_posterior,  weight_columns=None,
                  variables=("energy_densityc2", "pressurec2"),
                        x_points=np.linspace(3e13, 5e15, 100), **kwargs):
    """
    Default call to get p-eps quantiles
    """
    return get_quantiles(eos_posterior, weight_columns=weight_columns,
                         variables=variables, x_points=x_points,
                         selection_rule="nearest_neighbor",**kwargs)
def get_cs2_of_rho_quantiles(eos_posterior,  weight_columns=None,
                  variables=("baryon_density", "cs2c2"),
                        x_points=np.linspace(2.8e13, 2.8e15, 100), **kwargs):
    """
    Default call to get cs2-rho quantiles
    """
    return get_quantiles(eos_posterior, weight_columns=weight_columns,
                         variables=variables, x_points=x_points,
                         selection_rule="nearest_neighbor",
                         **kwargs)

def get_r_of_m_quantiles(eos_posterior,  weight_columns=None,
                  variables=("M", "R"),
                        x_points=np.linspace(0.8, 2.1, 100),
                         **kwargs):
    """
    Default call to get m-r quantiles
    """
    return get_quantiles(eos_posterior, weight_columns=weight_columns,
                         variables=variables, x_points=x_points, use_macro=True,
                         **kwargs)

def get_lambda_of_m_quantiles(eos_posterior,  weight_columns=None,
                  variables=("M", "Lambda"),
                        x_points=np.linspace(1.0, 2.2, 100),
                               **kwargs):
    """
    Default call to get m-lambda quantiles
    """
    return get_quantiles(eos_posterior, weight_columns=weight_columns,
                         variables=variables, x_points=x_points, use_macro=True,
                         **kwargs
                         )



if __name__ == "__main__":
    # This could eventually be turned into an executable of it's own
    pass
