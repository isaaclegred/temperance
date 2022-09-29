
raise ImportError("MOVE THIS SOMEWHERE ELSE, THERE IS NO REASON FOR IT TO HAVE IT's OWN FILE")

import os
import numpy as np

from argparse import ArgumentParser



### non-standard libraries
from temperance.core import result
from  temperance.core.result import EoSPosterior

from universality.utils import (utils, io)
from universality.properties import samples
from universality import plot

# This is the eos set we use most often
DEFAULT_EOS_DATA = {"eos_dir":"/home/philippe.landry/nseos/eos/gp/mrgagn",
                    "eos_column" : "eos",
                    "eos_per_dir":1000,
                    "eos_basename" : 'eos-draw-%(draw)06d.csv',
                    "macro_basename": 'macro-draw-%(draw)06d.csv'
                    "branches_dir" : "/home/isaac.legred/local_mrgagn_big_with_cs2c2",
                    "branches_basename" : ('macro-draw-%(draw)06d-branches.csv',
                                           "rho",
                                           "start_baryon_density", "end_baryon_density")}

DEFAULT_MAX_NUM_SAMPLES = 10000 # arbitrary

def get_quantiles(eos_posterior,  weight_columns=None,
                  variables=("baryon_density", "pressurec2"),
                  x_points=np.linspace(2.8e13, 2.8e15, 100),
                  max_num_samples=DEFAULT_MAX_NUM_SAMPLES,
                  eos_data=DEFAULT_EOS_DATA, selection_rule="random",
                  use_macro=False,
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
    samples = eos_posterior.sample(max_num_samples)
    weights = eos_posterior.get_weight(weight_columns)
    truth = weights > weight_threshold
    data = samples["eos"]
    template = (eos_data["eos_basename"]
                if not use_macro else
                eos_data["macro_basename"])
    return samples.process2quantiles(data=data, tmp=eos_data["eos_basename"],
                              mod=eos_data["eos_per_dir"], xcolumn=variables[0],
                              ycolumn=variables[1], x_test=x_points,
                              weights=weights, selection_rule=selection_rule)
    
    

if __name__ == "__main__":
    # This could eventually be turned into an executable of it's own

