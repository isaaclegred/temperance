import numpy as np
import pandas as pd


import temperance as tmpy
import temperance.sampling.eos_prior as eos_prior
import tempernace.core.result as result
from temperance.core.result import EoSPosterior 


def extract_gp_from_posterior(
        eos_posterior, weight_columns,
        eos_prior_set=eos_prior.EoSPriorSet.get_default(),
        max_num_eos=1000,
        variables=("log(pressurec2)", "phi"),
        gpr_path_template='draw-gpr-%(draw)06d.csv',
        load_eos_kwargs={"skipfooter": 5}):
    indices = eos_posterior.sample(n=max_num_eos, weight_columns=weight_columns,
                                   replace=True)
    
    
    
    
