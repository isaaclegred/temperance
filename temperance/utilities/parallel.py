import subprocess
import multiprocessing
import numpy as np
from temperance.sampling import eos_prior
import temperance as tmpy
from  temperance.core.result import EoSPosterior
import pandas as pd
def f(x):
    return x ** 2

def get_desired_property(on):
    return prior_set.get_property(on, dependent_variables=["R"],
                           extraction=eos_prior.Extraction("interpolation", True, "M",
                                                           np.array([1.6, 1.8])) ,
                           default_values = [1.477*1.6], verbose=True)

def distribute(on, num_procs, function,  *args, **kwargs):
    """
    Distribute the call to task with args *args, and *kwargs, the first 
    argument to task should be any subset of the array on
    """    
    with multiprocessing.Pool(processes=num_procs) as pool:
        output = pool.map(function, on)
        print(output)
    return output


Astro_eos_likelihood = EoSPosterior.from_csv("/home/isaac.legred/PTAnalysis/Analysis/collated_np_all_post.csv")
prior_set = tmpy.sampling.eos_prior.EoSPriorSet.get_default()
eoss =  np.array(Astro_eos_likelihood.samples[Astro_eos_likelihood.eos_column])
eos_sets =np.array_split(eoss, 25)
def main():
    # We base the EoSs we use in our analysis solely on those we have an astrophysical weight for
    evaluations = distribute(eos_sets, 25, get_desired_property)
    return evaluations

if __name__ == "__main__":
    evaluations = main()
    print(evaluations)
    for i, eos_set in enumerate(eos_sets):
        evaluations[i]["eos"] = eos_set
    output = pd.concat(evaluations)
    output.to_csv("R_of_M_astro.csv", index=False)
