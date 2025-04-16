"""
Demonstrating how to weigh mass-radius samples based on EoSs in an h5 file (rather 
than an EoSPriorSet object)
"""
import temperance as tmpy 
import temperance.core.result as result
from temperance.core.result import EoSPosterior
from temperance.sampling.eos_prior import EoSPriorH5
import temperance.weighing.weigh_by_density_estimate as weigh_by_density_estimate

import pandas as pd
import numpy as np

class FlatMassPrior:
    def __init__(self, m_min, m_max, seed=None):
        self.m_min = m_min
        self.m_max = m_max
        self.rng = np.random.default_rng(seed)

    def sample(self, size):
        return self.rng.uniform(self.m_min, self.m_max, size)
if __name__ == "__main__":
    fake_nicer_samples = pd.DataFrame({"M":1.4 + .2 * np.random.rand(1000), "R":12.5 + 1.5 * np.random.rand(1000)})
    # replace Mmax with the correct array of Mmaxs
    test_eos_posterior = EoSPosterior(samples = pd.DataFrame({"eos":np.arange(10000), "Mmax":np.ones(10000)*2.0}), label="test")
    this_h5_path = "/home/isaac.legred/lwp/Examples/LCEHL_EOS_posterior_samples_PSR.h5"
    test_eos_prior_h5 = EoSPriorH5(h5_path=this_h5_path, eos_subgroup_path="eos", macro_subgroup_path="ns", index_to_key=lambda x : f"eos_{int(x)}")
    mr_samples = weigh_by_density_estimate.generate_mr_samples(test_eos_posterior, test_eos_prior_h5,
                                                 FlatMassPrior, num_samples_per_eos=100, mass_prior_kwargs={"m_min":1.0, "m_max":2.0})
    print(mr_samples)
