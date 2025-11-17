"""
Demonstrating how to weigh mass-radius samples based on EoSs in an h5 file (rather 
than an EoSPriorSet object)
"""
import temperance as tmpy 
import temperance.core.result as result
from temperance.core.result import EoSPosterior
from temperance.sampling.eos_prior import EoSPriorH5
import temperance.weighing.weigh_by_density_estimate as weigh_by_density_estimate
from temperance.weighing.flow import get_uniform_flow_density_estimate
from temperance.weighing import flow as tmflow
import h5py
import temperance.sampling.branched_interpolator as b_interp
import torch as torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class FlatMassPrior:
    def __init__(self, m_min, m_max, seed=None):
        self.m_min = m_min
        self.m_max = m_max
        self.rng = np.random.default_rng(seed)

    def sample(self, size):
        return self.rng.uniform(self.m_min, self.m_max, size)
if __name__ == "__main__":
    fake_nicer_samples = pd.DataFrame({"M":1.4 + .2 * np.random.randn(1000), "R":12 + 0.5 * np.random.randn(1000), "Prior":np.ones(1000)})
    # replace Mmax with the correct array of Mmaxs
    test_eos_posterior = EoSPosterior(samples = pd.DataFrame({"eos":np.arange(10000), "Mmax":np.ones(10000)*2.0}), label="test")
    # Make a fake EOS Posterior
    fake_eoss = h5py.File("fake_eos_samples.h5", "w")
    eos_group = fake_eoss.create_group("eos")
    ns_group = fake_eoss.create_group("ns")
    for i in range(10000):
        this_eos = pd.DataFrame({"logn":np.linspace(-1, 0.5, 50), "logp":np.log10(0.16 * (10**np.linspace(-1, 0.5))**2.34)})
        this_ns = pd.DataFrame({"central_baryon_density":np.geomspace(1e13, 3e15, 50) ,"M":np.linspace(0.1, 2.0, 50), "R":12.5 + 1.25 * np.random.randn()* np.sin(np.linspace(0, 3*np.pi, 50)), "Lambda":300 + 200 * np.cos(np.linspace(0, 3*np.pi, 50))})
        eos_group.create_dataset(f"eos_{i}", data=this_eos.to_records())
        ns_group.create_dataset(f"eos_{i}", data=this_ns.to_records())
    class prior_for_nicer:
        @staticmethod
        def log_prob(samples):
            return torch.zeros(len(samples))
    test_eos_prior_h5 = EoSPriorH5(h5_path="fake_eos_samples.h5", eos_subgroup_path="eos", macro_subgroup_path="ns", index_to_key=lambda x : f"eos_{int(x)}")
    mr_samples = weigh_by_density_estimate.generate_mr_samples(test_eos_posterior, test_eos_prior_h5,
                                                 FlatMassPrior, num_samples_per_eos=100, mass_prior_kwargs={"m_min":1.0, "m_max":2.0})
    
    likelihood = weigh_by_density_estimate.get_normalizing_flow_mr_likelihood_estimate(fake_nicer_samples, prior_distribution=prior_for_nicer)
    print("mr_samples are", np.array(mr_samples[["M", "R"]]))
    likelihood_vals = likelihood(np.array(mr_samples[["M", "R"]]))
    print("likelihood vals are", likelihood_vals)
    plt.hist(mr_samples["R"], weights=likelihood_vals, bins=30, density=True, alpha=0.5, label="weighed by flow likelihood")
    plt.hist(mr_samples["R"], bins=30, alpha=0.5, density=True, label="unweighted")
    plt.savefig("weighed_mr_samples.pdf", bbox_inches="tight")
    plt.xlabel("R (km)")
    plt.ylabel("Counts")
    plt.legend()
    print(mr_samples)
