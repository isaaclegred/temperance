import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mpl

import temperance as tmpy
import temperance.sampling.eos_prior as eos_prior
import temperance.core.result as result
from temperance.core.result import EoSPosterior 
import temperance.plotting.envelope as envelope
envelope.get_defaults(mpl, 18)


def plot_covariance(cov, axis_values=None, plot_kwargs = {"norm":mpl.colors.Normalize(vmin=0, vmax=.5), "cmap":mpl.cm.plasma}):
    plt.matshow(cov,  **plot_kwargs)
    default_x_ticks = np.arange(0, len(cov), 100)
    print(axis_values)
    if axis_values is not None:
        print("x_ticks being set~")
        x_ticks = [f'{x:.2f}' for x in np.array(axis_values[default_x_ticks])/np.log(10)]
    else:
        x_ticks = default_x_ticks
    print(x_ticks)
    plt.xticks(default_x_ticks, labels=np.array(x_ticks))
    plt.yticks(default_x_ticks, labels=np.array(x_ticks))
    plt.colorbar()
def extract_gp_from_posterior(
        eos_posterior, weight_columns,
        eos_prior_set=eos_prior.EoSPriorSet.get_default(),
        max_num_eos=2000,
        variables=("log(pressurec2)", "phi"),
        gpr_path_template='draw-gpr-%(draw)06d.csv',
        load_eos_kwargs={"skipfooter": 5, "engine":"python"}):
    indices = np.array(eos_posterior.sample(size=max_num_eos, weight_columns=weight_columns,
                                            replace=True)["eos"],dtype=int)
    # get a file to store all of the results by copying the first
    # sampled dataframe, change it's name so we know which file it came from
    default_file = pd.read_csv(
        eos_prior_set.get_eos_path(indices[0],
                                   explicit_path=gpr_path_template,
        ), **load_eos_kwargs)
    default_data = default_file#[[*variables]]
    print(default_data)
    tracks = pd.DataFrame(np.empty((len(default_file[variables[0]]), len(indices)), dtype=np.float64), columns=indices)
    for i, eos_index in enumerate(indices):
        tracks[eos_index] =np.array( pd.read_csv(
            eos_prior_set.get_eos_path(
                int(eos_index),
                explicit_path=gpr_path_template,
            ),
            **load_eos_kwargs)[variables[1]])
    raw_data = np.array(tracks)[:, :] # just the values of the dependent variable
    mean = np.mean(raw_data, axis=1)
    cov = np.cov(raw_data)
    print("cov shape is ", cov)
    return mean, cov, default_data[variables[0]]

if __name__ == "__main__":
    eos_posterior=EoSPosterior.from_csv(
        "~/PTAnalysis/Analysis/collated_dbhf_2507_post.csv", label="dbhf_2507d")
    weight_columns=[result.WeightColumn("logweight_total")]
    #weight_columns=[]
    mean, cov, logp= extract_gp_from_posterior(eos_posterior, weight_columns=weight_columns,
                                               max_num_eos=3000)
    print(mean)
    print(cov)
    print(logp)
    plot_covariance(cov, axis_values=logp)
    plt.xlabel(r"$\log_{10}(p/c^2) [\mathrm{g}/\mathrm{cm}^3]$")
    plt.ylabel(r"$\log_{10}(p/c^2) [\mathrm{g}/\mathrm{cm}^3]$")
    plt.xlim(100, 400)
    plt.ylim(100, 400)
    plt.savefig(f"covariance_diagnostic_{eos_posterior.label}.pdf")
    
    
