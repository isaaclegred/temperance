import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate as interpolate
import scipy.linalg as linalg
from  sklearn.mixture import GaussianMixture

import temperance as tmpy
import temperance.sampling.eos_prior as eos_prior
import temperance.core.result as result
from temperance.core.result import EoSPosterior 
import temperance.plotting.envelope as envelope
envelope.get_defaults(mpl, 18)



def plot_covariance(cov, axis_values=None,
                    plot_kwargs = {
                        "norm": mpl.colors.Normalize(vmin=0, vmax=.5),
                        "cmap": mpl.cm.plasma}):
    plt.matshow(cov,  **plot_kwargs)
    default_x_ticks = np.arange(0, len(cov), 100)
    if axis_values is not None:
        x_ticks = [f'{x:.2f}' for x in np.array(axis_values[default_x_ticks])/np.log(10)]
    else:
        x_ticks = default_x_ticks
    plt.xticks(default_x_ticks, labels=np.array(x_ticks))
    plt.yticks(default_x_ticks, labels=np.array(x_ticks))
    plt.colorbar()

def get_phi_of_logp(
        eos_posterior, weight_columns, n_components=1,
        eos_prior_set=eos_prior.EoSPriorSet.get_default(),
        max_num_eos=2000, variables=("log(pressurec2)", "phi"),
        gpr_path_template='draw-gpr-%(draw)06d.csv',
        load_eos_kwargs={"skipfooter": 5, "engine":"python"},
        interpolation_logp=None, covariance_type="full"):
    indices = np.array(eos_posterior.sample(
        size=max_num_eos, weight_columns=weight_columns,
        replace=False)["eos"], dtype=int)
    print("indices are", indices)
    print("default_file is", eos_prior_set.get_eos_path(
            indices[0],
            explicit_path=gpr_path_template))
    default_file = pd.read_csv(
        eos_prior_set.get_eos_path(
            indices[0],
            explicit_path=gpr_path_template),
        **load_eos_kwargs)
    default_data = default_file # [[*variables]]
    size =  len(interpolation_logp) if interpolation_logp is not None else len(
        np.array(default_data[variables[0]]))
    tracks = pd.DataFrame(
        np.empty((size, len(indices)), dtype=np.float64), columns=indices)
    for i, eos_index in enumerate(indices):
        data = pd.read_csv(
            eos_prior_set.get_eos_path(
                int(eos_index),
                explicit_path=gpr_path_template,
            ),
            **load_eos_kwargs)
        if interpolation_logp is not None:
            if i % 100 == 0:
                print("interp_logp", interpolation_logp)
                print("data logp", data[variables[0]])
            # print(default_data[variables[0]], tracks[eos_index])
            tracks[eos_index] = interpolate.griddata(
                np.array(data[variables[0]]),
                np.array(data[variables[1]]), interpolation_logp, method="linear")
        else:
            tracks[eos_index] = data[variables[1]]
            
    raw_data = np.transpose(np.array(tracks)[:, :]) # just the values of the dependent var
    return raw_data, interpolation_logp if interpolation_logp is not None else np.array(default_data[variables[0]])
    
def extract_classification_model_from_posterior(
        eos_posterior, weight_columns, n_components=1,
        eos_prior_set=eos_prior.EoSPriorSet.get_default(),
        max_num_eos=2000, variables=("log(pressurec2)", "phi"),
        gpr_path_template='draw-gpr-%(draw)06d.csv',
        load_eos_kwargs={"skipfooter": 5, "engine":"python"},
        interpolation_logp=None, covariance_type="full"):
    indices = np.array(eos_posterior.sample(
        size=max_num_eos, weight_columns=weight_columns,
        replace=False)["eos"], dtype=int)
    print("indices are", indices)
    print("default_file is", eos_prior_set.get_eos_path(
            indices[0],
            explicit_path=gpr_path_template))
    default_file = pd.read_csv(
        eos_prior_set.get_eos_path(
            indices[0],
            explicit_path=gpr_path_template),
        **load_eos_kwargs)
    default_data = default_file # [[*variables]]
    size =  len(interpolation_logp) if interpolation_logp is not None else len(
        np.array(default_data[variables[0]]))
    tracks = pd.DataFrame(
        np.empty((size, len(indices)), dtype=np.float64), columns=indices)
    for i, eos_index in enumerate(indices):
        data = pd.read_csv(
            eos_prior_set.get_eos_path(
                int(eos_index),
                explicit_path=gpr_path_template,
            ),
            **load_eos_kwargs)
        if interpolation_logp is not None:
            if i % 100 == 0:
                print("interp_logp", interpolation_logp)
                print("default data logp", default_data[variables[0]])
            # print(default_data[variables[0]], tracks[eos_index])
            tracks[eos_index] = interpolate.griddata(
                np.array(data[variables[0]]),
                np.array(data[variables[1]]), interpolation_logp, method="linear")
        else:
            tracks[eos_index] = data[variables[1]]
            
    raw_data = np.transpose(np.array(tracks)[:, :]) # just the values of the dependent variable
    print("nans at", np.where(raw_data != raw_data))
    gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type).fit(raw_data)

    predicted_classes = gm.predict(raw_data)
    classes = [raw_data[predicted_classes==i, :] for i in np.unique(predicted_classes)]
    print("zeroth class is raw_data?", np.all(raw_data == classes[0]))
    print("zeroth class data shape", classes[0].shape)
    print("classes are",classes)
    classes_gms = [(np.mean(class_data, axis=0),
                    np.cov(class_data, rowvar=False)) for class_data in classes]
    print("classes are:", [class_[1].shape  for class_ in classes_gms])
    weights = [sum(predicted_classes==i) for i in np.unique(predicted_classes)]
    print("predicted_classes are ", predicted_classes)
    print(np.mean(classes[0],  axis=0))
    print(classes_gms[0][0])
    return classes_gms, interpolation_logp if interpolation_logp is not None else np.array(default_data[variables[0]]), weights
def extract_mixture_model_from_posterior(
        eos_posterior, weight_columns, n_components=1,
        eos_prior_set=eos_prior.EoSPriorSet.get_default(),
        max_num_eos=2000, variables=("log(pressurec2)", "phi"),
        gpr_path_template='draw-gpr-%(draw)06d.csv',
        load_eos_kwargs={"skipfooter": 5, "engine":"python"},
        interpolation_logp=None):
    indices = np.array(eos_posterior.sample(
        size=max_num_eos, weight_columns=weight_columns,
        replace=True)["eos"], dtype=int)
    print("getting data from", eos_prior_set.get_eos_path(
            indices[0],
            explicit_path=gpr_path_template))
    default_file = pd.read_csv(
        eos_prior_set.get_eos_path(
            indices[0],
            explicit_path=gpr_path_template),
        **load_eos_kwargs)
    default_data = default_file # [[*variables]]
    size =  len(interpolation_logp) if interpolation_logp is not None else len(
        np.array(default_data[variables[0]]))
    tracks = pd.DataFrame(
        np.empty((size, len(indices)), dtype=np.float64), columns=indices)
    for i, eos_index in enumerate(indices):
        data = np.array(pd.read_csv(
            eos_prior_set.get_eos_path(
                int(eos_index),
                explicit_path=gpr_path_template,
            ),
            **load_eos_kwargs)[variables[1]])
        if interpolation_logp is not None:
            # print(default_data[variables[0]], tracks[eos_index])
            if i % 100 == 0:
                print("interp_logp", interpolation_logp)
                print("default data logp", default_data[variables[0]])
            tracks[eos_index] = interpolate.griddata(
                np.array(default_data[variables[0]]),
                data, interpolation_logp)
        else:
            tracks[eos_index] = data
            
    raw_data = np.transpose(np.array(tracks)[:, :]) # just the values of the dependent variable
    gm = GaussianMixture(n_components=n_components).fit(raw_data)
    return gm, interpolation_logp if interpolation_logp is not None else np.array(default_data[variables[0]])
def extract_gp_from_posterior(
        eos_posterior, weight_columns, 
        eos_prior_set=eos_prior.EoSPriorSet.get_default(),
        max_num_eos=2000,
        variables=("log(pressurec2)", "phi"),
        gpr_path_template='draw-gpr-%(draw)06d.csv',
        load_eos_kwargs={"skipfooter": 5, "engine":"python"},
        interpolation_logp=None):
    indices = np.array(eos_posterior.sample(
        size=max_num_eos, weight_columns=weight_columns,
        replace=True)["eos"], dtype=int)
    # get a file to store all of the results by copying the first
    # sampled dataframe, change it's name so we know which file it came from
    default_file = pd.read_csv(
        eos_prior_set.get_eos_path(
            indices[0],
            explicit_path=gpr_path_template),
        **load_eos_kwargs)
    default_data = default_file # [[*variables]]
    size =  len(interpolation_logp) if interpolation_logp is not None else len(
        np.array(default_data[variables[0]]))
    tracks = pd.DataFrame(
        np.empty((size, len(indices)), dtype=np.float64), columns=indices)
    for i, eos_index in enumerate(indices):
        data = np.array(pd.read_csv(
            eos_prior_set.get_eos_path(
                int(eos_index),
                explicit_path=gpr_path_template,
            ),
            **load_eos_kwargs)[variables[1]])
        if interpolation_logp is not None:
            # print(default_data[variables[0]], tracks[eos_index])
            tracks[eos_index] = interpolate.griddata(
                np.array(default_data[variables[0]]),
                data, interpolation_logp)
        else:
            tracks[eos_index] = data        
    raw_data = np.array(tracks)[:, :] # just the values of the dependent variable
    mean = np.mean(raw_data, axis=1)
    cov = np.cov(raw_data - mean[:, np.newaxis])
    return mean, cov, interpolation_logp if interpolation_logp is not None else np.array(default_data[variables[0]])

if __name__ == "__main__":
    eos_posterior = EoSPosterior.from_csv(
        "~/PTAnalysis/Analysis/collated_dbhf_2507_post.csv", label="dbhf_2507d")
    weight_columns = [result.WeightColumn("logweight_total")]
    # weight_columns=[]
    # mean, cov, logp= extract_gp_from_posterior(eos_posterior, weight_columns=weight_columns,
    #                                            max_num_eos=3000)
    gm, logp  = extract_mixture_model_from_posterior(eos_posterior,n_components=4, weight_columns=[],max_num_eos=3000)
    print("means", gm.means_)
    cov = gm.covariances_[0]
    # print(mean)
    print("cov", cov.shape, cov)
    print("logp", logp.shape, logp)
    plot_covariance(cov, axis_values=logp)
    plt.xlabel(r"$\log_{10}(p/c^2) [\mathrm{g}/\mathrm{cm}^3]$")
    plt.ylabel(r"$\log_{10}(p/c^2) [\mathrm{g}/\mathrm{cm}^3]$")
    plt.xlim(100, 400)
    plt.ylim(100, 400)
    plt.savefig(f"covariance_diagnostic_{eos_posterior.label}_gm.pdf")
    
    
