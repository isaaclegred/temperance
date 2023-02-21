# Plotting extracted quantiles, and overplotting tablulated equations of state.
#################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from dataclasses import dataclass, field

import temperance as tmpy
import temperance.core.result as result
from temperance.core.result import EoSPosterior
from  temperance.sampling.eos_prior import EoSPriorSet





import lal
import scipy
import random
from random import randint

import csv
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp


c_cgs=lal.C_SI*100
rhonuc=2.8e14


def get_defaults(matplotlib, fontsize=18):
    print("Warning! Modifying matplotlib defaults.")
    matplotlib.rcParams['figure.figsize'] = (9.7082039325, 6.0)
    matplotlib.rcParams['xtick.labelsize'] = fontsize
    matplotlib.rcParams['ytick.labelsize'] = fontsize
    matplotlib.rcParams['axes.labelsize'] = fontsize
    matplotlib.rcParams['legend.fontsize'] = fontsize
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    matplotlib.rcParams['font.sans-serif']= ['Bitstream Vera Sans']
    matplotlib.rcParams['text.usetex']= True
    matplotlib.rcParams['mathtext.fontset']= 'dejavuserif'
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['ytick.right'] = True





def get_file_type(variables, explicit_file_type=None):
    ''' 
    Find the file type based on the variables requested
    If you ask for inconsistent variables (i.e. M and baryon_density), the code will give you 
    the one that you ask for first, and it will only  fail when  "draw_curve" tries to find data 
    in the file that doesn't exist
    '''
    if explicit_file_type is not None:
        return explcit_file_type
    if ("M" in variables or "R" in variables or "Lambda" in variables or "rhoc" in variables):
        return "macro-draw-"
    elif ("baryon_density"  in variables or "pressurec2" in variables or "energy_densityc2" in variables or "cs2c2" in variables):
        return "eos-draw-"
    else :
        raise("unrecognized variables", variable)
def get_logweights(weight_file, eos_column="eos", logweight_column="logweight_total"):
    '''
    Get a Dictionary of logweights from a file with eoss and logweights
    '''
    eos = np.array(pd.read_csv(weight_file)["eos"])
    logweight = np.array(pd.read_csv(weight_file)[logweight_column])
    return {eos[i] : logweight[i] for i in range(len(eos))}

def overlay_eos(eos_file, variables=("M", "R"), label=None, ax=None, preprocessing=None):
    if ax is None:
        ax = plt.gca()
    eos = pd.read_csv(eos_file)
    data = eos[variables[0]], eos[variables[1]]
    if preprocessing is not None:
        data = preprocessing(data)
    ax.plot(*data, label=label)

# Data will come out in the order the variables are put in, if using explicit_file_type make sure it agrees with the variables being used, otherwise 
# the function will just find the right file to grab.  This can return arbitrary dimensional data, the more variables the more data. 
def draw_curve(eos_dir="/home/philippe.landry/nseos/eos/gp/mrgagn/", eos_per_dir=1000,
               num_dirs=2299, variables=("M","R"), explicit_file_type=None, 
               logweights=None, logweight_threshold=-10.0, known_index=None):
    '''
    Draw a curve from either an eos or macro file, if no logweights are provided, then draw from all of the possible eoss based on
    the num_dirs variable, otherwise, draw from the list of eoss in the logweights dictionary (logweights.keys())
    '''
    num_eos = eos_per_dir  * num_dirs
    # logweight doesn't matter if eos is fixed apriori
    if known_index is not None:
        draw_index = known_index
   
    # Use the list of logweights to draw from
    elif logweights is not None:
        logweight_sufficient = False
        while not logweight_sufficient:
            draw_index = np.random.choice(np.array(list(logweights.keys()), dtype=int))
            logweight_sufficient =  (logweights[draw_index] > logweight_threshold)
    else:
        draw_index = np.random.randint(num_eos)
    
    
    draw_dir = draw_index // eos_per_dir
    drawn_file = eos_dir + "DRAWmod" +str(eos_per_dir) + "-"+ str(draw_dir).zfill(6) + "/" + get_file_type(variables) + str(draw_index).zfill(6) + ".csv"
    data_curve = (np.array(pd.read_csv(drawn_file)[variable_name]) for i, variable_name in enumerate(variables))
    return data_curve
def apply_if(f, x, condition):
    '''
    Return f(x) if `condition` is satisfied, otherwise return x
    '''
    if condition:
        return f(x)
    else:
        return x
# If we don't want the units in /c2 units
def regularize_c2_scaling(data_curve, variables):
    '''
    If any `variables` contain the substring c2, this indicates they are divided by c2, this function multiplies back this c2, in cgs, to these variables.
    '''
    fix_this_scaling = lambda name : ("c2" in name)
    modified_data_curve = (apply_if(lambda data : data*c_cgs**2, data, fix_this_scaling(variables[i])) for (i, data) in enumerate(data_curve))
    return modified_data_curve
def normalize_density(data_curve, variables):
    '''
    If any `variables` are "baryon_density", change their units to units of rho_nuc
    '''
    fix_this_scaling = lambda name : ("baryon_density" in name)
    modified_data_curve = (apply_if(lambda data : data/rhonuc, data, fix_this_scaling(variables[i])) for (i, data) in enumerate(data_curve))
    return modified_data_curve
######################################################
# PLOT FAIR DRAWS
######################################################
def plot_fair_draws(these_logweights, these_vars, eos_dir, N=1, regularize_cs2_scaling=True, divide_by_rho_nuc=False, eos_per_dir=1000,color=None, lw=.7):
    for i in range(N):
        data = draw_curve(eos_dir = eos_dir, eos_per_dir=eos_per_dir,
                          variables=these_vars, logweights=these_logweights)
        if regularize_cs2_scaling:
            data = regularize_c2_scaling(data, these_vars)
        if divide_by_rho_nuc:
            data = normalize_density(data, these_vars)
        plt.plot(*data, color=color, lw=lw)

# Special helper functions for normalizing a rho axis
get_x_label = lambda normalize_axis : '$\\rho \,(\mathrm{g/cm}^3)$' if not normalize_axis else  \
    '$\\rho \,( \\rho_{\mathrm{nuc}})$'
def get_x_lim(normalize_axis, zoom=False):
    if zoom:
        lims = (1.5*rhonuc,2.5*rhonuc) 
    else:
        lims = (4e13,2.8e15)
    if normalize_axis:
        lims = tuple([lim/rhonuc for lim in lims])
    return lims



################################
# DEFINE PLOTTING FUNCTIONS
################################



def plot_generic_p_rho_envelope(post_path, prior_path, post_color="magenta", prior_color="red", this_label='GENERIC', lower=5, center=50, upper=95, divide_by_rho_nuc=False, no_prior=False, prior_linestyle="-.", prior_label=None, lw=2.5, ax=None, make_labels=True):
    ######################################################
    # GENERIC PRIOR
    ######################################################
    if ax is None:
        ax = plt.gca()
    if not no_prior:
        if prior_label is None:
            prior_label = this_label + " (prior)"
        columns = open(prior_path, 'r').readline().strip().split(',')
        data = np.loadtxt(prior_path, delimiter=',', skiprows=1)
        baryon_density = np.array([float(_.split('=')[1][:-1]) for _ in columns[1:]])
        if divide_by_rho_nuc:
            baryon_density = baryon_density/rhonuc
        pressures50 = data[center,:]*c_cgs*c_cgs
        pressures5 = data[lower,:]*c_cgs*c_cgs
        pressures95 = data[upper,:]*c_cgs*c_cgs
            
        #plt.fill_between(baryon_density,pressures5[1:],pressures95[1:],color='r',alpha=0.2)
        ax.plot(baryon_density,pressures5[1:],c=prior_color,lw=lw,
                linestyle=prior_linestyle,label= f'{prior_label}'if make_labels else None)
        ax.plot(baryon_density,pressures95[1:],c=prior_color,lw=lw,
                linestyle=prior_linestyle)

    ######################################################
    # GENERIC POSTERIOR
    ######################################################

    columns = open(post_path, 'r').readline().strip().split(',')
    data = np.loadtxt(post_path, delimiter=',', skiprows=1)
    baryon_density = np.array([float(_.split('=')[1][:-1]) for _ in columns[1:]])
    if divide_by_rho_nuc:
        baryon_density = baryon_density/rhonuc
    pressures50 = data[center,:]*c_cgs*c_cgs
    pressures5 = data[lower,:]*c_cgs*c_cgs
    pressures95 = data[upper,:]*c_cgs*c_cgs

    ax.fill_between(baryon_density,pressures5[1:],pressures95[1:],alpha=0.4, color=post_color)
    #plt.plot(baryon_density,pressures50[1:],c='b')
    ax.plot(baryon_density,pressures5[1:],c=post_color,lw=lw,label=f'{this_label}' if make_labels else\
 None)
    ax.plot(baryon_density,pressures95[1:],c=post_color,lw=lw)

def complete_p_rho_plot(divide_by_rho_nuc=False, ax=None, zoom=False, legend_loc="best", legend_alpha=.5):
    if ax is None:
        ax = plt.gca()
    if divide_by_rho_nuc:
        local_rhonuc = 1
        local_rho2nuc = 2
        local_rho6nuc = 6
    else : 
        local_rhonuc = rhonuc
        local_rho2nuc = 2*rhonuc
        local_rho6nuc = 6*rhonuc
    if not divide_by_rho_nuc:
        ax.axvline(local_rhonuc,c='k')
        ax.axvline(local_rho2nuc,c='k')
        ax.axvline(local_rho6nuc,c='k')

    ax.tick_params(direction='in')
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.figure.tight_layout()
    ax.grid(alpha=0.5)
    ax.set_ylabel('$ p \,(\mathrm{dyn/cm}^2)$')

    ax.set_xlabel(get_x_label(divide_by_rho_nuc))
  
    ax.set_xlim(*get_x_lim(divide_by_rho_nuc))
    ax.set_ylim(4e31,1e37)
    ax.legend(frameon=True,fancybox=True,framealpha=legend_alpha,fontsize=20, loc=legend_loc)
    if divide_by_rho_nuc:
        plt.xticks([.5,1,2,3,4,5,6,7,8,9], labels =["0.5", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    if not divide_by_rho_nuc:
        ax.figure.text(3e14,0.6e32,'$\\rho_{\mathrm{nuc}}$',fontsize=28,rotation=90)
        ax.figure.text(6e14,0.8e32,'$2\\rho_{\mathrm{nuc}}$',fontsize=28,rotation=90)
        ax.figure.text(18e14,0.8e32,'$6\\rho_{\mathrm{nuc}}$',fontsize=28,rotation=90)





Msol_in_km=lal.MSUN_SI*lal.G_SI/lal.C_SI/lal.C_SI/1000
 
def plot_generic_mr_envelope(post_path, prior_path,
                             post_color="magenta", prior_color="red",
                             this_label='GENERIC', lower=5, median=50,
                             upper=95, no_prior=False,
                             prior_linestyle="-.", lw=2.5, prior_label=None,
                             ax=None,):
    if ax is not None:
        plt.sca(ax)
    if not no_prior:
        if prior_label is None:
            prior_label = this_label + " (prior)"
        columns = open(prior_path, 'r').readline().strip().split(',')
        data = np.loadtxt(prior_path, delimiter=',', skiprows=1)
        mass = [float(_.split('=')[1][:-1]) for _ in columns[1:]]
        radius50 = data[median,:]
        radius5 = data[lower,:]
        radius95 = data[upper,:]
        
        #plt.fill_between(radius5[1:],radius95[1:],mass,color='r',alpha=0.2)
        plt.plot(radius5[1:],mass,c=prior_color,lw=lw,label=f'{prior_label}',linestyle=prior_linestyle)
        plt.plot(radius95[1:],mass,c=prior_color,lw=lw,linestyle=prior_linestyle)
        
    columns = open(post_path, 'r').readline().strip().split(',')
    data = np.loadtxt(post_path, delimiter=',', skiprows=1)
    mass = [float(_.split('=')[1][:-1]) for _ in columns[1:]]
    radius50 = data[median,:]
    radius5 = data[lower,:]
    radius95 = data[upper,:]

    plt.fill_betweenx(mass, radius5[1:],radius95[1:],alpha=0.4, color=post_color)
    #plt.plot(mass,radius50[1:],c='b')
    plt.plot(radius5[1:],mass,c=post_color,lw=lw,label=this_label)
    plt.plot(radius95[1:],mass,c=post_color,lw=lw)

def plot_envelope(post_path, prior_path,
                             post_color="magenta", prior_color="red",
                             this_label='GENERIC', lower=5, median=50,
                             upper=95, no_prior=False,
                             prior_linestyle="-.", lw=2.5, prior_label=None,
                             ax=None, x_scale=1 , y_scale= lambda x,y : y):
    if ax is not None:
        plt.sca(ax)
    if not no_prior:
        if prior_label is None:
            prior_label = this_label + " (prior)"
        columns = open(prior_path, 'r').readline().strip().split(',')
        data = np.loadtxt(prior_path, delimiter=',', skiprows=1, dtype=float)
        ind = np.array([float(_.split('=')[1][:-1]) for _ in columns[1:]], dtype=float)*x_scale
        dep50 = y_scale(ind, data[median, 1:])
        dep5 = y_scale(ind, data[lower, 1:])
        dep95 = y_scale(ind, data[upper, 1:])

        #plt.fill_between(radius5[1:],radius95[1:],mass,color='r',alpha=0.2)
        plt.plot(ind, dep5,c=prior_color,lw=lw,label=f'{prior_label}',linestyle=prior_linestyle)
        plt.plot(ind, dep95,ind,c=prior_color,lw=lw,linestyle=prior_linestyle)
        
    columns = open(post_path, 'r').readline().strip().split(',')
    data = np.loadtxt(post_path, delimiter=',', skiprows=1, dtype=float)
    ind = np.array([float(_.split('=')[1][:-1]) for _ in columns[1:]], dtype=float) * x_scale
    dep50 = np.array(y_scale(ind, data[median, 1:]), dtype=float)
    dep5 = np.array(y_scale(ind, data[lower, 1:]), dtype=float)
    dep95 = np.array(y_scale(ind, data[upper, 1:]), dtype=float)

    plt.fill_between(ind, dep5,dep95,alpha=0.4, color=post_color)
    #plt.plot(mass,radius50[1:],c='b')
    plt.plot(ind,dep5,c=post_color,lw=lw,label=this_label)
    plt.plot(ind, dep95,c=post_color,lw=lw) 

def complete_mr_plot(ax=None, ylim=(.8, 2.2), xlim=(8,14)):
    if ax is not None:
        plt.sca(ax)
    plt.tight_layout()
    plt.grid(alpha=0.5)
    plt.xlabel('$ R$ (km)')
    plt.ylabel('$M \, (M_{\odot})$')
    plt.ylim(ylim)
    plt.xlim(xlim)


    plt.yticks([1,1.4,1.8,2.2])
    plt.tick_params(direction='in')
    plt.legend(frameon=True,fancybox=True,framealpha=0.7,loc="lower left",fontsize=20)


def plot_generic_cs2_envelope(post_path, prior_path, post_color="magenta", prior_color="red", this_label='GENERIC', lower=5, center=50, upper=95, divide_by_rho_nuc=False, no_prior=False, prior_linestyle="-.", lw=2.5, prior_label=None, ax=None):
    ######################################################
    # GENERIC PRIOR
    ######################################################
    if ax is not None:
        plt.sca(ax)
    if not no_prior:
        if prior_label is None:
            prior_label = this_label + " (prior)"
        columns = open(prior_path, 'r').readline().strip().split(',')
        data = np.loadtxt(prior_path, delimiter=',', skiprows=1)
        baryon_density = np.array([float(_.split('=')[1][:-1]) for _ in columns[1:]])
        if divide_by_rho_nuc:
            baryon_density = baryon_density/rhonuc
            cs2s50 = data[50,:]
            cs2s5 = data[5,:]
            cs2s95 = data[95,:]
            plt.plot(baryon_density,cs2s5[1:],c=prior_color,lw=lw,linestyle=prior_linestyle, label=f'{prior_label}')
            plt.plot(baryon_density,cs2s95[1:],c=prior_color,lw=lw, linestyle=prior_linestyle)
    
    ######################################################
    # GENERIC POST
    ######################################################
    columns = open(post_path, 'r').readline().strip().split(',')
    data = np.loadtxt(post_path, delimiter=',', skiprows=1)
    baryon_density = np.array([float(_.split('=')[1][:-1]) for _ in columns[1:]])
    if divide_by_rho_nuc:
        baryon_density = baryon_density/rhonuc
    cs2s50 = data[50,:]
    cs2s5 = data[5,:]
    cs2s95 = data[95,:]

    plt.fill_between(baryon_density,cs2s5[1:],cs2s95[1:],color=post_color,alpha=0.25)
    plt.plot(baryon_density,cs2s5[1:],c=post_color,lw=lw,label=this_label)
    plt.plot(baryon_density,cs2s95[1:],c=post_color,lw=lw)
def complete_cs2_plot(divide_by_rho_nuc=False, log_cs2_axis=True, ax=None):
    if ax is not None:
        plt.sca(ax)
    if divide_by_rho_nuc:
        local_rhonuc = 1
        local_rho2nuc = 2
        local_rho6nuc = 6
    else : 
        local_rhonuc = rhonuc
        local_rho2nuc = 2*rhonuc
        local_rho6nuc = 6*rhonuc
    if not divide_by_rho_nuc:
        plt.axvline(local_rhonuc,c='k')
        plt.axvline(local_rho2nuc,c='k')
        plt.axvline(local_rho6nuc,c='k')
    plt.axhline(1/3, c='k')

    plt.tick_params(direction='in')

    plt.xscale('log')
    if log_cs2_axis:
        plt.yscale('log')

    plt.tight_layout()
    plt.grid(alpha=0.5)
    plt.ylabel('$ c_s^2/c^2$')
    plt.xlabel(get_x_label(divide_by_rho_nuc))
    plt.xlim(get_x_lim(divide_by_rho_nuc))
    plt.ylim(0.005,1.1)
    plt.legend(frameon=True,fancybox=True,framealpha=.4,fontsize=18, loc="lower right")
    if not divide_by_rho_nuc:
        plt.text(3e14,0.6e32,'$\\rho_{\mathrm{nuc}}$',fontsize=28,rotation=90)
        plt.text(6e14,0.8e32,'$2\\rho_{\mathrm{nuc}}$',fontsize=28,rotation=90)
        plt.text(18e14,0.8e32,'$6\\rho_{\mathrm{nuc}}$',fontsize=28,rotation=90)
        plt.text(.3*rhonuc,0.37,'$c_s^2/c^2 = 1/3$',fontsize=22)
    else:
        plt.text(.3,0.37,'$c_s^2/c^2 = 1/3$',fontsize=22)
        plt.xticks([.5,1,2,3,4,5,6,7,8,9], labels =["0.5", "1", "2", "3", "4", "5", "6", "7", "8", "9"])



@dataclass
class PlottableQuantiles:
    label : str
    quantiles : pd.DataFrame
    posterior : EoSPosterior
    weight_columns : list[result.WeightColumn] = None
    prior_set : EoSPriorSet=EoSPriorSet.get_default()
    color : str = None
    lw : float = 2.0
    scale_x : float = 1.0
    scale_y : float = 1.0
    fill_alpha : float = 0.1
    flip_axes : bool = False
    plot_kwargs : dict = field(default_factory=dict)
    def plot(self, ax, lower=5, upper=95, middle=50, plot_middle=False, plot_draws=None,
             no_quantiles=False, **plot_kwargs):

        if 'quantile' in self.quantiles.columns:
            self.quantiles.pop('quantile')
        if ax is None:
            ax = plt.gca()
        x_vals = np.array(list(map(
            lambda column : float(column.split("=")[1][:-1]), self.quantiles.columns))) * self.scale_x
        y_middle = self.quantiles.loc[50,:] * self.scale_y
        y_low = self.quantiles.loc[5,:] * self.scale_y
        y_high = self.quantiles.loc[95,:] * self.scale_y
        if not(no_quantiles):
            if not(self.flip_axes):
                ax.fill_between(x_vals, y_low, y_high, color=self.color, alpha=self.fill_alpha)
                ax.plot(x_vals,y_low,c=self.color,lw=self.lw,label=self.label,**self.plot_kwargs)
                ax.plot(x_vals,y_high,c=self.color,lw=self.lw, **self.plot_kwargs)
            else:
                ax.plot(y_low, x_vals,c=self.color,lw=self.lw,label=self.label, **self.plot_kwargs)
                ax.plot(y_high, x_vals,c=self.color,lw=self.lw, **self.plot_kwargs)
                ax.fill_betweenx(x_vals, y_low, y_high, alpha = self.fill_alpha, color=self.color)
        if plot_draws is not None and plot_draws != {}:
            num_draws = plot_draws["num_draws"]
            x_var = plot_draws["x_var"]
            y_var = plot_draws["y_var"]
            is_macro = plot_draws["is_macro"]
            indices = self.posterior.sample(
                columns=[self.posterior.eos_column],
                size=num_draws, weight_columns=self.weight_columns)
            for counter, draw_index in enumerate(indices[self.posterior.eos_column]):
                if is_macro:
                    data = pd.read_csv(self.prior_set.get_macro_path(int(draw_index)))
                    if counter == 0:
                        
                        plot_macro(data, columns=(x_var, y_var), color=self.color,
                                   label = self.label, **self.plot_kwargs)

                    else :
                        plot_macro(data, columns=(x_var, y_var), color=self.color, **self.plot_kwargs)

                else:
                    data = pd.read_csv(self.prior_set.get_eos_path(int(draw_index)))
                    if counter == 0:
                        ax.plot(data[x_var], data[y_var], color=self.color, label = self.label, **self.plot_kwargs)
                    else:
                        ax.plot(data[x_var], data[y_var], color=self.color, **self.plot_kwargs)
                        
                

        return ax
    
def default_initialize(figsize =(9.7082039325, 6) ):
    fig =  plt.figure(figsize=figsize)
    ax = plt.gca()
    return fig, ax

def plot_envelope(plottable_quantiles, initialize=default_initialize, initialize_kwargs={},plot_kwargs={}, complete_plot=None, complete_kwargs={}):
    if initialize is not None:
        fig, ax = initialize(**initialize_kwargs)
    for plottable_quantile in plottable_quantiles:
        ax = plottable_quantile.plot(ax, **plot_kwargs)
    if complete_plot is not None:
        ax = complete_plot(**complete_kwargs)
    return ax
def default_complete_plot(ax, variables,names,  ranges, legend_kwargs={"framealpha":.2, "loc":"best"}):
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel(names[variables[0]])
    ax.set_ylabel(names[variables[1]])
    ax.set_xlim(*ranges[variables[0]])
    ax.set_ylim(*ranges[variables[1]])
    ax.legend(**legend_kwargs)
    return ax

def trim_unstable(injection_data, M_column="M"):
    """
    Trim the final unstable branch off of TOV data in 
    dataframe format, 
    """
    last_stable = np.where(np.diff(injection_data[M_column]) > 0)[0][-1]
    return injection_data.loc[: last_stable + 1, :] 

def get_rhoc_mmax(injection_data, M_column="M", rhoc_column="rhoc"):
    return np.array(trim_unstable(injection_data)[[M_column, rhoc_column]])[-1, :]
def plot_macro(data, columns=("R", "M"), trim_collapse=True, mark_max_mass=True, M_column="M", **kwargs):
    if trim_collapse:
        trimmed_data = trim_unstable(data, M_column=M_column)
    else:
        trimmed_data = data
    plt.plot(trimmed_data[columns[0]], trimmed_data[columns[1]], **kwargs)
    if mark_max_mass:
        plt.scatter(trimmed_data[columns[0]].tail(1).item(), trimmed_data[columns[1]].tail(1).item(),
                    s=2, color="red")
    
