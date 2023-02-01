import numpy as np
import pandas as pd
from dataclasses import dataclass 
from itertools import chain

import universality as unvs
import universality.plot as uplot

import matplotlib
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt


import temperance.core.result as result


def get_default_plot_settings(matplotlib):
    matplotlib.rcParams['xtick.labelsize'] = 18.0
    matplotlib.rcParams['ytick.labelsize'] = 18.0
    matplotlib.rcParams['axes.labelsize'] = 18.0
    matplotlib.rcParams['legend.fontsize'] = 18.0
    matplotlib.rcParams['font.family']= 'Times New Roman'
    matplotlib.rcParams['font.sans-serif']= ['Bitstream Vera Sans']
    matplotlib.rcParams['text.usetex']= True
    matplotlib.rcParams['mathtext.fontset']= 'stixsans'
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['ytick.right'] = True
    
def generate_labels(name, extraction_variable, extraction_values,
                    column_abbreviation, column_subscript=None,
                    column_units=None, column_is_log=False,
                    column_log_base=None):
    labels = {}
    has_column_subscript = column_subscript is not None
    if not has_column_subscript:
        column_subscript = {}
    if column_log_base is None :
        column_log_base = ""
    log_prefix = f" \\log_{{ {column_log_base} }} \\left(" if column_is_log else  ""
    log_suffix = f"\\right)" if column_is_log else ""
    for extraction_value in extraction_values:
        if not has_column_subscript:
            column_subscript[extraction_value] = extraction_value
        labels[f"{name}({extraction_variable}={extraction_value})"]= (
        f"${log_prefix}{column_abbreviation}_{{{column_subscript[extraction_value]}}}" +
            f"\ [{column_units}]" + f"{log_suffix}$")
    return labels 

_KNOWN_LABELS={"Mmax" : r"$M_{\max}$",
               **generate_labels("R", "M", [1.4, 1.6, 1.8, 2.0], "R",
                                 column_units="\mathrm{km}"),
               **generate_labels("pressurec2", "baryon_density",
                                 ["2.8e+14", "5.6e+14", "1.68e+15", "1.96e+15" ],
                                 "p", column_is_log=True,
                                 column_log_base = "10",
                                 column_subscript=dict(zip( ["2.8e+14", "5.6e+14", "1.68e+15", "1.96e+15" ], [1.0, 2.0, 6.0, 8.0])), column_units="\mathrm{g}/\mathrm{cm}^3"),
               **generate_labels("Lambda", "M", [1.4, 1.6, 1.8, 2.0], "Lambda",
                                 column_units="\mathrm{km}")

               }

def get_default_label(column_name):
    if column_name in _KNOWN_LABELS.keys():
        return _KNOWN_LABELS[column_name]
    else:
        return column_name

@dataclass
class PlottableColumn:
    name : str
    label : str
    plot_range : tuple[float] = None
    bandwidth : float = None
    true_value : float = None
    log_column : bool = False
    column_multiplier : float = None
    alternate_units : tuple[str, float, float] = None
    def get_plottable_data(self, samples):
        column_data  = np.array(samples[self.name])
        if self.log_column:
            column_data = np.log(column_data)
        if self.column_multiplier is not None:
            column_data*=self.column_multiplier
        return column_data
    def get_plottable_true_value(self):
        true_value = self.true_value
        if true_value is None:
            return None
        if self.log_column:
            true_value = np.log(true_value)
        if self.column_multiplier is not None:
            true_value *= self.column_multiplier
        return true_value
            
            

@dataclass
class PlottableSamples:
    """
    Arbitrary samples from some posterior distribution to be plotted
    """
    label: str
    samples: pd.DataFrame
    weight_columns_to_use: list[result.WeightColumn]
    max_samples:int =None
    color: str = None
    alpha: float = None
    linestyle: str = "-"
    linewidth: float = 2.0
    def get_data(self, plottable_columns:list[PlottableColumn]):
        data = pd.DataFrame()
        for column in plottable_columns:
            data[column.name] = column.get_plottable_data(self.samples)
        return data

@dataclass
class PlottableEoSSamples:
    label :str
    posterior: result.EoSPosterior
    weight_columns_to_use: list[result.WeightColumn]
    additional_properties : pd.DataFrame 
    color: str
    alpha: float = None
    linestyle: str = "-"
    linewidth: float = 2.0
    def guarantee_consistent_order(self):
        self.additional_properties = pd.merge(
            self.posterior.samples[posterior.eos_column],
            self.additional_properties,
            on=self.posterior.eos_column)
    def get_data(self, plottable_columns:list[PlottableColumn]):
        eos_file_columns = [self.posterior.eos_column]
        additional_columns = [self.posterior.eos_column]
        for column in plottable_columns:
            if column.name in self.posterior.samples.keys():
                eos_file_columns.append(column.name)
            elif column.name in self.additional_properties.keys():
                additional_columns.append(column.name)
            else:
                raise KeyError(f"Column {column.name} not found in either the "
                               "EoS or additional_properties files")
        # This is reasonable assuming that (1) most of the time the EoS
        # file will not contain all of the data we need and (2)
        # all the data we need is a small subset of the total data avaiable
        # in both files
        joint_data = pd.merge(self.posterior.samples[eos_file_columns],
                              self.additional_properties[additional_columns],
                              on=self.posterior.eos_column)
        joint_data.pop(self.posterior.eos_column)
        for column in plottable_columns:
            joint_data[column.name] = column.get_plottable_data(joint_data)
        return joint_data
    def get_prior(self, prior_prefix="prior", color=None,
                  weight_columns_to_use=[],
                  alpha=None, linestyle="--", linewidth=2.0):
        return PlottableEoSSamples(
            label=f"{prior_prefix}({self.label})",
            posterior=self.posterior,
            weight_columns_to_use=weight_columns_to_use,
            additional_properties=self.additional_properties,
            color=color,
            alpha=alpha,
            linestyle=linestyle,
            linewidth=linewidth
        )        
        
    

def get_property_columns(plottable_samples):
    """
    Get all of the columns that represent NS properties (not being eos
    or weight columns)
    """
    property_columns = []
    eos_posterior = plottable_samples.posterior
    additional_properties = plottable_samples.additional_properties
    for column_name in chain(eos_posterior.samples.keys(),
                             additional_properties.keys()):
        if column_name !=  eos_posterior.eos_column and (
                column_name not in [weight_column.name for weight_column in
                                    eos_posterior.weight_columns_available]):
            property_columns.append(
                PlottableColumn(column_name,
                                get_default_label(column_name)))
    return property_columns


def corner_samples(plottable_samples, use_universality=False,
                   columns_to_plot=None, legend=True, fig=None,
                   *args, **kwargs):
    """
    Make a corner plot of certain samples from some distribution
    """
    column_names = [column.name for column in columns_to_plot]
    bandwidths = [column.bandwidth for column in columns_to_plot]
    truths = [column.true_value for column in columns_to_plot]
    ranges = [column.plot_range for column in columns_to_plot]
    column_labels= [column.label for column in columns_to_plot]
    if use_universality:
        lines = [] # hack for getting legend in a controlled  way
        for samples in plottable_samples:
            # legend getting hack (0,1  and  0,1 don't mean anything)
            if legend:
                lines.append((Line2D([0,1], [0,1], linestyle=samples.linestyle,
                                     color=samples.color), samples.label))
                data = np.array(samples.get_data(columns_to_plot))
                weights_df = result.get_total_weight(samples.samples,
                                              samples.weight_columns_to_use)
                weights = weights_df["total_weight"]
            fig = uplot.kde_corner(data,
                               bandwidths=bandwidths,
                               truths=truths,
                               ranges=ranges,
                               weights=weights,
                               labels=column_labels,
                               fig=fig,
                               color=samples.color,
                               alpha=samples.alpha,
                               linewidth=samples.linewidth,
                               linestyle=samples.linestyle,
                               **kwargs)
        if legend:
            leg = fig.legend([line[0] for line in lines],[line[1] for line in lines],
                             loc="upper right", frameon=True,
                             fancybox=True, fontsize=18)
            for line in leg.get_lines():
                line.set_linewidth(samples.linewidth)
        return fig

    
def corner_eos(plottable_samples,  use_universality=True,
               columns_to_plot=None, legend=True, fig=None,  *args, **kwargs):
    """
    Make a corner plot of the data represented in the eos_posterior file 
    using the weight_columns in weight_columns_to_use
    
    if use_universality is true pass the extra arguments to universality, 
    otherwise pass the extra arguments to seaborn plotting utilities (
    the arguments will be different depending on the plotting utilities
    currently I'll use seaborn.pairplot)
    """
    if columns_to_plot is None:
        columns_to_plot = get_proprty_columns(plottable_samples[0])
    
    column_names = [column.name for column in columns_to_plot]

    
    bandwidths = [column.bandwidth for column in columns_to_plot]
    truths = [[column.get_plottable_true_value()] for column in columns_to_plot]
    ranges = [column.plot_range for column in columns_to_plot]
    column_labels= [column.label for column in columns_to_plot]
    alternate_column_units = [
        column.alternate_units for column in columns_to_plot]
    
    
    
    if use_universality:
        lines = [] # hack for getting legend in a controlled  way
        for samples in plottable_samples:


            # legend getting hack (0,1  and  0,1 don't mean anything)
            if legend:
                lines.append((Line2D([0,1], [0,1], linestyle=samples.linestyle,
                                     color=samples.color), samples.label))
            data = np.array(samples.get_data(columns_to_plot))
            weights_df = samples.posterior.get_total_weight(
                samples.weight_columns_to_use)
            weights = np.array(pd.merge(weights_df,
                                        samples.additional_properties,
                                        on=samples.posterior.eos_column)["total_weight"])
            fig = uplot.kde_corner(data,
                                   bandwidths=bandwidths,
                                   truths=truths,
                                   ranges=ranges,
                                   weights=weights,
                                   labels=column_labels,
                                   fig=fig,
                                   color=samples.color,
                                   alpha=samples.alpha,
                                   linewidth=samples.linewidth,
                                   linestyle=samples.linestyle,
                                   alternate_units=alternate_column_units,
                                   **kwargs)
        if legend:
            leg = fig.legend([line[0] for line in lines],[line[1] for line in lines],
                             loc="upper right", frameon=True,
                             fancybox=True, fontsize=18)
            for line in leg.get_lines():
                line.set_linewidth(samples.linewidth)
        return fig
