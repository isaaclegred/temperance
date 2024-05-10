import numpy as np
import pandas as pd
import universality
from universality import stats
from dataclasses import dataclass
import scipy.interpolate as interpolate

import temperance.core.result as result 
from temperance.core.result import EoSPosterior
from temperance.sampling.eos_prior import EoSPriorSet

from temperance.plotting.corner import  PlottableSamples


@dataclass
class SamplesColumn:
    name : str
    label : str
    plot_range : tuple[float] = None
    bandwidth : float = None
    true_value : float = None
    log_column : bool = False
    column_multiplier : float = None
    def get_sample_data(self, samples):
        column_data  = np.array(samples[self.name])
        if self.log_column:
            column_data = np.log(column_data)
        if self.column_multiplier is not None:
            column_data*=self.column_multiplier
        return column_data
    def get_true_value(self):
        true_value = self.true_value
        if self.log_column:
            true_value = np.log(true_value)
        if self.column_multiplier is not None:
            true_value *= self.column_multiplier
        return true_value

class Statistic1D:
    def __init__(self, name, operation):
        """
        Operation should take 1-D array of samples and weights
        """
        self.name = name
        self.operation=operation
    def __call__(self, samples, weight_columns,
                 sample_columns,
                 *args, **kwargs):
        weights = result.get_total_weight(samples, weight_columns)
        
        return {self.name : self.operation(samples, weights,
                                           *args, **kwargs)}

def _quantiles_operation(sample_array, weights,
                         quantiles_desired=[.05, .5, .95]):
    order = np.argsort(sample_array)
    invcdf = interpolate.interp1d(np.cumsum(weights[order]),
                                  sample_array[order])
    output = {}
    outdata = invcdf(np.array(quantiles_desired))
    return dict(zip(quantiles_desired, outdata))
    
Quantiles = Statistic1D("Quantiles",
                        operation=_quantiles_operation)
    

class DensityEstimate:
    def __init__(self, samples, weight_columns, sample_columns,
                 density_function,
                 *args, **kwargs ):
        self.samples = samples
        self.weight_columns = weight_columns
        self.sample_columns = sample_columns
        self.density_function = density_function(self.samples,
                                                 self.weight_columns,
                                                 self.sample_columns)
    def __call__(self, samples):
        return self.density_function(samples)
    def entropy(self, samples):
        return self.density_function.entropy(samples)
    def information(self, samples):
        return self.density_function.information(samples)
    def argmax(self, samples):
        return self.density_function.argmax(samples)

    
        
        
        
