import universality
from universality.utils import (utils, io, units)
from universality import kde
from universality import stats
from universality import plot

import temperance.core.stats
import temperance.core.result as result

import numpy as np
import pandas as pd

class kde_function:
    """
    A wrapper around the universality kde for use in the 
    stats.DensityEstimate, 
    
    
    """
    def __init__(self, samples, weight_columns, sample_columns):
        """
        ========
        Requires
        ========
        samples : 
        dictionary like (such as dataframe), implements 
        samples[column_name] ->  indiviudal column samples,
        and samples.shape -> (num_samples, num_columns)
        weight columns:
        iterable of temperance.core.result.WeightColumns, 
        weights to use in kde
        sample_columns:
        iterable of temperance.core.stats.SamplesColumn,
        columns to build the kde over
        """
        self.data = np.array(samples[[column.name
                                      for column in sample_columns]])
        self.variances = np.array([column.bandwidth
                                   for column in sample_columns] )
        self.weights = result.get_total_weight(samples, weight_columns,
                                               weights_as_array=True)
        self.samples = samples
        self.weight_columns = weight_columns
        self.sample_columns = sample_columns
    def samples_to_input(self, samples):
        return np.transpose(np.array([samples[column] for column in samples]))
        
    def __call__(self, samples):
        #print(kde.vects2flatgrid(*self.samples_to_input(samples)))
        print("samples_to_input", self.samples_to_input(samples).shape)
        return kde.logkde(
            self.samples_to_input(samples),
            self.data,
            self.variances,
            weights=self.weights,
    )
    def argmax(self, samples):
        return stats.logkde2argmax(self.samples_to_input(samples), self.logkde)
    def information(self, samples):
        return stats.logkde2entropy(self.samples_to_input(samples), self.logkde)
    def entropy(self, samples):
        stats.logkde2entropy(self.samples_to_input(samples), self.logkde)

