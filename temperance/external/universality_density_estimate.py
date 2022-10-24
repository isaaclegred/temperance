import universality
from universality.utils import (utils, io, units)
from universality import kde
from universality import stats
from universality import plot

import temperance.core.stats
import temperance.core.result

class kde_function:
    """
    A wrapper around the universality kde for use in the 
    stats.DensityEstimate, 
    
    
    """
    def __init__(samples, weight_columns, sample_columns):
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
        self.data = np.array(samples[column.name
                                     for column in sample_columns])
        self.variances = np.array([column.bandwidth
                                   for column in sample_columns] )
        self.weights = result.get_total_weight(samples, weight_columns,
                                               weights_as_array=True)
        self.samples = samples
        self.weight_columns = weight_columns
        self.sample_columns = sample_columns


    )
    def samples_to_input(samples):
        return [np.array(column) for column in (*np.array(samples).tolist())]
    def __call__(samples):
        return kde.logkde(
            kde.vects2flatgrid(*samples_to_input(samples)),
            self.data,
            self.variances,
            weights=self.weights,
    )
    def argmax(samples):
        return stats.logkde2argmax(samples_to_input(samples), logkde)
    def information(samples):
        return stats.logkde2entropy(samples_to_input(samples), logkde)
    def entropy(samples):
        stats.logkde2entropy(samples_to_input(samples), logkde)

