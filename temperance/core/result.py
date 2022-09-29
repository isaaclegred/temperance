import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import universality
from universality import stats
import bilby
import seaborn as sns
from dataclasses import dataclass
import warnings


# Utilities
# to be factored out
def get_property_list(class_list, target_property):
    map(class_list, lambda elt : elt.target_property)


class Prior:
    def __init__(self, name="original", function=lambda sample : 1):
        self.name = name
        self.function = function
    def __call__(self, sample):
        return self.function(sample)
        

class MarginalizationResult:
    def __init__(self, marg_samples, logvar, counts, neffs, prior=Prior()):
        self.marg_samples = marg_samples
        self.logvar = logvar
        self.counts = counts
        self.neffs = neffs
        self.prior = prior

class InferenceResult:
    """
    The result of a hierarchical likelihood calculation, also 
    contains utilites to marginalize over the computed likelihood
    values to produce a likelihood for each EoS
    """
    def __init__(self, samples, marg_samples=None, eos_column="eos",
                 eos_directory=None, samples_path=None, default_marginalization_prior=Prior()):
        self.samples = samples
        self.marg_samples = marg_samples
        self.eos_column = eos_column
        
        # Need  to think about this more
        self.prior = default_marginalization_prior
        
        self.weights =np.exp(np.array(samples["logweight"]))
        self.posterior= self.weights/np.sum(self.weights)
        
        
        self.eos_directory = eos_directory
        self.eos_indices = np.unique(np.array(self.samples[self.eos_column]))
        
        self.auxillary_likelihood = {}
        
        self.marginalization_results = []
        
        # get the result with the default marginalization
        if self.marg_samples is None and default_marginalization_prior is not None:
            self.marginalization_results.append(self.get_marginalization(prior=default_marginalization_prior))
            
    
    
    def to_h5(self):
        raise NotImplemented
    
    @staticmethod
    def from_h5():
        raise NotImplemented 
    
    def get_marginalization(self, prior=Prior()):
        """
        Get all of the marginalization quantites associated with 
        the EoS represented in samples, i.e. construct a logweight
        for each unique eos
        """
        
        posteriors = np.zeros_like(self.eos_indices)
        squares = np.zeros_like(self.eos_indices)
        counts = np.zeros_like(self.eos_indices, dtype=int)
        neffs = np.zeros_like(self.eos_indices, dtype=float)
        for index, eos in enumerate(self.eos_indices):
            posteriors[index], squares[index], counts[index], neffs[index] = (
                self.marginalize_over_samples(eos,
                                              prior=prior)
                                                                             )
            
        marg_samples = pd.DataFrame({"eos" : self.eos_indices,
                                     "logweight_total": posteriors})
        logvar = squares + np.log(1. - np.exp(2*posteriors -
                                              squares - np.log(counts)))
        counts = counts
        neffs = neffs
        return MarginalizationResult(marg_samples=marg_samples, logvar=logvar, 
                                     counts=counts, neffs=neffs, prior=prior)
    def get_eos_data(self, eos, column):
        """
        Get the data subset just corresponding to the EoS eos
        """
        return self.samples[self.samples[self.eos_column] == eos][ column]
    def marginalize_over_samples(self, eos, weight_column="logweight", 
                                 track_variance=True, prior=Prior()):
        eos_data = self.get_eos_data(eos, weight_column)
        lmw = np.log((self.prior(eos)*np.exp(np.array(eos_data))).sum())
        lmw2 = np.log((self.prior(eos)*np.exp(2*np.array(eos_data))).sum())
        count = len(eos_data)
        neff = stats.neff(np.exp(np.array(eos_data)))
        return lmw, lmw2, count, neff
    
    def sample(self, column=None, size=1, posterior=None, weight_columns=None,
               **kwargs):
        """
        Sample from the posterior, either a known posterior
        on the eos's or a set of weight_columns can be used
        to induce weights for sampling
        kwargs are passed to pandas.DataFrame.sample
        """
        if posterior is not None and weight_columns is not None:
            raise ValueError("only one of posterior and weight_columns can"
                             "be specified")
        # get the weights from the weight_columns
        posterior = (self.get_weight(weight_columns)
                     if weight_columns is not None
                     else posterior)
        if column is not None:
            return np.array(self.samples.sample(n=size,
                                                weights=posterior,
                                                **kwargs)[column]) 
        return self.samples.sample(n=size,
                                   Weights=posterior,
                                   **kwargs)
    def corner(self, columns, num_samples_to_use=1000, *args, **kwargs):
        reduced_samples = self.samples.sample(n=num_samples_to_use,
                                              weights =self.posterior)
        sns.pairplot(self.samples, vars=columns, corner=True,  kind="kde",
                     plot_kws={"weights":self.posterior, **kwargs},
                     diag_kws={"weights":self.posterior, **kwargs})
    def get_auxillary_weighted_posterior(self, auxillary_likelihoods):
        for likelihood_name in auxillary_likelihoods:
            likelihood = self.auxillary_likelihood[likelihood_name]
            for eos in likelihood.keys():
                eos_data = np.array(self.get_eos_data(eos, "logweight"))
                eos_data += likelihood["eos"]
        
    
    def __getitem__(self, key):
        return self.samples[key]
    def __setitem__(self, key, value):
        self.samples[key] = value
        self.consistent_marginalization = False

@dataclass
class WeightColumn:
    name: str
    is_log: bool = True
    is_inverted: bool = False

def get_weight_columns(samples, weight_columns):
    """
    Get either return the weight columns
    """
    if weight_columns is not None:
        iterator = iter(weight_columns) # Must be iterable
        if isinstance(weight_columns[0], WeightColumn, False):
            # User has already phrased it in terms of weight
            # columns
            return weight_columns
        if isinstance(weight_column[0], tuple):
            # a list of tuples of name - is_log pairs
            return [WeightColumn(column_name, is_log, False) for column_name,
                    is_log in weight_columsn]
    weight_columns = []
    for column_name in samples.keys():
        if "log" in column_name and "weight" in column_name:
            weight_columns.append(WeightColumn(column_name, True, False))
        elif "weight" in column_name:
            weight_columns.append(WeightColumn(column_name, False, False))
    return weight_columns


            
class EoSPosterior:
    """
    This class is a convenience for compatibility with current code
    """
    def __init__(self, samples, eos_column="eos",  weight_columns=None):
        self.samples = samples
        self.num_samples = samples.shape[0]
        self.weight_columns_available = get_weight_columns(samples, weight_columns)
        self.eos_column=eos_column
        self.conditions = None

    @staticmethod
    def from_csv(samples_path, eos_column="eos", weight_columns=None):
        """
        Construct an EoSPosterior object from a csv containing samples
        """
        samples = pd.read_csv(samples_path)
        return EoSPosterior(samples=samples, eos_column=eos_column,
                            weight_columns=weight_columns)
    def get_column_weight(self, weight_column):
        if weight_column not in self.weight_columns_available:
            if weight_column.name in list(map(self.weight_columns_available), lambda column : column.name):
               warnings.warn(f"Using a weight_column which is present in the EoSPosterior, "
               "but with using logweight tag {weight_column.is_log} and invert tag "
               "{weight_column.inverted}, at least one of which is inconsistent with the posterior "
               "object", category=RuntimeWarning)
            else:
                raise f"Weight column {weight_column.name} is not present in eos posterior!"
        
        data = np.array(self.samples[weight_column.name])
        weights = np.exp(data) if weight_column.is_log else data
        weights = 1/weights if weight_column.is_inverted else weights
        return weights

    def get_column_logweight(self, weight_column):
        if weight_column not in self.weight_columns_available:
            if weight_column.name in list(map(self.weight_columns_available), lambda column : column.name):
               warnings.warn(f"Using a weight_column which is present in the EoSPosterior, "
               "but with using logweight tag {weight_column.is_log} and invert tag "
               "{weight_column.inverted}, at least one of which is inconsistent with the posterior "
               "object", category=RuntimeWarning)
            else:
                raise f"Weight column {weight_column.name} is not present in eos posterior!"
        
        data = np.array(self.samples[weight_column.name])
        logweights = np.log(data) if not weight_column.is_log else data
        logweights = -logweights if weight_column.is_inverted else logweights
        return logweights
    
    def get_total_weight(self, weight_columns_to_use):
        """
        get the total weights as a dataframe 
        """
        total_logweights = np.ones(self.samples.shape[0])
        for weight_column in weight_columns_to_use:
            total_logweights += self.get_column_logweight(weight_column)

        weights_df = self.samples[[self.eos_column]].copy()
        weights_df["total_weight"] = np.transpose(np.exp(total_logweights))
        return weights_df

    def condition(self, evaluated_criteria, weight_is_log = False):
        """
        Add additional weight columns corresponding to this criteria for these EoSs,
        the criteria should have the form (criteria_name:string, 
        evaluated_criteria: dataframe with 2 columns, eos_column, and some other column with weight
        corresponding to the criteria) 
        """
        weight_key = [column for column in evaluated_criteria.keys() if column != self.eos_column][0]
        if not weight_is_log:
            evaluated_criteria[weight_key] = np.log(evaluated_criteria[weight_key])
            
        if self.conditions is None:
            self.conditions = {weight_key : evaluated_criteria}
        else:
            self.conditions[weight_key] = evaluated_criteria
        weight_column_name = f"logweight_{weight_key}"
        self.samples = self.samples.merge(evaluated_criteria, on=self.eos_column)
        self.samples.rename(columns={weight_key: weight_column_name}, inplace=True)
        self.weight_columns_available.append(
            WeightColumn(weight_column_name, is_log=True, is_inverted=False)
        )
