import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import universality
from universality import stats
import bilby
import seaborn as sns
from dataclasses import dataclass
import warnings
import copy 

# Utilities
# to be factored out
def get_property_list(class_list, target_property):
    map(class_list, lambda elt : elt.target_property)

_default_uniform_mass_data = {"low": 1.0, "high": 2.5,
                              "mass_1_name": "m1",
                              "mass_2_name": "m2"}    
def uniform_mass_samples(
        N=1, seed=None,
        low=_default_uniform_mass_data["low"],
        high=_default_uniform_mass_data["high"],
        mass_1_name=_default_uniform_mass_data["mass_1_name"],
        mass_2_name=_default_uniform_mass_data["mass_2_name"]
):
    rng = np.random.default_rng(seed)
    samples = rng.uniform(low=low, high=high, size=(N,2))
    samples.sort(axis=1) # Sorted has lower values first
    # So we flip them to have mass_1 > mass_2
    pd.DataFrame({mass_1_name:samples[:,1], mass_2_name: samples[:, 0]})

def uniform_mass_pdf(
        sample,            
        low=_default_uniform_mass_data["low"],
        high=_default_uniform_mass_data["high"],
        mass_1_name=_default_uniform_mass_data["mass_1_name"],
        mass_2_name=_default_uniform_mass_data["mass_2_name"],
        no_range_check = False):
    """
    We only have support on high > m_1 > m_2 > low,
    and on this range it is uniform
    if no_range_check is True assume the sample is in the region of support
    """
    if (not no_range_check):
        return np.where(np.logical_and(np.logical_and(high >= sample[mass_1_name],  sample[mass_2_name] >= low), sample[mass_1_name]>=  sample[mass_2_name]), 
         1/(high - low)**2 * 2, 0.0)
    else:
        return 1/(high - low)**2 * 2
    
        

class Prior:
    """
    A function representing a prior, 
    it should be able to sample a prior
    and compute the pdf at a point
    Default is a uniform prior on (1, 3) 
    for component masses, with the added restriction that
    mass_1 is greater than mass_2

    """
    def __init__(self, name="original", pdf=uniform_mass_pdf,
                 sample=uniform_mass_samples):
        self.name = name
        self.pdf = pdf
        self.sample = sample 
        
    def __call__(self, sample, *args, **kwargs):
        return self.pdf(sample, *args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

def marginalize_over_samples(data,
                             column_to_compute, compute_column_is_log,
                             track_variance=True, prior=Prior()):
    """
    Marginalize over the rows (which will in general represent samples)
    of data in order to compute the marginalized value of the column
    `column_to_compute` under the prior `prior`

    Mostly taken from universality (it may be more convenient to use
    universality.utils.marginilize)
    """
    count = data.shape[0]
    prior_weights = prior(data)
    if compute_column_is_log:
        lmw = np.log((prior_weights*np.exp(np.array(data[column_to_compute]))).sum())
        if track_variance:
            lmw2 = np.log(
                (prior_weights*np.exp(2*np.array(data[column_to_compute]))).sum())
        else :
            lmw2 = np.nan
        neff = stats.neff(prior_weights)
        return lmw, lmw2, count, neff, prior_weights 
    else:
        mw = np.sum(prior_weights*data[column_to_compute])
        if track_variance:
            mw2 = prior_weights * np.array(data[column_to_compute])**2
            
        return mw, mw2, count, neff, prior_weights
    # TODO : make this precision-loss proof


    
class MarginalizationResult:
    def __init__(self, marg_samples, logvar, counts, neffs, prior=Prior(),
                 eos_column="eos"):
        self.marg_samples = marg_samples
        self.logvar = logvar
        self.counts = counts
        self.neffs = neffs
        self.prior = prior
        self.eos_column = eos_column
        

class InferenceResult:
    """
    The result of a hierarchical likelihood calculation, also 
    contains utilites to marginalize over the computed likelihood
    values to produce a likelihood for each EoS
    """
    def __init__(self, samples, marg_samples=None, eos_column="eos",
                 eos_directory=None, samples_path=None, weight_columns=None,
                 default_marginalization_prior=Prior(),
                 default_weight_column_name="logweight"):
        self.samples = samples
        self.marg_samples = marg_samples
        self.eos_column = eos_column
        
        # Need  to think about this more
        self.prior = default_marginalization_prior
        self.prior_evaluations = {}
    
        self.weight_columns_available = get_weight_columns(samples, weight_columns)
        self.default_weight_column = (
            self.weight_columns_available[
                list(map(lambda column:column.name,
                     self.weight_columns_available)).index(
                         default_weight_column_name)]
        )
        
        self.weights = np.exp(np.array(
            samples[self.default_weight_column.name]))
        self.posterior= self.weights/np.sum(self.weights)
        
        
        self.eos_directory = eos_directory
        self.eos_indices = np.unique(np.array(self.samples[self.eos_column]))
        
        self.auxillary_likelihood = {}
        
        self.marginalization_results = []

        # get the result with the default marginalization
        if self.marg_samples is None and default_marginalization_prior is not None:
            default_marginalization, prior_evaluations = (
                self.get_marginalization(prior=default_marginalization_prior))
            self.marginalization_results.append(default_marginalization)
            self.prior_evaluations.update(prior_evaluations)
            
    
    
    def to_h5(self):
        raise NotImplemented
    
    @staticmethod
    def from_h5():
        raise NotImplemented 
    
    def get_marginalization(self, weight_column=None,
                            prior=Prior()):
        """
        Marginalize over the likelihood for each eos to construct a single
        likelihood for each unique eos, stored as a logweight
        """
        if weight_column is None:
            weight_column = self.default_weight_column
        posteriors = np.zeros_like(self.eos_indices)
        squares = np.zeros_like(self.eos_indices)
        counts = np.zeros_like(self.eos_indices, dtype=int)
        neffs = np.zeros_like(self.eos_indices, dtype=float)
        prior_evaluations = {f"Prior_{prior.name}" :
                             np.ndarray(self.samples.shape[0])}
        for index, eos in enumerate(self.eos_indices):
            eos_data = self.get_eos_data(eos)
            (posteriors[index], squares[index], counts[index], neffs[index],
            prior_evaluations[f"Prior_{prior.name}"][eos_data.index]) = (
                marginalize_over_samples(
                    eos_data,
                    prior=prior,
                    column_to_compute=weight_column.name,
                    compute_column_is_log=weight_column.is_log
            )
                                                                             )
            
        marg_samples = pd.DataFrame({"eos" : self.eos_indices,
                                     "logmargweight": posteriors})
        logvar = squares + np.log(1. - np.exp(2*posteriors -
                                              squares - np.log(counts)))
        counts = counts
        neffs = neffs
        return MarginalizationResult(marg_samples=marg_samples, logvar=logvar, 
                                     counts=counts, neffs=neffs, prior=prior,
                                     eos_column=self.eos_column), prior_evaluations
    def get_eos_data(self, eos, columns=None):
        """
        Get the data subset just corresponding to the EoS eos
        """
        if columns is None:
            columns = self.samples.columns
        return self.samples[self.samples[self.eos_column] == eos][columns]

    def get_total_weights(self, weight_columns):
        return get_total_weight(self.samples, weight_columns)
        
    def sample(self, columns=None, size=1, posterior=None, weight_columns=None,
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
        posterior = (self.get_total_weight(weight_columns)
                     if weight_columns is not None
                     else posterior)
        if columns is not None:
            return np.array(self.samples.sample(n=size,
                                                weights=posterior,
                                                **kwargs)[column]) 
        return self.samples.sample(n=size,
                                   weights=posterior,
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
    def get_inverse(self):
        """
        Get the identical weight column, but inverted, i.e. representing the 
        complementary weight.  
        """
        return WeightColumn(self.name, self.is_log, not(self.is_inverted))
def get_weight_columns(samples, weight_columns):
    """
    return the weight columns identified in this posterior 
    (Note that whether they are logged and inverted has to
    be guessed from the column name, so it is wise to check
    that the column properties have been identified correctly)
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

def get_column_weight(samples:pd.DataFrame, weight_column:WeightColumn):
    """
    Get the weight corresponding to a certain weight column by applying
    the correct combinations of exponentiations and inversion,
    Returns an array of weights in the order of the EoSs in self.samples
    """
    def invert_weight_column(weights):
        if np.any(weights==0):
            # This is a hack, if the weight column has zero weights,
            #inverting them will make them singularly relevant,
            # i.e. they get weight 1 and everything else gets weight
            # 0
            return np.array(weights == 0, dtype=float)
        else:
            return 1/weights
    data = np.array(samples[weight_column.name])
    weights = np.exp(data) if weight_column.is_log else data
    weights = invert_weight_column(weights) if weight_column.is_inverted else weights
    return weights
def get_column_logweight(samples:pd.DataFrame, weight_column:WeightColumn):
    """
    Get the logwweight corresponding to a certain weight column by applying
    the correct combinations of logs and inversions,
    Returns an array of logweights in the order of the EoSs in self.samples
    """
    def invert_logweight(logweights):
        # we avoid infinite weights
        if np.any(logweights==-np.inf):
            return np.log(np.array((logweights==-np.inf), dtype=float)
                                )
        else:
            return -logweights
    data = np.array(samples[weight_column.name])
    logweights = np.log(data) if not weight_column.is_log else data
    logweights = invert_logweight(logweights) if weight_column.is_inverted else logweights
    return logweights



def get_total_weight(samples:pd.DataFrame,
                     weight_columns:list[WeightColumn],index_columns=None,
                     weights_as_array=False):
    """
    get the total weights as a dataframe,
    if desired, copy columns into the weights dataframe 
    (helpful for indexing with, for example, the EoS)
    """
    total_logweights = np.zeros(samples.shape[0])
    for weight_column in weight_columns:
        total_logweights += get_column_logweight(samples, weight_column)
        
    if index_columns is not None:
        weights_df = samples[index_columns].copy()
    else:
        weights_df  = pd.DataFrame()
    if weights_as_array:
        return np.transpose(np.exp(total_logweights))
    weights_df["total_weight"] = np.transpose(np.exp(total_logweights))
    return weights_df

def get_logical_or_weight_column(samples:pd.DataFrame, weight_columns:list[WeightColumn], result_name=None, **kwargs):
    """
    Instead of getting the and of the weights (that is the product of the weights), get 
    the logical or of the columns, so that if any is true, the  
    """
    if result_name is None:
        result_name = "weight_{weight_column_1.name.split['weight_'][-1]}_or_{weight_column_2.name.split('weight')[-1]}"
    for weight_column in weight_columns:
        weight_column.is_inverted=not(weight_column.is_inverted)
    
    output_weights = get_total_weight(samples, weight_columns, **kwargs)
    
    output_weights[result_name] = 1.0 - np.array(output_weights["total_weight"]) # not eveything
    output_weights.pop("total_weight")
    
    result_weight_column = WeightColumn(name=result_name, is_log=False, is_inverted=False)
    return output_weights, result_weight_column
        
        


class EoSPosterior:
    """
    This class is a convenience for compatibility with current code
    """
    def __init__(self, samples, eos_column="eos",  weight_columns=None, label="posterior", eos_prior_weights=None):
        self.samples = samples
        self.num_samples = samples.shape[0]
        self.weight_columns_available = get_weight_columns(samples, weight_columns)
        self.eos_column=eos_column
        self.conditions = None
        self.label = label
        if eos_prior_weights is None:
            self.eos_prior_weights=np.ones(self.num_samples)
        else:
            self.eos_prior_weights = eos_prior_weights
        

    @staticmethod
    def from_csv(samples_path, eos_column="eos", weight_columns=None, **kwargs):
        """
        Construct an EoSPosterior object from a csv containing samples
        """
        samples = pd.read_csv(samples_path)
        return EoSPosterior(samples=samples, eos_column=eos_column,
                            weight_columns=weight_columns, **kwargs)

    @staticmethod
    def from_marginalized_likelihood(marginalized_likelihood, **kwargs):
        """
        Construct an EoSPosterior from a marginalized likelihood
        """
        # This will extract the weight columns automatically;
        # this could be made more robust by setting them according
        # to their names (which are predicatable probably?)
        return EoSPosterior(samples=marginalized_likelihood.marg_samples,
                            eos_column=marginalized_likelihood.eos_column,
                            weight_columns = None, **kwargs)
    def merge(self, other, explicit_copy_columns=[]):
        """
        Combine sevaral EoSPosterors into a single posterior, by copying the 
        weight columns from one EoSPosterior into another

        Repeated columns are given a suffix based on their labels
        """

        # This is a hack, if somebody invents a better way to join w/o copying 
        # RHS duplicate columns please fix this

        other_names = list(map(lambda weight_column : weight_column.name,
                               other.weight_columns_available))
        to_copy = [self.eos_column] + other_names + list(explicit_copy_columns)
        return  EoSPosterior(
            self.samples.join(other.samples[to_copy], on=self.eos_column,
                              how="inner",
                              lsuffix=f"{other.label}"),
            eos_column=self.eos_column, weight_columns=None)
        
    def get_column_weight(self, weight_column):
        """
        Get the weight corresponding to a certain weight column by applying 
        the correct combinations of exponentiations and inversion, 
        Returns an array of weights in the order of the EoSs in self.samples
        """
        if weight_column not in self.weight_columns_available:
            if weight_column.name in list(map(self.weight_columns_available, lambda column : column.name)):
               warnings.warn(f"Using a weight_column which is present in the EoSPosterior, "
               "but with using logweight tag {weight_column.is_log} and invert tag "
               "{weight_column.inverted}, at least one of which is inconsistent with the posterior "
               "object", category=RuntimeWarning)
            else:
                raise f"Weight column {weight_column.name} is not present in eos posterior!"
        
            return get_column_weight(self.samples, weight_column)

    def get_column_logweight(self, weight_column):
        if weight_column not in self.weight_columns_available:
            if weight_column.name in list(map(lambda column : column.name,
                                              self.weight_columns_available
                                          )):
               warnings.warn(f"Using a weight_column which is present in the EoSPosterior, "
               "but with using logweight tag {weight_column.is_log} and invert tag "
               "{weight_column.inverted}, at least one of which is inconsistent with the posterior "
               "object", category=RuntimeWarning)
            else:
                raise ValueError(f"Weight column {weight_column.name} is not present in eos posterior!,  available weight columns are {self.weight_columns_available}")
        
        return get_column_logweight(self.samples, weight_column)
    
    def get_total_weight(self, weight_columns_to_use):
        """
        get the total weights as a dataframe,
        somewhat different than the free function above
        """
        total_logweights = np.zeros(self.samples.shape[0])
        for weight_column in weight_columns_to_use:
            column_logweight = self.get_column_logweight(weight_column)
            total_logweights += column_logweight

        weights_df = self.samples[[self.eos_column]].copy()
        weights_df["total_weight"] = np.transpose(np.exp(total_logweights))
        return weights_df

    def sample(self, columns=None, size=1, posterior=None, weight_columns=None,
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
        posterior = (np.array(
            self.get_total_weight(weight_columns)["total_weight"])
                     if weight_columns is not None
                     else posterior)
        if columns is not None:
            return self.samples.sample(n=size,
                                       weights=posterior,
                                       axis=0,
                                       ignore_index=True,
                                       **kwargs)[columns] 
        return self.samples.sample(n=size,
                                   weights=posterior,
                                   axis=0,
                                   ignore_index=True,
                                   **kwargs)

    def condition(self, evaluated_criteria, weight_is_log = False,
                  include_negation=False, weight_key=None):
        """
        Add additional weight columns corresponding to this criteria for these EoSs,
        the criteria should have the form (criteria_name:string, 
        evaluated_criteria: dataframe with 2 columns, eos_column, 
        and some other column with weight
        corresponding to the criteria) 
        """
        if weight_key is None:
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
        if include_negation:
            inv_weight_column_name = f"logweight_{weight_key}"
            self.weight_columns_available.append(WeightColumn(weight_column_name, is_log=True, is_inverted=True))
    def estimate_evidence(self, weight_columns_to_use=[], prior=None,
                          prior_weight_columns=None):
        """
        Estimate the evidence associated with this EOS Posterior,
        int P(d|e)pi(e)de, with e the eos, 
        """
        weights = self.get_total_weight(weight_columns_to_use)["total_weight"]
        if prior is None:
            if prior_weight_columns is None:
                prior = self.eos_prior_weights
            else:
                prior = self.get_total_weight(prior_weight_columns)["total_weight"]
        print(max(weights))
        prior /= sum(prior)
        count = len(weights)
        squares = weights**2
        # TODO : find a way to do this that is computationally efficient and
        # doesn't look weird
        evidence = np.sum(weights * prior)
        var_evidence = np.sum(prior**2)*np.sum((squares - evidence**2) * prior)
        return evidence, var_evidence

    def maximum_likelihood(self, weight_columns_to_use, argmax=False):
        """
        Compute the maximum likelihood of the posterior given the weight 
        columns provided.
        
        If argmax is true, return the index of the maximum rather than the maximum itself.
        """
        max_function = np.argmax if argmax else np.max
        return max_function(self.get_total_weight(weight_columns_to_use)["total_weight"])

    def compute_neff(self, weight_columns_to_use=[], threshold=0.0):
        """
        Compute the number of effective samples in the posterior using 
        weight_columns_to_use, do not include samples which have a weight of 
        threshold relative to the maxiumum weight in the posterior.
        """
        total_weight = np.array(self.get_total_weight(weight_columns_to_use)["total_weight"])
        nonnegligable = np.where(total_weight / np.sum(total_weight) > threshold)[0]
        print(total_weight[nonnegligable])
        return stats.neff(total_weight[nonnegligable])

    def add_weight_column(self, weight_column, indexed_weights):
        """
        Modify the EoSPosterior object to add a new weight
        Args:
           self: EoSPosterior
           weight_column: WeightColumn
           indexed_weights: pf.DataFrame, or indexable with columns self.eos_column,
           and an additional column with weight_column.name

        Returns:
        None

        """
        
        self.samples = pd.merge(self.samples, indexed_weights, on=self.eos_column)
        self.weight_columns_available.append(weight_column)


    
    def add_property(self, eos_prior_data, add_to=None, **kwargs):
        """
        Extract a propert for the EoS prior process which was used to generate
        this posterior, the user has to know the details of the process as specified
        in an eos_prior_data. If add_to is not None, use this dataframe as the 
        destination for the collated samples.  Otherwise use the EoS posterior itself.
        
        **kwargs will be passed to the eos_prior_data.get_property function
        which does all of the heavy lifting.  

        Note that the destination add_to must have the same EoSs in the same order as
        the eos posterior for this call to work, no merge is done on keys.  

        Note this doesn't return the thing it's adding to,
        (maybe that should change)
        """
        new_samples =  eos_prior_data.get_property(self.samples[self.eos_column], **kwargs)
        if add_to is not None:
            for key in new_samples.keys():
                add_to[key] = new_samples[key]
        else:
            for key in new_samples.keys():
                self.samples[key] = new_samples[key]
                
    def add_logical_or_weight_column(self, weight_columns, result_name=None):
        result_weight, result_weight_column = get_logical_or_weight_column(
            samples=self.samples, weight_columns=weight_columns, result_name=result_name,
            index_columns=self.eos_column)

        print(result_weight)
        print(result_weight[result_name])
        self.weight_columns_available.append(result_weight_column)
        
    def add_logical_or_weight_column(self, weight_columns, result_name=None):
        if result_name is None:
            result_name = "weight_{weight_column_1.name.split['weight_'][-1]}_or_{weight_column_2.name.split('weight')[-1]}"
        to_negate_weights = copy.deepcopy(weight_columns)
        for weight_column in to_negate_weights :
            weight_column.is_inverted=not(weight_column.is_inverted)

        output_weights = self.get_total_weight(to_negate_weights)
        
        output_weights[result_name] = 1.0 - np.array(output_weights["total_weight"]) # not eveything
        output_weights.pop("total_weight")

        result_weight_column = WeightColumn(name=result_name, is_log=False, is_inverted=False)
        self.samples[result_name]= output_weights[result_name]
        self.weight_columns_available.append(result_weight_column)

