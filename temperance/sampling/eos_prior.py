import numpy as np
import pandas as pd
import os
import h5py

from dataclasses import dataclass


from universality.utils import io
from universality.properties import samples

# This is the eos set we use most often
DEFAULT_EOS_DATA = {"eos_dir":"/home/philippe.landry/nseos/eos/gp/mrgagn",
                    "eos_column" : "eos",
                    "eos_per_dir": 1000,
                    "eos_basename" : 'eos-draw-%(draw)06d.csv',
                    "macro_basename": 'macro-draw-%(draw)06d.csv',
                    "branches_dir" : "/home/isaac.legred/local_mrgagn_big_with_cs2c2",
                    "branches_basename" : ('macro-draw-%(draw)06d-branches.csv',
                                           "rhoc",
                                           "start_baryon_density",
                                           "end_baryon_density")}

@dataclass
class Extraction:
    """
    Get a particular EoS common quantity from a prior (i.e. extract per eos)

    If dynamic is not None, we use this column to decide extraction values 
    """
    strategy : str
    macro : bool
    independent_variable : str = None
    values : list[float] = None
    dynamic : list[str] = None
    branched : bool = True
    selection_rule :str = "random"
_known_extraction_strategies = ["extremize", "interpolation"]

@dataclass
class EoSPriorSet:
    """
    This class encapsulates the current way we store samples from
    an EoS Process.  It's currently agnostic to the sampling method,
    e.g. parametric, nonparametric, whatever, but it prefers the 
    standard hierarchical directory structure we use to store these
    draws
    """
    eos_dir : str
    eos_column : str
    eos_per_dir : int
    macro_dir : str 
    eos_path_template : str = 'eos-draw-%(draw)06d.csv'
    macro_path_template: str = 'macro-draw-%(draw)06d.csv'
    branches_data: tuple[str] = (
        '/home/isaac.legred/local_mrgagn_big_with_cs2c2/DRAWmod1000-%(moddraw)06d/macro-draw-%(draw)06d-branches.csv',
        "rhoc",
        "start_baryon_density", "end_baryon_density")

    
    @staticmethod
    def get_default():
        return EoSPriorSet(eos_dir=DEFAULT_EOS_DATA["eos_dir"],
                           eos_column=DEFAULT_EOS_DATA["eos_column"],
                           eos_per_dir=DEFAULT_EOS_DATA["eos_per_dir"],
                           macro_dir=DEFAULT_EOS_DATA["eos_dir"]
                           
        )
    def get_eos_path(self, eos_number, subdir=None, explicit_path=None):
        if subdir == None:
            subdir = f"DRAWmod{self.eos_per_dir}-{eos_number//self.eos_per_dir:06d}"
        if explicit_path is None:
            local_path = self.eos_path_template%{"draw" : eos_number}
        else:
            local_path = explicit_path%{"draw": eos_number}
        return os.path.join(self.eos_dir, subdir,
                            local_path)
    def get_macro_path(self, eos_number, subdir=None):
        if subdir is None:
            subdir = f"DRAWmod{self.eos_per_dir}-{eos_number//self.eos_per_dir:06d}"
        return os.path.join(self.macro_dir, subdir,
                            self.macro_path_template%{"draw" : eos_number})
    def get_path_template(self, base=None, eos_dir=None, macro=False):
        """
        This is the format universality expects this path to be in
        """
        if base is None:
            if macro:
                base = self.macro_path_template
            else:
                base = self.eos_path_template
        if eos_dir is None:
            eos_dir=self.eos_dir
        return os.path.join(eos_dir,
                            f"DRAWmod{self.eos_per_dir}-%(moddraw)06d", 
                            base)
    
    def get_property(self, eos_indices,
                     dependent_variables=["R"],
                     extraction=Extraction("interpolation", True, "M",  [1.6, 1.8]), **kwargs):
        if extraction.strategy == "extremize":
            # call to process2extrema
            
            return eos_indices
        elif extraction.strategy == "interpolation":
            # call process2samples
            if extraction.dynamic is not None:
                dynamic_kwargs = {"dynamic_x_test" : np.array(extraction.dynamic)}
            else:
                dynamic_kwargs = {}
            if extraction.branched:
                branches_kwargs = {"branches_mapping":self.branches_data}
            else:
                branches_kwargs = {}
            print("path tempalte is", self.get_path_template(macro=extraction.macro))
            data = samples.process2samples(
                np.array(eos_indices),
                self.get_path_template(macro=extraction.macro),
                self.eos_per_dir,
                extraction.independent_variable,
                dependent_variables,
                static_x_test=extraction.values,
                **dynamic_kwargs,
                **branches_kwargs,
                **kwargs)
            ref_columns = extraction.dynamic.columns if extraction.dynamic is not None else []
            columns=samples.outputcolumns(dependent_variables,
                                          extraction.independent_variable,
                                          reference_values=[] if extraction.values is None else extraction.values,
                                          reference_columns=ref_columns)
            return pd.DataFrame(data, columns=columns)
        else:
            raise ValueError(f"Unknown extraction strategy, {extraction.strategy},"
                             "try one of {_known_extraction_strategies}")
        
@dataclass
class EoSPriorDistribution:
    """
    A wrapper around a prior distribution for an EoS.  This should implement a `sample` and 
    `log_prob` method.  

    Arguments:
        eos_sampler: A class which can sample from the eos prior
        variables: A dictionary mapping variable names to the names they have in the eos_sampler
        constructor_args: arguments to pass to the eos_sampler constructor
        constructor_kwargs: keyword arguments to pass to the eos_sampler constructor

    """
    def __init__(self, 
                 eos_sampler, 
                 independent_grid=None,
                 variables={"lnrho" : "log(baryon_density)", 
                 "phi" : "phi"}, 
                 *constructor_args, **constructor_kwargs):
        self.variables = variables
        self.independent_grid = independent_grid
        self.eos_sampler = eos_sampler(*constructor_args, **constructor_kwargs) 
        self.dependent = list(variables.keys())[1]
        self.independent = list(variables.keys())[0]
    def sample(self, num_samples=1, as_array=False):
        """
        Sample from the eos prior
        Arguments:
            num_samples: The number of samples to generate
            as_array: If True, return the samples as an array, otherwise return a DataFrame
                If `as_array` is True, we only return the dependent variable, relying on the 
                user to keep track of what the independent variable is.
        Returns:
            samples: A DataFrame or array of samples
        """
        if num_samples == 1:
            dependent_vals = self.eos_sampler.sample()
            if (as_array):
                return dependent_vals
            return pd.DataFrame(
                np.array([self.independent_grid, dependent_vals]), 
                columns=self.variables.keys())
        samples = []
        for _ in range(num_samples):
            dependent_vals = self.eos_sampler.sample()
            if (as_array):
                return dependent_vals
            samples.append(pd.DataFrame(
                np.array([self.independent_grid, dependent_vals]), 
                columns=self.variables.keys()))
        return samples
    def log_prob(self, samples):
        if isinstance(samples, list[pd.DataFrame]):
            [self.eos_sampler.log_prob(sample[self.dependent]) for sample in samples]
        if isinstance(samples, pd.DataFrame):
            samples = samples[self.dependent]
            return self.eos_sampler.log_prob(samples)
        if isinstance(samples, np.ndarray):
            # Assume this is just the an array of n dependent samples
            return self.eos_sampler.log_prob(samples)

        
class EoSPriorH5:
    def __init__(self, h5_path, eos_subgroup_path="eos", macro_subgroup_path="ns", index_to_key=lambda x : str(x)):
        self.h5_path = str(h5_path)
        self.h5_path = h5_path        
        self.h5_file = h5py.File(name=self.h5_path)
        self.eos_subgroup = self.h5_file[eos_subgroup_path]
        self.macro_subgroup = self.h5_file[macro_subgroup_path]
        self.index_to_key = index_to_key
    def get_eos(self, eos_index):
        return self.eos_subgroup[self.index_to_key(eos_index)]
    def get_macro(self, eos_index):
        return self.macro_subgroup[self.index_to_key(eos_index)]
    def __getitem__(self, name):
        match name:
            case "ns" : return self.macro_subgroup
            case "eos" : return self.eos_subgroup
    
