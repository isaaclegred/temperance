import numpy as np
import pandas as pd
import os

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
                dynamic_kwargs = {"dynamic_x_text" : np.array(extraction.dynamic)}
            else:
                dynamic_kwargs = {}
            if extraction.branched:
                branches_kwargs = {"branches_mapping":self.branches_data}
            else:
                branches_kwargs = {}
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
            ref_columns = extaction.dynamic.columns if extraction.dynamic is not None else []
            columns=samples.outputcolumns(dependent_variables,
                                          extraction.independent_variable,
                                          reference_values=extraction.values,
                                          reference_columns=ref_columns)
            return pd.DataFrame(data, columns=columns)
        else:
            raise ValueError(f"Unknown extraction strategy, {extraction.strategy},"
                             "try one of {_known_extraction_strategies}")
        
