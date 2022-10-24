import numpy as np
import pandas as pd
import os

from dataclasses import dataclass

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
    def get_eos_path(self, eos_number, subdir=None):
        if subdir == None:
            subdir = f"DRAWmod{self.eos_per_dir}-{eos_number//self.eos_per_dir:06d}"
        return os.path.join(self.eos_dir, subdir,
                            self.eos_path_template%{"draw" : eos_number})
    def get_macro_path(self, eos_number, subdir=None):
        if subdir is None:
            subdir = f"DRAWmod{self.eos_per_dir}-{eos_number//self.eos_per_dir:06d}"
        return os.path.join(self.macro_dir, subdir,
                            self.eos_path_template%{"draw" : eos_number})
    def get_path_template(self, base=None, eos_dir=None):
        """
        This is the format universality expects this path to be in
        """
        if base is None:
            base = self.eos_path_template
        if eos_dir is None:
            eos_dir=self.eos_dir
        return os.path.join(eos_dir,
                            f"DRAWmod{self.eos_per_dir}-%(moddraw)06d", 
                            base)
