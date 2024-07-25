import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy import integrate as integrate

import temperance.utilities.units as tmu

def read_compose_3d_table(thermo_path, nb_path, t_path, yq_path ):
    """
    Read a 3-d table from compose
    Args: (all str)
      thermo_path : the eos file ending in .thermo
      nb_path: the eos file ending in .nb
      t_path : the eos file ending in .t
      yq_path: the eos file ending in .yq
    returns:  thermo, nb, t, yq: dataframes containing the tables;
              see the compose docs for how to interpret these files
      
    """
    with open(thermo_path) as open_file:
        meta_info = open_file.readline()
        split_info=  meta_info.split()
        m_n = float(split_info[0])
        m_p = float(split_info[1])
        has_electrons = bool(split_info[2])
    thermo = pd.read_csv(thermo_path, skiprows = [0],
                         names=["iT", "inb", "iYq",
                                "p_per_nb", "s_per_nb", "mub_per_mn_minus_1",
                                "muq_per_mn", "mul_per_mn", "f_per_nbmn_minus_1",
                                "e_per_nbmn_minus_1"],
                         delim_whitespace=True, index_col=False)
    nb = pd.read_csv(nb_path, skiprows=[0, 1], names=["nb"])
    t = pd.read_csv(t_path, skiprows=[0, 1], names=["T"])
    yq = pd.read_csv(yq_path, skiprows=[0, 1], names=["Yq"])
    return (thermo, nb, t, yq)
def find_2d_eos_by_optimizing_ye(thermo, nb, t, yq):
    """
    Compute a 2-d EoS by assuming the system has reached chemical equalibrium.
    Args: thermo, nb, t, yq: dataframes 
    Returns: thermo under chemical equalibrium, i.e. with y_q fixed so that 
    free energy is minimized for each density and temperature.
    """
    #return  thermo.sort_values("f_per_nbmn_minus_1").groupby(["iT", "inb"]).head(1).sort_values(["inb","iT"])
    #return  thermo[thermo["iYq"] == min(thermo["iYq"])]
    return  thermo.sort_values("e_per_nbmn_minus_1").groupby(["iT", "inb"]).head(1).sort_values(["inb","iT"])

def plot_2d_eos(thermo, nb, t):
    NB, T = np.meshgrid(nb["nb"], t["T"])
    plt.contourf(NB, T,  np.reshape(np.array(limited_thermo["p_per_nb"]),
                                    (max(limited_thermo["iT"]),
                                     max(limited_thermo["inb"]))),
                 norm=LogNorm(),
                 levels=np.geomspace(.01,1e5, 30))
    plt.xlabel("$n_b\\ [\\mathrm{fm}^-3]$")
    #plt.xscale("log")
    plt.ylabel("$T\\ [\\mathrm{MeV}]$")
    #plt.yscale("log")
    plt.title("$p\\ [\mathrm{MeV}/\mathrm{fm}^{-3}]$")
    plt.colorbar()
    plt.show()

def find_1d_eos_by_taking_T_equal_minT(thermo, nb, t):
    """
    Return a 1-d EoS by taking an EoS in chemical equalibrium and setting the
    temperature equal to the minimum value in the table.  
    Args: thermo (eos in chemical equalibium), nb, t: dataframes
    Returns: thermo_cold EoS at minimal temperature in table for each density
    """
    return thermo.loc[thermo["iT"] == min(thermo["iT"])]
def plot_1d_eos(thermo, nb, units="default"):
    if units == "default":
        plt.plot(nb, cold_eos["p_per_nb"])


def cold_eos_to_cgs_standard(cold_thermo, nb, enfoce_first_law=False):
    nb = np.array(nb["nb"])
    p_in_mev_per_fm_3 = cold_thermo["p_per_nb"] * nb
    p_in_g_per_cm_3 = tmu.nuclear_density_to_cgs(p_in_mev_per_fm_3)
    rho_in_g_per_cm_3 = tmu.nuclear_baryon_number_density_to_cgs_mass_density(nb)
    e_in_g_per_cm_3 = (cold_thermo["e_per_nbmn_minus_1"] + 1.0) * rho_in_g_per_cm_3
    # The first law of thermodynamics almost certainly won't be satisfied
    # we take p + e to be the "real variables" since the baryon density is made up
    if enfoce_first_law:
        h = np.exp(integrate.cumtrapz(1/(p_in_g_per_cm_3+e_in_g_per_cm_3), x=p_in_g_per_cm_3, initial=0.0))
        rho_in_g_per_cm_3 = (e_in_g_per_cm_3 + p_in_g_per_cm_3)/h
    return pd.DataFrame({"baryon_density":rho_in_g_per_cm_3,
                        "energy_densityc2":e_in_g_per_cm_3,
                        "pressurec2": p_in_g_per_cm_3})
