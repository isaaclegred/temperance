import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy import integrate as integrate

import temperance.utilities.units as tmu

def download_compose_3d_files(toplevel_url, eos_name, dest_dir=".", force=False):
    """
    Download a CompOSE 3-D EoS table from a URL into local CSV files.

    Mirrors Examples/Compose/get_eos.sh: fetches eos.thermo, eos.nb, eos.t,
    and eos.yq from *toplevel_url* and saves them as
    ``<eos_name>_thermo.csv``, ``<eos_name>_nb.csv``, ``<eos_name>_t.csv``,
    ``<eos_name>_yq.csv`` inside *dest_dir*.

    If all four files already exist on disk the download is skipped and the
    existing paths are returned.  Pass ``force=True`` to re-download even when
    cached files are present.

    Parameters
    ----------
    toplevel_url : str
        Base URL of the CompOSE EoS directory, e.g.
        ``"https://compose.obspm.fr/download/..."``
    eos_name : str
        Label used to build output file names, e.g. ``"qmc_rmf3"``.
    dest_dir : str, optional
        Directory in which to write the files (default: current directory).
    force : bool, optional
        If True, re-download even if cached files exist (default: False).

    Returns
    -------
    thermo_path, nb_path, t_path, yq_path : str
        Paths to the four downloaded files, ready to pass to
        ``read_compose_3d_table``.
    """
    import os
    import urllib.request

    os.makedirs(dest_dir, exist_ok=True)
    toplevel_url = toplevel_url.rstrip("/")

    remote_files = [
        ("thermo", "eos.thermo"),
        ("nb",     "eos.nb"),
        ("t",      "eos.t"),
        ("yq",     "eos.yq"),
    ]

    paths = {}
    for key, remote_name in remote_files:
        dest = os.path.join(dest_dir, f"{eos_name}_{key}.csv")
        paths[key] = dest

    if not force and all(os.path.exists(p) for p in paths.values()):
        print(f"Using cached CompOSE files for '{eos_name}' in '{dest_dir}'")
        return paths["thermo"], paths["nb"], paths["t"], paths["yq"]

    for key, remote_name in remote_files:
        url  = f"{toplevel_url}/{remote_name}"
        dest = paths[key]
        if not force and os.path.exists(dest):
            print(f"  Cached: {dest}")
        else:
            print(f"  Downloading {url} -> {dest}")
            urllib.request.urlretrieve(url, dest)

    return paths["thermo"], paths["nb"], paths["t"], paths["yq"]

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
def find_2d_eos_by_optimizing_ye(thermo, nb, t, yq, order=0):
    """
    Compute a 2-d EoS by assuming the system has reached chemical equalibrium.
    Args: thermo, nb, t, yq: dataframes 
    Returns: thermo under chemical equalibrium, i.e. with y_q fixed so that 
    free energy is minimized for each density and temperature.
    """
    #return  thermo.sort_values("f_per_nbmn_minus_1").groupby(["iT", "inb"]).head(1).sort_values(["inb","iT"])
    #return  thermo[thermo["iYq"] == min(thermo["iYq"])]
    if order == 0:
        # Oth order: just take the minimum free energy for each (iT, inb) pair
        return  thermo.sort_values("e_per_nbmn_minus_1").groupby(["iT", "inb"]).head(1).sort_values(["inb","iT"])
    elif order == 1:
        print("First order not defined for minimization, try order=2")
        pass
    elif order == 2:
        # For each (iT, inb) group: fit a quadratic to e vs Yq, find the analytic
        # vertex Yq_opt = -b/(2a), then linearly interpolate all thermo columns at
        # that Yq.  Falls back to the discrete minimum when the group has fewer than
        # 3 Yq points or the vertex lies outside the sampled Yq range.
        yq_vals = np.asarray(yq["Yq"])  # 1-based: iYq maps to yq_vals[iYq - 1]
        value_cols = [c for c in thermo.columns if c not in ("iT", "inb", "iYq")]

        rows = []
        for (iT_val, inb_val), group in thermo.groupby(["iT", "inb"]):
            group_s = group.sort_values("iYq")
            Yq_g = yq_vals[group_s["iYq"].values.astype(int) - 1]
            e_g  = group_s["e_per_nbmn_minus_1"].values

            Yq_opt = None
            if len(Yq_g) >= 3:
                [_, b, a] = np.polynomial.Polynomial.fit(Yq_g, e_g, 2).coef
                if a > 0:
                    vertex = -b / (2.0 * a)
                    if Yq_g[0] <= vertex <= Yq_g[-1]:
                        Yq_opt = vertex

            if Yq_opt is None:
                # Fall back: pick the row with the smallest discrete e
                row = group_s.iloc[np.argmin(e_g)].to_dict()
            else:
                row = {"iT": iT_val, "inb": inb_val, "iYq": np.nan}
                for col in value_cols:
                    row[col] = float(np.interp(Yq_opt, Yq_g, group_s[col].values))

            rows.append(row)

        return (pd.DataFrame(rows, columns=["iT", "inb", "iYq"] + value_cols)
                  .sort_values(["inb", "iT"])
                  .reset_index(drop=True))
    
def find_1d_eos_by_taking_T_equal_Tmin_then_optimizing_ye(thermo, nb, t, yq, order=0):
    """
    Return a cold (T = T_min), beta-equilibrium 1-D EoS.

    For each baryon density the Yq that minimises e_per_nbmn_minus_1 is found
    and all thermodynamic quantities are evaluated there.

    order=0 : pick the discrete Yq grid point with the lowest e.
    order=2 : fit a quadratic to e(Yq) for each density, find the analytic
              vertex Yq_opt = -b/(2a), and linearly interpolate all thermo
              columns at that Yq.  Falls back to the discrete minimum when
              fewer than 3 Yq points are available or the vertex lies outside
              the sampled range.

    The returned DataFrame is decoupled from the 3-D grid indices:
      - ``nb``     : baryon density [fm^-3]
      - ``Yq_opt`` : beta-equilibrium charge fraction
      - all thermo columns (p_per_nb, e_per_nbmn_minus_1, …) evaluated at Yq_opt
    """
    zero_T_thermo = thermo.loc[thermo["iT"] == min(thermo["iT"])]
    nb_vals  = np.asarray(nb["nb"])
    yq_vals  = np.asarray(yq["Yq"])
    value_cols = [c for c in zero_T_thermo.columns if c not in ("iT", "inb", "iYq")]

    if order == 0:
        rows = []
        for inb_val, group in zero_T_thermo.groupby("inb"):
            group_s  = group.sort_values("iYq")
            Yq_g     = yq_vals[group_s["iYq"].values.astype(int) - 1]
            e_g      = group_s["e_per_nbmn_minus_1"].values
            idx_min  = np.argmin(e_g)
            row = {"nb": nb_vals[int(inb_val) - 1], "Yq_opt": float(Yq_g[idx_min])}
            for col in value_cols:
                row[col] = group_s[col].values[idx_min]
            rows.append(row)
        return (pd.DataFrame(rows, columns=["nb", "Yq_opt"] + value_cols)
                  .reset_index(drop=True))

    elif order == 1:
        print("First order not defined for minimization, try order=2")

    elif order == 2:
        rows = []
        for inb_val, group in zero_T_thermo.groupby("inb"):
            group_s = group.sort_values("iYq")
            Yq_g    = yq_vals[group_s["iYq"].values.astype(int) - 1]
            e_g     = group_s["e_per_nbmn_minus_1"].values

            Yq_opt = None
            if len(Yq_g) >= 3:
                a, b, _ = np.polyfit(Yq_g, e_g, 2)
                if a > 0:
                    vertex = -b / (2.0 * a)
                    if Yq_g[0] <= vertex <= Yq_g[-1]:
                        Yq_opt = vertex

            if Yq_opt is None:
                idx_min = np.argmin(e_g)
                Yq_opt  = float(Yq_g[idx_min])
                row = {"nb": nb_vals[int(inb_val) - 1], "Yq_opt": Yq_opt}
                for col in value_cols:
                    row[col] = float(group_s[col].values[idx_min])
            else:
                row = {"nb": nb_vals[int(inb_val) - 1], "Yq_opt": Yq_opt}
                for col in value_cols:
                    row[col] = float(np.interp(Yq_opt, Yq_g, group_s[col].values))

            rows.append(row)

        return (pd.DataFrame(rows, columns=["nb", "Yq_opt"] + value_cols)
                  .reset_index(drop=True))



def find_2d_eos_by_taking_T_equal_minT(thermo, nb, t):
    """
    Return a 2-d EoS by taking an EoS and setting the
    temperature equal to the minimum value in the table.  
    Args: thermo (eos in chemical equalibium), nb, t: dataframes
    Returns: thermo_cold EoS at minimal temperature in table for each density
            (will still depend on y_q).
    """
    return thermo.loc[thermo["iT"] == min(thermo["iT"])]

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
