import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import temperance as tmpy
import temperance.external.read_3d_compose_table as tmcomp

if __name__ == "__main__":
    prefix = "sfho"
    thermo_path = f"./{prefix}_thermo.csv"
    nb_path = f"./{prefix}_nb.csv"
    t_path = f"./{prefix}_t.csv"
    yq_path = f"./{prefix}_yq.csv"
    #thermo_ns_path = f"./{prefix}_thermo_ns.csv"
    #nb_ns_path = f"./{prefix}_nb_ns.csv"
    thermo, nb, t, yq = tmcomp.read_compose_3d_table(thermo_path,
                                              nb_path, t_path, yq_path)
    #limited_thermo = thermo[thermo["iYq"] == 4]


    limited_thermo = tmcomp.find_2d_eos_by_optimizing_ye(thermo, nb, t, yq)
    limited_thermo.to_csv("sfho_chemical_equalibrium_thermo.csv", index=False)
    cold_eos = tmcomp.find_1d_eos_by_taking_T_equal_minT(limited_thermo, nb, t)
    ns_thermo, nb_ns, t, yq  = tmcomp.read_compose_3d_table(
        thermo_path, nb_path, t_path, yq_path)
    
    print(limited_thermo)
    #cgs_eos= tmcomp.cold_eos_to_cgs_standard(ns_thermo, nb_ns, enfoce_first_law=False)
    cgs_eos_inferred = tmcomp.cold_eos_to_cgs_standard(cold_eos, nb, enfoce_first_law=False)
    
    cgs_eos_inferred.to_csv(f"{prefix}_cgs.csv")
    #plt.plot(cgs_eos["baryon_density"], cgs_eos["pressurec2"], color="deepskyblue")
    plt.plot(cgs_eos_inferred["baryon_density"], cgs_eos_inferred["pressurec2"], color="darkred", linestyle="--")
    #plt.plot(nb["nb"], cold_eos["p_per_nb"])
    plt.xlabel(r"$\rho\ [\mathrm{g}/\mathrm{cm}^3]$")
    plt.ylabel(r"$p\ [\mathrm{g}/\mathrm{cm}^3]$") 
    
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    plt.savefig(f"compose_{prefix}_eos.pdf")

