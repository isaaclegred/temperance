import pandas as pd
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import os

import matplotlib.pyplot as plt
import temperance.solving.solve_relativistic as solve_relativistic
import matplotlib.cm as cm
import temperance.plotting.envelope as envelope
import matplotlib as mpl
envelope.get_defaults(mpl, fontsize=20)
import tqdm as tqdm


def extrapolate_eos_to_high_density(eos, e_max=1e20, n_points=10000, name="eos", fix=False):
    cs2 = np.gradient(eos["pressurec2"], eos["energy_densityc2"])
    cs2 = np.where(cs2<0, 1e-6, cs2)
    e = eos["energy_densityc2"]
    phi = np.log(1/cs2 - 1)
    valid_indices = np.where(cs2 <= 1)[0]
    e = e[valid_indices]
    phi = phi[valid_indices]
    print("e:", e)
    print("phi:", phi)
    use_for_extrapolation =  - 25
    coeffs = np.polyfit(np.log(e)[use_for_extrapolation:], np.floor(phi)[use_for_extrapolation:], deg=2)
    log_e_extrapolate = np.linspace(np.log(max(e)), np.log(e_max), n_points)
    phi_extrapolate = np.nan_to_num(np.polyval(coeffs, log_e_extrapolate), nan=6)
    plt.clf()
    plt.plot(np.log(e), phi, label="original cs2")
    plt.plot(log_e_extrapolate, phi_extrapolate, label="extrapolated cs2")
    plt.xscale("log")
    plt.xlabel("log(energy density [c2])")
    plt.ylabel("phi")
    plt.legend()
    plt.savefig(f"phi_extrapolation_{name}.pdf")
    e_extrapolate = np.exp(log_e_extrapolate)
    cs2_extrapolate = 1 / (1 + np.exp(phi_extrapolate))

    p_extrapolate = integrate.cumulative_trapezoid(cs2_extrapolate, e_extrapolate, initial=0) + max(eos["pressurec2"][valid_indices])
    h_extrapolate = np.exp(integrate.cumulative_trapezoid(1 / (e_extrapolate + p_extrapolate), p_extrapolate, initial=0))  * (max(eos.loc[valid_indices]["pressurec2"]) + max(eos.loc[valid_indices]["energy_densityc2"])) / (max(eos.loc[valid_indices]["baryon_density"]))
    rho_extrapolate = (e_extrapolate + p_extrapolate) / h_extrapolate


    plt.clf()
    plt.plot(e_extrapolate, cs2_extrapolate, label="extrapolated cs2")
    plt.xscale("log")
    plt.xlabel("energy density [c2]")
    plt.ylabel("cs2")
    plt.legend()
    plt.savefig(f"cs2_extrapolation_{name}.pdf")
    eos = pd.concat([eos.loc[valid_indices],  pd.DataFrame({"baryon_density" : rho_extrapolate[1:], "pressurec2":p_extrapolate[1:],
    "energy_densityc2" : e_extrapolate[1:]})], ignore_index=True)
    if fix:
        # Fix thermodynamic  badness via reintigrating the TOV equations to get a consistent EoS
        h_new = np.exp(integrate.cumulative_trapezoid(1 / (eos["energy_densityc2"] + eos["pressurec2"]), eos["pressurec2"], initial=0))  * (min(eos["pressurec2"]) + min(eos["energy_densityc2"])) / (min(eos["baryon_density"]))
        rho_new = (eos["energy_densityc2"] + eos["pressurec2"]) / h_new
        eos = pd.DataFrame({"baryon_density" : rho_new, "pressurec2":eos["pressurec2"], "energy_densityc2" : eos["energy_densityc2"]})
    eos.to_csv("./NewEoS/extrapolated_"+name+".csv", index=False)
    return eos
    # append to eos

def get_high_density_MR_curve(eos, central_densities):
    masses = []
    radii = []
    solution = solve_relativistic.get_tov_family(
        eos, central_densities, outpath=None)
    masses = solution["M"]
    radii = solution["R"]
    return np.array(masses), np.array(radii)

def plot_high_density_MR_curve(eos, central_energy_densities, label=None, fig=None, **kwargs):
    masses, radii = get_high_density_MR_curve(eos, central_energy_densities)
    fig.plot(radii, masses, label=label,  **kwargs)
    return fig

def plot_multiple_MR_curves(eos_list, central_densities, labels=None, colors=None):
    fig, ax = plt.subplots()
    for i, eos in enumerate(eos_list):
        label = labels[i] if labels is not None else None
        color = colors[i] if colors is not None else None
        plot_high_density_MR_curve(eos, central_densities, label=label, fig=ax, color=color)
    ax.set_xlabel(r"$R\ [\rm{km}]$", fontdict={'size':16})
    ax.set_ylabel(r"$M\ [M_{\odot}]$", fontdict={'size':16})
    if labels is not None:
        ax.legend()
    return fig


def get_tov_slope_field(eos, lnh, e_tildes = np.linspace(1e-2, .5, 20), vs = np.linspace(1e-2, .5, 20), fig=None, test_lyapunov_function=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 5.6))
    ax = fig.gca()
    cs2 = eos.cs2_of_logenthalpy(lnh)
    w = eos.p_of_logenthalpy(lnh) / eos.e_of_logenthalpy(lnh)
    # get etilde and v grid based on cs2
    # print("lnh is", lnh)
    # print("cs2 is",  cs2)
    etilde = e_tildes
    v = vs
    def etilde_dot(cs2, w, etilde, v):
        return etilde * (2*(1-2*v)/(w * etilde + v) - (1+w)/cs2)
    def v_dot(cs2, w, etilde, v):
        return ((1-2*v)/(w * etilde + v) * (etilde - v))
    E, V = np.meshgrid(etilde, v)
    dE = etilde_dot(cs2, w, E, V)
    dV = v_dot(cs2, w, E, V)
    ax.quiver(E, V, dE, dV)
    if test_lyapunov_function is not None:
        if test_lyapunov_function == "default":
            def test_lyapunov_funtion(cs2, w, E, V):
                v_star = 2 * cs2/(4 * cs2 + (1+w)**2)
                e_star = v_star
                b = 3
                c= 10
                L_e =  3/8*(E - e_star)  -1/8 * (V-v_star)
                L_v =  7/8*(V - v_star)  -1/8* (E - e_star)
                return L_e, L_v
        L_e, L_v = test_lyapunov_funtion(cs2, w, E, V)
        failed = np.where(L_e*dE + L_v*dV>0)
        plt.contour(E, V, L_e*dE + L_v*dV, levels=[0], colors="red", linewidths=2)
        print("Lyapunov test results", list(zip(E[failed], V[failed])))

    ax.set_xlabel(r"$\tilde{e}$")
    ax.set_ylabel("$v$")
    plt.plot(etilde, etilde, color="black", linestyle="--")
    plt.plot(1/w * ((2 * (1-2*v) * cs2)/(1+w) - v), v, color="black", linestyle="--")
    return fig, ax


def plot_tov_trajectories(eos, central_densities, lnhs_to_display, label="eos", make_radius_cartoon = True):


    tov_solutions = []
    for rhoc in central_densities:
        lnhs, tov_solution = solve_relativistic.solve_tov(eos, np.array([rhoc]), points_to_solve_for=2000)
        u = tov_solution[:, 0]
        v = tov_solution[:, 1]
        lnh = lnhs
        e = eos.e_of_logenthalpy(lnh)
        solution = pd.DataFrame({"lnh": lnh, "etilde": 4 * np.pi * e * u, "v": v, "u": u})
        print(solution)
        tov_solutions.append(solution)

    for lnh_to_display in tqdm.tqdm(lnhs_to_display):
        e_tildes_to_display = np.linspace(1e-3, 0.5, 20)
        vs_to_display = np.linspace(1e-3, 0.34, 20)
        fig, ax = get_tov_slope_field(eos, lnh_to_display, e_tildes=e_tildes_to_display, vs=vs_to_display, test_lyapunov_function=None)
        central_logenthalpies = eos.logenthalpy_of_rho(central_densities)
        max_central_lnh = max(central_logenthalpies)
        local_v_curve = []
        local_u_curve = []

        color_factor = .85
        color_map = mpl.colormaps["inferno"]
        if make_radius_cartoon:
            ax2  = fig.add_axes([0.6, 0.55, 0.3, 0.3])
            ax2.add_patch(plt.Circle((0, 0), 1.0, color="black", alpha=0.5))
            ax2.set_xticks([-1.0, -.5,0.0, 0.5, 1.0])
            ax2.set_yticks([-1.0, -.5,0.0, 0.5, 1.0])
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.grid()
        for solution in tov_solutions:
            etilde_to_display = interpolate.griddata(solution["lnh"], solution["etilde"], np.linspace(max(solution["lnh"]), lnh_to_display, 100))
            v_to_display = interpolate.griddata(solution["lnh"], solution["v"], np.linspace(max(solution["lnh"]), lnh_to_display, 100))
            ax.plot(etilde_to_display, v_to_display, color=color_map(color_factor * solution["lnh"][0] / max_central_lnh), lw=2)
            ax.scatter(etilde_to_display[-1], v_to_display[-1], color=color_map(color_factor * solution["lnh"][0] / max_central_lnh), s=18)
            local_v_curve.append(v_to_display[-1])
            local_u_curve.append(etilde_to_display[-1])
            #ax.plot(solution["etilde"], solution["v"], color=cm.viridis(solution["lnh"][0] / max_central_lnh))
            if make_radius_cartoon:
                # make a plot of a circle representing a NS and a dot showing the fractional distance to the surface for each solution
                radius = np.sqrt(interpolate.griddata(solution["lnh"], solution["u"], lnh_to_display))
                max_radius = np.max(np.sqrt(solution["u"]))
                ax2.scatter(radius / max_radius, 0, color=color_map(color_factor * solution["lnh"][0] / max_central_lnh), s=50)
                ax2.add_patch(plt.Circle((0, 0), radius/max_radius, linestyle="--", color=color_map(color_factor * solution["lnh"][0] / max_central_lnh), fill=False, lw=2))
                ax2.set_aspect("equal")
        ax.plot(local_u_curve, local_v_curve, color="black", lw=2, linestyle="--", label="TOV endpoints")
        ax.set_xlim(min(e_tildes_to_display), max(e_tildes_to_display))
        ax.set_ylim(min(vs_to_display), max(vs_to_display))
        ax.set_title(rf"$\ln h = {lnh_to_display:.3f}$", fontdict={'size':18})
        fig.savefig(f"tov_trajectories/{label}/snapshot_lnh_{lnh_to_display:06f}.pdf")


if __name__ == "__main__":
    eos = solve_relativistic.polytropic_eos(K=100, Gamma=2.0)
    #central_densities = np.geomspace(1e-6, 1, 50)
    # central_densities = np.array([1e-3])
    # label="gamma2p0_single"

    # # if os.path.exists(f"tov_trajectories/{label}"):
    # #     os.remove(f"tov_trajectories/{label}")
    # os.makedirs(f"tov_trajectories/{label}", exist_ok=True)
    # plot_tov_trajectories(eos, central_densities, lnhs_to_display=np.linspace(1e-3, 185e-3,  31), label=label)
    # central_densities = np.array([1e-1])
    # label="gamma2p0_single_high"

    # # if os.path.exists(f"tov_trajectories/{label}"):
    # #     os.remove(f"tov_trajectories/{label}")
    # os.makedirs(f"tov_trajectories/{label}", exist_ok=True)
    # plot_tov_trajectories(eos, central_densities, lnhs_to_display=np.linspace(1e-1, 3,  31), label=label)

    # plt.savefig("tov_trajectories.pdf")

    # central_densities = np.geomspace(1e-6, 0.2, 50)
    # label="gamma2_new"

    # # if os.path.exists(f"tov_trajectories/{label}"):
    # #     os.remove(f"tov_trajectories/{label}")
    # os.makedirs(f"tov_trajectories/{label}", exist_ok=True)
    # plot_tov_trajectories(eos, central_densities, lnhs_to_display=np.geomspace(1e-3, 3,  31), label=label, make_radius_cartoon=False)


    # eos_sfho_33 = solve_tov.interpolated_eos(extrapolate_eos_to_high_density(pd.read_csv("./NewEoS/sfho_2p0_cs2_p33.csv"),name="sfho_2p0_cs2_p33"))
    # central_densities = np.geomspace(1e-6, 0.2, 50)
    # label="sfho_2p0_cs2_p33"
    # # if os.path.exists(f"tov_trajectories/{label}"):
    # #     os.remove(f"tov_trajectories/{label}")
    # os.makedirs(f"tov_trajectories/{label}", exist_ok=True)
    # plot_tov_trajectories(eos_sfho_33, central_densities, lnhs_to_display=np.geomspace(1e-3, 3,  31), label=label, make_radius_cartoon=False)
    eos_dbhf_2507 = solve_relativistic.interpolated_eos(extrapolate_eos_to_high_density(pd.read_csv("./NewEoS/dbhf_2507_extended.csv"),name="dbhf_2507", fix=True))
    central_densities = np.geomspace(.8e-3, 4.0  * .00045, 50)
    label="dbhf_2507"
    # if os.path.exists(f"tov_trajectories/{label}"):
    #     os.remove(f"tov_trajectories/{label}")
    os.makedirs(f"tov_trajectories/{label}", exist_ok=True)
    plot_tov_trajectories(eos_dbhf_2507, central_densities, lnhs_to_display=np.geomspace(3e-2, .32,  31), label=label, make_radius_cartoon=False)


    # sfho_eos_table= pd.read_csv("~/ComposeEoSs/SFHo/sfho.csv")
    # eos_sfho = solve_tov.interpolated_eos(extrapolate_eos_to_high_density(sfho_eos_table, name="sfho"))
    # eos_bsk22 = solve_tov.interpolated_eos(extrapolate_eos_to_high_density(pd.read_csv("~/ComposeEoSs/Bsk22/bsk22.csv"), name="bsk22"))
    # eos_sfho_33 = solve_tov.interpolated_eos(extrapolate_eos_to_high_density(pd.read_csv("./NewEoS/sfho_2p0_cs2_p33.csv"),name="sfho_2p0_cs2_p33"))
    # plot_multiple_MR_curves([eos, eos_sfho, eos_bsk22, eos_sfho_33], central_densities=np.geomspace(1e-4, 3.5e-1, 100), labels=[r"$\Gamma=2$", "SFHo", "BSk22", "SFHo : 2 : 0.33"], colors=["darkorange", "navy", "coral", "deepskyblue"])
    # #plot_multiple_MR_curves([eos_sfho], central_densities=np.geomspace(1e-4, 3.89e-3, 100), labels=["SFHo"])
    # test_rs = np.linspace(6, 14, 100)
    # plt.plot(test_rs, test_rs * 1/6 / 1.477, color="black", linestyle="--", label="$C=1/6$")
    # plt.plot(test_rs, test_rs * 1/4 / 1.477, color="black", linestyle="--", label="$C=1/4$")
    # plt.plot(test_rs, test_rs * 1/3 / 1.477, color="black", linestyle="--", label="$C=1/3$")
    # plt.xlim(7, 14)
    # plt.ylim(0, 2.4)


    # plt.savefig("mr_comparison.pdf")