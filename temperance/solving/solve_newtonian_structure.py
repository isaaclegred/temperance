"""
Solve the Newtonian stellar structure equations using a log-enthalpy
formulation analogous to Lindblom (1992), with all relativistic corrections
removed.

See solve_tov.py for the GR counterpart and analytic_eos.py for EOS classes.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .analytic_eos import (
        xp, ode_solver, print_val,
        polytropic_eos, css_eos, interpolated_eos,
    )
except ImportError:
    _inner = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, _inner)
    from solving.analytic_eos import (
        xp, ode_solver, print_val,
        polytropic_eos, css_eos, interpolated_eos,
    )

_EOS_TYPES = (interpolated_eos, polytropic_eos, css_eos)


def _wrap_eos(eos):
    if not isinstance(eos, _EOS_TYPES):
        return interpolated_eos(eos)
    return eos


def newtonian_tidal_deformability_rhs(eta, u, du, rho, one_over_cs2, ell=2):
    """
    RHS of the Newtonian tidal Love number ODE.

    Obtained from the GR Lindblom form by taking the Newtonian limit:
      - metric factor f = 1 - 2v  ->  1
      - energy density e  ->  baryon density rho
      - pressure p  ->  0 (subdominant compared to rho)

    The resulting structure is:

        deta/d(-lnh) = -prefactor * (eta(eta-1) + A*eta - B)

    where
        prefactor = du / (2u)
        A = 2 * (1 - 2 pi u rho)
        B = ell*(ell+1) - 4 pi u rho * (1 + 1/cs^2)

    The factor (1 + 1/cs^2) vs the GR (3 + 1/cs^2): the "3" in GR arises
    from pressure contributions to the Tolman mass; it is absent in Newtonian.
    """
    prefactor = 1 / (2*u) * du
    A = 2 * (1 - 2 * xp.pi * u * rho)
    B = (ell + 1) * ell - 4 * xp.pi * u * rho * (1 + one_over_cs2)
    return -prefactor * (eta * (eta - 1) + A * eta - B)


def newtonian_solver(lnhc, eos, termination_lnh=1e-14, ell=2, points_to_solve_for=500):
    """
    Solve the Newtonian stellar structure equations using log-enthalpy as the
    independent variable.

    State vector: [u, v, eta, v_b] where
        u   = r^2
        v   = m(r) / r
        eta = tidal deformability variable (Newtonian)
        v_b = M_baryon / r  (coincides with v in Newtonian since e = rho)

    Structure equations (Newtonian limit of the Lindblom TOV form):
        du/d(-lnh) = 2 * u / v
        dv/d(-lnh) = (4 pi u rho - v) / v

    Compared to the GR equations, the relativistic corrections removed are:
        - metric factor (1 - 2v) in the denominator of cf
        - 4 pi p in the denominator of cf (pressure << density in Newtonian)
        - energy density e replaced by baryon density rho in dv
        - sqrt(1 - 2v) factor removed from dv_b
    """
    initial_stepsize = xp.array(1e-5) * lnhc
    rhoc = eos.rho_of_logenthalpy(xp.array([lnhc]))[0]
    # u = r^2, v = m(r)/r, eta = tidal variable, v_b = M_baryon/r
    initial_state = xp.array([6/np.sqrt(4 * np.pi * rhoc)*initial_stepsize, 2*initial_stepsize, ell, 0.0])

    def rhs(state, minus_lnh):
        lnh = -minus_lnh
        if lnh < 0.0:
            return xp.array([100, 100, 1000, 1000])
        u = state[0]
        v = state[1]
        eta = state[2]
        v_b = state[3]
        rho = eos.rho_of_logenthalpy(xp.array([lnh]))[0]
        cs2 = eos.cs2_of_logenthalpy(xp.array([lnh]))[0]

        v_over_u = v / u
        cf = 1 / v_over_u
        du = 2 * cf
        dv = (4 * xp.pi * rho * cf - 1.0) 

        one_over_cs2 = 1 / cs2
        deta = newtonian_tidal_deformability_rhs(eta, u, du, rho, one_over_cs2, ell=ell)

        dv_b = (4 * xp.pi * rho - v_b / u) * cf

        return xp.array([du, dv, deta, dv_b])

    lnhs_solved = xp.linspace(-lnhc, -termination_lnh, points_to_solve_for)
    solution = ode_solver(rhs, initial_state, lnhs_solved)
    return -lnhs_solved, solution


def eta_to_love_number_newtonian(eta_R, ell=2):
    """
    Convert the surface value of eta to the Newtonian tidal Love number k_ell.

    Matches the interior tidal potential solution to the exterior vacuum solution
    (A r^ell + B r^{-(ell+1)}) at the stellar surface. The Love number is
    determined by the ratio B/A:

        k_ell = (ell - eta_R) / (2 * (eta_R + ell + 1))

    Parameters
    ----------
    eta_R : float
        Value of the logarithmic tidal-potential derivative at the surface.
    ell : int
        Tidal harmonic. Defaults to 2 (quadrupole).

    Returns
    -------
    float
        Newtonian tidal Love number k_ell.
    """
    return (ell - eta_R) / (2 * (eta_R + ell + 1))


def solve_newtonian(eos, rho_c, *args, **kwargs):
    """
    Solve the Newtonian stellar structure equations for a given central density
    and equation of state.
    """
    eos = _wrap_eos(eos)
    rho_c = float(xp.asarray(rho_c).flat[0])
    print("rhoc is", rho_c)
    lnhc = eos.logenthalpy_of_rho(xp.array([rho_c]))[0]
    print("solving for logenthalpy", lnhc)
    return newtonian_solver(lnhc, eos, *args, **kwargs)


def construct_newtonian_star(eos, rho_c, *args, **kwargs):
    """
    Construct a Newtonian star; return mass and density as a function of radius.
    """
    lnhs, sols = solve_newtonian(eos, rho_c, *args, **kwargs)
    u = sols[:, 0]
    r = np.sqrt(u)
    return pd.DataFrame({"baryon_density": eos.rho_of_logenthalpy(lnhs), "r": r, "m": sols[:, 1] * r, "m_baryon": sols[:, 3] * r})


def get_newtonian_family(eos, densities, outpath=None):
    """
    Solve the Newtonian structure equations for a range of central densities
    and return a DataFrame of mass, radius, and tidal Love number.
    """
    Rs = []
    Ms = []
    k2s = []
    rhocs = []
    Ms_baryon = []
    eos = _wrap_eos(eos)
    for density in densities:
        density = xp.array([density])
        lnhs, sols = solve_newtonian(eos, density)
        u_term = sols[-1, 0]
        v_term = sols[-1, 1]
        eta_term = sols[-1, 2]
        vb_term = sols[-1, 3]
        Rs.append(np.sqrt(u_term) * 1.477)  # in km
        Ms.append(np.sqrt(u_term) * v_term)  # in solar masses
        k2s.append(eta_to_love_number_newtonian(eta_term))
        rhocs.append(eos.rho_of_logenthalpy(xp.array([lnhs[0]]))[0] * 2.8e14 / .00045)
        Ms_baryon.append(np.sqrt(u_term) * vb_term)
    data = pd.DataFrame({"central_baryon_density": np.array(rhocs), "M": np.array(Ms), "R": np.array(Rs), "k2": np.array(k2s), "M_baryon": np.array(Ms_baryon)})
    if outpath is not None:
        data.to_csv(outpath, index=False)
    return data


if __name__ == "__main__":
    eos = polytropic_eos(K=123.6, Gamma=2.0)
    tabulation_densities = np.geomspace(.00045e-10, .00045e1, 200)
    poly_pressure = eos.p_of_rho(tabulation_densities)
    poly_energy = eos.e_of_rho(tabulation_densities)
    poly_eos = pd.DataFrame({"baryon_density": tabulation_densities * 2.8e14 / .00045, "pressurec2": poly_pressure * 2.8e14 / .00045, "energy_densityc2": poly_energy * 2.8e14 / .00045})
    poly_eos.to_csv("poly_newtonian.csv", index=False)

    densities = np.linspace(.00045, 8 * .00045, 100)
    family = get_newtonian_family(eos, densities, outpath="macro-poly_newtonian.csv")
    plt.plot(family["R"], family["M"])
    plt.xlabel("R (km)")
    plt.ylabel("M (Msun)")
    plt.xlim(10, 20)
    plt.ylim(.2, 2.5)
    plt.savefig("mr-poly_newtonian.pdf", bbox_inches="tight")
    plt.clf()
    plt.plot(family["M"], family["k2"])
    plt.xlabel("M (Msun)")
    plt.ylabel("k2")
    plt.xlim(.2, 2.5)
    plt.savefig("mk2-poly_newtonian.pdf", bbox_inches="tight")
