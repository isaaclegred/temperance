"""
Solve the Tolman-Oppenheimer-Volkoff equations using Lindblom's log-enthalpy
formulation (Lindblom 1992).

The independent variable is the log-enthalpy lnh = ln(h), which decreases
monotonically from the centre (lnh = lnh_c) to the surface (lnh → 0).
Integration is carried out in the variable -lnh so the solver always steps
in the positive direction.

State vector
------------
    u   = r^2            (avoids the r=0 singularity)
    v   = m(r) / r       (compactness proxy; C = v at the surface)
    eta = tidal Love number variable (logarithmic derivative of metric pert.)
    v_b = M_baryon / r   (baryon mass enclosed)

The TOV equations in this form are (Lindblom 1992, Eqs. 4-5):

    du/d(-lnh) =  2 (1-2v) / (4π p + v/u)
    dv/d(-lnh) =  (4π e - v/u) * (1-2v) / (4π p + v/u)

with the common factor  cf = (1-2v) / (4π p + v/u).

Units
-----
All quantities in geometric units (G = c = 1).  Radii are in units where
1 solar mass ≈ 1.477 km.
"""

import os
import sys
import numpy as np
import scipy.integrate
import pandas as pd
from scipy.special import hyp2f1

import matplotlib.pyplot as plt

try:
    from .analytic_eos import (
        xp, ode_solver, print_val,
        polytropic_eos, css_eos, interpolated_eos,
    )
except ImportError:
    # Allow direct script execution: add the inner temperance/ dir to sys.path.
    _inner = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, _inner)
    from solving.analytic_eos import (
        xp, ode_solver, print_val,
        polytropic_eos, css_eos, interpolated_eos,
    )

_EOS_TYPES = (interpolated_eos, polytropic_eos, css_eos)


def _wrap_eos(eos):
    """Return eos unchanged if already a known type, else wrap in interpolated_eos."""
    if not isinstance(eos, _EOS_TYPES):
        return interpolated_eos(eos)
    return eos


# ---------------------------------------------------------------------------
# Tidal deformability ODE
# ---------------------------------------------------------------------------

def lindblom_tidal_deformability_rhs(eta, u, du, v, e, p, one_over_cs2, ell=2):
    """
    RHS of the tidal Love-number ODE in Lindblom's log-enthalpy variables.

    The perturbation variable eta = d ln H / d ln r tracks the logarithmic
    gradient of the even-parity metric perturbation H.  Its equation of
    motion is (Hinderer 2008, adapted to Lindblom variables):

        deta/d(-lnh) = -prefactor * [eta(eta-1) + A*eta - B]

    where

        f        = 1 - 2v                             (metric factor)
        prefactor = du / (2u)                          (= d ln r / d(-lnh))
        A        = (2/f) * [1 - 3v - 2π u (e + 3p)]
        B        = (1/f) * [ℓ(ℓ+1) - 4π u (e+p) (3 + 1/cs²)]

    Parameters
    ----------
    eta : float
        Current value of the tidal variable.
    u : float
        r^2 at the current point.
    du : float
        d(r^2)/d(-lnh) at the current point.
    v : float
        m(r)/r at the current point.
    e : float
        Total energy density at the current point.
    p : float
        Pressure at the current point.
    one_over_cs2 : float
        Inverse adiabatic sound speed squared (1/cs²).
    ell : int
        Tidal harmonic (default 2 for quadrupole).

    Returns
    -------
    float
        deta / d(-lnh).
    """
    f = 1.0 - 2.0 * v
    prefactor = du / (2.0 * u)
    A = (2.0 / f) * (1.0 - 3.0 * v - 2.0 * xp.pi * u * (e + 3.0 * p))
    B = (1.0 / f) * (ell * (ell + 1) - 4.0 * xp.pi * u * (e + p) * (3.0 + one_over_cs2))
    return -prefactor * (eta * (eta - 1.0) + A * eta - B)


# ---------------------------------------------------------------------------
# Core TOV integrator
# ---------------------------------------------------------------------------

def lindblom_solver(lnhc, eos, termination_lnh=1e-14, ell=2, points_to_solve_for=500):
    """
    Integrate the TOV equations from the centre outward using log-enthalpy.

    Initial conditions are Taylor-expanded near the centre (r → 0):

        u_0  = [6 / sqrt(4π e_c)] * δ_lnh / (1 + 3 w_c)
        v_0  = 2 * δ_lnh / (1 + 3 w_c)
        eta_0 = ell
        v_b_0 = 0

    where  w_c = p_c / e_c  is the central equation-of-state parameter and
    δ_lnh = 1e-5 * lnh_c  is the first integration step.

    Parameters
    ----------
    lnhc : float
        Central log-enthalpy (lnh at r=0).
    eos : EOS object
        Must implement e_of_logenthalpy, p_of_logenthalpy, cs2_of_logenthalpy,
        rho_of_logenthalpy.
    termination_lnh : float
        Log-enthalpy value where integration stops (proxy for the stellar
        surface, where lnh → 0).  Default 1e-14.
    ell : int
        Tidal harmonic for the Love-number integration.  Default 2.
    points_to_solve_for : int
        Number of equally-spaced log-enthalpy grid points.  Default 500.

    Returns
    -------
    lnhs : ndarray, shape (points_to_solve_for,)
        Log-enthalpy grid, decreasing from lnhc to termination_lnh.
    solution : ndarray, shape (points_to_solve_for, 4)
        Columns: [u, v, eta, v_b] on the lnhs grid.
    """
    initial_stepsize = xp.array(1e-5) * lnhc
    ec = eos.e_of_logenthalpy(xp.array([lnhc]))[0]
    w_initial = eos.p_of_logenthalpy(xp.array([lnhc]))[0] / ec
    print_val("Central log-enthalpy: {lnhc}", lnhc=lnhc)
    print_val("Central energy density: {ec}", ec=ec)

    initial_state = xp.array([
        6.0 / xp.sqrt(4.0 * xp.pi * ec) * initial_stepsize / (1.0 + 3.0 * w_initial),
        2.0 * initial_stepsize / (1.0 + 3.0 * w_initial),
        float(ell),
        0.0,
    ])

    def rhs(state, minus_lnh):
        lnh = -minus_lnh
        if lnh < 0.0:
            # Guard: solver may evaluate RHS slightly past the surface.
            return xp.array([100.0, 100.0, 1000.0, 1000.0])

        u, v, eta, v_b = state[0], state[1], state[2], state[3]
        p   = eos.p_of_logenthalpy(xp.array([lnh]))[0]
        e   = eos.e_of_logenthalpy(xp.array([lnh]))[0]
        cs2 = eos.cs2_of_logenthalpy(xp.array([lnh]))[0]
        rho = eos.rho_of_logenthalpy(xp.array([lnh]))[0]

        v_over_u = v / u
        cf = (1.0 - 2.0 * v) / (4.0 * xp.pi * p + v_over_u)

        du   = 2.0 * cf
        dv   = (4.0 * xp.pi * e - v_over_u) * cf
        deta = lindblom_tidal_deformability_rhs(eta, u, du, v, e, p, 1.0 / cs2, ell=ell)
        # Extra sqrt(1-2v) from the GR volume element for the baryon mass integral
        dv_b = (4.0 * xp.pi * rho / xp.sqrt(1.0 - 2.0 * v) - v_b / u) * cf

        return xp.array([du, dv, deta, dv_b])

    lnhs_solved = xp.linspace(-lnhc, -termination_lnh, points_to_solve_for)
    solution = ode_solver(rhs, initial_state, lnhs_solved)
    return -lnhs_solved, solution


# ---------------------------------------------------------------------------
# Love number / tidal deformability
# ---------------------------------------------------------------------------

def eta_to_love_number(eta_R, C, ell=2):
    """
    Convert the surface tidal variable eta_R and compactness C = M/R to the
    dimensionless tidal deformability Lambda = (2/3) k_ell / C^5.

    The Love number k_2 is computed following Hinderer (2008) / Flanagan &
    Hinderer (2008).  The hypergeometric function F = 2F1(3, 5; 6; 2C) and
    its radial derivative appear in the exterior matching condition.

    Parameters
    ----------
    eta_R : float
        Value of the tidal variable at the stellar surface.
    C : float
        Compactness M/R (equals v at the last integration point).
    ell : int
        Tidal harmonic.  Only ell=2 is implemented.

    Returns
    -------
    float
        Dimensionless tidal deformability Lambda = (2/3) k_2 / C^5.
    """
    if ell != 2:
        raise NotImplementedError("Only ell=2 is implemented")
    if C**6 == 0.0:
        print("C = 0.0?", C)
        return np.inf 
    f_R = 1.0 - 2.0 * C
    F = hyp2f1(3.0, 5.0, 6.0, 2.0 * C)

    def dFdz():
        z = 2.0 * C
        return (5.0 / (2.0 * z**6)) * (
            z * (-60.0 + z * (150.0 + z * (-110.0 + 3.0 * z * (5.0 + z))))
            / (z - 1.0) ** 3
            + 60.0 * np.log(1.0 - z)
        )

    RdFdr = -2.0 * C * dFdz()
    k2 = 0.5 * (eta_R - 2.0 - 4.0 * C / f_R) / (
        RdFdr - F * (eta_R + 3.0 - 4.0 * C / f_R)
    )
    return (2.0 / 3.0) * (k2 / C**5)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def solve_tov(eos, rho_c, *args, **kwargs):
    """
    Solve the TOV equations for a single central density.

    Parameters
    ----------
    eos : EOS object or DataFrame
        Equation of state.  If a DataFrame is passed it is wrapped in
        interpolated_eos automatically.
    rho_c : float
        Central baryon density (geometric units).
    *args, **kwargs
        Forwarded to lindblom_solver (e.g. termination_lnh, ell,
        points_to_solve_for).

    Returns
    -------
    lnhs : ndarray
    solution : ndarray, shape (N, 4)
        Columns: [u, v, eta, v_b].
    """
    eos = _wrap_eos(eos)
    rho_c = float(xp.asarray(rho_c).flat[0])
    lnhc = eos.logenthalpy_of_rho(xp.array([rho_c]))[0]
    return lindblom_solver(lnhc, eos, *args, **kwargs)


def construct_star(eos, rho_c, *args, **kwargs):
    """
    Integrate the TOV equations and return a radial profile DataFrame.

    Parameters
    ----------
    eos : EOS object or DataFrame
    rho_c : float
        Central baryon density.

    Returns
    -------
    pandas.DataFrame
        Columns: baryon_density, r, m, m_baryon  (all in geometric units).
    """
    lnhs, sols = solve_tov(eos, rho_c, *args, **kwargs)
    r = np.sqrt(sols[:, 0])
    return pd.DataFrame({
        "baryon_density": eos.rho_of_logenthalpy(lnhs),
        "r":       r,
        "m":       sols[:, 1] * r,
        "m_baryon": sols[:, 3] * r,
    })


def get_tov_family(eos, densities, outpath=None):
    """
    Compute global properties for a sequence of central densities.

    Parameters
    ----------
    eos : EOS object or DataFrame
    densities : array-like
        Central baryon densities to solve for (geometric units).
    outpath : str, optional
        If given, write the result DataFrame to this CSV path.

    Returns
    -------
    pandas.DataFrame
        Columns: central_baryon_density, M [Msun], R [km], Lambda, M_baryon [Msun].
    """
    eos = _wrap_eos(eos)
    Rs, Ms, Lambdas, rhocs, Ms_baryon = [], [], [], [], []

    for density in densities:
        density = xp.array([density])
        lnhs, sols = solve_tov(eos, density)

        r_surf  = np.sqrt(sols[-1, 0])
        C       = float(sols[-1, 1])       # v at surface = M/R
        eta_R   = float(sols[-1, 2])
        vb_surf = float(sols[-1, 3])

        Rs.append(r_surf * 1.477)          # km
        Ms.append(r_surf * C)              # solar masses
        Lambdas.append(eta_to_love_number(eta_R, C))
        rhocs.append(float(eos.rho_of_logenthalpy(xp.array([lnhs[0]]))[0]) * 2.8e14 / 0.00045)
        Ms_baryon.append(r_surf * vb_surf)

    data = pd.DataFrame({
        "central_baryon_density": np.array(rhocs),
        "M":        np.array(Ms),
        "R":        np.array(Rs),
        "Lambda":   np.array(Lambdas),
        "M_baryon": np.array(Ms_baryon),
    })
    if outpath is not None:
        data.to_csv(outpath, index=False)
    return data


# ---------------------------------------------------------------------------
# Quick-start example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    eos = polytropic_eos(K=123.6, Gamma=2.0)

    tabulation_densities = np.geomspace(0.00045e-10, 0.00045e1, 200)
    poly_eos = pd.DataFrame({
        "baryon_density":    tabulation_densities * 2.8e14 / 0.00045,
        "pressurec2":        eos.p_of_rho(tabulation_densities) * 2.8e14 / 0.00045,
        "energy_densityc2":  eos.e_of_rho(tabulation_densities) * 2.8e14 / 0.00045,
    })
    poly_eos.to_csv("poly_100_2.csv", index=False)

    densities = np.linspace(0.00045, 8 * 0.00045, 100)
    family = get_tov_family(eos, densities, outpath="macro-poly_100_2.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(family["R"], family["M"])
    ax1.set_xlabel("R (km)")
    ax1.set_ylabel("M (Msun)")
    ax1.set_xlim(10, 20)
    ax1.set_ylim(0.2, 2.5)
    ax2.plot(family["M"], family["Lambda"])
    ax2.set_yscale("log")
    ax2.set_xlabel("M (Msun)")
    ax2.set_ylabel("Lambda")
    ax2.set_xlim(0.2, 2.5)
    ax2.set_ylim(1, 1e4)
    fig.tight_layout()
    fig.savefig("mr-mlambda-poly_100_2.pdf", bbox_inches="tight")
