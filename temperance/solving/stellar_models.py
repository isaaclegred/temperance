"""
Stellar structure models for use with the oscillation mode solvers.

Two classes:
  - NewtonianStellarModel : integrates the Newtonian structure equations
  - RelativisticStellarModel : wraps the Lindblom TOV solver in solve_tov.py

Both expose a common set of attributes so that mode solvers can be written
against a single interface.

Common interface (all arrays are on a 1-D radial grid, index 0 = centre)
-----------------------------------------------------------------------
r         : np.ndarray  radial coordinate (geometric units, km, or cm)
rho       : np.ndarray  baryon mass density
p         : np.ndarray  pressure
Gamma1    : np.ndarray  adiabatic index d ln p / d ln rho |_s
cs2       : np.ndarray  adiabatic sound speed squared
m         : np.ndarray  enclosed gravitational mass
R         : float       stellar radius (same units as r)
M         : float       total gravitational mass

Newtonian-only
--------------
phi       : np.ndarray  Newtonian gravitational potential (negative inside star)

Relativistic-only
-----------------
e         : np.ndarray  total energy density
nu        : np.ndarray  metric potential  g_tt = -e^{2 nu}
lam       : np.ndarray  metric potential  g_rr =  e^{2 lam}  (lam = -0.5 ln(1-2m/r))
C         : float       compactness  M/R
"""

from __future__ import annotations

import numpy as np
import scipy.integrate
import scipy.interpolate

from .solve_relativistic import lindblom_solver
from .solve_newtonian_structure import newtonian_solver
from .analytic_eos import polytropic_eos, interpolated_eos, css_eos


# ---------------------------------------------------------------------------
# Newtonian stellar model
# ---------------------------------------------------------------------------

class NewtonianStellarModel:
    """
    Stellar structure model for use with the Newtonian oscillation solver.

    Integrates the Newtonian stellar structure equations in log-enthalpy form
    (solve_newtonian_structure.newtonian_solver):

        du/d(-lnh) = 2 u / v                  [u = r², v = m/r]
        dv/d(-lnh) = (4π ρ u - v) / v

    The Newtonian gravitational potential Φ is computed by integrating
    dΦ/dr = m/r² outward and matching to the exterior solution Φ(R) = -M/R.

    Parameters
    ----------
    eos : object
        Equation of state (polytropic_eos, css_eos, or interpolated_eos).
    rho_c : float
        Central baryon density (geometric units, consistent with eos).
    n_points : int
        Number of log-enthalpy grid points passed to newtonian_solver.
    termination_lnh : float
        Log-enthalpy value at which integration stops (surface proxy).
    """

    def __init__(
        self,
        eos,
        rho_c: float,
        *,
        n_points: int = 500,
        termination_lnh: float = 1e-14,
    ):
        if not isinstance(eos, (interpolated_eos, polytropic_eos, css_eos)):
            eos = interpolated_eos(eos)
        self.eos = eos
        self.rho_c = rho_c
        self.n_points = n_points
        self.termination_lnh = termination_lnh

        # Solved profile arrays (populated by solve())
        self.r: np.ndarray | None = None
        self.rho: np.ndarray | None = None
        self.p: np.ndarray | None = None
        self.m: np.ndarray | None = None
        self.phi: np.ndarray | None = None
        self.Gamma1: np.ndarray | None = None
        self.cs2: np.ndarray | None = None
        self.R: float | None = None
        self.M: float | None = None

        self.solve()

    def solve(self) -> None:
        """Populate profile arrays using the Newtonian structure solver."""
        lnhc = self.eos.logenthalpy_of_rho(np.array([self.rho_c]))[0]
        lnhs, sols = newtonian_solver(
            lnhc,
            self.eos,
            termination_lnh=self.termination_lnh,
            points_to_solve_for=self.n_points,
        )

        u = sols[:, 0]   # r^2
        v = sols[:, 1]   # m(r)/r
        r = np.sqrt(u)
        m = v * r

        self.lnh = lnhs
        self.r = r
        self.m = m
        self.rho = self.eos.rho_of_logenthalpy(lnhs)
        self.p = self.eos.p_of_logenthalpy(lnhs)
        # cs2_of_logenthalpy returns the GR sound speed dp/de = (dp/drho)/(de/drho).
        # de/drho = exp(lnh) for a cold EOS (the specific relativistic enthalpy h).
        # The Newtonian adiabatic index Gamma1 = (rho/p) * dp/drho
        #   = (rho/p) * cs2_GR * exp(lnh).
        cs2_gr = self.eos.cs2_of_logenthalpy(lnhs)
        self.cs2 = cs2_gr * np.exp(lnhs)   # Newtonian dp/drho
        self.Gamma1 = self.rho / self.p * self.cs2

        # Newtonian gravitational potential: dΦ/dr = m/r², Φ(R) = -M/R
        self.R = float(r[-1])
        self.M = float(m[-1])
        dphi_dr = m / r**2
        phi_unnorm = scipy.integrate.cumulative_trapezoid(dphi_dr, r, initial=0.0)
        phi_surface = -self.M / self.R
        self.phi = phi_unnorm - phi_unnorm[-1] + phi_surface

    def interpolate(self, quantity: str):
        """
        Return a scipy interpolator for a stored profile array.

        Parameters
        ----------
        quantity : str
            Name of a profile attribute ('p', 'rho', 'm', 'phi', 'Gamma1', 'cs2').

        Returns
        -------
        scipy.interpolate.interp1d
        """
        arr = getattr(self, quantity)
        if arr is None:
            raise RuntimeError("Model has not been solved yet.")
        return scipy.interpolate.interp1d(self.r, arr, kind="cubic",
                                          bounds_error=False, fill_value=0.0)


# ---------------------------------------------------------------------------
# Relativistic stellar model (wraps Lindblom TOV solver)
# ---------------------------------------------------------------------------

class RelativisticStellarModel:
    """
    Stellar structure in general relativity, via the Lindblom (1992) form
    of the TOV equations solved in solve_tov.py.

    The Lindblom solver uses log-enthalpy h = ln(1 + integral dp/(e+p)) as
    the independent variable and the state vector

        [u, v, eta, v_b]

    where  u = r^2,  v = m(r)/r,  eta = tidal-deformability variable,
    v_b = M_baryon / r.  This class unpacks those variables onto a physical
    radial grid and computes the metric potentials needed by the mode solvers.

    Parameters
    ----------
    eos : object
        Equation of state (polytropic_eos, css_eos, or interpolated_eos from
        solve_tov.py, or any tabulated EOS accepted by solve_tov.py).
    rho_c : float
        Central baryon density (geometric units, consistent with eos).
    n_points : int
        Number of log-enthalpy grid points passed to lindblom_solver.
    termination_lnh : float
        Log-enthalpy value at which integration stops (surface proxy).
    """

    def __init__(
        self,
        eos,
        rho_c: float,
        *,
        n_points: int = 500,
        termination_lnh: float = 1e-14,
    ):
        # Wrap bare tabulated EOS if needed (mirrors solve_tov.solve_tov)
        if not isinstance(eos, (interpolated_eos, polytropic_eos, css_eos)):
            eos = interpolated_eos(eos)
        self.eos = eos
        self.rho_c = rho_c
        self.n_points = n_points
        self.termination_lnh = termination_lnh

        # Solved profile arrays (populated by solve())
        self.r: np.ndarray | None = None
        self.rho: np.ndarray | None = None
        self.p: np.ndarray | None = None
        self.e: np.ndarray | None = None
        self.m: np.ndarray | None = None
        self.lam: np.ndarray | None = None   # metric potential lambda
        self.nu: np.ndarray | None = None    # metric potential nu
        self.Gamma1: np.ndarray | None = None
        self.cs2: np.ndarray | None = None
        self.lnh: np.ndarray | None = None   # log-enthalpy grid
        self.R: float | None = None
        self.M: float | None = None
        self.C: float | None = None          # compactness M/R

        self.solve()

    def solve(self) -> None:
        """
        Call lindblom_solver and unpack the solution into physical profile arrays.

        Metric potentials
        -----------------
        The Lindblom state variable v = m(r)/r, so:

            e^{-2 lambda} = 1 - 2 v   =>   lambda = -0.5 * ln(1 - 2 v)

        The time-like potential nu is obtained by integrating from the surface
        inward.  At the surface r=R, continuity with the Schwarzschild exterior
        gives  nu(R) = 0.5 * ln(1 - 2 M/R).  Inside the star:

            d nu / d lnh = (m + 4 pi r^3 p) / (r (1-2m/r))  *  r / (d lnh / dr)^{-1}

        which in Lindblom variables simplifies to  d nu / d lnh = -1  plus
        a correction from the metric factor, giving the relation

            nu(r) = nu(R) - integral_r^R  d nu/dr  dr

        where  d nu/dr = (m + 4 pi r^3 p) / (r^2 (1 - 2 m/r)).
        The integration is done numerically on the radial grid.
        """
        lnhc = self.eos.logenthalpy_of_rho(np.array([self.rho_c]))[0]
        lnhs, sols = lindblom_solver(
            lnhc,
            self.eos,
            termination_lnh=self.termination_lnh,
            points_to_solve_for=self.n_points,
        )

        u = sols[:, 0]        # r^2
        v = sols[:, 1]        # m(r)/r

        r = np.sqrt(u)
        m = v * r

        self.lnh = lnhs
        self.r = r
        self.m = m
        self.rho = self.eos.rho_of_logenthalpy(lnhs)
        self.p = self.eos.p_of_logenthalpy(lnhs)
        self.e = self.eos.e_of_logenthalpy(lnhs)
        self.cs2 = self.eos.cs2_of_logenthalpy(lnhs)

        # Adiabatic index: Gamma1 = (rho + p)/p * cs2  in GR
        self.Gamma1 = (self.e + self.p) / self.p * self.cs2

        # Metric potential lambda
        self.lam = -0.5 * np.log(1.0 - 2.0 * v)

        # Metric potential nu: integrate d nu/dr outward then shift at surface
        self.nu = self._integrate_nu(r, m, self.p, v)

        self.R = float(r[-1])
        self.M = float(m[-1])
        self.C = self.M / self.R
    def _dnu_dr(self, r: float) -> float:
        """
        Compute d nu / dr at a scalar radius r using the TOV relation:

            d nu / dr = (m + 4 pi r^3 p) / (r^2 (1 - 2 m/r))
        """
        m_r = float(np.interp(r, self.r, self.m))
        p_r = float(np.interp(r, self.r, self.p))
        return (m_r + 4.0 * np.pi * r**3 * p_r) / (r**2 * (1.0 - 2.0 * m_r / r))

    def _integrate_nu(
        self,
        r: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """
        Compute nu(r) by integrating d nu/dr = (m + 4 pi r^3 p) / (r^2 (1-2m/r))
        from the centre outward, then shift so that nu(R) = 0.5 ln(1 - 2 M/R).
        """
        dnu_dr = (m + 4.0 * np.pi * r**3 * p) / (r**2 * (1.0 - 2.0 * v))
        nu_unnorm = scipy.integrate.cumulative_trapezoid(dnu_dr, r, initial=0.0)
        nu_surface = 0.5 * np.log(1.0 - 2.0 * self.M / self.R) if self.M is not None \
                     else 0.5 * np.log(1.0 - 2.0 * v[-1])
        return nu_unnorm - nu_unnorm[-1] + nu_surface

    # ------------------------------------------------------------------
    # Convenience interpolators (populated after solve())
    # ------------------------------------------------------------------

    def interpolate(self, quantity: str):
        """
        Return a scipy interpolator for a stored profile array.

        Parameters
        ----------
        quantity : str
            Name of a profile attribute, e.g. 'p', 'rho', 'e', 'm',
            'lam', 'nu', 'Gamma1', 'cs2'.

        Returns
        -------
        scipy.interpolate.interp1d
        """
        arr = getattr(self, quantity)
        if arr is None:
            raise RuntimeError("Model has not been solved yet.")
        return scipy.interpolate.interp1d(self.r, arr, kind="cubic",
                                          bounds_error=False, fill_value=0.0)
