"""
Stellar structure models for use with the oscillation mode solvers.

Classes:
  - NewtonianStellarModel : integrates the Newtonian structure equations
  - RelativisticStellarModel : wraps the Lindblom TOV solver in solve_tov.py
  - UniformDensityStellarModel, TolmanVIIStellarModel,
    PrescribedDensityStellarModel : relativistic stars without an EOS
    (closed form, or a prescribed e(r)), useful as benchmarks

All expose a common set of attributes so that mode solvers can be written
against a single interface.  The relativistic models also share the
exterior-aware accessors of RelativisticProfile (mass(r), pressure(r),
energy_density(r), nu_of_r(r), lam_of_r(r)).

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
import scipy.optimize

from .solve_relativistic import lindblom_solver
from .solve_newtonian_structure import newtonian_solver
from .analytic_eos import polytropic_eos, interpolated_eos, css_eos, piecewise_eos


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
# Relativistic profiles: shared accessors
# ---------------------------------------------------------------------------

class RelativisticProfile:
    """
    Accessors shared by every relativistic stellar model.

    A subclass sets the profile arrays r, m, p, e, nu (index 0 = centre, or
    the first point off it), then calls ``_build_profile()``.  The accessors

        mass(r), pressure(r), energy_density(r), nu_of_r(r), lam_of_r(r)

    are valid for all r > 0: inside the star they interpolate the profile
    (monotone cubic, PCHIP), outside they are the Schwarzschild values
    m = M, p = e = 0, nu = 0.5 ln(1 - 2M/r), lam = -0.5 ln(1 - 2M/r).
    (``nu`` and ``lam`` are the profile arrays, hence the ``_of_r`` names.)
    """

    def _build_profile(self) -> None:
        r = np.asarray(self.r, dtype=float)
        self.R = float(r[-1])
        self.M = float(self.m[-1])
        self.C = self.M / self.R
        with np.errstate(divide="ignore", invalid="ignore"):
            self.lam = np.where(r > 0, -0.5 * np.log(1.0 - 2.0 * np.asarray(self.m) / r), 0.0)
        # include the centre so the accessors are defined on [0, R]
        arrays = {"m": self.m, "p": self.p, "e": self.e, "nu": self.nu}
        if r[0] > 0:
            r = np.concatenate([[0.0], r])
            arrays = {key: np.concatenate([[0.0 if key == "m" else value[0]], value])
                      for key, value in arrays.items()}
        self._profile = {key: scipy.interpolate.PchipInterpolator(r, value, extrapolate=False)
                         for key, value in arrays.items()}

    def _inside(self, r, key, outside):
        r = np.asarray(r, dtype=float)
        inside = self._profile[key](np.clip(r, 0.0, self.R))
        return np.where(r <= self.R, inside, outside)

    def mass(self, r):
        """Enclosed gravitational mass m(r); M outside the star."""
        return self._inside(r, "m", self.M)

    def pressure(self, r):
        """Pressure p(r); 0 outside the star."""
        return self._inside(r, "p", 0.0)

    def energy_density(self, r):
        """Total energy density e(r); 0 outside the star."""
        return self._inside(r, "e", 0.0)

    def nu_of_r(self, r):
        """Metric potential nu(r), g_tt = -e^{2 nu}; Schwarzschild outside."""
        r = np.asarray(r, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            exterior = 0.5 * np.log(1.0 - 2.0 * self.M / r)
        return self._inside(r, "nu", exterior)

    def lam_of_r(self, r):
        """Metric potential lam(r) = -0.5 ln(1 - 2 m(r)/r), g_rr = e^{2 lam}."""
        r = np.asarray(r, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 0, -0.5 * np.log(1.0 - 2.0 * self.mass(r) / r), 0.0)

    _ACCESSORS = {"m": "mass", "p": "pressure", "e": "energy_density",
                  "nu": "nu_of_r", "lam": "lam_of_r"}

    def interpolate(self, quantity: str):
        """
        Return a callable r -> quantity(r) for a stored profile array.

        For 'm', 'p', 'e', 'nu' and 'lam' this is the exterior-aware accessor
        (m = M, nu and lam Schwarzschild, p = e = 0 for r > R).  Other
        profiles (e.g. 'rho', 'cs2', 'Gamma1') are interpolated inside the
        star and are 0 outside.
        """
        if quantity in self._ACCESSORS:
            return getattr(self, self._ACCESSORS[quantity])
        arr = getattr(self, quantity)
        if arr is None:
            raise RuntimeError(f"Model has no '{quantity}' profile.")
        return scipy.interpolate.interp1d(self.r, arr, kind="cubic",
                                          bounds_error=False, fill_value=0.0)


# ---------------------------------------------------------------------------
# Relativistic stellar model (wraps Lindblom TOV solver)
# ---------------------------------------------------------------------------

class RelativisticStellarModel(RelativisticProfile):
    """
    Stellar structure in general relativity, via the Lindblom (1992) form
    of the TOV equations solved in solve_relativistic.py.

    The Lindblom solver uses log-enthalpy h = ln(1 + integral dp/(e+p)) as
    the independent variable and the state vector

        [u, v, eta, v_b]

    where  u = r^2,  v = m(r)/r,  eta = tidal-deformability variable,
    v_b = M_baryon / r.  This class unpacks those variables onto a physical
    radial grid and computes the metric potentials needed by the mode solvers.

    The central state is given by exactly one of ``rho_c`` (central baryon
    density) or ``lnh_c`` (central log-enthalpy); ``from_logenthalpy`` and
    ``from_central_pressure`` are the corresponding constructors.

    Parameters
    ----------
    eos : object
        Equation of state (polytropic_eos, css_eos, or interpolated_eos from
        solve_tov.py, or any tabulated EOS accepted by solve_tov.py).
    rho_c : float
        Central baryon density (geometric units, consistent with eos).
    lnh_c : float
        Central log-enthalpy, instead of rho_c.
    n_points : int
        Number of log-enthalpy grid points passed to lindblom_solver.
    termination_lnh : float
        Log-enthalpy value at which integration stops (surface proxy).
    lnh_grid : array-like, optional
        Output grid of log-enthalpies, decreasing from lnh_c to the surface,
        instead of the default uniform grid of n_points.  A grid that is
        dense near lnh = 0 resolves the crust, e.g.
        ``np.geomspace(lnh_c, 1e-10, 2000)``.
    """

    def __init__(
        self,
        eos,
        rho_c: float | None = None,
        *,
        lnh_c: float | None = None,
        n_points: int = 500,
        termination_lnh: float = 1e-14,
        lnh_grid=None,
    ):
        if (rho_c is None) == (lnh_c is None):
            raise ValueError("give exactly one of rho_c and lnh_c")
        # Wrap bare tabulated EOS if needed (mirrors solve_tov.solve_tov)
        if not isinstance(eos, (interpolated_eos, polytropic_eos, css_eos, piecewise_eos)):
            eos = interpolated_eos(eos)
        self.eos = eos
        self.rho_c = rho_c
        self.lnh_c = (float(lnh_c) if lnh_c is not None
                      else float(self.eos.logenthalpy_of_rho(np.array([rho_c]))[0]))
        if self.rho_c is None:
            self.rho_c = float(self.eos.rho_of_logenthalpy(np.array([self.lnh_c]))[0])
        self.n_points = n_points
        self.termination_lnh = termination_lnh
        self.lnh_grid = lnh_grid

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

    @classmethod
    def from_logenthalpy(cls, eos, lnh_c: float, **kwargs):
        """Star with central log-enthalpy lnh_c."""
        return cls(eos, lnh_c=lnh_c, **kwargs)

    @classmethod
    def from_central_pressure(cls, eos, p_c: float, **kwargs):
        """Star with central pressure p_c (geometric units)."""
        if not isinstance(eos, (interpolated_eos, polytropic_eos, css_eos, piecewise_eos)):
            eos = interpolated_eos(eos)
        lnh_c = float(np.atleast_1d(eos.logenthalpy_of_p(np.array([p_c])))[0])
        return cls(eos, lnh_c=lnh_c, **kwargs)

    @property
    def central_pressure(self) -> float:
        return float(self.p[0])

    @property
    def central_energy_density(self) -> float:
        return float(self.e[0])

    def solve(self) -> None:
        """
        Call lindblom_solver and unpack the solution into physical profile arrays.

        Metric potentials
        -----------------
        The Lindblom state variable v = m(r)/r, so:

            e^{-2 lambda} = 1 - 2 v   =>   lambda = -0.5 * ln(1 - 2 v)

        For a barotropic EOS the time-like potential follows exactly from the
        log-enthalpy (hydrostatic equilibrium, d nu = -d lnh):

            nu(r) = nu(R) - [lnh(r) - lnh(R)],   nu(R) = 0.5 ln(1 - 2 M/R),

        so no quadrature of d nu / dr is needed.
        """
        lnhs, sols = lindblom_solver(
            self.lnh_c,
            self.eos,
            termination_lnh=self.termination_lnh,
            points_to_solve_for=self.n_points,
            lnh_grid=self.lnh_grid,
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

        R, M = float(r[-1]), float(m[-1])
        self.nu = 0.5 * np.log(1.0 - 2.0 * M / R) - (lnhs - lnhs[-1])
        self._build_profile()

    def _dnu_dr(self, r: float) -> float:
        """
        Compute d nu / dr at a scalar radius r using the TOV relation:

            d nu / dr = (m + 4 pi r^3 p) / (r^2 (1 - 2 m/r))
        """
        m_r = float(np.interp(r, self.r, self.m))
        p_r = float(np.interp(r, self.r, self.p))
        return (m_r + 4.0 * np.pi * r**3 * p_r) / (r**2 * (1.0 - 2.0 * m_r / r))


# ---------------------------------------------------------------------------
# Analytic and prescribed-density models (no EOS)
# ---------------------------------------------------------------------------

class PrescribedDensityStellarModel(RelativisticProfile):
    """
    TOV star with a prescribed energy-density profile e(r) on [0, R], no
    EOS needed.  The central pressure is found by shooting on p(R) = 0, and
    nu by quadrature of d nu / dr (there is no enthalpy to use).

    rho, cs2 and Gamma1 are not defined (None).

    Parameters
    ----------
    e_of_r : callable
        Energy density as a function of r.
    R : float
        Stellar radius.
    n_points : int
        Number of radial grid points.
    p_c_bracket : (float, float), optional
        Bracket on ln p_c for the shooting; default
        (ln(1e-8 e_c), ln(1e3 e_c)).
    """

    rho = cs2 = Gamma1 = None

    def __init__(self, e_of_r, R: float, *, n_points: int = 4000, p_c_bracket=None):
        r = np.linspace(0.0, R, n_points)

        def rhs(x, y):
            m, p = y
            e = e_of_r(x)
            if x == 0.0:
                return [0.0, 0.0]
            return [4.0 * np.pi * x**2 * e,
                    -(e + p) * (m + 4.0 * np.pi * x**3 * p) / (x * (x - 2.0 * m))]

        def integrate(p_c):
            r0 = 1e-8 * R
            y0 = [4.0 / 3.0 * np.pi * r0**3 * e_of_r(0.0), p_c]
            return scipy.integrate.solve_ivp(rhs, (r0, R), y0, t_eval=np.r_[r0, r[1:]],
                                             rtol=1e-11, atol=1e-14, method="DOP853")

        def surface_pressure(log_pc):
            sol = integrate(np.exp(log_pc))
            if sol.status != 0 or sol.y.shape[1] < len(r):
                return 1.0  # diverged: central pressure too large
            return sol.y[1, -1] / np.exp(log_pc)

        e_c = e_of_r(0.0)
        lo, hi = p_c_bracket or (np.log(1e-8 * e_c), np.log(1e3 * e_c))
        log_pc = scipy.optimize.brentq(surface_pressure, lo, hi, xtol=1e-14)
        m, p = integrate(np.exp(log_pc)).y
        self.r = r
        self.m = m
        self.p = np.maximum(p, 0.0)
        self.e = np.array([e_of_r(x) for x in r])
        M = float(m[-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            dnu = np.where(r > 0, (m + 4.0 * np.pi * r**3 * self.p) / (r * (r - 2.0 * m)), 0.0)
        nu = scipy.integrate.cumulative_trapezoid(dnu, r, initial=0.0)
        self.nu = nu - nu[-1] + 0.5 * np.log(1.0 - 2.0 * M / R)
        self._build_profile()

    @property
    def central_pressure(self) -> float:
        return float(self.p[0])

    @property
    def central_energy_density(self) -> float:
        return float(self.e[0])


class TolmanVIIStellarModel(PrescribedDensityStellarModel):
    """
    Tolman VII star, e(r) = e_c (1 - r^2 / R^2), with M = (8 pi / 15) e_c R^3,
    for a given compactness.  The pressure is obtained numerically (the
    closed form of Neary, Ishak & Lake 2001 would also do), which doubles as
    a check of the solver.
    """

    def __init__(self, R_over_M: float, *, M: float = 1.0, n_points: int = 4000):
        R = R_over_M * M
        e_c = 15.0 * M / (8.0 * np.pi * R**3)
        super().__init__(lambda r: e_c * (1.0 - (r / R) ** 2), R, n_points=n_points)


class UniformDensityStellarModel(RelativisticProfile):
    """
    Uniform-density star (Schwarzschild interior solution), in closed form.
    Requires R/M > 9/4 (Buchdahl).

    rho, cs2 and Gamma1 are not defined (None).
    """

    rho = cs2 = Gamma1 = None

    def __init__(self, R_over_M: float, *, M: float = 1.0, n_points: int = 4000):
        if R_over_M <= 9.0 / 4.0:
            raise ValueError("a uniform-density star requires R/M > 9/4")
        R = R_over_M * M
        e0 = 3.0 * M / (4.0 * np.pi * R**3)
        r = np.linspace(0.0, R, n_points)
        a = np.sqrt(1.0 - 2.0 * M * r**2 / R**3)
        b = np.sqrt(1.0 - 2.0 * M / R)
        self.r = r
        self.m = M * (r / R) ** 3
        self.p = e0 * (a - b) / (3.0 * b - a)
        self.e = np.full_like(r, e0)
        self.nu = np.log(1.5 * b - 0.5 * a)
        self._build_profile()

    @property
    def central_pressure(self) -> float:
        return float(self.p[0])

    @property
    def central_energy_density(self) -> float:
        return float(self.e[0])
