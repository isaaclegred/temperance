"""
Radial and nonradial oscillation modes of neutron stars.

Two top-level modules:
  - NewtonianModes:    modes in Newtonian gravity (Cowling or full)
  - RelativisticModes: modes in full GR (fluid + spacetime perturbations)

Within NewtonianModes there is a helper class NewtonianTidalOverlap for
computing overlap integrals of eigenfunctions with the external tidal field.

Convention for mode labels
--------------------------
ell : int >= 0
    Angular degree. ell=0 -> radial, ell=1 -> dipole (spurious in Newtonian),
    ell>=2 -> nonradial (f, p, g modes).
m : int, |m| <= ell
    Azimuthal order.  Frequencies are m-degenerate for non-rotating stars,
    but m is kept for future extension to rotating stars.
n : int >= 0
    Radial order.  n=0 is the fundamental (f) mode, n>=1 are p-modes.
    Negative n labels g-modes in some conventions (not used here).

For GR only: w-modes (purely spacetime) are returned when ell >= 2 and the
solver is run in spacetime-perturbation mode.  Axial (odd-parity) w-modes
trapped in the potential well of an ultracompact star are computed by
RelativisticModes.trapped_axial_modes; strongly damped curvature w-modes and
polar w-modes are not implemented yet.

Units: frequencies are in the units of the stellar model (geometric, 1/Msun
for temperance's models); convert with temperance.utilities.units
(OMEGA_TO_HZ for omega, MSUN_TO_S for damping times).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
import scipy.integrate
import scipy.interpolate
import scipy.optimize
from dataclasses import dataclass, field, replace
from typing import Optional

from .stellar_models import NewtonianStellarModel, RelativisticStellarModel


# ---------------------------------------------------------------------------
# Shared result types
# ---------------------------------------------------------------------------

@dataclass
class OscillationMode:
    """Container for a single oscillation mode solution."""
    ell: int
    m: int
    n: int
    omega: complex                     # angular frequency (units of the stellar model)
    frequency: float                   # Re(omega) / (2 pi), same units
    damping_time: Optional[float]      # 1 / |Im(omega)|, same units (None if undamped)
    xi_r: Optional[np.ndarray] = None  # radial displacement eigenfunction on radial grid
    xi_h: Optional[np.ndarray] = None  # horizontal displacement eigenfunction
    pressure_perturbation: Optional[np.ndarray] = None
    radial_grid: Optional[np.ndarray] = None  # r-coordinates for eigenfunctions
    mode_type: str = "unknown"         # "f", "p", "g", "w", or "radial"
    solver_info: dict = field(default_factory=dict)
    quality_factor: Optional[float] = None   # Re(omega) / (-2 Im(omega)) for damped modes
    trapped: Optional[bool] = None           # Re(omega)^2 inside a potential well (axial w-modes)

    @property
    def is_radial(self) -> bool:
        return self.ell == 0


@dataclass
class TidalOverlapResult:
    """Container for a tidal overlap integral."""
    ell: int
    n: int
    Q_nl: float          # dimensionless overlap integral Q_{n ell}
    kappa_nl: float      # effective tidal coupling constant
    mode: OscillationMode


@dataclass
class PotentialWell:
    """
    A well in the axial potential V_l(r): a minimum bounded on the outside
    by a barrier maximum.  Modes with V_min < Re(omega)^2 < V_max are
    trapped, leaking out only by tunnelling through the barrier.
    """
    exists: bool
    r_min: float = np.nan       # radius of the well minimum
    V_min: float = np.nan
    r_max: float = np.nan       # radius of the bounding barrier maximum
    V_max: float = np.nan
    barrier_location: str = ""  # "interior" / "exterior"


def _count_nodes(y: np.ndarray) -> int:
    """Count zero-crossings (nodes) in array y, ignoring exact zeros."""
    signs = np.sign(y[y != 0.0])
    return int(np.sum(np.diff(signs) != 0))

# ---------------------------------------------------------------------------
# Newtonian module
# ---------------------------------------------------------------------------

class NewtonianModes:
    """
    Compute oscillation modes of a neutron star in Newtonian gravity.

    The fluid perturbation equations (Cowling approximation or full, depending
    on ``cowling``) are solved as an eigenvalue problem on the stellar
    background provided by a TOV solution or Newtonian structure solution.

    Parameters
    ----------
    stellar_model : NewtonianStellarModel
        Solved Newtonian structure model providing r, rho, p, Gamma1, phi.
    cowling : bool
        If True, freeze gravitational potential perturbations (Cowling
        approximation).  Faster but less accurate, especially for g-modes.
    """

    def __init__(self, stellar_model: NewtonianStellarModel, *, cowling: bool = True):
        self.model = stellar_model
        self.cowling = cowling
        self._build_interpolators()

    def _build_interpolators(self) -> None:
        """Build cubic interpolators for all background profile quantities."""
        m = self.model
        kw = dict(kind="linear", bounds_error=False, fill_value=0.0)
        self._rho_i    = scipy.interpolate.interp1d(m.r, m.rho,    **kw)
        self._p_i      = scipy.interpolate.interp1d(m.r, m.p,      **kw)
        self._m_i      = scipy.interpolate.interp1d(m.r, m.m,      **kw)
        self._cs2_i    = scipy.interpolate.interp1d(m.r, m.cs2,    **kw)
        self._Gamma1_i = scipy.interpolate.interp1d(m.r, m.Gamma1, **kw)

    # ------------------------------------------------------------------
    # Radial modes  (ell = 0)
    # ------------------------------------------------------------------

    def radial_mode_rhs(self, r: float, Z: np.ndarray, omega2: float, version_1: bool = False) -> list:
        """
        RHS of the Newtonian radial oscillation ODE.

        Cowling (self.cowling=True): state = [y1, y2], delta_phi frozen to zero.
        Full    (self.cowling=False): state = [y1, y2, y4], y4 = d(delta_phi)/dr.

        Variables
        ---------
        y1 = xi_r                      (radial displacement)
        y2 = -delta_p                  (Gamma1*p * div xi, Lagrangian pressure pert.)
        y4 = d(delta_phi)/dr           (gradient of gravitational potential pert.)

        Equations
        ---------
            dy1/dr = y2 / (Gamma1*p) - 2*y1/r
            dy2/dr = + 4/r * dp/dr * y1 - omega^2 * rho * y1 [- 4*pi*rho^2*y1 if Cowling]
            dy4/dr = 4*pi*(rho*g/cs^2*y1 - rho/(Gamma1*p)*y2) - 2*y4/r
                                                       [only in full case]

        The 4*pi*rho term in dy2 comes from d(g)/dr in the *background* Poisson
        equation and is present in both Cowling and full versions.
        """
        rho = float(self._rho_i(r))
        p   = float(self._p_i(r))
        G1  = float(self._Gamma1_i(r))
        m_r = float(self._m_i(r))
        cs2 = float(self._cs2_i(r))

        g   = m_r / r**2
        G1p = G1 * p

        y1, y2 = Z[0], Z[1]
        y4 = 0.0 if self.cowling else float(Z[2])

        dy1 = y2 / G1p - 2.0 * y1 / r
        dy2 =  -4/r * g * rho * y1 - rho * omega2 * y1 

        if self.cowling:
            return [dy1, dy2 - 4 * np.pi * rho**2 * y1]

        delta_rho = rho * g / cs2 * y1 - rho / G1p * y2
        dy4 = 4.0 * np.pi * delta_rho - 2.0 * y4 / r
        return [dy1, dy2, dy4]

    def _shoot_radial(self, omega2: float):
        """
        Integrate radial equations from center to surface for trial omega^2.

        Returns
        -------
        r_grid  : ndarray
        y1      : ndarray   xi_r on r_grid
        y2_surf : float     -delta_p at r=R  (eigenvalue condition: must be 0)
        """
        m = self.model
        r0, r1 = m.r[0], m.r[-2]   # start just off centre, stop just inside surface

        rho_c = float(self._rho_i(r0))
        G1_c  = float(self._Gamma1_i(r0))
        p_c   = float(self._p_i(r0))

        # Near-centre Taylor expansion (xi_r = r, div xi = 3 => y2 = 3 Gamma1 p):
        y1_0 = r0
        y2_0 = 3.0 * G1_c * p_c
        # delta_phi ~ const - (2*pi*rho_c/3)*r^2, so y4 = d(delta_phi)/dr ~ -4*pi*rho_c*r
        ic = [y1_0, y2_0] if self.cowling else [y1_0, y2_0, -4.0 * np.pi * rho_c * r0]

        sol = scipy.integrate.solve_ivp(
            lambda r, Z: self.radial_mode_rhs(r, Z, omega2),
            [r0, r1], ic,
            method="RK45", dense_output=False,
            # rtol=1e-9, atol=1e-11,
            # max_step=(r1 - r0) / 200,
        )
        return sol.t, sol.y[0], float(sol.y[1, -1])

    def find_single_radial_mode(self, omega_guess: float) -> OscillationMode:
        """
        Find the radial mode nearest to omega_guess by root-finding on the
        surface boundary condition y2(R) = 0.  This corresponds to Delta p = 0 at the surface.

        Parameters
        ----------
        omega_guess : float
            Initial guess for the angular eigenfrequency (geometric units).

        Returns
        -------
        OscillationMode
            ell=0, m=0; n is inferred from node count of xi_r.
        """
        omega2_guess = omega_guess**2

        def bc(omega2):
            _, _, y2s = self._shoot_radial(omega2)
            print(f"omega^2 = {omega2:.4g}, surface BC = {y2s:.4g}")
            return y2s

        # Bracket by stretching the guess ±50 %
        lo, hi = 0.5 * omega2_guess, 1.5 * omega2_guess
        # Widen until we have a sign change
        for _ in range(20):
            if bc(lo) * bc(hi) < 0:
                break
            lo *= 0.5
            hi *= 2.0
        else:
            raise RuntimeError(
                f"Could not bracket a root near omega_guess={omega_guess:.4g}. "
                "Try a different initial guess."
            )

        omega2_found = scipy.optimize.brentq(bc, lo, hi, xtol=1e-16, rtol=1e-10)
        r_grid, y1, _ = self._shoot_radial(omega2_found)
        n_nodes = _count_nodes(y1)
        print(f"Found root: omega^2 = {omega2_found:.6g}, n_nodes = {n_nodes}")
        omega_found = np.sqrt(complex(omega2_found))

        return OscillationMode(
            ell=0, m=0, n=n_nodes,
            omega=complex(omega_found),
            frequency=omega_found / (2.0 * np.pi),
            damping_time=None,
            xi_r=y1 / np.abs(y1).max(),
            radial_grid=r_grid,
            mode_type="radial",
            solver_info={"omega2": omega2_found},
        )

    def compute_radial_mode(self, n: int, n_scan: int = 300) -> OscillationMode:
        """
        Compute the n-th Newtonian radial oscillation mode (ell=0).

        Scans omega^2 from near zero up to a generous upper bound, locates all
        sign changes in the surface boundary condition y2(R), then refines the
        n-th bracket with Brent's method.  The radial order is verified by
        counting nodes in xi_r.

        Parameters
        ----------
        n : int >= 0
            Radial order.  n=0 has no nodes (fundamental), n=1 has one node, …
        n_scan : int
            Number of trial frequencies used in the initial scan.

        Returns
        -------
        OscillationMode
            ell=0, m=0, mode_type='radial'.
        """
        m = self.model
        # Characteristic dynamical frequency scale
        omega_char = float(np.sqrt(m.M / m.R**3))
        # This could be bad if very close to the maximum mass, some modes will flip sign
        omega2_min = -omega_char**2
        # Upper bound: empirically (n+2)*pi overtone multiples with a safety factor
        omega2_max = ((n + 3) * np.pi * omega_char) ** 2 * 4.0

        omega2_scan = np.linspace(omega2_min, omega2_max, n_scan)

        def bc(omega2):
            _, _, y2s = self._shoot_radial(omega2)
            return y2s

        bc_vals = np.array([bc(o2) for o2 in omega2_scan])

        # Find brackets where bc changes sign
        brackets = []
        for i in range(len(bc_vals) - 1):
            if np.isfinite(bc_vals[i]) and np.isfinite(bc_vals[i + 1]):
                if bc_vals[i] * bc_vals[i + 1] < 0.0:
                    brackets.append((omega2_scan[i], omega2_scan[i + 1]))

        if n >= len(brackets):
            raise ValueError(
                f"Mode n={n} not found: only {len(brackets)} mode(s) detected "
                f"in omega^2 range [{omega2_min:.3g}, {omega2_max:.3g}]. "
                "Increase n_scan or increase the scan range via omega2_max."
            )

        lo, hi = brackets[n]
        omega2_found, diagnostics = scipy.optimize.brentq(bc, lo, hi, xtol=1e-16, rtol=1e-10, full_output=True)
        r_grid, y1, _ = self._shoot_radial(omega2_found)
        n_nodes = _count_nodes(y1)
        omega_found = np.sqrt(complex(omega2_found))

        return OscillationMode(
            ell=0, m=0, n=n_nodes,
            omega=complex(omega_found),
            frequency=omega_found / (2.0 * np.pi),
            damping_time=None,
            xi_r=y1 / np.abs(y1).max(),
            radial_grid=r_grid,
            mode_type="radial",
            solver_info={"omega2": omega2_found, "n_nodes_found": n_nodes},
        )

    def compute_radial_spectrum(self, n_max: int) -> list[OscillationMode]:
        """Return the first n_max+1 radial modes (n=0 … n_max)."""
        # This implementation is bad: it should look for modes and regardless of 
        # which it finds store it as long as it hasn't already been found.
        return [self.compute_radial_mode(n) for n in range(n_max + 1)]

    # ------------------------------------------------------------------
    # Nonradial modes  (ell >= 2)
    # ------------------------------------------------------------------

    def _nonradial_central_ic(self, x0: float, ell: int, config=1) -> list:
        """
        Near-centre Taylor initial conditions for the Newtonian nonradial ODE.

        The regular solution near r = 0 behaves as r^ell for the displacement.
        For the Cowling approximation (self.cowling=True) the state is [y1, y2];
        for the full system (self.cowling=False) the state is [y1, y2, y3, y4].

        Variables follow the Christensen-Dalsgaard convention (see
        compute_nonradial_mode docstring):
            y1 = xi_r / r
            y2 = ell*(ell+1)/R * xi_h     (horizontal displacement combination)
            y3 = -x phi' / (g r)          (potential perturbation; full only)
            y4 = x^2 d/dx(y3/x)           (potential gradient; full only)

        Returns
        -------
        list of float
            Initial condition vector at r = r0.
        """
        if config == 1:
            return np.array([1, ell+1, 0, 0]) * x0**(ell-1)
        elif config == 2:
            return np.array([1, ell+1, 1, ell-2]) * x0**(ell-1)
        
        raise ValueError(f"Invalid config {config} for nonradial central IC.")
    def _nonradial_boundary_ic(self, x1: float, ell: int, config: int, sigma: float) -> list:
        """
        Near-surface Taylor initial conditions for the Newtonian nonradial ODE.

        The regular solution near r = R behaves as (R-r)^s with s = l-1.  The
        state vector is the same as for the central IC.

        Returns
        -------
        list of float
            Initial condition vector at r = r1.
        """
        if config == 1:
            return np.array([1, ell * (ell+1) / sigma**2, 0, 0])
        elif config == 2:
            return np.array([1, 0.0, 1, -ell])
        
        raise ValueError(f"Invalid config {config} for nonradial boundary IC.")
    

    def nonradial_mode_rhs(self, x: float, Z: np.ndarray, omega2: float, ell: int) -> list:
        """
        RHS of the Newtonian nonradial oscillation ODE system.

        Independent variable is x = r/R in [0, 1].  Internally r = x*R.

        Cowling (self.cowling=True):  state Z = [y1, y2],       phi' frozen.
        Full    (self.cowling=False): state Z = [y1, y2, y3, y4].

        Variables (Christensen-Dalsgaard convention):
            y1 = xi_r / r
            y2 = ell*(ell+1)/R * xi_h
            y3 = -x phi' / (g r)          (full only)
            y4 = x^2 d/dx(y3/x)           (full only)

        sigma^2 = omega^2 R^3 / (G M) is the dimensionless eigenfrequency;
        omega2 = sigma^2 * G*M/R^3 is passed directly.

        Parameters
        ----------
        x : float
            Dimensionless radius x = r/R in [0, 1].
        Z : ndarray
            Current state vector (length 2 for Cowling, 4 for full).
        omega2 : float
            Trial angular eigenfrequency squared.
        ell : int
            Angular degree (>= 2).

        Returns
        -------
        list of float
            dZ/dx evaluated at x.
        """
        R   = self.model.R
        r   = x * R

        rho = float(self._rho_i(r))
        p   = float(self._p_i(r))
        G1  = float(self._Gamma1_i(r))
        m_r = float(self._m_i(r))
        cs2 = float(self._cs2_i(r))

        # Christensen-Dalsgaard structure coefficients (G=1, Newtonian)
        # V_g = -1/Gamma1 * dlnp/dlnr = G m rho / (Gamma1 p r)
        V_g = m_r * rho / (G1 * p * r)
        # U = c1 = 4 pi r^3 rho / m
        U   = 4.0 * np.pi * r**3 * rho / m_r
        # A = 1/Gamma1 dlnp/dlnr - dlnrho/dlnr; using HSE and drho/dr = dp/(cs2 dr)
        A   = m_r / r * (1.0 / cs2 - rho / (G1 * p))
        # eta = l(l+1) g / (omega^2 r) = l(l+1) m / (omega^2 r^3)
        ll1 = ell * (ell + 1)
        eta = ll1 * m_r / (omega2 * r**3)

       

        # Equations (11-14): x dy/dx = (...), so dy/dx = (...) / x
        if self.cowling:
            y1, y2 = Z[0], Z[1]
            dy1 = ((V_g - 2.0) * y1 + (1.0 - V_g / eta) * y2) / x
            dy2 = ((ll1 - eta * A) * y1 + (A - 1.0) * y2) / x
            return [dy1, dy2]

        y1, y2, y3, y4 = Z[0], Z[1], Z[2], Z[3]
        dy1 = ((V_g - 2.0) * y1 + (1.0 - V_g / eta) * y2 - V_g * y3) / x
        dy2 = ((ll1 - eta * A) * y1 + (A - 1.0) * y2 + eta * A * y3) / x
        dy3 = (y3 + y4) / x
        dy4 = (-A * U * y1 - U * V_g / eta * y2
               + (ll1 + U * (A - 2.0) + U * V_g) * y3
               + 2.0 * (1.0 - U) * y4) / x
        return [dy1, dy2, dy3, dy4]

    def _shoot_nonradial(self, omega2: float, ell: int, reconstruct: bool = False):
        """
        Integrate nonradial equations from centre to surface for trial omega^2.

        Construct 4 solutions:
        Config 1 and 2 from the center
        and Config 1 and 2 from the surface.
        Then match them at an intermediate point (arbitrary, but for simplicity half the radius)
        Parameters
        ----------
        omega2 : float
            Trial angular eigenfrequency squared.
        ell : int
            Angular degree.
        reconstruct : bool
            If True and not Cowling, find the null-space linear combination of
            the two center solutions that satisfies the surface BCs and
            integrate it from centre to surface.  Should be True only when
            called at a converged eigenvalue (to recover the eigenfunction).
            Ignored in the Cowling approximation (always correct there).

        Returns
        -------
        x_grid  : ndarray   dimensionless radii x = r/R from the center integration
        y1      : ndarray   xi_r / r on x_grid  (center config-1 solution)
        bc      : float     eigenvalue condition:
                            Cowling -> y2 at the surface (must vanish)
                            Full    -> det of normalized 4x4 matching matrix (must vanish)
        """
        m    = self.model
        R    = m.R
        M    = m.M

        x0 = m.r[0]  / R    # dimensionless center start (just off r=0)
        x1 = m.r[-2] / R    # dimensionless surface stop  (just inside R)
        x_m = 0.5            # matching point

        ode = lambda x, Z: self.nonradial_mode_rhs(x, Z, omega2, ell)

        # ------------------------------------------------------------------
        # Cowling approximation: simple center -> surface shooting.
        # Eigenvalue condition: y2(x=x1) = 0  (Delta p = 0 at surface).
        # ------------------------------------------------------------------
        if self.cowling:
            ic = self._nonradial_central_ic(x0, ell, config=1)[:2]
            sol = scipy.integrate.solve_ivp(
                ode, [x0, x1], ic, method="RK45",
            )
            return sol.t, sol.y[0], float(sol.y[1, -1])

        # ------------------------------------------------------------------
        # Full system: 2-sided matching at x_m.
        #
        # Dimensionless frequency sigma^2 = omega^2 R^3 / (G M).
        # Use abs so the IC formula sigma^-2 stays finite while scanning
        # negative omega^2 (g-mode / stability regime).
        # ------------------------------------------------------------------
        sigma = np.sqrt(abs(omega2) * R**3 / M)

        # Center config 1: integrate to surface with dense output so we can
        # cheaply evaluate the state at x_m without a separate integration.
        ic_c1  = self._nonradial_central_ic(x0,  ell, config=1)
        sol_c1 = scipy.integrate.solve_ivp(
            ode, [x0, x1], ic_c1, method="RK45", dense_output=True,
        )
        Z_c1 = sol_c1.sol(x_m)   # 4-vector at matching point

        # Center config 2: integrate only to matching point
        ic_c2  = self._nonradial_central_ic(x0,  ell, config=2)
        sol_c2 = scipy.integrate.solve_ivp(
            ode, [x0, x_m], ic_c2, method="RK45",
        )
        Z_c2 = sol_c2.y[:, -1]

        # Surface configs: integrate backward from surface to matching point
        ic_s1  = self._nonradial_boundary_ic(x1, ell, config=1, sigma=sigma)
        sol_s1 = scipy.integrate.solve_ivp(
            ode, [x1, x_m], ic_s1, method="RK45",
        )
        Z_s1 = sol_s1.y[:, -1]

        ic_s2  = self._nonradial_boundary_ic(x1, ell, config=2, sigma=sigma)
        sol_s2 = scipy.integrate.solve_ivp(
            ode, [x1, x_m], ic_s2, method="RK45",
        )
        Z_s2 = sol_s2.y[:, -1]

        # ------------------------------------------------------------------
        # Matching matrix: columns are the 4 solution vectors at x_m.
        # Eigenvalue condition: det = 0.
        # Normalize each column to prevent scale differences from swamping
        # the determinant.
        # ------------------------------------------------------------------
        mat   = np.column_stack([Z_c1, Z_c2, Z_s1, Z_s2])
        norms = np.linalg.norm(mat, axis=0)
        norms[norms == 0.0] = 1.0
        det   = np.linalg.det(mat / norms)

        if not reconstruct:
            return sol_c1.t, sol_c1.y[0], det

        # Reconstruct the true eigenfunction: find the null vector of the
        # (column-normalised) matching matrix.  The null vector v gives
        # coefficients for [c1, c2, s1, s2]; mapping back through the column
        # norms gives the weights for the original (un-normalised) solutions.
        _, _, Vt = np.linalg.svd(mat / norms)
        print(mat)
        print("norms are", norms)
        v = Vt[-1]                          # approximate null vector
        a1 = v[0] / norms[0]
        a2 = v[1] / norms[1]

        ic_eigen = a1 * np.asarray(ic_c1, dtype=float) + a2 * np.asarray(ic_c2, dtype=float)
        sol_eigen = scipy.integrate.solve_ivp(ode, [x0, x1], ic_eigen, method="RK45")
        return sol_eigen.t, sol_eigen.y[0], det

    def find_single_nonradial_mode(self, ell: int, m: int, omega_guess: float) -> OscillationMode:
        """
        Find the nonradial mode nearest to omega_guess by root-finding on the
        surface boundary condition.

        Brackets the root by stretching the initial guess, then refines with
        Brent's method.  The radial order n is inferred from the node count
        of xi_r after convergence.

        Parameters
        ----------
        ell : int >= 2
            Angular degree.
        m : int
            Azimuthal order (|m| <= ell); stored in result, does not affect
            the eigenvalue for non-rotating stars.
        omega_guess : float
            Initial guess for the angular eigenfrequency (rad/s or geometric
            units consistent with the stellar model).

        Returns
        -------
        OscillationMode
            ell, m as given; n inferred from node count; damping_time=None.
        """
        omega2_guess = omega_guess ** 2

        def match(omega2):
            _, _, val = self._shoot_nonradial(omega2, ell)
            return val

        lo, hi = 0.5 * omega2_guess, 1.5 * omega2_guess
        for _ in range(20):
            if match(lo) * match(hi) < 0:
                break
            lo *= 0.5
            hi *= 2.0
        else:
            raise RuntimeError(
                f"Could not bracket a root near omega_guess={omega_guess:.4g}. "
                "Try a different initial guess."
            )

        omega2_found = scipy.optimize.brentq(match, lo, hi, xtol=1e-16, rtol=1e-10)

        x_grid, y1, _ = self._shoot_nonradial(omega2_found, ell, reconstruct=True)
        n_nodes = _count_nodes(y1)
        omega_found = np.sqrt(complex(omega2_found))
        mode_type = "f" if n_nodes == 0 else "p"

        return OscillationMode(
            ell=ell, m=m, n=n_nodes,
            omega=complex(omega_found),
            frequency=omega_found / (2.0 * np.pi),
            damping_time=None,
            xi_r=y1 / np.abs(y1).max(),
            radial_grid=x_grid * self.model.R,
            mode_type=mode_type,
            solver_info={"omega2": omega2_found},
        )

    def compute_nonradial_mode(
        self, ell: int, m: int, n: int, *, mode_family: str = "p"
    ) -> OscillationMode:
        """
        Compute a nonradial oscillation mode with given (ell, m, n).

        Solves the coupled system for fluid perturbations (xi_r, xi_h, p').
        In the Cowling approximation this reduces to the two-variable
        Cowling equations; otherwise the full four-variable system including
        the potential perturbation phi' is used.

        Follow the approach outlined in https://arxiv.org/pdf/0710.3106 (Christensen-Dalsgaard 2007)

        Define the following variables
        x = r/R
        A1 = q/x^3 [q = m(r)/M]
        A2 = V_g = -1/Gamma1 * dlnp/dlnr = G * m * rho / (Gamma1 * p * r)
        A3 = Gamma1
        A4 = A = 1/Gamma1 * dlnp/dlnr - dlnrho/dlnr = N^2 / (g/R)  ( N the Brunt-Vaisala frequency)
        A5 = c1 = 4 pi * r^3 * rho / m

        The system consists of the following ODEs for the state vector Y = [y1, y2, y3, y4] where
        y1 = xi_r / r
        y2 = x * (p'/rho + phi') * (l(l+1) / omega^2 r^2) = (l)(l+1) / R * xi_h
        y3 = -x phi'/(gr)
        y4 = x^2 d/dx(y3/x)

        omega^2 = GM/R^3 * sigma^2  defines the dimensionless frequnecy sigma

        
        Parameters
        ----------
        ell : int >= 2
            Angular degree.
        m : int
            Azimuthal order; must satisfy |m| <= ell.  Frequency is
            m-independent for non-rotating stars; the parameter is stored
            in the result for bookkeeping.
        n : int >= 0
            Radial order.  n=0 selects the fundamental (f) mode,
            n=1,2,... select p-modes.  Negative values may be used
            for g-mode families in future extensions.
        mode_family : {"p", "g", "f"}
            Which branch of the spectrum to target.  "f" is equivalent
            to n=0 on the p-branch.

        Returns
        -------
        OscillationMode
        """
        if ell < 2:
            raise ValueError(
                f"Nonradial modes require ell >= 2; got ell={ell}. "
                "Use compute_radial_mode for ell=0."
            )
        if abs(m) > ell:
            raise ValueError(f"|m|={abs(m)} > ell={ell}")

        mod = self.model
        omega_char = float(np.sqrt(mod.M / mod.R**3))
        omega2_min = 1e-4 * omega_char**2
        omega2_max = ((n + 3) * np.pi * omega_char) ** 2 * 4.0

        n_scan = 300
        omega2_scan = np.linspace(omega2_min, omega2_max, n_scan)

        def match(omega2):
            _, _, val = self._shoot_nonradial(omega2, ell)
            return val

        match_vals = np.array([match(o2) for o2 in omega2_scan])

        brackets = []
        for i in range(len(match_vals) - 1):
            if np.isfinite(match_vals[i]) and np.isfinite(match_vals[i + 1]):
                if match_vals[i] * match_vals[i + 1] < 0.0:
                    brackets.append((omega2_scan[i], omega2_scan[i + 1]))

        if n >= len(brackets):
            raise ValueError(
                f"Mode n={n} not found: only {len(brackets)} mode(s) detected "
                f"in omega^2 range [{omega2_min:.3g}, {omega2_max:.3g}]. "
                "Increase n_scan or widen the scan range."
            )

        lo, hi = brackets[n]
        omega2_found, diagnostics = scipy.optimize.brentq(
            match, lo, hi, xtol=1e-16, rtol=1e-10, full_output=True
        )

        x_grid, y1, _ = self._shoot_nonradial(omega2_found, ell, reconstruct=True)
        n_nodes = _count_nodes(y1)
        omega_found = np.sqrt(complex(omega2_found))
        mode_type = "f" if n_nodes == 0 else "p"

        return OscillationMode(
            ell=ell, m=m, n=n_nodes,
            omega=complex(omega_found),
            frequency=omega_found / (2.0 * np.pi),
            damping_time=None,
            xi_r=y1 / np.abs(y1).max(),
            radial_grid=x_grid * mod.R,
            mode_type=mode_type,
            solver_info={"omega2": omega2_found, "n_nodes_found": n_nodes},
        )

    def compute_nonradial_spectrum(
        self, ell: int, n_max: int, *, mode_family: str = "p"
    ) -> list[OscillationMode]:
        """Return modes n=0 … n_max for given ell (m=0 by convention)."""
        return [
            self.compute_nonradial_mode(ell, 0, n, mode_family=mode_family)
            for n in range(n_max + 1)
        ]


class NewtonianTidalOverlap:
    """
    Compute overlap integrals of oscillation eigenfunctions with the
    external tidal field.

    The dimensionless overlap integral Q_{n ell} controls the tidal
    excitability of each mode and appears in the dynamical tide response
    of the star.  For a mode with eigenfunction xi and tidal potential U_ell:

        Q_{n ell} = (1/MR^ell) * integral_0^R rho * xi_r * dU_ell/dr r^2 dr

    plus a surface term from the horizontal displacement.

    Parameters
    ----------
    stellar_model : NewtonianStellarModel
        Same background model passed to NewtonianModes.
    mode_calculator : NewtonianModes
        Instance used to obtain (or reuse) eigenfunctions.
    """

    def __init__(self, stellar_model: NewtonianStellarModel, mode_calculator: NewtonianModes):
        self.model = stellar_model
        self.modes = mode_calculator

    def compute_overlap(self, ell: int, n: int) -> TidalOverlapResult:
        """
        Compute the tidal overlap integral Q_{n ell} for the mode (ell, n).

        Retrieves or computes the eigenfunction for mode (ell, m=0, n), then
        evaluates the radial integral of the eigenfunction against the tidal
        potential U_ell ~ r^ell P_ell(cos theta).

        Parameters
        ----------
        ell : int >= 2
            Angular degree of the tidal harmonic.
        n : int >= 0
            Radial order of the mode.

        Returns
        -------
        TidalOverlapResult
            Contains Q_nl, kappa_nl, and the underlying OscillationMode.
        """
        raise NotImplementedError

    def compute_overlap_spectrum(
        self, ell: int, n_max: int
    ) -> list[TidalOverlapResult]:
        """Return overlap integrals for modes n=0 … n_max at given ell."""
        return [self.compute_overlap(ell, n) for n in range(n_max + 1)]

    def effective_tidal_coupling(self, ell: int, n_max: int) -> float:
        """
        Compute the total effective tidal coupling constant kappa_ell by
        summing contributions from modes n=0 … n_max.

        This is related to the static tidal Love number k_ell in the limit
        that all mode frequencies are large compared to the tidal frequency.

        Parameters
        ----------
        ell : int >= 2
        n_max : int
            Truncation order for the mode sum.

        Returns
        -------
        float
            Sum_{n=0}^{n_max} kappa_{n ell}.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Relativistic module
# ---------------------------------------------------------------------------

class RelativisticModes:
    """
    Compute oscillation modes of a neutron star in full general relativity.

    Fluid perturbations are described by the Lindblom-Detweiler (1983) or
    Detweiler-Lindblom (1985) equations.  Spacetime perturbations are
    included via the Regge-Wheeler (odd parity) and Zerilli (even parity)
    equations, enabling computation of the gravitational wave damping time
    and w-modes.

    Parameters
    ----------
    stellar_model : RelativisticStellarModel
        Solved TOV model providing r, rho, p, e, Gamma1, m, nu, lam.
    include_spacetime : bool
        If True, couple fluid and spacetime perturbations (full problem).
        If False, freeze metric perturbations (GR Cowling approximation).
    """

    def __init__(self, stellar_model: RelativisticStellarModel, *, include_spacetime: bool = True):
        self.model = stellar_model
        self.include_spacetime = include_spacetime
        self._axial_cache = {}
        self._trapped_cache = {}
        self._build_interpolators()

    def _build_interpolators(self) -> None:
        """
        Build cubic interpolators for the background profiles.  Models
        without an EOS (e.g. UniformDensityStellarModel) have no rho, cs2 or
        Gamma1; those interpolators are then None, which is fine for the
        axial modes but not for the fluid (radial) modes.
        """
        m = self.model
        kw = dict(kind="cubic", bounds_error=False, fill_value=0.0)
        def interpolator(values):
            return None if values is None else scipy.interpolate.interp1d(m.r, values, **kw)
        self._rho_i    = interpolator(m.rho)
        self._p_i      = interpolator(m.p)
        self._e_i      = interpolator(m.e)
        self._m_i      = interpolator(m.m)
        self._cs2_i    = interpolator(m.cs2)
        self._Gamma1_i = interpolator(m.Gamma1)
        self._nu_i     = interpolator(m.nu)
        self._lam_i    = interpolator(m.lam)
        # dν/dr from the TOV equation: (m + 4π r³ p) / (r² (1 − 2m/r))
        with np.errstate(divide="ignore", invalid="ignore"):
            dnu_dr = np.where(m.r > 0, (m.m + 4.0 * np.pi * m.r**3 * m.p)
                              / (m.r**2 * (1.0 - 2.0 * m.m / m.r)), 0.0)
        self._dnu_dr_i = interpolator(dnu_dr)

    # ------------------------------------------------------------------
    # Radial modes  (ell = 0)
    # ------------------------------------------------------------------

    def radial_mode_rhs(self, r: float, Z: np.ndarray, omega2: float) -> list:
        """
        RHS of the GR radial oscillation ODE 
        See https://journals.aps.org/prd/pdf/10.1103/PhysRevD.108.103035.

        The situation is morally speaking identical to the Newtonian case

        State vector Z = [y1, y2] where
            y1 = ...  (Displacement, xi)
            y2 = ...  (Lagrangian pressure perturbation, Delta p)

        Background quantities available via self._*_i interpolators:
            rho, p, e, m, cs2, Gamma1, nu, lam

        where  e^{2 lam} = 1/(1 - 2m/r),  e^{2 nu} = -g_tt.

        """
        # Get background quantities
        e = float(self._e_i(r))
        p   = float(self._p_i(r))
        G1  = float(self._Gamma1_i(r))
        m_r = float(self._m_i(r))
        nu = float(self._nu_i(r))
        lam = float(self._lam_i(r))
        cs2 = float(self._cs2_i(r))
        dnu_dr = float(self._dnu_dr_i(r))

        e2lam  = np.exp(2.0 * lam)          # 1 / (1 - 2m/r)
        em2nu  = np.exp(-2.0 * nu)          # 1 / (-g_tt)

        y1, y2 = Z[0], Z[1]

        # Eq. (1): dξ/dr = (dν/dr − 3/r) ξ − ΔP / (r Γ P)
        dy1 = (dnu_dr - 3.0 / r) * y1 - y2 / (r * G1 * p)

        # Eq. (2): d(ΔP)/dr
        #   = [e^{2λ}(ω² e^{-2ν} − 8π P) + (dν/dr)(4/r + dν/dr)] (e+P) r ξ
        #     − [dν/dr + 4π(e+P) r e^{2λ}] ΔP
        coeff_xi = (e2lam * (omega2 * em2nu - 8.0 * np.pi * p)
                    + dnu_dr * (4.0 / r + dnu_dr)) * (e + p) * r
        coeff_dp = dnu_dr + 4.0 * np.pi * (e + p) * r * e2lam
        dy2 = coeff_xi * y1 - coeff_dp * y2

        return [dy1, dy2]

    def _shoot_radial(self, omega2: float):
        """
        Integrate GR radial equations from center to surface for trial omega^2.

        Returns
        -------
        r_grid  : ndarray
        y1      : ndarray   first state variable on r_grid
        y2_surf : float     surface boundary value  (eigenvalue condition: must be 0)
        """
        m = self.model
        r0, r1 = m.r[0], m.r[-2]

        # TODO: set near-centre Taylor-expansion initial conditions
        ic = self._radial_central_ic(r0)

        sol = scipy.integrate.solve_ivp(
            lambda r, Z: self.radial_mode_rhs(r, Z, omega2),
            [r0, r1], ic,
            method="RK45", dense_output=False,
            rtol=1e-9, atol=1e-11,
            max_step=(r1 - r0) / 200,
        )
        return sol.t, sol.y[0], float(sol.y[1, -1])

    def _radial_central_ic(self, r0: float) -> list:
        """
        Near-centre Taylor initial conditions for the GR radial ODE.

        Near r = 0, ξ approaches a constant A (regular at the origin) while
        ΔP must cancel the 3/r singularity in Eq. (1).  Substituting
        ξ = A, dξ/dr = 0 into Eq. (1) and keeping the leading 1/r terms:

            0 = −(3/r) A − ΔP / (r Γ P)   ⟹   ΔP = −3 Γ P A

        Normalising with A = 1:
            y1(r₀) = 1
            y2(r₀) = −3 Γ₁(r₀) P(r₀)

        The eigenvalue condition at the surface is y2(R) = ΔP(R) = 0.
        """
        G1_c = float(self._Gamma1_i(r0))
        p_c  = float(self._p_i(r0))
        return [1.0, -3.0 * G1_c * p_c]

    def find_single_radial_mode(self, omega_guess: float) -> OscillationMode:
        """
        Find the GR radial mode nearest to omega_guess by root-finding on the
        surface boundary condition.

        Parameters
        ----------
        omega_guess : float
            Initial guess for the angular eigenfrequency (geometric units).

        Returns
        -------
        OscillationMode
            ell=0, m=0, damping_time=None; n inferred from node count of y1.
        """
        omega2_guess = omega_guess**2

        def bc(omega2):
            _, _, y2s = self._shoot_radial(omega2)
            return y2s

        lo, hi = 0.5 * omega2_guess, 1.5 * omega2_guess
        for _ in range(20):
            if bc(lo) * bc(hi) < 0:
                break
            lo *= 0.5
            hi *= 2.0
        else:
            raise RuntimeError(
                f"Could not bracket a root near omega_guess={omega_guess:.4g}."
            )

        omega2_found = scipy.optimize.brentq(bc, lo, hi, xtol=1e-16, rtol=1e-10)
        r_grid, y1, _ = self._shoot_radial(omega2_found)
        n_nodes = _count_nodes(y1)
        omega_found = np.sqrt(complex(omega2_found))

        return OscillationMode(
            ell=0, m=0, n=n_nodes,
            omega=complex(omega_found),
            frequency=omega_found / (2.0 * np.pi),
            damping_time=None,
            xi_r=y1 / np.abs(y1).max(),
            radial_grid=r_grid,
            mode_type="radial",
            solver_info={"omega2": omega2_found},
        )

    def compute_radial_mode(self, n: int, n_scan: int = 300) -> OscillationMode:
        """
        Compute the n-th GR radial oscillation mode (ell=0).

        Scans omega^2 from near zero to a generous upper bound, locates all
        sign changes in the surface boundary condition, then refines the n-th
        bracket with Brent's method.  Radial order is verified by counting
        nodes in the first state variable.

        Frequencies are real (radial modes do not emit gravitational waves).
        The stability criterion omega^2 > 0 gives the onset of radial
        instability at the mass maximum.

        Parameters
        ----------
        n : int >= 0
            Radial order.  n=0 is the fundamental radial mode.
        n_scan : int
            Number of trial frequencies used in the initial scan.

        Returns
        -------
        OscillationMode
            ell=0, m=0, damping_time=None.
        """
        m = self.model
        omega_char = float(np.sqrt(m.M / m.R**3))
        omega2_min = -omega_char**2
        omega2_max = ((n + 3) * np.pi * omega_char) ** 2 * 4.0

        omega2_scan = np.linspace(omega2_min, omega2_max, n_scan)

        def bc(omega2):
            _, _, y2s = self._shoot_radial(omega2)
            return y2s

        bc_vals = np.array([bc(o2) for o2 in omega2_scan])

        brackets = []
        for i in range(len(bc_vals) - 1):
            if np.isfinite(bc_vals[i]) and np.isfinite(bc_vals[i + 1]):
                if bc_vals[i] * bc_vals[i + 1] < 0.0:
                    brackets.append((omega2_scan[i], omega2_scan[i + 1]))

        if n >= len(brackets):
            raise ValueError(
                f"Mode n={n} not found: only {len(brackets)} mode(s) detected "
                f"in omega^2 range [{omega2_min:.3g}, {omega2_max:.3g}]. "
                "Increase n_scan or the scan range."
            )

        lo, hi = brackets[n]
        omega2_found = scipy.optimize.brentq(bc, lo, hi, xtol=1e-16, rtol=1e-10)
        r_grid, y1, _ = self._shoot_radial(omega2_found)
        n_nodes = _count_nodes(y1)
        omega_found = np.sqrt(complex(omega2_found))

        return OscillationMode(
            ell=0, m=0, n=n_nodes,
            omega=complex(omega_found),
            frequency=omega_found / (2.0 * np.pi),
            damping_time=None,
            xi_r=y1 / np.abs(y1).max(),
            radial_grid=r_grid,
            mode_type="radial",
            solver_info={"omega2": omega2_found, "n_nodes_found": n_nodes},
        )

    def compute_radial_spectrum(self, n_max: int) -> list[OscillationMode]:
        """Return the first n_max+1 radial modes (n=0 … n_max)."""
        return [self.compute_radial_mode(n) for n in range(n_max + 1)]

    # ------------------------------------------------------------------
    # Nonradial modes  (ell >= 2)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Axial (odd-parity) perturbations
    # ------------------------------------------------------------------
    #
    # Chandrasekhar & Ferrari 1991; Kokkotas & Schmidt 1999, Sec. 6:
    #
    #     d^2 Z / dr*^2 + [omega^2 - V_l(r)] Z = 0,   dr*/dr = e^{lam - nu},
    #     V_l = e^{2 nu} [ l(l+1)/r^2 + 4 pi (e - p) - 6 m / r^3 ],
    #
    # which is the Regge-Wheeler potential outside the star.  Axial
    # perturbations of a non-rotating perfect fluid do not couple to the
    # fluid, so this is a single ODE.  Time dependence e^{-i omega t}: damped
    # modes have Im(omega) < 0, with Q = Re(omega) / (-2 Im(omega)).
    #
    # Boundary conditions: Z ~ r^{l+1} at the centre and purely outgoing
    # Z ~ e^{+i omega r*} at infinity.  The outgoing solution starts at large r
    # from the asymptotic series
    #     Z = e^{i omega r*} sum_k a_k r^{-k},
    #     2 i omega (k+1) a_{k+1} = [k(k+1) - l(l+1)] a_k - 2M (k^2 - 4) a_{k-1},
    # is integrated inward to the surface and matched to the interior
    # solution through the Wronskian of (Z, dZ/dr*).
    #
    # LIMITATION: the exterior is integrated along the real r axis.  That is
    # accurate for weakly damped (trapped, Q >~ 3) modes but not for strongly
    # damped curvature w-modes (|Im omega| r_inf >> 1), which need a complex-r
    # rotation or the Benhar-Berti-Ferrari continued fraction.
    #
    # Moved here from TrappedNullGeos (trapgeo/axial.py).

    def axial_potential(self, r, ell: int):
        """V_l(r): the axial potential, interior + Regge-Wheeler exterior."""
        model, L = self.model, ell * (ell + 1)
        r = np.asarray(r, dtype=float)
        m, e, p = model.mass(r), model.energy_density(r), model.pressure(r)
        return np.exp(2.0 * model.nu_of_r(r)) * (L / r**2 + 4.0 * np.pi * (e - p) - 6.0 * m / r**3)

    def tortoise(self, r):
        """r*(r), with r*(0) = 0: interior quadrature, Schwarzschild outside."""
        model = self.model
        r = np.atleast_1d(np.asarray(r, dtype=float))
        r_int = np.linspace(0.0, model.R, 4000)
        rr = np.maximum(r_int, 1e-12)
        rs_int = scipy.integrate.cumulative_trapezoid(np.exp(model.lam_of_r(rr) - model.nu_of_r(rr)),
                                                      r_int, initial=0.0)
        M = model.M
        exterior = lambda x: x + 2.0 * M * np.log(x / (2.0 * M) - 1.0)
        return np.where(r <= model.R, np.interp(r, r_int, rs_int),
                        rs_int[-1] + exterior(np.maximum(r, model.R)) - exterior(model.R))

    def _axial_interior(self, ell):
        """Cached interpolators of V_l and dr*/dr inside the star."""
        if ell not in self._axial_cache:
            model = self.model
            r = np.asarray(model.r, dtype=float)
            r = r[r > 0]
            self._axial_cache[ell] = (
                scipy.interpolate.PchipInterpolator(r, self.axial_potential(r, ell)),
                scipy.interpolate.PchipInterpolator(r, np.exp(model.lam_of_r(r) - model.nu_of_r(r))),
            )
        return self._axial_cache[ell]

    def find_potential_well(self, ell: int, r_out_factor: float = 8.0, n: int = 6000) -> PotentialWell:
        """Locate an inner well of V_l bounded by an outer barrier maximum."""
        model = self.model
        r = np.linspace(model.R * 1e-3, r_out_factor * max(model.M, model.R / 3), n)
        r = np.union1d(r, [model.R * (1 - 1e-9), model.R * (1 + 1e-9)])
        V = self.axial_potential(r, ell)
        i_maxima = np.nonzero((V[1:-1] >= V[:-2]) & (V[1:-1] > V[2:]))[0] + 1
        i_minima = np.nonzero((V[1:-1] <= V[:-2]) & (V[1:-1] < V[2:]))[0] + 1
        if i_minima.size == 0 or i_maxima.size == 0:
            return PotentialWell(False)
        i_lo = i_minima[np.argmin(V[i_minima])]
        outer = i_maxima[i_maxima > i_lo]
        if outer.size == 0:
            return PotentialWell(False)
        i_hi = outer[np.argmax(V[outer])]
        return PotentialWell(True, r[i_lo], V[i_lo], r[i_hi], V[i_hi],
                             "interior" if r[i_hi] < model.R else "exterior")

    def _axial_interior_solution(self, omega, ell, r_eval=None, rtol=1e-10, atol=1e-14):
        model = self.model
        V_i, g_i = self._axial_interior(ell)
        r0 = model.R * 1e-4
        y0 = np.array([r0 ** (ell + 1), (ell + 1) * r0**ell * np.exp(float(model.nu_of_r(r0)))], dtype=complex)
        w2 = omega**2
        rhs = lambda r, y: [g_i(r) * y[1], g_i(r) * (V_i(r) - w2) * y[0]]
        return scipy.integrate.solve_ivp(rhs, (r0, model.R * (1 - 1e-12)), y0, rtol=rtol,
                                         atol=atol * r0 ** (ell + 1), method="DOP853", t_eval=r_eval)

    def _axial_series(self, omega, ell, r, n_series):
        M, L = self.model.M, ell * (ell + 1)
        a = [1.0 + 0j, -L / (2j * omega)]
        for k in range(1, n_series):
            a.append(((k * (k + 1) - L) * a[k] - 2 * M * (k * k - 4) * a[k - 1]) / (2j * omega * (k + 1)))
        a = np.array(a)
        k = np.arange(len(a))
        u = np.sum(a * r ** (-k))
        du = np.sum(-k * a * r ** (-k - 1.0))
        f = 1.0 - 2.0 * M / r
        return u, 1j * omega * u + f * du   # (Z, dZ/dr*) up to e^{i omega r*}

    def _axial_exterior_solution(self, omega, ell, r_eval=None, rtol=1e-10,
                                 r_inf_factor=200.0, n_series=12):
        M, L = self.model.M, ell * (ell + 1)
        r_inf = max(r_inf_factor * M, 40.0 * ell / max(abs(omega.real), 1e-3))
        Z, Y = self._axial_series(omega, ell, r_inf, n_series)
        w2 = omega**2

        def rhs(r, y):
            f = 1.0 - 2.0 * M / r
            V = f * (L / r**2 - 6.0 * M / r**3)
            return [y[1] / f, (V - w2) * y[0] / f]

        return scipy.integrate.solve_ivp(rhs, (r_inf, self.model.R), np.array([Z, Y]), rtol=rtol,
                                         atol=1e-30, method="DOP853", t_eval=r_eval)

    def axial_wronskian(self, omega, ell: int):
        """
        Normalised Wronskian of the regular interior solution and the purely
        outgoing exterior solution at r = R; it vanishes at an axial mode.
        """
        omega = complex(omega)
        interior = self._axial_interior_solution(omega, ell)
        exterior = self._axial_exterior_solution(omega, ell)
        Zi, Yi = interior.y[0, -1], interior.y[1, -1]
        Ze, Ye = exterior.y[0, -1], exterior.y[1, -1]
        norm = (np.sqrt(abs(Zi) ** 2 + abs(Yi / omega) ** 2)
                * np.sqrt(abs(Ze) ** 2 + abs(Ye / omega) ** 2))
        return (Zi * Ye - Ze * Yi) / (norm * abs(omega))

    def wkb(self, omega_r: float, ell: int, well: PotentialWell | None = None):
        """
        Bohr-Sommerfeld phase (integral of k dr* across the allowed region of
        the well) and the Gamow estimate of Q for a real frequency omega_r.
        Returns (phase, Q); (nan, nan) if there is no well or no allowed region.
        """
        well = well or self.find_potential_well(ell)
        if not well.exists:
            return np.nan, np.nan
        r = np.linspace(self.model.R * 1e-4, well.r_max * 6, 20000)
        rs = self.tortoise(r)
        V = self.axial_potential(r, ell)
        E = omega_r**2
        allowed = (V < E) & (r < well.r_max)
        if not allowed.any():
            return np.nan, np.nan
        k = np.sqrt(np.clip(E - V, 0, None))
        phase = trapezoid(np.where(allowed, k, 0.0), rs)
        crossing_time = trapezoid(np.where(allowed & (k > 0), 1.0 / np.maximum(k, 1e-30), 0.0), rs)
        forbidden = (V > E) & (r > well.r_min)
        theta = trapezoid(np.where(forbidden, np.sqrt(np.clip(V - E, 0, None)), 0.0), rs)
        return phase, 2.0 * E * crossing_time * np.exp(2.0 * theta)

    def well_phase(self, ell: int, well: PotentialWell | None = None) -> float:
        """Bohr-Sommerfeld phase of the whole well (at the barrier top), in units of pi."""
        well = well or self.find_potential_well(ell)
        if not well.exists:
            return 0.0
        phase, _ = self.wkb(np.sqrt(well.V_max) * (1 - 1e-6), ell, well)
        return phase / np.pi

    def find_axial_mode(self, omega_guess: complex, ell: int, *, well: PotentialWell | None = None,
                        tol: float = 1e-12, residual_tol: float = 1e-7) -> OscillationMode:
        """
        Refine an axial quasinormal mode from a complex frequency guess (a
        root of axial_wronskian) and characterise it: quality factor,
        whether it is trapped in the potential well, node count of the
        interior eigenfunction and the WKB estimate of Q.

        The root counts as converged if the root finder says so or the
        normalised Wronskian there is below ``residual_tol`` (it is ~1 away
        from modes).  For very high-Q modes the finder can stall at the
        Wronskian's numerical noise floor (~1e-10) and report failure even
        though it has found the root.
        """
        def residual(x):
            W = self.axial_wronskian(x[0] + 1j * x[1], ell)
            return [W.real, W.imag]
        omega_guess = complex(omega_guess)
        solution = scipy.optimize.root(residual, [omega_guess.real, omega_guess.imag], method="hybr",
                                       options={"xtol": tol})
        omega = complex(solution.x[0], solution.x[1])
        residual_value = float(abs(self.axial_wronskian(omega, ell)))
        well = well or self.find_potential_well(ell)
        return self._characterise_axial_mode(omega, ell, well,
                                             bool(solution.success) or residual_value < residual_tol,
                                             residual_value)

    def _characterise_axial_mode(self, omega, ell, well, converged, residual_value=np.nan):
        Q = omega.real / (-2.0 * omega.imag) if omega.imag < 0 else np.inf
        trapped = bool(well.exists and well.V_min < omega.real**2 < well.V_max)
        r_in = np.linspace(self.model.R * 1e-4, self.model.R * (1 - 1e-12), 800)
        Z_in = self._axial_interior_solution(omega, ell, r_eval=r_in).y[0]
        # interior eigenfunction rotated to be mostly real, for counting nodes
        phase = np.angle(Z_in[np.argmax(abs(Z_in))])
        nodes = _count_nodes((Z_in * np.exp(-1j * phase)).real)
        ratio = np.nan
        if well.exists:
            # amplitude in the well vs just beyond the barrier, outer solution scaled to match at R
            r_out = np.linspace(self.model.R, max(well.r_max * 1.5, self.model.R * 1.01), 200)[::-1]
            Z_out = self._axial_exterior_solution(omega, ell, r_eval=r_out).y[0]
            scale = Z_in[-1] / Z_out[-1] if Z_out[-1] != 0 else 1.0
            ratio = float(np.max(abs(Z_in)) / max(abs(Z_out[0] * scale), 1e-300))
        _, wkb_Q = self.wkb(omega.real, ell, well) if well.exists else (np.nan, np.nan)
        return OscillationMode(
            ell=ell, m=0, n=nodes,
            omega=omega,
            frequency=omega.real / (2.0 * np.pi),
            damping_time=1.0 / abs(omega.imag) if omega.imag != 0 else None,
            radial_grid=r_in,
            mode_type="w",
            quality_factor=float(Q),
            trapped=trapped,
            solver_info={"parity": "odd", "converged": converged, "wronskian": residual_value,
                         "n_nodes": nodes,
                         "well_amplitude_ratio": ratio, "wkb_Q": float(wkb_Q)},
        )

    def trapped_axial_modes(self, ell: int, *, n_scan: int = 150, max_modes: int = 20,
                            wkb_prefilter: float | None = 0.25):
        """
        Weakly damped axial modes with Re(omega)^2 inside the potential well
        (between V_min and the barrier top), sorted by Re(omega).

        The well band is scanned along the real axis for minima of
        |Wronskian|, each refined with find_axial_mode.  ``wkb_prefilter``
        skips the (expensive) search when the Bohr-Sommerfeld phase of the
        whole well is below that many pi; the n = 0 quasi-bound state needs
        about 0.5 pi, the margin guards against WKB error at low ell.

        The result is cached per (ell, settings); the modes returned are
        copies, so changing them does not affect later calls.

        Returns (well, modes).
        """
        key = (ell, n_scan, max_modes, wkb_prefilter)
        if key not in self._trapped_cache:
            self._trapped_cache[key] = self._scan_trapped_axial_modes(ell, n_scan, max_modes, wkb_prefilter)
        well, modes = self._trapped_cache[key]
        return well, [replace(mode, solver_info=dict(mode.solver_info)) for mode in modes]

    def _scan_trapped_axial_modes(self, ell, n_scan, max_modes, wkb_prefilter):
        well = self.find_potential_well(ell)
        if not well.exists:
            return well, []
        if wkb_prefilter is not None and self.well_phase(ell, well) < wkb_prefilter:
            return well, []
        omega_lo = np.sqrt(max(well.V_min, 0.0)) * 1.0001
        omega_hi = np.sqrt(well.V_max) * 0.9999
        grid = np.linspace(omega_lo, omega_hi, n_scan)
        absW = np.array([abs(self.axial_wronskian(w - 1e-8j, ell)) for w in grid])
        candidates = np.nonzero((absW[1:-1] < absW[:-2]) & (absW[1:-1] < absW[2:]))[0] + 1
        modes = []
        for i in candidates[:max_modes]:
            mode = self.find_axial_mode(grid[i] - 1j * 1e-4 * grid[i], ell, well=well)
            omega = mode.omega
            if (not mode.solver_info["converged"] or omega.imag > 0
                    or any(abs(omega - other.omega) < 1e-7 * abs(omega) for other in modes)):
                continue
            modes.append(mode)
        modes.sort(key=lambda mode: mode.omega.real)
        return well, modes

    def minimal_trapping_ell(self, threshold: float = 0.5, ell_max: int = 256):
        """
        Smallest ell whose axial well supports an n = 0 quasi-bound state in
        the Bohr-Sommerfeld sense (well_phase >= threshold * pi), by
        bisection in ell (the phase grows roughly linearly with ell at large
        ell; assumed monotonic).  np.inf if none up to ell_max.
        """
        ok = lambda ell: self.well_phase(int(ell)) >= threshold
        if ok(2):
            return 2
        if not ok(ell_max):
            return np.inf
        lo, hi = 2, ell_max
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        return hi

    # ------------------------------------------------------------------
    # Nonradial modes  (ell >= 2)
    # ------------------------------------------------------------------

    def compute_nonradial_mode(
        self,
        ell: int,
        m: int,
        n: int,
        *,
        mode_family: str = "p",
        parity: str = "even",
    ) -> OscillationMode:
        """
        Compute a nonradial quasi-normal mode with given (ell, m, n) in GR.

        Implemented: axial (parity="odd") w-modes trapped in the potential
        well of an ultracompact star, via trapped_axial_modes; n indexes them
        in order of increasing Re(omega).  Not implemented yet: polar
        (even-parity) modes, i.e. the fluid f, p, g modes (Lindblom-Detweiler
        equations) and polar w-modes, and strongly damped curvature w-modes.

        Parameters
        ----------
        ell : int >= 2
            Angular degree.
        m : int
            Azimuthal order (|m| <= ell).
        n : int >= 0
            Radial order.  For w-modes n labels the w_I overtone sequence.
        mode_family : {"p", "g", "f", "w"}
            Which branch of the spectrum to target.
        parity : {"even", "odd"}
            Perturbation parity.  Most fluid modes are even (polar); odd
            (axial) fluid modes are trivial for a non-rotating star but
            axial w-modes are physical.

        Returns
        -------
        OscillationMode
            omega is complex; damping_time = 1 / |Im(omega)|.
        """
        if ell < 2:
            raise ValueError(
                f"Nonradial modes require ell >= 2; got ell={ell}. "
                "Use compute_radial_mode for ell=0."
            )
        if abs(m) > ell:
            raise ValueError(f"|m|={abs(m)} > ell={ell}")
        if parity not in ("even", "odd"):
            raise ValueError(f"parity must be 'even' or 'odd', got '{parity}'")
        if parity == "odd":
            if mode_family != "w":
                raise ValueError("odd-parity (axial) fluid modes are trivial for a non-rotating star; "
                                 "use mode_family='w'")
            _, modes = self.trapped_axial_modes(ell)
            if n >= len(modes):
                raise ValueError(
                    f"axial w-mode n={n} not found: this star has {len(modes)} trapped axial "
                    f"mode(s) at ell={ell}. Strongly damped curvature w-modes are not implemented.")
            return replace(modes[n], m=m)
        raise NotImplementedError("polar (even-parity) nonradial modes are not implemented yet")

    def compute_nonradial_spectrum(
        self,
        ell: int,
        n_max: int,
        *,
        mode_family: str = "p",
        parity: str = "even",
    ) -> list[OscillationMode]:
        """Return modes n=0 … n_max for given ell (m=0 by convention)."""
        return [
            self.compute_nonradial_mode(
                ell, 0, n, mode_family=mode_family, parity=parity
            )
            for n in range(n_max + 1)
        ]

    def compute_w_modes(self, ell: int, n_max: int) -> list[OscillationMode]:
        """
        The trapped axial (odd-parity) w-modes for given ell, at most
        n_max + 1 of them in order of increasing Re(omega); fewer if the star
        has fewer (none unless it has a potential well).
        """
        _, modes = self.trapped_axial_modes(ell)
        return modes[: n_max + 1]
