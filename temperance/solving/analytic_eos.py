"""
Analytic and tabulated equations of state for neutron star structure calculations.

All EOS classes expose a common log-enthalpy interface used by the TOV and
Newtonian structure solvers.  The log-enthalpy is defined as

    lnh = ln(h)  where  h = exp( integral dp / (e + p) )

so that lnh = 0 at zero pressure and lnh = lnh_c > 0 at the centre.

Common interface
----------------
Every EOS class provides the following callables (all accept array arguments):

    rho_of_logenthalpy(lnh)   -> baryon density
    p_of_logenthalpy(lnh)     -> pressure
    e_of_logenthalpy(lnh)     -> total energy density
    cs2_of_logenthalpy(lnh)   -> adiabatic sound speed squared  dp/de|_s
    logenthalpy_of_rho(rho)   -> inverse: log-enthalpy from baryon density
    p_of_rho(rho)             -> pressure from baryon density
    e_of_rho(rho)             -> energy density from baryon density

Units
-----
All quantities are in geometric units (G = c = 1) unless noted otherwise.
The conversion from CGS used by interpolated_eos is

    rho [geom] = rho [g/cm^3] * (0.00045 / 2.8e14)

so that nuclear saturation density (~2.8e14 g/cm^3) maps to ~0.00045 in
geometric units.
"""

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate

# ---------------------------------------------------------------------------
# JAX / NumPy backend selection
# The try block attempts to import JAX for JIT-compiled ODE integration.
# If JAX is unavailable, the code falls back to numpy + scipy transparently.
# ---------------------------------------------------------------------------

try:
    raise ImportError  # remove this line to enable JAX
    import jax
    from jax import config
    from jax.scipy.integrate import trapezoid

    config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    xp = jax.numpy
    xp.trapz = trapezoid
    from jax.experimental import ode
    from jax.scipy.interpolate import RegularGridInterpolator as interp1d
    ode_solver = ode.odeint

    def print_val(statement, val):
        jax.debug.print(statement, val=val)

except ImportError:
    xp = np
    from scipy.interpolate import interp1d as spinterp1d
    interp1d = lambda x, y, method, **kwargs: spinterp1d(x[0], y, **kwargs)
    ode_solver = scipy.integrate.odeint

    def print_val(statement, **kwargs):
        val = list(kwargs.items())[0]
        print(f"{statement} = {val}")

try:
    import jax.scipy as jsp
except ImportError:
    import scipy as jsp


# Heavy / optional dependencies — imported lazily at point of use so that
# lightweight classes (polytropic_eos, interpolated_eos, …) remain importable
# without requiring 'universality', 'jax', or a fully-installed package tree.
# ---------------------------------------------------------------------------
# EOS classes
# ---------------------------------------------------------------------------

class polytropic_eos:
    """
    Simple polytropic equation of state:  p = K * rho^Gamma.

    Parameters
    ----------
    K : float
        Polytropic constant (geometric units).
    Gamma : float
        Adiabatic index used for the pressure-density relation.
    Gamma1 : float, optional
        Out-of-equilibrium (dynamical) adiabatic index used for the sound
        speed.  Defaults to Gamma when omitted (isentropic case).
    """

    def __init__(self, K, Gamma, Gamma1=None):
        self.K = K
        self.Gamma = Gamma
        self.Gamma1 = Gamma if Gamma1 is None else Gamma1

        self.rho_of_logenthalpy = lambda lnh: (
            (xp.exp(lnh) - 1.0) * (self.Gamma - 1.0) / (self.K * self.Gamma)
        ) ** (1.0 / (Gamma - 1.0))

        self.p_of_logenthalpy = lambda lnh: (
            self.K * self.rho_of_logenthalpy(lnh) ** self.Gamma
        )
        self.e_of_logenthalpy = lambda lnh: (
            self.rho_of_logenthalpy(lnh)
            + self.p_of_logenthalpy(lnh) / (self.Gamma - 1.0)
        )
        # cs2 = (dp/drho) / (de/drho) = Gamma1 * p / (rho * h)
        self.cs2_of_logenthalpy = lambda lnh: (
            self.Gamma1
            * self.K
            * self.rho_of_logenthalpy(lnh) ** (self.Gamma - 1.0)
            / xp.exp(lnh)
        )
        self.logenthalpy_of_rho = lambda rho: xp.log(
            1.0
            + self.K * self.Gamma * rho ** (self.Gamma - 1.0) / (self.Gamma - 1.0)
        )
        self.p_of_rho = lambda rho: self.K * rho ** self.Gamma
        self.e_of_rho = lambda rho: rho + self.p_of_rho(rho) / (self.Gamma - 1.0)

class energy_density_polytrope:
    """
    Polytropic EOS expressed in terms of energy density: p = K * e^Gamma.

    Unlike polytropic_eos (which uses baryon density rho), the independent
    thermodynamic variable here is the total energy density e.  This form is
    convenient for piecewise constructions because pressure continuity at
    segment boundaries directly constrains K without reference to baryon density.

    The log-enthalpy integral dp/(e+p) has a closed form:

        lnh(e) = Gamma/(Gamma-1) * ln(1 + K * e^(Gamma-1))

    with inverse:

        e(lnh) = ((exp((Gamma-1)/Gamma * lnh) - 1) / K)^(1/(Gamma-1))

    The adiabatic (GR) sound speed is cs2 = Gamma1 * p / (e+p).

    Parameters
    ----------
    K : float
        Polytropic constant (geometric units).
    Gamma : float
        Adiabatic index for the equilibrium relation p = K * e^Gamma.
        Must satisfy Gamma > 1.
    Gamma1 : float, optional
        Dynamical adiabatic index controlling the sound speed.
        Defaults to Gamma (isentropic).
    e_max : float
        Upper end of the precomputed (rho -> lnh) lookup table.
        Should exceed the maximum energy density expected in the star.
    n_table : int
        Number of points in the lookup table.
    """

    def __init__(self, K, Gamma, Gamma1=None, e_max=1.0, n_table=2000):
        self.K = K
        self.Gamma = Gamma
        self.Gamma1 = Gamma if Gamma1 is None else Gamma1

        # ------------------------------------------------------------------
        # Closed-form relations in terms of e
        # ------------------------------------------------------------------
        def _e_of_lnh(lnh):
            # Invert lnh = Gamma/(Gamma-1) * ln(1 + K*e^(Gamma-1))
            return ((xp.exp(lnh * (Gamma - 1.0) / Gamma) - 1.0) / K) ** (1.0 / (Gamma - 1.0))

        self.e_of_logenthalpy = _e_of_lnh

        self.p_of_logenthalpy = lambda lnh: K * _e_of_lnh(lnh) ** Gamma

        self.rho_of_logenthalpy = lambda lnh: (
            (_e_of_lnh(lnh) + K * _e_of_lnh(lnh) ** Gamma) / xp.exp(lnh)
        )

        # cs2 = Gamma1 * p / (e + p)  (GR adiabatic sound speed)
        self.cs2_of_logenthalpy = lambda lnh: (
            self.Gamma1 * K * _e_of_lnh(lnh) ** Gamma
            / (_e_of_lnh(lnh) + K * _e_of_lnh(lnh) ** Gamma)
        )

        # ------------------------------------------------------------------
        # Precompute (rho -> lnh) lookup table for the inverse direction
        # ------------------------------------------------------------------
        e_table   = np.linspace(e_max * 1e-8, e_max, n_table)
        lnh_table = Gamma / (Gamma - 1.0) * np.log(1.0 + K * e_table ** (Gamma - 1.0))
        p_table   = K * e_table ** Gamma
        rho_table = (e_table + p_table) / np.exp(lnh_table)

        kw = dict(kind="cubic", bounds_error=False, fill_value="extrapolate")
        self.logenthalpy_of_rho = scipy.interpolate.interp1d(rho_table, lnh_table, **kw)
        self.p_of_rho           = scipy.interpolate.interp1d(rho_table, p_table,   **kw)
        self.e_of_rho           = scipy.interpolate.interp1d(rho_table, e_table,   **kw)

        # Direct e-based accessors (used internally by piecewise_eos)
        self.p_of_e   = lambda e: K * e ** Gamma
        self.e_of_p   = lambda p: (p / K) ** (1.0 / Gamma)
        self.lnh_of_e = lambda e: Gamma / (Gamma - 1.0) * xp.log(1.0 + K * e ** (Gamma - 1.0))
class css_eos:
    """
    Constant-speed-of-sound EOS above a transition energy density e0.

    Below e0 the matter is pressure-free (p = 0, modelling a low-density
    crust); above e0 the pressure is linear in energy density with slope cs2.

    Parameters
    ----------
    cs2 : float
        Constant adiabatic sound speed squared (in units of c^2).
    e0 : float
        Transition energy density (geometric units) where the CSS phase begins.
    """

    def __init__(self, cs2, e0):
        self.cs2 = cs2
        self.e0 = e0
        prefactor = e0 * (e0 / (1 + cs2)) ** (-1.0 / (1 + cs2))

        self.p_of_e = lambda e: xp.where(e < self.e0, 0, self.cs2 * (e - self.e0))
        self.rho_of_e = lambda e: xp.where(
            e < self.e0,
            e,
            (e - self.e0 * self.cs2) ** (1.0 / (1 + self.cs2)) * prefactor,
        )
        self.enthalpy_of_e = lambda e: (e + self.p_of_e(e)) / self.rho_of_e(e)
        self.e_of_rho = lambda rho: xp.where(
            rho < self.e0,
            rho,
            (rho / prefactor) ** (1 + self.cs2) / (1 + self.cs2) + self.e0 * self.cs2,
        )
        self.e_of_logenthalpy = lambda lnh: (
            e0 * (1 + 2 * self.cs2) * xp.exp((1 + self.cs2 / self.cs2) * lnh) / (1 + self.cs2)
        )
        self.p_of_logenthalpy = lambda lnh: self.p_of_e(self.e_of_logenthalpy(lnh))
        self.rho_of_logenthalpy = lambda lnh: self.rho_of_e(self.e_of_logenthalpy(lnh))
        self.cs2_of_logenthalpy = lambda lnh: xp.full_like(lnh, self.cs2)
        self.logenthalpy_of_rho = lambda rho: xp.log(
            self.enthalpy_of_e(self.e_of_rho(rho))
        )
        self.p_of_rho = lambda rho: self.p_of_e(self.e_of_rho(rho))
        self.e_of_rho = self.e_of_rho


class interpolated_eos:
    """
    Tabulated EOS read from a DataFrame and interpolated with splines.

    The log-enthalpy grid is constructed by numerically integrating
    dp / (e + p) over the tabulated pressure column.  The sound speed is
    derived as d(lnh) / d(ln rho) via numerical differentiation.

    Parameters
    ----------
    eos : pandas.DataFrame
        Table with columns:
          - ``baryon_density``    baryon number density (g/cm^3 or cgs)
          - ``pressurec2``        pressure / c^2            (same units)
          - ``energy_densityc2``  energy density / c^2      (same units)
    conversion_factor : float
        Multiplicative factor converting cgs columns to geometric units.
        Default converts g/cm^3 at nuclear saturation (~2.8e14) to ~0.00045.
    method : str
        Interpolation scheme passed to the backend (``"cubic"`` by default).
    """

    def __init__(self, eos, conversion_factor=1.0 / 2.8e14 * 0.00045, method="cubic"):
        self.eos = eos
        logenthalpy = xp.array(
            scipy.integrate.cumulative_trapezoid(
                1.0 / (eos["pressurec2"] + eos["energy_densityc2"]),
                eos["pressurec2"],
                initial=0,
            )
        )
        baryon_density = xp.array(eos["baryon_density"]) * conversion_factor
        pressurec2 = xp.array(eos["pressurec2"]) * conversion_factor
        energy_densityc2 = xp.array(eos["energy_densityc2"]) * conversion_factor
        cs2 = xp.array(np.gradient(logenthalpy, np.log(baryon_density)))

        self.cs2_of_logenthalpy = interp1d((logenthalpy,), cs2, method=method)
        self.rho_of_logenthalpy = interp1d((logenthalpy,), baryon_density, method=method)
        self.p_of_rho = interp1d((baryon_density,), pressurec2, method=method)
        self.e_of_rho = interp1d((baryon_density,), energy_densityc2, method=method)
        self.p_of_logenthalpy = interp1d((logenthalpy,), pressurec2, method=method)
        self.e_of_logenthalpy = interp1d((logenthalpy,), energy_densityc2, method=method)
        self.logenthalpy_of_rho = interp1d((baryon_density,), logenthalpy, method=method)


class piecewise_eos:
    """
    Piecewise EoS assembled from energy_density_polytrope and CSS segments.

    Segments are joined with pressure continuity enforced at each boundary.
    The cumulative log-enthalpy is tracked analytically across segments.
    A precomputed table spanning all segments backs the public interface.

    Parameters
    ----------
    segments : list of dict
        Each dict describes one segment.  Two types are supported:

        Polytropic segment::

            {"type": "polytrope", "Gamma": float, "e_max": float}
            {"type": "polytrope", "Gamma": float, "K": float, "e_max": float}

        K is inferred from pressure continuity at the lower boundary for all
        segments after the first; it must be given explicitly for the first.
        An optional ``"Gamma1"`` key overrides the dynamical adiabatic index.

        CSS segment::

            {"type": "css", "cs2": float, "e_max": float}

        p = p_lo + cs2 * (e - e_lo) where (e_lo, p_lo) are set automatically
        from the previous segment's upper boundary.  CSS segments must not
        be the first segment (p=0 at e=0 would produce a singular lnh).

    n_table : int
        Number of grid points per segment used when building the table.

    Examples
    --------
    >>> eos = piecewise_eos([
    ...     {"type": "polytrope", "Gamma": 4/3, "K": 0.01,  "e_max": 1e-4},
    ...     {"type": "polytrope", "Gamma": 2.5,              "e_max": 5e-4},
    ...     {"type": "css",       "cs2":  0.6,               "e_max": 2e-3},
    ... ])
    """

    def __init__(self, segments, n_table=500):
        if not segments:
            raise ValueError("segments must be non-empty.")
        first = segments[0]
        if first["type"] == "polytrope" and "K" not in first:
            raise ValueError("The first polytropic segment must specify K.")
        if first["type"] == "css":
            raise ValueError("The first segment cannot be CSS (lnh diverges at e=0 with p=0).")

        e_pieces   = []
        p_pieces   = []
        lnh_pieces = []
        cs2_pieces = []

        e_lo   = 0.0
        p_lo   = 0.0
        lnh_lo = 0.0

        for i, seg in enumerate(segments):
            e_hi    = seg["e_max"]
            # First segment avoids e=0; subsequent segments skip the shared
            # lower boundary (already the last point of the previous segment).
            e_start = e_hi * 1e-8 if e_lo == 0.0 else e_lo
            e_grid  = np.linspace(e_start, e_hi, n_table + 1)[1:] if i > 0 else np.linspace(e_start, e_hi, n_table)

            if seg["type"] == "polytrope":
                Gamma  = seg["Gamma"]
                Gamma1 = seg.get("Gamma1", Gamma)
                K      = seg["K"] if i == 0 else p_lo / e_lo ** Gamma

                p_grid = K * e_grid ** Gamma

                # lnh = lnh_lo + Gamma/(Gamma-1) * ln((1 + K*e^(Gamma-1)) / ref)
                # ref = 1 + K*e_lo^(Gamma-1);  at e_lo=0 this is 1.
                ref      = 1.0 + K * e_lo ** (Gamma - 1.0) if e_lo > 0.0 else 1.0
                lnh_grid = lnh_lo + Gamma / (Gamma - 1.0) * np.log(
                    (1.0 + K * e_grid ** (Gamma - 1.0)) / ref
                )
                cs2_grid = Gamma1 * p_grid / (e_grid + p_grid)

            elif seg["type"] == "css":
                cs2_val  = seg["cs2"]
                p_grid   = p_lo + cs2_val * (e_grid - e_lo)

                # integral_{e_lo}^{e} dp/(e'+p') = cs2/(1+cs2) * ln((e+p(e)) / (e_lo+p_lo))
                lnh_grid = lnh_lo + cs2_val / (1.0 + cs2_val) * np.log(
                    (e_grid + p_grid) / (e_lo + p_lo)
                )
                cs2_grid = np.full(n_table, cs2_val)

            else:
                raise ValueError(f"Unknown segment type '{seg['type']}'.")

            e_pieces.append(e_grid)
            p_pieces.append(p_grid)
            lnh_pieces.append(lnh_grid)
            cs2_pieces.append(cs2_grid)

            e_lo   = e_hi
            p_lo   = float(p_grid[-1])
            lnh_lo = float(lnh_grid[-1])

        e_arr   = np.concatenate(e_pieces)
        p_arr   = np.concatenate(p_pieces)
        lnh_arr = np.concatenate(lnh_pieces)
        cs2_arr = np.concatenate(cs2_pieces)
        rho_arr = (e_arr + p_arr) / np.exp(lnh_arr)

        kw = dict(kind="cubic", bounds_error=False, fill_value="extrapolate")
        self.e_of_logenthalpy   = scipy.interpolate.interp1d(lnh_arr, e_arr,   **kw)
        self.p_of_logenthalpy   = scipy.interpolate.interp1d(lnh_arr, p_arr,   **kw)
        self.rho_of_logenthalpy = scipy.interpolate.interp1d(lnh_arr, rho_arr, **kw)
        self.cs2_of_logenthalpy = scipy.interpolate.interp1d(lnh_arr, cs2_arr, **kw)
        self.logenthalpy_of_rho = scipy.interpolate.interp1d(rho_arr, lnh_arr, **kw)
        self.p_of_rho           = scipy.interpolate.interp1d(rho_arr, p_arr,   **kw)
        self.e_of_rho           = scipy.interpolate.interp1d(rho_arr, e_arr,   **kw)


class analytic_composition_dependent_eos:
    """
    Take a 1-D EoS and add leading-order composition dependent corrections
    around beta-equilibrium

    Parameters
    ----------
    eos_1d : object
        A 1D EOS object.
    saturation_params : dict
        Dictionary containing the saturation parameters:
        - ``"nsat"``: baryon density at saturation (in geometric units).
        - ``"EB"``: binding energy per baryon at saturation (in geometric units).
    symmetry_params : dict, optional
        Dictionary containing the symmetry energy parameter:
        - ``"S"``: symmetry energy at saturation (in fm^-3).
        - ``"L"``: slope of the symmetry energy at saturation (in MeV).
   
    nuclear_masses : dict, optional
        Dictionary containing the nuclear masses.
    density_range : tuple, optional
        The range of baryon densities to consider (in geometric units).
    """

    def __init__(self, eos_1d, saturation_params={"EB": -16, "nsat": 0.16}, symmetry_params=None, nuclear_masses=None, density_range=None):
        from temperance.solving import nuclear_properties  # jax dependency — lazy
        self.nsat = saturation_params["nsat"]
        self.E_bind = saturation_params["EB"]
        self.K0 = saturation_params.get("K0", 240.0)
        if symmetry_params is not None:
            self.S = symmetry_params["S"]
            self.L = symmetry_params["L"]
            self.Ksym = symmetry_params["Ksym"]
            self.Ybeta = symmetry_params.get("Ybeta", 0.04)
        else:
            S0, L0, Ksym ,Ybeta  = nuclear_properties.get_symmetry_parameters(density_range=density_range, **nuclear_masses, E0=self.E_bind, K0=self.K0, n0=self.nsat, beta_equilibrated_energy_density=eos_1d.e_of_rho,)
            self.S = S0
            self.L = L0
            self.Ksym = Ksym
            self.Ybeta = Ybeta
        self.eos_1d = eos_1d

        # Nuclear masses in MeV; use provided values or standard defaults.
        _DEFAULT_MASSES = {"mp": 938.272, "mn": 939.565, "mN": 931.494, "me": 0.511}
        _nm = nuclear_masses if nuclear_masses is not None else {}
        self._mp = _nm.get("mp", _DEFAULT_MASSES["mp"])
        self._mn = _nm.get("mn", _DEFAULT_MASSES["mn"])
        self._mN = _nm.get("mN", _DEFAULT_MASSES["mN"])
        self._me = _nm.get("me", _DEFAULT_MASSES["me"])

        # Precompute Ybeta(n) by solving beta equilibrium on a density grid so
        # the specific internal energy can be evaluated at arbitrary composition.
        if density_range is not None:
            _n_lo = float(np.min(density_range))
            _n_hi = float(np.max(density_range))
        else:
            _n_lo, _n_hi = 0.3 * self.nsat, 6.0 * self.nsat
        _n_grid = np.linspace(_n_lo, _n_hi, 40)
        _Ybeta_arr = np.empty(len(_n_grid))
        for _i, _n_i in enumerate(_n_grid):
            try:
                _e_i = float(np.atleast_1d(eos_1d.e_of_rho(np.array([_n_i])))[0])
                _sol = nuclear_properties.get_beta_eq_charge_fraction(
                    _n_i, self._mp, self._mn, self._mN, self._me,
                    self.E_bind, self.K0, self.nsat, _e_i,
                )
                _Ybeta_arr[_i] = _sol.root
            except Exception:
                _Ybeta_arr[_i] = float(self.Ybeta)
        self._Ybeta_of_n = scipy.interpolate.interp1d(
            _n_grid, _Ybeta_arr, kind="linear", fill_value="extrapolate"
        )

        # specific_internal_energy_of_nB_and_yq(nB, Yq) -> e / (mN * nB) - 1
        #
        # Total energy density = beta-eq energy (from 1D EoS)
        #   + nuclear out-of-equilibrium correction: nB * S(nB) * [(1-2Yq)^2 - (1-2Ybeta)^2]
        #   + change in lepton energy: e_lep(nB, Yq) - e_lep(nB, Ybeta)
        def _sie(nB, Yq):
            nB = np.atleast_1d(np.asarray(nB, dtype=float))
            Yq = np.atleast_1d(np.asarray(Yq, dtype=float))
            e_beta = np.array([
                float(np.atleast_1d(self.eos_1d.e_of_rho(np.array([_n])))[0])
                for _n in nB
            ])
            S_n = np.asarray(nuclear_properties.compute_symmetry_energy_per_nucleon(
                nB, self.nsat, self.S, self.L, self.Ksym
            ), dtype=float)
            Ybeta_n = np.clip(self._Ybeta_of_n(nB), 1e-6, 0.5)
            delta_e_nuc = nB * S_n * ((1.0 - 2.0 * Yq)**2 - (1.0 - 2.0 * Ybeta_n)**2)
            e_lep_yq, _ = nuclear_properties.default_electron_energy_density_and_chemical_potenial(
                nB, Yq, self._me, 197.3
            )
            e_lep_beta, _ = nuclear_properties.default_electron_energy_density_and_chemical_potenial(
                nB, Ybeta_n, self._me, 197.3
            )
            e_total = (e_beta + delta_e_nuc
                       + np.asarray(e_lep_yq, float) - np.asarray(e_lep_beta, float))
            return e_total / (self._mN * nB) - 1.0

        self.specific_internal_energy_of_nB_and_yq = _sie




class interpolated_eos_density_and_charge_fraction:
    """
    A 2-D EoS interpolated from a table of baryon density and charge fraction.
    """
    def __init__(self, thermo, nb, Yq, method="cubic"):
        """
        Initialize the 2-D interpolated EoS.

        Parameters
        ----------
        thermo : pandas.DataFrame
            DataFrame containing the thermodynamic quantities.
        nb : pandas.DataFrame
            DataFrame containing the baryon density values.
        Yq : pandas.DataFrame
            DataFrame containing the charge fraction values, with column ``"Yq"``.
        method : str, optional
            Interpolation method to use (default is "cubic").
        """
        self.thermo = thermo
        self.nb = nb
        self.Yq = Yq
        self.method = method

        nb_vals = np.asarray(nb["nb"])   # (n_nb,)
        Yq_vals = np.asarray(Yq["Yq"])  # (n_Yq,)
        n_nb, n_Yq = len(nb_vals), len(Yq_vals)

        if "inb" in thermo.columns and "iYq" in thermo.columns:
            # CompOSE path: sort by (inb, iYq) → nb-outer, Yq-inner → shape (n_nb, n_Yq)
            thermo_s = thermo.sort_values(["inb", "iYq"])
            p_2d = thermo_s["p_per_nb"].values.reshape(n_nb, n_Yq)
            e_2d = thermo_s["e_per_nbmn_minus_1"].values.reshape(n_nb, n_Yq)
        else:
            # MUSES / meshgrid(nb, Yq) path: Yq-outer, nb-inner → transpose to (n_nb, n_Yq)
            p_2d = thermo["p_per_nb"].values.reshape(n_Yq, n_nb).T
            e_2d = thermo["e_per_nbmn_minus_1"].values.reshape(n_Yq, n_nb).T

        self.pressure_of_nB_and_yq = scipy.interpolate.RegularGridInterpolator(
            (nb_vals, Yq_vals), p_2d, method=method, bounds_error=False, fill_value=None
        )
        self.specific_internal_energy_of_nB_and_yq = scipy.interpolate.RegularGridInterpolator(
            (nb_vals, Yq_vals), e_2d, method=method, bounds_error=False, fill_value=None
        )
    @staticmethod
    def from_3d_compose_table(thermo, nb, T, Yq):
        """
        Create a 2-D interpolated EoS from a 3-D table by fixing the charge fraction.

        Parameters
        ----------
        thermo : pandas.DataFrame
            DataFrame containing the thermodynamic quantities.
        nb : pandas.DataFrame
            DataFrame containing the baryon density values.
        T : pandas.DataFrame
            DataFrame containing the temperature values.
        Yq : pandas.DataFrame
            DataFrame containing the charge fraction values, with column ``"Yq"``.

        Returns
        -------
        interpolated_eos_density_and_charge_fraction
            An instance of the 2-D interpolated EoS with fixed charge fraction.
        """
        import temperance.external.read_3d_compose_table as read_3d_compose_table  # lazy
        # Find the thermo under chemical equilibrium (minimizing free energy).
        thermo_eq = read_3d_compose_table.find_2d_eos_by_taking_T_equal_minT(thermo, nb, T)

        # Return an instance of the 2-D interpolated EoS.
        return interpolated_eos_density_and_charge_fraction(thermo_eq, nb, Yq)
    @staticmethod
    def from_muses_format(table, T_ref=None, M_N=939.565, n_nb=50, n_yq=20):
        """
        Create an interpolated_eos_density_and_charge_fraction from a MUSES-format table.

        The MUSES format is a 10-column headerless CSV/whitespace file with columns:
          1  Temperature (T)                       [MeV]
          2  Baryon chemical potential (mu_B)       [MeV]
          3  Strange chemical potential (mu_S)      [MeV]
          4  Electric charge chemical potential (mu_Q) [MeV]
          5  Baryon density (n_B)                  [1/fm^3]
          6  Strange density (n_S)                 [1/fm^3]
          7  Electric charge density (n_Q)         [1/fm^3]
          8  Energy density (epsilon)              [MeV/fm^3]
          9  Pressure (P)                          [MeV/fm^3]
         10  Entropy density (s)                   [1/fm^3]

        Parameters
        ----------
        table : str or pandas.DataFrame
            Path to the MUSES file, or a pre-loaded DataFrame.  If a path,
            the file is read with automatic separator detection (comma or
            whitespace).
        T_ref : float or None
            Temperature slice to select.  Defaults to the minimum temperature
            present in the file (i.e. the cold slice).
        M_N : float
            Nucleon mass [MeV] used to compute the dimensionless binding energy
            e / (n_B * M_N) - 1.  Default is the free neutron mass 939.565 MeV.
        n_nb : int
            Number of points in the regular baryon-density output grid.
        n_yq : int
            Number of points in the regular charge-fraction output grid.

        Returns
        -------
        interpolated_eos_density_and_charge_fraction
        """
        import pandas as pd
        from scipy.interpolate import griddata

        MUSES_COLS = [
            "T", "mu_B", "mu_S", "mu_Q",
            "n_B", "n_S", "n_Q",
            "e", "p",
            "s",
        ]

        if isinstance(table, str):
            df = pd.read_csv(
                table, header=None, names=MUSES_COLS,
                sep=r"\s+|,", engine="python",
            )
        else:
            df = table.copy()
            df.columns = MUSES_COLS[: len(df.columns)]

        # Select temperature slice
        if T_ref is None:
            T_ref = float(df["T"].min())
        df = df[np.isclose(df["T"], T_ref, atol=1e-8 * max(1.0, abs(T_ref)))].copy()
        df = df[df["n_B"] > 0.0].copy()   # drop vacuum rows

        # Derived quantities
        df["Y_Q"] = df["n_Q"] / df["n_B"]
        df["p_per_nb"] = df["p"] / df["n_B"]
        df["e_per_nbmn_minus_1"] = df["e"] / (df["n_B"] * M_N) - 1.0

        # Build a regular (n_B, Y_Q) output grid by interpolating the
        # scattered (n_B, Y_Q) points from the MUSES sweep.
        points = df[["n_B", "Y_Q"]].values

        nb_grid = np.linspace(df["n_B"].min(), df["n_B"].max(), n_nb)
        Yq_grid = np.linspace(df["Y_Q"].min(), df["Y_Q"].max(), n_yq)
        NB, YQ  = np.meshgrid(nb_grid, Yq_grid)
        grid_pts = np.column_stack([NB.ravel(), YQ.ravel()])

        p_grid = griddata(points, df["p_per_nb"].values,          grid_pts, method="linear")
        e_grid = griddata(points, df["e_per_nbmn_minus_1"].values, grid_pts, method="linear")

        thermo = pd.DataFrame({
            "p_per_nb":             p_grid,
            "e_per_nbmn_minus_1":   e_grid,
        })
        nb_df = pd.DataFrame({"nb": nb_grid})
        Yq_df = pd.DataFrame({"Yq": Yq_grid})

        return interpolated_eos_density_and_charge_fraction(thermo, nb_df, Yq_df)
        