import numpy as np
import scipy
import pandas as pd
try :
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
except ImportError:
    xp = np
    ode_solver = scipy.integrate.odeint

try :
    import jax.scipy as jsp
except ImportError:
    import scipy as jsp
try:
    import universality.tov as utov
except ImportError:
    pass

class interpolated_eos:
    def __init__(self, eos):
        self.eos = eos
        logenthalpy = xp.array(scipy.integrate.cumulative_trapezoid(
            1/(eos["pressurec2"] + eos["energy_densityc2"]), 
            eos["pressurec2"], initial=0) + np.log(eos["energy_densityc2"][0]+ eos["pressurec2"][0]) - np.log(eos["baryon_density"][0]))
        jax.debug.print("Value of logenthalpy: {logenthalpy}", logenthalpy=logenthalpy)
        print(logenthalpy[:].shape)
        baryon_density = xp.array(eos["baryon_density"])/2.8e14 * .00045
        jax.debug.print("Value of baryon_density: {baryon_density}", baryon_density=baryon_density)
        print(baryon_density.shape)
        pressurec2 = xp.array(eos["pressurec2"])/2.8e14 * .00045
        energy_densityc2 = xp.array(eos["energy_densityc2"])/2.8e14 * .00045
        cs2 = xp.array(np.gradient(logenthalpy, np.log(baryon_density)))
        self.cs2_of_logenthalpy = interp1d((logenthalpy,), cs2, method="linear")
        self.rho_of_logenthalpy = interp1d((logenthalpy,), baryon_density, method="linear")
        self.p_of_rho = interp1d((baryon_density,), pressurec2, method="linear")
        self.e_of_rho = interp1d((baryon_density,), energy_densityc2, method="linear")
        self.p_of_logenthalpy = interp1d((logenthalpy,), pressurec2, method="linear")
        self.e_of_logenthalpy = interp1d((logenthalpy,), energy_densityc2, method="linear")
        self.logenthalpy_of_rho = interp1d((baryon_density,), logenthalpy, method="linear")
        

def lindblom_tidal_deformability_rhs(eta, u, du,  v, e, p, one_over_cs2, ell=2):
    """
    RHS of the tidal love number ode using the lindblom form of TOV
    """
    f = 1-2*v
    prefactor = 1 / (2*u) * du
    A = 2 / f * (1 - 3 * v - 2 * xp.pi * u * (e + p))
    B = 1 / f * ((ell + 1) * ell) - 4 * xp.pi * u * (e + p) * (3 + one_over_cs2)
    return -prefactor * (eta * (eta- 1) + A * eta - B)

def lindblom_solver(lnhc, eos, max_iter=1000, tol=1e-6, termination_lnh=1e-14, ell=2):
    """
    Solve the TOV equations out to a surface of constant logenthalpy
    """
    initial_stepsize = xp.array(1e-3)
    ec = eos.e_of_logenthalpy(xp.array([lnhc]))[0]
    jax.debug.print("Value of ec: {ec}", ec=ec)
    # u = r^2, v = m(r) / r, eta=ell
    initial_state = xp.array([initial_stepsize**2, 4/3*xp.pi*initial_stepsize**2*ec, ell])
    jax.debug.print("Initial state: {x}", x=initial_state)
    def rhs(state, minus_lnh):
        lnh = -minus_lnh
        u = state[0]
        v = state[1]
        eta = state[2]
        p = eos.p_of_logenthalpy(xp.array([lnh]))[0]
        e = eos.e_of_logenthalpy(xp.array([lnh]))[0]
        cs2 = eos.cs2_of_logenthalpy(xp.array([lnh]))[0]
        
        # jax.debug.print("Value of lnh: {x}", x=lnh)
        # jax.debug.print("Value of p: {p}", p=p)
        # jax.debug.print("Value of e: {e}", e=e)
        # common factor
        v_over_u = v / u
        cf =  (1-2*v) /(4 * xp.pi * p + v_over_u)
        # jax.debug.print("Value of cf: {cf}", cf=cf)
        du=  2 * cf
        dv =  (4 * xp.pi * e  - v_over_u) * cf
        # jax.debug.print("Value of du: {du}", du=du)
        # jax.debug.print("Value of dv: {dv}", dv=dv)
        one_over_cs2 = 1/cs2
        deta = lindblom_tidal_deformability_rhs(eta, u, du, v, e, p, one_over_cs2, ell=ell)

        return xp.array([du, dv, deta])
    lnhs_solved = xp.linspace(-lnhc, -termination_lnh, 20)
    solution = ode_solver(rhs, initial_state, lnhs_solved)
    return lnhs_solved, solution



def solve_tov(eos, rho_c, solver="Lindblom", max_iter=1000, tol=1e-6, *args, **kwargs):
    """
    Solve the Tolman-Oppenheimer-Volkoff equations for a given central density 
    and equation of state.

    By default we use a jaxified version of the lindblom solver based on
    solving out to a surface of constant logenthalpy
    """
    if not isinstance(eos, interpolated_eos):
        eos = interpolated_eos(eos)
    if solver == "Lindblom":
        lnhc = eos.logenthalpy_of_rho(rho_c)[0]
        return lindblom_solver(lnhc, eos, max_iter=max_iter, tol=tol, *args, **kwargs)

        
if __name__ == "__main__":
    eos = pd.read_csv("./rmf-1109-000338-bps.csv")
    lnhs, sols = solve_tov(eos, xp.array([1.7e-3]))
    print(sols[-1, :])
       
