import numpy as np
import scipy
import pandas as pd
# For tidal love number calculation
from scipy.special import hyp2f1

# For plotting
import matplotlib.pyplot as plt

try :
    raise ImportError
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
    print("falling back to numpy")
    xp = np
    from scipy.interpolate import interp1d
    ode_solver = scipy.integrate.odeint
    from scipy.interpolate import interp1d as spinterp1d
    interp1d = lambda x, y, method, **kwargs : spinterp1d(x[0], y, **kwargs) 
    def print_val(statement, **kwargs):
        val = list(kwargs.items())[0]
        print(f"{statement} = {val}")
try :
    import jax.scipy as jsp
except ImportError:
    import scipy as jsp
try:
    import universality.tov as utov
except ImportError:
    pass

class polytropic_eos:
    def __init__(self, K, Gamma):
        self.K = K
        self.Gamma = Gamma
        self.rho_of_logenthalpy = lambda lnh: ((xp.exp(lnh) - 1.0) * (self.Gamma - 1.0) / (self.K * self.Gamma))**(1/(Gamma - 1.0))
        self.p_of_logenthalpy = lambda lnh: self.K * self.rho_of_logenthalpy(lnh)**self.Gamma
        self.e_of_logenthalpy = lambda lnh: self.rho_of_logenthalpy(lnh) + self.p_of_logenthalpy(lnh) / (self.Gamma - 1.0)
        # dp/drho/ de/drho = dp/drho / h
        self.cs2_of_logenthalpy = lambda lnh: self.Gamma * self.K * (self.rho_of_logenthalpy(lnh)**(self.Gamma - 1.0)) / xp.exp(lnh)
        self.logenthalpy_of_rho = lambda rho: xp.log(1.0 + self.K * (self.Gamma) 
                                            *(rho**(self.Gamma - 1.0)) / (self.Gamma - 1.0) ) 
        self.p_of_rho = lambda rho: self.K * rho**self.Gamma
        self.e_of_rho = lambda rho: rho + self.p_of_rho(rho) / (self.Gamma - 1.0)

class css_eos:
    def __init__(self, cs2, e0):
        """
        cs2 the constant speed of sound, and e0 the energy density where we switch to a 
        cs2 = 0 EoS
        """
        self.cs2 = cs2
        self.e0 = e0
        prefactor = (e0) * (e0 / (1 + cs2))**(-1/(1 + cs2))
        self.p_of_e = lambda e: xp.where(e < self.e0, 0, self.cs2 * (e - self.e0))
        self.rho_of_e = lambda e: xp.where( e < self.e0, e, (e-self.e0*self.cs2)**(1/(1 + self.cs2 )) * prefactor)
        self.enthalpy_of_e = lambda e: (e + self.p_of_e(e)) / self.rho_of_e(e)
        self.e_of_rho = lambda rho: xp.where(rho < self.e0, rho, (rho/prefactor)**(1+self.cs2)/(1+self.cs2)+ self.e0*self.cs2)
        self.e_of_logenthalpy = lambda lnh: e0 * (1 + 2*self.cs2) * xp.exp((1+self.cs2/self.cs2) * lnh ) / (1 + self.cs2)
        self.p_of_logenthalpy = lambda lnh: self.p_of_e(self.e_of_logenthalpy(lnh))
        self.rho_of_logenthalpy = lambda lnh: self.rho_of_e(self.e_of_logenthalpy(lnh))
        self.cs2_of_logenthalpy = lambda lnh: xp.full_like(lnh, self.cs2)
        self.logenthalpy_of_rho = lambda rho: xp.log(self.enthalpy_of_e(self.e_of_rho(rho)))


class interpolated_eos:
    def __init__(self, eos, conversion_factor = 1/2.8e14 * .00045):
        self.eos = eos
        logenthalpy = xp.array(scipy.integrate.cumulative_trapezoid(
            1/(eos["pressurec2"] + eos["energy_densityc2"]), 
            eos["pressurec2"], initial=0) + np.log(eos["energy_densityc2"][0]+ eos["pressurec2"][0]) - np.log(eos["baryon_density"][0]))
        print_val("Value of logenthalpy: {logenthalpy}", logenthalpy=logenthalpy)
        print(logenthalpy[:].shape)
        baryon_density = xp.array(eos["baryon_density"]) * conversion_factor
        print_val("Value of baryon_density: {baryon_density}", baryon_density=baryon_density)
        print(baryon_density.shape)
        pressurec2 = xp.array(eos["pressurec2"]) * conversion_factor
        energy_densityc2 = xp.array(eos["energy_densityc2"]) * conversion_factor

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
    A = 2 / f * (1 - 3 * v - 2 * xp.pi * u * (e +3*p))
    B = 1 / f * ((ell + 1) * ell - 4 * xp.pi * u * (e + p) * (3 + one_over_cs2))
    return -prefactor * (eta * (eta- 1) + A * eta - B)

def lindblom_solver(lnhc, eos, max_iter=1000, tol=1e-6, termination_lnh=1e-14, ell=2, points_to_solve_for=500):
    """
    Solve the TOV equations out to a surface of constant logenthalpy
    """
    # Initial step in logenthalpy
    initial_stepsize = xp.array(1e-5) * lnhc
    ec = eos.e_of_logenthalpy(xp.array([lnhc]))[0]
    print_val("Logenthalpy central value  is:{lnhc}", lnhc=lnhc)
    print_val("Value of ec: {ec}", ec=ec)
    w_initial = eos.p_of_logenthalpy(xp.array([lnhc]))[0] / ec
    # u = r^2, v = m(r) / r, eta=ell
    initial_state = xp.array([6/np.sqrt(4 * np.pi * ec)*initial_stepsize/(1+3*w_initial), 2*initial_stepsize/(1+3*w_initial), ell, 0.0])
    print_val("Initial state: {x}", x=initial_state)
    def rhs(state, minus_lnh):
        lnh = -minus_lnh
        if lnh < 0.0:
            # this is dumb, but can't guarantee the solver won't try to call the 
            # rhs beyond the surface
            du=100
            dv=100
            deta = 1000
            dmb = 1000
            return xp.array([du, dv, deta, dmb])
        u = state[0]
        v = state[1]
        eta = state[2]
        v_b = state[3]
        #print_val("Value of lnh: {x}", x=lnh)
        p = eos.p_of_logenthalpy(xp.array([lnh]))[0]
        e = eos.e_of_logenthalpy(xp.array([lnh]))[0]
        cs2 = eos.cs2_of_logenthalpy(xp.array([lnh]))[0]
        rho = eos.rho_of_logenthalpy(xp.array([lnh]))[0]

        # jax.debug.print("Value of p: {p}", p=p)
        # jax.debug.print("Value of e: {e}", e=e)
        # common factor
        v_over_u = v / u
        cf =  (1-2*v) / (4 * xp.pi * p + v_over_u)
        # jax.debug.print("Value of cf: {cf}", cf=cf)
        du=  2 * cf
        dv =  (4 * xp.pi * e  - v_over_u) * cf
        # jax.debug.print("Value of du: {du}", du=du)
        # jax.debug.print("Value of dv: {dv}", dv=dv)
        one_over_cs2 = 1/cs2
        deta = lindblom_tidal_deformability_rhs(eta, u, du, v, e, p, one_over_cs2, ell=ell)

        # Baryon mass equation is a bit different
        dv_b = (4 * xp.pi * rho / xp.sqrt(1 - 2*v) - v_b / u) * cf

        return xp.array([du, dv, deta, dv_b])


    lnhs_solved = xp.linspace(-lnhc, -termination_lnh, points_to_solve_for)
    solution = ode_solver(rhs, initial_state, lnhs_solved)
    return -lnhs_solved, solution

def eta_to_love_number(etaR, C, ell=2):
    """
    Convert the surface value of eta to the tidal love number k2

    Heavily borrowed from Philippe Landry's `nsstruc` package
    """
    if ell != 2:
        raise NotImplementedError("Only ell=2 is implemented")
    fR = 1.-2.*C
    F = hyp2f1(3.,5.,6.,2.*C) # a hypergeometric function
    def dFdz():
        z = 2.*C
        return (5./(2.*z**6.))*(z*(-60.+z*(150.+z*(-110.+3.*z*(5.+z))))/(z-1.)**3+60.*np.log(1.-z))
    RdFdr = -2.*C*dFdz() # log derivative of hypergeometric function
    k2el = 0.5*(etaR-2.-4.*C/fR)/(RdFdr-F*(etaR+3.-4.*C/fR)) # gravitoelectric quadrupole Love number
    return (2./3.)*(k2el/C**5)


def solve_tov(eos, rho_c, solver="Lindblom", max_iter=1000, tol=1e-6, *args, **kwargs):
    """
    Solve the Tolman-Oppenheimer-Volkoff equations for a given central density 
    and equation of state.

    By default we use a jaxified version of the lindblom solver based on
    solving out to a surface of constant logenthalpy
    """
    if not (isinstance(eos, interpolated_eos) or isinstance(eos, polytropic_eos) or isinstance(eos, css_eos)):
        eos = interpolated_eos(eos)
    if solver == "Lindblom":
        print("rhoc is" , rho_c)
        lnhc = eos.logenthalpy_of_rho(rho_c)[0]
        print("solving for logenthalpy", lnhc)
        return lindblom_solver(lnhc, eos, max_iter=max_iter, tol=tol, *args, **kwargs)

def get_tov_family(eos, densities, outpath=None):
    """
    Solve the TOV Equations for a range of central densities and return the mass and radius
    of the resulting stars. The mass and radius are returned in a pandas dataframe.
    The mass is in solar masses and the radius is in km.
    The function also saves the data to a csv file if outpath is provided.
    """
    Rs = []
    Ms = []
    Lambdas = []
    rhocs = []
    Ms_baryon = []
    if not (isinstance(eos, interpolated_eos) or isinstance(eos, polytropic_eos) or isinstance(eos, css_eos)):
        eos = interpolated_eos(eos)
    for density in densities:
        density = xp.array([density])
        lnhs, sols = solve_tov(eos, density)
        u_term = sols[-1, 0]
        v_term =sols[-1, 1]
        eta_term = sols[-1, 2]
        vb_term = sols[-1, 3]
        Rs.append(np.sqrt(u_term) * 1.477) # in km
        Ms.append(np.sqrt(u_term) * v_term) # in solar masses
        Lambdas.append(eta_to_love_number(eta_term, v_term) )
        rhocs.append(eos.rho_of_logenthalpy(xp.array([lnhs[0]]))[0]*2.8e14/.00045)
        Ms_baryon.append(np.sqrt(u_term) * vb_term)
    data = pd.DataFrame({"central_baryon_density": np.array(rhocs),  "M":np.array(Ms), "R":np.array(Rs), "Lambda":np.array(Lambdas), "M_baryon": np.array(Ms_baryon)})
    if outpath is not None:
        data.to_csv(outpath, index=False)
    return data
    
if __name__ == "__main__":
    #eos = pd.read_csv("./qmc-rmf-3.csv")
    eos = polytropic_eos(K=123.6, Gamma=2.0)
    tabulation_densities = np.geomspace(.00045e-10,.00045e1, 200)
    poly_pressure = eos.p_of_rho(tabulation_densities)
    poly_energy = eos.e_of_rho(tabulation_densities)
    poly_eos = pd.DataFrame({"baryon_density":tabulation_densities*2.8e14/.00045, "pressurec2":poly_pressure*2.8e14/.00045, "energy_densityc2":poly_energy*2.8e14/.00045})
    poly_eos.to_csv("poly_100_2.csv", index=False)


    densities=  np.linspace(.00045, 8 * .00045, 100)
    family= get_tov_family(eos, densities, outpath="macro-poly_100_2.csv")
    plt.plot(family["R"], family["M"])
    plt.xlabel("R (km)")
    plt.ylabel("M (Msun)")
    plt.xlim(10, 20)
    plt.ylim(.2, 2.5)
    # Compare to universality solution
    umacro = pd.read_csv("umacro-poly_100_2.csv")
    plt.plot(umacro["R"], umacro["M"], ls="--", label="Universality")
    plt.savefig("mr-poly_100_2.pdf", bbox_inches="tight")
    plt.clf()
    plt.plot(family["M"], family["Lambda"])
    plt.yscale("log")
    plt.xlabel("M (Msun)")
    plt.ylabel("Lambda")
    plt.plot(umacro["M"], umacro["Lambda"], ls="--", label="Universality")
    plt.xlim(.2, 2.5)
    plt.ylim(1, 1e4)
    plt.savefig("mlambda-poly_100_2.pdf", bbox_inches="tight")
       
