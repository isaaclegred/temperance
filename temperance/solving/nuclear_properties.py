import numpy as np 
import pandas as pd
from jax import jit
import jax.numpy as jnp
xp = jnp
import jax
jax.config.update('jax_platform_name', 'cpu')
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import scipy.integrate as integrate
# import jax.lax.custom_root as custom_root
# Diagnostic plotting
import matplotlib.pyplot as plt

import temperance.core.result as result
from temperance.core.result import EoSPosterior
from temperance.sampling.eos_prior import EoSPriorSet
import astropy.units as u
import astropy.constants as c

# We work in units where c = 1, 

def cgs_to_per_fm3(rest_mass_density):
    """
    Convert a quantity from cgs units to per fm^3
    Arguments:
      rest_mass_density: The quantity to convert
    Returns:
      The number of baryons per fm^3
    """
    # TODO standardize units
    gram_per_mev = (1.0 *u.MeV / c.c**2 / u.g).decompose().value
    cm_3_per_fm_3 = (1.0*u.fm**3 / u.cm**3).decompose().value
    mev_per_baryon= (1.0* c.m_p /  (u.MeV/c.c**2) ).decompose().value
    return rest_mass_density  / (gram_per_mev * mev_per_baryon / cm_3_per_fm_3)
def cgs_to_mev_per_fm3(energy_density):
    """
    Convert a quantity from cgs units to MeV per fm^3
    Arguments:
      energy_density: The quantity to convert
    Returns:
      The energy density in MeV per fm^3
    """
    # TODO standardize units
    gram_per_mev = (1.0 *u.MeV / c.c**2 / u.g).decompose().value
    cm_3_per_fm_3 = (1.0*u.fm**3 / u.cm**3).decompose().value
    return energy_density  / (gram_per_mev / cm_3_per_fm_3)
# Follow the outline of 2102.10074
@jit
def default_symmetric_matter_energy_per_nucleon(n, n0, E0, K0):
    """
    Compute the symmetric matter energy density at a given density
    Arguments:
      n: The density at which to compute the energy density
      n0: The nuclear saturation density
      E0: The energy density at saturation
      K0: The incompressibility at saturation
    Returns:
      The energy per nucleon
    """
    z = (n/n0)
    return E0 + 1/2 * K0 * (1/3 * (z - 1)) ** 2
@jit
def compute_symmetry_energy_per_nucleon(n, n0, S0, L, K):
    """
    Compute the symmetry energy per nucleon at a given density
    Arguments:
      n: The density at which to compute the energy per nucleon
      S0: The symmetry energy at saturation
      L: The slope at saturation
      K: The skewness at saturation
    Returns:
      The symmetry energy per nucleon at the given density
    """
    z = (n/n0)
    return S0 + 1/3 * L * (z-1) + 1/18 * K * (z-1) ** 2
@jit
def default_energy_per_nucleon(n, Yq, n0, E0, K0, J0, L0, Ksym0):
    """
    Compute the energy per nucleon at a given density
    Arguments:
      n: The density at which to compute the energy per nucleon
      Yq: The charge fraction
      n0: The nuclear saturation density
      E0: The energy density at saturation
      K0: The incompressibility at saturation
      J0: The symmetry energy at saturation
      L0: The slope at saturation
      Ksym0: The symmetry energy at saturation
    Returns:
      The energy per nucleon at the given density
    """
    Esym = default_symmetric_matter_energy_per_nucleon(n, n0, E0, K0)
    z = (n/n0)
    S = compute_symmetry_energy_per_nucleon(n, n0, J0, L0, Ksym0)
    return Esym + S * (1 - 2 * Yq) ** 2

def get_electron_fraction_from_energy_and_symmetry_energy(
        total_energy, E_SNM, symmetry_energy):
    """
    Compute the electron fraction from the energy per nucleon and the symmetry energy
    Arguments:
      total_energy: The total energy per nucleon
      E_SNM: The binding energy per nucleon in symmetric matter
      symmetry_energy: The symmetry energy per nucleon
    Returns:
      The electron fraction
    """
    #E = E_SNM + (1-2*Yq)**2 * E_sym
    return 1/2 * (1 - np.sqrt((total_energy - E_SNM) / symmetry_energy))
@jit
def default_electron_energy_density_and_chemical_potenial(n, Yq, me, hbar):
    """
    Compute the energy per electron and the chemical potential
    Arguments:
      n: The density at which to compute the energy per electron
      Yq: The proton fraction
    Returns:
      The energy per electron and the chemical potential
    """
    ne = n * Yq
    xe = hbar * (3 * xp.pi ** 2 * ne) ** (1/3) / me
    electron_mass_times_c_over_hbar = me / hbar # ~1/(386  fm )
    electron_energy_density = (electron_mass_times_c_over_hbar)**3 * me/(8 * xp.pi**2) * (xe * (2 * xe**2 + 1) * xp.sqrt(xe**2 + 1) - xp.log(xe + xp.sqrt(xe**2 + 1)))
    chemical_potential = xp.sqrt(hbar**2*(3 * xp.pi**2*ne)**(2/3) + me**2)
    return electron_energy_density, chemical_potential 

def get_beta_eq_charge_fraction(n,
        mp, mn, mN, me, E0, K0, n0, beta_equilibrated_energy_density,
        get_electron_energy_and_chemical_potential=
        default_electron_energy_density_and_chemical_potenial,
        get_symmetric_energy_per_nucleon=default_symmetric_matter_energy_per_nucleon):
    """
    Compute the beta equilibrium equation
    Arguments:
      n: The density at which to compute the equation
      mp: The proton mass
      mn: The neutron mass
      mN: The nucleon mass
      beta_equilibrated_energy_density: The energy density at beta equilibrium
      get_electron_energy_and_chemical_potential: A function which computes the
      electron energy density and chemical potential
      get_symmetric_energy_per_nucleon: A function which computes the symmetric
      energy per nucleon

    Returns:
        The beta equilibrium equation, which can has a root at Yq = beta-equlibrium electron fraction
      """
    symmetric_energy_per_nucleon = get_symmetric_energy_per_nucleon(n, n0, E0, K0)

    delta_m = mp - mn 
    def beta_eq(Yq):
        # hbar in units of MeV *  fm / c
        electron_energy_density, chemical_potential = get_electron_energy_and_chemical_potential(n, Yq, me, 197.3)
        potential = (1-2*Yq) / 4 * (delta_m + chemical_potential)
        specific_energy_difference = (
          beta_equilibrated_energy_density - electron_energy_density) / n
        #print("eos beta equil energy density per particle", beta_equilibrated_energy_density / n)
        return ( potential - 
                (specific_energy_difference) + 
                symmetric_energy_per_nucleon)
    #print("bracket is", beta_eq(0.0001, verbose=False), beta_eq(.7, verbose=False))
    return optimize.root_scalar(beta_eq, bracket=[.000001, .99]) 
def eos_to_internal_energy_density(eos, density=0.16 * 2.8e14/0.16, energy_density_conversion=cgs_to_mev_per_fm3):
    """
    Compute the energy density at saturation for an EoS
    Arguments:
      eos: The EoS to compute the energy density at saturation
      saturation_density: The density at which to compute the energy density
      energy_density_conversion: A function which converts the energy density to the desired units
    Returns:
      The energy density at saturation
    """
    alpha = 1.0
    beta = alpha 
    #print("density is", 1/cgs_to_per_fm3(1.0) * density)
    index = 1
    h = np.exp(integrate.cumulative_simpson(1/(eos["pressurec2"] + eos["energy_densityc2"])[index:], x = eos["pressurec2"][index:], initial=0) ) * ((eos["pressurec2"] + eos["energy_densityc2"])/ eos["baryon_density"])[index]
    reintegrated_rho =np.array((eos["energy_densityc2"] + eos["pressurec2"])[index:] / h)
    return np.array(energy_density_conversion(interpolate.griddata(alpha * eos["baryon_density"], beta * eos["energy_densityc2"]- alpha*eos["baryon_density"], 1/cgs_to_per_fm3(1.0) * density, method="cubic")), dtype=float)
    #return np.array(energy_density_conversion(interpolate.griddata(reintegrated_rho, np.array(eos["energy_densityc2"][index:] - reintegrated_rho) , 1/cgs_to_per_fm3(1.0) * density, method="cubic")), dtype=float)


def get_symmetry_parameters(
    n,
    mp, mn, mN, me, E0, K0, n0, beta_equilibrated_energy_density,
    get_electron_energy_and_chemical_potential=
    default_electron_energy_density_and_chemical_potenial,
    get_symmetric_energy_per_nucleon=default_symmetric_matter_energy_per_nucleon,
    verbose=False):

    """
    Compute the symmetry parameters at saturation.
    Arguments:
      n: The range of densities to use to estimate the derivatives of the symmetry parameters
      mp: The proton mass
      mn: The neutron mass
      mN: The nucleon mass
      me: The electron mass
      E0: The energy density at saturation for symmetric matter
      K0: The incompressibility at saturation for symmetric matter
      n0: The nuclear saturation density
      beta_equilibrated_energy_density: The energy density at beta equilibrium
      get_electron_energy_and_chemical_potential: A function which computes the
      electron energy density and chemical potential
      get_symmetric_energy_per_nucleon: A function which computes the symmetric
      energy per nucleon
    Returns:
      The symmetry parameters S0, L0, Ksym, Ybeta
      S0: The symmetry energy at saturation
      L0: The slope of the symmetry energy at saturation
      Ksym: The skewness of the symmetry energy at saturation
      Ybeta: The beta equilibrium charge at saturation
    """
    symmetric_internal_energy = get_symmetric_energy_per_nucleon(n, n0, E0, K0)
    Y_q_beta = np.empty(len(n))
    for i in range(len(n)):
        try:
          sol =get_beta_eq_charge_fraction(n[i],
            mp, mn, mN, me, E0, K0, n0, beta_equilibrated_energy_density[i],
            get_electron_energy_and_chemical_potential=
            get_electron_energy_and_chemical_potential,
            get_symmetric_energy_per_nucleon=get_symmetric_energy_per_nucleon)
          #print("sol is", sol)
          Y_q_beta[i] = sol.root
        except ValueError:
          print("ERROR in rootfind for electron fraction")
          return (np.nan, np.nan, np.nan, np.nan)
    electron_energy_density, _ = get_electron_energy_and_chemical_potential(n, Y_q_beta, me, 197.3)
    delta_E_beta = beta_equilibrated_energy_density/n - electron_energy_density/n - symmetric_internal_energy
    # Compute S(n)  
    Sn = ((beta_equilibrated_energy_density - electron_energy_density)/n   - symmetric_internal_energy)/(1-2*Y_q_beta)**2
    # Evaluate at saturation 
    S0 = interpolate.griddata(n, Sn, n0)
    # Same for L
    Ln = 3 * n * np.gradient(Sn, n, edge_order=2) 
    L0 = interpolate.griddata(n, Ln, n0)
    # Same for K
    Ksymn = 3 * n**2 * np.gradient(Ln/n, n, edge_order=2)
    Ksym = interpolate.griddata(n, Ksymn, n0)
    # Same for charge fraciton
    Ybeta = interpolate.griddata(n, Y_q_beta, n0)
    return S0, L0, Ksym, Ybeta


def toy_example(eos_post, eos_prior, eos_samples=100):
    """
    Generate a toy example of the energy per nucleon
    and compute the equilibrium electron fraction for 
    a set of EoSs
    """
    rng = np.random.default_rng()
    eos_samples = eos_post.sample(size=eos_samples, weight_columns=[])
    # Masses in MeV  (TODO standardize units)
    mn = 939.5
    mp = 938.2
    me = 0.511
    # Atomic mass unit
    mN = 931.5
    symmetry_params = {}
    for eos_index in eos_samples["eos"]:
        print(eos_index)
       
        n0 = rng.normal(0.164, 0.007)
        E0 = rng.normal(-15.86, .57)
        K0 = rng.normal(215, 40)
        delta_n = .001
        
        n_test = np.linspace(n0-delta_n,n0+delta_n, 5)
        
        eos = pd.read_csv(eos_prior.get_eos_path(int(eos_index)))
        energy_density_in_eq = eos_to_internal_energy_density(eos, n_test)

        #sol = get_beta_eq_charge_fraction(n0, mp, mn, mN, me, E0, K0, n0, energy_density_in_eq)
        #charge_fractions[int(eos_index)] = sol.root
       
        symmetry_params[eos_index] = get_symmetry_parameters(n_test, mp, mn, mN, me, E0, K0, n0, energy_density_in_eq)
        print("inferred params", symmetry_params[eos_index])
    return symmetry_params



if __name__ == "__main__":
  rmf_eos_set=1109
  for key in ["1e14"]:
    eos_set_identifier=f"maxpc2-{key}"
    eos_tag = f"{rmf_eos_set}_{eos_set_identifier}_mrgagn_01d000_00d010"
    rmf_dir  = f"~/RMFGP/gp-rmf-inference/conditioned-priors/many-draws/{rmf_eos_set}/{eos_tag}"
    eos_post = EoSPosterior.from_csv(
        f"~/RMFGP/RealData/{eos_tag}_post.csv", label="astro" + "-" + eos_set_identifier + f"-{rmf_eos_set}")
    eos_prior = EoSPriorSet(
        eos_dir=rmf_dir, eos_column="eos", eos_per_dir=1000, macro_dir=rmf_dir,
        branches_data=None)
    #eos_post = EoSPosterior.from_csv("~/NewPulsar/collated_np_all_post.csv", label="astro")
    #eos_prior = EoSPriorSet.get_default()
    symmetry_params = toy_example(eos_post, eos_prior)
    symmetry_values = np.array(list(symmetry_params.values()))
    print(symmetry_values)
    symmetry_data = pd.DataFrame({"eos": np.array(list(symmetry_params.keys()))})
    symmetry_data["S0"] = symmetry_values[:, 0]
    symmetry_data["L"] = symmetry_values[:, 1]
    symmetry_data["Ksym"] = symmetry_values[:, 2]
    symmetry_data.to_csv("symmetry_data_limited.csv", index=False)
    print(symmetry_params) 





