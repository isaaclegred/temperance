import numpy as np
import astropy
import astropy.units as u
import astropy.constants as constant

baryon_mass_in_mev = (constant.m_n/(u.MeV/constant.c**2)).si.value
baryon_mass_in_g = (constant.m_n/u.g).si.value

def cgs_density_to_nuclear(rho):
    return rho / (u.MeV/constant.c**2/u.fm**3/(u.g/u.cm**3)).si.value

def nuclear_density_to_cgs(rho):
    return rho/cgs_density_to_nuclear(1.0)
    

def nuclear_baryon_number_density_to_cgs_mass_density(nb):
    mass_density_with_dimensions = constant.m_n * nb * u.fm**-3
    return (mass_density_with_dimensions * u.cm**3/u.g).si.value

