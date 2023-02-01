import numpy as np
import astropy
import astropy.units as u
import astropy.constants as constant

def cgs_density_to_nuclear(rho):
    return rho / (u.MeV/constant.c**2/u.fm**3/(u.g/u.cm**3)).si.value

baryon_mass_in_mev = (constant.m_n/(u.MeV/constant.c**2)).si.value
