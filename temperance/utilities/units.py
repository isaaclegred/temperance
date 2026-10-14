"""
Unit conventions.

temperance works in geometric units, G = c = Msun = 1, so the unit of
length (and of time, in light-travel distance) is G Msun / c^2.

The density conversion is a *convention*: temperance has always mapped
nuclear saturation density, 2.8e14 g/cm^3, to 0.00045, i.e.
CGS_DENSITY_TO_GEOM = 0.00045 / 2.8e14.  With KM_PER_MSUN below, the exact
factor would map 2.8e14 g/cm^3 to 4.534e-4 (0.75% larger).  The rounded
value is kept so tabulated EOSs give the same stars as before.

The geometric constants are plain numbers.  astropy is only imported by the
nuclear-unit helpers at the bottom, when they are first used.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Geometric units
# ---------------------------------------------------------------------------

#: G Msun / c^2 in km (the geometric unit of length)
KM_PER_MSUN = 1.4766250614
#: speed of light in km/s
SPEED_OF_LIGHT_KM_S = 299792.458
#: nuclear saturation (mass) density, the conventional value, in g/cm^3
RHO_SAT_CGS = 2.8e14
#: nuclear saturation density in geometric units (temperance convention)
RHO_SAT_GEOM = 0.00045
#: multiply a density (or pressure / c^2) in g/cm^3 by this to get geometric units
CGS_DENSITY_TO_GEOM = RHO_SAT_GEOM / RHO_SAT_CGS
#: multiply an angular frequency in units of 1/Msun by this to get a frequency in Hz
OMEGA_TO_HZ = SPEED_OF_LIGHT_KM_S / (2.0 * np.pi * KM_PER_MSUN)
#: multiply a time in units of Msun (e.g. a damping time 1/|Im omega|) by this to get seconds
MSUN_TO_S = KM_PER_MSUN / SPEED_OF_LIGHT_KM_S

# ---------------------------------------------------------------------------
# Nuclear units
# ---------------------------------------------------------------------------

def _astropy():
    import astropy.units as u
    import astropy.constants as constant
    return u, constant


def __getattr__(name):
    # baryon_mass_in_mev / baryon_mass_in_g need astropy, so compute them on first use
    u, constant = _astropy()
    if name == "baryon_mass_in_mev":
        return (constant.m_n/(u.MeV/constant.c**2)).si.value
    if name == "baryon_mass_in_g":
        return (constant.m_n/u.g).si.value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def cgs_density_to_nuclear(rho):
    u, constant = _astropy()
    return rho / (u.MeV/constant.c**2/u.fm**3/(u.g/u.cm**3)).si.value

def nuclear_density_to_cgs(rho):
    return rho/cgs_density_to_nuclear(1.0)
    

def nuclear_baryon_number_density_to_cgs_mass_density(nb):
    u, constant = _astropy()
    mass_density_with_dimensions = constant.m_n * nb * u.fm**-3
    return (mass_density_with_dimensions * u.cm**3/u.g).si.value

