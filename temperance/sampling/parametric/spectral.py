#Test Sampling eos's from spectral coefficients
import numpy as np
import piecewise_polytrope as pyeos
import scipy.interpolate as interp
import lalsimulation as lalsim
import lalinference as lalinf
import lal
import argparse
from matplotlib import pyplot as plt
import astropy.constants as const
import astropy.units as u


c = const.c.cgs.value
# Characteristic refinement number (probably should be changed on a case by case basis)
N = 100
M_sun_si = const.M_sun.si.value

# PARAMETRIC EOS
# Model of equation of state prior to sample from: 
# Need to sample gamma0, gamma1, gamma2, gamma3
# See https://arxiv.org/pdf/1805.11217.pdf for an introdcution


# Legacy support, should be removed at some point
p_range = (1e31, 1e37)
p_0 = 3.9e32

# This is the prior used in the introductory paper above
gamma0_range = (0.2, 2.0)
gamma1_range = (-1.6, 1.7)
gamma2_range = (-.6, .6)
gamma3_range = (-.02, .02)


# This is the preimage of the prior
# used in https://arxiv.org/pdf/2001.01747.pdf,
# it gets mapped to a relevant range of gammas
r0_range = (-4.37722, 4.91227)
r1_range = (-1.82240, 2.06387)
r2_range = (-.32445, .36469)
r3_range = (-.09529, .11046)


# This maps r's drawn from distributions (such
# as a distributioon with the values above as uniform
# bounds) and returns values of gammma which are more
# likely physical.  This mapping may also be inducing
# correlations in the priors; need to investigate further.
def map_rs_to_gammas(r0, r1, r2, r3):
    S = np.matrix([[.43801, -0.53573, +0.52661, -0.49379],
                   [-0.76705, +0.17169, +0.31255, -0.53336],
                   [+0.45143, 0.67967, -0.19454, -0.54443],
                   [+0.12646, 0.47070, 0.76626, 0.41868]])
    mu_r = np.matrix([[0.89421],[0.33878],[-0.07894],[+0.00393]])
    sigma_r = np.matrix([[0.35700,0,0,0],[0,0.25769,0,0],[0,0,0.05452,0],[0,0,0,0.00312]])
    rs = np.matrix([[r0],[r1], [r2], [r3]])
    return sigma_r * S**(-1) * rs  + mu_r





# This can be called as a script, in that case it produces a single "draw file" which contains
# a tabulated eos of (pressure, energy density, baryon density)
parser = argparse.ArgumentParser(description='Get the number of draws needed, could be expanded')
parser.add_argument("--num-draws", type=int, dest="num_draws", default=1)
parser.add_argument("--dir-index", type=int, dest="dir_index", default=0)
parser.add_argument("--prior-tag", type=str, dest="prior_tag", default="uniform")

# This class is meant to hose all of the functions needed to interact with a
# paramaterized eos, without actually exposing the client to any of the lalsimulation
# functions (which can be somewhat volatile and don't come with object orientation)
class eos_spectral:
    def __init__(self,gamma0, gamma1, gamma2, gamma3):
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3

        self.eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(
            self.gamma0,
            self.gamma1, 
            self.gamma2, 
            self.gamma3)

        self.family = lalsim.CreateSimNeutronStarFamily(self.eos)
    def get_params(self):
       return [self.gamma0, self.gamma1, self.gamma2, self.gamma3]
    # Get the eos family from the paramaters. 
    def get_eos(self):
        return self.eos
    def get_fam(self):
        return self.family
    # Evaluate the eos in terms of epsilon(p)
    def eval_energy_density(self, p):
        if isinstance(p, list) or isinstance(p, np.ndarray):    
            eps = np.zeros(len(p))  
            for i, pres in enumerate(p):
                eps[i] = lalsim.SimNeutronStarEOSEnergyDensityOfPressure(pres,self.eos)    
        else:
            eps = lalsim.SimNeutronStarEOSEnergyDensityOfPressure(p, self.eos)
        return eps
    # Evaluate the phi parameter as used in the non-parametric papers,
    # Not currently used
    def eval_phi(self, p):
        if isinstance(p, list) or isinstance(p, np.ndarray):    
            eps = np.zeros(len(p))  
            for i, pres in enumerate(p):
                eps[i] = lalsim.SimNeutronStarEOSEnergyDensityDerivOfPressure(pres,self.eos)    
        else:
             eps = lalsim.SimNeutronStarEOSEnergyDensityDerivOfPressure(p, self.eos)
        return eps
    # Evaluate the baryon density at a particular pressure
    def eval_baryon_density(self, p):
        if isinstance(p, list) or isinstance(p, np.ndarray):    
            rho = np.zeros(len(p))  
            for i, pres in enumerate(p):
                rho [i] = lalsim.SimNeutronStarEOSRestMassDensityOfPseudoEnthalpy(
                    lalsim.SimNeutronStarEOSPseudoEnthalpyOfPressure(pres,self.eos), self.eos)    
        else:
            rho  = lalsim.SimNeutronStarEOSRestMassDensityOfPseudoEnthalpy(
                lalsim.SimNeutronStarEOSPseudoEnthalpyOfPressure(p,self.eos), self.eos) 
        return rho
    # Evluate the speed of sound at a particular pressure
    def eval_speed_of_sound(self, p):
        if isinstance(p, list) or isinstance(p, np.ndarray):
            cs = np.zeros(len(p))
            for i, pres in enumerate(p):
                try:
                    h = lalsim.SimNeutronStarEOSPseudoEnthalpyOfPressure(pres, self.eos)
                    cs[i] = lalsim.SimNeutronStarEOSSpeedOfSound(h, self.eos)
                except:
                    print(pres, "failed to produce a valid sound speed")
                    break
        else:
            cs  = lalsim.SimNeutronStarEOSSpeedOfSound(p, self.eos)
        return cs
    #Evaluate the exponent polytope (I don't really know if this works, wouldn't recommend using)
    def eval_Gamma(self, p):
        x = np.log( p/p_0)
        return np.exp(self.gamma0 + self.gamma1 * x + self.gamma2 * x**2 + self.gamma3 * x**3 )
    # Return true if the local speed of sound is larger than the speed of light at the highest pressure allowed for a 
    # certain EOS
    def is_causal(self, ps):
        c_si = c/100 # The speed of sound in SI
        cs = self.eval_speed_of_sound(ps)
        cs_max = max(cs)
        print("cs_max is", cs_max)
        return cs_max < c_si*1.1
    def get_max_M(self):
        return lalsim.SimNeutronStarMaximumMass(self.family)/lal.MSUN_SI 
    # This function claims to check to see if the adiabatic exponent is bounded in the
    # range [.6, 4.5], it only works as well as the function which evaluates the gamma
    def is_confined(self, ps):
        if (.6 < self.eval_Gamma(ps).all() < 4.5):
            return True
        
############################################################
# Implemented Priors         
############################################################             

# Draws froam a uniform distribution, with the bounds as provided
# in the intro paper, basically useless because of a feature (bug)?
# in lalsim that causes a segfault if parameters produce an EOS
# with insufficient points to do the TOV integration.  So any code that
# runs this function is liable to fail catastrophically
def get_eos_realization_uniform_spec (gamma0_range = gamma0_range,
                                      gamma1_range= gamma1_range,
                                      gamma2_range=gamma2_range, 
                                      gamma3_range = gamma3_range):
    # There's some problem with configurations not working if the parameters are too close together
    gamma0 = np.random.uniform(*gamma0_range)
    gamma1 = np.random.uniform(*gamma1_range)
    gamma2 = np.random.uniform(*gamma2_range)
    gamma3 = np.random.uniform(*gamma3_range) 
    try:
        print([gamma0, gamma1, gamma2, gamma3])
        return eos_spectral(gamma0, gamma1, gamma2, gamma3)    

    except : 
        # try again :(
        return get_eos_realization_uniform_spec (gamma0_range = gamma0_range,
                                      gamma1_range= gamma1_range,
                                      gamma2_range=gamma2_range, 
                                      gamma3_range = gamma3_range)
        

# This is something of a wrapper of the lalinference
# function which checks to see if a particular combo
# of parameters will (1) create an EOS which can be used
# to successfully integrate TOV, and (2) create a physically
# plausible solution (perhaps the checks could be relaxed
# to expand the prior but its not really possible to dispense
# with it completely)
def criteria(gamma0, gamma1, gamma2, gamma3):
    vars = lalinf.Variables()
    no_vary = lalinf.lalinference.LALINFERENCE_PARAM_FIXED
    lalinf.lalinference.AddREAL8Variable(vars, "SDgamma0", gamma0, no_vary )
    lalinf.lalinference.AddREAL8Variable(vars, "SDgamma1", gamma1, no_vary)
    lalinf.lalinference.AddREAL8Variable(vars, "SDgamma2",  gamma2, no_vary)
    lalinf.lalinference.AddREAL8Variable(vars, "SDgamma3",  gamma3, no_vary)
    lalinf.lalinference.AddREAL8Variable(vars, "mass1",  1.4 , no_vary)
    lalinf.lalinference.AddREAL8Variable(vars, "mass2",  1.4 , no_vary)

    
    a = lal.CreateStringVector("Hi")
    process_ptable = lalinf.ParseCommandLineStringVector(a)
    success_param = lalinf.EOSPhysicalCheck(vars, process_ptable)
    if success_param == 0:
        return True
    else :
        return False
# Get an EOS sampled from a uniform prior that is supposed to 
# satisy the criteria, this is also fairly useless because
# most draws are not physical, so it takes an incredibly long
# time to get a sizeable sample of reasonable EOSs
def get_eos_realization_uniform_constrained_spec (gamma0_range = gamma0_range,
                                                  gamma1_range= gamma1_range,
                                                  gamma2_range=gamma2_range,
                                                  gamma3_range = gamma3_range):
    gamma0 = np.random.uniform(*gamma0_range)
    gamma1 = np.random.uniform(*gamma1_range)
    gamma2 = np.random.uniform(gamma2_range[0], gamma2_range[1])
    gamma3 = np.random.uniform(gamma3_range[0], gamma3_range[1])
    try: # Sometimes get random exceptions
        accepted = False
        while not(accepted):
            gamma0 = np.random.uniform(*gamma0_range)
            gamma1 = np.random.uniform(*gamma1_range)
            gamma2 = np.random.uniform(gamma2_range[0], gamma2_range[1])
            gamma3 = np.random.uniform(gamma3_range[0], gamma3_range[1])
            while(not criteria(gamma0, gamma1, gamma2, gamma3)):
                gamma0 = np.random.uniform(*gamma0_range)
                gamma1 = np.random.uniform(*gamma1_range)
                gamma2 = np.random.uniform(gamma2_range[0], gamma2_range[1])
                gamma3 = np.random.uniform(gamma3_range[0], gamma3_range[1])
            this_polytrope = eos_spectral(gamma0, gamma1, gamma2, gamma3)
            if  3.5 > this_polytrope.get_max_M() and 1.97 < this_polytrope.get_max_M():
                accepted = True
                
            
    except :
        # Try again
        return get_eos_realization_uniform_constrained_spec(gamma0_range = gamma0_range,
                                                            gamma1_range= gamma1_range,
                                                            gamma2_range=gamma2_range,
                                                            gamma3_range = gamma3_range)
    return this_polytrope
    

# Inspired by  https://arxiv.org/pdf/2001.01747.pdf appendix B, see there for help
# This prior uses a separate domain which gets mapped into gamma-space, in doing this
# it targets the most viable regions of parameter space, it is the de facto best choice
# for sampling
def get_eos_realization_mapped_constrained_spec (r0_range = r0_range,
                                                 r1_range= r1_range,
                                                 r2_range= r2_range,
                                                 r3_range = r3_range):
    r0 = np.random.uniform(*r0_range)
    r1 = np.random.uniform(*r1_range)
    r2 = np.random.uniform(*r2_range)
    r3 = np.random.uniform(*r3_range)

    gammas = map_rs_to_gammas(r0, r1, r2, r3)
    gamma0 = gammas[0,0]
    gamma1 = gammas[1,0]
    gamma2 = gammas[2,0]
    gamma3 = gammas[3,0]
    failure = False
    try :
        if not criteria(gamma0, gamma1, gamma2, gamma3):
            failure = True
        this_polytrope = eos_spectral(gamma0, gamma1, gamma2, gamma3)
        if 3 < this_polytrope.get_max_M() or 1.7 > this_polytrope.get_max_M():
            failure = True
        
    except :
        # Try again
        return get_eos_realization_mapped_constrained_spec(r0_range = r0_range,
                                                            r1_range= r1_range,
                                                            r2_range=r2_range,
                                                            r3_range = r3_range)
    if failure:
         return get_eos_realization_mapped_constrained_spec(r0_range = r0_range,
                                                            r1_range= r1_range,
                                                            r2_range= r2_range,
                                                            r3_range = r3_range)
    print(gamma0, gamma1, gamma2, gamma3)
    return this_polytrope



# Inspired by  https://arxiv.org/pdf/2001.01747.pdf appendix B, see there for help
def get_eos_realization_mapped_gaussian_constrained_spec (r0_range = r0_range,
                                                 r1_range= r1_range,
                                                 r2_range= r2_range,
                                                 r3_range = r3_range):
    ################################################################
    ranges= [r0_range, r1_range, r2_range, r3_range]
    means = [np.mean(this_range) for this_range in ranges]
    cov = 1/6*np.diag([np.std(this_range) for this_range in ranges])
    [r0, r1, r2, r3] = np.random.multivariate_normal(means, cov)  
    ################################################################
    gammas = map_rs_to_gammas(r0, r1, r2, r3)
    gamma0 = gammas[0,0]
    gamma1 = gammas[1,0]
    gamma2 = gammas[2,0]
    gamma3 = gammas[3,0]
    failure = False
    try :
        if not criteria(gamma0, gamma1, gamma2, gamma3):
            failure = True
        this_polytrope = eos_spectral(gamma0, gamma1, gamma2, gamma3)
        if 3 < this_polytrope.get_max_M() or 1.9 > this_polytrope.get_max_M():
            failure = True
    except :
        # Try again
        return get_eos_realization_mapped_gaussian_constrained_spec(r0_range = r0_range,
                                                            r1_range= r1_range,
                                                            r2_range=r2_range,
                                                            r3_range = r3_range)
    if failure:
        return get_eos_realization_mapped_gaussian_constrained_spec(r0_range = r0_range,
                                                            r1_range= r1_range,
                                                            r2_range= r2_range,
                                                            r3_range = r3_range)
    # This is a lot of printing, but makes it possible to diagnose the prior more easily
    print(gamma0, gamma1, gamma2, gamma3)
    return this_polytrope









# Stitch EoS onto the known EoS below nuclear saturation density. 
# Use Sly log(p1) = 34.384, gamma1 = 3.005, gamma2 = 2.988, gamma3 = 2.851
# There's some subtlety here related to where the pressure is known.  Here
# it is given at the dividing line between rho_1 and rho_2,  but we want it
# at the stitching point (I think this is fine though, because we can just 
# evaluate the modelt at the stiching point)

#SLy model
sly_polytrope_model = pyeos.eos_polytrope(34.384, 3.005, 2.988, 2.851)


def get_draw_function_from_tag(prior_tag):
    if prior_tag == "uniform":
        return get_eos_realization_mapped_constrained_spec
    elif prior_tag == "Uniform":
        return get_eos_realization_mapped_constrained_spec
    elif prior_tag == "gaussian":
        return get_eos_realization_mapped_gaussian_constrained_spec
    elif prior_tag == "Gaussian":
        return get_eos_realization_mapped_gaussian_constrained_spec
    elif prior_tag == "unmapped":
        print("actually using unmapped")
        return get_eos_realization_uniform_constrained_spec
    elif prior_tag == "Unmapped":
        return get_eos_realization_uniform_constrained_spec
    else:
        print("couldn't identify the spectral prior tag, using the uniform prior")
        return get_eos_realization_mapped_constrained_spec


 
def create_eos_draw_file(name, draw_function):
    #print(name)
    
    eos_poly = draw_function()
    if True:
    # FIXME WORRY ABOUT CGS VS SI!!!!! (Everything is in SI till the last step :/ ) 
        p_small = np.linspace(1.0e12, 1.3e30, 1000)
        p_main = np.geomspace(1.3e30, 9.0e36, 1100)
        eps_small=  eos_poly.eval_energy_density(p_small)
        eps_main = eos_poly.eval_energy_density(p_main)
        rho_b_small = eos_poly.eval_baryon_density(p_small)
        rho_b_main = eos_poly.eval_baryon_density(p_main)
        p = np.concatenate([p_small, p_main])
        eps = np.concatenate([eps_small, eps_main])
        rho_b = np.concatenate([rho_b_small, rho_b_main])
        data = np.transpose(np.stack([p/c**2*10 , eps/c**2*10, rho_b/10**3])) # *10 because Everything above is done in SI
        np.savetxt(name,data, header = 'pressurec2,energy_densityc2,baryon_density',
                   fmt='%.10e', delimiter=",", comments="")
        return eos_poly.get_params()
    else :
        return create_eos_draw_file(name)

if __name__ == "__main__":
    args = parser.parse_args()
    num_draws = args.num_draws
    dir_index = args.dir_index
    prior_tag = args.prior_tag
    print(prior_tag)
    parameters_used = []
    eos_nums = np.ndarray((num_draws, 1))
    for i in range(num_draws):
        eos_num = dir_index*num_draws + i
        name = "eos-draw-" + "%06d" %(eos_num) + ".csv"
        params = create_eos_draw_file(name, get_draw_function_from_tag(prior_tag))
        parameters_used.append(params)
        eos_nums[i,0] = eos_num
    metadata = np.concatenate([eos_nums, np.array(parameters_used)], axis=1)
    np.savetxt("eos_metadata-"+"%06d" %(dir_index) + ".csv", metadata, header="eos, gamma0, gamma1, gamma2, gamma3", delimiter=",", comments="")
