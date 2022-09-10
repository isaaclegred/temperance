# Test Sampling eos's from parametric values
import numpy as np
import scipy.interpolate as interp
import lalsimulation as lalsim
import lal
import lalinference as lalinf
import argparse
import matplotlib as mpl
mpl.use("agg")
from matplotlib import pyplot as plt
import astropy.constants as const
import astropy.units as u


c = const.c.cgs.value
# Characteristic refinement number (probably should be changed on a case by case basis)
N = 200
M_sun_si = const.M_sun.si.value

# Model of equation of state prior to sample from: 
# Need to sample gamma1, gamma2, gamma3, and logp1
# From the Lackey and Wade paper (I don't actually )
logp1_range = (33.6, 35.4)
gamma1_range  = (1.9, 4.5)
gamma2_range = (1.1, 5)
gamma3_range = (1.1, 5)

parser = argparse.ArgumentParser(description='Get the number of draws needed, could be expanded')
parser.add_argument("--num-draws", type=int, dest="num_draws", default=1)
parser.add_argument("--dir-index", type=int, dest="dir_index", default=0)
parser.add_argument("--prior-tag", type=str, dest="prior_tag", default="uniform")



# need
class eos_polytrope:
    def __init__(self,logp1, gamma1, gamma2, gamma3):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.logp1 = logp1
        self.eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(
                                                             self.logp1-1,
                                                            self.gamma1, 
                                                            self.gamma2, 
                                                            self.gamma3)
        self.family = lalsim.CreateSimNeutronStarFamily(self.eos)

    def get_params(self):
        return [self.gamma1, self.gamma2, self.gamma3, self.logp1]
   
    # Get the eos family from the paramaters. 
    def get_eos(self):
        return self.family.eos
    # Evaluate the eos in terms of epsilon(p)
    def eval_energy_density(self, p):
        if isinstance(p, list) or isinstance(p, np.ndarray):    
            eps = np.zeros(len(p))  
            for i, pres in enumerate(p):
                eps[i] = lalsim.SimNeutronStarEOSEnergyDensityOfPressure(pres,self.eos)    
        else:
             eps = lalsim.SimNeutronStarEOSEnergyDensityOfPressure(p, self.eos)
        return eps
    def eval_phi(self, p):
        if isinstance(p, list) or isinstance(p, np.ndarray):    
            eps = np.zeros(len(p))  
            for i, pres in enumerate(p):
                eps[i] = lalsim.SimNeutronStarEOSEnergyDensityDerivOfPressure(pres,self.eos)    
        else:
             eps = lalsim.SimNeutronStarEOSEnergyDensityDerivOfPressure(p, self.eos)
        return eps
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
            h = lalsim.SimNeutronStarEOSPseudoEnthalpyOfPressure(p, self.eos)
            cs  = lalsim.SimNeutronStarEOSSpeedOfSound(h, self.eos)
        return cs
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
    # Return true if the local speed of sound is larger than the speed of light at the highest pressure allowed for a 
    # certain EOS
    def is_causal(self):
        p_max = lalsim.SimNeutronStarEOSMaxPressure(self.eos)
        c_s_max= lalsim.SimNeutronStarEOSSpeedOfSound(
                    lalsim.SimNeutronStarEOSPseudoEnthalpyOfPressure(p_max,self.eos), self.eos)
        return c_s_max < c/100*1.1   # Conversion from cgs to SI (leave some leeway like in 1805.11217)
    def is_M_big_enough(self):
        m_max = lalsim.SimNeutronStarMaximumMass(self.family)
        return m_max > 1.76 * M_sun_si
        

             

# We also need gamma1 > gamma2 > gamma3 ? and thermodynamic stability? , so I guess we sample gamma1 first 
# and then constrain the others based on this. This is the (somewhat) uniform prior on the gamma's, I think
# I still need to glue together the 
def get_eos_realization_uniform_poly (logp1_range = logp1_range, gamma1_range= gamma1_range, gamma2_range=gamma2_range, 
                                        gamma3_range = gamma3_range):
    # There's some problem with configurations not working if the parameters are too close together,
    # so I tried to force them apart without losing too much of the prior
    eps = .1
    gamma1 = np.random.uniform(*gamma1_range)
    gamma2 = np.random.uniform(gamma2_range[0]+eps, gamma1 - eps)
    gamma3 = np.random.uniform(gamma3_range[0], gamma2 - eps) 
    logp1 = np.random.uniform(*logp1_range)
    return eos_polytrope(logp1, gamma1, gamma2, gamma3)    
 

# Enforce conditions ahead of time
def criteria(logp1, gamma1, gamma2, gamma3):
    vars = lalinf.Variables()
    no_vary = lalinf.lalinference.LALINFERENCE_PARAM_FIXED
    lalinf.lalinference.AddREAL8Variable(vars, "logp1", logp1, no_vary )
    lalinf.lalinference.AddREAL8Variable(vars, "gamma1", gamma1, no_vary)
    lalinf.lalinference.AddREAL8Variable(vars, "gamma2",  gamma2, no_vary)
    lalinf.lalinference.AddREAL8Variable(vars, "gamma3",  gamma3, no_vary)
    lalinf.lalinference.AddREAL8Variable(vars, "mass1",  1.4 , no_vary)
    lalinf.lalinference.AddREAL8Variable(vars, "mass2",  1.4 , no_vary)
    
    a = lal.CreateStringVector("Hi")
    process_ptable = lalinf.ParseCommandLineStringVector(a)
    success_param = lalinf.EOSPhysicalCheck(vars, process_ptable)
    if success_param == 0:
        return True
    else :
        return False


def get_eos_realization_uniform_constrained_poly (logp1_range = logp1_range,
                                                  gamma1_range= gamma1_range,
                                                  gamma2_range=gamma2_range,
                                                  gamma3_range = gamma3_range):
    # There's some problem with configurations not working if the parameters are too close together,
    # so I tried to force them apart without losing too much of the prior
    gamma1 = np.random.uniform(*gamma1_range)
    gamma2 = np.random.uniform(*gamma2_range)
    gamma3 = np.random.uniform(*gamma3_range)
    logp1 = np.random.uniform(*logp1_range)
    this_polytrope = eos_polytrope(logp1, gamma1, gamma2, gamma3)
    if criteria(logp1, gamma1, gamma2, gamma3):
        return this_polytrope
    else:
        return get_eos_realization_uniform_constrained_poly(logp1_range = logp1_range,
                                                            gamma1_range= gamma1_range,
                                      gamma2_range=gamma2_range, gamma3_range = gamma3_range)



def get_eos_realization_gaussian_constrained_poly (logp1_range = logp1_range,
                                              gamma1_range= gamma1_range,
                                              gamma2_range=gamma2_range,
                                              gamma3_range = gamma3_range):
    ################################################################
    ranges= [logp1_range, gamma1_range, gamma2_range, gamma3_range]
    means = [np.mean(this_range) for this_range in ranges]
    cov = 1/6*np.diag([np.std(this_range) for this_range in ranges])
    [logp1, gamma1, gamma2, gamma3] = np.random.multivariate_normal(means, cov) 
    
    

    
    ################################################################
    failure = False
    try :
        if not criteria(logp1, gamma1, gamma2, gamma3):
            failure = True
        this_polytrope = eos_polytrope(logp1, gamma1, gamma2, gamma3)
        if 3 < this_polytrope.get_max_M() or 1.9 > this_polytrope.get_max_M():
            failure = True
    except :
        # Try again
        return get_eos_realization_gaussian_constrained_poly(logp1_range = logp1_range,
                                                             gamma1_range= gamma1_range,
                                                             gamma2_range=gamma2_range,
                                                             gamma3_range = gamma3_range)
    if failure:
        return get_eos_realization_gaussian_constrained_poly(logp1_range = logp1_range,
                                                             gamma1_range= gamma1_range,
                                                             gamma2_range=gamma2_range,
                                                             gamma3_range = gamma3_range)
        # This is a lot of printing, but makes it possible to diagnose the prior more easily
    print(logp1, gamma1, gamma2, gamma3)
    return this_polytrope

# Stitch EoS onto the known EoS below nuclear saturation density. 
# Use Sly log(p1) = 34.384, gamma1 = 3.005, gamma2 = 2.988, gamma3 = 2.851
# There's some subtlety here related to where the pressure is known.  Here
# it is given at the dividing line between rho_1 and rho_2,  but we want it
# at the stitching point (I think this is fine though, because we can just 
# evaluate the modelt at the stiching point)

#SLy model
sly_polytrope_model = eos_polytrope(34.384, 3.005, 2.988, 2.851)


def get_draw_function_from_tag(prior_tag):
    if prior_tag == "uniform":
        return get_eos_realization_uniform_constrained_poly
    elif prior_tag == "Uniform":
        return get_eos_realization_uniform_constrained_poly
    elif prior_tag == "gaussian":
        return get_eos_realization_gaussian_constrained_poly
    elif prior_tag == "Gaussian":
        return get_eos_realization_gaussian_constrained_poly
    elif prior_tag == "unconstrained":
        print("you're nuts")
        return get_eos_realization_uniform_poly
    elif prior_tag == "Unconstrained":
        print("you're nuts")
        return get_eos_realization_uniform_poly
    else:
        print("couldn't identify the piecewise prior tag, using the uniform prior")
        return get_eos_realization_uniform_constrained_poly

def create_eos_draw_file(name, draw_function):
    eos_poly = draw_function()
    p_small = np.linspace(1.0e12, 1.3e30, 800)
    p_main = np.geomspace (1.3e30, 9.0e36, 900)
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

if __name__ == "__main__":
    args = parser.parse_args()
    num_draws = args.num_draws
    dir_index = args.dir_index
    prior_tag = args.prior_tag
    parameters_used = []
    eos_nums = np.ndarray((num_draws, 1))
    for i in range(num_draws):
        eos_num = dir_index*num_draws + i
        name = "eos-draw-" + "%06d" %(eos_num) + ".csv"
        params = create_eos_draw_file(name, get_draw_function_from_tag(prior_tag))
        parameters_used.append(params)
        eos_nums[i,0] = eos_num
    metadata = np.concatenate([eos_nums, np.array(parameters_used)], axis=1)
    np.savetxt("eos_metadata-"+"%06d" %(dir_index) + ".csv",
               metadata, header="eos, Gamma1, Gamma2, Gamma3, logp1", delimiter=",", comments="")
