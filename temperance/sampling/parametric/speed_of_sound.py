#Sampling EoSs from a parametrization of the speed of sound.
import numpy as np
import piecewise_polytrope as pyeos
import scipy.interpolate as interp
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy
import lalsimulation as lalsim
import lalinference as lalinf
import lal
import argparse
import matplotlib as mpl
mpl.use("agg")
from matplotlib import pyplot as plt
import astropy.constants as const
import astropy.units as u


c = const.c.cgs.value
# Characteristic refinement number (probably should be changed on a case by case basis)
N = 100
M_sun_si = const.M_sun.si.value

# PARAMETRIC EOS
# Model of equation of state prior to sample from: 


#https://arxiv.org/pdf/1812.08188.pdf


# Legacy support, should be removed at some point
p_range = (1e31, 1e37)
p_0 = 3.9e32



# We want to transition to the model slightly below
rho_0 = 2.8e14 # g/cm**3
rho_0_si = 2.8e17 # kg/m**3
rho_small = rho_0/2
c2si = (2.998e8)**2

a1_range=(.5, 1.5)
a2_range=(1.3, 5)
a3_ratio_range=(.05, 3)
a4_range=(1.5, 21)
a5_range=(.1, 1)
#a6 is special
#a6_range=

# This can be called as a script, in that case it produces a single "draw file" which contains
# a tabulated eos of (pressure, energy density, baryon density)
parser = argparse.ArgumentParser(description='Get the number of draws needed, could be expanded')
parser.add_argument("--num-draws", type=int, dest="num_draws", default=1)
parser.add_argument("--dir-index", type=int, dest="dir_index", default=0)
parser.add_argument("--prior-tag", type=str, dest="prior_tag", default="uniform")
parser.add_argument("--diagnostic", type=bool, dest="diagnostic", default=False)
sly_polytrope_model = pyeos.eos_polytrope(34.384, 3.005, 2.988, 2.851)
sly_p_1 = 1e32 #(SI, i.e. Pa)  This is an arbitrary cutoff point
sly_eps_1 = sly_polytrope_model.eval_energy_density(sly_p_1) #SI J/cm**3
sly_rho_1 = sly_polytrope_model.eval_baryon_density(sly_p_1) #SI g/cm**3
print(sly_rho_1)

# Eval f1(val) if val > thresh and f2(val) otherwise
# doesn't work with arrays
def function_switch(high_val_func, low_val_func, thresh, val):
    if val > thres:
        return high_val_func(val)
    else:
        return low_val_func(val)

# Return a function which gives the speed of sound at all points
def get_cs2c2(a1, a2, a3, a4, a5, a6):
    fun = lambda x : a1*np.exp(-1/2*(x-a2)**2/a3**2) + a6 +(1/3 - a6)/(1 + np.exp(-a5*(x-a4)))
    return fun
def tabulate_values(eps_min, eps_max, cs_2c2, p_min, rho_min):
    # Find low density eos
    # they glue to a particular eos but I think it's better
    # to glue to SLy for consistency.  
    # p_min should be decided upon ahead of time, and (eps_min, p_min)
    # should be a point on the SLy eps-p curve which is low enough
    # to be considered "known"
    eps_vals = np.geomspace(eps_min, eps_max, 500)
    def dp_and_rho(eps, p_and_rho):
        p = p_and_rho[0]
        rho = p_and_rho[1]
        # Define the pressure and density differentials
        dp = cs_2c2(eps/rho_0_si/c2si)
        drho = rho/(eps + p) # Don't think too hard about the units
        return np.array([dp, drho])
    tabulated_eos = integrate.solve_ivp(dp_and_rho, ((1-10**-10)*eps_min, (1+10**-10)*eps_max), np.array([p_min, rho_min]), t_eval=eps_vals)
    eps = tabulated_eos.t
    p = tabulated_eos.y[0,:]
    rho =tabulated_eos.y[1,:]
    # Note the order of the returns
    return p, eps, rho

# An EoS model based on the speed of sound parametrization

class eos_speed_of_sound:
    def compute_a6(self):
        # In the paper this is made to match some EFT, but here we want to match it to SLy to make
        # it consistent at low denisites with the other EoSs
        # Shoot for it?
        to_match  = self.sly_model.eval_speed_of_sound(sly_p_1)**2/c2si
        eps_match = self.sly_model.eval_energy_density(sly_p_1)
        def diff(self, a_6_guess):
            cs2_guess = self.construct_cs2_helper(a_6_guess)
            error = to_match - cs2_guess(eps_match/(c2si*rho_0_si))
            return error
        result = optimize.root_scalar(lambda a_6_guess : diff(self, a_6_guess), x0=0, x1=1, fprime = lambda x : -1 + 1/(1+np.exp(-self.a5*(eps_match/(c2si*rho_0_si) - self.a4))))
        return result.root

    def __init__(self, a1, a2, a3, a4, a5):

        self.x =np.linspace(0,16,1000) #eps/(m_N n_0)
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.sly_model= sly_polytrope_model
        self.eps_of_p = None
        self.rho_of_p = None
        # The lower point is  arbitrary, we don't want to include the endpoint
        self.p_small = np.geomspace(10**5, sly_p_1, 400, endpoint=False)
        self.eps_small = sly_polytrope_model.eval_energy_density(self.p_small)
        self.rho_small = sly_polytrope_model.eval_baryon_density(self.p_small)
        self.p_main = None
        self.eps_main = None
        self.rho_main = None
        # need to find a6 by gluing, should have a procedure to do this right
        # off the bat i think (edit : implemented, need to check if works)
        self.a6  = self.compute_a6()
        self.cs2 = self.construct_cs2_helper(self.a6)
        eps_min = sly_eps_1
        eps_max = 1e20*c2si # in SI, kg/m^3, relatively arbitrary
        self.p_main, self.eps_main, self.rho_main = tabulate_values(eps_min, eps_max, self.cs2, sly_p_1, sly_rho_1)
    def get_params(self):
        return [self.a1, self.a2, self.a3, self.a4, self.a5, self.a6]

        # Evaluate the eos in terms of epsilon(p)
    def eval_energy_density(self, p, use_low_without_eval=False):
        # if you know you're using the low-p definition ahead of time
        if use_low_without_eval:
            return sly_polytrope_model.eval_energy_density(p1)
        if self.eps_of_p == None:
            #Interpolate a function eps(p)
            self.eps_of_p = scipy.interpolate.interp1d(np.concatenate([self.p_small, self.p_main]), np.concatenate([self.eps_small, self.eps_main]), kind='cubic')
        return self.eps_of_p(p)
    # Evaluate the baryon density at a particular pressure
    def eval_baryon_density(self, p):
        #interpolate a function rho(p)
        if self.rho_of_p is None:
            self.rho_of_p = scipy.interpolate.interp1d(np.concatenate([self.p_small, self.p_main]), np.concatenate([self.rho_small, self.rho_main]), kind='cubic')
        return self.rho_of_p(p)
    # Evluate the speed of sound at a particular pressure
    def eval_speed_of_sound(self, p):
        return self.cs2(self.eval_energy_density(p)/rho_0_si/c2si)
    # thin wrapper around cs2-getter that actually computes the thing
    def construct_cs2_helper(self, a_6):
        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4
        a5 = self.a5
        a6 = a_6
        return get_cs2c2(a1, a2, a3, a4, a5, a6)



def get_eos_realization_sos(a1_range=a1_range, a2_range=a2_range, a3_ratio_range=a3_ratio_range, a4_range=a4_range, a5_range=a5_range):
    a1=np.random.uniform(*a1_range)
    a2=np.random.uniform(*a2_range)
    # This one is defined weirdly
    a3=np.random.uniform(*a3_ratio_range) * a2
    a4=np.random.uniform(*a4_range)
    a5=np.random.uniform(*a5_range)
    return eos_speed_of_sound(a1, a2, a3, a4, a5)
def create_eos_draw_file(name, tag): # Tag does nothing!
    eos_poly = get_eos_realization_sos(a1_range, a2_range, a3_ratio_range, a4_range, a5_range)

    # FIXME WORRY ABOUT CGS VS SI!!!!! (Everything is in SI till the last step  ) 
    p_small = eos_poly.p_small
    p_main = eos_poly.p_main
    eps_small=  eos_poly.eps_small
    # If the original eps range was not large enough, this may fail
    # given a very soft EoS, such EoSs are probably not realistic 
    eps_main=eos_poly.eps_main
    rho_b_main=eos_poly.rho_main
    rho_b_small = eos_poly.rho_small
    p = np.concatenate([p_small, p_main])
    eps = np.concatenate([eps_small, eps_main])
    rho_b = np.concatenate([rho_b_small, rho_b_main])
    # Many bad things could conceivably happen in the
    # interpolation, but if c_s^2 > 0 the whole time, then
    # p(eps) is invertible to eps(p) and everything should
    # be fine ( I lied :( )
    x = eps/c2si/rho_0_si
    cs2s = eos_poly.cs2(x)
    if np.all( np.logical_and(np.logical_and(cs2s > 0, cs2s < 1.05), eps>0)):
        data = np.transpose(np.stack([p/c**2*10 , eps/c**2*10, rho_b/10**3])) # *10 because Everything above is done in SI
        if args.diagnostic:
            # plot the best range for seeing what's wrong
            plt.loglog(x[400:], cs2s[400:])
        print(name)
        print("speed of sound parameters", "(", eos_poly.a1, ", ", eos_poly.a2, ", ", eos_poly.a3, ", ", eos_poly.a4, ", ", eos_poly.a5, ", ", eos_poly.a6, ")" )
        np.savetxt(name,data, header = 'pressurec2,energy_densityc2,baryon_density',
                   fmt='%.10e', delimiter=",", comments="")
        return eos_poly.get_params()
    else :
        if args.diagnostic:
            print("speed of sound is unphysical")
        return create_eos_draw_file(name, tag)

get_draw_function_from_tag = lambda x : x
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

    if args.diagnostic:
        plt.show()
        plt.savefig("diagnostic_plot.png")
    metadata = np.concatenate([eos_nums, np.array(parameters_used)], axis=1)
    np.savetxt("eos_metadata-"+"%06d" %(dir_index) + ".csv",
               metadata, header="eos, Gamma1, Gamma2, Gamma3, logp1", delimiter=",", comments="")
