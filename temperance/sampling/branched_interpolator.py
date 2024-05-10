"""
Utilities for dealing with EoSs with multiple stable branches
"""
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

def array_in(data, target_range):
    """
    return an array of shape `data.shape` which 
    stores whether each element of data is
    in the range `target_range`
    """
    (min_val, max_val) = target_range
    return np.logical_and(data>=min_val, data<=max_val)
def get_branches(eos, properties = ["R", "Lambda", "I"]):
    """
    Get each of the stable branches of an EoS, stored in a list.

    Each branch is a pd.DataFrame that represents the M-Lambda
    curve of the that stable branch, it has a column "branch_id"
    which represents a constant, unique, integer identifier for branch
    """
    if not (isinstance(eos, pd.DataFrame)):
        if isinstance(eos, np.recarray):
            eos = pd.DataFrame.from_records(eos)
        else:
            # Worst case but we still cover it. This is the definition
            # of what an EoS must be able to do
            property_and_data = {macro_property : eos[macro_property] for macro_property in properties }
            eos = pd.DataFrame({"M": eos["M"], "Lambda": eos["Lambda"]})
    def segments(split_eos):
        segment_diff = split_eos.index - np.arange(split_eos.shape[0])
        return segment_diff
    # The first point is stable if the first difference is stable
    initially_stable = (np.diff(eos["M"]) >= 0)[0]
    stable = eos.loc[[initially_stable, *(np.diff(eos["M"]) >= 0)], :].copy(deep=True)
    unstable = eos.loc[[not (initially_stable), *(np.diff(eos["M"]) < 0)], :].copy(deep=True)
    stable["branch_id"] = segments(stable)
    branch_ids = np.unique(np.array(segments(stable)))
    branches = []

    for branch_id in branch_ids:
        branches.append(stable[stable["branch_id"] == branch_id])
        if branches[-1].shape[0] == 1:
            branch = branches[-1]
            M = np.array(branch["M"])
            properties_data = {macro_property : np.array(branch[macro_property]) for macro_property in properties}
            # only one point on the branch
            # add a second, this branch will never
            # be used because it is too small,
            # but we keep it anyway
            M_new = [M[0], M[0]*(1+2*np.finfo(float).eps)]
            for macro_property in properties:
                properties_data[macro_property] = np.array([properties_data[macro_property][0], properties_data[macro_property][0]*(1+2*np.finfo(float).eps)])
            properties_data["M"] = M_new
            new_branch = properties_data
            new_branch["branch_id"] = branch_id
            branches[-1] = new_branch
    return branches

def get_macro_interpolators(branches, properties):
    """
    Get an interpolator for each stable branch as Lambda(M) is well defined, if an mvalue is passed which is not
    on the stable branch then a value of 0 is returned, which represents a black hole.
    """
    interpolators =  [interpolate.interp1d(
        np.array(branch["M"]),
        np.array([branch[macro_property] for macro_property in properties]),
        bounds_error=False, fill_value=0.0) for branch in branches]
    if len(interpolators) ==0:
        # No stable branches
        print ("This code will not work")
    return interpolators
def get_macro_of_m_evaluations(macro_interpolators, black_hole_values,  branches, m):
    """
    Get the values of 
    """
    results = []
    found_branches_for_ms=[]
    for i, branch in enumerate(branches):
        m_is_on_this_branch = array_in(
            m,
            (min(branch["M"]), max(branch["M"])))
        found_branches_for_ms.append(m_is_on_this_branch)        
        m_branch = m[np.where(m_is_on_this_branch)[0]]
        macro_of_m = macro_interpolators[i]
        properties_on_branch = macro_of_m(m_branch)
        macro_data = {key : properties_on_branch[i, :] for i, key in enumerate(black_hole_values.keys())}
        results.append({ "m": m_branch, **macro_data})
    #If no stable branch was found
    no_stable_branch = np.sum(np.array([*found_branches_for_ms]), axis=0) == 0
    # We set the deformability to 0 (black hole value)
    # and add an "overall unstable" branch to the results
    results.append(
        {"m": m[no_stable_branch],
         **{macro_property:
            black_hole_values[macro_property](m[no_stable_branch])
            for macro_property in black_hole_values.keys()}})
    
    return results
def get_macro_from_m_and_eos(m, eos, black_hole_values):
    """
    Evaluate the m values on each stable branch,
    thus get interpolated properties, such as Lambda(M), for each stable branch
    """
    properties = list(black_hole_values.keys())
    branches = get_branches(eos, properties)
    macro_interpolators = get_macro_interpolators(branches, properties)
    macro_of_m_evaluations = get_macro_of_m_evaluations(macro_interpolators,black_hole_values, branches, m)
    return macro_of_m_evaluations

def choose_macro_per_m(m, eos, black_hole_values, choice_function=None, branches=None, interpolators=None, only_lambda=True):
    """
    Get a single lambda value for each m, choosing between branches using the choice_function
    if none, sample branches randomly: do not call lambdas which are not chosen
    This will be slow, do not call it if there are not multiple stable branches.
    """
    properties = list(black_hole_values.keys())
    if branches is None:
        branches = get_branches(eos, properties)
    if len(branches) == 1:
        if only_lambda:
            interpolator = get_macro_interpolators(branches, properties=["Lambda"])[0]
            return {"m":m, "Lambda":interpolator(m)[0, :]}
        
        [stable, unstable] = get_macro_from_m_and_eos(m, eos, black_hole_values)

        if interpolators is not None:
            if only_lambda:
                derived =  {key: interpolators[0](m)[0,:] for i, key in enumerate (black_hole_values.keys())}
                return  {"m" : m , **derived}
                         
        if unstable["m"].shape[0] == 0:
            return {"m" :stable["m"], **{macro_property: stable[macro_property] for macro_property in black_hole_values.keys()}}
        elif stable["m"].shape[0] == 0:
            return {"m": unstable["m"], **{macro_property: unstable[macro_property] for macro_property in black_hole_values.keys()}}
        return {"m" :np.concatenate([stable["m"], unstable["m"]]),
                **{macro_property: np.concatenate([stable[macro_property], unstable[macro_property]]) for macro_property in black_hole_values.keys()}}
    if interpolators is None:
        interpolators = get_macro_interpolators(branches, properties)
    branch_availability= np.empty((len(m),  len(branches)))
    property_data = np.empty(shape=(len(black_hole_values), len(m)))
    for i, branch in enumerate(branches):

        branch_availability[:, i] = array_in(m,  (min(branch["M"]), max(branch["M"])))
    for j, m_val in enumerate(m):
        if np.all(branch_availability[j, :] < 1.0):
            # No stable branch, use black hole values
            for i, black_hole_function in enumerate(black_hole_values.values()):
                property_data[i,j] = black_hole_function(m_val)
        else:
            branch_to_use = np.random.choice(
                np.arange(len(branches)), p = branch_availability[j, :]/np.sum(branch_availability[j, :]))
        
            property_data[:, j] = interpolators[branch_to_use](m_val)
    return {"m": m, **{macro_property:property_data[i, :] for i, macro_property in enumerate(black_hole_values.keys())}}
    
        

if __name__ == "__main__":
    print("Example")
    # A silly example as a test case
    # TODO factor this into a real test
    rhoc = np.linspace(1.0, 3.0, 100)
    test_M = (rhoc-2)**3 + (rhoc-2) + 1
    test_Lambda = 1000*(rhoc+1)**-6
    test_R = 11.0 - 0.01 * rhoc
    test_eos = pd.DataFrame({"M" : test_M, "Lambda":test_Lambda, "R":test_R })
    plt.plot(test_R, test_M)
    plt.savefig("diagnostic_branched_mr", bbox_inches="tight")
    print("branches:", get_branches(test_eos))
    print("evaluating on [.7, 1.0, 1.3]")
    #print(get_macro_from_m_and_eos(np.array([.7, 1.0, 1.3]), test_eos, {"Lambda": lambda m : 0 }, only_lambda=True))
    print("choosing specific branch at random")
    print(choose_macro_per_m(np.array([.7, 1.0, 1.3, 1.7]), test_eos, {"Lambda": lambda m : 0 }))
