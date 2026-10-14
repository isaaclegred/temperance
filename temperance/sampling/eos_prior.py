import numpy as np
import pandas as pd
import os
import h5py

from dataclasses import dataclass

from .stability import stable_branches


# This is the eos set we use most often
DEFAULT_EOS_DATA = {"eos_dir":"/home/philippe.landry/nseos/eos/gp/mrgagn",
                    "eos_column" : "eos",
                    "eos_per_dir": 1000,
                    "eos_basename" : 'eos-draw-%(draw)06d.csv',
                    "macro_basename": 'macro-draw-%(draw)06d.csv',
                    "branches_dir" : "/home/isaac.legred/local_mrgagn_big_with_cs2c2",
                    "branches_basename" : ('macro-draw-%(draw)06d-branches.csv',
                                           "rhoc",
                                           "start_baryon_density",
                                           "end_baryon_density")}

@dataclass
class Extraction:
    """
    Get a particular EoS common quantity from a prior (i.e. extract per eos)

    If dynamic is not None, we use this column to decide extraction values 
    """
    strategy : str
    macro : bool
    independent_variable : str = None
    values : list[float] = None
    dynamic : list[str] = None
    branched : bool = True
    selection_rule :str = "random"
_known_extraction_strategies = ["extremize", "interpolation"]
_known_selection_rules = ["random", "min", "max", "nearest_neighbor"]

# Column-name formats for extracted properties, matching universality
VALUE_COLUMN_TEMPLATE = "%s(%s=%s)"      # y(x=value), for static x values
REFERENCE_COLUMN_TEMPLATE = "%s(%s@%s)"  # y(x@column), for dynamic x values


def output_columns(columns, reference, reference_values=(), reference_columns=()):
    """
    Names of the columns returned by get_property: for each dependent column,
    one per static reference value, then one per dynamic reference column.
    Mirrors universality.properties.samples.outputcolumns.
    """
    names = []
    for column in columns:
        names += [VALUE_COLUMN_TEMPLATE % (column, reference, val) for val in reference_values]
        names += [REFERENCE_COLUMN_TEMPLATE % (column, reference, col) for col in reference_columns]
    return names

@dataclass
class BaseEoSPriorSet:
    """
    This class encapsulates the current way we store samples from
    an EoS Process.  It's currently agnostic to the sampling method,
    e.g. parametric, nonparametric, whatever, but it prefers the 
    standard hierarchical directory structure we use to store these
    draws

    This base class only knows where the draws live.  Use EoSPriorSet to
    extract properties, or UniversalityEoSPriorSet for the legacy
    universality-based extraction.
    """
    eos_dir : str
    eos_column : str
    eos_per_dir : int
    macro_dir : str 
    eos_path_template : str = 'eos-draw-%(draw)06d.csv'
    macro_path_template: str = 'macro-draw-%(draw)06d.csv'
    branches_data: tuple[str] = (
        '/home/isaac.legred/local_mrgagn_big_with_cs2c2/DRAWmod1000-%(moddraw)06d/macro-draw-%(draw)06d-branches.csv',
        "rhoc",
        "start_baryon_density", "end_baryon_density")

    
    @classmethod
    def get_default(cls):
        return cls(eos_dir=DEFAULT_EOS_DATA["eos_dir"],
                           eos_column=DEFAULT_EOS_DATA["eos_column"],
                           eos_per_dir=DEFAULT_EOS_DATA["eos_per_dir"],
                           macro_dir=DEFAULT_EOS_DATA["eos_dir"]
                           
        )
    def get_eos_path(self, eos_number, subdir=None, explicit_path=None):
        if subdir == None:
            subdir = f"DRAWmod{self.eos_per_dir}-{eos_number//self.eos_per_dir:06d}"
        if explicit_path is None:
            local_path = self.eos_path_template%{"draw" : eos_number}
        else:
            local_path = explicit_path%{"draw": eos_number}
        return os.path.join(self.eos_dir, subdir,
                            local_path)
    def get_macro_path(self, eos_number, subdir=None):
        if subdir is None:
            subdir = f"DRAWmod{self.eos_per_dir}-{eos_number//self.eos_per_dir:06d}"
        return os.path.join(self.macro_dir, subdir,
                            self.macro_path_template%{"draw" : eos_number})
    def get_path_template(self, base=None, eos_dir=None, macro=False):
        """
        This is the format universality expects this path to be in
        """
        if base is None:
            if macro:
                base = self.macro_path_template
            else:
                base = self.eos_path_template
        if eos_dir is None:
            eos_dir=self.eos_dir
        return os.path.join(eos_dir,
                            f"DRAWmod{self.eos_per_dir}-%(moddraw)06d", 
                            base)

    def get_eos(self, eos_index):
        """The EoS table for draw `eos_index`, as a DataFrame."""
        return pd.read_csv(self.get_eos_path(int(eos_index)))

    def get_macro(self, eos_index):
        """The macroscopic (e.g. M, R, Lambda) table for draw `eos_index`, as a DataFrame."""
        return pd.read_csv(self.get_macro_path(int(eos_index)))


@dataclass
class UniversalityEoSPriorSet(BaseEoSPriorSet):
    """
    Legacy EoS prior set that extracts properties with universality's
    properties.samples.process2samples, reading stable branches from
    precomputed branches files (`branches_data`).  Requires universality;
    prefer EoSPriorSet.
    """
    def get_property(self, eos_indices,
                     dependent_variables=["R"],
                     extraction=Extraction("interpolation", True, "M",  [1.6, 1.8]), **kwargs):
        """
        Extract `dependent_variables` for each EoS in `eos_indices`, as
        described by `extraction`.  Extra kwargs are passed through to
        universality's process2samples (e.g. selection_rule, default_values).

        Returns a DataFrame with one row per EoS and columns named by
        `output_columns`.
        """
        if extraction.strategy == "extremize":
            # call to process2extrema
            
            return eos_indices
        elif extraction.strategy == "interpolation":
            from universality.properties import samples
            if extraction.dynamic is not None:
                dynamic_kwargs = {"dynamic_x_test" : np.array(extraction.dynamic)}
            else:
                dynamic_kwargs = {}
            if extraction.branched:
                branches_kwargs = {"branches_mapping":self.branches_data}
            else:
                branches_kwargs = {}
            print("path template is", self.get_path_template(macro=extraction.macro))
            data = samples.process2samples(
                np.array(eos_indices),
                self.get_path_template(macro=extraction.macro),
                self.eos_per_dir,
                extraction.independent_variable,
                dependent_variables,
                static_x_test=extraction.values,
                **dynamic_kwargs,
                **branches_kwargs,
                **kwargs)
            ref_columns = extraction.dynamic.columns if extraction.dynamic is not None else []
            columns = output_columns(dependent_variables,
                                     extraction.independent_variable,
                                     reference_values=[] if extraction.values is None else extraction.values,
                                     reference_columns=ref_columns)
            return pd.DataFrame(data, columns=columns)
        else:
            raise ValueError(f"Unknown extraction strategy, {extraction.strategy},"
                             f"try one of {_known_extraction_strategies}")


@dataclass
class EoSPriorDistribution:
    """
    A wrapper around a prior distribution for an EoS.  This should implement a `sample` and 
    `log_prob` method.  

    Arguments:
        eos_sampler: A class which can sample from the eos prior
        variables: A dictionary mapping variable names to the names they have in the eos_sampler
        constructor_args: arguments to pass to the eos_sampler constructor
        constructor_kwargs: keyword arguments to pass to the eos_sampler constructor

    """
    def __init__(self, 
                 eos_sampler, 
                 independent_grid=None,
                 variables={"lnrho" : "log(baryon_density)", 
                 "phi" : "phi"}, 
                 *constructor_args, **constructor_kwargs):
        self.variables = variables
        self.independent_grid = independent_grid
        self.eos_sampler = eos_sampler(*constructor_args, **constructor_kwargs) 
        self.dependent = list(variables.keys())[1]
        self.independent = list(variables.keys())[0]
    def sample(self, num_samples=1, as_array=False):
        """
        Sample from the eos prior
        Arguments:
            num_samples: The number of samples to generate
            as_array: If True, return the samples as an array, otherwise return a DataFrame
                If `as_array` is True, we only return the dependent variable, relying on the 
                user to keep track of what the independent variable is.
        Returns:
            samples: A DataFrame or array of samples
        """
        if num_samples == 1:
            dependent_vals = self.eos_sampler.sample()
            if (as_array):
                return dependent_vals
            return pd.DataFrame(
                np.array([self.independent_grid, dependent_vals]), 
                columns=self.variables.keys())
        samples = []
        for _ in range(num_samples):
            dependent_vals = self.eos_sampler.sample()
            if (as_array):
                return dependent_vals
            samples.append(pd.DataFrame(
                np.array([self.independent_grid, dependent_vals]), 
                columns=self.variables.keys()))
        return samples
    def log_prob(self, samples):
        if isinstance(samples, list[pd.DataFrame]):
            [self.eos_sampler.log_prob(sample[self.dependent]) for sample in samples]
        if isinstance(samples, pd.DataFrame):
            samples = samples[self.dependent]
            return self.eos_sampler.log_prob(samples)
        if isinstance(samples, np.ndarray):
            # Assume this is just the an array of n dependent samples
            return self.eos_sampler.log_prob(samples)

    # -----------------------------------------------------------------
    # Per-EoS property extraction, shared by every prior that can look up
    # its draws by index (EoSPriorSet, EoSPriorH5).  Subclasses
    # provide get_eos and get_macro; everything else is common.
    # -----------------------------------------------------------------
    def get_eos(self, eos_index):
        """The EoS table for draw `eos_index` (DataFrame-like)."""
        raise NotImplementedError

    def get_macro(self, eos_index):
        """The macroscopic (e.g. M, R, Lambda) table for draw `eos_index` (DataFrame-like)."""
        raise NotImplementedError

    def get_property(self, eos_indices,
                     dependent_variables=["R"],
                     extraction=Extraction("interpolation", True, "M",  [1.6, 1.8]),
                     verbose=False, **kwargs):
        """
        Extract `dependent_variables` for each EoS in `eos_indices`, as
        described by `extraction`.  Extra kwargs are passed through to
        `get_property_per_eos`; `verbose` prints progress.

        Returns a DataFrame with one row per EoS and columns named by
        `output_columns`, the same layout as UniversalityEoSPriorSet.get_property.
        """
        if extraction.strategy == "extremize":
            # call to process2extrema
            
            return eos_indices
        elif extraction.strategy != "interpolation":
            raise ValueError(f"Unknown extraction strategy, {extraction.strategy},"
                             f"try one of {_known_extraction_strategies}")
        static_x_test = [] if extraction.values is None else list(extraction.values)
        if extraction.dynamic is not None:
            dynamic_x_test = np.array(extraction.dynamic)
            ref_columns = extraction.dynamic.columns
        else:
            dynamic_x_test = None
            ref_columns = []
        if not static_x_test and dynamic_x_test is None:
            raise ValueError("extraction must specify `values` or `dynamic`")
        eos_indices = list(eos_indices)
        data = []
        for i, eos_index in enumerate(eos_indices):
            if verbose:
                print(f"\r    {i + 1}/{len(eos_indices)} eos {eos_index}", end="", flush=True)
            data.append(self.get_property_per_eos(
                eos_index, dependent_variables, extraction,
                static_x_test + ([] if dynamic_x_test is None else list(dynamic_x_test[i])),
                **kwargs))
        if verbose:
            print()
        data = np.array(data)
        columns = output_columns(dependent_variables,
                                 extraction.independent_variable,
                                 reference_values=static_x_test,
                                 reference_columns=ref_columns)
        return pd.DataFrame(data.reshape(-1, len(columns)), columns=columns)

    def get_property_per_eos(self, eos_index, dependent_variables, extraction, x_test,
                             selection_rule=None, default_values=None, rng=None,
                             mass_column="M", radius_column="R"):
        """
        Read `dependent_variables` off one EoS at the values `x_test` of
        `extraction.independent_variable`.

        eos_index: the draw to use; its table comes from get_macro if
          extraction.macro, else get_eos
        dependent_variables: list of column names to extract
        extraction: the Extraction being performed.  If extraction.branched
          and extraction.macro, the table is split into stable branches on
          the fly (see find_stable_branches); otherwise the whole table is
          one branch.
        x_test: list of x values for this EoS (static values, then this
          EoS's dynamic values)
        selection_rule: how to choose when an x value lies on several
          branches; defaults to extraction.selection_rule.  One of
            "random": a branch chosen uniformly, independently for each x
            "min"/"max": smallest/largest value (one dependent variable only)
            "nearest_neighbor": the tabulated row whose x is closest, over
              all branches (no interpolation; default_values only if there
              are no stable points at all)
        default_values: one value per dependent variable, used where an x
          value is on no branch; defaults to NaN
        rng: numpy Generator for the "random" rule
        mass_column, radius_column: the columns used to find the stable
          branches (turning-point rule, see sampling.stability); without a
          radius column every run of non-decreasing mass counts as stable

        Returns a 1D array of length len(dependent_variables) * len(x_test),
        ordered as output_columns: for each dependent variable, each x.
        """
        table = self.get_macro(eos_index) if extraction.macro else self.get_eos(eos_index)
        table = pd.DataFrame(np.asarray(table)) if not isinstance(table, pd.DataFrame) else table
        selection_rule = extraction.selection_rule if selection_rule is None else selection_rule
        if selection_rule not in _known_selection_rules:
            raise ValueError(f"selection_rule={selection_rule!r} not understood, "
                             f"try one of {_known_selection_rules}")
        if selection_rule in ("min", "max") and len(dependent_variables) != 1:
            raise ValueError(f'selection_rule="{selection_rule}" needs exactly one dependent variable')
        if default_values is None:
            default_values = [np.nan] * len(dependent_variables)
        if len(default_values) != len(dependent_variables):
            raise ValueError("need exactly one default value per dependent variable")

        x = np.asarray(table[extraction.independent_variable], dtype=float)
        y = np.asarray(table[list(dependent_variables)], dtype=float)  # (Nrows, Ny)
        if extraction.branched and extraction.macro:
            radius = (np.asarray(table[radius_column], dtype=float)
                      if radius_column is not None and radius_column in table else None)
            branches = find_stable_branches(np.asarray(table[mass_column], dtype=float), radius)
        else:
            branches = [np.ones(len(x), dtype=bool)]

        if selection_rule == "nearest_neighbor":
            rows = np.flatnonzero(np.logical_or.reduce(branches)) if branches else []
            if len(rows) == 0:
                return np.repeat(default_values, len(x_test)).astype(float)
            nearest = rows[np.argmin(np.abs(x[rows][None, :] - np.asarray(x_test)[:, None]), axis=1)]
            return y[nearest].T.flatten()

        # candidates[i] collects one (Ny,) value per branch covering x_test[i]
        candidates = [[] for _ in x_test]
        for branch in branches:
            values = _interpolate_on_branch(x[branch], y[branch], x_test)
            for i, value in enumerate(values):
                if value is not None:
                    candidates[i].append(value)

        rng = np.random.default_rng() if rng is None else rng
        chosen = np.empty((len(x_test), len(dependent_variables)))
        for i, values in enumerate(candidates):
            if not values:
                chosen[i] = default_values
            elif selection_rule == "random":
                chosen[i] = values[rng.integers(len(values))]
            elif selection_rule == "min":
                chosen[i] = np.min(values, axis=0)
            else:
                chosen[i] = np.max(values, axis=0)
        return chosen.T.flatten()


    def get_quantiles(self, eos_indices, xcolumn, ycolumn, x_test, quantiles,
                      weights=None, macro=False, branched=None,
                      selection_rule="random", default_value=None, rng=None,
                      quantile_type="sym"):
        """
        Weighted quantiles of `ycolumn` across the EoSs `eos_indices`, at
        each value in `x_test` of `xcolumn`.  Mirrors universality's
        properties.samples.process2quantiles.

        weights: (N,) weight of each EoS, default uniform
        macro: read the macroscopic tables (get_macro) rather than get_eos
        branched: select stable branches on the fly; defaults to `macro`
        selection_rule, rng: as in get_property_per_eos
        default_value: value used where an x is on no branch of an EoS; the
          default (None) excludes that EoS from the quantiles at that x, and
          x values no EoS reaches get NaN quantiles

        Returns (quantiles, median): arrays of shape (len(quantiles),
        len(x_test)) and (len(x_test),).
        """
        if quantile_type != "sym":
            raise NotImplementedError(f'quantile_type="{quantile_type}" is not supported, only "sym"')
        from temperance.core.weights import quantile
        x_test = list(x_test)
        extraction = Extraction("interpolation", macro, xcolumn, x_test,
                                branched=macro if branched is None else branched,
                                selection_rule=selection_rule)
        samples = self.get_property(
            eos_indices, [ycolumn], extraction, rng=rng,
            default_values=[np.nan if default_value is None else default_value]).to_numpy()
        weights = (np.full(len(samples), 1.0 / len(samples)) if weights is None
                   else np.asarray(weights, dtype=float))

        qs = np.empty((len(quantiles), len(x_test)))
        median = np.empty(len(x_test))
        for i in range(len(x_test)):
            y = samples[:, i]
            keep = ~np.isnan(y)
            if not keep.any():  # no EoS reaches this x
                qs[:, i] = median[i] = np.nan
                continue
            qs[:, i] = quantile(y[keep], quantiles, weights=weights[keep])
            median[i] = quantile(y[keep], [0.5], weights=weights[keep])[0]
        return qs, median

def find_stable_branches(mass, radius=None, **kwargs):
    """
    Boolean masks (in table order) for each stable branch of a sequence of
    stars ordered by central density.  With ``radius`` this is the
    turning-point rule, which rejects the high-density spiral; without it
    every run of non-decreasing mass counts.  See
    temperance.sampling.stability.stable_branches for the other options.
    """
    return stable_branches(mass, radius=radius, **kwargs)


def _interpolate_on_branch(x, y, x_test):
    """
    y (Nrows, Ny) linearly interpolated at each x_test along one branch, or
    None where x_test is outside the branch's range.  x must be monotonic
    on the branch unless every requested value is an exact match.
    """
    out = [None] * len(x_test)
    if len(x) == 0:
        return out
    order = np.argsort(x, kind="stable")
    monotonic = np.all(np.diff(x) >= 0) or np.all(np.diff(x) <= 0)
    for i, X in enumerate(x_test):
        if not (x[order[0]] <= X <= x[order[-1]]):
            continue
        exact = np.flatnonzero(x == X)
        if len(exact) == 1:
            out[i] = y[exact[0]]
        elif monotonic:
            out[i] = np.array([np.interp(X, x[order], y[order, j]) for j in range(y.shape[1])])
        else:
            raise RuntimeError(
                f"{X} is not an exact tabulated value and x is not monotonic on this "
                "branch, so it cannot be interpolated; use a branched extraction or "
                'selection_rule="nearest_neighbor"')
    return out


@dataclass
class EoSPriorSet(BaseEoSPriorSet, EoSPriorDistribution):
    """
    EoS prior set stored in the standard draw directories.  Properties are
    extracted with the per-EoS extraction shared with EoSPriorH5
    (EoSPriorDistribution), selecting stable branches on the fly; no
    universality needed.  BaseEoSPriorSet supplies get_eos and get_macro.
    """


# Name used for EoSPriorSet before it became the default
NativeEoSPriorSet = EoSPriorSet


class EoSPriorH5(EoSPriorDistribution):
    """
    EoS draws stored in an HDF5 file, one dataset per draw in an EoS group
    and a macro (stellar properties) group, e.g. the Zenodo 17352004 sets::

        EoSPriorH5(path, "EoS", "Macro", index_to_key=lambda i: f"{i:05d}")

    get_eos / get_macro return DataFrames.  The file is opened read-only;
    use the object as a context manager, or call close(), to release it.

    Parameters
    ----------
    h5_path : path
    eos_subgroup_path, macro_subgroup_path : str
        Groups holding the EoS and macro tables.
    index_to_key : callable
        Maps an EoS index to its dataset name.
    key_to_index : callable
        The inverse, used by indices(); default int.
    """

    def __init__(self, h5_path, eos_subgroup_path="eos", macro_subgroup_path="ns",
                 index_to_key=lambda x : str(x), key_to_index=int):
        self.h5_path = str(h5_path)
        self.h5_file = h5py.File(self.h5_path, mode="r")
        self.eos_subgroup = self.h5_file[eos_subgroup_path]
        self.macro_subgroup = self.h5_file[macro_subgroup_path]
        self.index_to_key = index_to_key
        self.key_to_index = key_to_index

    @staticmethod
    def _as_frame(dataset):
        return pd.DataFrame(dataset[()])

    def get_eos(self, eos_index):
        """The EoS table of draw eos_index, as a DataFrame."""
        return self._as_frame(self.eos_subgroup[self.index_to_key(eos_index)])

    def get_macro(self, eos_index):
        """The macro table of draw eos_index, as a DataFrame."""
        return self._as_frame(self.macro_subgroup[self.index_to_key(eos_index)])

    def keys(self):
        """Dataset names of the draws, sorted."""
        return sorted(self.eos_subgroup.keys())

    def indices(self):
        """EoS indices of the draws, in the order of keys()."""
        return [self.key_to_index(key) for key in self.keys()]

    def __len__(self):
        return len(self.eos_subgroup)

    def close(self):
        self.h5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __getitem__(self, name):
        match name:
            case "ns" : return self.macro_subgroup
            case "eos" : return self.eos_subgroup


class EoSPriorFamily:
    """
    A family of EoS priors indexed by a hyperparameter, e.g. the Zenodo
    17352004 sets, one file per transition pressure p_t::

        family = EoSPriorFamily.from_glob("data/*_ns_data.h5", r"maxpc2-([0-9.e+]+)_",
                                          hyperparameter="p_t", eos_subgroup_path="EoS",
                                          macro_subgroup_path="Macro",
                                          index_to_key=lambda i: f"{i:05d}")
        for p_t, prior in family: ...

    Parameters
    ----------
    members : dict hyperparameter value -> EoSPriorDistribution
    hyperparameter : str
        Name of the hyperparameter (for labels and tables).
    paths : dict, optional
        value -> file path, used by posteriors() to read weights.
    """

    def __init__(self, members, hyperparameter="hyperparameter", paths=None):
        self.members = dict(sorted(members.items()))
        self.hyperparameter = hyperparameter
        self.paths = dict(paths or {})

    @classmethod
    def from_glob(cls, pattern, value_regex, hyperparameter="hyperparameter", value_type=float,
                  **prior_kwargs):
        """
        One EoSPriorH5 per file matching ``pattern``; the hyperparameter value
        is group 1 of ``value_regex`` searched in the file name.  Files the
        regex does not match are skipped.  prior_kwargs go to EoSPriorH5.
        """
        import glob
        import re
        members, paths = {}, {}
        for path in sorted(glob.glob(str(pattern))):
            match = re.search(value_regex, os.path.basename(path))
            if match:
                value = value_type(match.group(1))
                members[value] = EoSPriorH5(path, **prior_kwargs)
                paths[value] = path
        return cls(members, hyperparameter=hyperparameter, paths=paths)

    def values(self):
        return list(self.members)

    def __iter__(self):
        return iter(self.members.items())

    def __getitem__(self, value):
        return self.members[value]

    def __len__(self):
        return len(self.members)

    def posteriors(self, **from_h5_kwargs):
        """
        {value: EoSPosterior} with the weights of each member's file (see
        EoSPosterior.from_h5; kwargs are passed to it).
        """
        from temperance.core.result import EoSPosterior
        if set(self.paths) != set(self.members):
            raise ValueError("posteriors() needs a file path for every member")
        return {value: EoSPosterior.from_h5(self.paths[value], **from_h5_kwargs) for value in self.members}

    def log_evidences(self, posteriors=None, weight_columns=None):
        """
        {value: ln Z}, Z = EoSPosterior.estimate_evidence: the mean
        likelihood weight over each member's (equally weighted) prior draws.
        """
        posteriors = posteriors or self.posteriors()
        out = {}
        for value, posterior in posteriors.items():
            columns = weight_columns if weight_columns is not None else posterior.weight_columns_available
            evidence, _ = posterior.estimate_evidence(columns)
            out[value] = float(np.log(evidence))
        return out

    def close(self):
        for member in self.members.values():
            if hasattr(member, "close"):
                member.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
