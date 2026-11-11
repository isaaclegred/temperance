"""
Monte Carlo samples of the observables of neutron stars drawn from an EoS
distribution, for evaluating single-event likelihoods on.

For each EoS, masses are drawn from a mass prior and the requested
observables (e.g. R, Lambda) are read off that EoS's stable branches.
Without an EoS distribution, only masses are drawn.

Neutron stars or compact objects (the ``objects`` argument):
  - "neutron_stars" (default): every mass is below the EoS's maximum mass
    (the ``m_max_column`` of the EoS distribution, e.g. the TOV maximum
    mass); the mass prior is truncated there.
  - "compact_objects": the mass prior alone sets the range of masses, which
    the EoS does not limit; a mass with no stable neutron star is a black
    hole and gets the black-hole value of each observable (R: the
    Schwarzschild radius, Lambda: 0).

sample_single_neutron_star    one star per sample (e.g. a NICER M-R likelihood)
sample_neutron_star_binary    two stars per sample (e.g. a GW likelihood)
uniform_in_components_mass_prior
                              a bilby prior on binary masses, uniform in the
                              component masses within chirp-mass and
                              mass-ratio bounds
generate_mr_samples           sample_single_neutron_star with observables=["R"]
generate_gw_samples           sample_neutron_star_binary with observables=["Lambda"]
                              and both masses drawn independently from a 1D prior

Mass frames (read this before comparing to gravitational-wave results)
----------------------------------------------------------------------
bilby names masses by frame: ``mass_1``, ``mass_2``, ``chirp_mass``,
``total_mass`` are *detector-frame* (redshifted, (1 + z) m) and
``mass_1_source``, ``chirp_mass_source``, ... are *source-frame*.

temperance's EoS calculations involve no redshift: a neutron star's mass is
its physical, source-frame mass.  So in this module ``mass_1``, ``mass_2``,
``chirp_mass`` and ``mass_ratio`` always mean **source-frame** quantities,
even though bilby would read those names as detector-frame.  This is only a
naming choice: bilby's mass priors and conversions (e.g.
UniformInComponentsChirpMass, generate_mass_parameters) are pure arithmetic
between mass parameters, so they apply unchanged to source-frame masses under
the plain names.  The mass ratio, q = mass_2 / mass_1 <= 1, is the same in
both frames.

Consequences:
  - A bilby prior over ``chirp_mass`` / ``mass_ratio`` (or ``mass_1`` /
    ``mass_2``) passed here is taken to describe source-frame masses.
  - A prior written with bilby's ``*_source`` names is also accepted (those
    names are unambiguous); its masses are returned as ``mass_1``, ``mass_2``.
  - When comparing with GW posterior samples, use their source-frame columns
    (``mass_1_source``, ``chirp_mass_source``, ...), or convert their
    detector-frame masses with the redshift; do **not** match our
    ``mass_1`` to their ``mass_1``.
"""

import copy

import numpy as np
import pandas as pd

import temperance.sampling.branched_interpolator as b_interp
from temperance.utilities.units import KM_PER_MSUN

# bilby (a dependency) is imported where used, as it is slow to import

#: Value of each observable for a mass with no stable neutron star (a black hole):
#: R is the Schwarzschild radius in km, Lambda is 0.
DEFAULT_BLACK_HOLE_VALUES = {
    "R": lambda m: 2 * m * KM_PER_MSUN,
    "Lambda": lambda m: np.zeros_like(np.asarray(m, dtype=float)),
}


def _black_hole_values(observables, black_hole_values):
    # user-supplied values override the defaults, observable by observable
    values = {**DEFAULT_BLACK_HOLE_VALUES, **(black_hole_values or {})}
    missing = [name for name in observables if name not in values]
    if missing:
        raise ValueError(f"no black-hole value for observable(s) {missing}; "
                         "pass black_hole_values={name: function of mass}")
    # keep only the requested observables, in the requested order:
    # choose_macro_per_m interpolates exactly the keys of this dict
    return {name: values[name] for name in observables}


_OBJECTS = ("neutron_stars", "compact_objects")


def _check_objects(objects):
    if objects not in _OBJECTS:
        raise ValueError(f"objects must be one of {_OBJECTS}, got {objects!r}")


def _samples_per_eos(eos_distribution, eos_prior_set, num_samples_per_eos, max_total_samples,
                     external_get_macro):
    """
    How many samples to draw for each EoS, or None to draw masses only.

    - no EoS arguments: masses only, max_total_samples of them (required);
    - num_samples_per_eos only: that many per EoS;
    - max_total_samples only: max_total_samples // (number of EoSs) per EoS;
    - both: num_samples_per_eos per EoS, if that is not more than
      max_total_samples in total.
    """
    # Masses only: the EoS arguments must all be absent, and the total is required
    if eos_distribution is None:
        if eos_prior_set is not None or num_samples_per_eos is not None or external_get_macro is not None:
            raise ValueError("eos_prior_set, num_samples_per_eos and external_get_macro "
                             "need an eos_distribution")
        if max_total_samples is None:
            raise ValueError("without an eos_distribution, pass max_total_samples "
                             "(the number of masses to draw)")
        return None
    # With EoSs: each EoS's macro table must be reachable
    if eos_prior_set is None and external_get_macro is None:
        raise ValueError("an eos_distribution needs an eos_prior_set (or external_get_macro) "
                         "to look up each EoS's macro table")
    n_eos = len(eos_distribution.samples)
    # Only a total: share it equally (integer division, so up to n_eos - 1 fewer)
    if num_samples_per_eos is None:
        if max_total_samples is None:
            raise ValueError("pass num_samples_per_eos, max_total_samples, or both")
        per_eos = max_total_samples // n_eos
        if per_eos == 0:
            raise ValueError(f"max_total_samples={max_total_samples} is fewer than one sample "
                             f"for each of the {n_eos} EoSs")
        print(f"Drawing {per_eos * n_eos} of at most {max_total_samples} samples "
              f"({per_eos} for each of {n_eos} EoSs)")
        return per_eos
    # A per-EoS count, optionally capped by the total
    if max_total_samples is not None and num_samples_per_eos * n_eos > max_total_samples:
        raise ValueError(f"num_samples_per_eos={num_samples_per_eos} for {n_eos} EoSs is "
                         f"{num_samples_per_eos * n_eos} samples, more than "
                         f"max_total_samples={max_total_samples}")
    return num_samples_per_eos


def sample_single_neutron_star(
    eos_distribution=None,
    eos_prior_set=None,
    mass_prior_class=None,
    num_samples_per_eos=None,
    observables=("R",),
    mass_prior_kwargs={"m_min": 1.0},
    black_hole_values=None,
    m_max_column="Mmax",
    external_get_macro=None,
    max_total_samples=None,
    objects="neutron_stars",
):
    """
    Draw neutron stars (or compact objects) from each EoS in an EoS
    distribution: masses from a mass prior, and the requested observables at
    those masses.  Without an EoS distribution, draw masses only.

    How many samples (see also the Returns section):
      - eos_distribution, eos_prior_set and num_samples_per_eos all None:
        max_total_samples masses from the mass prior, with no observables;
      - num_samples_per_eos: that many samples from each EoS;
      - max_total_samples without num_samples_per_eos:
        max_total_samples // (number of EoSs) from each EoS, so possibly
        fewer than max_total_samples in all (the number drawn is printed);
      - both: num_samples_per_eos from each EoS; an error if that is more
        than max_total_samples in all.

    Arguments:
      eos_distribution: An EoSPosterior (or prior) whose samples list the
        EoSs, with each EoS's maximum mass in ``m_max_column``
      eos_prior_set: The EoS prior (e.g. EoSPriorSet or EoSPriorH5) the
        eos_distribution was drawn from; its get_macro gives each EoS's table
      mass_prior_class: A constructor for the mass prior (required), called
        with mass_prior_kwargs and implementing ``sample(n)``.  For neutron
        stars it is also given m_max = the EoS's maximum mass (or the
        smaller of that and an m_max in mass_prior_kwargs); for compact
        objects, exactly mass_prior_kwargs.
      num_samples_per_eos: the number of monte carlo samples used to resolve
        each eos, for most applications should be large enough that
        num_samples_per_eos * (width of mass measurement) / (maximum mass - 1) >> 1
      observables: columns of the macro tables to read off at each mass,
        e.g. ["R"] or ["R", "Lambda"]
      mass_prior_kwargs: additional kwargs to pass to the mass_prior_class
        constructor
      black_hole_values: {observable: function of mass} for masses with no
        stable star, overriding DEFAULT_BLACK_HOLE_VALUES (R: Schwarzschild
        radius in km, Lambda: 0); needed for any other observable
      m_max_column: the column of eos_distribution.samples with each EoS's
        maximum mass
      external_get_macro: optional function eos index -> macro table, used
        instead of eos_prior_set.get_macro
      max_total_samples: the most samples to draw in all
      objects: "neutron_stars" (masses below each EoS's maximum mass) or
        "compact_objects" (the mass prior alone sets the range; masses with
        no stable star are black holes); see the module docstring
    Returns:
      With an EoS distribution, a DataFrame with columns eos, Mmax, M and the
      observables, with the same number of rows per EoS, in the order of
      eos_distribution.samples.  Without one, a DataFrame with the single
      column M.
    """
    if mass_prior_class is None:
        raise TypeError("sample_single_neutron_star needs a mass_prior_class")
    _check_objects(objects)
    per_eos = _samples_per_eos(eos_distribution, eos_prior_set, num_samples_per_eos,
                               max_total_samples, external_get_macro)
    if per_eos is None:
        # No EoSs: just masses from the prior as configured.  There is no
        # maximum mass to truncate at, so `objects` has no effect here.
        # (deepcopy: the default mass_prior_kwargs dict is shared between calls)
        mass_prior = mass_prior_class(**copy.deepcopy(mass_prior_kwargs))
        return pd.DataFrame({"M": np.asarray(mass_prior.sample(max_total_samples), dtype=float)})
    assert eos_distribution is not None   # (checked by _samples_per_eos)

    black_hole_values = _black_hole_values(list(observables), black_hole_values)

    def get_macro(eos_index):
        if external_get_macro is not None:
            return external_get_macro(eos_index)
        assert eos_prior_set is not None   # (checked by _samples_per_eos)
        return eos_prior_set.get_macro(int(eos_index))

    eoss_to_use = eos_distribution.samples[[eos_distribution.eos_column, m_max_column]]
    # One block of per_eos rows per EoS; columns are eos, maximum mass, mass,
    # observables.  (A float array, so the eos index comes back as a float,
    # as it always has.)
    output_data = np.empty((len(eoss_to_use) * per_eos, 3 + len(observables)))

    for i, eos_index in enumerate(eoss_to_use[eos_distribution.eos_column]):
        m_max = eoss_to_use[m_max_column].iloc[i]
        # A fresh copy per EoS, so adding m_max never leaks into the caller's
        # dict or into the next EoS
        mass_prior_kwargs_local = copy.deepcopy(mass_prior_kwargs)
        if objects == "neutron_stars":
            # Truncate at this EoS's maximum mass, or at the prior's own upper
            # bound if that is lower; the TOV limit always wins
            mass_prior_kwargs_local["m_max"] = min(mass_prior_kwargs_local.get("m_max", np.inf), m_max)
        mass_prior = mass_prior_class(**mass_prior_kwargs_local)
        eos_table = get_macro(int(eos_index))
        mass_samples = mass_prior.sample(per_eos)
        # Read each observable off the EoS's stable branches at each mass.  A
        # mass on several (twin) branches gets one of them at random (numpy's
        # global RNG); a mass on none gets the black-hole value.  The masses
        # come back in the order they were given.
        samples = b_interp.choose_macro_per_m(
            mass_samples,
            eos_table,
            black_hole_values=black_hole_values,
            only_lambda=False,
        )
        rows = slice(i * per_eos, (i + 1) * per_eos)
        output_data[rows, 0] = eos_index
        output_data[rows, 1] = m_max
        output_data[rows, 2] = samples["m"]
        for j, name in enumerate(observables):
            output_data[rows, 3 + j] = samples[name]
    return pd.DataFrame(
        data=output_data, columns=[eos_distribution.eos_column, "Mmax", "M", *observables]
    )


#: Column-name stems for binary observables, following bilby's
#: posterior-sample names where they exist (lambda_1, lambda_2).
BINARY_OBSERVABLE_NAMES = {"Lambda": "lambda"}

# mass parameters bilby names by frame (mass_ratio and symmetric_mass_ratio are frame-free)
_FRAME_DEPENDENT_MASSES = ("mass_1", "mass_2", "chirp_mass", "total_mass")


def uniform_in_components_mass_prior(chirp_mass, mass_ratio, mass_1=None, mass_2=None):
    """
    A bilby prior on binary masses that is uniform in the component masses
    (mass_1, mass_2), restricted to bounds on the chirp mass and mass ratio
    and, optionally, on each component mass.

    Built from bilby's UniformInComponentsChirpMass and
    UniformInComponentsMassRatio (which together are uniform in the
    components), plus Constraint priors on mass_1 / mass_2 enforced by
    bilby's rejection sampling through generate_mass_parameters.

    All masses are **source-frame**, under bilby's plain names (see the
    module docstring).

    Arguments:
      chirp_mass: (minimum, maximum) source-frame chirp mass
      mass_ratio: (minimum, maximum) of q = mass_2 / mass_1, within (0, 1]
      mass_1, mass_2: optional (minimum, maximum) of each component mass,
        e.g. mass_1=(1.0, m_max) to keep both stars below an EoS's maximum
        mass (mass_1 is the heavier)
    Returns:
      bilby.core.prior.PriorDict
    """
    from bilby.core.prior import Constraint, PriorDict
    from bilby.gw.conversion import generate_mass_parameters
    from bilby.gw.prior import UniformInComponentsChirpMass, UniformInComponentsMassRatio

    # bilby evaluates Constraint priors on conversion_function(sample), so
    # generate_mass_parameters is what makes mass_1 / mass_2 available to the
    # component-mass constraints below (samples failing them are redrawn)
    priors = PriorDict(conversion_function=generate_mass_parameters)
    # Sampled in (chirp_mass, mass_ratio); these two priors carry the Jacobian
    # that makes the joint distribution uniform in (mass_1, mass_2)
    priors["chirp_mass"] = UniformInComponentsChirpMass(*chirp_mass, name="chirp_mass")
    priors["mass_ratio"] = UniformInComponentsMassRatio(*mass_ratio, name="mass_ratio")
    for name, bounds in (("mass_1", mass_1), ("mass_2", mass_2)):
        if bounds is not None:
            priors[name] = Constraint(*bounds, name=name)
    return priors


def _component_masses(samples):
    """
    Source-frame (mass_1, mass_2), with mass_1 >= mass_2, from samples of a
    bilby prior written either with plain mass names (taken as source frame,
    see the module docstring) or with bilby's *_source names.
    """
    from bilby.gw.conversion import generate_mass_parameters

    # Which naming the prior uses.  Plain names are source frame by this
    # module's convention; *_source names are source frame by bilby's.  Both
    # at once would mean two different things to bilby, so refuse that.
    keys = set(samples)
    plain = keys & set(_FRAME_DEPENDENT_MASSES)
    source = keys & {f"{name}_source" for name in _FRAME_DEPENDENT_MASSES}
    if plain and source:
        raise ValueError(f"the mass prior mixes plain mass names {sorted(plain)} with source-frame names "
                         f"{sorted(source)}; here plain names already mean source frame, so use one or the other")
    # generate_mass_parameters fills in the component masses from whichever
    # pair of mass parameters the prior samples (e.g. chirp_mass and
    # mass_ratio); source=True makes it read and write the *_source names
    converted = generate_mass_parameters(dict(samples), source=bool(source))
    suffix = "_source" if source else ""
    mass_1 = np.asarray(converted[f"mass_1{suffix}"], dtype=float)
    mass_2 = np.asarray(converted[f"mass_2{suffix}"], dtype=float)
    # Relabel so that mass_1 is the heavier (bilby's q = mass_2 / mass_1 <= 1)
    return np.maximum(mass_1, mass_2), np.minimum(mass_1, mass_2)


def _sample_component_masses(prior, n, m_max=None, max_draws=1000):
    """
    n source-frame (mass_1, mass_2) pairs from a bilby prior; with m_max,
    from the prior conditioned on both masses being <= m_max (by
    rejection: draws of n are repeated until n pass, up to max_draws times).
    """
    if m_max is None:
        return _component_masses(prior.sample(n))
    # Rejection sampling gives exactly the prior conditioned on the cut, for
    # any prior and parameterisation (no need to know how to add a constraint
    # to it); the cost is the fraction of the prior above m_max.
    kept_1, kept_2, n_kept = [], [], 0
    for _ in range(max_draws):
        mass_1, mass_2 = _component_masses(prior.sample(n))
        below = mass_1 <= m_max            # mass_1 is the heavier, so both are below
        kept_1.append(mass_1[below])
        kept_2.append(mass_2[below])
        n_kept += int(below.sum())
        if n_kept >= n:
            # keep the first n accepted, in the order they were drawn
            return np.concatenate(kept_1)[:n], np.concatenate(kept_2)[:n]
    raise ValueError(f"only {n_kept} of {n * max_draws} binaries drawn from the mass prior had both masses "
                     f"below the maximum mass {m_max:.4g}; the prior has (almost) no support there "
                     "(use objects='compact_objects', or a prior that reaches below the maximum mass)")


def _binary_columns(observables):
    # Grouped by observable, then star: lambda_1, lambda_2, R_1, R_2, ...
    # (sample_neutron_star_binary writes observable j of star k to column 4 + 2 j + k)
    names = []
    for observable in observables:
        stem = BINARY_OBSERVABLE_NAMES.get(observable, observable)
        names += [f"{stem}_1", f"{stem}_2"]
    return names


def sample_neutron_star_binary(
    eos_distribution=None,
    eos_prior_set=None,
    mass_prior=None,
    num_samples_per_eos=None,
    observables=("Lambda",),
    black_hole_values=None,
    m_max_column="Mmax",
    external_get_macro=None,
    max_total_samples=None,
    objects="neutron_stars",
):
    """
    Draw binaries of two neutron stars (or compact objects) from each EoS in
    an EoS distribution: component masses from a bilby prior on binary
    masses, and the requested observables of each star at its mass, both from
    the same EoS.  Without an EoS distribution, draw the component masses
    only.

    **Mass frame:** mass_1 and mass_2 here are source-frame masses, although
    bilby would read those names as detector-frame; see the module docstring.

    How many samples is decided as in sample_single_neutron_star (from
    eos_distribution, num_samples_per_eos and max_total_samples).

    Arguments:
      eos_distribution, eos_prior_set, num_samples_per_eos, m_max_column,
      external_get_macro, max_total_samples: as for sample_single_neutron_star
      mass_prior: (required) a bilby PriorDict over binary masses, e.g. from
        uniform_in_components_mass_prior or a bilby BNSPriorDict, written
        with plain mass names (chirp_mass and mass_ratio, or mass_1 and
        mass_2, taken as source frame) or with *_source names; or a function
        m_max -> PriorDict, called for each EoS with its maximum mass (e.g.
        for a population that depends on it).  Its samples come from
        bilby's random number generator (seed with
        bilby.core.utils.random.seed).
      observables: macro-table columns to read off for each star, e.g.
        ["Lambda"] or ["Lambda", "R"]
      black_hole_values: as for sample_single_neutron_star: masses with no
        stable star get these values (Lambda: 0)
      objects: "neutron_stars": the mass prior conditioned on both masses
        being below each EoS's maximum mass (by rejection sampling, which
        works for any prior); "compact_objects": the mass prior as it is,
        with masses above the maximum mass being black holes.  See the
        module docstring.
    Returns:
      With an EoS distribution, a DataFrame with columns eos, Mmax, mass_1,
      mass_2 and, for each observable, its value for each star: lambda_1,
      lambda_2 for Lambda (bilby's names), <observable>_1, <observable>_2
      otherwise; the same number of rows per EoS.  mass_1 >= mass_2
      (components are relabelled where the prior gives mass_2 > mass_1).
      Without an EoS distribution, a DataFrame with columns mass_1, mass_2.
    """
    if mass_prior is None:
        raise TypeError("sample_neutron_star_binary needs a mass_prior")
    _check_objects(objects)
    # A function of m_max (to build a prior per EoS) rather than a PriorDict:
    # PriorDicts have .sample, plain functions do not
    prior_for = mass_prior if callable(mass_prior) and not hasattr(mass_prior, "sample") else None
    per_eos = _samples_per_eos(eos_distribution, eos_prior_set, num_samples_per_eos,
                               max_total_samples, external_get_macro)
    if per_eos is None:
        # No EoSs: just component masses.  There is no maximum mass, so a
        # function of it cannot be used, and `objects` has no effect here.
        if prior_for is not None:
            raise ValueError("without an eos_distribution there is no maximum mass to pass to a "
                             "mass_prior function; pass a PriorDict")
        mass_1, mass_2 = _component_masses(mass_prior.sample(max_total_samples))
        return pd.DataFrame({"mass_1": mass_1, "mass_2": mass_2})
    assert eos_distribution is not None   # (checked by _samples_per_eos)

    black_hole_values = _black_hole_values(list(observables), black_hole_values)

    def get_macro(eos_index):
        if external_get_macro is not None:
            return external_get_macro(eos_index)
        assert eos_prior_set is not None   # (checked by _samples_per_eos)
        return eos_prior_set.get_macro(int(eos_index))

    eoss_to_use = eos_distribution.samples[[eos_distribution.eos_column, m_max_column]]
    # One block of per_eos rows per EoS
    columns = [eos_distribution.eos_column, "Mmax", "mass_1", "mass_2", *_binary_columns(observables)]
    output_data = np.empty((len(eoss_to_use) * per_eos, len(columns)))

    for i, eos_index in enumerate(eoss_to_use[eos_distribution.eos_column]):
        m_max = eoss_to_use[m_max_column].iloc[i]
        prior = prior_for(m_max) if prior_for is not None else mass_prior
        # Neutron stars: condition the prior on both masses <= this EoS's
        # maximum mass.  Compact objects: no cut, heavy masses become black holes.
        mass_1, mass_2 = _sample_component_masses(
            prior, per_eos, m_max if objects == "neutron_stars" else None)
        eos_table = get_macro(int(eos_index))
        rows = slice(i * per_eos, (i + 1) * per_eos)
        output_data[rows, 0] = eos_index
        output_data[rows, 1] = m_max
        output_data[rows, 2] = mass_1
        output_data[rows, 3] = mass_2
        # Each star separately, on the same EoS: observables on the stable
        # branches, a random branch for twin-star masses, black-hole values
        # where there is no stable star (see sample_single_neutron_star)
        for k, masses in enumerate((mass_1, mass_2)):
            star = b_interp.choose_macro_per_m(masses, eos_table, black_hole_values=black_hole_values,
                                               only_lambda=False)
            for j, name in enumerate(observables):
                output_data[rows, 4 + 2 * j + k] = star[name]
    return pd.DataFrame(data=output_data, columns=columns)


def generate_mr_samples(
    eos_posterior,
    eos_prior_set,
    mass_prior_class,
    num_samples_per_eos,
    mass_prior_kwargs={"m_min": 1.0},
    external_get_macro=None
):
    """
    Mass-radius samples for evaluating a mass-radius likelihood:
    sample_single_neutron_star with observables=["R"].  As before, masses are
    limited by each EoS's maximum mass unless mass_prior_kwargs has an m_max,
    which is then used as given (objects="compact_objects").

    Returns:
      mr_samples: a DataFrame of samples with eos, eos maximum tov mass, mass, radius.
      There will generically be many m-r samples for each eos.
    """
    return sample_single_neutron_star(
        eos_posterior, eos_prior_set, mass_prior_class, num_samples_per_eos,
        observables=["R"], mass_prior_kwargs=mass_prior_kwargs,
        external_get_macro=external_get_macro,
        # The original function used an m_max given in mass_prior_kwargs
        # as is (even above the EoS's maximum mass), and only otherwise
        # truncated at the maximum mass; these two modes reproduce that
        objects="compact_objects" if "m_max" in mass_prior_kwargs else "neutron_stars",
    )


class _IndependentComponentMasses:
    """
    Makes a one-dimensional mass prior (anything with ``sample(n)``) usable as
    a binary mass prior: ``sample(n)`` draws mass_1 and mass_2 independently
    from it, as {"mass_1": ..., "mass_2": ...} (source frame, unordered;
    sample_neutron_star_binary relabels them heavier-first).
    """

    def __init__(self, mass_prior):
        self.mass_prior = mass_prior

    def sample(self, n):
        return {"mass_1": np.asarray(self.mass_prior.sample(n), dtype=float),
                "mass_2": np.asarray(self.mass_prior.sample(n), dtype=float)}


def generate_gw_samples(
    eos_posterior,
    eos_prior_set,
    mass_prior_class,
    num_samples_per_eos,
    mass_prior_kwargs={"m_min": 1.0},
    external_get_macro=None
):
    """
    Mass-tidal deformability samples for evaluating a gravitational-wave
    likelihood: sample_neutron_star_binary with observables=["Lambda"], with
    both component masses drawn independently from one mass prior.

    The mass prior is mass_prior_class(**mass_prior_kwargs), used as given
    for every EoS (as the original version of this function intended), so
    the binaries are compact objects: a component above an EoS's maximum
    mass is a black hole with Lambda = 0.  For a prior on the binary itself
    (e.g. uniform in components within chirp-mass bounds), or for neutron
    stars only, call sample_neutron_star_binary directly.

    Arguments:
      eos_posterior: An EoSPosterior, with each EoS's maximum mass in "Mmax"
      eos_prior_set: The EoSPriorSet corresponding to the eos_posterior
      mass_prior_class: A constructor for the mass prior, which must
        implement ``sample(n)``; its random numbers are its own (e.g. numpy's)
      num_samples_per_eos: the number of monte carlo samples used to resolve the likelihood
      mass_prior_kwargs: additional kwargs to pass to the mass_prior_class
        constructor
    Returns:
      mlambda_samples: a DataFrame of samples with eos, Mmax, mass_1, mass_2,
      lambda_1, lambda_2 (mass_1 >= mass_2, source frame; see the module
      docstring).  There will generically be many pairs of m-lambda samples
      for each eos.
    """
    # deepcopy: the default mass_prior_kwargs dict is shared between calls
    mass_prior = _IndependentComponentMasses(mass_prior_class(**copy.deepcopy(mass_prior_kwargs)))
    return sample_neutron_star_binary(
        eos_posterior, eos_prior_set, mass_prior, num_samples_per_eos,
        observables=["Lambda"], external_get_macro=external_get_macro,
        objects="compact_objects",
    )
