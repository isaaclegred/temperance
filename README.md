# temperance

Tools for inferring the dense-matter equation of state (EoS) from
astrophysical data, built around **samples of EoSs with likelihood
weights**: read sets of EoS draws, compute neutron-star properties for each
EoS, weigh the EoSs by observations, and summarise the resulting posterior.

temperance works in geometric units, G = c = M☉ = 1 (see
[Units](#units)).

## Installation

```bash
pip install -e .            # from a clone of this repository
pip install -e ".[test]"    # also install pytest
```

Optional extras: `[jax]` (JAX backend), `[ml]` (torch and pyro, for
normalising-flow density estimates), `[lal]` (lalsuite).
[universality](https://github.com/reedessick/universality) is optional: it is
only used by the legacy `UniversalityEoSPriorSet` and by
`corner_eos(..., use_universality=True)`.

## Overview

| Package | Contents |
|---|---|
| `temperance.core` | `EoSPosterior` and `WeightColumn` (`result`); weighted statistics (`weights`); KDEs (`kde`); mass populations (`populations`) |
| `temperance.sampling` | Sets of EoS draws (`eos_prior`: `EoSPriorSet`, `EoSPriorH5`, `EoSPriorFamily`, `Extraction`); radial stability (`stability`) |
| `temperance.solving` | EoS models (`analytic_eos`); TOV solver (`solve_relativistic`); stellar models (`stellar_models`); sequences of stars (`sequences`); oscillation modes (`solve_oscillations`) |
| `temperance.weighing` | Likelihood weights from observations: pulsar masses (`weigh_by_pulsar`), mass–radius and GW data via density estimates (`weigh_by_density_estimate`) |
| `temperance.plotting` | Corner plots (`corner`), credible envelopes (`get_quantiles`, `envelope`) |
| `temperance.utilities` | Unit constants (`units`); parallel evaluation over EoSs (`parallel`) |

## Core ideas

**`EoSPosterior`** (`core/result.py`) is a table with one row per EoS (an
`eos` index column) plus columns of weights and of per-EoS properties. A
**`WeightColumn`** holds no data: it says how to turn one column into weights
(whether it stores a log weight, and whether it is inverted). The posterior
for a set of observations is the product of the weights of their columns,
and most methods take a list of `WeightColumn`s to say which to use.

**EoS priors** (`sampling/eos_prior.py`) give access to the EoS and
macroscopic (M, R, Λ, ...) tables of each draw:

- `EoSPriorSet` — draws stored as CSV files in `DRAWmod<N>-<k>/` directories;
- `EoSPriorH5` — draws stored in an HDF5 file, one dataset per draw (e.g.
  the Zenodo record 17352004 sets of
  [Phys. Rev. D 112, 063003](https://doi.org/10.1103/9kh9-xfpd));
- `EoSPriorFamily` — several priors indexed by a hyperparameter, e.g. one
  HDF5 file per transition pressure.

`get_property` reads properties off each draw, as described by an
**`Extraction`**: values at given masses (or other variables), or maxima /
minima over the radially stable stars. Stable branches are found with the
Bardeen–Thorne–Meltzer turning-point rule (`sampling/stability.py`), which
keeps twin-star branches and rejects the high-density spiral.

## Examples

### Weigh a set of EoS draws and ask a question of the posterior

```python
from temperance.core.result import EoSPosterior
from temperance.sampling.eos_prior import EoSPriorH5, Extraction

path = "1109_maxpc2-1e12_mrgagn_01d000_00d010_ns_data.h5"
prior = EoSPriorH5(path, "EoS", "Macro", index_to_key=lambda i: f"{i:05d}")
posterior = EoSPosterior.from_h5(path)            # weights from the "Weights" group

# R at 1.4 Msun, and the maximum mass, for every EoS
posterior.add_property(prior, dependent_variables=["R"],
                       extraction=Extraction("interpolation", True, "M", [1.4]))
posterior.add_property(prior, dependent_variables=["M"],
                       extraction=Extraction("extremize", True))

posterior.probability(lambda s: s["R(M=1.4)"] > 12.5)   # P, sigma, n_eff, ...
posterior.probability(lambda s: s["max(M)"] > 2.3)
```

Other statistics on `EoSPosterior`: `conditional_probability`,
`estimate_stratified` (an expensive quantity evaluated on a random
subsample of a cheaply screened set of EoSs), `histogram`,
`estimate_evidence`, `compute_neff` / `compute_neff_kish`; and
`average_over_posteriors` (evidence-weighted average over, e.g., the members
of an `EoSPriorFamily`).

### Solve stars and sequences

```python
import numpy as np
from temperance.solving.analytic_eos import piecewise_eos
from temperance.solving.sequences import solve_sequence
from temperance.solving.stellar_models import RelativisticStellarModel

# a hadronic polytrope, a first-order phase transition and a causal core
eos = piecewise_eos([
    {"type": "rest_mass_polytrope", "Gamma": 2.0, "K": 100.0, "rho_max": 9e-4},
    {"type": "jump", "delta_e": 1e-3},
    {"type": "css", "cs2": 1.0, "e_max": 0.2},
])

star = RelativisticStellarModel.from_logenthalpy(eos, 0.6)
star.M, star.R, star.mass(2 * star.R), star.nu_of_r(2 * star.R)   # accessors valid outside the star

stars = solve_sequence(eos, lnh_c=np.linspace(0.05, 1.5, 60),
                       star_properties={"C": lambda model: model.C})
stars[stars["stable"]][["M", "R_km", "C"]]
```

Equations of state: `polytropic_eos`, `css_eos`, `piecewise_eos`
(polytropes, constant-speed-of-sound segments and latent-heat jumps) and
`interpolated_eos` (a table with `baryon_density`, `pressurec2` and
`energy_densityc2` columns in g/cm³, and optionally `cs2c2`).
`UniformDensityStellarModel`, `TolmanVIIStellarModel` and
`PrescribedDensityStellarModel` give EoS-free benchmark stars with the same
interface.

### Evaluate something expensive on many EoSs

```python
from temperance.utilities.parallel import map_over_eos
from temperance.utilities.units import KM_PER_MSUN

def maximum_compactness(prior, i):
    macro = prior.get_macro(i)
    return float((macro["M"] / (macro["R"] / KM_PER_MSUN)).max())

results = map_over_eos(maximum_compactness, range(len(prior)), prior=prior,
                       n_procs=8, checkpoint="compactness.pkl")
```

Workers run with single-threaded BLAS, failures are recorded per EoS
instead of stopping the run, and with `checkpoint` a rerun skips the EoSs
already done.

### Mass populations

Populations are one-dimensional
[bilby](https://lscsoft.docs.ligo.org/bilby/) priors; `core/populations.py`
truncates them at each EoS's maximum mass:

```python
from bilby.core.prior import TruncatedGaussian
from temperance.core.populations import population_fraction, truncated_pdf

galactic_dns = TruncatedGaussian(1.33, 0.09, minimum=1.0, maximum=3.0)
truncated_pdf(galactic_dns, [1.3, 1.4], m_max=2.1)
```

`population_fraction` gives the fraction of a population with some
property (with explicit rules for twin-star branches), and
`weigh_by_pulsar.weigh_EoSs_by_mass_measurement(..., population=,
mass_likelihood=)` uses the same population in a pulsar-mass likelihood.

### Plots

- **Corner plots:** describe the posteriors to plot with
  `plotting.corner.PlottableEoSSamples` and the properties with
  `PlottableColumn`, then call `corner_eos`. See
  `Examples/EoSPopulation.ipynb`.
- **Credible envelopes** (e.g. the 90% region of pressure as a function of
  density): `plotting.get_quantiles.get_quantiles` computes quantiles from an
  `EoSPosterior` and an EoS prior; `plotting/envelope.py` has plotting
  helpers. See `Examples/get_envelopes.py`.

More examples are in `Examples/`: stellar profiles (`Profiles/`),
oscillation modes (`Modes/`), CompOSE tables (`Compose/`) and weighing
mass–radius samples (`WeighingSamples/`).

## Units

All quantities are geometric with M☉ = 1 unless noted.
`temperance.utilities.units` defines the conversions once:

| Constant | Meaning |
|---|---|
| `KM_PER_MSUN` | G M☉ / c² = 1.4766 km, the unit of length |
| `CGS_DENSITY_TO_GEOM` | g/cm³ → geometric (a convention: 2.8×10¹⁴ g/cm³ ↦ 0.00045) |
| `OMEGA_TO_HZ`, `MSUN_TO_S` | angular frequency → Hz, time → s |

Tabulated EoSs are read in g/cm³ and converted with `CGS_DENSITY_TO_GEOM`.
Macroscopic tables report R in km.

## Development

```bash
pytest test
```

Tests run on GitHub Actions for Python 3.11 and 3.12 on every pull request.
Comparisons against universality are skipped when it is not installed.
