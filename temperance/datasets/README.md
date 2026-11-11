# Datasets

Published measurements useful for EoS inference, one CSV table per kind of
data. GitHub renders the CSVs as searchable tables, and any CSV reader can
load them. In Python:

```python
from temperance import datasets

datasets.list_tables()          # ['gravitational_waves', 'nicer_mass_radius', 'pulsar_masses']
nicer = datasets.load_table("nicer_mass_radius")
nicer[nicer["name"] == "J0740+6620"][["analysis", "radius", "radius_minus", "radius_plus", "samples_url"]]
lower, upper = datasets.credible_interval(nicer, "radius")
```

## Tables

| Table | Contents |
|---|---|
| `pulsar_masses.csv` | Neutron-star masses from binaries: radio timing, optical spectroscopy, spider light curves, X-ray binaries. Includes the 129 non-GW entries of the Fan et al. 2024 population sample |
| `nicer_mass_radius.csv` | NICER pulse-profile mass–radius measurements |
| `gravitational_waves.csv` | Compact binaries with a neutron star: source-frame masses, tidal deformability, radii |
| `columns.csv` | The column dictionary: type, unit, whether required, whether part of the key, and description of every column |

## Format

- **One row per analysis.** The columns marked `key` in `columns.csv`
  identify a row: `(name, analysis)` in every table, plus `component` in
  `pulsar_masses`, where both stars of a double neutron star have rows. A
  pulsar or event analysed several times has several rows, e.g.
  `J0740+6620` with `riley2021`, `miller2021`, `salmi2024` and
  `dittmann2024`. `analysis` is the first author and publication year, with
  a suffix when one paper reports several analyses (`lvk2019_low_spin`,
  `lvk2019_high_spin`).
- **Shared columns** in every table: `name`, `analysis`, `reference`,
  `arxiv`, `doi`, `samples_url` (the public release of the posterior
  samples, blank if none), `samples_file` (the file in that release, where
  known) and `notes` (caveats, and results that have no column).
- **Credible intervals.** A quantity `q` is a median with offsets, `q`,
  `q_minus`, `q_plus`, so the interval is `[q - q_minus, q + q_plus]`. Its
  probability is in the row's `credible_level`. A quantity reported only as
  a range or one-sided limit (e.g. GW component masses with a low-spin
  prior, or an upper limit on Λ̃) uses `q_lower` and `q_upper` instead.
  Each row uses one form or the other, and `credible_interval` returns the
  bounds for either.
- **Blank** means not reported. Values are copied as the papers quote them.
- **Units:** masses in M☉, radii in km. GW masses are source frame, so the
  columns carry an explicit `_source` (unlike temperance's sampling code,
  where plain `mass_1` is source frame, and bilby, where it is detector
  frame).

`load_table` checks a table against the dictionary. Every declared column
must be present and no other. Required columns must be filled and values
must parse as their declared type. The key must be unique, and each
interval must be given completely in one form.
`test/datasets/test_datasets.py` runs these checks on every table.

## Adding data

- **A new analysis:** add a row to the table.
- **A new column:** add a row for it to `columns.csv`, and the column to the
  table.
- **A new kind of data:** create `<table>.csv` with the shared columns plus
  its own. Add its own columns to `columns.csv` with `table` set to the new
  name; the shared columns are the rows with `table` = `*`. `list_tables`
  and `load_table` pick it up without code changes.

## Caveats for inference

- **Masses counted twice.** Every NICER analysis of J0740+6620, J0437−4715
  and J1231−1411 uses a radio-timing mass prior (`radio_mass_prior`). Don't
  multiply its likelihood by the same pulsar's entry in `pulsar_masses`
  without accounting for this.
- **Analyses that overlap.** Successive analyses of one source share data
  (e.g. the J1614−2230 timing analyses, and the J0740+6620 NICER analyses).
  Choose one per source rather than combining them.
- **`GW170817 / lvk2018_parametrized_eos`** already requires
  M_max > 1.97 M☉.
- **`J1231-1411 / salmi2024`** converged only with a restricted radius
  prior.
- **`J0740+6620 / dittmann2024`**: the mass shown is the radio value the
  paper quotes.
- **Pulsar masses** mostly have no public posteriors; see
  [Neutron-star mass population](#neutron-star-mass-population).
- **Not yet included:** the 2024 reanalysis of J0030+0451 by
  Vinciguerra et al. (ApJ 961, 62), whose text gives no per-model credible
  intervals, and LVK catalogue reanalyses (GWTC-2.1, GWTC-3) of the events
  listed.

## Neutron-star mass population

`pulsar_masses` contains every non-GW entry of Table I of Fan et al. 2024
([arXiv:2309.12644](https://arxiv.org/abs/2309.12644)), the sample used by
Biswas & Rosswog 2024 ([arXiv:2408.15192](https://arxiv.org/abs/2408.15192)).
These rows are tagged `fan2024` in `compilations`. The table also has newer
or alternative analyses of the same systems, which are untagged:

```python
pulsars = datasets.load_table("pulsar_masses")
fan = pulsars[pulsars["compilations"].str.split().apply(lambda tags: "fan2024" in tags)]
```

Fan et al.'s seven GW masses (GW170817, GW190425, GW191219, GW200105 and
GW200115) are events in `gravitational_waves.csv`; use their posterior
samples.

**Three kinds of measurement.** A row gives one of:
- a mass (`mass`);
- a mass function with a total mass (`mass_function`, `total_mass`), from
  the advance of periastron;
- a mass function with a mass ratio (`mass_function`, `mass_ratio`), from
  the companion's radial velocities.

For the last two, the marginal likelihood of the pulsar mass is given in
Alsing et al. 2018 ([arXiv:1709.07889](https://arxiv.org/abs/1709.07889),
Eqs. 3–4). `PulsarMeasurement.from_row(row).get_mass_likelihood()`
(`temperance.weighing.pulsar_measurements`) builds the likelihood of any
row. `mass_ratio` is the pulsar mass over the companion mass, so the minimum
pulsar mass is f q (1 + q)². The implementations in Farr & Chatziioannou's
[Stan code](https://github.com/farr/AlsingNSMassReplication) and Fan et
al.'s script use a form written for the inverse ratio, which makes these
systems nearly uninformative there.

**Where the values come from.** Fan et al.'s rows are copied from their Table
I, with `reference` pointing to the original paper. We checked Fan et al.'s
values against the cited papers and corrected them where they differ; each
change is in `notes` and listed below. The `credible_level` of these rows is
0.683, as Fan et al. treat them, but several are not Gaussian 1σ results;
`notes` says which.

**Posteriors.** Of the 129 analyses in Fan et al.'s sample, only the two
NICER ones publish a posterior. Among the newer analyses, two more do:
- Fonseca et al. 2021, J0740+6620: PDFs on grids;
- Voisin et al. 2020, J0337+1715: an MCMC chain of the timing fit, from which
  the masses must be reconstructed.

`data_url` links the underlying data (timing models and arrival times, or
optical light curves) for 42 rows, from which a likelihood can be rebuilt.
Many timing papers say their data are available "on request".

**Newer analyses.** These include NANOGrav 15-yr, MeerKAT and later papers on
individual systems. Some revisions are large, e.g. J0348+0432 drops from 2.01
to 1.806 M☉ (saffer2025). The NANOGrav 15-yr masses (agazie2023) use the
Shapiro delay alone, so for some pulsars they are far less precise than, or
inconsistent with, fits that also use other relativistic effects.
Collection-level resources:

| Resource | What it has |
|---|---|
| Fan et al.'s [`calc_pmmax.py`](https://github.com/ShaoPengTang/ShowMyWorks/blob/main/calc_pmmax.py) | The exact inputs of Table I as Python lists, and the split-normal and mass-function likelihoods they sample (the samples are not released) |
| Farr & Chatziioannou's [`pulsars.h5`](https://media.githubusercontent.com/media/farr/AlsingNSMassReplication/master/pulsars.h5) | Alsing et al.'s 74 sources plus J0740+6620, as Gaussian masses, (f, M_T) and (f, q); the [repository](https://github.com/farr/AlsingNSMassReplication) has the likelihoods in Stan |
| You et al. 2025 ([arXiv:2412.05524](https://arxiv.org/abs/2412.05524)), [Zenodo 14375273](https://zenodo.org/records/14375273) | 90 neutron stars in spreadsheets, with orbital parameters; mass samples that are Gaussian approximations to the published values, with accreted-mass corrections for recycled pulsars |
| [Freire's neutron-star mass page](https://www3.mpifr-bonn.mpg.de/staff/pfreire/NS_masses.html) | Current radio-timing masses with a changelog, including systems newer than Fan et al.'s sample; HTML only |
| Pulsar-timing-array releases: [NANOGrav 15-yr](https://zenodo.org/records/16051178), [EPTA DR2](https://zenodo.org/records/8300645), [PPTA DR3](https://doi.org/10.25919/w0nw-jt05), [MeerKAT](https://doi.org/10.57891/j0vh-5g31) | Timing models and arrival times only; no mass chains or grids |

**When there is no posterior:**
- **Timing data public:** recompute a χ² grid over companion mass and
  inclination (tempo or PINT), as the timing papers do.
- **Only summary numbers:** a split normal fitted to the quoted interval is
  the usual approximation, as in Fan et al. The quoted values are posteriors
  under the authors' priors, which matters for broad measurements.

**Population caveats:**
- **Correlated masses in double neutron stars.** The two masses are
  correlated through the precisely measured total mass. Treating them as
  independent loses this.
- **Double pulsar.** The `kramer2021` masses of J0737−3039A/B use a
  moment-of-inertia prior from GW170817, so they depend on the EoS.
  `kramer2021_eos_free` does not.
- **Duplicate.** PSR J2039−5617 appears twice in Fan et al. (clark2021 and
  strader2019).
- **Companions that are not neutron stars.** NGC 1851A's companion
  (`J0514-4002A`) is a white dwarf (dutta2025), and J1906+0746's may be.
- **Masses counted twice.** The NICER rows (J0030+0451 riley2019,
  J0740+6620 riley2021) are the same analyses as in `nicer_mass_radius`.
- **Model-dependent masses.** Spider and X-ray-binary masses depend on the
  model of the companion, and burst masses (ozel2016) on burst modelling.

**Corrections to Fan et al. 2024** (details in `notes`):
- **J1930−1852:** the total mass is 2.59(4), not 2.54(3).
- **J1325−6253:** the total mass is 2.57(6), not 2.57183.
- **J1829+2456:** the 1σ errors are 0.004; 0.007 is the 95% error.
- **J1946+2052:** the source is Stovall et al. 2018, not Swiggum et al. 2015.
- **J2215+5135:** the source is Kandel & Romani 2020 (ApJ 892, 101), not ApJ
  903, 39.
- **J1738+0333:** the errors are +0.07/−0.06, not +0.7/−0.6.
- **J1141−6545:** the source is Bhat et al. 2008. NANOGrav 11-yr does not
  include this pulsar.
- **B1516+02B:** the source is Freire et al. 2008, ApJ 679, 1433.
- **J0045−7319:** the source is Thorsett & Chakrabarty 1999; "Nice 2003" does
  not give the value.
- **The six bursters:** their masses are tabulated in Özel & Freire 2016,
  and 4U 1724−207 repeats the SAX J1748.9−2021 entry.

Fan et al.'s values also include some that are not Gaussian 1σ results, such
as X 1822−371 (a hard range) and 4FGL J2039.5−5617 (half a wider interval).
These are noted in the rows but not changed.
