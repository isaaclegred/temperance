# temperance
A library for interfacing astrophysical data with inference of the nuclear equation of state

See `temperance/core/result` for the `EoSPosterior` class which represents the
fundamental probability distribution on EoSs.  In broader terms, an `EoSPosteror` contains
the data needed to evaluate certain individual event likelihoods.  We represent these
likelihoods as `core.result.WeightColumn` objects.  Note that the `WeightColumn` object doesn't contain any data, but rather contains instructions for how to compute weights given a column
in an `EoSPosterior.samples` dataframe.  A variety of operations are available to compute
statistics from `EoSPosterior` objects and a list of `WeightColumns`
 
See ./Examples/EoSPopulation.ipynb for an example of how to use some of the plotting utilities
available in temperance.  The basic idea of corner plots is that there are 2 main ingredients,
probabilities (per EoS) and values (per EoS), therefore the probability distribution
on EoSs induces a probability distribution on some invariant property of neutron stars, such as
the TOV maximum mass.  We need to pass the data which we want to plot as well as specification
for how it should be plotted.    To make a corner plot, first we need to define the posterior
which will be plotted, and the properties that will be plotted, these are contained in
`plotting.corner.PlottableSamples` and `plotting.corner.PlottableColumns` objects respectively.
These can then be passed to the `plotting.corner.corner_eos` function.  


I'm planning to make a plotting example for quantile/envelope plotting as well, but the basic
structure is similar to corner plots, except that quantiles are precomputed in a separate step.
See `./temperance/plotting/get_quantiles.py` for details about getting quantiles.  This
process also requires an `EoSPosterior` object. This will extract integer quantiles of some
quantity as a function of some other, for example pressure as a function of baryon density.
These can be plotted, to give, for example, a 90% credible region for the pressure at
any density.  There's some plotting utilites for this available in `plotting/envelope.py`, but
homebrewed plotting solutions also work well here.  