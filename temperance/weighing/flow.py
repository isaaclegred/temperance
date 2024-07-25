import torch
import pyro
import pyro.distributions.transforms as T
from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.transforms import Planar, SplineCoupling
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
def generate_flow_density_estimate(data, weights):
    """
    Generate a density estimate using normalizing flows.
    Arguments:
      data: (n-d array) The data to estimate the density of
      weights: The weights of the data points. Array of the same length as the data.
    Returns:
        A normalizing flow distribution which can be used to estimate the density of the data.
    """
    n_dim = data.shape[1]

    # Step 1: Define the base distribution - in this case, a standard normal distribution in n dimensions
    base_distribution = Normal(torch.zeros(n_dim), torch.ones(n_dim))

    # Step 2: Define the transformations - in this case, a sequence of Planar flows
    num_flows = 10  # for example
    flow_transforms = [Planar(input_dim=n_dim) for _ in range(num_flows)]
    spline_transform = T.spline_coupling(input_dim=n_dim, count_bins=12)
    #pyro
    #flow_transforms = pyro.nn.AutoRegressiveNN(n_dim, [n_dim], param_dims=[n_dim, n_dim, 1], hidden_activation=torch.tanh)

    # Step 3: Create the normalizing flow
    flow_distribution = TransformedDistribution(base_distribution, [spline_transform])

    # Step 4: Generate some data from the flow
    #data = flow_distribution.sample((100,))

    # Step 5: Train the flow to fit the data
    # This is left as an exercise - you would typically use maximum likelihood, i.e., optimize the log prob of the data under the flow
    # Note that this is non-trivial and involves e.g. computing the Jacobian of the flow
    smoke_test = ('CI' in os.environ)
    steps = 1 if smoke_test else 2001
    print("steps", steps)
    #data=data[np.random.choice(np.arange(data.shape[0]), data.shape[0], p=weights, replace=True), :]
    dataset = torch.tensor(data, dtype=torch.float)
    weight_set = torch.tensor(weights, dtype=torch.float)
    optimizer = torch.optim.Adam(spline_transform.parameters(), lr=5e-3)
    for step in range(steps):
        optimizer.zero_grad()
        loss = -(flow_distribution.log_prob(dataset) * weight_set).mean()
        loss.backward()
        optimizer.step()
        flow_distribution.clear_cache()

        if step % 200 == 0:
            print('step: {}, loss: {}'.format(step, loss.item()))
    # Step 6: Sample from the flow
    return flow_distribution
    