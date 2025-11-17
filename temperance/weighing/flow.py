import torch
import pyro
import pyro.distributions.transforms as T
from pyro.distributions import Normal, TransformedDistribution, Uniform
from pyro.distributions.transforms import Planar, SplineCoupling
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cpu")  # change to "cuda" if you have a GPU

def safe_exp(x):
    return torch.exp(x).detach().numpy()
def get_uniform_flow_density_estimate(ranges):
    """
    Get a uniform distribution in pytorch over the ranges specified
    Arguments:
      ranges: List of tuples specifying the (min, max) for each dimension
    """
    UniformDistribution = Uniform(torch.tensor([r[0] for r in ranges], dtype=torch.float), torch.tensor([r[1] for r in ranges], dtype=torch.float))
    return UniformDistribution

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
    if weights is None:
        weights = np.ones(data.shape[0]) / data.shape[0]

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
    

    # ---------- Affine coupling layer (RealNVP style) ----------
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim, mask):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask.float())
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim * 2)
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x):
        masked = x * self.mask
        st = self.net(masked)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * 0.9
        y = masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = ((1 - self.mask) * s).sum(dim=1)
        return y, log_det
    
    def inverse(self, y):
        masked = y * self.mask
        st = self.net(masked)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * 0.9
        x = masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        log_det = -((1 - self.mask) * s).sum(dim=1)
        return x, log_det

# ---------- RealNVP flow ----------
class RealNVP(nn.Module):
    def __init__(self, dim, n_coupling, hidden_dim):
        super().__init__()
        self.dim = dim
        masks = []
        for i in range(n_coupling):
            mask = torch.arange(dim) % 2
            if i % 2 == 1:
                mask = 1 - mask
            masks.append(mask)
        self.layers = nn.ModuleList([AffineCoupling(dim, hidden_dim, m) for m in masks])
        self.register_buffer("base_mean", torch.zeros(dim))
        self.register_buffer("base_std", torch.ones(dim))
    
    def forward(self, x):
        log_det_tot = x.new_zeros(x.shape[0])
        z = x
        for layer in self.layers:
            z, ld = layer.forward(z)
            log_det_tot = log_det_tot + ld
        return z, log_det_tot
    
    def inverse(self, z):
        log_det_tot = z.new_zeros(z.shape[0])
        x = z
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            log_det_tot = log_det_tot + ld
        return x, log_det_tot
    
    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_base = -0.5 * (((z - self.base_mean) / self.base_std) ** 2).sum(dim=1) - 0.5 * self.dim * math.log(2 * math.pi)
        return log_base + log_det
    
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.dim, device=self.base_mean.device)
        x, _ = self.inverse(z)
        return x

# ---------- Training utility ----------
def build_and_train_flow(data_np, n_coupling=6, hidden_dim=128, lr=1e-3, epochs=200, batch_size=512, print_every=50):
    """
    Train a RealNVP on `data_np` (numpy array shape (N, D)).
    Returns (model, history) where model has .log_prob(torch.tensor) and .sample(n).
    """
    assert isinstance(data_np, np.ndarray)
    dim = data_np.shape[1]
    dataset = TensorDataset(torch.from_numpy(data_np.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = RealNVP(dim=dim, n_coupling=n_coupling, hidden_dim=hidden_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    
    history = {"loss": []}
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        n = 0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            logp = model.log_prob(batch)
            loss = -logp.mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.shape[0]
            n += batch.shape[0]
        epoch_loss = epoch_loss / n
        history["loss"].append(epoch_loss)
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}  loss={epoch_loss:.5f}")
    return model, history
def generate_improved_flow_density_estimate(data, weights=None):
    """
    Generate a density estimate using normalizing flows.
    Arguments:
      data: (n-d array) The data to estimate the density of
      weights: The weights of the data points. Array of the same length as the data.
    Returns:
        A normalizing flow distribution which can be used to estimate the density of the data.
    """
    model, history = build_and_train_flow(data, n_coupling=6, hidden_dim=128, lr=1e-3, epochs=200, batch_size=512, print_every=50)
    return model