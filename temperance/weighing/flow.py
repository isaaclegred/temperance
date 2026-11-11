"""
Normalizing-flow density estimates (a RealNVP in torch), used for
likelihoods estimated from posterior samples (see weigh_by_density_estimate).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
    UniformDistribution = torch.distributions.Uniform(torch.tensor([r[0] for r in ranges], dtype=torch.float), torch.tensor([r[1] for r in ranges], dtype=torch.float))
    return UniformDistribution


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
def build_and_train_flow(data_np, n_coupling=6, hidden_dim=128, lr=1e-3, epochs=200, batch_size=512, print_every=50,
                         weights=None, validation_fraction=0.0, patience=20):
    """
    Train a RealNVP on `data_np` (numpy array shape (N, D)), maximising the
    (weighted, if `weights` is given) mean log probability of the data.
    With validation_fraction > 0, that fraction of the data is held out;
    training stops once the held-out loss has not improved for `patience`
    epochs, and the model with the lowest held-out loss is returned (a flow
    trained too long overfits its samples).
    Returns (model, history) where model has .log_prob(torch.tensor) and .sample(n).
    """
    assert isinstance(data_np, np.ndarray)
    dim = data_np.shape[1]
    if weights is None:
        weights = np.ones(len(data_np))
    data = torch.from_numpy(data_np.astype(np.float32))
    weights = torch.from_numpy(np.asarray(weights, dtype=np.float32))
    order = torch.randperm(len(data))
    n_validation = int(validation_fraction * len(data))
    validation, train = order[:n_validation], order[n_validation:]
    loader = DataLoader(TensorDataset(data[train], weights[train]), batch_size=batch_size, shuffle=True)

    model = RealNVP(dim=dim, n_coupling=n_coupling, hidden_dim=hidden_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    def weighted_loss(x, w):
        return -(w * model.log_prob(x)).sum() / w.sum()

    history = {"loss": [], "validation_loss": []}
    best = (np.inf, None, 0)   # (held-out loss, model state, epoch)
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        n = 0
        for batch, batch_weights in loader:
            batch, batch_weights = batch.to(device), batch_weights.to(device)
            opt.zero_grad()
            loss = weighted_loss(batch, batch_weights)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.shape[0]
            n += batch.shape[0]
        epoch_loss = epoch_loss / n
        history["loss"].append(epoch_loss)
        if n_validation:
            with torch.no_grad():
                validation_loss = weighted_loss(data[validation].to(device), weights[validation].to(device)).item()
            history["validation_loss"].append(validation_loss)
            if validation_loss < best[0]:
                best = (validation_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
        if print_every and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch {epoch}/{epochs}  loss={epoch_loss:.5f}"
                  + (f"  held-out loss={history['validation_loss'][-1]:.5f}" if n_validation else ""))
        if n_validation and epoch - best[2] >= patience:
            if print_every:
                print(f"Epoch {epoch}: held-out loss last improved at epoch {best[2]}; stopping")
            break
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, history


class StandardizedFlow(nn.Module):
    """
    A flow trained on standardized data, (x - shift) / scale, used on the
    original scale: log_prob includes the Jacobian -sum(log scale).
    """

    def __init__(self, model, shift, scale):
        super().__init__()
        self.model = model
        self.register_buffer("shift", torch.as_tensor(shift, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def log_prob(self, x):
        return self.model.log_prob((x - self.shift) / self.scale) - torch.log(self.scale).sum()

    def sample(self, num_samples):
        return self.model.sample(num_samples) * self.scale + self.shift


def generate_improved_flow_density_estimate(data, weights=None, **training_kwargs):
    """
    Generate a density estimate using normalizing flows (a RealNVP).
    The data are standardized (by their weighted mean and standard
    deviation) before training, since each coupling layer can only rescale
    by a bounded factor; the returned flow works on the original scale.
    By default 20% of the data are held out to stop training before the flow
    overfits (see build_and_train_flow).
    Arguments:
      data: (n-d array) The data to estimate the density of
      weights: The weights of the data points. Array of the same length as the data.
      training_kwargs: passed to build_and_train_flow (e.g. epochs, hidden_dim,
        validation_fraction, patience)
    Returns:
        A normalizing flow distribution which can be used to estimate the density of the data.
    """
    data = np.asarray(data, dtype=float)
    weights = np.ones(len(data)) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    shift = np.sum(weights[:, None] * data, axis=0)
    scale = np.sqrt(np.sum(weights[:, None] * (data - shift) ** 2, axis=0))
    settings = dict(n_coupling=6, hidden_dim=128, lr=1e-3, epochs=200, batch_size=512, print_every=50,
                    validation_fraction=0.2, patience=20)
    settings.update(training_kwargs)
    # weights rescaled to mean 1, so the loss has the same size as an unweighted one
    model, history = build_and_train_flow((data - shift) / scale, weights=weights * len(data), **settings)
    return StandardizedFlow(model, shift, scale)