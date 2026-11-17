import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.integrate import trapezoid

torch = pytest.importorskip("torch")

from temperance.weighing import flow   # noqa: E402
from temperance.weighing.weigh_by_density_estimate import get_normalizing_flow_mr_likelihood_estimate   # noqa: E402

# a NICER-like mass-radius posterior: scales of 0.1 Msun and 0.5 km, correlated
MEAN = np.array([2.2, 12.5])
COVARIANCE = np.array([[0.1 ** 2, 0.5 * 0.1 * 0.5], [0.5 * 0.1 * 0.5, 0.5 ** 2]])
TRUTH = stats.multivariate_normal(MEAN, COVARIANCE)


def _test_points(n=2000, seed=10):
    points = TRUTH.rvs(n, random_state=seed)
    return points[TRUTH.logpdf(points) > TRUTH.logpdf(MEAN) - 0.5 * 2.5 ** 2]   # within 2.5 sigma


def _rms_log_error(log_density, points):
    """rms error of a log density, up to a constant (which EoS weights do not depend on)."""
    error = log_density - TRUTH.logpdf(points)
    return np.sqrt(np.mean((error - np.median(error)) ** 2))


def _log_prob(model, points):
    with torch.no_grad():
        return model.log_prob(torch.tensor(points, dtype=torch.float)).numpy()


def test_flow_recovers_a_density_on_its_own_scale():
    torch.manual_seed(0)
    model = flow.generate_improved_flow_density_estimate(TRUTH.rvs(2000, random_state=1), print_every=0)
    points = _test_points()
    assert _rms_log_error(_log_prob(model, points), points) < 0.15
    # the standardization's Jacobian is included: the density integrates to 1
    m, r = np.meshgrid(np.linspace(1.6, 2.8, 241), np.linspace(9.5, 15.5, 241), indexing="ij")
    density = np.exp(_log_prob(model, np.column_stack([m.ravel(), r.ravel()]))).reshape(m.shape)
    assert trapezoid(trapezoid(density, r[0], axis=1), m[:, 0]) == pytest.approx(1.0, abs=0.05)


def test_flow_uses_the_weights():
    # samples from a broad distribution, weighted towards the narrow one
    broad = stats.multivariate_normal(MEAN, 4 * COVARIANCE)
    samples = broad.rvs(6000, random_state=2)
    torch.manual_seed(0)
    model = flow.generate_improved_flow_density_estimate(samples, weights=TRUTH.pdf(samples) / broad.pdf(samples),
                                                         print_every=0)
    points = _test_points()
    assert _rms_log_error(_log_prob(model, points), points) < 0.15


def test_training_stops_at_the_best_held_out_loss():
    torch.manual_seed(0)
    data = (TRUTH.rvs(500, random_state=3) - MEAN) / np.sqrt(np.diag(COVARIANCE))
    model, history = flow.build_and_train_flow(data, epochs=400, validation_fraction=0.2, patience=10,
                                               print_every=0)
    best = int(np.argmin(history["validation_loss"]))
    assert len(history["loss"]) == best + 1 + 10 < 400


def test_likelihood_divides_out_a_non_flat_prior():
    # posterior samples = Gaussian likelihood x a non-flat prior, with the log prior in "Prior"
    prior = stats.multivariate_normal(MEAN + [0.1, 0.5], np.diag([0.2 ** 2, 1.0 ** 2]))
    draws = prior.rvs(200_000, random_state=4)
    weights = TRUTH.pdf(draws)
    chosen = np.random.default_rng(5).choice(len(draws), 4000, p=weights / weights.sum())
    posterior = pd.DataFrame(draws[chosen], columns=["M", "R"])
    posterior["Prior"] = prior.logpdf(draws[chosen])
    torch.manual_seed(0)
    likelihood = get_normalizing_flow_mr_likelihood_estimate(posterior, print_every=0)
    points = _test_points()
    assert _rms_log_error(np.log(likelihood(points)), points) < 0.2


# --- GW: a posterior with a known likelihood, Gaussian in chirp mass, q and Lambda-tilde, under a prior
# flat in (mass_1, mass_2, lambda_1, lambda_2)

GW_OBSERVED = dict(chirp_mass=1.1979, mass_ratio=0.89, lambda_tilde=745.0)
GW_ERRORS = dict(chirp_mass=0.002, mass_ratio=0.1, lambda_tilde=100.0)


def _gw_log_likelihood(theta):
    from bilby.gw.conversion import component_masses_to_chirp_mass, lambda_1_lambda_2_to_lambda_tilde
    m1, m2, l1, l2 = theta.T
    values = dict(chirp_mass=component_masses_to_chirp_mass(m1, m2), mass_ratio=m2 / m1,
                  lambda_tilde=lambda_1_lambda_2_to_lambda_tilde(l1, l2, m1, m2))
    return -0.5 * sum(((values[k] - GW_OBSERVED[k]) / GW_ERRORS[k]) ** 2 for k in values)


def _gw_posterior_samples(n, seed, log_prior=lambda theta: 0.0, n_draws=3_000_000):
    """Importance sampling: chirp mass ~ normal, q ~ U(0.3, 1), lambdas ~ U(0, 3000)."""
    from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
    rng = np.random.default_rng(seed)
    mc = rng.normal(GW_OBSERVED["chirp_mass"], 2 * GW_ERRORS["chirp_mass"], n_draws)
    m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(mc, rng.uniform(0.3, 1.0, n_draws))
    theta = np.column_stack([m1, m2, rng.uniform(0, 3000, n_draws), rng.uniform(0, 3000, n_draws)])
    # a prior flat in (m1, m2) has density m1^2 / chirp mass in (chirp mass, q)
    log_w = (_gw_log_likelihood(theta) + log_prior(theta) + np.log(m1 ** 2 / mc)
             - stats.norm.logpdf(mc, GW_OBSERVED["chirp_mass"], 2 * GW_ERRORS["chirp_mass"]))
    w = np.exp(log_w - log_w.max())
    return theta[rng.choice(n_draws, n, p=w / w.sum())]


@pytest.fixture(scope="module")
def gw_test_points():
    points = _gw_posterior_samples(2000, seed=20)
    return points, _gw_log_likelihood(points)


def _gw_rms_log_error(likelihood, points, truth):
    error = np.log(likelihood(points)) - truth
    return np.sqrt(np.mean((error - np.median(error)) ** 2))


def test_gw_likelihood_recovers_a_known_likelihood(gw_test_points):
    from temperance.weighing.weigh_by_density_estimate import get_normalizing_flow_gw_likelihood_estimate
    samples = pd.DataFrame(_gw_posterior_samples(5000, seed=21), columns=["mass_1", "mass_2", "lambda_1", "lambda_2"])
    samples["Prior"] = 0.0
    torch.manual_seed(0)
    likelihood = get_normalizing_flow_gw_likelihood_estimate(samples, print_every=0)
    points, truth = gw_test_points
    assert _gw_rms_log_error(likelihood, points, truth) < 0.45
    # the labels of the two stars do not matter
    np.testing.assert_allclose(likelihood(points[:10]), likelihood(points[:10][:, [1, 0, 3, 2]]))
    # a black hole (lambda = 0) is moved to the smallest sampled lambda: finite, not zero
    black_hole = points[:10].copy()
    black_hole[:, 2] = 0.0
    floor = black_hole.copy()
    floor[:, 2] = samples["lambda_1"].min()
    np.testing.assert_allclose(likelihood(black_hole), likelihood(floor))


def test_gw_likelihood_divides_out_a_non_flat_prior(gw_test_points):
    from temperance.weighing.weigh_by_density_estimate import get_normalizing_flow_gw_likelihood_estimate
    log_prior = lambda theta: -theta[:, 2] / 1000 - theta[:, 3] / 1000   # favouring small lambdas
    theta = _gw_posterior_samples(5000, seed=22, log_prior=log_prior)
    samples = pd.DataFrame(theta, columns=["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"])
    samples["Prior"] = log_prior(theta)
    torch.manual_seed(0)
    likelihood = get_normalizing_flow_gw_likelihood_estimate(
        samples, columns=("mass_1_source", "mass_2_source", "lambda_1", "lambda_2"), print_every=0)
    points, truth = gw_test_points
    assert _gw_rms_log_error(likelihood, points, truth) < 0.5
