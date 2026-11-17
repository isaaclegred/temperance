"""Run Examples/SyntheticInference/synthetic_inference.py at a small size, so the example keeps working."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

# the NICER and GW likelihoods are normalizing flows (the "ml" extra)
pytest.importorskip("torch")

EXAMPLE = Path(__file__).resolve().parents[2] / "Examples" / "SyntheticInference" / "synthetic_inference.py"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("synthetic_inference", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def result(example):
    return example.run(n_eos=40, n_mc=40, n_stars=25, seed=2, verbose=False, n_gw_samples=3000)


def test_observations_follow_the_true_eos(example, result):
    # the samplers drew the injected stars from the true EoS's macro table
    true_macro = example.macro_table(example.TRUE_EOS, 25)
    stable = true_macro.iloc[:int(true_macro["M"].idxmax()) + 1]
    observations = result["observations"]
    star = observations["nicer_star"]
    assert star["R"] == pytest.approx(np.interp(star["M"], stable["M"], stable["R"]), rel=1e-2)
    masses = observations["pulsar_masses"]
    assert len(masses) == example.N_HEAVIEST and np.all(np.diff(masses) <= 0)
    assert example.M_MIN <= masses.min() and masses.max() <= result["truth"]["Mmax"]
    for event in observations["gw_events"]:
        assert example.M_MIN <= event["true"]["mass_2"] <= event["true"]["mass_1"] <= result["truth"]["Mmax"]
        # the released posterior samples follow the likelihood around the observed values
        samples = event["samples"]
        assert len(samples) == 3000 and (samples["mass_1"] >= samples["mass_2"]).all()
        chirp_mass = example.gw_observables(*(samples[c].to_numpy() for c in ("mass_1", "mass_2", "lambda_1", "lambda_2")))[0]
        assert np.median(chirp_mass) == pytest.approx(event["chirp_mass"], abs=3 * example.CHIRP_MASS_ERROR)


def test_weights_and_summary(example, result):
    posterior, columns = result["posterior"], result["weight_columns"]
    weights = posterior.get_total_weight(columns)["total_weight"].to_numpy()
    assert len(weights) == 40 and np.all(np.isfinite(weights)) and weights.sum() > 0
    # EoSs that cannot support the heaviest observed pulsar (to 5 sigma) carry no weight
    heaviest = max(p.mass for p in result["observations"]["heavy_pulsars"])
    light = posterior.samples["Mmax"].to_numpy() < heaviest - 5 * example.PULSAR_MASS_ERROR
    assert weights[light].sum() < 1e-6 * weights.sum()
    summary = result["summary"]
    assert summary.index[0] == "prior" and summary.loc["prior", "n_eff"] == pytest.approx(40)
    assert "all data, ignoring the selection" in summary.index
    assert summary["n_eff"].iloc[-2] <= summary["n_eff"].iloc[0]


def test_binary_population(example):
    # the density of ordered pairs integrates to 1 below each cutoff, and the sampler stays inside it
    from scipy.integrate import trapezoid
    for m_max in (1.9, 2.3, 2.8):
        grid = np.linspace(example.M_MIN, m_max, 1501)
        m1, m2 = np.meshgrid(grid, grid, indexing="ij")
        density = example.binary_population_density(m1, m2, np.full(m1.shape, m_max))
        assert trapezoid(trapezoid(density, grid, axis=1), grid) == pytest.approx(1.0, abs=0.01)
        pairs = example.PairedBinaries(m_max, rng=0).sample(1000)
        assert np.all((example.M_MIN <= pairs["mass_2"]) & (pairs["mass_2"] <= pairs["mass_1"])
                      & (pairs["mass_1"] <= m_max))
