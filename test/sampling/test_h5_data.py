"""
Tests for HDF5 EoS sets: EoSPriorH5, EoSPriorFamily and EoSPosterior.from_h5,
on small files laid out like the Zenodo 17352004 sets.
"""

import h5py
import numpy as np
import pandas as pd
import pytest

from temperance.core.result import EoSPosterior, WeightColumn
from temperance.sampling.eos_prior import EoSPriorFamily, EoSPriorH5, Extraction

N_EOS = 6
KEY = lambda i: f"{i:05d}"


def write_set(path, weights, radius_offset=0.0):
    """EoS / Macro / Weights groups, one dataset per draw, keys 00000, 00001, ..."""
    pc = np.geomspace(1e34, 1e36, 30)
    with h5py.File(path, "w") as f:
        for i, weight in enumerate(weights):
            eos = pd.DataFrame({"pressurec2": pc, "energy_densityc2": 10 * pc, "baryon_density": 9 * pc,
                                "cs2c2": np.full(30, 0.3)})
            mass = (2.0 + 0.05 * i) * (1 - np.exp(-np.arange(30) / 8))
            macro = pd.DataFrame({"central_pressurec2": pc, "M": mass,
                                  "R": 13 - 0.05 * i - radius_offset - np.arange(30) / 15})
            f.create_dataset(f"EoS/{KEY(i)}", data=eos.to_records(index=False))
            f.create_dataset(f"Macro/{KEY(i)}", data=macro.to_records(index=False))
            f.create_dataset(f"Weights/{KEY(i)}", data=weight)


@pytest.fixture
def zenodo_like(tmp_path):
    paths = {}
    for p_t, scale in [(1e12, 1.0), (1e13, 2.0), (3e12, 0.5)]:
        weights = scale * np.linspace(0.1, 1.0, N_EOS)
        if p_t == 1e12:
            weights[3] = np.nan
        path = tmp_path / f"1109_maxpc2-{p_t:.0e}_mrgagn_ns_data.h5"
        write_set(path, weights)
        paths[p_t] = path
    (tmp_path / "unrelated.h5").write_bytes(b"")
    return tmp_path, paths


def open_prior(path):
    return EoSPriorH5(path, "EoS", "Macro", index_to_key=KEY)


def test_h5_prior_container(zenodo_like):
    _, paths = zenodo_like
    with open_prior(paths[1e12]) as prior:
        assert len(prior) == N_EOS
        assert prior.keys() == [KEY(i) for i in range(N_EOS)]
        assert prior.indices() == list(range(N_EOS))
        eos, macro = prior.get_eos(2), prior.get_macro(2)
        assert isinstance(eos, pd.DataFrame) and list(eos.columns) == ["pressurec2", "energy_densityc2",
                                                                        "baryon_density", "cs2c2"]
        assert isinstance(macro, pd.DataFrame) and macro["M"].iloc[-1] == pytest.approx(2.1 * (1 - np.exp(-29 / 8)))
        handle = prior.h5_file
    assert not handle  # closed on leaving the context


def test_h5_prior_extraction(zenodo_like):
    _, paths = zenodo_like
    with open_prior(paths[1e12]) as prior:
        out = prior.get_property(prior.indices(), ["R"], Extraction("interpolation", True, "M", [1.4]))
    assert out.shape == (N_EOS, 1) and out["R(M=1.4)"].notna().all()


def test_posterior_from_h5_zeroes_nonfinite_weights(zenodo_like):
    _, paths = zenodo_like
    with pytest.warns(RuntimeWarning, match="1 non-finite weight"):
        posterior = EoSPosterior.from_h5(paths[1e12])
    assert list(posterior.samples["eos"]) == list(range(N_EOS))
    assert posterior.weight_columns_available == [WeightColumn("weight", is_log=False)]
    weights = posterior.get_total_weight(posterior.weight_columns_available)["total_weight"].to_numpy()
    expected = np.linspace(0.1, 1.0, N_EOS)
    expected[3] = 0.0
    np.testing.assert_allclose(weights, expected)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # a zero weight is not a warning
        posterior.get_total_weight(posterior.weight_columns_available)


def test_posterior_from_h5_can_drop_nonfinite_weights(zenodo_like):
    _, paths = zenodo_like
    with pytest.warns(RuntimeWarning, match="dropped"):
        posterior = EoSPosterior.from_h5(paths[1e12], nonfinite="drop")
    assert list(posterior.samples["eos"]) == [0, 1, 2, 4, 5]
    with pytest.raises(ValueError):
        EoSPosterior.from_h5(paths[1e12], nonfinite="keep")


def test_prior_family(zenodo_like):
    directory, paths = zenodo_like
    family = EoSPriorFamily.from_glob(directory / "*.h5", r"maxpc2-([0-9.e+]+)_", hyperparameter="p_t",
                                      eos_subgroup_path="EoS", macro_subgroup_path="Macro", index_to_key=KEY)
    with family:
        assert family.values() == [1e12, 3e12, 1e13]          # sorted, unrelated file skipped
        assert len(family) == 3 and family.hyperparameter == "p_t"
        assert [value for value, _ in family] == family.values()
        assert family[3e12].get_macro(0)["M"].iloc[0] == 0.0
        with pytest.warns(RuntimeWarning):
            posteriors = family.posteriors()
        log_z = family.log_evidences(posteriors)
    base = np.linspace(0.1, 1.0, N_EOS)
    with_nan_zeroed = base.copy()
    with_nan_zeroed[3] = 0.0
    assert log_z[1e12] == pytest.approx(np.log(with_nan_zeroed.mean()))
    assert log_z[3e12] == pytest.approx(np.log(0.5 * base.mean()))
    assert log_z[1e13] == pytest.approx(np.log(2.0 * base.mean()))
    assert not family[1e12].h5_file   # closed with the family
