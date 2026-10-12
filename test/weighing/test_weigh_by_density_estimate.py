"""
Tests for temperance.weighing.weigh_by_density_estimate.
"""

import h5py
import numpy as np
import pandas as pd
import pytest

from temperance.core.result import EoSPosterior
from temperance.sampling.eos_prior import EoSPriorH5, EoSPriorSet, UniversalityEoSPriorSet
from temperance.weighing import weigh_by_density_estimate as wbde

NUM_EOS = 3
MASSES = np.array([1.1, 1.4, 1.9])


class FixedMassPrior:
    """Always 'samples' MASSES, so results are deterministic."""
    def __init__(self, m_min=1.0, m_max=2.0):
        pass

    def sample(self, num_samples):
        return MASSES[:num_samples]


@pytest.fixture
def macro_tables(tmp_path):
    rhoc = np.linspace(1e14, 2e15, 40)
    tables = {}
    with h5py.File(tmp_path / "draws.h5", "w") as f:
        for draw in range(NUM_EOS):
            mass = (2.0 + 0.1 * draw) * (1 - np.exp(-rhoc / 6e14))
            table = pd.DataFrame({"rhoc": rhoc, "M": mass, "R": 13.0 - draw - rhoc / 1e15,
                                  "Lambda": 500 * (1.4 / mass)**6})
            subdir = tmp_path / "DRAWmod1000-000000"
            subdir.mkdir(exist_ok=True)
            table.to_csv(subdir / f"macro-draw-{draw:06d}.csv", index=False)
            f.create_dataset(f"ns/eos_{draw}", data=table.to_records(index=False))
            f.create_dataset(f"eos/eos_{draw}", data=table.to_records(index=False))
            tables[draw] = table
    return tmp_path, tables


def test_generate_mr_samples_same_for_every_prior_type(macro_tables):
    root, tables = macro_tables
    posterior = EoSPosterior(pd.DataFrame({"eos": np.arange(NUM_EOS),
                                           "Mmax": [tables[d]["M"].max() for d in range(NUM_EOS)]}))
    priors = {
        "EoSPriorSet": EoSPriorSet(eos_dir=str(root), eos_column="eos", eos_per_dir=1000, macro_dir=str(root)),
        "UniversalityEoSPriorSet": UniversalityEoSPriorSet(eos_dir=str(root), eos_column="eos",
                                                           eos_per_dir=1000, macro_dir=str(root)),
        "EoSPriorH5": EoSPriorH5(root / "draws.h5", index_to_key=lambda x: f"eos_{int(x)}"),
    }
    results = {name: wbde.generate_mr_samples(posterior, prior, FixedMassPrior, len(MASSES))
               for name, prior in priors.items()}
    results["external_get_macro"] = wbde.generate_mr_samples(
        posterior, None, FixedMassPrior, len(MASSES), external_get_macro=lambda i: tables[int(i)])
    priors["EoSPriorH5"].h5_file.close()

    expected = results.pop("EoSPriorSet")
    assert np.isfinite(expected["R"]).all()
    for draw in range(NUM_EOS):
        rows = expected[expected["eos"] == draw]
        np.testing.assert_allclose(rows["R"], np.interp(MASSES, tables[draw]["M"], tables[draw]["R"]))
    for name, result in results.items():
        pd.testing.assert_frame_equal(result, expected, obj=name)
