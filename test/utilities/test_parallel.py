"""
Tests for temperance.utilities.parallel.map_over_eos.
"""

import os
import pickle

import h5py
import numpy as np
import pandas as pd
import pytest

from temperance.sampling.eos_prior import EoSPriorH5
from temperance.utilities.parallel import EoSEvaluationError, map_over_eos

INDICES = list(range(12))

# "fork" in a process with threads (pytest, h5py, BLAS) is deliberate here; see map_over_eos
pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")


def square(index):
    return index**2


@pytest.mark.parametrize("n_procs, chunksize", [(1, 1), (1, 5), (3, 1), (3, 4)])
def test_results_in_order(n_procs, chunksize):
    out = map_over_eos(square, INDICES[::-1], n_procs=n_procs, chunksize=chunksize)
    assert list(out) == INDICES[::-1]
    assert out == {i: i**2 for i in INDICES}


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_picklable_function_with_other_start_methods(start_method):
    out = map_over_eos(square, INDICES, n_procs=2, chunksize=3, start_method=start_method)
    assert out == {i: i**2 for i in INDICES}


def test_lambdas_and_prior_work_in_workers():
    offset = 7
    out = map_over_eos(lambda prior, i: prior["scale"] * i + offset, INDICES, prior={"scale": 3}, n_procs=2)
    assert out == {i: 3 * i + 7 for i in INDICES}


def test_h5_prior_in_workers(tmp_path):
    path = tmp_path / "set.h5"
    with h5py.File(path, "w") as f:
        for i in INDICES:
            f.create_dataset(f"Macro/{i:05d}", data=pd.DataFrame({"M": [0.5, 1.0 + 0.1 * i]}).to_records(index=False))
            f.create_dataset(f"EoS/{i:05d}", data=pd.DataFrame({"p": [1.0]}).to_records(index=False))
    with EoSPriorH5(path, "EoS", "Macro", index_to_key=lambda i: f"{i:05d}") as prior:
        out = map_over_eos(lambda prior, i: float(prior.get_macro(i)["M"].max()), INDICES, prior=prior, n_procs=3)
        assert out[11] == pytest.approx(2.1)
        assert prior.get_macro(0)["M"].iloc[1] == pytest.approx(1.0)   # parent's file still open


def test_errors_are_recorded_or_raised():
    def odd_fails(i):
        if i % 2:
            raise ValueError(f"bad {i}")
        return i
    for n_procs in (1, 2):
        out = map_over_eos(odd_fails, INDICES, n_procs=n_procs)
        assert out[0] == 0 and isinstance(out[3], EoSEvaluationError)
        assert "bad 3" in out[3].message and "ValueError" in out[3].traceback
        with pytest.raises(ValueError, match="bad"):
            map_over_eos(odd_fails, INDICES, n_procs=n_procs, on_error="raise")


def test_checkpoint_resumes(tmp_path):
    path = tmp_path / "run.pkl"
    calls = []

    def recorded(i):
        calls.append(i)
        return -i
    map_over_eos(recorded, INDICES[:5], checkpoint=path)
    with open(path, "rb") as file:
        assert pickle.load(file) == {i: -i for i in INDICES[:5]}
    calls.clear()
    out = map_over_eos(recorded, INDICES, checkpoint=path, checkpoint_every=3)
    assert calls == INDICES[5:]                 # the first five came from the checkpoint
    assert out == {i: -i for i in INDICES}
    with open(path, "rb") as file:
        assert pickle.load(file) == out


def test_checkpoint_written_during_parallel_run(tmp_path):
    path = tmp_path / "run.pkl"
    seen = []

    def progress(done, total):
        if os.path.exists(path):
            with open(path, "rb") as file:
                seen.append(len(pickle.load(file)))
    map_over_eos(square, INDICES, n_procs=2, checkpoint=path, checkpoint_every=4, progress=progress)
    # partial checkpoints were readable while the run went on
    assert {4, 8} <= set(seen)


def test_progress_counts():
    reports = []
    map_over_eos(square, INDICES, chunksize=4, progress=lambda done, total: reports.append((done, total)))
    assert reports == [(4, 12), (8, 12), (12, 12)]


def blas_threads(_):
    import threadpoolctl
    np.dot(np.ones((50, 50)), np.ones((50, 50)))     # make sure BLAS is loaded
    return max((info["num_threads"] for info in threadpoolctl.threadpool_info()), default=None)


def test_single_thread_blas_in_workers():
    pytest.importorskip("threadpoolctl")
    unlimited = map_over_eos(blas_threads, range(2), n_procs=2, single_thread_blas=False)
    if None in unlimited.values() or max(unlimited.values()) == 1:
        pytest.skip("no multithreaded BLAS that threadpoolctl controls (e.g. Apple Accelerate)")
    assert set(map_over_eos(blas_threads, range(4), n_procs=2).values()) == {1}


def test_environment_restored():
    before = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    map_over_eos(square, INDICES, n_procs=2)
    assert {name: os.environ.get(name) for name in before} == before


def test_bad_on_error():
    with pytest.raises(ValueError):
        map_over_eos(square, INDICES, on_error="ignore")
