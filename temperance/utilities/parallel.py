"""
Evaluate a function on many EoSs in parallel processes.

    results = map_over_eos(fn, indices, prior=prior, n_procs=8, checkpoint="run.pkl")

- Workers run with single-threaded BLAS by default: numpy's default
  threading oversubscribes a process pool (3-10x slower in practice).
- With ``checkpoint``, results are saved as they arrive and a rerun skips
  the indices already done, so a long run can be summarised while it
  continues, or resumed after an interruption.
- With ``on_error="record"`` (default) an EoS whose evaluation fails gets
  an EoSEvaluationError instead of stopping the run.

Processes are started with "fork" where available, so ``fn`` and ``prior``
need not be picklable (lambdas and open HDF5 files are fine); priors with a
``reopen()`` method (e.g. EoSPriorH5) reopen their files in each worker,
because HDF5 handles must not be shared across a fork.  Forking a process
that is running threads can deadlock a child if a thread holds a lock at
the time (Python >= 3.12 warns about it); if that bites, pass
start_method="spawn" or "forkserver" with a picklable fn and prior (e.g. a
module-level function, and a prior built from a path inside it).
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

_BLAS_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass
class EoSEvaluationError:
    """Stands in for the result of an EoS whose evaluation raised."""
    index: object
    message: str
    traceback: str


# Set in each worker (inherited under fork, passed to the initializer otherwise)
_WORKER = {}


def _limit_blas_threads():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:   # environment variables alone only help processes not yet started
        return None
    return threadpool_limits(1)


def _initialize_worker(fn, prior, single_thread_blas):
    if prior is not None and hasattr(prior, "reopen"):
        prior.reopen()
    _WORKER.update(fn=fn, prior=prior,
                   limits=_limit_blas_threads() if single_thread_blas else None)


def _evaluate(fn, prior, index, on_error):
    try:
        return fn(index) if prior is None else fn(prior, index)
    except Exception as error:   # noqa: BLE001 - recorded for the caller
        if on_error == "raise":
            raise
        return EoSEvaluationError(index, repr(error), traceback.format_exc())


def _work(chunk, on_error):
    return [(index, _evaluate(_WORKER["fn"], _WORKER["prior"], index, on_error)) for index in chunk]


def _load_checkpoint(path):
    if path is None or not os.path.exists(path):
        return {}
    with open(path, "rb") as file:
        return pickle.load(file)


def _save_checkpoint(path, results):
    directory = os.path.dirname(os.path.abspath(path))
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as file:
        pickle.dump(results, file)
    os.replace(file.name, path)   # atomic: a reader never sees a partial file


def map_over_eos(fn, indices, *, prior=None, n_procs=1, chunksize=1, checkpoint=None,
                 checkpoint_every=100, single_thread_blas=True, on_error="record",
                 start_method=None, progress=None):
    """
    Evaluate ``fn`` on each EoS index.

    Parameters
    ----------
    fn : callable
        ``fn(prior, index)`` if ``prior`` is given, else ``fn(index)``.
    indices : iterable
        EoS indices (any hashable values).
    prior : object, optional
        Passed to fn, e.g. an EoSPriorSet or EoSPriorH5.
    n_procs : int
        Number of worker processes; 1 runs in this process.
    chunksize : int
        Indices sent to a worker at a time.
    checkpoint : path, optional
        Pickle file of {index: result}.  Indices already in it are not
        recomputed; it is rewritten every ``checkpoint_every`` new results
        and at the end.
    single_thread_blas : bool
        Limit BLAS / OpenMP to one thread in each worker (and in this
        process when n_procs == 1).
    on_error : "record" or "raise"
        "record" returns an EoSEvaluationError for an index whose
        evaluation raised; "raise" propagates the exception.
    start_method : str, optional
        multiprocessing start method; default "fork" where available,
        else "spawn" (which needs fn and prior to be picklable).
    progress : callable, optional
        Called as progress(n_done, n_total) as results arrive.

    Returns
    -------
    dict index -> result, in the order of ``indices``.
    """
    if on_error not in ("record", "raise"):
        raise ValueError("on_error must be 'record' or 'raise'")
    indices = list(indices)
    results = _load_checkpoint(checkpoint)
    todo = [index for index in indices if index not in results]
    n_new = 0
    n_done_before = len(indices) - len(todo)

    def collect(pairs):
        nonlocal n_new
        for index, value in pairs:
            results[index] = value
            n_new += 1
            if checkpoint is not None and n_new % checkpoint_every == 0:
                _save_checkpoint(checkpoint, results)
        if progress is not None:
            progress(n_done_before + n_new, len(indices))

    if todo and n_procs <= 1:
        limits = _limit_blas_threads() if single_thread_blas else None
        try:
            for start in range(0, len(todo), chunksize):
                collect((index, _evaluate(fn, prior, index, on_error)) for index in todo[start:start + chunksize])
        finally:
            if limits is not None:
                limits.restore_original_limits()
    elif todo:
        if start_method is None:
            start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
        context = multiprocessing.get_context(start_method)
        saved = {name: os.environ.get(name) for name in _BLAS_VARIABLES}
        if single_thread_blas:   # inherited by spawned workers before they import numpy
            os.environ.update({name: "1" for name in _BLAS_VARIABLES})
        try:
            with ProcessPoolExecutor(n_procs, mp_context=context, initializer=_initialize_worker,
                                     initargs=(fn, prior, single_thread_blas)) as pool:
                futures = [pool.submit(_work, todo[start:start + chunksize], on_error)
                           for start in range(0, len(todo), chunksize)]
                for future in as_completed(futures):
                    collect(future.result())
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    if checkpoint is not None and n_new:
        _save_checkpoint(checkpoint, results)
    return {index: results[index] for index in indices}
