"""Shared TabArena runner settings used by the 2026-10-26 TabAbax runs.

The layout matches NVIDIA's structured-data-models benchmark scripts.
The experiment arguments below are the ones those runs actually set.
Machine-specific patches from the notebooks (offline dataset materialization,
datetime fill, shared-weight shims, Elo plots) are not included.
"""

from __future__ import annotations

import gc
from pathlib import Path

def tababax_model_cls():
    from tabarena.models.tababax.model import TabAbaxModel

    return TabAbaxModel


def cache_config(results: Path, tabarena_cache: str | None, data_foundry: str | None):
    if tabarena_cache is None and data_foundry is None:
        return None
    from tabarena.caching import CacheConfig

    cache = Path(tabarena_cache).resolve() if tabarena_cache else results
    return CacheConfig(
        tabarena=cache,
        data_foundry=Path(data_foundry).resolve() if data_foundry else cache / "data_foundry",
        openml=cache / "openml",
        huggingface=cache / "huggingface",
        results=results,
    )


def manual_config(checkpoint: str | None) -> dict:
    """One manual config. No search space. Defaults stay on the estimator unless overridden."""
    if checkpoint is None:
        return {"allow_auto_download": True}
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"model_path": str(path.resolve()), "allow_auto_download": False}


def run_jobs(context, jobs, result_dir: Path) -> list:
    """Run the whole job list in one call, as the H20 scripts did.

    The NVIDIA example loops one job at a time and clears the CUDA cache.
    The published numbers came from a single ``run_jobs`` call:

    - ``register=False``, so a later classification or regression pass is not
      given a task list cut down by the first pass;
    - ``debug_mode=True``, so a failed split is raised instead of recorded as a skip;
    - ``cache_mode='default'``, so an existing ``results.pkl`` is reused.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    results = context.run_jobs(
        jobs,
        expname=result_dir,
        register=False,
        debug_mode=True,
        cache_mode="default",
    )
    gc.collect()
    return list(results)


def compare_kwargs(output_dir: Path, subset: list[str]) -> dict:
    """Leaderboard arguments from the H20 scripts, with plotting turned off."""
    return {
        "output_dir": output_dir,
        "subset": subset,
        "plot": False,
        "return_results": True,
        "fillna": "auto",
        "calibration_method": "auto",
        "average_seeds": False,
        "score_on_val": False,
        "remove_imputed": False,
    }
