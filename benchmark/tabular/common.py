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


def run_jobs(context, jobs, result_dir: Path) -> None:
    """Run each job on its own, without registering or plotting.

    ``cache_mode='default'`` skips a split when its results file is already on disk.
    ``register=False`` keeps a later classification or regression pass from
    seeing a task list that was cut down by the first pass.
    """
    import torch

    result_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        try:
            context.run_jobs(
                jobs=[job],
                expname=result_dir,
                register=False,
                cache_mode="default",
            )
        finally:
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
