"""Shared settings for the TabArena and BeyondArena runners.

Each runner builds one manual configuration and passes the whole job list to a
single ``run_jobs`` call. Plotting, dataset materialization workarounds, and the
shared-weight shims from the original notebooks are not included.
"""

from __future__ import annotations

import gc
from pathlib import Path

#: The BeyondArena board the numbers in ``results/`` were scored on: ``core2k``, 2000 splits over
#: 142 datasets. ``core`` is the narrower 507-split protocol the arena used before October 2026.
BEYONDARENA_SUBSET = "core2k"


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


def manual_config(checkpoint: str | None, revision: str | None = None) -> dict:
    """One manual config. No search space. Defaults stay on the estimator unless overridden.

    ``revision`` pins the Hugging Face commit the checkpoint is resolved against, so the
    pre-fetch and the estimator cannot fetch different bytes. Left unset, the wrapper's own
    pin applies. A local ``checkpoint`` is loaded as a file, and the revision only matters
    when that file is missing and the download fallback runs.
    """
    config: dict = {}
    if checkpoint is None:
        config["allow_auto_download"] = True
    else:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        config["model_path"] = str(path.resolve())
        config["allow_auto_download"] = False
    if revision is not None:
        config["checkpoint_revision"] = revision
    return config


def run_jobs(context, jobs, result_dir: Path) -> list:
    """Run the whole job list in one call.

    The whole list goes to a single ``run_jobs`` call:

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


def beyondarena_subset(subset: list[str] | None, problem: str) -> list[str]:
    """The subset expression for one problem type: ``core2k``, the problem type, then any extra.

    ``subset`` entries are AND-ed onto the board and the problem type instead of replacing
    them, so ``--subset core2k`` and the default produce the same expression and a run can
    never silently drop back to ``core`` or score both problem types on one pass.
    """
    expression = [BEYONDARENA_SUBSET, problem, *(subset or [])]
    return list(dict.fromkeys(expression))


def prefetch_checkpoints(tasks: list[str], revision: str | None = None) -> dict[str, str]:
    """Download the released checkpoints ``tasks`` need, at the pinned revision, before any job runs.

    The wrapper pins the revision on its own, so a run started without this call still loads
    the same bytes. The call front-loads the download, which keeps a multi-hour job from
    stalling on a first-split download, and it prints the commit the run is scored against.
    """
    from huggingface_hub import hf_hub_download

    from tabarena.models.tababax.model import CHECKPOINT_REVISION, HF_CHECKPOINTS, HF_REPO_ID

    revision = revision or CHECKPOINT_REVISION
    print(f"TabAbax checkpoints: {HF_REPO_ID} at {revision}")
    return {
        task: hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CHECKPOINTS[task], revision=revision)
        for task in tasks
    }


def compare_kwargs(output_dir: Path, subset: list[str]) -> dict:
    """Leaderboard arguments used for the scored run, with plotting turned off."""
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
