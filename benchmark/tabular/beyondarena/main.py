"""Run TabAbax on BeyondArena.

Same bundle settings as the 2026-10-26 run: the core subset, one manual
config, ``tabarena_default`` preprocessing, required text cache, and no
feature shuffle. The default subset is ``core`` plus the problem type.
"""

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.contexts import BeyondArenaContext
from tabarena.utils.config_utils import ConfigGenerator

from benchmark.tabular.common import cache_config, manual_config, run_jobs, tababax_model_cls


def _one(
    context,
    checkpoint: str | None,
    problem: str,
    result_dir: Path,
    dataset: str | None,
    skip: list[str],
    subset: list[str] | None,
) -> None:
    generator = ConfigGenerator(
        model_cls=tababax_model_cls(),
        name="TabAbax",
        search_space={},
        manual_configs=[manual_config(checkpoint)],
    )
    experiments = BeyondArenaExperimentBundle(
        models=[(generator, 0)],
        outer_experiments=True,
        preprocessing_pipelines=["tabarena_default"],
        text_cache_mode="require",
        shuffle_features=False,
    ).build_experiments()
    if len(experiments) != 1:
        raise RuntimeError(f"expected 1 experiment, got {len(experiments)}")
    for experiment in experiments:
        experiment.experiment_kwargs["require_warmup"] = False
    jobs = context.build_jobs(
        experiments,
        subset=subset or ["core", problem],
        dataset_names=[dataset] if dataset else None,
    )
    if skip:
        kept = []
        for job in jobs:
            name = str(getattr(job, "dataset", "") or getattr(getattr(job, "task", None), "dataset", ""))
            if not any(name == item or name.startswith(item + "-") for item in skip):
                kept.append(job)
        jobs = kept
    run_jobs(context, jobs, result_dir / problem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("classification", "regression", "all"), default="all")
    parser.add_argument("--classifier-checkpoint", default=None)
    parser.add_argument("--regressor-checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--subset", action="append", default=None)
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument("--tabarena-cache", default=None)
    parser.add_argument("--data-foundry-cache", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result_dir = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parents[1] / "beyondarena_out" / "TabAbax"
    )
    context = BeyondArenaContext(
        cache_config=cache_config(result_dir, args.tabarena_cache, args.data_foundry_cache),
    )
    if args.problem in {"classification", "all"}:
        _one(
            context,
            args.classifier_checkpoint,
            "classification",
            result_dir,
            args.dataset,
            args.skip,
            args.subset,
        )
    if args.problem in {"regression", "all"}:
        _one(
            context,
            args.regressor_checkpoint,
            "regression",
            result_dir,
            args.dataset,
            args.skip,
            args.subset,
        )


if __name__ == "__main__":
    main()
