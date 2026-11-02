"""Run TabAbax on TabArena v0.1.

One manual config, the full training table as context (no bagging), the default
preprocessing pipeline, and no feature shuffle. Classification and regression
load separate checkpoints.
"""

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import ConfigGenerator

from benchmark.tabular.common import cache_config, manual_config, run_jobs, tababax_model_cls

EXPECTED = {"classification": 38, "regression": 13, "all": 51}


def _one(context, checkpoint: str | None, problem: str, result_dir: Path, dataset: str | None) -> None:
    generator = ConfigGenerator(
        model_cls=tababax_model_cls(),
        name="TabAbax",
        search_space={},
        manual_configs=[manual_config(checkpoint)],
    )
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(generator, 0)],
        outer_experiments=True,
        preprocessing_pipelines=["default"],
        shuffle_features=False,
    ).build_experiments()
    if len(experiments) != 1:
        raise RuntimeError(f"expected 1 experiment, got {len(experiments)}")
    jobs = context.build_jobs(
        experiments,
        subset=[problem],
        dataset_names=[dataset] if dataset else None,
    )
    if dataset is None and len(jobs) != EXPECTED[problem]:
        raise RuntimeError(
            f"TabArena {problem} produced {len(jobs)} jobs, expected {EXPECTED[problem]}"
        )
    run_jobs(context, jobs, result_dir / problem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("classification", "regression", "all"), default="all")
    parser.add_argument("--classifier-checkpoint", default=None)
    parser.add_argument("--regressor-checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--tabarena-cache", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result_dir = Path(args.output) if args.output else Path(__file__).resolve().parents[1] / "tabarena_out" / "TabAbax"
    context = TabArenaContext(cache_config=cache_config(result_dir, args.tabarena_cache, None))
    if args.problem in {"classification", "all"}:
        _one(context, args.classifier_checkpoint, "classification", result_dir, args.dataset)
    if args.problem in {"regression", "all"}:
        _one(context, args.regressor_checkpoint, "regression", result_dir, args.dataset)


if __name__ == "__main__":
    main()
