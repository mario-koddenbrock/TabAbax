"""Print a BeyondArena leaderboard from local result files. This script does not draw figures.

Result pickles are produced by ``main.py`` and are not committed. The comparison
arguments are the ones used on the H20 run, with ``plot=False``.
"""

from pathlib import Path

from tabarena.contexts import BeyondArenaContext
from tabarena.end_to_end import EndToEnd
from tabarena.models import MethodMetadata

from benchmark.tabular.common import compare_kwargs

ROOT = Path(__file__).resolve().parents[1] / "beyondarena_out" / "TabAbax"
OUT = Path(__file__).resolve().parents[1] / "beyondarena_evals" / "TabAbax"


def main() -> None:
    if next(ROOT.rglob("results.pkl"), None) is None:
        raise FileNotFoundError(f"no BeyondArena results under {ROOT}. Run main.py first.")
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = MethodMetadata.baseline(method="TabAbax", compute="gpu", artifact_dir=OUT / "artifacts")
    processed = EndToEnd.from_path_raw(
        path_raw=ROOT,
        method_metadata=metadata,
        task_metadata=BeyondArenaContext().task_metadata_collection,
        backend="native",
    )
    processed.get_results().to_csv(OUT / "method_results_per_split.csv", index=False)
    context = BeyondArenaContext.from_new_methods(processed.to_method_metadata_lst())
    for subset_name in ("classification", "regression", "all"):
        subset_dir = OUT / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        leaderboard, _per_task = context.compare(**compare_kwargs(subset_dir, ["core", subset_name]))
        print(f"\n{subset_name}")
        print(context.leaderboard_to_website_format(leaderboard).to_string(index=False))


if __name__ == "__main__":
    main()
