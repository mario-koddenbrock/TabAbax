"""Turn TabArena result files into a leaderboard table. This script does not draw figures."""

from pathlib import Path

from tabarena.contexts import TabArenaContext
from tabarena.end_to_end import EndToEnd
from tabarena.models import MethodMetadata

ROOT = Path(__file__).resolve().parents[1] / "tabarena_out" / "TabAbax"
OUT = Path(__file__).resolve().parents[1] / "tabarena_evals" / "TabAbax"


def main() -> None:
    if next(ROOT.rglob("results.pkl"), None) is None:
        raise FileNotFoundError(f"no TabArena results under {ROOT}")
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = MethodMetadata.baseline(method="TabAbax", compute="gpu", artifact_dir=OUT / "artifacts")
    processed = EndToEnd.from_path_raw(
        path_raw=ROOT,
        method_metadata=metadata,
        task_metadata=TabArenaContext().task_metadata_collection,
        backend="native",
    )
    processed.get_results().to_csv(OUT / "method_results_per_split.csv", index=False)
    context = TabArenaContext.from_new_methods(processed.to_method_metadata_lst())
    leaderboard = context.compare(output_dir=OUT, only_valid_tasks=["TabAbax"], plot=False)
    print(context.leaderboard_to_website_format(leaderboard).to_string(index=False))


if __name__ == "__main__":
    main()
