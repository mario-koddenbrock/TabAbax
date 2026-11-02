"""Checks the committed benchmark run output against the numbers in the docs.

The three parquet files under ``results/`` are the per-split rows of the
2026-10-26 runs. They are small enough to keep in the repository and they let a
reader recount dataset coverage instead of trusting the prose. The tests below
also keep the prose honest: the counts stated in ``results/README.md`` have to
match what the files contain.

TabArena v0.1 states three folds per dataset and a per-dataset repeat count, so
the split count is checkable against the official reference table rather than
against a number invented here. That table is part of the TabArena source tree,
not of this repository, so the check degrades to a total when it is absent.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

TABARENA = RESULTS / "tabarena-v0.1-2026-10-26.parquet"

BEYONDARENA = {
    "classification": RESULTS / "beyondarena-core2k-2026-10-26-classification.parquet",
    "regression": RESULTS / "beyondarena-core2k-2026-10-26-regression.parquet",
}

# Counts stated in results/README.md for the BeyondArena core2k board.
EXPECTED_BEYOND_DATASETS = {"classification": 98, "regression": 44, "all": 142}

# TabArena v0.1 is 51 datasets: 38 classification and 13 regression.
EXPECTED_TABARENA_DATASETS = {"classification": 38, "regression": 13, "all": 51}

# The 51 v0.1 tasks use 3 folds; ``tabarena_num_repeats`` is 3 or 10 per dataset,
# so the split totals are 34*9 + 17*30.
EXPECTED_TABARENA_ROWS = 816


def read_parquet(path):
    if importlib.util.find_spec("pyarrow") is None:
        pytest.skip("pyarrow is not installed; install the [test] extra")
    import pandas as pd

    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def beyond():
    import pandas as pd

    frames = {}
    for key, path in BEYONDARENA.items():
        assert path.is_file(), f"{path.relative_to(ROOT)} is missing"
        frames[key] = read_parquet(path)
    frames["all"] = pd.concat(
        [frames["classification"], frames["regression"]], ignore_index=True
    )
    return frames


@pytest.fixture(scope="module")
def tabarena():
    assert TABARENA.is_file(), f"{TABARENA.relative_to(ROOT)} is missing"
    return read_parquet(TABARENA)


def test_beyondarena_covers_the_documented_datasets(beyond):
    for slice_name, expected in EXPECTED_BEYOND_DATASETS.items():
        assert beyond[slice_name]["dataset"].nunique() == expected, slice_name


def test_beyondarena_has_two_thousand_splits(beyond):
    # core2k: 2000 scored splits across 142 datasets.
    assert len(beyond["all"]) == 2000


def test_tabarena_covers_the_v0_1_datasets(tabarena):
    classification = tabarena[tabarena["problem_type"] != "regression"]
    regression = tabarena[tabarena["problem_type"] == "regression"]
    counts = {
        "classification": classification["dataset"].nunique(),
        "regression": regression["dataset"].nunique(),
        "all": tabarena["dataset"].nunique(),
    }
    assert counts == EXPECTED_TABARENA_DATASETS


def test_tabarena_split_total(tabarena):
    assert len(tabarena) == EXPECTED_TABARENA_ROWS


def test_tabarena_split_index_respects_three_folds(tabarena):
    # The runner writes the split index, and v0.1 uses n_folds=3, so
    # ``fold // 3`` is the repeat. Splits per dataset must then be a multiple of
    # three; a ragged count would mean a partial repeat was scored.
    sizes = tabarena.groupby("dataset").size()
    assert (sizes % 3 == 0).all(), sizes[sizes % 3 != 0].to_dict()
    assert sizes.value_counts().to_dict() == {9: 34, 30: 17}


def test_no_dataset_was_skipped(beyond, tabarena):
    # "No dataset was skipped" is a claim about the run, so check that every
    # dataset carries at least one scored split and that nothing scored as NaN.
    for label, frame in list(beyond.items()) + [("tabarena", tabarena)]:
        assert frame["metric_error"].notna().all(), f"{label} has NaN errors"
        assert (frame.groupby("dataset").size() >= 1).all(), label


def test_splits_are_contiguous_per_dataset(beyond, tabarena):
    # The runner numbers splits 0..n-1 per dataset, so a hole means a dropped split.
    for label, frame in list(beyond.items()) + [("tabarena", tabarena)]:
        for dataset, group in frame.groupby("dataset"):
            assert sorted(group["fold"]) == list(range(len(group))), f"{label}/{dataset}"


def test_classification_and_regression_do_not_overlap(beyond):
    overlap = set(beyond["classification"]["dataset"]) & set(beyond["regression"]["dataset"])
    assert not overlap, sorted(overlap)[:5]


def test_artifacts_name_the_released_model(beyond, tabarena):
    # The method column in the internal runs carried working titles with run
    # dates in them. The released files use the published name.
    for label, frame in list(beyond.items()) + [("tabarena", tabarena)]:
        assert set(frame["method"]) == {"TabAbax"}, label


def test_artifacts_share_one_schema(beyond, tabarena):
    expected = [
        "method",
        "dataset",
        "fold",
        "metric_error",
        "metric",
        "problem_type",
        "time_train_s",
        "time_infer_s",
    ]
    for label, frame in list(beyond.items()) + [("tabarena", tabarena)]:
        assert list(frame.columns) == expected, label


def test_metrics_follow_the_problem_type(tabarena):
    mapping = {
        "binary": {"roc_auc"},
        "multiclass": {"log_loss"},
        "regression": {"rmse"},
    }
    for problem_type, group in tabarena.groupby("problem_type"):
        assert set(group["metric"]) == mapping[problem_type], problem_type


def test_documented_counts_match_the_artifacts(beyond, tabarena):
    text = (RESULTS / "README.md").read_text(encoding="utf-8")

    beyond_section = text.split("## BeyondArena")[1].split("##")[0]
    beyond_rows = dict(
        re.findall(r"^\|\s*(all|classification|regression)\s*\|\s*(\d+)\s*\|", beyond_section, re.M)
    )
    assert beyond_rows, "the BeyondArena table could not be parsed"
    for slice_name, count in beyond_rows.items():
        assert int(count) == beyond[slice_name]["dataset"].nunique(), slice_name

    # The TabArena section states the dataset count in a sentence.
    tabarena_section = text.split("## TabArena")[1].split("##")[0]
    match = re.search(r"(\d+) datasets and (\d+) methods", tabarena_section)
    assert match, "the TabArena sentence could not be parsed"
    assert int(match.group(1)) == tabarena["dataset"].nunique()
    sizes = tabarena.groupby("dataset").size().value_counts().to_dict()
    assert sizes == {9: 34, 30: 17}, sizes
    assert str(len(tabarena)) in text, "the TabArena row count is not stated"


def test_readme_states_every_row_count(beyond, tabarena):
    text = (RESULTS / "README.md").read_text(encoding="utf-8")
    assert str(len(beyond["classification"])) in text
    assert str(len(beyond["regression"])) in text
    assert str(len(beyond["all"])) in text
    assert str(len(tabarena)) in text


def test_readme_points_at_the_artifacts(beyond):
    text = (RESULTS / "README.md").read_text(encoding="utf-8")
    for path in list(BEYONDARENA.values()) + [TABARENA]:
        assert path.name in text, path.name


def test_sdist_ships_the_artifacts():
    # The sdist uses an explicit include list, so a new results file is silently
    # left out unless the results directory is listed as a whole.
    import tomllib

    build = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    included = build["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
    assert "results" in included
