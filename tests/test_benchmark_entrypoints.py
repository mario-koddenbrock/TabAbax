"""Checks that the run entry point and the evaluation entry point ask for the same board.

``results/README.md`` says the recorded numbers are the BeyondArena ``core2k`` board: 2000
splits. A leaderboard that filters ``core`` (507 splits) reports a different board, so the
two entry points have to agree. These tests read the expressions the scripts build rather
than the source text, so a future edit that re-spells either side fails here.
"""

import ast
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark" / "tabular"

SCRIPTS = ["beyondarena/main", "beyondarena/evaluate", "tabarena/main", "tabarena/evaluate"]


def test_board_constant_is_core2k():
    from benchmark.tabular.common import BEYONDARENA_SUBSET

    assert BEYONDARENA_SUBSET == "core2k"


def test_default_subset_is_the_problem_type_on_core2k():
    from benchmark.tabular.common import beyondarena_subset

    assert beyondarena_subset(None, "classification") == ["core2k", "classification"]
    assert beyondarena_subset(None, "regression") == ["core2k", "regression"]


def test_explicit_subset_keeps_the_board_and_the_problem_type():
    # ``--subset core2k`` is what the README recommends, and it has to equal the default
    # rather than replace the problem-type filter.
    from benchmark.tabular.common import beyondarena_subset

    assert beyondarena_subset(["core2k"], "classification") == ["core2k", "classification"]


def test_extra_subset_entries_are_and_ed_on():
    from benchmark.tabular.common import beyondarena_subset

    assert beyondarena_subset(["large"], "regression") == ["core2k", "regression", "large"]


def test_a_default_run_and_its_leaderboard_agree():
    # The runner passes its ``--subset`` list, the evaluator passes none. Both go through
    # ``beyondarena_subset``, so the scored board and the printed board have to match.
    from benchmark.tabular.common import beyondarena_subset

    for problem in ("classification", "regression", "all"):
        assert beyondarena_subset(None, problem) == beyondarena_subset(["core2k"], problem)


@pytest.mark.parametrize("name", SCRIPTS)
def test_entry_points_do_not_hardcode_a_legacy_subset(name):
    # A literal ``"core"`` is the pre-October-2026 protocol. It belongs in prose only.
    tree = ast.parse((BENCHMARK / f"{name}.py").read_text(encoding="utf-8"))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "core" not in literals, f"{name} filters the legacy core board"


def test_beyondarena_entry_points_share_the_subset_helper():
    # Both side of the BeyondArena pair must route through the one helper, otherwise the
    # two boards can drift apart again.
    for name in ("beyondarena/main", "beyondarena/evaluate"):
        tree = ast.parse((BENCHMARK / f"{name}.py").read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "benchmark.tabular.common"
            for alias in node.names
        }
        assert "beyondarena_subset" in imported, name


def test_prefetch_pins_the_wrapper_revision(monkeypatch):
    # The run script must resolve the commit the TabArena wrapper pins, so a run that
    # front-loads the download and the estimator that loads it cannot fetch different bytes.
    # The wrapper is faked, so this checks the wiring and not the commit itself.
    pin = "sentinel-revision"

    wrapper = types.ModuleType("tabarena.models.tababax.model")
    wrapper.CHECKPOINT_REVISION = pin
    wrapper.HF_REPO_ID = "tababax-ai/TabAbax"
    wrapper.HF_CHECKPOINTS = {
        "classification": "tababax-classifier.ckpt",
        "regression": "tababax-regressor.ckpt",
    }
    monkeypatch.setitem(sys.modules, "tabarena", types.ModuleType("tabarena"))
    monkeypatch.setitem(sys.modules, "tabarena.models", types.ModuleType("tabarena.models"))
    monkeypatch.setitem(sys.modules, "tabarena.models.tababax", types.ModuleType("tabarena.models.tababax"))
    monkeypatch.setitem(sys.modules, "tabarena.models.tababax.model", wrapper)

    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kwargs: calls.append(kwargs) or f"/cache/{kwargs['filename']}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    from benchmark.tabular.common import prefetch_checkpoints

    assert prefetch_checkpoints(["classification"]) == {
        "classification": "/cache/tababax-classifier.ckpt"
    }
    assert calls[0] == {
        "repo_id": "tababax-ai/TabAbax",
        "filename": "tababax-classifier.ckpt",
        "revision": pin,
    }


def test_prefetch_only_fetches_the_requested_problem_types(monkeypatch):
    # A classification-only run should not pull the regression checkpoint.
    wrapper = types.ModuleType("tabarena.models.tababax.model")
    wrapper.CHECKPOINT_REVISION = "sentinel-revision"
    wrapper.HF_REPO_ID = "tababax-ai/TabAbax"
    wrapper.HF_CHECKPOINTS = {
        "classification": "tababax-classifier.ckpt",
        "regression": "tababax-regressor.ckpt",
    }
    monkeypatch.setitem(sys.modules, "tabarena", types.ModuleType("tabarena"))
    monkeypatch.setitem(sys.modules, "tabarena.models", types.ModuleType("tabarena.models"))
    monkeypatch.setitem(sys.modules, "tabarena.models.tababax", types.ModuleType("tabarena.models.tababax"))
    monkeypatch.setitem(sys.modules, "tabarena.models.tababax.model", wrapper)

    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kwargs: calls.append(kwargs) or kwargs["filename"]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    from benchmark.tabular.common import prefetch_checkpoints

    assert set(prefetch_checkpoints(["regression"])) == {"regression"}
    assert [call["filename"] for call in calls] == ["tababax-regressor.ckpt"]
