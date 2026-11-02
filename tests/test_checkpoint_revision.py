"""Checkpoint resolution checks that never reach the network or run inference.

``checkpoint_revision`` exists so a benchmark can pin the Hub commit its weights came from; a
regression here would silently let a republished checkpoint change the bytes behind a published
score, so the argument must reach every ``hf_hub_download`` call.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

import tababax._sklearn.classifier as classifier_module
import tababax._sklearn.regressor as regressor_module
from tababax import TabAbaxClassifier, TabAbaxRegressor

REVISION = "039e4f239d06a4d631e38c15bc23ef195ed8cf79"


class _MissingFromCache(Exception):
    """Stands in for ``huggingface_hub``'s cache miss so the download branch runs without a request."""


def _record_calls(monkeypatch, module) -> list[dict]:
    """Capture the kwargs of every ``hf_hub_download`` call the module's ``_load_model`` makes."""
    calls: list[dict] = []

    def fake_download(*args, **kwargs):
        calls.append(dict(kwargs))
        raise _MissingFromCache

    monkeypatch.setattr(module, "hf_hub_download", fake_download)
    # The download branch is only reached after the cache lookup reports a miss.
    monkeypatch.setattr(module, "LocalEntryNotFoundError", _MissingFromCache)
    return calls


def test_classifier_passes_revision_to_the_hub(monkeypatch):
    calls = _record_calls(monkeypatch, classifier_module)
    model = TabAbaxClassifier(checkpoint_revision=REVISION, device="cpu")

    with pytest.raises(_MissingFromCache):
        model._load_model()

    assert calls, "the loader never asked the Hub for a checkpoint"
    assert {call["revision"] for call in calls} == {REVISION}
    assert {call["filename"] for call in calls} == {"tababax-classifier.ckpt"}


def test_regressor_passes_revision_to_the_hub(monkeypatch):
    calls = _record_calls(monkeypatch, regressor_module)
    model = TabAbaxRegressor(checkpoint_revision=REVISION, device="cpu")

    with pytest.raises(_MissingFromCache):
        model._load_model()

    assert calls, "the loader never asked the Hub for a checkpoint"
    assert {call["revision"] for call in calls} == {REVISION}
    assert {call["filename"] for call in calls} == {"tababax-regressor.ckpt"}


def test_revision_defaults_to_the_repository_head():
    """Unset revision keeps the historical behaviour: resolve the repo's default branch."""
    assert TabAbaxClassifier(device="cpu").checkpoint_revision is None
    assert TabAbaxRegressor(device="cpu").checkpoint_revision is None
