"""Constructor checks that do not download checkpoints or run inference."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from tababax import TabAbaxClassifier, TabAbaxRegressor


def test_classifier_init_does_not_download():
    model = TabAbaxClassifier(allow_auto_download=False)
    assert model.n_estimators == 8
    assert model.allow_auto_download is False
    assert not hasattr(model, "model_")


def test_regressor_init_does_not_download():
    model = TabAbaxRegressor(allow_auto_download=False)
    assert model.n_estimators == 8
    assert model.allow_auto_download is False
    assert not hasattr(model, "model_")
