def test_public_api():
    from tababax import TababaxClassifier, TababaxRegressor, InferenceConfig

    assert TababaxClassifier.__name__ == "TababaxClassifier"
    assert TababaxRegressor.__name__ == "TababaxRegressor"
    assert InferenceConfig is not None
