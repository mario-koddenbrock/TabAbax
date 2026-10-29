def test_public_api():
    from tababax import TabAbaxClassifier, TabAbaxRegressor, InferenceConfig

    assert TabAbaxClassifier.__name__ == "TabAbaxClassifier"
    assert TabAbaxRegressor.__name__ == "TabAbaxRegressor"
    assert InferenceConfig is not None
