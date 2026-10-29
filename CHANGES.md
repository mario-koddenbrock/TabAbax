# Changes

## 1.0.0

Private inference release of Tababax.

The package exports `TababaxClassifier` and `TababaxRegressor`. Pre-training, fine-tuning, forecasting, and explanation entry points are not included. Checkpoints are not included.

The inference source is Apache-2.0. The checkpoints downloaded from `tababax-ai/tababax` are CC-BY-NC-4.0. `requirements.txt` lists the inference dependencies. Omitting `model_path` downloads `tababax-classifier.ckpt` or `tababax-regressor.ckpt`.
