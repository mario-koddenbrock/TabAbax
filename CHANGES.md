# Changes

## 1.0.0

Inference release of TabAbax, derived from the TabICL v2 inference code.

The package exports `TabAbaxClassifier` and `TabAbaxRegressor`. Pre-training, fine-tuning, forecasting, and explanation entry points are not included. Checkpoints are not included.

The model code is derived from TabICL (https://github.com/soda-inria/tabicl), Copyright (c) 2025, Soda team @ Inria, and remains BSD-3-Clause. See `NOTICE`. Packaging added around it is Apache-2.0. The checkpoints downloaded from `tababax-ai/TabAbax` are CC-BY-NC-4.0. `requirements.txt` lists the inference dependencies. Omitting `model_path` downloads `tababax-classifier.ckpt` or `tababax-regressor.ckpt`.
