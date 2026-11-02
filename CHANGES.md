# Changes

## Unreleased

- Restored explicit TabICLv2 provenance and attribution.
- Added third-party notices for TabICL, scikit-learn, TabR, PyTabKit, and rotary-embedding-torch.
- Clarified the separation between source-code licenses and the checkpoint license.
- Added model-card and benchmark provenance documentation.
- Added TabArena and BeyondArena runner scripts under `benchmark/`, and recorded the 2026-10-26 Elo summary.
- This commit corrects the earlier omission of TabICL attribution and clarifies the licensing of derived and original portions. Git history was not rewritten.

## 1.0.0

Inference release of TabAbax, derived from the TabICL v2 inference code.

The package exports `TabAbaxClassifier` and `TabAbaxRegressor`. Pre-training, fine-tuning, forecasting, and explanation entry points are not included. Checkpoints are not included.

The model code is derived from TabICL (https://github.com/soda-inria/tabicl), Copyright (c) 2025, Soda team @ Inria, and remains BSD-3-Clause. See `NOTICE`. Packaging added around it is Apache-2.0. The checkpoints downloaded from `tababax-ai/TabAbax` are CC-BY-NC-4.0. `requirements.txt` lists the inference dependencies. Omitting `model_path` downloads `tababax-classifier.ckpt` or `tababax-regressor.ckpt`.
