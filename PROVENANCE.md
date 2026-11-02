# Code Provenance

## TabICLv2

The inference implementation is derived from:

https://github.com/soda-inria/tabicl

Reference:

Tag `v2.2.0`, commit `0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3`.

https://github.com/soda-inria/tabicl/commit/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3

Compared with that commit, these files match once the TabAbax attribution
header is removed: `_cgroup_memory.py`, `attention.py`, `encoders.py`,
`inference.py`, `inference_config.py`, `layers.py`, `learning.py`,
`quantile_dist.py`, `ssmax.py`, and `_torch_devices.py`.

`rope.py` and `sklearn_utils.py` match that commit apart from the header
and the third-party license sentences added in their docstrings.
`kv_cache.py` matches apart from the header and the rename of
`TabICLCache` to `TabAbaxCache`.

These files are modified derivatives of that commit: `embedding.py`,
`interaction.py`, `model.py` (from `tabicl.py`), `_sklearn/base.py`,
`classifier.py`, `preprocessing.py`, and `regressor.py`.
`embedding.py` adds `FourierFeatureEmbedding` and `ColInteraction`.

Derived areas:

- src/tababax/_model/
- src/tababax/_sklearn/base.py
- src/tababax/_sklearn/classifier.py
- src/tababax/_sklearn/regressor.py
- src/tababax/_sklearn/preprocessing.py
- src/tababax/_sklearn/sklearn_utils.py
- src/tababax/_torch_devices.py

The TabICL-derived portions retain the BSD 3-Clause License.

## TabAbax modifications

TabAbax changes include:

- renamed public classes and package namespace;
- modified architecture components;
- changed training and prior configuration, which are not in this repository;
- newly trained checkpoints, distributed separately under CC-BY-NC-4.0;
- inference integration and memory-management changes where applicable.

This repository contains inference code only.

It does not contain:

- pre-training code;
- synthetic prior generation code;
- training datasets;
- optimizer configuration;
- full checkpoint-generation scripts.

The released checkpoints were trained from a TabICLv2-derived implementation
with additional architectural and prior changes.

## Other third-party sources

- rotary-embedding-torch: MIT, Copyright (c) 2021 Phil Wang
- scikit-learn: BSD-3-Clause, commit 1eb422d6c5 of sklearn/utils/validation.py
- TabR: MIT, commit 75105013189c76bc4f247633c2fb856bc948e579
- PyTabKit: Apache-2.0, commit 949bf81e3964f65a33dd2c252c3713c239c17b2d

See NOTICE for the applicable notices and license texts.
