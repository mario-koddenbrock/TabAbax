---
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: tabular-classification
tags:
  - tabular
  - tabular-foundation-model
  - tabicl-derived
  - tabarena
---

# TabAbax

TabAbax is a research model derived from the TabICLv2 inference implementation.

Source implementation:
https://github.com/soda-inria/tabicl

The TabICL-derived inference portions remain under the BSD 3-Clause License.
The checkpoint weights in this repository are distributed under CC-BY-NC-4.0 and are not licensed for commercial use.

This file is the model card to upload to https://huggingface.co/tababax-ai/TabAbax.
It is not yet the live model card. The inference code is derived from TabICL v2.2.0, commit 0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3.

## Relationship to TabICL

TabAbax is not an independent implementation from scratch. The TabAbax team modified the architecture and training/prior configuration and trained the released checkpoints.

This repository contains the checkpoints only. The inference code is at https://github.com/tababax-research/TabAbax.
That repository does not contain the pre-training code, the synthetic prior, the training datasets, the optimizer configuration, or the checkpoint-generation scripts.

## Files

- `tababax-classifier.ckpt`: classifier checkpoint
- `tababax-regressor.ckpt`: regressor checkpoint

## Intended use

Research and non-commercial evaluation of tabular classification and regression models.

## Limitations

Results depend on dataset preprocessing, ensemble settings, hardware, and benchmark version. The reported results should not be interpreted as a guarantee on unseen datasets.

The BeyondArena and TabArena numbers in the code repository are an internal comparison dated 2026-10-26. The raw per-split files, hardware record, seeds, benchmark commit, and checkpoint hash are not published with them. See `results/README.md` in https://github.com/tababax-research/TabAbax.

## Citation

No TabAbax technical report or paper citation is available yet.

## Third-party notices

See:
https://github.com/tababax-research/TabAbax/blob/main/NOTICE
