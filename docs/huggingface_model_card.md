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

The inference code is derived from TabICL v2.2.0, commit `0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3`.
That commit is the version this card compares against. It is not a record of the commit that
was copied when the code was first taken from TabICL. See `PROVENANCE.md` in the code repository
for which files match that version and which are modified.

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

The 2026-10-26 comparison ran on NVIDIA H20. No dataset was skipped. The weights are the renamed files in this repository. Per-split `results.pkl` files are produced by `benchmark/tabular/*/main.py` and are not committed; the flattened per-split rows from that run are committed in `results/` in the code repository: `beyondarena-core2k-2026-10-26-classification.parquet` (1446 rows, 98 datasets), `beyondarena-core2k-2026-10-26-regression.parquet` (554 rows, 44 datasets) and `tabarena-v0.1-2026-10-26.parquet` (816 rows, 51 datasets). Together the BeyondArena pair is the 2000-split `core2k` board. The benchmark revisions and software versions from that run were not kept, so a rerun reproduces the procedure rather than these exact scores. See `results/README.md` in https://github.com/tababax-research/TabAbax.

## Citation

No TabAbax technical report or paper citation is available yet.

## Third-party notices

See:
https://github.com/tababax-research/TabAbax/blob/main/NOTICE
