# Results recorded on 2026-10-26

The figure `docs/beyondarena_elo.png` and the numbers below come from an internal comparison
run on 2026-10-26.
The per-split results of the BeyondArena and TabArena runs are committed here:

| File | Benchmark | Slices | Datasets | Rows |
| --- | --- | --- | ---: | ---: |
| `beyondarena-core2k-2026-10-26-classification.parquet` | BeyondArena core2k | classification | 98 | 1446 |
| `beyondarena-core2k-2026-10-26-regression.parquet` | BeyondArena core2k | regression | 44 | 554 |
| `tabarena-v0.1-2026-10-26.parquet` | TabArena v0.1 | classification and regression | 51 | 816 |

The two BeyondArena files hold 2000 splits on the core2k subset, which is
`core` widened to 2000 splits; that is the subset the Elo figure was drawn on.
The TabArena file is one table for both problem types.

Each row is one dataset, fold and metric. The columns are `method`, `dataset`,
`fold`, `metric_error`, `metric`, `problem_type`, `time_train_s` and
`time_infer_s`. The `method` column reads `TabAbax`; the internal run recorded
working titles with run dates in them, and those were dropped at release.

TabArena v0.1 uses three folds and a per-dataset repeat count of 3 or 10, which
is why the file holds 34 datasets at 9 splits and 17 at 30. The `fold` column in
that file is the split index `0..n-1`, not the fold within a repeat: with
`n_folds=3` the repeat is `fold // 3`, and every dataset in the file stays inside
its configured repeat count, so no split was dropped. BeyondArena numbers splits
`0..n-1` per dataset in the same way, so a missing split shows up as a hole.

What these files do not carry: the leaderboard is scored against baselines from
the official cache, so the Elo values below are not recomputable from them alone.
The benchmark revisions that were installed, the machine record, and the code
that drew the figure were not kept either. See "Run record" for what is missing.
`benchmark/` contains the run and leaderboard commands, without the plotting code,
so a rerun reproduces the procedure and can land on different Elo values.

The position column is the place of this model when the printed table is sorted by Elo.
The average rank is the `rank` column of that same table. The two are not the same number.

## BeyondArena, core2k protocol

41 methods.

The board is the `core2k` subset: both `benchmark/tabular/beyondarena/main.py` and
`evaluate.py` filter it, so the run and the leaderboard cover the same 2000 splits.

| Slice | Datasets | Elo | Elo+ | Elo- | Average rank | Position |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 142 | 1237.7 | 29.0 | 29.1 | 15.40 | 10 |
| classification | 98 | 1218.8 | 30.2 | 33.7 | 16.26 | 11 |
| regression | 44 | 1286.7 | 64.7 | 43.2 | 13.47 | 8 |

The source note for these figures is `default`.

## TabArena

The combined board prints 51 datasets and 99 methods.

| Slice | Elo | Elo+ | Elo- | Average rank | Position |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 1676.0 | 74.0 | 58.3 | 17.66 | 11 |
| classification | 1646.7 | 81.7 | 72.4 | 20.24 | 12 |
| regression | 2023.7 | 179.7 | 95.8 | 10.12 | 7 |

The classification and regression notes do not print a dataset count. The runner in `benchmark/tabular/tabarena/main.py` expects the TabArena v0.1 split, 38 classification datasets and 13 regression datasets.

## Run record

The 2026-10-26 comparison ran on one NVIDIA H20 96 GB with two Intel Xeon
Platinum 8480+ CPUs, 1 TB of RAM and a 2 TB system disk. That is the machine the
original pull request reported, and it is the machine these numbers come from.
What was not kept is the software record: the TabArena and BeyondArena commits
that were installed, the CUDA, driver and PyTorch versions, and the code that
drew the figure. The hardware is known; the pinned software state is not.

These are **internal comparison numbers, not an official leaderboard rank**.
The Elo values come from comparing TabAbax against baseline results in the
official cache on 2026-10-26. They are not verified by the TabArena maintainers,
and the "10th of 41" position is the place in that internal table, not a rank
the project publishes or endorses. The committed parquet files hold the TabAbax
rows only, so the Elo column cannot be recomputed from them alone.

No dataset was skipped. The weights are the checkpoints now published at
[tababax-ai/TabAbax](https://huggingface.co/tababax-ai/TabAbax)
under the filenames `tababax-classifier.ckpt` and `tababax-regressor.ckpt`.
They were renamed for release. The weight upload is commit
`039e4f239d06a4d631e38c15bc23ef195ed8cf79`.
A later Hub commit, `815239c50664fd73c1e4ee2d80414e9933757bc0`,
only updates the model-card README. The weight files were not downloaded again
for this note. They are CC-BY-NC-4.0.

`benchmark/tabular/*/main.py` is the command that runs the jobs.
`evaluate.py` reads the `results.pkl` files that command writes and prints the
leaderboard. Those pickles stay local; the flattened per-split rows are what the
three parquet files above hold. Running `main.py` reproduces the procedure; it does
not replay the 2026-10-26 run.

Why the parquet files are not a full replay either: the run also wrote
`results.pkl` files holding the fitted-configuration records and the fold
assignments, and the leaderboard reads those. The parquet files are the scored
rows, which is enough to recount coverage and per-dataset error, not enough to
re-derive the Elo column against the baseline cache.

Not recorded with the 2026-10-26 notes: the TabArena and BeyondArena
commits, CPU, RAM, disk, CUDA, and PyTorch versions. The runner uses the
estimator default `random_state=42` when the caller does not override it.
