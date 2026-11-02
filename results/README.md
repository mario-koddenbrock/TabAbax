# Results recorded on 2026-10-26

The figure `docs/beyondarena_elo.png` and the numbers below come from an internal comparison
run on 2026-10-26.
They are not the output of a run in this repository: the raw per-split files, the
benchmark revisions that were installed, the machine record, and the code that drew
the figure were not kept.
`benchmark/` contains the run and leaderboard commands, without the plotting code, so a
rerun reproduces the procedure and can land on different Elo values.

The position column is the place of this model when the printed table is sorted by Elo.
The average rank is the `rank` column of that same table. The two are not the same number.

## BeyondArena, core protocol

41 methods.

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

The 2026-10-26 comparison ran on NVIDIA H20. No dataset was skipped.
The weights are the checkpoints now published at
[tababax-ai/TabAbax](https://huggingface.co/tababax-ai/TabAbax)
under the filenames `tababax-classifier.ckpt` and `tababax-regressor.ckpt`.
They were renamed for release. The weight upload is commit
`039e4f239d06a4d631e38c15bc23ef195ed8cf79`.
A later Hub commit, `815239c50664fd73c1e4ee2d80414e9933757bc0`,
only updates the model-card README. The weight files were not downloaded again
for this note. They are CC-BY-NC-4.0.

`benchmark/tabular/*/main.py` is the command that runs the jobs.
`evaluate.py` reads the `results.pkl` files that command writes and prints the
leaderboard. Those pickles are run outputs. They are not committed. Running
`main.py` reproduces the procedure; it does not replay the 2026-10-26 run.

Not recorded with the 2026-10-26 notes: the TabArena and BeyondArena
commits, CPU, RAM, disk, CUDA, and PyTorch versions. The runner uses the
estimator default `random_state=42` when the caller does not override it.
