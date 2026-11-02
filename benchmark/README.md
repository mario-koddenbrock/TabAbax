# Benchmarks

Tabular evaluation entry points for TabAbax. The directory layout follows
[NVIDIA/structured-data-models `benchmark/`](https://github.com/NVIDIA/structured-data-models/tree/main/benchmark):
one `main` module that runs jobs, and one `evaluate` module that reads the
result files and prints a leaderboard. These scripts do not draw figures.

The experiment settings are the ones used for the internal runs summarized in
`results/README.md`:

- one manual configuration, no hyperparameter search;
- `outer_experiments=True`, so the training table is the context and AutoGluon bagging is not applied;
- TabArena preprocessing pipeline `default`;
- BeyondArena preprocessing pipeline `tabarena_default`, with `text_cache_mode="require"`;
- `shuffle_features=False`;
- classification and regression are separate passes because they load different checkpoints.

The checkpoint weights are CC-BY-NC-4.0. A run that leaves the checkpoint
arguments empty downloads `tababax-ai/TabAbax`. That download is not licensed
for commercial use.

Install TabArena with the data and preprocessing extras before running:

```bash
pip install -e .
pip install "tabarena[data-foundry,preprocessing]"
```

The installed TabArena must contain `tabarena.models.tababax.model.TabAbaxModel`.
That class lives in the TabAbax TabArena branch. It is not in upstream TabArena
until a pull request is merged.

## TabArena

TabArena v0.1 has 51 datasets: 38 classification and 13 regression.

```bash
python -m benchmark.tabular.tabarena.main --problem all
python -m benchmark.tabular.tabarena.main --problem classification --dataset blood-transfusion-service-center
python -m benchmark.tabular.tabarena.evaluate
```

Pass `--classifier-checkpoint` and `--regressor-checkpoint` to score local files
instead of the Hugging Face download. Pass `--tabarena-cache` when the official
artifacts are already on disk.

## BeyondArena

The default task filter is `core`, then `classification` or `regression`.
The 2026-10-26 summary used 142 core datasets: 98 classification and 44 regression.

```bash
python -m benchmark.tabular.beyondarena.main --problem all
python -m benchmark.tabular.beyondarena.main --problem regression --dataset QSAR_fish_toxicity
python -m benchmark.tabular.beyondarena.evaluate
```

`--data-foundry-cache` points at a local BeyondArena download. `--skip DATASET`
drops a dataset by its name prefix. The 2026-10-26 run did not skip any
dataset, and it ran on NVIDIA H20. Leave `--skip` unset to match that run.

The weights for that run are the files published as `tababax-classifier.ckpt`
and `tababax-regressor.ckpt` at `tababax-ai/TabAbax`, revision
`039e4f239d06a4d631e38c15bc23ef195ed8cf79`.
