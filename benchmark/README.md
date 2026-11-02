# Benchmarks

The directory layout follows
[NVIDIA/structured-data-models `benchmark/`](https://github.com/NVIDIA/structured-data-models/tree/main/benchmark):
`main` runs the jobs, `evaluate` reads the local `results.pkl` files and prints a leaderboard.
Neither script draws figures, and neither script commits those pickles.
NVIDIA's scripts work the same way: the pickles are the output of a run, not files in the repository.

The job that is executed is not NVIDIA's model list. It is the H20 run, pointed at TabAbax:

| Setting | H20 script | This runner |
| --- | --- | --- |
| TabArena bundle | `TabArenaV0pt1ExperimentBundle`, `outer_experiments=True`, pipeline `default`, `shuffle_features=False` | same |
| BeyondArena bundle | `BeyondArenaExperimentBundle`, pipeline `tabarena_default`, `text_cache_mode="require"`, `shuffle_features=False`, subset `core` plus the problem type, `require_warmup=False` | same |
| Checkpoints | classification and regression are separate passes | `--classifier-checkpoint` and `--regressor-checkpoint`; omit them to download the published renamed files |
| `run_jobs` | one call, `register=False`, `debug_mode=True`, `cache_mode="default"` | same |
| Leaderboard | `compare` with `fillna="auto"`, `calibration_method="auto"`, `average_seeds=False`, `score_on_val=False`, `remove_imputed=False` | same arguments, `plot=False` |
| Model class on H20 | `TabICLv2Model` plus a local source tree | `TabAbaxModel`, which loads the same renamed checkpoints |

The H20 run did not skip any dataset. Do not pass `--skip` if the goal is to repeat that run.
Left out on purpose: Elo plots, the lighttabc import shim, the AutoGluon shared-weight shim, the datetime fill patch, and the offline dataset materializer. Those were environment workarounds. They are not part of the scored configuration.

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

`--data-foundry-cache` points at a local BeyondArena download.
Leave `--skip` unset. The H20 run scored every dataset.
The weight files are `tababax-classifier.ckpt` and `tababax-regressor.ckpt`
in commit `039e4f239d06a4d631e38c15bc23ef195ed8cf79`.
Hub commit `815239c50664fd73c1e4ee2d80414e9933757bc0` only changes the model-card text.
