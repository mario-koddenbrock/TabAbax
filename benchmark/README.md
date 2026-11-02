# Benchmarks

`main.py` runs the jobs. `evaluate.py` reads the local `results.pkl` files that
`main.py` wrote and prints a leaderboard. Neither script draws figures, and
neither script commits those pickles: the pickles are the output of a run, not
files in the repository.

| File | Role |
| --- | --- |
| `tabular/common.py` | Shared settings: the checkpoint arguments, the cache configuration, the single `run_jobs` call, and the leaderboard arguments. |
| `tabular/tabarena/main.py` | Runs TabArena v0.1, 51 datasets: 38 classification and 13 regression. |
| `tabular/tabarena/evaluate.py` | Prints the TabArena leaderboard for the pickles under `tabular/tabarena_out/TabAbax`. |
| `tabular/beyondarena/main.py` | Runs BeyondArena core2k, 142 datasets: 98 classification and 44 regression. |
| `tabular/beyondarena/evaluate.py` | Prints the BeyondArena leaderboard for the pickles under `tabular/beyondarena_out/TabAbax`. |

Both runners use one manual configuration, the full training table as context
with no bagging, and no feature shuffle. Classification and regression are
separate passes because they load separate checkpoints.

BeyondArena splits come from its `core2k` subset, which is the board the numbers
in `results/` were scored on. It is the default: no flag is needed.

```bash
python -m benchmark.tabular.beyondarena.main --problem all
```

`core2k` is a subset registered by the installed BeyondArena build that widens the
507-split `core` subset to 2000 splits. `main.py` always ANDs it with the problem
type, so `--problem all` is two passes: `core2k` and `classification`, then
`core2k` and `regression`. A run cannot fall back to `core`.

`--subset` adds one more filter instead of replacing the default. It takes a
subset name the context registers, so `--subset core2k` is a no-op and
`--subset large` narrows the board to the large tables:

```bash
python -m benchmark.tabular.beyondarena.main --problem regression --subset large
```

`evaluate.py` reads the same subset, so the printed leaderboard covers the board
the run scored.

| Setting | Value |
| --- | --- |
| TabArena bundle | `TabArenaV0pt1ExperimentBundle`, `outer_experiments=True`, pipeline `default`, `shuffle_features=False` |
| BeyondArena bundle | `BeyondArenaExperimentBundle`, pipeline `tabarena_default`, `text_cache_mode="require"`, `shuffle_features=False`, subset `core2k` plus the problem type, `require_warmup=False` |
| Checkpoints | `--classifier-checkpoint` and `--regressor-checkpoint`; omit them to download the published files at the pinned commit |
| `run_jobs` | one call, `register=False`, `debug_mode=True`, `cache_mode="default"` |
| Leaderboard | `compare` with `fillna="auto"`, `calibration_method="auto"`, `average_seeds=False`, `score_on_val=False`, `remove_imputed=False`, and `plot=False` |

No dataset is skipped. Do not pass `--skip` if the goal is to score the full
board.

## What a rerun reproduces

These scripts reproduce the evaluation procedure: the bundles, the pipelines, the
job call, the checkpoint handling, and the leaderboard arguments.

Not in this repository:

- the per-split `results.pkl` files, which a run writes;
- the TabArena and BeyondArena revisions that were installed;
- the machine record: CPU, RAM, disk, driver, CUDA, and PyTorch versions;
- the TabAbax TabArena branch, which supplies `TabAbaxModel`.

A rerun on different hardware, or against a newer TabArena or BeyondArena, can
produce different Elo values from the same code.

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

The task filter is `core2k`, then `classification` or `regression`.
The scored board has 142 datasets and 2000 splits: 98 classification and 44 regression.

```bash
python -m benchmark.tabular.beyondarena.main --problem all
python -m benchmark.tabular.beyondarena.main --problem regression --dataset QSAR_fish_toxicity
python -m benchmark.tabular.beyondarena.evaluate
```

`--data-foundry-cache` points at a local BeyondArena download.
Leave `--skip` unset to score every dataset.
`--checkpoint-revision` pins the Hub commit; leave it unset for the commit the wrapper pins.
The weight files are `tababax-classifier.ckpt` and `tababax-regressor.ckpt`
in commit `039e4f239d06a4d631e38c15bc23ef195ed8cf79`.
Hub commit `815239c50664fd73c1e4ee2d80414e9933757bc0` only changes the model-card text.

### Pinned weights

Omitting the checkpoint arguments downloads the released files at commit
`039e4f239d06a4d631e38c15bc23ef195ed8cf79`, the commit the TabArena wrapper pins, rather
than whatever the Hub's default branch holds when the run starts. The runner fetches the
files it needs before the first job and prints the commit; the estimator then loads them
from that same revision.

`--checkpoint-revision` overrides the commit for both the pre-fetch and the estimator.
Passing a local `--classifier-checkpoint` or `--regressor-checkpoint` skips the pre-fetch
for that problem type and leaves the file unmodified.
