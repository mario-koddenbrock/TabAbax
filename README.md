# TabAbax

Scikit-learn estimators for tabular classification and regression. `fit` stores the training table as context. `predict` runs one forward pass of a pretrained model. The model weights are not updated.

This repository is the inference package. It is what AutoGluon and TabArena import. It does not contain pre-training, fine-tuning, data priors, or checkpoints.

## Install

Python >= 3.10.

From a clone, install the inference dependencies and then the package:

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` lists torch, scikit-learn, numpy, scipy, pandas, einops, psutil, tqdm, and huggingface-hub. The same list is in `pyproject.toml`, so this also works:

```bash
pip install -e .
```

From the private GitHub repository:

```bash
pip install "git+https://github.com/tababax-research/TabAbax.git"
```

## Weights

Checkpoints are not in this repository. Pass a local file and refuse any download:

```python
TabAbaxClassifier(
    model_path="path/to/classifier.ckpt",
    allow_auto_download=False,
)
```

The official checkpoints will be published separately on Hugging Face. This release does not select a checkpoint on its own. Pass `model_path`.

## Usage

```python
from tababax import TabAbaxClassifier, TabAbaxRegressor

clf = TabAbaxClassifier(
    model_path="path/to/classifier.ckpt",
    allow_auto_download=False,
)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.predict_proba(X_test)

reg = TabAbaxRegressor(
    model_path="path/to/regressor.ckpt",
    allow_auto_download=False,
)
reg.fit(X_train, y_train)
reg.predict(X_test)
```

Inputs may be NumPy arrays or pandas DataFrames. String and categorical columns are encoded inside `fit`. The default prediction uses an ensemble of dataset views; `n_estimators=1` is a single forward pass. Ensemble members are forwarded one at a time unless `batch_size` is raised.

## AutoGluon and TabArena

The integration surface is the two estimators:

- `tababax.TabAbaxClassifier`: `fit`, `predict`, `predict_proba`
- `tababax.TabAbaxRegressor`: `fit`, `predict`

No extra server or training entry point is required. A caller constructs the estimator, passes `model_path`, and uses the scikit-learn API.

## Layout

```
src/tababax/_model/     architecture and in-context forward pass
src/tababax/_sklearn/   estimators, preprocessing, checkpoint loading
tests/test_import.py          public import check
```

`_model/learning.py` is the in-context forward path. It is not a pre-training loop.

## License

The draft product terms are in `LICENSE`. They allow internal evaluation and non-commercial research. Commercial and production use needs a separate written agreement. The contact and the final text will be added before a public release.

Portions of the source are derived from TabICL and remain under the BSD 3-Clause License. That notice is in `NOTICE` and is kept with the source.
