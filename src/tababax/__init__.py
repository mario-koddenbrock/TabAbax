import os

# The Xet downloader can sit at zero bytes on Windows. Plain HTTP completes.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from ._model import InferenceConfig
from ._sklearn import TababaxClassifier, TababaxRegressor

__all__ = [
    "TababaxClassifier",
    "TababaxRegressor",
    "InferenceConfig",
]
