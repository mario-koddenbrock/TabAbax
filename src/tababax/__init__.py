import os
import sys

# The Xet downloader can sit at zero bytes on Windows. Plain HTTP completes.
# Limited to Windows so that importing this package does not switch the download backend for
# every later Hugging Face call in the same process (and its subprocesses) on other platforms.
if sys.platform == "win32":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from ._model import InferenceConfig
from ._sklearn import TabAbaxClassifier, TabAbaxRegressor

__all__ = [
    "TabAbaxClassifier",
    "TabAbaxRegressor",
    "InferenceConfig",
]
