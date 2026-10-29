"""Private. Users should not import from this subpackage.

``Tababax`` is wrapped by :class:`tababax.TababaxClassifier`
and :class:`tababax.TababaxRegressor`.
``InferenceConfig`` is re-exported at :class:`tababax.InferenceConfig`.
"""

from .model import Tababax
from .inference_config import InferenceConfig
