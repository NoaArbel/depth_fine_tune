"""
Stubs out heavy ML dependencies (peft, transformers, sklearn) so that
loss-function unit tests can import from src.train without loading model weights.
"""
import sys
from unittest.mock import MagicMock

for mod in [
    "peft",
    "transformers",
    "sklearn",
    "sklearn.metrics",
    "timm",
]:
    sys.modules.setdefault(mod, MagicMock())
