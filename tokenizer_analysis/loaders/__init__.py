"""
Data loaders for tokenizer analysis.
"""

from .multilingual_data import load_multilingual_data, load_language_data
from .code_data import CodeDataLoader

__all__ = ["load_multilingual_data", "load_language_data", "CodeDataLoader"]

from .._migration import make_module_getattr

__getattr__ = make_module_getattr(__name__, __all__)