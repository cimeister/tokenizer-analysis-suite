"""
Data loaders for tokenizer analysis.

Contains loaders for various datasets including morphological data.
"""

from .morphological import MorphologicalDataLoader
from .multilingual_data import load_multilingual_data, load_language_data

__all__ = ["MorphologicalDataLoader", "load_multilingual_data", "load_language_data"]