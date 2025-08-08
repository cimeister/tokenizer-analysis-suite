"""
Tokenizer Analysis Module

A modular framework for comprehensive tokenizer comparison and analysis.
Supports both pairwise and multi-tokenizer comparisons with various metrics
including morphological alignment, information-theoretic measures, and
segmentation analysis.
"""

__version__ = "1.0.0"
__author__ = "Tokenizer Analysis Project"

from .metrics.base import BaseMetrics
from .metrics.basic import BasicTokenizationMetrics
from .metrics.information_theoretic import InformationTheoreticMetrics
from .metrics.morphological import MorphologicalMetrics
from .metrics.gini import TokenizerGiniMetrics
from .loaders import MorphologicalDataLoader
from .visualization import TokenizerVisualizer
from .main import UnifiedTokenizerAnalyzer, create_analyzer_from_raw_inputs, create_analyzer_from_tokenized_data

__all__ = [
    "BaseMetrics",
    "BasicTokenizationMetrics",
    "InformationTheoreticMetrics", 
    "MorphologicalMetrics",
    "TokenizerGiniMetrics",
    "MorphologicalDataLoader",
    "TokenizerVisualizer",
    "UnifiedTokenizerAnalyzer",
    "create_analyzer_from_raw_inputs",
    "create_analyzer_from_tokenized_data"
]