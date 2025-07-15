"""
Constants for tokenizer analysis framework.

This module defines all magic numbers and configuration constants used throughout
the tokenizer analysis codebase to improve maintainability and reduce errors.
"""

from typing import List


class TextProcessing:
    """Constants for text processing and loading operations."""
    
    # Text length thresholds
    MIN_PARAGRAPH_LENGTH = 10
    MIN_LINE_LENGTH = 10  
    MIN_SENTENCE_LENGTH = 20
    MIN_CONTENT_LENGTH = 50
    
    # Text chunking and processing
    DEFAULT_CHUNK_SIZE = 500
    TRUNCATION_DISPLAY_LENGTH = 100
    MAX_TEXTS_FALLBACK = 10
    
    # File processing limits
    LARGE_ARRAY_THRESHOLD = 100
    ARRAY_SAMPLING_POINTS = 50


class Statistics:
    """Constants for statistical computations and analysis."""
    
    # Entropy analysis defaults
    DEFAULT_RENYI_ALPHAS: List[float] = [1.0, 2.0, 3.0]
    SHANNON_ENTROPY_ALPHA = 1.0
    
    # Fallback values and ratios
    FALLBACK_WORDS_PER_TOKEN = 0.5
    DEFAULT_SAFE_DIVIDE_VALUE = 0.0
    
    # Statistical display precision
    PERCENTAGE_MULTIPLIER = 100
    DEFAULT_PRECISION = 4


class Validation:
    """Constants for validation and error checking."""
    
    # Minimum counts for analysis
    MIN_WORD_LENGTH = 2
    MIN_LANGUAGES_FOR_GINI = 2
    MIN_LANGUAGES_FOR_COMPARISON = 1
    MIN_TOKENIZERS_FOR_COMPARISON = 2
    MIN_TOKENIZERS_FOR_PLOTS = 2
    
    # Display limits for error messages and debugging
    MAX_ERROR_DISPLAY_COUNT = 5
    MAX_TOKEN_DISPLAY_COUNT = 20
    MAX_EXAMPLE_DISPLAY_COUNT = 20
    
    # Array truncation for output
    MAX_ARRAY_DISPLAY_LENGTH = 5


class DataProcessing:
    """Constants for data processing and sampling operations."""
    
    # Random sampling
    DEFAULT_RANDOM_SEED = 42
    DEFAULT_MAX_TEXTS_PER_LANGUAGE = 1000
    DEFAULT_MAX_SAMPLES = 2000
    
    # Array processing
    STEP_SIZE_FOR_LARGE_ARRAYS = 50
    OVERLAP_THRESHOLD = 0
    
    # Default values for missing data
    DEFAULT_RANK_VALUE = 0.0
    DEFAULT_COST_VALUE = 0.0
    DEFAULT_UTILIZATION_VALUE = 0.0


class FileFormats:
    """Constants for file format handling."""
    
    # Supported file extensions
    JSON_EXTENSIONS = ['.json']
    TEXT_EXTENSIONS = ['.txt', '.text']
    PARQUET_EXTENSIONS = ['.parquet']
    
    # Text column names to search for in datasets
    TEXT_COLUMN_NAMES = ['text', 'content', 'sentence', 'document', 'passage']
    
    # File reading parameters
    DEFAULT_ENCODING = 'utf-8'
    ERROR_HANDLING = 'replace'


class Morphology:
    """Constants for morphological analysis."""
    
    # Token cleaning prefixes and suffixes
    BYTE_PREFIXES = ['▁', 'Ġ']
    CONTINUATION_PREFIXES = ['##']
    SUFFIX_PATTERNS = ['</w>', '@@']
    
    # Morphological analysis thresholds
    MIN_MORPHEME_LENGTH = 1
    MAX_MORPHEME_OVERLAP = 1.0
    
    # Punctuation for word cleaning
    PUNCTUATION = '.,!?;:"()[]{}'


class PlotConfiguration:
    """Constants for visualization and plotting."""
    
    # Plot dimensions and styling
    DEFAULT_FIGURE_SIZE = (12, 8)
    DEFAULT_DPI = 300
    
    # Color and style defaults
    DEFAULT_COLORMAP = 'viridis'
    DEFAULT_ALPHA = 0.7
    
    # Text and label limits
    MAX_LABEL_LENGTH = 50
    MAX_LEGEND_ITEMS = 10


# Convenience imports for commonly used constants
MIN_PARAGRAPH_LENGTH = TextProcessing.MIN_PARAGRAPH_LENGTH
MIN_LINE_LENGTH = TextProcessing.MIN_LINE_LENGTH
MIN_SENTENCE_LENGTH = TextProcessing.MIN_SENTENCE_LENGTH
MIN_CONTENT_LENGTH = TextProcessing.MIN_CONTENT_LENGTH
DEFAULT_CHUNK_SIZE = TextProcessing.DEFAULT_CHUNK_SIZE

DEFAULT_RENYI_ALPHAS = Statistics.DEFAULT_RENYI_ALPHAS
SHANNON_ENTROPY_ALPHA = Statistics.SHANNON_ENTROPY_ALPHA
FALLBACK_WORDS_PER_TOKEN = Statistics.FALLBACK_WORDS_PER_TOKEN
DEFAULT_SAFE_DIVIDE_VALUE = Statistics.DEFAULT_SAFE_DIVIDE_VALUE

MIN_LANGUAGES_FOR_GINI = Validation.MIN_LANGUAGES_FOR_GINI
MIN_TOKENIZERS_FOR_COMPARISON = Validation.MIN_TOKENIZERS_FOR_COMPARISON
MIN_TOKENIZERS_FOR_PLOTS = Validation.MIN_TOKENIZERS_FOR_PLOTS
MAX_ERROR_DISPLAY_COUNT = Validation.MAX_ERROR_DISPLAY_COUNT
MAX_TOKEN_DISPLAY_COUNT = Validation.MAX_TOKEN_DISPLAY_COUNT

DEFAULT_RANDOM_SEED = DataProcessing.DEFAULT_RANDOM_SEED
DEFAULT_MAX_TEXTS_PER_LANGUAGE = DataProcessing.DEFAULT_MAX_TEXTS_PER_LANGUAGE

TEXT_COLUMN_NAMES = FileFormats.TEXT_COLUMN_NAMES
DEFAULT_ENCODING = FileFormats.DEFAULT_ENCODING