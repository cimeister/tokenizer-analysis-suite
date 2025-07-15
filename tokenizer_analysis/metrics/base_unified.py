"""
Base metrics class for unified TokenizedData interface - skeleton only.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
from collections import defaultdict
import logging

from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider
from ..constants import (
    DEFAULT_SAFE_DIVIDE_VALUE,
    Statistics,
    Validation,
    MAX_ERROR_DISPLAY_COUNT
)

logger = logging.getLogger(__name__)


class BaseMetrics(ABC):
    """Base class for tokenizer metrics using TokenizedData interface - skeleton only."""
    
    def __init__(self, input_provider: InputProvider):
        """
        Initialize base metrics.
        
        Args:
            input_provider: InputProvider instance providing tokenized data
        """
        self.input_provider = input_provider
        self.tokenizer_names = input_provider.get_tokenizer_names()
        self.language_metadata = None  # Can be set by subclasses
    
    def get_tokenized_data(self) -> Dict[str, List[TokenizedData]]:
        """Get tokenized data organized by tokenizer."""
        return self.input_provider.get_tokenized_data()
    
    def get_vocab_size(self, tokenizer_name: str) -> int:
        """Get vocabulary size for a tokenizer."""
        return self.input_provider.get_vocab_size(tokenizer_name)
    
    def get_languages(self, tokenizer_name: Optional[str] = None) -> List[str]:
        """Get available languages."""
        return self.input_provider.get_languages(tokenizer_name)
    
    @staticmethod
    def compute_basic_stats(values: List[float]) -> Dict[str, float]:
        """Compute basic statistics for a list of values."""
        if not values:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0,
                'sum': 0
            }
            
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values),
            'sum': sum(values)
        }
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = DEFAULT_SAFE_DIVIDE_VALUE) -> float:
        """Safely divide two numbers, returning default if denominator is zero."""
        return numerator / denominator if denominator != 0 else default
    
    def compute_pairwise_comparisons(self, values: Dict[str, float], metric_name: str = "metric") -> Dict[str, Dict[str, Any]]:
        """Compute pairwise comparisons between tokenizers."""
        comparisons = {}
        
        tokenizer_list = list(values.keys())
        for i, tok1 in enumerate(tokenizer_list):
            for j, tok2 in enumerate(tokenizer_list[i+1:], i+1):
                val1, val2 = values[tok1], values[tok2]
                
                comparison_key = f"{tok1}_vs_{tok2}"
                comparisons[comparison_key] = {
                    'tokenizer_1': tok1,
                    'tokenizer_2': tok2,
                    'value_1': val1,
                    'value_2': val2,
                    'difference': val1 - val2,
                    'ratio': self.safe_divide(val1, val2, 1.0),
                    'percent_difference': self.safe_divide(abs(val1 - val2), (val1 + val2) / 2, 0.0) * Statistics.PERCENTAGE_MULTIPLIER
                }
        
        return comparisons
    
    @staticmethod
    def empty_stats() -> Dict[str, float]:
        """Return empty statistics dictionary with zero values."""
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'count': 0,
            'sum': 0
        }
    
    @staticmethod
    def validate_non_empty_data(data: Any, name: str) -> None:
        """
        Validate that data is not empty.
        
        Args:
            data: Data to validate
            name: Name of the data for error messages
            
        Raises:
            ValueError: If data is empty
        """
        if not data:
            raise ValueError(f"{name} cannot be empty")
    
    @staticmethod
    def validate_minimum_count(items: List[Any], min_count: int, name: str) -> None:
        """
        Validate that a list has at least min_count items.
        
        Args:
            items: List to validate
            min_count: Minimum required count
            name: Name of the items for error messages
            
        Raises:
            ValueError: If items list is too short
        """
        if len(items) < min_count:
            raise ValueError(f"{name} must have at least {min_count} items, got {len(items)}")
    
    @staticmethod
    def validate_positive_number(value: float, name: str) -> None:
        """
        Validate that a number is positive.
        
        Args:
            value: Number to validate
            name: Name of the value for error messages
            
        Raises:
            ValueError: If value is not positive
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    
    @staticmethod
    def truncate_for_display(items: List[Any], max_count: int = MAX_ERROR_DISPLAY_COUNT) -> List[Any]:
        """
        Truncate a list for display purposes.
        
        Args:
            items: List to truncate
            max_count: Maximum number of items to keep
            
        Returns:
            Truncated list
        """
        if len(items) <= max_count:
            return items
        return items[:max_count]
    
    @staticmethod
    def format_list_for_display(items: List[Any], max_count: int = MAX_ERROR_DISPLAY_COUNT) -> str:
        """
        Format a list for display, truncating if necessary.
        
        Args:
            items: List to format
            max_count: Maximum number of items to show
            
        Returns:
            Formatted string representation
        """
        if len(items) <= max_count:
            return str(items)
        
        truncated = items[:max_count]
        return f"{truncated}... (showing {max_count}/{len(items)})"
    
    @abstractmethod
    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """
        Compute metrics using TokenizedData format.
        
        Args:
            tokenized_data: Optional dict mapping tokenizer names to TokenizedData lists.
                          If None, will use input_provider's data.
            
        Returns:
            Metrics results dictionary
        """
        pass


class TokenizedDataProcessor:
    """Utility class for processing TokenizedData objects."""
    
    @staticmethod
    def group_by_language(tokenized_data: List[TokenizedData]) -> Dict[str, List[TokenizedData]]:
        """Group TokenizedData objects by language."""
        grouped = defaultdict(list)
        for data in tokenized_data:
            grouped[data.language].append(data)
        return dict(grouped)
    
    @staticmethod
    def extract_tokens(tokenized_data: List[TokenizedData]) -> List[List[int]]:
        """Extract token lists from TokenizedData objects."""
        return [data.tokens for data in tokenized_data]
    
    @staticmethod
    def extract_texts(tokenized_data: List[TokenizedData]) -> List[str]:
        """Extract text strings from TokenizedData objects (where available)."""
        return [data.text for data in tokenized_data if data.text is not None]
    
    @staticmethod
    def flatten_all_tokens(tokenized_data: List[TokenizedData]) -> List[int]:
        """Flatten all tokens into a single list."""
        all_tokens = []
        for data in tokenized_data:
            all_tokens.extend(data.tokens)
        return all_tokens
    
    @staticmethod
    def count_total_tokens(tokenized_data: List[TokenizedData]) -> int:
        """Count total number of tokens across all data."""
        return sum(len(data.tokens) for data in tokenized_data)
    
    @staticmethod
    def get_unique_tokens(tokenized_data: List[TokenizedData]) -> set:
        """Get set of all unique token IDs."""
        unique_tokens = set()
        for data in tokenized_data:
            unique_tokens.update(data.tokens)
        return unique_tokens
    
    @staticmethod
    def validate_consistency(tokenized_data: List[TokenizedData], 
                           expected_tokenizer: Optional[str] = None,
                           expected_languages: Optional[List[str]] = None) -> bool:
        """Validate that TokenizedData objects are consistent."""
        if not tokenized_data:
            return False
        
        # Check tokenizer consistency
        if expected_tokenizer:
            if not all(data.tokenizer_name == expected_tokenizer for data in tokenized_data):
                return False
        
        # Check language consistency
        if expected_languages:
            found_languages = set(data.language for data in tokenized_data)
            if not found_languages.issubset(set(expected_languages)):
                return False
        
        # Check basic data integrity
        for data in tokenized_data:
            if not data.tokens or not isinstance(data.tokens, list):
                return False
            if not all(isinstance(token, int) for token in data.tokens):
                return False
        
        return True