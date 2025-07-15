#!/usr/bin/env python3
"""
Normalization configuration for tokenizer metrics.

This module provides flexible normalization options for tokenizer analysis,
allowing metrics to be calculated per line, byte, character, or word.
"""

from enum import Enum
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
import re
from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Sequence

class NormalizationMethod(Enum):
    """Enumeration of available normalization methods."""
    LINES = "lines"
    BYTES = "bytes" 
    CHARACTERS = "characters"
    WORDS = "words"


class PretokenizationMethod(Enum):
    """Enumeration of available pretokenization methods for word-based normalization."""
    PYTHON_SPLIT = "python_split"  # Default Python .split()
    HUGGINGFACE_WHITESPACE = "hf_whitespace"  # HuggingFace whitespace pretokenizer
    REGEX_WHITESPACE = "regex_whitespace"  # Custom regex for whitespace
    CUSTOM_REGEX = "custom_regex"  # User-defined regex pattern


@dataclass
class NormalizationConfig:
    """Configuration for normalization methods in tokenizer analysis."""
    
    # Primary normalization method
    method: NormalizationMethod = NormalizationMethod.BYTES
    
    # Pretokenization method (used when method is WORDS)
    pretokenization: PretokenizationMethod = PretokenizationMethod.PYTHON_SPLIT
    
    # Custom regex pattern (used when pretokenization is CUSTOM_REGEX)
    custom_regex: Optional[str] = None
    
    # Whether to include empty splits (affects word counting)
    include_empty_splits: bool = False
    
    # Text encoding (used for byte-based normalization)
    encoding: str = "utf-8"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.pretokenization == PretokenizationMethod.CUSTOM_REGEX and not self.custom_regex:
            raise ValueError("custom_regex must be provided when using CUSTOM_REGEX pretokenization")
        
        if self.encoding not in ["utf-8", "hf"]:
            raise ValueError("encoding must be 'utf-8' or 'hf'")
        
        if self.encoding == "hf" and self.method != NormalizationMethod.BYTES:
            raise ValueError("HuggingFace encoding ('hf') can only be used with BYTES normalization method")
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NormalizationConfig':
        """Create NormalizationConfig from dictionary."""
        # Convert string values to enums
        if 'method' in config_dict:
            config_dict['method'] = NormalizationMethod(config_dict['method'])
        
        if 'pretokenization' in config_dict:
            config_dict['pretokenization'] = PretokenizationMethod(config_dict['pretokenization'])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert NormalizationConfig to dictionary."""
        return {
            'method': self.method.value,
            'pretokenization': self.pretokenization.value,
            'custom_regex': self.custom_regex,
            'include_empty_splits': self.include_empty_splits,
            'encoding': self.encoding
        }


class TextNormalizer:
    """Helper class for normalizing text according to configuration."""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self._hf_pretokenizer = None
        self._compiled_regex = None
        
        # Initialize pretokenizer if needed
        if self.config.method == NormalizationMethod.WORDS:
            self._initialize_pretokenizer()
        elif self.config.method == NormalizationMethod.BYTES and self.config.encoding == "hf":
            self._initialize_byte_pretokenizer()
    
    def _initialize_pretokenizer(self):
        """Initialize the pretokenizer based on configuration for word-level normalization."""
        if self.config.pretokenization == PretokenizationMethod.HUGGINGFACE_WHITESPACE:
            self._hf_pretokenizer = Whitespace()
        
        elif self.config.pretokenization == PretokenizationMethod.CUSTOM_REGEX:
            if not self.config.custom_regex:
                raise ValueError("custom_regex must be provided")
            self._compiled_regex = re.compile(self.config.custom_regex)
        
        elif self.config.pretokenization == PretokenizationMethod.REGEX_WHITESPACE:
            # Default whitespace regex pattern
            self._compiled_regex = re.compile(r'\s+')
    
    def _initialize_byte_pretokenizer(self):
        """Initialize the pretokenizer for HuggingFace byte-level normalization."""
        self._hf_pretokenizer = Sequence([Whitespace(), ByteLevel(use_regex=False)])
    
    def get_normalization_count(self, text: str) -> int:
        """Get the normalization count for a text according to the configuration."""
        if not text:
            return 0
        
        if self.config.method == NormalizationMethod.LINES:
            return 1  # Each text is considered one line
        
        elif self.config.method == NormalizationMethod.BYTES:
            if self.config.encoding == "hf":
                return self._get_hf_byte_count(text)
            else:
                return len(text.encode(self.config.encoding))
        
        elif self.config.method == NormalizationMethod.CHARACTERS:
            return len(text)
        
        elif self.config.method == NormalizationMethod.WORDS:
            return self._count_words(text)
        
        else:
            raise ValueError(f"Unknown normalization method: {self.config.method}")
    
    def _get_hf_byte_count(self, text: str) -> int:
        """Get byte count using HuggingFace ByteLevel pretokenizer."""
        if not text.strip():
            return 0
        
        # Use the HuggingFace ByteLevel pretokenizer to get byte representation
        pretokenized = self._hf_pretokenizer.pre_tokenize_str(text)
        
        # Count the total bytes represented by all pretokenized segments
        total_bytes = 0
        for token_str, _ in pretokenized:
            # ByteLevel pretokenizer produces tokens that represent bytes
            # Each character in the token string represents one byte
            total_bytes += len(token_str)
        
        return total_bytes
    
    def _count_words(self, text: str) -> int:
        """Count words in text according to pretokenization method."""
        if self.config.pretokenization == PretokenizationMethod.PYTHON_SPLIT:
            words = text.split()
            return len(words) if not self.config.include_empty_splits else len(text.split(' '))
        
        elif self.config.pretokenization == PretokenizationMethod.HUGGINGFACE_WHITESPACE:
            # Use HuggingFace pretokenizer
            pretokenized = self._hf_pretokenizer.pre_tokenize_str(text)
            return len(pretokenized)
        
        elif self.config.pretokenization in [PretokenizationMethod.REGEX_WHITESPACE, PretokenizationMethod.CUSTOM_REGEX]:
            # Split by regex pattern
            parts = self._compiled_regex.split(text)
            if not self.config.include_empty_splits:
                parts = [p for p in parts if p.strip()]
            return len(parts)
        
        else:
            raise ValueError(f"Unknown pretokenization method: {self.config.pretokenization}")
    
    def get_short_description(self) -> str:
        """Get short description for plot labels."""
        if self.config.method == NormalizationMethod.LINES:
            return "line"
        elif self.config.method == NormalizationMethod.BYTES:
            return "byte"
        elif self.config.method == NormalizationMethod.CHARACTERS:
            return "char"
        elif self.config.method == NormalizationMethod.WORDS:
            return "word"
        else:
            return f"{self.config.method.value}"


def create_default_configs() -> Dict[str, NormalizationConfig]:
    """Create a set of common normalization configurations."""
    return {
        'bytes': NormalizationConfig(
            method=NormalizationMethod.BYTES,
            encoding='utf-8'
        ),
        'bytes_hf': NormalizationConfig(
            method=NormalizationMethod.BYTES,
            encoding='hf'
        ),
        'characters': NormalizationConfig(
            method=NormalizationMethod.CHARACTERS
        ),
        'lines': NormalizationConfig(
            method=NormalizationMethod.LINES
        ),
        'words_python': NormalizationConfig(
            method=NormalizationMethod.WORDS,
            pretokenization=PretokenizationMethod.PYTHON_SPLIT
        ),
        'words_hf': NormalizationConfig(
            method=NormalizationMethod.WORDS,
            pretokenization=PretokenizationMethod.HUGGINGFACE_WHITESPACE
        ),
        'words_regex': NormalizationConfig(
            method=NormalizationMethod.WORDS,
            pretokenization=PretokenizationMethod.REGEX_WHITESPACE
        )
    }


# Default configuration for backward compatibility
DEFAULT_NORMALIZATION_CONFIG = create_default_configs()['bytes']
LINES_CONFIG = create_default_configs()['lines'] 