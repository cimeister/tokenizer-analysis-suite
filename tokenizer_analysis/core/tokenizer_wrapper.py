"""
Tokenizer wrapper classes for unified tokenizer interface.

This module provides a common interface for different tokenizer types,
making it easy for users to integrate custom tokenizers into the framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class TokenizerWrapper(ABC):
    """
    Abstract base class for tokenizer wrappers.
    
    This class defines the interface that all tokenizers must implement
    to work with the tokenizer analysis framework.
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """Get tokenizer name/identifier."""
        pass
    
    @abstractmethod 
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        pass
    
    @abstractmethod
    def get_vocab(self) -> Optional[Dict[str, int]]:
        """Get vocabulary mapping. Returns None if not available."""
        pass
    
    @abstractmethod
    def can_encode(self) -> bool:
        """Whether this tokenizer can encode raw text."""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs. 
        
        Args:
            text: Raw text to encode
            
        Returns:
            List of token IDs
            
        Raises:
            NotImplementedError: If can_encode() returns False
        """
        pass
    
    @abstractmethod
    def can_pretokenize(self) -> bool:
        """Whether this tokenizer supports pretokenization."""
        pass
        
    @abstractmethod
    def pretokenize(self, text: str) -> List[str]:
        """
        Pretokenize text into subword units.
        
        Args:
            text: Raw text to pretokenize
            
        Returns:
            List of pretokenized string pieces
            
        Raises:
            NotImplementedError: If can_pretokenize() returns False
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'TokenizerWrapper':
        """
        Factory method to create tokenizer from config.
        
        Args:
            name: Name/identifier for this tokenizer
            config: Configuration dictionary
            
        Returns:
            TokenizerWrapper instance
        """
        pass
    
    def get_underlying_tokenizer(self):
        """
        Get the underlying raw tokenizer object if available.
        
        Returns:
            The raw tokenizer object or None if not available.
            
        Note:
            This method is primarily for specialized use cases like MorphScore
            that require direct access to tokenizer internals.
        """
        return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get additional tokenizer metadata."""
        return {
            "name": self.get_name(), 
            "vocab_size": self.get_vocab_size(),
            "can_encode": self.can_encode(),
            "can_pretokenize": self.can_pretokenize()
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.get_name()}', vocab_size={self.get_vocab_size()})"


class HuggingFaceTokenizer(TokenizerWrapper):
    """Wrapper for HuggingFace tokenizers."""
    
    def __init__(self, name: str, tokenizer, config: Dict[str, Any]):
        """
        Initialize HuggingFace tokenizer wrapper.
        
        Args:
            name: Tokenizer name
            tokenizer: HuggingFace tokenizer instance
            config: Original configuration dict
        """
        self._name = name
        self._tokenizer = tokenizer
        self._config = config
    
    def get_name(self) -> str:
        return self._name
    
    def get_vocab_size(self) -> int:
        return len(self._tokenizer.get_vocab())
    
    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab()
    
    def can_encode(self) -> bool:
        return True
    
    def encode(self, text: str) -> List[int]:
        result = self._tokenizer.encode(text)
        # Handle different return types
        if hasattr(result, 'ids'):
            return result.ids
        elif isinstance(result, list):
            return result
        elif isinstance(result, dict) and 'input_ids' in result:
            return result['input_ids']
        else:
            raise ValueError(f"Unexpected encoding result type: {type(result)}")
    
    def can_pretokenize(self) -> bool:
        return hasattr(self._tokenizer, 'pre_tokenizer') and self._tokenizer.pre_tokenizer is not None
    
    def pretokenize(self, text: str) -> List[str]:
        if not self.can_pretokenize():
            raise NotImplementedError(f"Tokenizer {self._name} does not support pretokenization")
        return [token for token, _ in self._tokenizer.pre_tokenizer.pre_tokenize_str(text)]
    
    def get_underlying_tokenizer(self):
        """Return the underlying HuggingFace tokenizer object."""
        return self._tokenizer
    
    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'HuggingFaceTokenizer':
        """Create HuggingFace tokenizer wrapper from config."""
        # Import here to avoid circular import
        from ..utils.tokenizer_utils import _load_huggingface_tokenizer
        tokenizer = _load_huggingface_tokenizer(config)
        return cls(name, tokenizer, config)

class UniMixLMTokenizer(TokenizerWrapper):
    """Wrapper for HuggingFace tokenizers."""
    
    def __init__(self, name: str, tokenizer, config: Dict[str, Any]):
        """
        Initialize Unimixlm tokenizer wrapper.
        
        Args:
            name: Tokenizer name
            tokenizer: Unimix tokenizer instance
            config: Original configuration dict
        """
        self._name = name
        self._tokenizer = tokenizer
        self._config = config
    
    def get_name(self) -> str:
        return self._name
    
    def get_vocab_size(self) -> int:
        return len(self._tokenizer.get_vocab())
    
    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab()
    
    def can_encode(self) -> bool:
        return True
    
    def encode(self, text: str) -> List[int]:
        from unimixlm.code.utils import encode_text_minimal as unimix_encode_text
        return unimix_encode_text(self._tokenizer, text)["input_ids"]
    
    def can_pretokenize(self) -> bool:
        return hasattr(self._tokenizer.base_tokenizer, 'pre_tokenizer') and self._tokenizer.base_tokenizer.pre_tokenizer is not None
    
    def pretokenize(self, text: str) -> List[str]:
        if not self.can_pretokenize():
            raise NotImplementedError(f"Tokenizer {self._name} does not support pretokenization")
        return [token for token, _ in self._tokenizer.base_tokenizer.pre_tokenizer.pre_tokenize_str(text)]
    
    def get_underlying_tokenizer(self):
        """Return the underlying HuggingFace tokenizer object."""
        return self._tokenizer
    
    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'UniMixLMTokenizer':
        """Create tokenizer wrapper from config."""
        from unimixlm.code.utils import load_tokenizer_from_config as unimix_load_tokenizer_from_config
        unimixlm_config = config.copy()
        unimixlm_config["class"] = unimixlm_config.get("unimixlm_class", "standard")
        tokenizer = unimix_load_tokenizer_from_config(unimixlm_config)
        return cls(name, tokenizer, config)

class PreTokenizedDataTokenizer(TokenizerWrapper):
    """Tokenizer wrapper for pre-tokenized data scenarios."""
    
    def __init__(self, name: str, vocab_size: int, vocab_dict: Optional[Dict[str, int]] = None):
        """
        Initialize pre-tokenized data tokenizer wrapper.
        
        Args:
            name: Tokenizer name
            vocab_size: Size of vocabulary
            vocab_dict: Optional vocabulary mapping
        """
        self._name = name
        self._vocab_size = vocab_size
        self._vocab_dict = vocab_dict or {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_vocab_size(self) -> int:
        return self._vocab_size
    
    def get_vocab(self) -> Optional[Dict[str, int]]:
        return self._vocab_dict if self._vocab_dict else None
    
    def can_encode(self) -> bool:
        return False
    
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError("PreTokenizedDataTokenizer cannot encode raw text")
    
    def can_pretokenize(self) -> bool:
        return False
    
    def pretokenize(self, text: str) -> List[str]:
        raise NotImplementedError("PreTokenizedDataTokenizer cannot pretokenize text")
    
    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'PreTokenizedDataTokenizer':
        """Create pre-tokenized data tokenizer wrapper from config."""
        vocab_size = config.get('vocab_size')
        vocab_dict = config.get('vocab_dict')
        if vocab_size is None:
            raise ValueError("PreTokenizedDataTokenizer requires vocab_size in config")
        return cls(name, vocab_size, vocab_dict)


# Registry for custom tokenizer classes
_TOKENIZER_REGISTRY: Dict[str, type] = {
    'huggingface': HuggingFaceTokenizer,
    'hf': HuggingFaceTokenizer,
    'transformers': HuggingFaceTokenizer,
    'standard': HuggingFaceTokenizer,  # Legacy alias
    'pretokenized': PreTokenizedDataTokenizer,
    'unimixlm': UniMixLMTokenizer
}


def register_tokenizer_class(class_name: str, tokenizer_class: type) -> None:
    """
    Register a custom tokenizer class.
    
    Args:
        class_name: Name to use in configs
        tokenizer_class: TokenizerWrapper subclass
    """
    if not issubclass(tokenizer_class, TokenizerWrapper):
        raise ValueError("tokenizer_class must be a subclass of TokenizerWrapper")
    _TOKENIZER_REGISTRY[class_name] = tokenizer_class
    logger.info(f"Registered tokenizer class: {class_name} -> {tokenizer_class.__name__}")


def create_tokenizer_wrapper(name: str, config: Dict[str, Any]) -> TokenizerWrapper:
    """
    Factory function to create appropriate tokenizer wrapper from config.
    
    Args:
        name: Tokenizer name
        config: Configuration dictionary
        
    Returns:
        TokenizerWrapper instance
        
    Raises:
        ValueError: If tokenizer class is not recognized
    """
    tokenizer_class_name = config.get('class', 'huggingface')  # Default to HF
    if tokenizer_class_name == 'standard':
        logger.warning("The 'standard' tokenizers class is deprecated. Tokenizers labelled as such class "
                       "are assumed to be 'huggingface' tokenizers.")
    
    if tokenizer_class_name not in _TOKENIZER_REGISTRY:
        available_classes = list(_TOKENIZER_REGISTRY.keys())
        raise ValueError(f"Unknown tokenizer class: {tokenizer_class_name}. "
                        f"Available classes: {available_classes}")
    
    tokenizer_class = _TOKENIZER_REGISTRY[tokenizer_class_name]
    return tokenizer_class.from_config(name, config)