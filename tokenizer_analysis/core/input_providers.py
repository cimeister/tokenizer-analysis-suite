"""
Input provider implementations for raw and pre-tokenized data.
"""

from typing import Dict, List, Any, Union, Optional
import logging
from .input_types import (
    InputProvider, TokenizedData, InputSpecification, 
    TokenizerProtocol, VocabularyProvider
)
from ..utils import encode_text

logger = logging.getLogger(__name__)


class RawTokenizationProvider(InputProvider):
    """Provider that tokenizes raw text on demand."""
    
    def __init__(self, specifications: Dict[str, InputSpecification]):
        """
        Initialize with raw tokenization specifications.
        
        Args:
            specifications: Dict mapping tokenizer names to InputSpecification objects
                           (all must be in raw mode)
        """
        self.specifications = specifications
        self._validate_specifications()
        self._tokenized_cache = {}
    
    def _validate_specifications(self):
        """Validate that all specifications are in raw mode."""
        for name, spec in self.specifications.items():
            if not spec.is_raw_mode:
                raise ValueError(f"Specification for {name} is not in raw mode")
    
    def get_tokenized_data(self) -> Dict[str, List[TokenizedData]]:
        """Get tokenized data by tokenizing raw texts."""
        if self._tokenized_cache:
            return self._tokenized_cache
        
        tokenized_data = {}
        
        for tok_name, spec in self.specifications.items():
            tokenized_data[tok_name] = []
            logger.info(f"Tokenizing data for {tok_name} tokenizer...")
            for language, text_data in spec.texts.items():
                try:
                    # Handle both single strings and lists of strings
                    if isinstance(text_data, str):
                        # Single string
                        text_list = [text_data]
                    elif isinstance(text_data, list):
                        # List of strings
                        text_list = text_data
                    else:
                        logger.error(f"Text for {language} is neither string nor list: {type(text_data)} - {text_data}")
                        raise ValueError(f"Text for {language} must be a string or list of strings, got {type(text_data)}")
                    
                    # Process each text in the list
                    for text in text_list:
                        # Validate text is a string
                        if not isinstance(text, str):
                            logger.error(f"Text item for {language} is not a string: {type(text)} - {text}")
                            raise ValueError(f"Text item for {language} must be a string, got {type(text)}")
                        
                        # Ensure text is not empty
                        if not text.strip():
                            logger.debug(f"Empty text for {language}, skipping")
                            continue
                        
                        # Tokenize the text
                        tokens_raw = encode_text(spec.tokenizer, text) #spec.tokenizer.encode(text)
                        
                        # Ensure tokens is a list of integers
                        if hasattr(tokens_raw, 'ids'):
                            # Handle tokenizers library Encoding object
                            tokens = tokens_raw.ids
                        elif isinstance(tokens_raw, dict) and "input_ids" in tokens_raw:
                            tokens = tokens_raw["input_ids"]
                        elif isinstance(tokens_raw, list):
                            tokens = tokens_raw
                        else:
                            logger.error(f"Unexpected token format from tokenizer {tok_name}: {type(tokens_raw)} - {tokens_raw}")
                            raise ValueError(f"Tokenizer {tok_name} returned unexpected format: {type(tokens_raw)}")
                        
                        # Validate tokens are integers
                        if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
                            logger.error(f"Tokens for {language} are not a list of integers: {type(tokens)} - {tokens}")
                            raise ValueError(f"Tokens for {language} must be a list of integers, got {type(tokens)}")
                        
                        # Create TokenizedData object
                        data = TokenizedData(
                            tokenizer_name=tok_name,
                            language=language,
                            tokens=tokens,
                            text=text,
                            metadata={
                                'source': 'raw_tokenization',
                                'tokenizer_metadata': spec.metadata,
                                'text_length': len(text)
                            }
                        )
                        
                        tokenized_data[tok_name].append(data)
                        logger.debug(f"Tokenized {language} text for {tok_name}: {len(tokens)} tokens")
                    
                except Exception as e:
                    logger.error(f"Error tokenizing {language} text for {tok_name}: {e}")
                    raise
        
        self._tokenized_cache = tokenized_data
        return tokenized_data
    
    def get_tokenizer_names(self) -> List[str]:
        """Get list of tokenizer names."""
        return list(self.specifications.keys())
    
    def get_vocab_size(self, tokenizer_name: str) -> int:
        """Get vocabulary size for a tokenizer."""
        if tokenizer_name not in self.specifications:
            raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
        
        tokenizer = self.specifications[tokenizer_name].tokenizer
        
        # Handle different tokenizer types
        if hasattr(tokenizer, 'vocab_size'):
            return tokenizer.vocab_size
        elif hasattr(tokenizer, 'get_vocab_size'):
            return tokenizer.get_vocab_size()
        elif hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
            return len(vocab) if vocab else 0
        else:
            logger.warning(f"Cannot determine vocab size for tokenizer {tokenizer_name}")
            return 0
    
    def get_languages(self, tokenizer_name: str = None) -> List[str]:
        """Get list of languages."""
        if tokenizer_name:
            if tokenizer_name not in self.specifications:
                raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
            return list(self.specifications[tokenizer_name].texts.keys())
        else:
            # Return all unique languages across all tokenizers
            all_languages = set()
            for spec in self.specifications.values():
                all_languages.update(spec.texts.keys())
            return sorted(list(all_languages))
    
    def get_tokenizer(self, tokenizer_name: str) -> TokenizerProtocol:
        """Get tokenizer object (useful for additional operations)."""
        if tokenizer_name not in self.specifications:
            raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
        return self.specifications[tokenizer_name].tokenizer


class PreTokenizedProvider(InputProvider):
    """Provider for pre-tokenized data."""
    
    def __init__(self, specifications: Dict[str, InputSpecification]):
        """
        Initialize with pre-tokenized specifications.
        
        Args:
            specifications: Dict mapping tokenizer names to InputSpecification objects
                           (all must be in pre-tokenized mode)
        """
        self.specifications = specifications
        self._validate_specifications()
    
    def _validate_specifications(self):
        """Validate that all specifications are in pre-tokenized mode."""
        for name, spec in self.specifications.items():
            if not spec.is_pretokenized_mode:
                raise ValueError(f"Specification for {name} is not in pre-tokenized mode")
    
    def get_tokenized_data(self) -> Dict[str, List[TokenizedData]]:
        """Get pre-tokenized data."""
        tokenized_data = {}
        
        for tok_name, spec in self.specifications.items():
            # Validate that all tokenized data has correct tokenizer name
            validated_data = []
            for data in spec.tokenized_data:
                if data.tokenizer_name != tok_name:
                    logger.warning(
                        f"Tokenizer name mismatch: expected {tok_name}, "
                        f"got {data.tokenizer_name}. Correcting..."
                    )
                    # Create corrected copy
                    corrected_data = TokenizedData(
                        tokenizer_name=tok_name,
                        language=data.language,
                        tokens=data.tokens,
                        text=data.text,
                        metadata=data.metadata
                    )
                    validated_data.append(corrected_data)
                else:
                    validated_data.append(data)
            
            tokenized_data[tok_name] = validated_data
        
        return tokenized_data
    
    def get_tokenizer_names(self) -> List[str]:
        """Get list of tokenizer names."""
        return list(self.specifications.keys())
    
    def get_vocab_size(self, tokenizer_name: str) -> int:
        """Get vocabulary size for a tokenizer."""
        if tokenizer_name not in self.specifications:
            raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
        return self.specifications[tokenizer_name].vocabulary.vocab_size
    
    def get_languages(self, tokenizer_name: str = None) -> List[str]:
        """Get list of languages."""
        if tokenizer_name:
            if tokenizer_name not in self.specifications:
                raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
            return list(set(data.language for data in self.specifications[tokenizer_name].tokenized_data))
        else:
            # Return all unique languages across all tokenizers
            all_languages = set()
            for spec in self.specifications.values():
                all_languages.update(data.language for data in spec.tokenized_data)
            return sorted(list(all_languages))
    
    def get_vocabulary(self, tokenizer_name: str) -> VocabularyProvider:
        """Get vocabulary provider (useful for additional operations)."""
        if tokenizer_name not in self.specifications:
            raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
        return self.specifications[tokenizer_name].vocabulary


class MixedInputProvider(InputProvider):
    """Provider that combines raw and pre-tokenized data."""
    
    def __init__(self, 
                 raw_specifications: Optional[Dict[str, InputSpecification]] = None,
                 pretokenized_specifications: Optional[Dict[str, InputSpecification]] = None):
        """
        Initialize with mixed specifications.
        
        Args:
            raw_specifications: Raw tokenization specs
            pretokenized_specifications: Pre-tokenized specs
        """
        self.raw_provider = None
        self.pretokenized_provider = None
        
        if raw_specifications:
            self.raw_provider = RawTokenizationProvider(raw_specifications)
        
        if pretokenized_specifications:
            self.pretokenized_provider = PreTokenizedProvider(pretokenized_specifications)
        
        if not self.raw_provider and not self.pretokenized_provider:
            raise ValueError("Must provide at least one type of specification")
        
        # Check for tokenizer name conflicts
        raw_names = set(raw_specifications.keys()) if raw_specifications else set()
        pretokenized_names = set(pretokenized_specifications.keys()) if pretokenized_specifications else set()
        
        conflicts = raw_names & pretokenized_names
        if conflicts:
            raise ValueError(f"Tokenizer name conflicts between raw and pre-tokenized: {conflicts}")
    
    def get_tokenized_data(self) -> Dict[str, List[TokenizedData]]:
        """Get combined tokenized data."""
        combined_data = {}
        
        if self.raw_provider:
            combined_data.update(self.raw_provider.get_tokenized_data())
        
        if self.pretokenized_provider:
            combined_data.update(self.pretokenized_provider.get_tokenized_data())
        
        return combined_data
    
    def get_tokenizer_names(self) -> List[str]:
        """Get list of all tokenizer names."""
        names = []
        
        if self.raw_provider:
            names.extend(self.raw_provider.get_tokenizer_names())
        
        if self.pretokenized_provider:
            names.extend(self.pretokenized_provider.get_tokenizer_names())
        
        return names
    
    def get_vocab_size(self, tokenizer_name: str) -> int:
        """Get vocabulary size for a tokenizer."""
        if self.raw_provider and tokenizer_name in self.raw_provider.get_tokenizer_names():
            return self.raw_provider.get_vocab_size(tokenizer_name)
        elif self.pretokenized_provider and tokenizer_name in self.pretokenized_provider.get_tokenizer_names():
            return self.pretokenized_provider.get_vocab_size(tokenizer_name)
        else:
            raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
    
    def get_languages(self, tokenizer_name: str = None) -> List[str]:
        """Get list of languages."""
        if tokenizer_name:
            if self.raw_provider and tokenizer_name in self.raw_provider.get_tokenizer_names():
                return self.raw_provider.get_languages(tokenizer_name)
            elif self.pretokenized_provider and tokenizer_name in self.pretokenized_provider.get_tokenizer_names():
                return self.pretokenized_provider.get_languages(tokenizer_name)
            else:
                raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
        else:
            # Return all unique languages across all providers
            all_languages = set()
            
            if self.raw_provider:
                all_languages.update(self.raw_provider.get_languages())
            
            if self.pretokenized_provider:
                all_languages.update(self.pretokenized_provider.get_languages())
            
            return sorted(list(all_languages))


def create_input_provider(specifications: Dict[str, InputSpecification]) -> InputProvider:
    """
    Factory function to create appropriate InputProvider based on specifications.
    
    Args:
        specifications: Dict mapping tokenizer names to InputSpecification objects
        
    Returns:
        Appropriate InputProvider instance
    """
    raw_specs = {}
    pretokenized_specs = {}
    
    for name, spec in specifications.items():
        if spec.is_raw_mode:
            raw_specs[name] = spec
        elif spec.is_pretokenized_mode:
            pretokenized_specs[name] = spec
        else:
            raise ValueError(f"Invalid specification for {name}: neither raw nor pre-tokenized mode")
    
    if raw_specs and pretokenized_specs:
        return MixedInputProvider(raw_specs, pretokenized_specs)
    elif raw_specs:
        return RawTokenizationProvider(raw_specs)
    elif pretokenized_specs:
        return PreTokenizedProvider(pretokenized_specs)
    else:
        raise ValueError("No valid specifications provided")