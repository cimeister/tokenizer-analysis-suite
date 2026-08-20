"""
Input provider implementations for raw and pre-tokenized data.
"""

from typing import Dict, List, TYPE_CHECKING
import dataclasses
import logging
import time
from .input_types import (
    InputProvider, TokenizedData, InputSpecification,
    VocabularyProvider, check_batch_pairing
)

if TYPE_CHECKING:
    from .tokenizer_wrapper import TokenizerWrapper

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
        # Copy each specification. get_tokenized_data() sets ``texts = None`` on
        # them once the texts have been encoded, to release the corpus; done on
        # the caller's own objects that leaves each one neither raw nor
        # pre-tokenized, so its own is_raw_mode is False and
        # spec.get_languages() raises TypeError on the None. dataclasses.replace
        # re-runs __post_init__, so the copies are validated on construction as
        # well. The texts dict itself is shared rather than deep-copied: only
        # the attribute is rebound here, never the dict's contents.
        self.specifications = {
            name: dataclasses.replace(spec) for name, spec in specifications.items()
        }
        self._validate_specifications()
        self._tokenized_cache = {}
        self._encode_times: Dict[str, List[float]] = {}  # tok_name -> per-sample seconds
    
    def _validate_specifications(self):
        """Validate that all specifications are in raw mode."""
        for name, spec in self.specifications.items():
            if not spec.is_raw_mode:
                raise ValueError(f"Specification for {name} is not in raw mode")
    
    def get_tokenized_data(self) -> Dict[str, List[TokenizedData]]:
        """Get tokenized data by tokenizing this provider's own raw texts.

        Registered corpora are read through ``get_corpus_data``.
        """
        if self._tokenized_cache:
            return self._tokenized_cache
        
        tokenized_data = {}
        
        for tok_name, spec in self.specifications.items():
            tokenized_data[tok_name] = []
            self._encode_times[tok_name] = []
            logger.info(f"Tokenizing data for {tok_name} tokenizer...")
            for language, text_data in spec.texts.items():
                try:
                    # Handle both single strings and lists of strings
                    if isinstance(text_data, str):
                        text_list = [text_data]
                    elif isinstance(text_data, list):
                        text_list = text_data
                    else:
                        logger.error(f"Text for {language} is neither string nor list: {type(text_data)} - {text_data}")
                        raise ValueError(f"Text for {language} must be a string or list of strings, got {type(text_data)}")

                    # Filter to valid texts
                    valid_texts = []
                    for text in text_list:
                        if not isinstance(text, str):
                            logger.error(f"Text item for {language} is not a string: {type(text)} - {text}")
                            raise ValueError(f"Text item for {language} must be a string, got {type(text)}")
                        if not text.strip():
                            logger.debug(f"Empty text for {language}, skipping")
                            continue
                        valid_texts.append(text)

                    if not valid_texts:
                        continue

                    # Batch-encode all valid texts for this language.
                    #
                    # This loop is deliberately not
                    # InputProvider._encode_corpus, which encodes the
                    # registered corpora. The two differ in two ways, both of
                    # which move published numbers:
                    #
                    # 1. This one raises when a tokenizer cannot encode a text;
                    #    _encode_corpus skips that tokenizer with a warning.
                    # 2. This one records per-sample encode times, published as
                    #    encoding_speed; _encode_corpus records none, so a
                    #    derived corpus does not enter that measurement.
                    #
                    # Unifying them would change which tokenizers are skipped
                    # and what encoding_speed measures.
                    t0 = time.perf_counter()
                    batch_results = spec.tokenizer.encode_batch_with_offsets(valid_texts)
                    batch_elapsed = time.perf_counter() - t0

                    per_sample_time = batch_elapsed / len(valid_texts)
                    self._encode_times[tok_name].extend(
                        [per_sample_time] * len(valid_texts)
                    )

                    check_batch_pairing(
                        tok_name, language, valid_texts, batch_results,
                        "prose corpus",
                    )
                    for text, (tokens, offsets) in zip(valid_texts, batch_results):
                        if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
                            logger.error(f"Tokens for {language} are not a list of integers: {type(tokens)} - {tokens}")
                            raise ValueError(f"Tokens for {language} must be a list of integers, got {type(tokens)}")

                        data = TokenizedData(
                            tokenizer_name=tok_name,
                            language=language,
                            tokens=tokens,
                            text=text,
                            offsets=offsets,
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
        
        # Cache language lists before freeing raw texts, since
        # get_languages() reads from spec.texts.
        self._languages_cache = {}
        for tok_name, spec in self.specifications.items():
            if spec.texts is not None:
                self._languages_cache[tok_name] = list(spec.texts.keys())
                spec.texts = None

        self._tokenized_cache = tokenized_data
        return tokenized_data
    
    @property
    def encode_times(self) -> Dict[str, List[float]]:
        """Per-sample encoding times (seconds) for each tokenizer.

        Populated after ``get_tokenized_data()`` has been called.
        """
        return self._encode_times

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
        cache = getattr(self, '_languages_cache', None)
        if tokenizer_name:
            if tokenizer_name not in self.specifications:
                raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
            if cache and tokenizer_name in cache:
                return list(cache[tokenizer_name])
            return list(self.specifications[tokenizer_name].texts.keys())
        else:
            all_languages = set()
            if cache:
                for langs in cache.values():
                    all_languages.update(langs)
            else:
                for spec in self.specifications.values():
                    all_languages.update(spec.texts.keys())
            return sorted(list(all_languages))
    
    def get_tokenizer(self, tokenizer_name: str) -> 'TokenizerWrapper':
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
        """Get the pre-tokenized data the specifications carry.

        Registered corpora are read through ``get_corpus_data``. Encoding one
        needs a tokenizer that accepts raw text, which pre-tokenized input does
        not always supply, so tokenizers without one are left out there.
        """
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
                        offsets=data.offsets,
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
        
        spec = self.specifications[tokenizer_name]
        
        # Try tokenizer first (new way)
        if spec.tokenizer is not None:
            return spec.tokenizer.get_vocab_size()
        # Fall back to vocabulary (legacy way)
        elif spec.vocabulary is not None:
            return spec.vocabulary.vocab_size
        else:
            raise ValueError(f"No vocabulary information available for tokenizer {tokenizer_name}")
    
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
        spec = self.specifications[tokenizer_name]
        # Return tokenizer if available (new way), otherwise vocabulary (legacy)
        return spec.tokenizer if spec.tokenizer is not None else spec.vocabulary
    
    def get_tokenizer(self, tokenizer_name: str) -> 'TokenizerWrapper':
        """Get tokenizer object (useful for additional operations)."""
        if tokenizer_name not in self.specifications:
            raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
        spec = self.specifications[tokenizer_name]
        if spec.tokenizer is not None:
            return spec.tokenizer
        else:
            raise ValueError(f"No tokenizer wrapper available for {tokenizer_name} (legacy mode)")


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
        # Combining the two modes in one run is unsupported. The provider that
        # did it was never constructed: the CLI selects one mode for the whole
        # run, and nothing else builds a mixed specification set. A run that
        # reached here would mix numbers measured by encoding text with numbers
        # measured from ids somebody else produced, with nothing in the output
        # saying which tokenizer came from which.
        raise ValueError(
            "Specifications mix raw text and pre-tokenized input: raw "
            f"{sorted(raw_specs)}, pre-tokenized {sorted(pretokenized_specs)}. "
            "One run analyses one mode. Build a separate provider for each."
        )
    elif raw_specs:
        return RawTokenizationProvider(raw_specs)
    elif pretokenized_specs:
        return PreTokenizedProvider(pretokenized_specs)
    else:
        raise ValueError("No valid specifications provided")