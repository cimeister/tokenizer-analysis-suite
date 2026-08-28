"""
Input provider implementations for raw and pre-tokenized data.
"""

from typing import Dict, List, TYPE_CHECKING
import dataclasses
import logging
import time
from .input_types import (
    encode_batch_timed,
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
                    # registered corpora. An earlier version of this comment
                    # said the two differ in two ways. They differ in at least
                    # these, and the first four move published numbers:
                    #
                    # 1. A tokenizer that cannot encode raw text: _encode_corpus
                    #    checks before encoding and skips it with a warning.
                    #    This loop has no such check, so whatever the wrapper
                    #    does propagates.
                    # 2. This loop records encode times, published as
                    #    encoding_speed; _encode_corpus records none, so a
                    #    derived corpus does not enter that measurement.
                    # 3. A provider that cannot supply the tokenizer object:
                    #    _encode_corpus catches that and skips the tokenizer.
                    #    Here spec.tokenizer is always present.
                    # 4. A tokenizer with no encode_batch_with_offsets: this
                    #    loop calls it regardless, _encode_corpus has a
                    #    per-text path.
                    # 5. Malformed ids from the batch: this loop checks the
                    #    type and names the language; _encode_corpus checks for
                    #    None ids and names the corpus and label.
                    # 6. The label passed to check_batch_pairing.
                    # 7. TokenizedData.metadata: populated here, empty there.
                    # 8. This loop caches its result and frees the source
                    #    texts afterwards; _encode_corpus does neither.
                    #
                    # Unifying them would change which tokenizers are skipped
                    # and what encoding_speed measures.
                    # The elapsed time covers the encode alone. Publishing a
                    # figure that also included the pairing check would inflate
                    # it by about a twentieth, and the comparison harness
                    # ignores this field by name, so nothing would catch it.
                    batch_results, batch_elapsed = encode_batch_timed(
                        spec.tokenizer.encode_batch_with_offsets, valid_texts,
                    )
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
        """Encoding time in seconds, one entry per text, for each tokenizer.

        Not a per-sample measurement despite there being one entry per sample:
        the texts of a language are encoded in one batch, and that batch's
        elapsed time divided by its size is written into every entry. Two
        entries from the same batch are therefore identical, and the spread
        within a language says nothing about individual texts.

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
            # PreTokenizedProvider raises for the same condition. Returning 0
            # sent a zero denominator into vocabulary_utilization, which guards
            # it to None, so the run reported nothing measurable for this
            # tokenizer instead of naming the one that cannot say how large it
            # is.
            raise ValueError(
                f"Cannot determine the vocabulary size of tokenizer "
                f"{tokenizer_name!r}: it has none of vocab_size, "
                "get_vocab_size or get_vocab. vocabulary_utilization and "
                "renyi_efficiency are both fractions of it, so the run would "
                "publish nothing measurable for this tokenizer and not say why."
            )
    
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
                    # Refused, not relabelled. This used to copy the record
                    # under the key's name at warning level, and every metric
                    # then scored one tokenizer's ids under another's name.
                    # Both name-agreement validators downstream read this
                    # method's output, so they saw the corrected name and
                    # nothing could catch the mismatch; only an id above the
                    # declared vocabulary size tripped anything at all.
                    # Which of the two names is wrong is not knowable here, so
                    # the error names both and leaves the choice to the caller.
                    raise ValueError(
                        f"The pre-tokenized data under key {tok_name!r} holds a "
                        f"{data.language!r} record labelled {data.tokenizer_name!r}. "
                        "Scoring it would report one tokenizer's ids under the "
                        "other's name. Correct the dump, or key it by the name "
                        "its records already carry."
                    )
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
            # sorted, not set order: an unordered list made per_language key
            # order hash-dependent, so two runs over one dump could differ.
            return sorted({data.language
                           for data in self.specifications[tokenizer_name].tokenized_data})
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