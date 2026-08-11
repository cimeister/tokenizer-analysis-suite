"""
Input data types and abstractions for tokenizer analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union, Protocol, TYPE_CHECKING
from abc import ABC, abstractmethod
import logging

if TYPE_CHECKING:
    from .tokenizer_wrapper import TokenizerWrapper

logger = logging.getLogger(__name__)


@dataclass
class TokenizedData:
    """Standardized format for tokenized text data."""
    
    tokenizer_name: str
    language: str
    tokens: List[int]
    text: Optional[str] = None  # Original text if available
    offsets: Optional[List[Tuple[int, int]]] = None  # Token-to-char offsets from encoding
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate tokenized data after initialization."""
        if not self.tokenizer_name:
            raise ValueError("tokenizer_name cannot be empty")
        if not self.language:
            raise ValueError("language cannot be empty")
        if not self.tokens:
            raise ValueError("tokens cannot be empty")
        if not isinstance(self.tokens, list) or not all(isinstance(t, int) for t in self.tokens):
            raise ValueError("tokens must be a list of integers")
        
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def token_count(self) -> int:
        """Get number of tokens."""
        return len(self.tokens)
    
    @property
    def unique_tokens(self) -> set:
        """Get set of unique token IDs."""
        return set(self.tokens)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'tokenizer_name': self.tokenizer_name,
            'language': self.language,
            'tokens': self.tokens,
            'text': self.text,
            'offsets': self.offsets,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenizedData':
        """Create TokenizedData from dictionary."""
        return cls(
            tokenizer_name=data['tokenizer_name'],
            language=data['language'],
            tokens=data['tokens'],
            text=data.get('text'),
            offsets=[tuple(o) for o in data['offsets']] if data.get('offsets') else None,
            metadata=data.get('metadata')
        )


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer objects."""
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        ...
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...


class VocabularyProvider(Protocol):
    """Protocol for objects that provide vocabulary information."""
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping (optional)."""
        ...


@dataclass
class InputSpecification:
    """Specification for input data to the analysis pipeline."""
    
    # For raw tokenization mode
    tokenizer: Optional['TokenizerWrapper'] = None
    texts: Optional[Dict[str, Union[str, List[str]]]] = None  # language -> text or list of texts
    
    # For pre-tokenized mode
    tokenizer_name: Optional[str] = None
    vocabulary: Optional[VocabularyProvider] = None  # Kept for backward compatibility, but tokenizer is preferred
    tokenized_data: Optional[List[TokenizedData]] = None
    
    # Common
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate input specification."""
        if self.metadata is None:
            self.metadata = {}
        
        # Validate mode consistency
        has_raw_inputs = self.tokenizer is not None and self.texts is not None
        has_tokenized_inputs = (self.tokenizer is not None and 
                               self.tokenized_data is not None)
        
        # Support legacy mode with tokenizer_name + vocabulary
        has_legacy_tokenized_inputs = (self.tokenizer_name is not None and 
                                     self.vocabulary is not None and 
                                     self.tokenized_data is not None)
        
        if not has_raw_inputs and not has_tokenized_inputs and not has_legacy_tokenized_inputs:
            raise ValueError(
                "Must provide either (tokenizer + texts) for raw mode or "
                "(tokenizer + tokenized_data) for pre-tokenized mode"
            )
        
        if has_raw_inputs and (has_tokenized_inputs or has_legacy_tokenized_inputs):
            raise ValueError(
                "Cannot provide both raw and pre-tokenized inputs simultaneously. "
                "Use separate InputSpecification objects."
            )
    
    @property
    def is_raw_mode(self) -> bool:
        """Check if this is raw tokenization mode."""
        return self.tokenizer is not None and self.texts is not None
    
    @property
    def is_pretokenized_mode(self) -> bool:
        """Check if this is pre-tokenized mode."""
        return (self.tokenizer is not None and self.tokenized_data is not None) or \
               (self.tokenizer_name is not None and 
                self.vocabulary is not None and 
                self.tokenized_data is not None)
    
    def get_tokenizer_name(self) -> str:
        """Get tokenizer name for both modes."""
        if self.is_raw_mode:
            return self.tokenizer.get_name() if hasattr(self.tokenizer, 'get_name') else getattr(self.tokenizer, 'name', 'unknown')
        elif self.tokenizer is not None:
            return self.tokenizer.get_name()
        else:
            return self.tokenizer_name or 'unknown'
    
    def get_languages(self) -> List[str]:
        """Get list of languages in this specification."""
        if self.is_raw_mode:
            return list(self.texts.keys())
        else:
            return list(set(td.language for td in self.tokenized_data))
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size, from whichever source this specification carries.

        Both branches were wrong. The raw branch read ``tokenizer.vocab_size``,
        an attribute ``TokenizerWrapper`` does not have (it exposes
        ``get_vocab_size()``), and the pre-tokenized branch read
        ``self.vocabulary``, which is None for the ``tokenizer +
        tokenized_data`` shape that
        ``main.create_analyzer_from_tokenized_data`` builds. So the method
        raised ``AttributeError`` on every specification the package itself
        constructs. The tokenizer is the preferred source in both modes, with
        the legacy ``vocabulary`` provider, whose protocol declares
        ``vocab_size`` as a property, as the fallback.
        """
        if self.tokenizer is not None:
            return self.tokenizer.get_vocab_size()
        if self.vocabulary is not None:
            return self.vocabulary.vocab_size
        raise ValueError(
            "This InputSpecification carries neither a tokenizer nor a "
            "vocabulary provider, so its vocabulary size cannot be read."
        )


class InputProvider(ABC):
    """Abstract base class for providing tokenized data to analysis pipeline."""
    
    @abstractmethod
    def get_tokenized_data(self) -> Dict[str, List[TokenizedData]]:
        """
        Get tokenized data organized by tokenizer name.
        
        Returns:
            Dictionary mapping tokenizer names to lists of TokenizedData objects
        """
        pass
    
    @abstractmethod
    def get_tokenizer_names(self) -> List[str]:
        """Get list of tokenizer names."""
        pass
    
    @abstractmethod
    def get_vocab_size(self, tokenizer_name: str) -> int:
        """Get vocabulary size for a tokenizer."""
        pass
    
    @abstractmethod
    def get_languages(self, tokenizer_name: str = None) -> List[str]:
        """Get list of languages. If tokenizer_name is None, return all languages."""
        pass
    
    def validate_data(self) -> bool:
        """Validate the provided data."""
        try:
            tokenized_data = self.get_tokenized_data()
            tokenizer_names = self.get_tokenizer_names()
            
            # Check consistency
            if set(tokenized_data.keys()) != set(tokenizer_names):
                logger.error("Mismatch between tokenized_data keys and tokenizer_names")
                return False
            
            # Validate each tokenizer's data
            for tok_name, data_list in tokenized_data.items():
                if not data_list:
                    logger.warning(f"No data for tokenizer {tok_name}")
                    continue
                
                for data in data_list:
                    if not isinstance(data, TokenizedData):
                        logger.error(f"Invalid data type for {tok_name}: {type(data)}")
                        return False
                    
                    if data.tokenizer_name != tok_name:
                        logger.error(f"Tokenizer name mismatch: expected {tok_name}, got {data.tokenizer_name}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False

def check_batch_pairing(tokenizer_name: str, language: str, texts, encoded, corpus: str) -> None:
    """Raise unless a batch result has one entry per text it was given.

    Both corpora pair a batch with its texts by position, which is the batch
    API's contract. Nothing verifies it. A backend returning fewer results than
    it was given would otherwise be consumed by ``zip``, which stops at the
    shorter side: the trailing texts vanish and every remaining pairing is
    still correct, so the loss is invisible in the output.

    This checks the count and nothing else. A backend that returned the right
    number of encodings in a different order would attach one text's offsets to
    another text, and neither this nor anything else in the pipeline catches
    that.

    Args:
        tokenizer_name: named in the error, so a multi-tokenizer run says which.
        language: named in the error for the same reason.
        texts: what was handed to the backend.
        encoded: what the backend returned.
        corpus: which corpus this is, for the error message.

    Raises:
        ValueError: naming both counts.
    """
    if len(encoded) == len(texts):
        return
    raise ValueError(
        f"Tokenizer {tokenizer_name!r} returned {len(encoded)} encodings for "
        f"the {len(texts)} {language!r} texts of the {corpus}. Pairing them by "
        "position would attach one text's offsets to another text, so the "
        "metric would be computed against the wrong source. Only the count is "
        "checked: a backend that reordered a batch of the right length would "
        "not be caught here."
    )
