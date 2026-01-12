"""
Tokenizer wrapper classes for unified tokenizer interface.

This module provides a common interface for different tokenizer types,
making it easy for users to integrate custom tokenizers into the framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import os
import glob
import json

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

    def get_unk_token_id(self) -> Optional[int]:
        """
        Get the UNK token ID if available.

        Returns:
            The UNK token ID or None if not available.
        """
        return None

    def has_unk_token(self) -> bool:
        """
        Check if tokenizer has an UNK token.

        Returns:
            True if tokenizer has an UNK token, False otherwise.
        """
        return self.get_unk_token_id() is not None
    
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
        logger.info("Creating HF tokenizer")
    
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

    def get_unk_token_id(self) -> Optional[int]:
        """Get the UNK token ID from HuggingFace tokenizer."""
        # Try direct access to unk_token_id
        if hasattr(self._tokenizer, 'unk_token_id'):
            return self._tokenizer.unk_token_id

        # Try getting it through the vocabulary
        vocab = self.get_vocab()
        if vocab:
            unk_candidates = ['<unk>', '[UNK]', '<UNK>', 'unk', 'UNK', '⁇']
            for candidate in unk_candidates:
                if candidate in vocab:
                    return vocab[candidate]

        # Try through unk_token and token_to_id
        if hasattr(self._tokenizer, 'unk_token') and hasattr(self._tokenizer, 'token_to_id'):
            if self._tokenizer.unk_token:
                return self._tokenizer.token_to_id(self._tokenizer.unk_token)

        return None

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
        self.tokenizer_class = self._config.get('unimixlm_class')
        if self.tokenizer_class == 'langspec':
            from tokenizers import Tokenizer
            self.per_lang_tok = {}
            vocab_tuples = self._get_hf_unigram_tokenizer_vocab(self._tokenizer)
            base_vocab_order = [i for i, j in vocab_tuples]
            language_paths = config.get('language_paths')
            for lang_code, lang_tok_path in language_paths.items():
                tok = Tokenizer.from_file(lang_tok_path)
                vocab_order_per_lang = self._get_hf_unigram_tokenizer_vocab(tok)
                assert [i for i, j in vocab_order_per_lang] == base_vocab_order
                self.per_lang_tok[lang_code] = {
                    "tokenizer": tok,
                    "scores": self._extract_log_scores(tok),
                    "path": lang_tok_path,
                } 
        logger.info("Creating UnimixLM tokenizer")
    
    @staticmethod
    def _get_hf_unigram_tokenizer_vocab(tokenizer):
        state = tokenizer.model.__getstate__()
        attributes = json.loads(state.decode("utf-8"))
        vocab = attributes["vocab"]
        if isinstance(vocab, dict):
            # This is the format the BPE vocab comes in
            tuples = sorted(vocab.items(), key=lambda x: x[1])
            return [(tk, -1) for tk, idx in tuples]
        return vocab
        
    @staticmethod
    def _extract_log_scores(tok) -> Dict[str, float]:
        """
        Return dict {token -> log_score} from a HF Unigram tokenizer.
        """
        vocab_tuples = UniMixLMTokenizer._get_hf_unigram_tokenizer_vocab(tok)
        return {tok_id: tok_tuple[1] for tok_id, tok_tuple in enumerate(vocab_tuples) }
        
    def get_name(self) -> str:
        return self._name
    
    def get_vocab_size(self) -> int:
        return len(self._tokenizer.get_vocab())
    
    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab()
    
    def can_encode(self) -> bool:
        return True
    
    def encode(self, text: str) -> List[int]:
        if self.tokenizer_class == 'langspec':
            best_lang, best_tokens, best_logp = None, [], float("-inf")

            for lang_code, info in self.per_lang_tok.items():
                enc = info["tokenizer"].encode(text)
                tok_ids = enc.ids
                logp = sum(info["scores"][t] for t in tok_ids)
    
                if logp > best_logp:
                    best_lang, best_tokens, best_logp = lang_code, tok_ids, logp
    
            return best_tokens
            
        else:
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
        from tokenizers import Tokenizer
        tokenizer_class = config.get('unimixlm_class')
        if tokenizer_class is not None:
            tokenizer = Tokenizer.from_file(config['path'])
        else:
            # Try loading as a HuggingFace tokenizer
            try: 
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config['path'])
            except Exception as e:
                logger.info(f"Tried to load tokenizer via default method and could not {e}")
                raise

        return cls(name, tokenizer, config)


class SentencePieceTokenizer(TokenizerWrapper):
    """Wrapper for SentencePiece tokenizers."""

    def __init__(self, name: str, sp_processor: "spm.SentencePieceProcessor", config: Dict[str, Any]):
        """
        Initialize SentencePiece tokenizer wrapper.

        Args:
            name: Tokenizer name
            sp_processor: sentencepiece.SentencePieceProcessor instance
            config: Original configuration dict
        """
        self._name = name
        self._sp = sp_processor
        self._config = config or {}
        # Optional flags (default False)
        self._add_bos = bool(self._config.get("add_bos", False))
        self._add_eos = bool(self._config.get("add_eos", False))

        logger.info("Creating SentencePiece tokenizer")

    def get_name(self) -> str:
        return self._name

    def get_vocab_size(self) -> int:
        return int(self._sp.get_piece_size())

    def get_vocab(self) -> Dict[str, int]:
        size = self._sp.get_piece_size()
        return {self._sp.id_to_piece(i): i for i in range(size)}

    def can_encode(self) -> bool:
        return True

    def encode(self, text: str) -> List[int]:
        # Return list of ids; optionally prepend/append BOS/EOS if configured and defined
        ids = self._sp.encode(text, out_type=int)

        if self._add_bos:
            bos = self._sp.bos_id()
            if bos is not None and bos >= 0:
                ids = [bos] + ids

        if self._add_eos:
            eos = self._sp.eos_id()
            if eos is not None and eos >= 0:
                ids = ids + [eos]

        return ids

    def can_pretokenize(self) -> bool:
        return True

    def pretokenize(self, text: str) -> List[str]:
        # Pieces correspond to subword tokens (e.g., "▁The", "re")
        pieces = self._sp.encode(text, out_type=str)
        pretokens: List[str] = []
        current = ""
    
        for p in pieces:
            if p.startswith("▁"):
                # flush previous
                if current:
                    pretokens.append(current)
                # start new (strip the boundary marker)
                current = p[1:]
            else:
                # continuation of the current pretoken
                current += p
    
        if current:
            pretokens.append(current)
    
        # NOTE: these are "normalized words" per SP's normalization rules,
        return pretokens

    def get_underlying_tokenizer(self):
        """Return the underlying SentencePieceProcessor object."""
        return self._sp

    def get_unk_token_id(self) -> Optional[int]:
        """Get the UNK token ID from SentencePiece tokenizer."""
        # SentencePiece exposes unk_id(); returns -1 if undefined
        try:
            unk_id = self._sp.unk_id()
            if unk_id is not None and unk_id >= 0:
                return int(unk_id)
        except Exception:
            pass

        # Fallbacks: check common UNK pieces in the vocab
        vocab = self.get_vocab()
        for candidate in ['<unk>', '[UNK]', '<UNK>', 'unk', 'UNK', '⁇']:
            if candidate in vocab:
                return vocab[candidate]

        # Last-ditch: ask processor to map a likely token; if unknown, it should map to unk
        try:
            return int(self._sp.piece_to_id("<unk>"))
        except Exception:
            return None

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> "SentencePieceTokenizer":
        """
        Internal function to load a SentencePiece tokenizer from configuration.
    
        Expected config keys:
          - path: path to a .model file OR a directory containing a .model
          - (optional) model_filename: explicit filename to prefer inside a directory
        """
        try:
            import sentencepiece as spm  # lazy import here
        except ImportError as e:
            raise RuntimeError(
                "sentencepiece is required to build SentencePieceTokenizer "
                "from model files. Install with `pip install sentencepiece`."
            ) from e
        sp = None
        if "path" not in config:
            raise ValueError("config must include 'path' to the SentencePiece model (.model or directory)")
    
        path = config["path"]
        prefer_filename = config.get("model_filename")  # optional: e.g., "sp.model"
    
        # Helper: create the processor
        def _init_from_model_file(model_file: str) -> spm.SentencePieceProcessor:
            logger.info(f"Loading SentencePiece model from: {model_file}")
            sp = spm.SentencePieceProcessor()
            # Newer sentencepiece supports load() and constructor arg model_file=
            # Using load() keeps compatibility.
            if not os.path.isfile(model_file):
                raise FileNotFoundError(f"SentencePiece model file not found: {model_file}")
            loaded = sp.load(model_file)
            if not loaded:
                # Some versions return False on failure
                raise RuntimeError(f"SentencePieceProcessor.load failed for: {model_file}")
            return sp
    
        # Strategy 1: Direct path to a model file
        if os.path.isfile(path) and path.endswith(".model"):
            try:
                sp = _init_from_model_file(path)
            except Exception as e:
                logger.warning(f"Failed to load SentencePiece model from file {path}: {e}")
    
        # Strategy 2: Directory containing a model
        if os.path.isdir(path):
            candidates: List[str] = []
    
            # If user provided a preferred model filename, try that first
            if prefer_filename:
                preferred = os.path.join(path, prefer_filename)
                if os.path.isfile(preferred):
                    candidates.append(preferred)
    
            # Common names often used
            common_names = ["sp.model", "sentencepiece.model", "tokenizer.model", "model.model"]
            for name in common_names:
                p = os.path.join(path, name)
                if os.path.isfile(p):
                    candidates.append(p)
    
            # Any *.model in the directory as fallback (sorted for determinism)
            globbed = sorted(glob.glob(os.path.join(path, "*.model")))
            for p in globbed:
                if p not in candidates:
                    candidates.append(p)
    
            # Try candidates in order
            for candidate in candidates:
                try:
                    sp = _init_from_model_file(candidate)
                    break
                except Exception as e:
                    logger.warning(f"Failed to load SentencePiece model from {candidate}: {e}")
    
        # Strategy 3: If the user passed something else (e.g., a bad extension), try appending .model
        if not path.endswith(".model") and os.path.isfile(path + ".model"):
            try:
                sp = _init_from_model_file(path + ".model")
            except Exception as e:
                logger.warning(f"Failed to load SentencePiece model from {path+'.model'}: {e}")
        if sp is not None:
            return cls(name, sp, config)
        # Give up
        raise ValueError(f"Could not load SentencePiece tokenizer from {path}.")


class CustomBPETokenizer(TokenizerWrapper):
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
        return self._tokenizer.encode(text).ids
    
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
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'CustomBPETokenizer':
        """Create HuggingFace tokenizer wrapper from config."""
        # Import here to avoid circular import
        from ..utils.tokenizer_utils import _load_custom_bpe_from_directory
        tokenizer = _load_custom_bpe_from_directory(config)
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
    'unimixlm': UniMixLMTokenizer,
    'custom_bpe': CustomBPETokenizer,
    'sentencepiece': SentencePieceTokenizer
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