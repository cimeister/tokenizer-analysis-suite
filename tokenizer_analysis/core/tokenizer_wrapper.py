"""
Tokenizer wrapper classes for unified tokenizer interface.

This module provides a common interface for different tokenizer types,
making it easy for users to integrate custom tokenizers into the framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Union, Any, Tuple
import json
import logging
import os
import glob
import warnings

from ..constants import GENERIC_SPECIAL_TOKENS, UNK_CANDIDATES

logger = logging.getLogger(__name__)

# Ids of tokenizer objects already warned about in resolve_special_token_strings.
# This suppresses duplicate warnings only; no result is cached under an id, so a
# recycled id can at worst drop a warning, never return another tokenizer's set.
_WARNED_MISSING_SPECIALS: set = set()


def _hf_added_tokens_decoder(tok) -> Optional[Dict[int, Any]]:
    """Return ``{id: AddedToken-like}`` for the tokens added to *tok* outside its
    learned vocabulary, or ``None`` when *tok* exposes no such metadata.

    transformers tokenizers carry the ``added_tokens_decoder`` attribute; a raw
    ``tokenizers.Tokenizer`` exposes the ``get_added_tokens_decoder()`` getter.
    ``None`` rather than ``{}`` keeps "this tokenizer added nothing" distinct from
    "this object cannot be asked", which is the difference between reporting an
    empty special-token set and falling back to GENERIC_SPECIAL_TOKENS.
    """
    dec = getattr(tok, 'added_tokens_decoder', None)
    if dec:
        return dec
    getter = getattr(tok, 'get_added_tokens_decoder', None)
    if callable(getter):
        try:
            return getter()
        except Exception as e:
            logger.debug("get_added_tokens_decoder failed: %s", e)
            return None
    return {} if dec is not None else None


def hf_special_token_strings(tok) -> Optional[Set[str]]:
    """Surface strings of every token *tok* declares special or added.

    The string counterpart of :meth:`HuggingFaceTokenizer.get_special_token_ids`:
    same two metadata channels (``all_special_tokens`` on transformers objects,
    the added-tokens decoder on both), same reasoning that an added token is an
    intentional addition rather than a learned merge. Returns ``None`` when
    neither channel can be read, which tells the caller this object cannot report
    its special tokens at all.

    Takes a raw HuggingFace tokenizer, not a wrapper, so the per-example entry
    points in ``tokenizer_analysis.per_example`` (which are handed the raw object)
    can use the same metadata as the pipeline.
    """
    strings: Set[str] = set()
    answered = False

    specials = getattr(tok, 'all_special_tokens', None)
    if specials is not None:
        answered = True
        strings.update(str(s) for s in specials if s is not None)

    dec = _hf_added_tokens_decoder(tok)
    if dec is not None:
        answered = True
        for added in dec.values():
            content = getattr(added, 'content', None)
            strings.add(str(added) if content is None else str(content))

    return strings if answered else None


def resolve_special_token_strings(tokenizer: Any) -> Set[str]:
    """The special-token strings *tokenizer* declares, or GENERIC_SPECIAL_TOKENS.

    *tokenizer* may be a :class:`TokenizerWrapper` or a raw HuggingFace tokenizer.
    When neither can report its declared special tokens this warns once per
    tokenizer object and returns the generic set, because that set is a guess: it
    holds the common spellings but no tokenizer-specific form such as llama3's
    ``<|reserved_special_token_0|>``.
    """
    declared: Optional[Set[str]] = None
    getter = getattr(tokenizer, 'get_special_token_strings', None)
    if callable(getter):
        try:
            declared = getter()
        except Exception as e:
            logger.debug("get_special_token_strings failed: %s", e)
    else:
        declared = hf_special_token_strings(tokenizer)

    if declared is not None:
        return set(declared)

    key = id(tokenizer)
    if key not in _WARNED_MISSING_SPECIALS:
        _WARNED_MISSING_SPECIALS.add(key)
        name_getter = getattr(tokenizer, 'get_name', None)
        try:
            name = name_getter() if callable(name_getter) else type(tokenizer).__name__
        except Exception:
            name = type(tokenizer).__name__
        logger.warning(
            "Tokenizer %s cannot report its declared special tokens; using the "
            "%d generic ones (GENERIC_SPECIAL_TOKENS). Any special token it "
            "spells differently will be counted as content text.",
            name, len(GENERIC_SPECIAL_TOKENS),
        )
    return set(GENERIC_SPECIAL_TOKENS)


def _setup_fast_decode(tok):
    """Detect and return a fast decode callable for skip_special_tokens=True, or None."""
    # Path A: transformers Fast tokenizer with Rust backend
    backend = getattr(tok, 'backend_tokenizer', None)
    if backend is not None and hasattr(backend, 'decode'):
        return lambda ids: backend.decode(ids, skip_special_tokens=True)

    # Path B: tiktoken-backed tokenizer
    try:
        import tiktoken
    except ImportError:
        return None

    enc = None
    for attr_name in ('tokenizer', 'model', 'encoder'):
        obj = getattr(tok, attr_name, None)
        if isinstance(obj, tiktoken.Encoding):
            enc = obj
            break
    if enc is None:
        for obj in vars(tok).values():
            if isinstance(obj, tiktoken.Encoding):
                enc = obj
                break
    if enc is None:
        return None

    special_ids = frozenset(tok.all_special_ids) if hasattr(tok, 'all_special_ids') else frozenset()
    if special_ids:
        return lambda ids: enc.decode([i for i in ids if i not in special_ids])
    return enc.decode


class TokenizerWrapper(ABC):
    """Abstract interface for tokenizer wrappers."""
    
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        pass
    
    @abstractmethod
    def get_vocab(self) -> Optional[Dict[str, int]]:
        """Get vocabulary mapping. Returns None if not available."""
        pass
    
    @abstractmethod
    def can_encode(self) -> bool:
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Raises NotImplementedError if can_encode() returns False."""
        pass
    
    @abstractmethod
    def can_pretokenize(self) -> bool:
        pass
        
    @abstractmethod
    def pretokenize(self, text: str) -> List[str]:
        """Raises NotImplementedError if can_pretokenize() returns False."""
        pass
    
    @abstractmethod
    def get_special_token_strings(self) -> Optional[Set[str]]:
        """Surface strings of the tokens this tokenizer declares special.

        Read them from the tokenizer's own metadata; never infer them from the
        surface form. Return an empty set when the tokenizer genuinely declares
        none, and ``None`` only when it cannot be asked. ``None`` makes callers
        warn, naming the tokenizer, and fall back to
        :data:`~tokenizer_analysis.constants.GENERIC_SPECIAL_TOKENS`.

        Abstract rather than defaulting to the empty set because the answer
        decides which vocabulary entries are deleted from reconstructed text and
        dropped from the UTF-8 content-token denominator. A wrapper that silently
        inherited "declares none" would publish content-token counts that include
        its BOS and EOS.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'TokenizerWrapper':
        """Create tokenizer wrapper from config."""
        pass

    def pretokenize_with_spans(
        self, text: str
    ) -> Optional[List[Tuple[str, Tuple[int, int]]]]:
        """Pretokenize and return ``[(surface, (start, end)), ...]`` over *text*.

        Returns ``None`` when the backend cannot report spans. Spans are the
        only way to ask what fraction of the source the pretokenizer kept:
        comparing surface string lengths instead counts a byte-level surface,
        where one CJK character becomes three characters, so the inflation hides
        real character loss.
        """
        return None

    def can_decode(self) -> bool:
        return False

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> Optional[str]:
        """Decode token IDs to text. Returns None if not supported."""
        return None

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        """Encode text and return ``(token_ids, offsets)``.

        ``offsets[i]`` is ``(start_char, end_char)`` in the original *text*
        for token *i*.  Tokens that do not map to source characters (e.g.
        ``<s>``, ``</s>``) should have offset ``(0, 0)``.

        The default implementation returns ``(self.encode(text), None)``
        (offsets not supported).  Subclasses should override when the
        underlying tokenizer library provides character-offset information.
        """
        return self.encode(text), None

    def encode_batch_with_offsets(
        self, texts: List[str]
    ) -> List[Tuple[List[int], Optional[List[Tuple[int, int]]]]]:
        """Encode a batch of texts, returning ``(token_ids, offsets)`` per text.

        The default implementation loops over :meth:`encode_with_offsets`.
        Subclasses should override when the underlying library provides a
        native batch API for better throughput.
        """
        return [self.encode_with_offsets(text) for text in texts]

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

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs to token strings.

        Default implementation reverses :meth:`get_vocab`.  Subclasses should
        override with a more efficient method when available.
        """
        vocab = self.get_vocab()
        if vocab:
            id_to_token = {v: k for k, v in vocab.items()}
            return [id_to_token.get(tid, f"<UNK_{tid}>") for tid in token_ids]
        return [f"<UNK_{tid}>" for tid in token_ids]

    def get_unk_token_id(self) -> Optional[int]:
        """
        Get the UNK token ID if available.

        Returns:
            The UNK token ID or None if not available.
        """
        return None

    def has_unk_token(self) -> bool:
        return self.get_unk_token_id() is not None

    def get_special_token_ids(self) -> set:
        """IDs of tokens declared special in the tokenizer's metadata. Default: none."""
        return set()

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
        self._fast_decode = _setup_fast_decode(tokenizer)
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
        result = self._tokenizer.encode(text, add_special_tokens=False)
        # Handle different return types
        if hasattr(result, 'ids'):
            return result.ids
        elif isinstance(result, list):
            return result
        elif isinstance(result, dict) and 'input_ids' in result:
            return result['input_ids']
        else:
            raise ValueError(f"Unexpected encoding result type: {type(result)}")

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        """Encode text using HuggingFace tokenizer and return offsets."""
        result = self._tokenizer.encode(text, add_special_tokens=False)
        if hasattr(result, 'ids') and hasattr(result, 'offsets'):
            return result.ids, result.offsets
        # transformers.PreTrainedTokenizerFast — use __call__ with offset mapping
        if callable(getattr(self._tokenizer, '__call__', None)):
            try:
                enc = self._tokenizer(text, return_offsets_mapping=True,
                                      add_special_tokens=False)
                if 'offset_mapping' in enc:
                    ids = enc['input_ids']
                    offsets = [tuple(pair) for pair in enc['offset_mapping']]
                    return ids, offsets
            except Exception:
                pass
        # Reuse result — don't call encode() again
        if hasattr(result, 'ids'):
            return result.ids, None
        elif isinstance(result, list):
            return result, None
        elif isinstance(result, dict) and 'input_ids' in result:
            return result['input_ids'], None
        else:
            raise ValueError(f"Unexpected encoding result type: {type(result)}")

    def encode_batch_with_offsets(
        self, texts: List[str]
    ) -> List[Tuple[List[int], Optional[List[Tuple[int, int]]]]]:
        # Path A: tokenizers.Tokenizer with native encode_batch
        if hasattr(self._tokenizer, 'encode_batch'):
            encodings = self._tokenizer.encode_batch(texts, add_special_tokens=False)
            results = []
            for enc in encodings:
                if hasattr(enc, 'ids') and hasattr(enc, 'offsets'):
                    results.append((enc.ids, enc.offsets))
                else:
                    results.append((enc.ids, None))
            return results

        # Path B: transformers.PreTrainedTokenizerFast — batched __call__
        if callable(getattr(self._tokenizer, '__call__', None)):
            try:
                batch_enc = self._tokenizer(
                    texts,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )
                if 'offset_mapping' in batch_enc:
                    return [
                        (ids, [tuple(p) for p in offsets])
                        for ids, offsets in zip(
                            batch_enc['input_ids'],
                            batch_enc['offset_mapping'],
                        )
                    ]
            except Exception:
                pass

        # Fallback: loop
        return [self.encode_with_offsets(text) for text in texts]

    def can_decode(self) -> bool:
        return True

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> Optional[str]:
        try:
            if skip_special_tokens and self._fast_decode is not None:
                return self._fast_decode(token_ids)
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        except Exception as e:
            logger.warning(f"HuggingFace decode failed for {self._name}: {e}")
            return None

    def can_pretokenize(self) -> bool:
        return hasattr(self._tokenizer, 'pre_tokenizer') and self._tokenizer.pre_tokenizer is not None

    def pretokenize(self, text: str) -> List[str]:
        if not self.can_pretokenize():
            raise NotImplementedError(f"Tokenizer {self._name} does not support pretokenization")
        return [token for token, _ in self._tokenizer.pre_tokenizer.pre_tokenize_str(text)]

    def pretokenize_with_spans(self, text: str):
        """Surfaces with their character spans, straight from the backend.

        ``pre_tokenize_str`` already returns the spans; the plain
        ``pretokenize`` above throws them away.
        """
        if not self.can_pretokenize():
            return None
        try:
            return list(self._tokenizer.pre_tokenizer.pre_tokenize_str(text))
        except Exception as e:
            logger.debug("pre_tokenize_str failed for %s: %s", self._name, e)
            return None

    def get_underlying_tokenizer(self):
        """Return the underlying HuggingFace tokenizer object."""
        return self._tokenizer

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs using the underlying HuggingFace tokenizer."""
        # tokenizers.Tokenizer uses id_to_token(); transformers.AutoTokenizer
        # uses convert_ids_to_tokens().  Try both.
        if hasattr(self._tokenizer, 'id_to_token'):
            return [self._tokenizer.id_to_token(tid) or f"<UNK_{tid}>" for tid in token_ids]
        if hasattr(self._tokenizer, 'convert_ids_to_tokens'):
            tokens = self._tokenizer.convert_ids_to_tokens(token_ids)
            result = []
            for t, tid in zip(tokens, token_ids):
                if isinstance(t, str):
                    result.append(t)
                elif isinstance(t, bytes):
                    try:
                        result.append(t.decode('utf-8'))
                    except UnicodeDecodeError:
                        result.append(t.decode('utf-8', errors='replace'))
                else:
                    result.append(f"<UNK_{tid}>")
            return result
        return super().convert_ids_to_tokens(token_ids)

    def _added_tokens_decoder(self):
        """Return {id: AddedToken-like} for declared added/special tokens, for either a
        transformers tokenizer or a raw tokenizers.Tokenizer; {} if unavailable."""
        return _hf_added_tokens_decoder(self._tokenizer) or {}

    def get_special_token_strings(self) -> Optional[Set[str]]:
        """Surface strings for the same tokens get_special_token_ids reports: the
        declared special tokens plus everything added outside the learned vocabulary.

        Both come from the tokenizer's own metadata. Measured on the bundled
        tokenizers/bpe.json this returns {'<s>', '</s>', '<pad>', '<unk>'}, on
        apertus 1000 strings and on llama3 256; the surface pattern this replaced
        matched none of bpe.json's four.
        """
        return hf_special_token_strings(self._tokenizer)

    def get_special_token_ids(self) -> set:
        """IDs of all tokens ADDED outside the learned vocabulary -- declared special tokens
        (<bos>, <pad>, <unk>, ...) and reserved/control tokens (<unused123>, [multimodal], ...).
        Taken from the tokenizer's own metadata (added_tokens / all_special_ids), never guessed
        from surface form. These are intentional additions, not learned merges, so callers
        exclude them from junk / unused-vocab statistics."""
        tok = self._tokenizer
        ids = set()
        try:
            ids.update(i for i in getattr(tok, 'all_special_ids', []) or [] if i is not None)
        except Exception:
            pass
        for i in self._added_tokens_decoder().keys():
            ids.add(int(i))
        return ids

    def get_unk_token_id(self) -> Optional[int]:
        """UNK token ID from the tokenizer's declared metadata only. A real UNK token is
        always a declared special token (e.g. transformers' unk_token_id, or an added_tokens
        entry with special=True whose content is a recognized UNK form). A bare 'unk' subword
        is never treated as UNK -- that earlier heuristic mislabeled ordinary subwords."""
        tok = self._tokenizer
        uid = getattr(tok, 'unk_token_id', None)   # transformers: authoritative metadata
        if uid is not None:
            return uid
        for i, a in self._added_tokens_decoder().items():
            content = getattr(a, 'content', None)
            if content is None:
                content = str(a)
            if getattr(a, 'special', False) and content in UNK_CANDIDATES:
                return int(i)
        return None

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'HuggingFaceTokenizer':
        """Create HuggingFace tokenizer wrapper from config."""
        # Import here to avoid circular import
        from ..utils.tokenizer_utils import _load_huggingface_tokenizer
        tokenizer = _load_huggingface_tokenizer(config)
        return cls(name, tokenizer, config)

class UniMixLMTokenizer(HuggingFaceTokenizer):
    """Wrapper for UniMixLM tokenizers.

    Extends :class:`HuggingFaceTokenizer` with language-specific (``langspec``)
    encoding that picks the best per-language tokenizer by log-probability, and
    pre-tokenization via ``base_tokenizer``.

    ``get_special_token_strings`` is inherited unchanged: ``__init__`` asserts
    that every langspec per-language tokenizer has the same vocabulary order as
    the base tokenizer, so the base tokenizer's declaration describes all of them.
    """

    def __init__(self, name: str, tokenizer, config: Dict[str, Any]):
        super().__init__(name, tokenizer, config)
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

    # ── langspec helpers ─────────────────────────────────────────────

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
        """Return dict {token -> log_score} from a HF Unigram tokenizer."""
        vocab_tuples = UniMixLMTokenizer._get_hf_unigram_tokenizer_vocab(tok)
        return {tok_id: tok_tuple[1] for tok_id, tok_tuple in enumerate(vocab_tuples)}

    # ── overrides for langspec encoding ──────────────────────────────

    def _langspec_best_encoding(self, text: str):
        """Find the best langspec encoding by log-probability.

        Returns the raw ``tokenizers.Encoding`` object from the winning
        language tokenizer, or ``None`` if no langspec tokenizers exist.
        """
        best_enc, best_logp = None, float("-inf")
        for info in self.per_lang_tok.values():
            enc = info["tokenizer"].encode(text)
            logp = sum(info["scores"][t] for t in enc.ids)
            if logp > best_logp:
                best_enc, best_logp = enc, logp
        return best_enc

    def encode(self, text: str) -> List[int]:
        if self.tokenizer_class == 'langspec':
            best_enc = self._langspec_best_encoding(text)
            return best_enc.ids if best_enc is not None else []
        return super().encode(text)

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        if self.tokenizer_class == 'langspec':
            best_enc = self._langspec_best_encoding(text)
            if best_enc is not None and hasattr(best_enc, 'offsets'):
                return best_enc.ids, best_enc.offsets
            return (best_enc.ids if best_enc is not None else []), None
        return super().encode_with_offsets(text)

    def encode_batch_with_offsets(
        self, texts: List[str]
    ) -> List[Tuple[List[int], Optional[List[Tuple[int, int]]]]]:
        # Langspec encoding evaluates each text against all per-language
        # tokenizers; cannot use HuggingFaceTokenizer's native batch path.
        return [self.encode_with_offsets(text) for text in texts]

    def get_underlying_tokenizer(self):
        """Return the base tokenizer.

        For langspec UniMixLM tokenizers, this returns the base tokenizer,
        not the per-language tokenizer selected during encoding.  Use
        ``encode()`` on the wrapper for langspec-aware encoding.
        """
        if self.tokenizer_class == 'langspec':
            logger.warning(
                "%s: get_underlying_tokenizer() returns the base tokenizer. "
                "For langspec-aware encoding, call encode() on the wrapper.",
                self._name,
            )
        return self._tokenizer

    # ── overrides for base_tokenizer pre-tokenization ────────────────

    def can_pretokenize(self) -> bool:
        return (
            hasattr(self._tokenizer, 'base_tokenizer')
            and hasattr(self._tokenizer.base_tokenizer, 'pre_tokenizer')
            and self._tokenizer.base_tokenizer.pre_tokenizer is not None
        )

    def pretokenize(self, text: str) -> List[str]:
        if not self.can_pretokenize():
            raise NotImplementedError(f"Tokenizer {self._name} does not support pretokenization")
        return [token for token, _ in self._tokenizer.base_tokenizer.pre_tokenizer.pre_tokenize_str(text)]

    # ── from_config ──────────────────────────────────────────────────

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'UniMixLMTokenizer':
        """Create tokenizer wrapper from config."""
        from tokenizers import Tokenizer
        tokenizer_class = config.get('unimixlm_class')
        if tokenizer_class is not None:
            tokenizer = Tokenizer.from_file(config['path'])
        else:
            from ..utils.tokenizer_utils import _load_huggingface_tokenizer
            tokenizer = _load_huggingface_tokenizer(config)

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

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        """Encode text using SentencePiece and return character offsets.

        Uses ``encode_as_immutable_proto`` which provides ``begin`` / ``end``
        byte offsets on each piece.  Converts byte offsets to character
        offsets so they index into the original Python string.
        """
        try:
            proto = self._sp.encode_as_immutable_proto(text)
        except Exception:
            return self.encode(text), None

        ids: List[int] = []
        offsets: List[Tuple[int, int]] = []

        # Build byte→char offset map for the UTF-8 encoding of *text*.
        text_bytes = text.encode("utf-8")
        byte_to_char: List[int] = []
        for char_idx, ch in enumerate(text):
            byte_to_char.extend([char_idx] * len(ch.encode("utf-8")))
        # Sentinel for end-of-string
        byte_to_char.append(len(text))

        for piece in proto.pieces:
            ids.append(piece.id)
            b_start = piece.begin
            b_end = piece.end
            if b_start == b_end:
                # Special token or unknown — no source coverage
                offsets.append((0, 0))
            else:
                c_start = byte_to_char[b_start] if b_start < len(byte_to_char) else len(text)
                c_end = byte_to_char[b_end] if b_end < len(byte_to_char) else len(text)
                offsets.append((c_start, c_end))

        # Prepend/append BOS/EOS with (0,0) offsets if configured
        if self._add_bos:
            bos = self._sp.bos_id()
            if bos is not None and bos >= 0:
                ids = [bos] + ids
                offsets = [(0, 0)] + offsets
        if self._add_eos:
            eos = self._sp.eos_id()
            if eos is not None and eos >= 0:
                ids = ids + [eos]
                offsets = offsets + [(0, 0)]

        return ids, offsets

    def can_decode(self) -> bool:
        return True

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> Optional[str]:
        try:
            ids_to_decode = list(token_ids)
            if skip_special_tokens:
                bos = self._sp.bos_id()
                eos = self._sp.eos_id()
                special_ids = set()
                if bos is not None and bos >= 0:
                    special_ids.add(bos)
                if eos is not None and eos >= 0:
                    special_ids.add(eos)
                if special_ids:
                    ids_to_decode = [
                        tid for tid in ids_to_decode
                        if tid not in special_ids
                    ]
            return self._sp.decode(ids_to_decode)
        except Exception as e:
            logger.warning(f"SentencePiece decode failed for {self._name}: {e}")
            return None

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

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs using SentencePiece id_to_piece."""
        vocab_size = self._sp.get_piece_size()
        result = []
        for tid in token_ids:
            if 0 <= tid < vocab_size:
                result.append(self._sp.id_to_piece(tid))
            else:
                result.append(f"<UNK_{tid}>")
        return result

    def get_special_token_strings(self) -> Optional[Set[str]]:
        """The configured bos/eos/unk/pad pieces, plus every piece SentencePiece
        itself types as control or unknown.

        The piece type is the model's own metadata, so user-defined control pieces
        (``<|im_start|>`` and the like, added at training time with
        ``--control_symbols``) are covered without matching on surface form.

        Byte-fallback pieces (``<0xNN>``) are deliberately left out although they
        look like markup: each stands for one content byte, and the UTF-8 metrics
        decode them to that byte. SentencePiece types them BYTE, not CONTROL, so
        the scan below already excludes them; likewise UNUSED pieces, which are
        pruned merges rather than special tokens.
        """
        specials: Set[str] = set()

        for id_getter in ('bos_id', 'eos_id', 'unk_id', 'pad_id'):
            try:
                piece_id = getattr(self._sp, id_getter)()
                if piece_id is not None and piece_id >= 0:
                    specials.add(self._sp.id_to_piece(piece_id))
            except Exception as e:
                logger.debug("SentencePiece %s() failed for %s: %s",
                             id_getter, self._name, e)

        # Older sentencepiece bindings name these IsControl/IsUnknown, newer ones
        # also expose the snake_case spelling.
        is_control = (getattr(self._sp, 'IsControl', None)
                      or getattr(self._sp, 'is_control', None))
        is_unknown = (getattr(self._sp, 'IsUnknown', None)
                      or getattr(self._sp, 'is_unknown', None))
        if callable(is_control) and callable(is_unknown):
            try:
                for piece_id in range(self._sp.get_piece_size()):
                    if is_control(piece_id) or is_unknown(piece_id):
                        specials.add(self._sp.id_to_piece(piece_id))
            except Exception as e:
                logger.debug("SentencePiece piece-type scan failed for %s: %s",
                             self._name, e)
        else:
            logger.debug(
                "SentencePiece build for %s exposes no piece-type predicate; "
                "special tokens are the bos/eos/unk/pad pieces only.", self._name,
            )
        return specials

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
        for candidate in UNK_CANDIDATES:
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
    """Wrapper for custom BPE tokenizers loaded from vocab.json and merges.txt."""

    def __init__(self, name: str, tokenizer, config: Dict[str, Any]):
        """
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

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        """Encode text using custom BPE tokenizer and return offsets."""
        encoding = self._tokenizer.encode(text)
        return encoding.ids, encoding.offsets

    def encode_batch_with_offsets(
        self, texts: List[str]
    ) -> List[Tuple[List[int], Optional[List[Tuple[int, int]]]]]:
        encodings = self._tokenizer.encode_batch(texts)
        return [(enc.ids, enc.offsets) for enc in encodings]

    def can_decode(self) -> bool:
        return True

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> Optional[str]:
        try:
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        except Exception as e:
            logger.warning(f"CustomBPE decode failed for {self._name}: {e}")
            return None

    def can_pretokenize(self) -> bool:
        return hasattr(self._tokenizer, 'pre_tokenizer') and self._tokenizer.pre_tokenizer is not None

    def pretokenize(self, text: str) -> List[str]:
        if not self.can_pretokenize():
            raise NotImplementedError(f"Tokenizer {self._name} does not support pretokenization")
        return [token for token, _ in self._tokenizer.pre_tokenizer.pre_tokenize_str(text)]

    def get_special_token_strings(self) -> Optional[Set[str]]:
        """Added-token surfaces from the underlying ``tokenizers.Tokenizer``.

        A vocab.json + merges.txt BPE has no special tokens of its own; any it
        reports were attached to the loaded tokenizer object, and that object's
        added-tokens metadata is the only place they are declared.
        """
        return hf_special_token_strings(self._tokenizer)

    def get_unk_token_id(self) -> Optional[int]:
        """Get the UNK token ID from custom BPE tokenizer."""
        try:
            if hasattr(self._tokenizer, 'unk_token_id'):
                return self._tokenizer.unk_token_id
            if hasattr(self._tokenizer, 'token_to_id'):
                return self._tokenizer.token_to_id('<unk>')
        except Exception:
            pass
        return None

    def get_underlying_tokenizer(self):
        """Return the underlying HuggingFace tokenizer object."""
        return self._tokenizer

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs using the underlying HuggingFace tokenizer."""
        return [self._tokenizer.id_to_token(tid) or f"<UNK_{tid}>" for tid in token_ids]

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

    def get_special_token_strings(self) -> Optional[Set[str]]:
        """``None``: this wrapper holds a vocab size and an optional vocab dict
        taken from the config, and the config declares no special tokens.

        An empty set would assert that the pre-tokenized corpus contains none,
        which nothing here establishes. ``None`` makes the caller warn, naming
        this tokenizer, and use GENERIC_SPECIAL_TOKENS instead.
        """
        return None

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'PreTokenizedDataTokenizer':
        """Create pre-tokenized data tokenizer wrapper from config."""
        vocab_size = config.get('vocab_size')
        vocab_dict = config.get('vocab_dict')
        if vocab_size is None:
            raise ValueError("PreTokenizedDataTokenizer requires vocab_size in config")
        return cls(name, vocab_size, vocab_dict)


class _ScriptTokTokenizer(TokenizerWrapper):
    """Shared wrapper for ``script_bpe`` tokenizers (SCRIPT BPE and MinGram).

    Both back ends (``script_bpe.BPETokenizer`` and ``script_bpe.MinGramModel``)
    share one contract: a ``.tokens`` dict mapping ``int`` id -> token object, a
    bundled ``.pretokenizer`` that mediates encode/decode, and no bos/eos/unk/pad
    special tokens. The concrete subclasses differ only in how ``from_config``
    loads the back end, so all shared behaviour lives here.

    The back end's native ids are not a dense range. Both tokenizers reserve id 0
    (pad); SCRIPT BPE is then dense from 1, but MinGram's EM pruning leaves gaps,
    so ``max(id)`` can far exceed the token count. Reporting native ids would
    inflate ``vocab_size`` (and the vocabulary-utilization denominator) by the gap
    count and make the size tokenizer-dependent. This wrapper therefore remaps
    native ids to a gap-free, origin-preserving range (see :meth:`__init__`):
    ``encode`` emits compacted ids, ``decode`` maps them back, and ``vocab_size``
    is the compacted bound. The remap is order-preserving but is not the id
    assignment of any downstream model.

    Token strings come from ``pretokenizer.tokens_repr`` rather than per-id
    ``decode``: a lone SCRIPT block/index token is not independently decodable,
    so ``decode([id])`` collapses most ids to the same empty/fragment string,
    whereas ``tokens_repr`` renders each token distinctly. Under the SCRIPT
    pretokenizer those strings carry ``<|BLOCK_...|>`` / ``<|SCRIPT_INDEX_...|>``
    markers for pieces that are not whole characters, so a token string is the
    token's rendered form, not always literal source text (this also reflects
    NFC normalization and digit regrouping applied by the pretokenizer). Metrics
    that read token strings (byte coverage, junk/dead-vocab, whitespace) describe
    this rendered form. Full-sequence :meth:`decode` uses the back end's own
    decoder and round-trips at the character level, except that the pretokenizer
    normalizes (NFC), regroups digits, and drops code points it cannot encode
    (unassigned or noncharacter), so inputs with those do not round-trip exactly.
    """

    def __init__(self, name: str, backend, config: Dict[str, Any]):
        """
        Args:
            name: Tokenizer name.
            backend: A loaded ``script_bpe`` tokenizer (BPETokenizer/MinGramModel).
            config: Original configuration dict.
        """
        self._name = name
        self._backend = backend
        self._config = config or {}
        self._pretokenizer = backend.pretokenizer
        # Remap the back end's (possibly sparse) native ids to a gap-free range,
        # preserving the origin: the smallest native id maps to itself and the
        # rest follow contiguously. A 1-based (id-0-reserved) tokenizer stays
        # 1-based, a 0-based (reindexed) one stays 0-based, and MinGram's pruning
        # gaps are removed. This keeps vocab_size at the token count (not the
        # sparse max-id) so the vocabulary-utilization denominator is not
        # inflated. The map is order-preserving but is not any downstream model's
        # id assignment; it only gives metrics a dense id space.
        native_ids = sorted(backend.tokens.keys())
        base = native_ids[0] if native_ids else 0
        self._native_to_dense: Dict[int, int] = {
            nid: base + rank for rank, nid in enumerate(native_ids)
        }
        self._dense_to_native: Dict[int, int] = {
            dense: nid for nid, dense in self._native_to_dense.items()
        }
        # Compacted id-space bound (== max dense id + 1 == base + token count).
        self._vocab_size: int = base + len(native_ids)
        self._id_to_str_cache: Optional[Dict[int, str]] = None
        self._vocab_cache: Optional[Dict[str, int]] = None

    def get_name(self) -> str:
        return self._name

    def get_vocab_size(self) -> int:
        # The compacted (gap-free) id-space bound: the token count, plus one for
        # a reserved id 0. The input validator and id-indexed metrics treat
        # vocab_size as a bound on emitted ids (id < vocab_size); encode emits
        # compacted ids, so this holds without the gap inflation that would bias
        # vocabulary utilization and make the size depend on the tokenizer.
        return self._vocab_size

    def _ensure_str_cache(self) -> Dict[int, str]:
        """Build once the compacted-id -> token-string map, and return it."""
        if self._id_to_str_cache is None:
            cache: Dict[int, str] = {}
            for nid, tok in self._backend.tokens.items():
                dense = self._native_to_dense[nid]
                try:
                    cache[dense] = self._pretokenizer.tokens_repr(tok.atomic_tokens)
                except Exception:
                    cache[dense] = f"<TOK_{dense}>"
            self._id_to_str_cache = cache
        return self._id_to_str_cache

    def get_vocab(self) -> Dict[str, int]:
        if self._vocab_cache is None:
            # tokens_repr is distinct per token for these tokenizers, so this is
            # a faithful full-vocabulary map (one entry per compacted id). Were
            # two ids to ever render alike, the later id would win here.
            self._vocab_cache = {s: d for d, s in self._ensure_str_cache().items()}
        # Return a copy so a caller mutating the result cannot corrupt the cache
        # (the HF/SP/custom_bpe wrappers likewise hand back a fresh dict).
        return dict(self._vocab_cache)

    def can_encode(self) -> bool:
        return True

    def encode(self, text: str) -> List[int]:
        # BPETokenizer.encode returns array.array; MinGramModel.encode returns
        # list[int]. Remap each native id to its compacted id (plain Python int).
        to_dense = self._native_to_dense
        return [to_dense[i] for i in self._backend.encode(text)]

    def can_decode(self) -> bool:
        return True

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> Optional[str]:
        # Inputs are compacted ids (what encode emits); map them back to native
        # ids for the back end. skip_special_tokens is a no-op (no special
        # tokens). BPETokenizer.decode defaults errors="replace"; MinGramModel
        # .decode takes no errors kwarg, so passing only ids works for both.
        try:
            native = [self._dense_to_native[i] for i in token_ids]
            return self._backend.decode(native)
        except Exception as e:
            logger.warning(f"script_bpe decode failed for {self._name}: {e}")
            return None

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        cache = self._ensure_str_cache()
        return [cache.get(i, f"<UNK_{i}>") for i in token_ids]

    def can_pretokenize(self) -> bool:
        return True

    def pretokenize(self, text: str) -> List[str]:
        # pretokenizer.pretokenize returns a list of atomic-token chunks
        # (array.array); decode each chunk back to its rendered substring.
        return [self._pretokenizer.decode(chunk)
                for chunk in self._pretokenizer.pretokenize(text)]

    def get_special_token_strings(self) -> Optional[Set[str]]:
        """Empty: the ``script_bpe`` back ends declare no special tokens.

        Both reserve id 0 for padding, but the reserved id is not a key of
        ``backend.tokens``, so it has no token string and never appears in the
        vocabulary this wrapper exposes. Every string :meth:`get_vocab` returns is
        a rendered token (see the class docstring), which is content.
        """
        return set()

    def get_underlying_tokenizer(self):
        # Deliberately None: these are not HuggingFace tokenizers, so MorphScore
        # and the HF-internal sanity checks skip cleanly instead of misreading a
        # foreign object.
        return None


class ScriptBPETokenizer(_ScriptTokTokenizer):
    """Wrapper for SCRIPT BPE tokenizers (``script_bpe.BPETokenizer``, ``sebpe-v1``)."""

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'ScriptBPETokenizer':
        """
        Expected config keys:
          - path: path to a saved script_bpe BPE tokenizer (.json or .json.gz).
        """
        if "path" not in config:
            raise ValueError(
                "ScriptBPETokenizer requires 'path' to a script_bpe tokenizer "
                "file (.json or .json.gz)")
        try:
            from script_bpe import BPETokenizer
        except ImportError as e:
            raise RuntimeError(
                "The 'script_bpe' package is required for the 'script_bpe' "
                "tokenizer class. Install it with `pip install -e "
                "/path/to/script_tok` or add its repo to PYTHONPATH."
            ) from e
        path = config["path"]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"script_bpe tokenizer file not found: {path}")
        backend = BPETokenizer.load(path)
        return cls(name, backend, config)


class MinGramTokenizer(_ScriptTokTokenizer):
    """Wrapper for MinGram tokenizers (``script_bpe.MinGramModel``, ``semingram-v1``)."""

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'MinGramTokenizer':
        """
        Expected config keys:
          - path: path to a saved script_bpe MinGram tokenizer (.json or .json.gz).
          - (optional) reindex: if True, densify token ids to 0..n-1 on load
            (default False; passed to MinGramModel.load).
        """
        if "path" not in config:
            raise ValueError(
                "MinGramTokenizer requires 'path' to a script_bpe tokenizer "
                "file (.json or .json.gz)")
        try:
            from script_bpe import MinGramModel
        except ImportError as e:
            raise RuntimeError(
                "The 'script_bpe' package is required for the 'mingram' "
                "tokenizer class. Install it with `pip install -e "
                "/path/to/script_tok` or add its repo to PYTHONPATH."
            ) from e
        path = config["path"]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"script_bpe tokenizer file not found: {path}")
        reindex = bool(config.get("reindex", False))
        backend = MinGramModel.load(path, reindex=reindex)
        return cls(name, backend, config)


# Registry for custom tokenizer classes
_TOKENIZER_REGISTRY: Dict[str, type] = {
    'huggingface': HuggingFaceTokenizer,
    'hf': HuggingFaceTokenizer,
    'transformers': HuggingFaceTokenizer,
    'standard': HuggingFaceTokenizer,  # Legacy alias
    'pretokenized': PreTokenizedDataTokenizer,
    'unimixlm': UniMixLMTokenizer,
    'custom_bpe': CustomBPETokenizer,
    'sentencepiece': SentencePieceTokenizer,
    'script_bpe': ScriptBPETokenizer,
    'mingram': MinGramTokenizer
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
        warnings.warn(
            "The 'standard' tokenizer class is deprecated; use 'huggingface' "
            f"instead. Tokenizer {name!r} will be loaded as 'huggingface'.",
            DeprecationWarning,
            stacklevel=2,
        )
    
    if tokenizer_class_name not in _TOKENIZER_REGISTRY:
        available_classes = list(_TOKENIZER_REGISTRY.keys())
        raise ValueError(f"Unknown tokenizer class: {tokenizer_class_name}. "
                        f"Available classes: {available_classes}")
    
    tokenizer_class = _TOKENIZER_REGISTRY[tokenizer_class_name]
    return tokenizer_class.from_config(name, config)