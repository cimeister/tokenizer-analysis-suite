"""
Shared base for the metric classes that read the unified TokenizedData
interface.

``BaseMetrics`` is the abstract class every metric subclasses. It defines the
per-tokenizer context the metrics resolve once and read per token (declared
special tokens, detected subword markers, probed character decode table), the
token-id to token-string conversion with its fallbacks, the statistics helpers
(``compute_basic_stats``, ``safe_divide``) and the input validators. ``TokenizedDataProcessor`` below it groups and
flattens TokenizedData lists. ``format_optional`` prints a None as 'n/a'.

The token-string reconstruction helpers here (``_process_token``,
``_clean_token``, ``_build_char_to_token_map``,
``_build_source_to_recon_map``) have no production caller. Each has its own
note recording what the reconstruction got wrong and why the metrics read the
encoder's character offsets instead.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Tuple
import re
import numpy as np
import scipy
import scipy.stats
from collections import defaultdict
import logging

from ..core.input_types import Corpus, TokenizedData
from ..core.input_providers import InputProvider
from ..core.tokenizer_wrapper import resolve_special_token_strings
from ..constants import (
    DEFAULT_SAFE_DIVIDE_VALUE,
    GENERIC_SPECIAL_TOKENS,
    MISSING_VALUE_DISPLAY,
    PERCENTAGE_MULTIPLIER,
    MAX_ERROR_DISPLAY_COUNT,
)

logger = logging.getLogger(__name__)


def format_optional(value: Any, spec: str = ".3f") -> str:
    """Format *value*, or return 'n/a' when it is None.

    The metrics publish None for a value they could not compute, so that a
    reader cannot mistake it for a measured zero. ``format(None, '.3f')``
    raises TypeError, and the usual guard against that, ``d.get(key, 0.0)``,
    returns the stored None when the key is present and prints 0.000 when it is
    not. Both outcomes are worse than the null they came from: one aborts the
    run's summary, the other reinstates in the console the zero the results file
    was careful not to publish.
    """
    if value is None:
        return MISSING_VALUE_DISPLAY
    return format(value, spec)


# Subword-marker strings that _process_token strips, and only when the
# tokenizer being processed actually shows it uses that marker (see
# BaseMetrics._detect_subword_markers). Each name records the tokenizer
# family the marker comes from. Stripping any of these unconditionally, from
# every tokenizer regardless of family, is the defect this module fixes: a
# byte-level BPE vocabulary that uses none of them still has ordinary content
# tokens matching the patterns (e.g. a markdown heading '###', a comment
# banner '################', or a bare '@@'), and unconditional stripping
# truncated or emptied them.
_WORDPIECE_CONTINUATION_PREFIX = '##'    # BERT / WordPiece continuing-subword prefix
_CLIP_BPE_END_OF_WORD_SUFFIX = '</w>'    # CLIP-style BPE end-of-word suffix
_SUBWORD_NMT_CONTINUATION_SUFFIX = '@@'  # subword-nmt continuation suffix


class BaseMetrics(ABC):
    """Base class for tokenizer metrics using TokenizedData interface."""

    # Bounded re-sync for _build_source_to_recon_map. The window caps how long
    # a divergence can be and still be recovered; the anchor is how many
    # characters must agree before the scan trusts the new alignment. A
    # 1-character anchor re-syncs on any coincidental single match.
    _RESYNC_WINDOW = 32
    _RESYNC_ANCHOR = 3

    # Pre-compiled regex patterns for subword marker handling. _CONTINUATION,
    # _END_WORD and _CONTINUATION_END are read in one place, _process_token,
    # and nowhere else; MorphScoreMetrics reads none of them. _SPACE_PREFIX has
    # no reference anywhere in the package. _process_token tests the decoded
    # first character against ' ' directly, because _DEFAULT_CHAR_DECODE has
    # already rewritten Ġ and ▁ to a space by the time that check runs.
    _SPACE_PREFIX = re.compile(r'^[Ġ▁ ]')
    _CONTINUATION = re.compile(r'^##')
    _END_WORD = re.compile(r'</w>$')
    _CONTINUATION_END = re.compile(r'@@$')
    # SentencePiece byte-fallback token form: a single token encoding one raw
    # byte as the literal 6-character string `<0xNN>` (NN = uppercase or
    # lowercase hex). Emitted by LLaMA-family / Mistral SP tokenizers with
    # `byte_fallback=True` for any byte (e.g. newline → `<0x0A>`, tab → `<0x09>`).
    _BYTE_FALLBACK = re.compile(r'<0x([0-9A-Fa-f]{2})>')

    # Known byte-level BPE / SentencePiece character remappings.
    _DEFAULT_CHAR_DECODE: Dict[str, str] = {
        'Ġ': ' ', '▁': ' ', 'Ċ': '\n', 'ĉ': '\t', 'č': '\r',
    }

    @classmethod
    def _decode_byte_fallback(cls, token: str) -> str:
        """Decode SentencePiece byte-fallback substrings in *token* to their raw bytes.

        Replaces every occurrence of `<0xNN>` (case-insensitive hex) with the
        single character `chr(int(NN, 16))`. A no-op for tokens that contain no
        byte-fallback substring (the common case).

        Why this exists: the per-character `_DEFAULT_CHAR_DECODE` table cannot
        express the 6-char → 1-char rewrite that byte-fallback needs. Without
        this step, `<0x0A>` reaches the char-to-token reconstruction as the
        literal six-char string instead of as `\\n`, which breaks the source-to-recon
        greedy match at every newline / non-ASCII byte for SP-byte-fallback
        tokenizers (EuroLLM, LLaMA family, etc.).
        """
        return cls._BYTE_FALLBACK.sub(lambda m: chr(int(m.group(1), 16)), token)

    def __init__(self, input_provider: InputProvider):
        self.input_provider = input_provider
        self.tokenizer_names = input_provider.get_tokenizer_names()
        self.language_metadata = None  # Can be set by subclasses
        # Both per-tokenizer caches below key on id(tokenizer) and store the
        # tokenizer object alongside the value. The object reference is the
        # point: CPython recycles the id of a freed object, so without it a
        # long-lived metrics instance (the module-level singletons in
        # per_example.py live for the whole process) hands one tokenizer the
        # cached data of a different, already-collected one.
        #
        # Measured through the public per_example API before this: 20 of 40
        # calls read another tokenizer's reverse vocabulary, and every
        # Tokenizer.from_file landed on the same recycled address so only one
        # cache entry was ever created. The metric consequence for one text was
        # n_digit_spans 0 against 2 and mean_digit_f1 nan against 0.0.
        self._tokenizer_vocab_cache: Dict[int, Tuple[Any, Dict[int, str]]] = {}
        self._warned_tokenizers: set = set()
        self._char_decode_table: Optional[Dict[str, str]] = None
        # Special-token strings of the tokenizer currently being processed, set
        # by the per-tokenizer loops the same way _char_decode_table is. None
        # means "not resolved for a specific tokenizer", and _process_token then
        # uses GENERIC_SPECIAL_TOKENS.
        self._special_tokens: Optional[Set[str]] = None
        self._special_token_cache: Dict[int, Tuple[Any, Set[str]]] = {}
        # Subword-marker strings ('##', '</w>', '@@') the tokenizer currently
        # being processed actually uses, set by the same per-tokenizer loops
        # that set _char_decode_table and _special_tokens (see
        # _set_tokenizer_context). None means "not resolved for a specific
        # tokenizer"; _process_token treats that the same as an empty set:
        # strip nothing, unlike _special_tokens, which falls back to
        # GENERIC_SPECIAL_TOKENS. There is no equivalent generic guess here:
        # a guessed marker is exactly the defect being fixed, so an
        # unresolved or undetected marker set both mean "strip nothing".
        self._subword_markers: Optional[Set[str]] = None
        self._subword_marker_cache: Dict[int, Tuple[Any, Set[str]]] = {}

    def _registered_corpus(self, name: str) -> Optional['Corpus']:
        """The corpus registered under *name* on the input provider, or None.

        None means the run registered nothing, and the metric then builds the
        corpus from its own constructor arguments. That path is what keeps the
        classes constructible on their own: per_example.py and
        scripts/run_ast_only.py both build a metric against a provider that
        carries no corpora, and so does most of the test suite.

        The registry is looked up by attribute rather than assumed, because a
        provider need not be an ``InputProvider`` subclass. Every provider in
        this package is one, including the stand-ins
        (``per_example._StubInputProvider``, the test suite's ``MockProvider``
        and ``scripts/run_ast_only._AstOnlyProvider``), so the attribute check
        is there for a caller's own duck-typed provider rather than for
        anything shipped here.
        """
        corpus_names = getattr(self.input_provider, 'corpus_names', None)
        if not callable(corpus_names):
            return None
        # set(), not the list: corpus_names() builds a fresh list on every call
        # and this runs once per corpus per metric construction.
        if name not in set(corpus_names()):
            return None
        return self.input_provider.get_corpus(name)

    def _corpus_or_refuse_arguments(
        self, name: str, arguments: Dict[str, Any]
    ) -> Optional['Corpus']:
        """The registered corpus under *name*, refusing arguments it overrides.

        A metric takes its corpus from the registry when the run put one there,
        and builds one from its own constructor arguments otherwise. When both
        are present the registry used to win and the arguments were dropped
        without a word, so a caller who passed ``math_data_path`` could be
        handed numbers measured on a different corpus with nothing in the
        output saying so.

        Returns None when nothing is registered, which is the signal to build
        the corpus from the arguments instead.
        """
        corpus = self._registered_corpus(name)
        if corpus is None:
            return None
        # "Supplied" is "not None and not False", not truthiness. An empty dict
        # is a request in this package, not the absence of one:
        # cli/run_analysis returns {} for --code-ast-config to mean "use the
        # bundled samples" and None to mean "disabled", and code_texts={} used
        # to mean "this metric reports no code domain". Reading either as
        # unsupplied let the registered corpus override an explicit request,
        # which is the substitution this check exists to refuse.
        supplied = sorted(
            key for key, value in arguments.items()
            if value is not None and value is not False
        )
        if supplied:
            raise ValueError(
                f"{type(self).__name__} was given {', '.join(supplied)}, but a "
                f"{name!r} corpus from {corpus.source!r} is already registered "
                "on the input provider. Using the registered corpus would "
                "report numbers measured on it under a request for the other "
                "one. Pass the arguments or register the corpus, not both."
            )
        return corpus

    def _register_corpus(self, corpus: 'Corpus') -> 'Corpus':
        """Register a corpus this metric built itself, and return it.

        The provider is where a corpus is encoded and memoized per tokenizer, so
        a metric that had to build its own still puts it there rather than
        keeping it. A provider that does not implement the registry is named
        here, rather than reaching the encode call three frames later as an
        AttributeError on a method nobody mentioned.
        """
        add_corpus = getattr(self.input_provider, 'add_corpus', None)
        if not callable(add_corpus):
            raise TypeError(
                f"{type(self.input_provider).__name__} does not implement "
                f"add_corpus, so the {corpus.name!r} corpus this metric built "
                "cannot be encoded. Subclass InputProvider, which implements "
                "the corpus registry."
            )
        add_corpus(corpus)
        return corpus

    def _resolve_special_tokens(self, tokenizer: Any) -> Set[str]:
        """Special-token strings declared by *tokenizer*, memoized per object.

        The memo is what keeps the fallback warning to one per tokenizer instead
        of one per token: _process_token runs over every token of every document.
        """
        key = id(tokenizer)
        cached = self._special_token_cache.get(key)
        if cached is not None and cached[0] is tokenizer:
            return cached[1]
        resolved = resolve_special_token_strings(tokenizer)
        self._special_token_cache[key] = (tokenizer, resolved)
        return resolved

    def _resolve_subword_markers(self, tokenizer: Any) -> Set[str]:
        """Subword-marker strings *tokenizer* actually uses, memoized per object.

        Same identity-keyed cache pattern as _resolve_special_tokens (the
        object reference, not just id(tokenizer), is stored. See the
        comment on _special_token_cache in __init__ for why: CPython can
        recycle a freed object's id, and without the reference check a
        long-lived metrics instance would hand one tokenizer another,
        already-collected tokenizer's marker set).
        """
        key = id(tokenizer)
        cached = self._subword_marker_cache.get(key)
        if cached is not None and cached[0] is tokenizer:
            return cached[1]
        resolved = self._detect_subword_markers(tokenizer)
        self._subword_marker_cache[key] = (tokenizer, resolved)
        return resolved

    @staticmethod
    def _detect_subword_markers(tokenizer: Any) -> Set[str]:
        """Which of the three known subword markers *tokenizer* actually emits.

        Two channels, declared checked first:

        1. Declared. Unwrap to the backend tokenizer the same way
           UTF8IntegrityMetrics._has_bytelevel_component does (via
           get_underlying_tokenizer(), then backend_tokenizer), and read the
           backend model's own continuing_subword_prefix / end_of_word_suffix
           fields. Both are real fields on tokenizers.models.BPE and
           tokenizers.models.WordPiece (and on the model block of a
           serialized tokenizer.json), verified by construction against the
           installed tokenizers version: a fresh WordPiece defaults
           continuing_subword_prefix to '##' even with no arguments, and a
           fresh BPE defaults both fields to None. subword-nmt's '@@' has no
           such field: its marker is baked into the vocabulary strings
           themselves, not a model parameter, so this channel cannot see it.

        2. Behavioral. Encode a probe word certain to fragment into several
           subwords under any trained vocabulary, and look at the raw piece
           strings for one of the three known marker forms. The probe is
           'supercalifragilisticexpialidocious': a 34-character invented word
           (from Mary Poppins) that essentially no trained subword vocabulary
           holds as a single token, so it reliably fragments regardless of
           tokenizer family, verified against a real bert-base-uncased
           WordPiece vocabulary (11 pieces, continuation pieces '##'-prefixed)
           and a small BPE trained with end_of_word_suffix='</w>' (30 pieces,
           last one '</w>'-suffixed).

        Returns the empty set when neither channel finds a marker in use.
        That is a deliberate default, not a gap in coverage: applying a
        WordPiece rule to a non-WordPiece vocabulary silently corrupts
        ordinary content tokens (measured: 35 cl100k_base vocabulary entries
        altered, 24 for o200k_base, 1 for the bundled tokenizers/bpe.json),
        whereas failing to strip a marker a tokenizer genuinely uses only
        leaves that marker in the reconstruction, where it surfaces as an
        unmappable span rather than as a silently wrong number.
        """
        markers: Set[str] = set()

        # Channel 1: declared
        backend = tokenizer
        if hasattr(backend, 'get_underlying_tokenizer'):
            try:
                backend = backend.get_underlying_tokenizer() or backend
            except Exception as e:
                logger.debug(
                    "Could not unwrap tokenizer for subword-marker detection: %s", e
                )
        backend = getattr(backend, 'backend_tokenizer', backend)
        model = getattr(backend, 'model', None)
        if model is not None:
            prefix = getattr(model, 'continuing_subword_prefix', None)
            if prefix == _WORDPIECE_CONTINUATION_PREFIX:
                markers.add(_WORDPIECE_CONTINUATION_PREFIX)
            elif not prefix and type(model).__name__ == 'WordPiece':
                # A WordPiece model whose binding does not expose the field
                # (or exposes it empty) still defaults to '##' in every
                # tokenizers version checked; this only covers that binding
                # gap, not a genuinely custom prefix (handled below).
                markers.add(_WORDPIECE_CONTINUATION_PREFIX)
            elif prefix:
                logger.debug(
                    "Tokenizer declares a non-standard continuing_subword_prefix "
                    "%r; _process_token only recognizes '##', so it is left in "
                    "place rather than guessed at.", prefix,
                )

            suffix = getattr(model, 'end_of_word_suffix', None)
            if suffix == _CLIP_BPE_END_OF_WORD_SUFFIX:
                markers.add(_CLIP_BPE_END_OF_WORD_SUFFIX)
            elif suffix:
                logger.debug(
                    "Tokenizer declares a non-standard end_of_word_suffix %r; "
                    "_process_token only recognizes '</w>', so it is left in "
                    "place rather than guessed at.", suffix,
                )

        if markers:
            return markers

        # Channel 2: behavioral
        if not hasattr(tokenizer, 'encode'):
            return markers
        probe = "supercalifragilisticexpialidocious"
        try:
            encoded = tokenizer.encode(probe)
        except Exception as e:
            logger.debug("Subword-marker probe encode failed: %s", e)
            return markers
        ids = list(encoded.ids if hasattr(encoded, 'ids') else encoded)
        if len(ids) < 2:
            return markers
        pieces: Optional[List[str]] = None
        try:
            if hasattr(tokenizer, 'convert_ids_to_tokens'):
                pieces = tokenizer.convert_ids_to_tokens(ids)
        except Exception as e:
            logger.debug(
                "Subword-marker probe convert_ids_to_tokens failed: %s", e
            )
        if not pieces:
            return markers
        # continuing_subword_prefix marks every piece but the first;
        # subword-nmt's '@@' marks every piece but the last;
        # end_of_word_suffix marks only the last.
        for piece in pieces[1:]:
            if not isinstance(piece, str):
                continue
            if piece.startswith(_WORDPIECE_CONTINUATION_PREFIX):
                markers.add(_WORDPIECE_CONTINUATION_PREFIX)
        for piece in pieces[:-1]:
            if isinstance(piece, str) and piece.endswith(_SUBWORD_NMT_CONTINUATION_SUFFIX):
                markers.add(_SUBWORD_NMT_CONTINUATION_SUFFIX)
        last = pieces[-1]
        if isinstance(last, str) and last.endswith(_CLIP_BPE_END_OF_WORD_SUFFIX):
            markers.add(_CLIP_BPE_END_OF_WORD_SUFFIX)
        return markers

    def get_tokenized_data(self) -> Dict[str, List[TokenizedData]]:
        """Get tokenized data organized by tokenizer."""
        return self.input_provider.get_tokenized_data()
    
    def get_vocab_size(self, tokenizer_name: str) -> int:
        """Get vocabulary size for a tokenizer."""
        return self.input_provider.get_vocab_size(tokenizer_name)
    
    def get_languages(self, tokenizer_name: Optional[str] = None) -> List[str]:
        """Get available languages."""
        return self.input_provider.get_languages(tokenizer_name)

    # ------------------------------------------------------------------
    # Shared token conversion / cleaning helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_char_decode_table(tokenizer: Any) -> Dict[str, str]:
        """Probe *tokenizer* to discover its character remapping table.

        Encodes known whitespace characters (space, newline, tab, carriage
        return) and inspects the raw token strings for non-matching
        characters.  Returns a mapping from encoded character to the
        original character, e.g. ``{'Ġ': ' ', 'Ċ': '\\n'}``.

        Returns an empty dict when the tokenizer does not remap characters
        (e.g. WordPiece), or when probing fails.
        """
        if not hasattr(tokenizer, 'encode'):
            return {}

        probes = [
            ('a a', 1, ' '),   # space
            ('a\na', 1, '\n'), # newline
            ('a\ta', 1, '\t'), # tab
            ('a\ra', 1, '\r'), # carriage return
        ]
        table: Dict[str, str] = {}
        for text, target_pos, original_char in probes:
            try:
                token_ids = tokenizer.encode(text)
            except Exception:
                continue
            if not token_ids:
                continue
            # Convert IDs → raw token strings
            raw_tokens: Optional[List[str]] = None
            try:
                if hasattr(tokenizer, 'convert_ids_to_tokens'):
                    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
            except Exception as exc:
                logger.debug("decode-table probe %r failed (%s: %s); that "
                             "character keeps no remap entry",
                             original_char, type(exc).__name__, exc)
            if not raw_tokens:
                continue
            # Look at the token at target_pos (the one after 'a')
            # and also search through all tokens for the remapping.
            for raw_tok in raw_tokens:
                if not isinstance(raw_tok, str) or len(raw_tok) < 1:
                    continue
                for ch in raw_tok:
                    if ch != original_char and ch not in 'aA' and ord(ch) > 127:
                        # This high-unicode char might be a remapping.
                        # Verify: does the token content make sense with this
                        # substitution?
                        if ch not in table:
                            table[ch] = original_char
        return table

    def _convert_ids_to_tokens(self, tokenizer: Any, token_ids: List[int]) -> List[str]:
        """Convert token IDs to strings with multiple fallback strategies.

        Fallback order:
        1. ``tokenizer.convert_ids_to_tokens``
        2. ``tokenizer.get_vocab`` → reverse mapping (cached)
        3. ``tokenizer.model.id_to_token``
        4. Placeholder strings ``<TOKEN_{id}>``
        """
        if not token_ids:
            return []

        tokenizer_id = id(tokenizer)

        # Fast path: use cached vocab reverse-mapping if available
        cached = self._tokenizer_vocab_cache.get(tokenizer_id)
        if cached is not None and cached[0] is tokenizer:
            id_to_token = cached[1]
            return [id_to_token.get(tid, f"<UNK_{tid}>") for tid in token_ids]

        try:
            if hasattr(tokenizer, 'convert_ids_to_tokens'):
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                if tokens and all(isinstance(t, str) for t in tokens):
                    return tokens
        except Exception as e:
            logger.debug("convert_ids_to_tokens failed: %s", e)

        try:
            vocab = None
            if hasattr(tokenizer, 'get_vocab'):
                vocab = tokenizer.get_vocab()
            if vocab:
                self._tokenizer_vocab_cache[tokenizer_id] = (tokenizer, {
                    v: (k.decode('utf-8') if isinstance(k, bytes) else str(k))
                    for k, v in vocab.items()
                })
                id_to_token = self._tokenizer_vocab_cache[tokenizer_id][1]
                return [id_to_token.get(tid, f"<UNK_{tid}>") for tid in token_ids]
        except Exception as e:
            logger.debug("Vocabulary lookup fallback failed: %s", e)

        try:
            if hasattr(tokenizer, 'model') and hasattr(tokenizer.model, 'id_to_token'):
                tokens = [tokenizer.model.id_to_token(tid) for tid in token_ids]
                if tokens and all(t is not None for t in tokens):
                    return [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in tokens]
        except Exception as e:
            logger.debug("Model id_to_token fallback failed: %s", e)

        if tokenizer_id not in self._warned_tokenizers:
            self._warned_tokenizers.add(tokenizer_id)
            logger.warning(
                "All token conversion methods failed for %s. Using placeholders.",
                type(tokenizer),
            )
        return [f"<TOKEN_{tid}>" for tid in token_ids]

    def _process_token(self, raw_token: str, preserve_space: bool = False) -> Optional[str]:
        """Shared token processing: strip subword markers, returning ``None`` for special tokens.

        Args:
            raw_token: Raw token string from the tokenizer vocabulary.
            preserve_space: If ``False`` (default), space-prefix markers (Ġ, ▁,
                leading space) are stripped entirely: the ``_clean_token`` path.
                If ``True``, space-prefix markers are replaced with a literal
                space: the ``_decode_raw_token`` path used for
                whitespace-preserving alignment.

        No metric calls this any more, through either wrapper. Both wrappers
        were used only by paths that measured positions in text rebuilt by
        concatenating cleaned token strings, and the rebuilt text is not the
        source: this method removes one leading space from a token
        rather than all of them, so a token whose surface is several spaces
        leaves residual spaces the source has no counterpart for, and the walk
        that mapped source positions into the reconstruction resynchronized on
        one of them and stayed wrong from there. The code metrics and the digit
        and operator metrics all read source character positions through the
        encoder's own offsets instead. Kept because it is unit-tested and
        because both wrappers delegate to it; do not wire it back into a metric.
        """
        # Membership in the tokenizer's declared set, not a surface pattern. The
        # pattern this replaced, ^(<\||\[).*(\|>|\])$, matched ordinary content
        # tokens and deleted them from the reconstruction: 2 vocabulary entries of
        # tokenizers/bpe.json (one of them '[...]'), 2 of apertus ('[]' and
        # '[][]') and 5 of llama3. Deleting '[...]' from the reconstruction of
        # 'y = a[...]' took the delimiter start and end alignment of that snippet
        # from 0.5 to 0.0 under tokenizers/bpe.json. The pattern also matched
        # neither '<s>' nor '</s>', so all 4 tokens bpe.json declares special and
        # 976 of apertus's 1000 were reconstructed as literal text.
        # self._special_tokens is None on the paths that never resolve a
        # tokenizer, where GENERIC_SPECIAL_TOKENS is the documented fallback.
        specials = (GENERIC_SPECIAL_TOKENS if self._special_tokens is None
                    else self._special_tokens)
        if raw_token in specials:
            return None

        # Decode SentencePiece byte-fallback tokens (`<0xNN>` → chr(NN)) before
        # the per-char table. The table can only express 1-char→1-char rewrites
        # and would leave the literal six-char `<0xNN>` in place, which then
        # corrupts char-to-token reconstruction for SP-byte-fallback tokenizers.
        raw_token = self._decode_byte_fallback(raw_token)

        # Build effective decode table: always start with defaults, overlay
        # probed table when available.  This is safe because the default
        # entries (Ġ→' ', Ċ→'\n', etc.) are harmless for tokenizers that
        # never emit those characters.
        if self._char_decode_table:
            table = {**self._DEFAULT_CHAR_DECODE, **self._char_decode_table}
        else:
            table = self._DEFAULT_CHAR_DECODE

        # Apply character decode table to ALL characters
        decoded = ''.join(table.get(ch, ch) for ch in raw_token)

        # Check subword markers on the decoded result, but only the ones this
        # specific tokenizer actually uses (self._subword_markers, resolved by
        # _detect_subword_markers). self._subword_markers is None on the paths
        # that never resolve a tokenizer; `or set()` treats that the same as
        # "resolved, uses none": strip nothing, rather than falling back
        # to stripping all three unconditionally. Applying, say, the WordPiece
        # '##' rule to a tokenizer that never declared or exhibited it is the
        # defect this gate exists to prevent: see _detect_subword_markers for
        # the measured damage (35 cl100k_base vocabulary entries altered, 24
        # for o200k_base, 1 for tokenizers/bpe.json) and for why "strip
        # nothing" is the safe default when detection is inconclusive.
        markers = self._subword_markers or set()
        if _WORDPIECE_CONTINUATION_PREFIX in markers and self._CONTINUATION.match(decoded):
            return decoded[2:]
        if _CLIP_BPE_END_OF_WORD_SUFFIX in markers and self._END_WORD.search(decoded):
            return decoded[:-4]
        if _SUBWORD_NMT_CONTINUATION_SUFFIX in markers and self._CONTINUATION_END.search(decoded):
            return decoded[:-2]

        # Handle leading space
        if decoded and decoded[0] == ' ':
            if preserve_space:
                return decoded
            return decoded[1:]

        return decoded

    def _clean_token(self, token: str) -> Optional[str]:
        """Strip subword markers from *token*, returning ``None`` for special tokens.

        No metric calls this any more: its only caller was
        ``_build_char_to_token_map``. See ``_process_token`` for why the
        reconstruction the two of them build is not what the metrics measure
        against. Kept because it is unit-tested; do not wire it back into a
        metric.
        """
        return self._process_token(token, preserve_space=False)

    def _build_char_to_token_map(
        self, token_strings: List[str]
    ) -> Tuple[str, List[int]]:
        """Build a mapping from character offset to token index.

        Returns ``(reconstructed_text, char_to_token)`` where
        ``char_to_token[i]`` is the token index that produced character *i*
        in the reconstructed text.

        No metric calls this any more. Operator isolation in
        ``DigitBoundaryMetrics.compute_per_text`` was the last caller, and it now
        resolves operators to tokens through the encoder's character offsets,
        which ``compute()`` already did. The reconstruction returned here is not
        the source text: concatenating cleaned token strings drops the space in
        ``"! ="``, so the operator regex reads a ``"!="`` there that the source
        does not contain. On ``"0! = 1, 5! = 120, and 20 >= 3."`` with
        tokenizers/bpe.json that reported 3 compound operators against the 1 the
        source has. Kept because it is unit-tested; do not wire it back into a
        metric.
        """
        reconstructed: List[str] = []
        char_to_token: List[int] = []

        for idx, raw_token in enumerate(token_strings):
            cleaned = self._clean_token(raw_token)
            if cleaned is None:
                continue
            for ch in cleaned:
                reconstructed.append(ch)
                char_to_token.append(idx)

        return "".join(reconstructed), char_to_token

    @staticmethod
    def _build_source_to_recon_map(
        source_text: str, recon_text: str
    ) -> List[Optional[int]]:
        """Map each source-text character position to its position in the
        reconstructed (whitespace-stripped) text.

        Characters that do not survive reconstruction (whitespace consumed by
        subword prefixes, for instance) get ``None``.

        This used to be a greedy forward scan that advanced the reconstruction
        pointer only on a match, so it could never re-synchronize: one character
        the reconstruction rendered differently left the pointer stuck and every
        later source character mapped to ``None``. That fires on ordinary input,
        because a byte-level BPE renders ``é`` as ``Ã©``, so the map died at the
        first non-ASCII character. Measured on FLORES with the bundled BPE, the
        share of digit spans that became unmeasurable was 2% for English, 44%
        for German, 84% for French, 97% for Russian and 100% for Arabic, which
        made the digit metrics effectively English-only. On the AST side an
        unmappable span is scored as misaligned, so adding one accented comment
        to a Python snippet dropped its alignment from 0.93 to 0.00.

        The scan now re-synchronizes after a divergence, by looking for a short
        run of characters that matches again within a bounded window on both
        sides. ``difflib.SequenceMatcher`` gives a better alignment but is not
        affordable here: on a 15,000-character snippet, the size this loader
        actually produces, it took 66 seconds against 3 milliseconds for the
        windowed scan, and it runs per snippet per tokenizer.

        The window bounds what can be recovered. A divergence longer than
        ``_RESYNC_WINDOW`` characters is not re-synchronized, and those source
        characters stay ``None``, which callers must treat as unmeasured rather
        than as misaligned.

        No metric calls this any more. The code metrics and the digit metrics
        both map source characters to tokens through the encoder's own offsets
        instead. The reason is that the reconstruction this maps into is built
        by concatenating cleaned token strings, and ``_process_token`` removes
        one leading space from a token rather than all of them, so a token whose
        surface is several spaces leaves residual spaces the source has no
        counterpart for. The re-synchronization above then matches one of
        them and every later position is off. On a four-snippet indented corpus
        that path measured 1 of 14 numbers for meta-llama/Meta-Llama-3-8B and 1
        of 14 for Qwen/Qwen2.5-7B, against 14 of 14 from the offsets. It is kept
        because it is unit-tested; do not wire it back into a metric.
        """
        source_to_recon: List[Optional[int]] = [None] * len(source_text)
        src_len, recon_len = len(source_text), len(recon_text)
        window, anchor = BaseMetrics._RESYNC_WINDOW, BaseMetrics._RESYNC_ANCHOR

        def _agreement(s: int, r: int) -> int:
            """How many of the next `anchor` source chars match from recon[r].

            Greedy, so source characters the reconstruction drops (whitespace,
            most often, since recon is whitespace-stripped) do not veto the
            match. A strict window comparison cannot be used for that reason:
            with spaces removed from recon, almost no 3-character source window
            has a literal counterpart.
            """
            matched = 0
            ri = r
            for si in range(s, min(s + window, src_len)):
                if matched >= anchor or ri >= recon_len:
                    break
                if source_text[si] == recon_text[ri]:
                    matched += 1
                    ri += 1
            return matched

        src_idx = recon_idx = 0
        while src_idx < src_len and recon_idx < recon_len:
            if source_text[src_idx] == recon_text[recon_idx]:
                source_to_recon[src_idx] = recon_idx
                src_idx += 1
                recon_idx += 1
                continue

            # Diverged. The old scan advanced only on a match, so it handled a
            # source character the reconstruction drops but could never step
            # over a character the reconstruction *adds* (a byte-level vocab
            # renders `é` as `Ã©`, and some post-processors prepend a token).
            # One such character left it stuck for the rest of the document.
            # Look ahead in the reconstruction for a position where the source
            # picks up again, and require several characters of agreement so a
            # coincidental single match does not pull the alignment off.
            limit = min(recon_idx + 1 + window, recon_len)
            jump = None
            for r in range(recon_idx + 1, limit):
                if recon_text[r] != source_text[src_idx]:
                    continue
                if _agreement(src_idx, r) >= min(anchor, src_len - src_idx):
                    jump = r
                    break

            if jump is not None:
                recon_idx = jump
                continue
            # This source character is absent from the reconstruction.
            src_idx += 1

        return source_to_recon

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_basic_stats(values: List[float]) -> Dict[str, Optional[float]]:
        """Compute basic statistics for a list of values.

        A dispersion over a single observation is undefined, so ``std`` and
        ``std_err`` are None when ``count`` is 1, not 0.0. Publishing 0.0 said
        the sample has no spread, which is a statement about a sample of one
        that no data supports, and the ``--input`` route makes a corpus that
        small an ordinary case rather than an edge one.

        The empty case defers to ``empty_stats()`` rather than repeating a
        second, contradictory answer to the same question: this function used
        to return six zeros where ``empty_stats()`` returns six Nones.
        """
        if not values:
            return BaseMetrics.empty_stats()

        n = len(values)
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values, ddof=1)) if n > 1 else None,
            'std_err': float(scipy.stats.sem(values)) if n > 1 else None,
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': n,
            'sum': sum(values)
        }
    
    @staticmethod
    def safe_divide(
        numerator: float,
        denominator: float,
        default: Optional[float] = DEFAULT_SAFE_DIVIDE_VALUE,
    ) -> Optional[float]:
        """Divide, returning *default* when the denominator is zero.

        *default* is ``None`` package-wide as of 1.0. It used to be ``0.0``,
        which made "there was nothing to measure" indistinguishable from "the
        measured value is zero" in every rate the pipeline publishes. A reader
        of the JSON could not tell a tokenizer that never emitted an UNK from
        one with no UNK token at all, or a domain with perfect whitespace
        fidelity from a domain containing no whitespace.

        Callers that genuinely want a number for the empty case must pass one
        explicitly, which makes the choice visible at the call site.
        """
        return numerator / denominator if denominator != 0 else default
    
    @staticmethod
    def empty_stats() -> Dict[str, Optional[float]]:
        """Statistics dict for the case where nothing could be measured.

        Every statistic is ``None``, not ``0.0``. ``count: 0`` is the tell that
        the entry is empty, but it sat beside six zeros that read as measured
        values, and consumers that charted `mean` drew a real-looking bar at
        the origin. ``count`` and ``sum`` stay numeric because zero of
        something is a true statement about the sample size.
        """
        return {
            'mean': None,
            'median': None,
            'std': None,
            'std_err': None,
            'min': None,
            'max': None,
            'count': 0,
            'sum': 0,
        }
    
    @abstractmethod
    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """If tokenized_data is None, uses input_provider data."""
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
    def flatten_all_tokens(tokenized_data: List[TokenizedData]) -> List[int]:
        """Flatten all tokens into a single list."""
        all_tokens = []
        for data in tokenized_data:
            all_tokens.extend(data.tokens)
        return all_tokens
    
    @staticmethod
    def get_unique_tokens(tokenized_data: List[TokenizedData]) -> set:
        """Get set of all unique token IDs."""
        unique_tokens = set()
        for data in tokenized_data:
            unique_tokens.update(data.tokens)
        return unique_tokens
    
