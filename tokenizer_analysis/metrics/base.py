"""
Base metrics class for unified TokenizedData interface - skeleton only.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Tuple
import re
import numpy as np
import scipy
import scipy.stats
from collections import defaultdict
import logging

from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider
from ..core.tokenizer_wrapper import resolve_special_token_strings
from ..constants import (
    DEFAULT_SAFE_DIVIDE_VALUE,
    GENERIC_SPECIAL_TOKENS,
    PERCENTAGE_MULTIPLIER,
    MAX_ERROR_DISPLAY_COUNT,
)

logger = logging.getLogger(__name__)

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

    # Pre-compiled regex patterns for subword marker handling.
    # Shared by DigitBoundaryMetrics, ASTBoundaryMetrics and MorphScoreMetrics.
    # Bounded re-sync for _build_source_to_recon_map. The window caps how long
    # a divergence can be and still be recovered; the anchor is how many
    # characters must agree before the scan trusts the new alignment. A
    # 1-character anchor re-syncs on any coincidental single match.
    _RESYNC_WINDOW = 32
    _RESYNC_ANCHOR = 3

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

    def _set_tokenizer_context(
        self,
        char_decode_table: Optional[Dict[str, str]],
        special_tokens: Optional[Set[str]],
        subword_markers: Optional[Set[str]],
    ) -> None:
        """Assign the three per-tokenizer attributes _process_token reads, together.

        One entry point for _char_decode_table, _special_tokens and
        _subword_markers so a call site cannot update two of them and forget
        the third, which is what happened before _subword_markers existed:
        every _char_decode_table assignment in this codebase had a
        _special_tokens assignment next to it, added by hand at each site.

        Takes already-resolved values rather than a tokenizer to resolve
        itself, because some call sites (ASTBoundaryMetrics.compute()'s
        per-snippet loop) resolve once per tokenizer into a dict outside a
        per-snippet loop and only assign per snippet; resolving here on every
        call would reintroduce the repeated tokenizer.encode() probing that
        precompute step exists to avoid.
        """
        self._char_decode_table = char_decode_table
        self._special_tokens = special_tokens
        self._subword_markers = subword_markers

    def _clear_tokenizer_context(self) -> None:
        """Reset the three per-tokenizer attributes to the "none resolved" state.

        _process_token's fallbacks then apply: GENERIC_SPECIAL_TOKENS for
        special tokens, the bundled default char-decode table, and "strip no
        subword marker" (the same state as __init__).
        """
        self._set_tokenizer_context(None, None, None)

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
            except Exception:
                pass
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
        """Strip subword markers from *token*, returning ``None`` for special tokens."""
        return self._process_token(token, preserve_space=False)

    def _build_char_to_token_map(
        self, token_strings: List[str]
    ) -> Tuple[str, List[int]]:
        """Build a mapping from character offset to token index.

        Returns ``(reconstructed_text, char_to_token)`` where
        ``char_to_token[i]`` is the token index that produced character *i*
        in the reconstructed text.
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
    def compute_basic_stats(values: List[float]) -> Dict[str, float]:
        """Compute basic statistics for a list of values."""
        if not values:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'std_err': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0,
                'sum': 0
            }
            
        n = len(values)
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values, ddof=1)) if n > 1 else 0.0,
            'std_err': float(scipy.stats.sem(values)) if n > 1 else 0.0,
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
                    'percent_difference': self.safe_divide(abs(val1 - val2), (val1 + val2) / 2, 0.0) * PERCENTAGE_MULTIPLIER
                }
        
        return comparisons
    
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
    
    @staticmethod
    def validate_non_empty_data(data: Any, name: str) -> None:
        """Raise ValueError if data is empty."""
        if not data:
            raise ValueError(f"{name} cannot be empty")
    
    @staticmethod
    def validate_minimum_count(items: List[Any], min_count: int, name: str) -> None:
        """Raise ValueError if len(items) < min_count."""
        if len(items) < min_count:
            raise ValueError(f"{name} must have at least {min_count} items, got {len(items)}")
    
    @staticmethod
    def validate_positive_number(value: float, name: str) -> None:
        """Raise ValueError if value <= 0."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    
    @staticmethod
    def truncate_for_display(items: List[Any], max_count: int = MAX_ERROR_DISPLAY_COUNT) -> List[Any]:
        if len(items) <= max_count:
            return items
        return items[:max_count]
    
    @staticmethod
    def format_list_for_display(items: List[Any], max_count: int = MAX_ERROR_DISPLAY_COUNT) -> str:
        if len(items) <= max_count:
            return str(items)
        
        truncated = items[:max_count]
        return f"{truncated}... (showing {max_count}/{len(items)})"
    
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
    
