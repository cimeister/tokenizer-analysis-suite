"""
AST Boundary Alignment metrics for code tokenization evaluation.

Parses source code into an AST using tree-sitter, then measures the fraction
of AST node boundaries (identifiers, keywords, operators, literals,
delimiters) that coincide with token boundaries produced by the tokenizer.

Five AST node categories are tracked independently:

1. **Identifiers**: variable names, function names, class names, etc.
2. **Keywords**: language keywords (``if``, ``for``, ``return``, ...).
3. **Operators**: ``+``, ``==``, ``&&``, etc.
4. **Literals**: string, numeric, and boolean literals.
5. **Delimiters**: ``(``, ``)``, ``{``, ``}``, ``[``, ``]``, ``;``, ``,``.
"""

import math
import os
import pickle
import signal
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import logging

from .base import BaseMetrics, format_optional
from ..constants import AGGREGATION_MICRO_POOLED
from ..core.input_types import CODE_CORPUS, Corpus, TokenizedData
from ..core.input_providers import InputProvider

# Import constants and helpers from the lightweight worker module.
# The worker is kept free of heavy imports (BaseMetrics, InputProvider, etc.)
# so it can be spawned as a subprocess without pulling in the tokenizer stack.
from ._treesitter_worker import (
    CATEGORIES as _CATEGORIES_TUPLE,
    ERROR_SPANS_KEY,
    IDENTIFIER_TYPES as _IDENTIFIER_TYPES,
    LITERAL_TYPES as _LITERAL_TYPES,
    DELIMITER_CHARS as _DELIMITER_CHARS,
    NON_OPERATOR_PUNCTUATION as _NON_OPERATOR_PUNCTUATION,
    KNOWN_KEYWORDS as _KNOWN_KEYWORDS,
    classify_node as _classify_node_fn,
    extract_leaf_spans as _extract_leaf_spans_fn,
    parse_snippets as _parse_snippets,
)

_WHITESPACE_SIGNIFICANT_LANGS: Set[str] = {"python", "haskell"}

# Languages whose leaf types this metric does not classify correctly, mapped to
# why. They are excluded from the results rather than scored, because a wrong
# number is worse than an absent one: the identifier category simply does not
# fire for them, so their alignment and fragmentation figures describe a
# fraction of the code and would sit in a table beside languages that were
# measured properly.
#
# Measured on the bundled samples: identifier share of classified leaves is
# 0.073 for Swift, 0.058 for Kotlin and 0.000 for Perl, against 0.19 to 0.37 for
# every supported language. The cause is that `classify_node` does not know
# these grammars' identifier node types.
#
# Adding them is a contained change: extend IDENTIFIER_TYPES in
# _treesitter_worker.py with the node types below and re-measure the identifier
# share. It was left undone rather than guessed at. If you want one of these
# languages, please open an issue at
# https://github.com/cimeister/tokenizer-intrinsic-evals/issues
_UNSUPPORTED_CODE_LANGS: Dict[str, str] = {
    "swift": "identifier node type 'simple_identifier' is not classified "
             "(identifier share 0.073 against 0.19-0.37 for supported languages)",
    "kotlin": "identifier node type 'simple_identifier' is not classified "
              "(identifier share 0.058)",
    "perl": "identifier node types 'varname' and 'function' are not classified "
            "(identifier share 0.000)",
}

logger = logging.getLogger(__name__)

# Default timeout (seconds) for each per-language tree-sitter subprocess.
# Override with TOKEVAL_PARSE_TIMEOUT_S. The subprocess exists because a
# corrupt parse can abort the whole process, so the timeout is what turns a
# hung grammar into a named warning. It is generous because it also has to
# cover a loaded machine: the php grammar has timed out here at 120s on 3
# snippets under a concurrent test run, which the log reports as php not
# measured rather than php scoring zero.
DEFAULT_PARSE_TIMEOUT_S = int(os.environ.get("TOKEVAL_PARSE_TIMEOUT_S", "120"))


def _treesitter_pack_version() -> str:
    """Return the installed tree-sitter-language-pack version, or 'unknown'."""
    try:
        import importlib.metadata as _md

        return _md.version("tree-sitter-language-pack")
    except Exception:
        return "unknown"


def _identifier_stats(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Summarize identifier records, excluding spans that could not be mapped.

    An identifier no token covered has ``num_tokens is None``. That is a
    failure to measure, so it is counted and reported under ``unmappable`` but
    kept out of both rates. Treating it as a fragmented identifier with a token
    count of -1 made ``avg_tokens_per_identifier`` negative for C# under every
    tokenizer, which is not a possible token count.

    Returns None when nothing was measurable, so the caller can omit the entry
    rather than publish a zero.
    """
    total = len(items)
    mapped = [it for it in items if it["num_tokens"] is not None]
    unmappable = total - len(mapped)
    if not mapped:
        return None
    return {
        "fragmentation_rate": float(
            sum(1 for it in mapped if it["fragmented"]) / len(mapped)
        ),
        "avg_tokens_per_identifier": float(
            sum(it["num_tokens"] for it in mapped) / len(mapped)
        ),
        "count": len(mapped),
        "unmappable": unmappable,
    }


def parse_snippets_fenced(
    code_snippets: Dict[str, List[str]],
    lang_to_treesitter: Dict[str, str],
    timeout_s: int = DEFAULT_PARSE_TIMEOUT_S,
) -> Tuple[Dict[str, list], Dict[str, str]]:
    """Parse code snippets with tree-sitter in one subprocess per language.

    Tree-sitter uses a C backend. Two things make in-process parsing unsafe.
    First, earlier metrics call ``tokenizer.encode()`` through a Rust/C backend
    that can corrupt heap metadata, so the first tree-sitter malloc afterwards
    crashes. Second, some grammars corrupt the heap on their own. Measured
    2026-07-28 with tree-sitter-language-pack 0.13.0: calling
    ``get_parser("haskell").parse(...)`` directly on snippet 1 of the bundled
    synthetic Haskell sample aborts the process with
    ``malloc(): mismatching next->prev_size``, in a fresh interpreter with no
    other grammar loaded. Either way the process dies by signal and no
    ``except`` clause can catch it.

    The worker parses inside a daemon thread, and that same snippet currently
    survives there and returns correct spans. Do not rely on that: it is a
    side effect of the allocation pattern, not a fix. The fence is what makes
    both outcomes safe, because a worker that does abort is reported as a
    crashed language rather than taking the caller down.

    Spawning one subprocess per language contains the damage: a grammar that
    crashes takes its own process down and the others still return. The worker
    returns only categorized byte-offset spans, which is the one thing that
    needs the C library; byte-to-char mapping and indentation extraction are
    pure Python and stay in the caller.

    Never add an in-process fallback here. Falling back would reintroduce
    exactly the crash this function exists to contain.

    Returns:
        ``(parsed_spans, dropped)`` where *parsed_spans* maps language to a list
        of categorized-span dicts, and *dropped* maps each language that
        produced no spans to the reason. A language in *dropped* was not
        measured; it did not score zero. Callers must propagate that
        distinction rather than letting an absent language read as an absent
        finding.
    """
    worker_path = os.path.join(os.path.dirname(__file__), "_treesitter_worker.py")
    parsed_spans: Dict[str, list] = {}
    dropped: Dict[str, str] = {}

    for lang, snippets in code_snippets.items():
        if not snippets:
            continue
        if lang_to_treesitter.get(lang) is None:
            dropped[lang] = "no_grammar"
            logger.debug("No tree-sitter grammar for %s; skipping.", lang)
            continue

        tmp_in = tmp_out = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pkl", prefix=f"ts_in_{lang}_", delete=False
            ) as f_in:
                tmp_in = f_in.name
                pickle.dump(({lang: snippets}, lang_to_treesitter), f_in)

            with tempfile.NamedTemporaryFile(
                suffix=".pkl", prefix=f"ts_out_{lang}_", delete=False
            ) as f_out:
                tmp_out = f_out.name

            proc = subprocess.run(
                [sys.executable, worker_path, tmp_in, tmp_out],
                capture_output=True,
                timeout=timeout_s,
            )
            stderr_msg = proc.stderr.decode(errors="replace").strip()

            if proc.returncode != 0:
                rc = proc.returncode
                if rc < 0:
                    try:
                        sig_name = signal.Signals(-rc).name
                    except (ValueError, AttributeError):
                        sig_name = f"signal {-rc}"
                    dropped[lang] = f"grammar_crash:{sig_name}"
                    logger.error(
                        "  %s: tree-sitter worker killed by %s (return code %d). "
                        "The %r grammar in tree-sitter-language-pack %s crashed the "
                        "parser process, so %s was NOT measured and is absent from "
                        "the results. This is a native crash in the grammar, not a "
                        "property of any tokenizer. stderr:\n%s",
                        lang, sig_name, rc, lang_to_treesitter.get(lang),
                        _treesitter_pack_version(), lang, stderr_msg or "(empty)",
                    )
                else:
                    dropped[lang] = f"worker_exit:{rc}"
                    logger.error(
                        "  %s: tree-sitter worker exited with code %d; %s was NOT "
                        "measured. stderr:\n%s",
                        lang, rc, lang, stderr_msg or "(empty)",
                    )
                continue

            if stderr_msg:
                logger.info("Tree-sitter worker [%s]: %s", lang, stderr_msg)

            if not os.path.exists(tmp_out) or os.path.getsize(tmp_out) == 0:
                dropped[lang] = "empty_worker_output"
                logger.error(
                    "  %s: worker exited successfully but wrote no output (%s); "
                    "%s was NOT measured.",
                    lang, tmp_out, lang,
                )
                continue

            with open(tmp_out, "rb") as f:
                lang_result = pickle.load(f)
            parsed_spans.update(lang_result)
            logger.debug("  %s: parsed %d snippet(s) in subprocess.", lang, len(snippets))

        except subprocess.TimeoutExpired:
            dropped[lang] = f"timeout:{timeout_s}s"
            logger.warning(
                "  %s: tree-sitter worker timed out after %ds for %d snippet(s); "
                "%s was NOT measured.",
                lang, timeout_s, len(snippets), lang,
            )
        except Exception as e:
            dropped[lang] = f"worker_error:{type(e).__name__}"
            logger.error(
                "  %s: tree-sitter worker failed with %s: %s. Not falling back to "
                "in-process parsing, which would risk crashing this process. %s was "
                "NOT measured.",
                lang, type(e).__name__, e, lang,
            )
        finally:
            for tmp_path in (tmp_in, tmp_out):
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    return parsed_spans, dropped


class ASTBoundaryMetrics(BaseMetrics):
    """AST boundary alignment metrics for code tokenization.

    The code this metric measures is the corpus the run registered on the input
    provider, or, when there is none, code this metric loads itself from config
    paths or from the bundled samples. A registered corpus is read back from the
    provider already encoded, so it is encoded once for the whole run rather
    than once per metric; code this metric loaded itself is encoded here. Either
    way it does **not** use the ``tokenized_data`` parameter passed to
    :meth:`compute`.
    """

    _CATEGORIES = _CATEGORIES_TUPLE

    # Timeout (seconds) for each per-language tree-sitter subprocess.
    _PER_LANG_TIMEOUT = DEFAULT_PARSE_TIMEOUT_S

    # The registered code corpus, or None when this metric loaded its own. A
    # class attribute rather than an instance one because the tests build this
    # class with object.__new__ and set only the attributes they exercise; an
    # attribute that exists only after __init__ ran turns those into
    # AttributeError.
    _code_corpus: Optional[Corpus] = None

    def __init__(
        self,
        input_provider: InputProvider,
        code_config: Optional[Dict[str, str]] = None,
        max_snippets_per_lang: Optional[int] = None,
        max_snippet_chars: Optional[int] = None,
    ):
        super().__init__(input_provider)

        # Tree-sitter availability (lazy)
        self._treesitter_available: Optional[bool] = None

        # Languages the parser could not handle in the last compute() call,
        # mapped to why. Surfaced in the results so an absent language is not
        # read as a language that scored zero.
        self._dropped_languages: Dict[str, str] = {}

        # Load code data
        from ..loaders.code_data import CodeDataLoader

        self.code_loader = CodeDataLoader(
            code_config,
            max_snippets_per_lang=max_snippets_per_lang,
            max_snippet_chars=max_snippet_chars,
        )
        self.max_snippets_per_lang = self.code_loader.max_snippets_per_lang
        self.max_snippet_chars = self.code_loader.max_snippet_chars

        self._code_corpus = self._registered_corpus(CODE_CORPUS)
        if self._code_corpus is not None:
            # The run resolved the code corpus once and registered it. Loading
            # it again here read every configured file a second time and applied
            # the caps a second time, with nothing checking that the two loads
            # agreed, while this metric and the code domain of operator
            # isolation both reported on "the" code corpus.
            #
            # It still goes through the loader rather than being read straight
            # off the corpus, so that get_code_snippets keeps applying
            # max_snippets_per_lang as its final safety net.
            self.code_loader.code_snippets.update(
                {lang: list(texts) for lang, texts in self._code_corpus.texts.items()}
            )
        else:
            if code_config:
                self.code_loader.load_all()
                # Synthetic samples are the documented behaviour for an empty
                # code_config only. Substituting them for a config that named real
                # paths reports code metrics computed on toy snippets under the name
                # of the corpus the user asked for: measured 0.562 full AST
                # alignment on synthetic against 0.493 on StarCoder for the same
                # tokenizer.
                if not self.code_loader.code_snippets:
                    raise ValueError(
                        "The code config named "
                        f"{', '.join(sorted(code_config))} but no snippet was read "
                        "from any of those paths. Check that the directories hold "
                        "files with the expected extensions."
                    )

            # With no code config, synthetic samples are the documented input.
            if not self.code_loader.code_snippets:
                synthetic = CodeDataLoader.generate_synthetic_samples()
                for lang, snippets in synthetic.items():
                    self.code_loader.code_snippets.setdefault(lang, []).extend(snippets)

    def _shared_code_encodings(
        self, active_tokenizers: List[Tuple[str, Any]]
    ) -> Optional[Dict[str, Dict[Tuple[str, str], TokenizedData]]]:
        """The registered code corpus already encoded, indexed by (language, text).

        Returns None when no code corpus is registered, which tells
        :meth:`compute` to encode each snippet itself as it always has.

        The provider encodes a registered corpus once and memoizes it, so
        reading the encoding from there rather than calling
        ``encode_with_offsets`` per snippet means the corpus is encoded once for
        the whole run instead of once per metric that reads it. It is also the
        faster encode, because the provider encodes one batch per language.
        Measured with tokenizers/bpe.json over the 1500 files the config
        benchmarks/open_source/code_ast_config.json names: 10.29 s of
        per-snippet ``encode_with_offsets`` against 4.00 s for the provider's
        batched encode, and 0.00 s for every read after the first.
        ``HuggingFaceTokenizer`` and ``CustomBPETokenizer`` are the wrappers
        with a native batch path. ``UniMixLMTokenizer`` subclasses the first but
        overrides ``encode_batch_with_offsets`` back to a per-text loop, because
        langspec encoding scores each text against every per-language tokenizer.
        For it, and for every wrapper that inherits the base implementation, the
        method is a per-text loop, so only the single-encoding part of this
        applies.

        The index is keyed by ``(language, text)`` and never by position. The
        provider's encode keeps only texts satisfying ``text and text.strip()``,
        so its list for a language is shorter than the loader's snippet list
        whenever a snippet is empty or whitespace-only. Pairing the two lists by
        index after one such drop would score every later AST span against the
        following snippet's tokens, and nothing in the results would show it.
        The one cause of such a snippet this package used to produce,
        ``max_snippet_chars`` truncating an indented file down to its leading
        whitespace, is fixed where it arose, in
        ``CodeDataLoader._load_language``. A corpus supplied by a caller can
        still contain one, so the key stays the text. Two identical snippets
        collapse onto one entry, which is harmless: identical text encodes to
        identical ids and identical offsets.
        """
        if self._code_corpus is None:
            return None
        encoded = self.input_provider.get_corpus_data(CODE_CORPUS)
        lookup: Dict[str, Dict[Tuple[str, str], TokenizedData]] = {}
        for tok_name, records in encoded.items():
            per_tokenizer: Dict[Tuple[str, str], TokenizedData] = {}
            for record in records:
                # A record carrying no text cannot be matched to a snippet.
                # InputProvider._encode_corpus always sets it; a provider that
                # supplies its own records for this corpus need not. Such a
                # record is left out of the index, and the snippet it was meant
                # for then raises from compute() naming the tokenizer.
                if record.text is None:
                    continue
                per_tokenizer[(record.language, record.text)] = record
            lookup[tok_name] = per_tokenizer
        missing = sorted(
            name for name, _ in active_tokenizers if name not in lookup
        )
        if missing:
            raise ValueError(
                f"Tokenizer(s) {missing} report can_encode() true, which is how "
                "this metric selects the tokenizers it scores, but the shared "
                f"{CODE_CORPUS!r} corpus holds no encoding for them. "
                "InputProvider._encode_corpus selects on _can_encode_raw_text, "
                "which additionally requires a callable encode attribute, so "
                "the two predicates disagree for these tokenizers. Encoding "
                "them here instead would undo the single encoding this path "
                "exists for, and skipping them would drop them from the AST "
                "results with nothing in the output saying they are absent."
            )
        return lookup

    # ------------------------------------------------------------------
    # Tree-sitter helpers
    # ------------------------------------------------------------------

    def _ensure_treesitter(self) -> bool:
        """Lazily import tree-sitter.  Returns ``True`` if available."""
        if self._treesitter_available is not None:
            return self._treesitter_available
        try:
            import tree_sitter_language_pack as _ts_pack  # noqa: F401
            self._treesitter_available = True
        except ImportError:
            logger.warning(
                "tree-sitter-language-pack not installed. "
                "AST boundary metrics disabled. "
                "Install with: pip install tree-sitter-language-pack"
            )
            self._treesitter_available = False
        return self._treesitter_available

    # ------------------------------------------------------------------
    # AST node classification & span extraction
    # Delegated to _treesitter_worker (single source of truth).
    # ------------------------------------------------------------------

    _classify_node = staticmethod(_classify_node_fn)
    _extract_leaf_spans = staticmethod(_extract_leaf_spans_fn)

    @staticmethod
    def _byte_to_char_offsets(source_bytes: bytes) -> List[int]:
        """Map each byte offset to a character offset.

        For pure-ASCII text this is the identity mapping.  Returns a list
        of length ``len(source_bytes) + 1`` so that ``end_byte`` lookups
        (exclusive) work without special-casing.
        """
        result: List[int] = []
        source_str = source_bytes.decode("utf-8")
        char_idx = 0
        for ch in source_str:
            n_bytes = len(ch.encode("utf-8"))
            for _ in range(n_bytes):
                result.append(char_idx)
            char_idx += 1
        # Sentinel for exclusive end positions
        result.append(char_idx)
        return result

    # ------------------------------------------------------------------
    # Identifier token counting
    # ------------------------------------------------------------------

    @staticmethod
    def _count_identifier_tokens(
        char_start: int,
        char_end: int,
        source_to_recon: List[Optional[int]],
        char_to_token: List[int],
    ) -> Optional[int]:
        """Count the number of distinct token indices spanning a source character range.

        Uses the ``source_to_recon`` + ``char_to_token`` coordinate chain.
        Returns ``None`` if the span cannot be mapped.

        No metric calls this any more. The identifier counts come from
        ``_count_identifier_tokens_offsets``, which reads the encoder's
        character offsets. The chain this walks ends in text rebuilt by
        concatenating cleaned token strings, and that reconstruction is not the
        source: see ``BaseMetrics._process_token`` and
        ``_build_source_char_to_token_map`` for what it got wrong and by how
        much. Kept because it is unit-tested, and because
        ``_count_identifier_tokens_fast`` is checked against it; do not wire it
        back into a metric.
        """
        recon_positions = []
        for pos in range(char_start, min(char_end, len(source_to_recon))):
            rp = source_to_recon[pos]
            if rp is not None:
                recon_positions.append(rp)

        if not recon_positions:
            return None

        recon_start = min(recon_positions)
        recon_end = max(recon_positions) + 1  # exclusive

        if recon_end > len(char_to_token):
            return None

        return len(set(char_to_token[recon_start:recon_end]))

    # ------------------------------------------------------------------
    # Whitespace-preserving token decoding
    # ------------------------------------------------------------------

    def _decode_raw_token(self, raw_token: str) -> Optional[str]:
        """Decode a raw token string preserving whitespace.

        Unlike ``_clean_token`` (which strips space prefixes entirely),
        this replaces Ġ / ▁ / leading-space markers with a literal space.
        Returns ``None`` for special tokens.

        Delegates to :meth:`BaseMetrics._process_token` with
        ``preserve_space=True`` so that the branch logic stays in sync
        with ``_clean_token``.

        No metric calls this any more. It decoded the tokens of the
        reconstructed text the AST alignment used to be measured against, and
        the alignment now reads the encoder's offsets (see
        ``_build_source_char_to_token_map``). See ``_process_token`` for what the
        reconstruction got wrong. Kept because it is unit-tested; do not wire it
        back into a metric.
        """
        return self._process_token(raw_token, preserve_space=True)

    # ------------------------------------------------------------------
    # Source char → token map (whitespace-inclusive)
    # ------------------------------------------------------------------

    @staticmethod
    def _map_from_offsets(
        source_len: int,
        offsets: List[Tuple[int, int]],
    ) -> List[Optional[int]]:
        """Build a source-char → token-index map from encoding offsets.

        Each entry in *offsets* is ``(start_char, end_char)`` in the
        original source text for the corresponding token.  Any zero-width
        span (``start == end``) is skipped: a special token reported as
        ``(0, 0)`` is the usual case, and a zero-width span reported at any
        other position is skipped on the same test.

        Returns a list of length *source_len* where entry *i* is the
        token index covering that position, or ``None``.
        """
        result: List[Optional[int]] = [None] * source_len
        for tok_idx, (start, end) in enumerate(offsets):
            if start == end:
                # Zero-width span: no source coverage. A special token
                # (<s>, </s>) reported as (0, 0) is the usual case.
                continue
            for pos in range(start, min(end, source_len)):
                result[pos] = tok_idx
        return result

    @staticmethod
    def _trim_word_start_space_offsets(
        source_code: str,
        offsets: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Drop the word-start space from each token's claimed source range.

        A ByteLevel or SentencePiece vocabulary marks a word start by keeping
        one space inside the token (``Ġfoo``, ``▁foo``). Whether the reported
        offsets include that space is set by the ``trim_offsets`` flag of the
        ByteLevel post-processor, which is part of the tokenizer's configuration
        and not part of how it splits text. Measured on ``def foo():`` with the
        same ``Ġfoo`` token in both vocabularies: gpt2 ships
        ``trim_offsets: false`` and reports (3, 7) for ``' foo'``; gpt-neox-20b
        ships ``trim_offsets: true`` and reports (4, 7) for ``'foo'``. Setting
        the flag on gpt2 leaves the token ids byte-identical and changes only
        the offsets.

        Reading the raw offsets publishes that flag rather than the
        tokenization: an AST node beginning at ``foo`` starts mid-token under
        gpt2 and token-initial under gpt-neox-20b, for the same split. Measured
        over 169296 AST spans on 15 languages, ``full_alignment_rate`` was
        0.4327 for gpt2 against 0.7697 for gpt-neox-20b on raw offsets, and
        0.7905 against 0.7697 after this trim.

        Only one space is dropped, and only from a token that has something
        other than whitespace after it:

        * A token whose range is entirely whitespace keeps all of it. An
          indentation token has nothing else to cover, and the indentation
          metric reads these same positions.
        * A leading newline or tab is kept. ``Ċ}`` is a merge the tokenizer
          learned across a line break, so the ``}`` really does start
          mid-token, whereas the space in ``Ġfoo`` is the marker every
          word-initial token in these vocabularies carries. Dropping newlines
          and tabs as well raises ``full_alignment_rate`` from 0.5238 to 0.5384
          for llama-3 and from 0.5455 to 0.5579 for mistral-nemo, and leaves
          gpt2, gemma-2, bert-base, gpt-neox-20b and xlm-roberta unchanged.

        This is the convention ``BaseMetrics._process_token`` already applied
        when the alignment was measured against reconstructed text: it decoded
        the marker to a space and stripped one leading space, never a newline
        or a tab.
        """
        limit = len(source_code)
        trimmed: List[Tuple[int, int]] = []
        for start, end in offsets:
            stop = min(end, limit)
            if (
                start < stop
                and source_code[start] == " "
                and source_code[start:stop].strip()
            ):
                trimmed.append((start + 1, end))
            else:
                trimmed.append((start, end))
        return trimmed

    def _build_source_char_to_token_map(
        self,
        source_code: str,
        token_strings: List[str],
        offsets: Optional[List[Tuple[int, int]]] = None,
        tokenizer_name: Optional[str] = None,
    ) -> List[Optional[int]]:
        """Map each source character (including whitespace) to a token index.

        Requires *offsets* from ``encode_with_offsets``, which is exact.
        *tokenizer_name* only names the tokenizer in the error raised when the
        offsets are missing. *token_strings* is read nowhere in this method: it
        was the input the removed reconstruction fallback rebuilt its text from,
        and the parameter is kept so the existing call sites and tests do not
        have to change.

        There used to be a fallback that rebuilt the text by concatenating token
        strings and walked the two in step. It advanced only on an exact
        character match and never re-synchronized, so it stopped at the first
        character a byte-level vocabulary renders differently and mapped
        everything after it to None. Measured on a 45-character French snippet
        with tokenizers/bpe.json: 19 of 45 characters agreed with the offsets
        mapping and 26 were unmapped, against 0 unmapped from offsets. It
        reported no warning, so its output was indistinguishable from a real
        measurement.

        Failing here is the right behaviour: a wrapper that returns no offsets
        (pre-tokenized data, and script_bpe or mingram on text whose alignment
        their wrapper could not establish) cannot support a metric defined on
        source character positions, and guessing produced numbers that looked
        valid.

        The offsets are normalized by
        :meth:`_trim_word_start_space_offsets` first, so that the map does not
        depend on the tokenizer's ``trim_offsets`` setting.

        Returns a list of length ``len(source_code)`` where entry *i* is
        the token index covering source char *i*, or ``None``.
        """
        if offsets is None:
            named = f"tokenizer {tokenizer_name!r}" if tokenizer_name else "this tokenizer"
            raise ValueError(
                f"Code AST metrics need character offsets, and {named}'s "
                "encode_with_offsets() returned none. Offsets are the only exact "
                "way to say which token covers which source character; the "
                "previous fallback guessed by matching token strings against the "
                "source and silently mismapped most non-ASCII text. Use a "
                "tokenizer that reports offsets, or disable the code metrics "
                "with --no-code-ast."
            )
        return self._map_from_offsets(
            len(source_code),
            self._trim_word_start_space_offsets(source_code, offsets),
        )

    @staticmethod
    def _infer_indent_unit(
        indentation: List[Tuple[str, int, int]],
    ) -> int:
        """Detect the indentation unit size (in characters) for a snippet.

        Collects all non-zero whitespace widths and returns their GCD.
        Falls back to 1 if no indented lines exist or GCD is 0.
        """
        from math import gcd
        widths = [len(ws.expandtabs()) for ws, _, _ in indentation if ws]
        if not widths:
            return 1
        result = widths[0]
        for w in widths[1:]:
            result = gcd(result, w)
        return result if result > 0 else 1

    @staticmethod
    def _extract_line_indentation(
        source_code: str,
    ) -> List[Tuple[str, int, int]]:
        """Extract leading whitespace info for each non-blank line.

        Returns a list of ``(ws_string, line_char_start, ws_char_end)``
        tuples.  Blank / whitespace-only lines are excluded.  Lines with
        no indentation return ``ws_string=""``.
        """
        results: List[Tuple[str, int, int]] = []
        offset = 0
        for line in source_code.split("\n"):
            if line.strip():  # non-blank
                stripped = line.lstrip()
                ws_len = len(line) - len(stripped)
                ws_string = line[:ws_len]
                results.append((ws_string, offset, offset + ws_len))
            offset += len(line) + 1  # +1 for the newline
        return results

    # ------------------------------------------------------------------
    # Boundary alignment check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_boundary_alignment(
        char_start: int,
        char_end: int,
        source_to_recon: List[Optional[int]],
        char_to_token: List[int],
    ) -> Optional[Dict[str, bool]]:
        """Check whether an AST node's boundaries align with token boundaries.

        Parameters use *source-code* character coordinates following the
        half-open ``[start, end)`` convention (mirrors tree-sitter):

        * *char_start*: inclusive start in source-code character coordinates
        * *char_end*: exclusive end in source-code character coordinates

        Positions are translated to the reconstructed-text coordinate space
        via *source_to_recon* before checking against *char_to_token*.

        Returns ``None`` if the span cannot be mapped.

        No metric calls this any more. The alignment comes from
        ``_check_boundary_alignment_offsets``, which reads the encoder's
        character offsets. The reconstructed-text coordinate space this
        translates into is built by concatenating cleaned token strings and is
        not the source: see ``BaseMetrics._process_token`` and
        ``_build_source_char_to_token_map`` for what it got wrong and by how
        much. Kept because it is unit-tested, and because
        ``_check_boundary_alignment_fast`` is checked against it; do not wire it
        back into a metric.
        """
        # Map start position
        recon_start = None
        for pos in range(char_start, min(char_end, len(source_to_recon))):
            if source_to_recon[pos] is not None:
                recon_start = source_to_recon[pos]
                break

        # Map end position (find last mapped char before char_end)
        recon_end = None
        for pos in range(min(char_end - 1, len(source_to_recon) - 1), char_start - 1, -1):
            if pos >= 0 and pos < len(source_to_recon) and source_to_recon[pos] is not None:
                recon_end = source_to_recon[pos] + 1  # exclusive
                break

        if recon_start is None or recon_end is None or recon_start >= recon_end:
            return None
        if recon_end > len(char_to_token):
            return None

        # Start boundary: token changes at recon_start compared to previous
        start_aligned = (
            recon_start == 0
            or char_to_token[recon_start] != char_to_token[recon_start - 1]
        )

        # End boundary: token changes at recon_end compared to previous
        end_aligned = (
            recon_end >= len(char_to_token)
            or char_to_token[recon_end - 1] != char_to_token[recon_end]
        )

        fully_aligned = start_aligned and end_aligned
        cross_boundary = not fully_aligned

        return {
            "start_aligned": start_aligned,
            "end_aligned": end_aligned,
            "fully_aligned": fully_aligned,
            "cross_boundary": cross_boundary,
        }

    # ------------------------------------------------------------------
    # Numpy-accelerated helpers (reconstructed-text coordinates)
    # ------------------------------------------------------------------
    #
    # Neither of the two ``_fast`` helpers below is called by compute() or
    # compute_per_text() any more. They read positions in the text rebuilt by
    # concatenating cleaned token strings, and that reconstruction drops one
    # leading space per token instead of all of them, so a token whose surface
    # is several spaces (``ĠĠĠ`` in Llama 3, OLMo 2, Qwen 2.5 and Mistral NeMo)
    # left residual spaces in it that the source has no counterpart for. The
    # source-to-reconstruction walk then resynchronized on one of those residual
    # spaces and every position after it was off. Measured over 169296 AST spans
    # on 15 languages: full_alignment_rate 0.0717 for llama-3 against 0.5238
    # from the offsets, and 0.2429 for bert-base against 1.0000.
    #
    # The alignment and identifier paths now use the ``_offsets`` helpers below.
    # These two are kept only because they are unit-tested against their
    # list-based counterparts; do not wire them back into a metric.

    @staticmethod
    def _check_boundary_alignment_fast(
        char_start: int,
        char_end: int,
        s2r_arr: np.ndarray,
        c2t_arr: np.ndarray,
        c2t_len: int,
    ) -> Optional[Dict[str, bool]]:
        """Fast boundary alignment using pre-built numpy arrays.

        Same semantics as :meth:`_check_boundary_alignment` but operates
        on numpy ``int64`` arrays (``-1`` for unmapped positions) to avoid
        repeated Python list-indexing overhead.

        No metric calls this any more; its only callers are the tests that
        check it against :meth:`_check_boundary_alignment`. See the block
        comment above for the measured reason the reconstructed-text
        coordinates both of them read were abandoned.
        """
        s2r_len = len(s2r_arr)

        # Forward scan for recon_start
        hi = min(char_end, s2r_len)
        if char_start >= hi:
            return None
        segment = s2r_arr[char_start:hi]
        mapped = segment >= 0
        if not mapped.any():
            return None
        first_offset = int(mapped.argmax())
        recon_start = int(segment[first_offset])

        # Backward scan for recon_end (exclusive)
        last_offset = len(segment) - 1 - int(np.flip(mapped).argmax())
        recon_end = int(segment[last_offset]) + 1

        if recon_start >= recon_end or recon_end > c2t_len:
            return None

        start_aligned = (
            recon_start == 0
            or int(c2t_arr[recon_start]) != int(c2t_arr[recon_start - 1])
        )
        end_aligned = (
            recon_end >= c2t_len
            or int(c2t_arr[recon_end - 1]) != int(c2t_arr[recon_end])
        )
        fully_aligned = start_aligned and end_aligned

        return {
            "start_aligned": start_aligned,
            "end_aligned": end_aligned,
            "fully_aligned": fully_aligned,
            "cross_boundary": not fully_aligned,
        }

    @staticmethod
    def _count_identifier_tokens_fast(
        char_start: int,
        char_end: int,
        s2r_arr: np.ndarray,
        c2t_arr: np.ndarray,
        c2t_len: int,
    ) -> Optional[int]:
        """Fast identifier token counting using pre-built numpy arrays.

        Same semantics as :meth:`_count_identifier_tokens`.

        No metric calls this any more; its only callers are the tests that
        check it against :meth:`_count_identifier_tokens`. See the block
        comment above for the measured reason the reconstructed-text
        coordinates both of them read were abandoned.
        """
        s2r_len = len(s2r_arr)
        hi = min(char_end, s2r_len)
        if char_start >= hi:
            return None

        segment = s2r_arr[char_start:hi]
        mapped = segment[segment >= 0]

        if len(mapped) == 0:
            return None

        recon_start = int(mapped.min())
        recon_end = int(mapped.max()) + 1  # exclusive

        if recon_end > c2t_len:
            return None

        return int(np.unique(c2t_arr[recon_start:recon_end]).size)

    # ------------------------------------------------------------------
    # Offsets-based helpers (source-code coordinates, used by compute())
    # ------------------------------------------------------------------
    #
    # Both take *s2t_arr*: one entry per source character holding the index of
    # the token that covers it, or ``-1`` where no token does. It comes from
    # ``_build_source_char_to_token_map``, which reads the tokenizer's encoding
    # offsets. Character positions are the ones tree-sitter reports, so no
    # reconstruction and no resynchronization are involved.

    @staticmethod
    def _check_boundary_alignment_offsets(
        char_start: int,
        char_end: int,
        s2t_arr: np.ndarray,
    ) -> Optional[Dict[str, bool]]:
        """Check whether an AST node's boundaries coincide with token boundaries.

        *char_start* is inclusive and *char_end* exclusive, in source-code
        character positions (tree-sitter's convention).

        The test is the one :meth:`_check_boundary_alignment_fast` applies,
        stated in source coordinates: the token index must change at the span
        start and at the span end. "At the span start" compares against the
        nearest covered character before the span, because the characters
        between two tokens are the ones no token covers (whitespace, under a
        tokenizer such as bert-base-uncased that drops it). A span starting at
        the first covered character of the snippet counts as start-aligned, and
        one ending at the last counts as end-aligned.

        Returns ``None`` when no character of the span is covered by any token.
        The caller counts that as unmappable instead of scoring it as
        misaligned.
        """
        s_len = int(s2t_arr.shape[0])
        hi = min(char_end, s_len)
        if char_start >= hi:
            return None

        segment = s2t_arr[char_start:hi]
        covered = segment >= 0
        if not covered.any():
            return None

        first_offset = int(covered.argmax())
        last_offset = len(segment) - 1 - int(np.flip(covered).argmax())
        first_token = int(segment[first_offset])
        last_token = int(segment[last_offset])

        prev = char_start + first_offset - 1
        while prev >= 0 and int(s2t_arr[prev]) < 0:
            prev -= 1
        start_aligned = prev < 0 or int(s2t_arr[prev]) != first_token

        nxt = char_start + last_offset + 1
        while nxt < s_len and int(s2t_arr[nxt]) < 0:
            nxt += 1
        end_aligned = nxt >= s_len or int(s2t_arr[nxt]) != last_token

        fully_aligned = start_aligned and end_aligned

        return {
            "start_aligned": start_aligned,
            "end_aligned": end_aligned,
            "fully_aligned": fully_aligned,
            "cross_boundary": not fully_aligned,
        }

    @staticmethod
    def _count_identifier_tokens_offsets(
        char_start: int,
        char_end: int,
        s2t_arr: np.ndarray,
    ) -> Optional[int]:
        """Count the distinct tokens covering a source character range.

        Same coordinates and same ``-1``-for-uncovered convention as
        :meth:`_check_boundary_alignment_offsets`. Characters no token covers
        contribute nothing to the count, so an identifier a tokenizer splits
        across three tokens returns 3 whether or not the tokenizer also covers
        the whitespace around it.

        Returns ``None`` when no character of the span is covered, which the
        caller reports under ``unmappable`` rather than as a token count.
        """
        s_len = int(s2t_arr.shape[0])
        hi = min(char_end, s_len)
        if char_start >= hi:
            return None

        segment = s2t_arr[char_start:hi]
        covered = segment[segment >= 0]
        if covered.size == 0:
            return None

        return int(np.unique(covered).size)

    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------

    def compute(
        self, tokenized_data=None
    ) -> Dict[str, Any]:
        """Compute AST boundary alignment metrics.

        .. note::

           *tokenized_data* is **not used**: the metric reads the registered
           code corpus, or the snippets it loaded itself, and encodes them with
           each tokenizer.
        """
        if not self._ensure_treesitter():
            return {
                "ast_boundary_alignment": {
                    "error": "tree-sitter-language-pack not installed",
                },
                "identifier_fragmentation": {},
                "indentation_consistency": {},
            }

        if not self.code_loader.code_snippets:
            return {
                "ast_boundary_alignment": {
                    "error": "No code data loaded",
                },
                "identifier_fragmentation": {},
                "indentation_consistency": {},
            }

        # acc: tok_name -> code_lang -> category -> list of alignment dicts
        acc: Dict[str, Dict[str, Dict[str, List[Dict]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        # unmappable_acc: tok_name -> code_lang -> category -> count of spans no
        # token covered. Offsets cover every character a tokenizer encoded, so
        # this is expected to be 0; a non-zero count means the encoding left
        # part of the source uncovered and the affected spans were not measured.
        # They are excluded from the rates rather than scored as misaligned,
        # which is what the previous code did and what made every tokenizer
        # report the same span count. Under that code, 37128 of llama-3's 54216
        # identifier spans in a 15-language run had not been measured at all,
        # and every one of them was published as a boundary it had missed.
        unmappable_acc: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        # parse_error_acc: lang -> count of tree-sitter ERROR spans. An ERROR
        # span is a region of the source the grammar could not parse, so it is
        # not a syntax node and is not scored against token boundaries. It is
        # counted here and published so a reader can see how much of the corpus
        # failed to parse. There is one count per language rather than one per
        # tokenizer because parsing happens before any tokenizer is involved.
        parse_error_acc: Dict[str, int] = defaultdict(int)

        # ident_acc: tok -> lang -> [{text, num_tokens, fragmented}]
        ident_acc: Dict[str, Dict[str, List[Dict]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # indent_acc: tok -> lang -> [per-line records]
        # Each record: {"depth": int, "num_ws_tokens": int,
        #               "ws_width": int}
        indent_acc: Dict[str, Dict[str, List[Dict]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # ----- Phase 1: parse all snippets with tree-sitter -----
        # Tree-sitter uses a C backend.  On some platforms, earlier metrics
        # (DigitBoundaryMetrics, MorphScore, etc.) call tokenizer.encode()
        # through a Rust/C backend that corrupts heap metadata.  By the
        # time AST metrics run, the heap is already corrupted and the first
        # tree-sitter malloc triggers a crash.
        #
        # Fix: run tree-sitter work in *subprocesses* with their own clean
        # heaps.  One subprocess is spawned **per language** so that a
        # pathological snippet in one language cannot stall all others.
        # Each subprocess gets a moderate timeout; languages that exceed
        # it are skipped with a warning.
        #
        # The subprocess returns ONLY categorized byte-offset spans
        # (the one thing that requires tree-sitter's C library).
        # byte_to_char mapping and indentation extraction are pure Python
        # and are computed here in the main process.
        #
        # Structure: parsed_spans[code_lang] = [categorized_spans_dict, ...]

        from ..loaders.code_data import CodeDataLoader

        code_snippets = {
            lang: self.code_loader.get_code_snippets(lang)
            for lang in self.code_loader.get_languages()
            if lang not in _UNSUPPORTED_CODE_LANGS
        }
        skipped = [
            lang for lang in self.code_loader.get_languages()
            if lang in _UNSUPPORTED_CODE_LANGS
        ]
        if skipped:
            logger.warning(
                "Skipping %d language(s) whose leaf types this metric does not "
                "classify correctly, so they are absent from the results rather "
                "than scored on a fraction of their code: %s. See "
                "_UNSUPPORTED_CODE_LANGS in metrics/code_ast.py.",
                len(skipped),
                "; ".join(f"{lang} ({_UNSUPPORTED_CODE_LANGS[lang]})"
                          for lang in sorted(skipped)),
            )
        lang_to_treesitter = CodeDataLoader._LANG_TO_TREESITTER

        total_input = sum(len(v) for v in code_snippets.values())
        logger.info(
            "Phase 1: parsing %d snippet(s) across %d language(s) via "
            "per-language subprocesses (timeout=%ds each).",
            total_input, len(code_snippets), self._PER_LANG_TIMEOUT,
        )

        parsed_spans, dropped_languages = parse_snippets_fenced(
            code_snippets, lang_to_treesitter, timeout_s=self._PER_LANG_TIMEOUT
        )
        self._dropped_languages = dropped_languages
        if dropped_languages:
            logger.warning(
                "Phase 1: %d language(s) could not be parsed and are absent from "
                "the AST results (not scored zero): %s",
                len(dropped_languages),
                ", ".join(f"{k} ({v})" for k, v in sorted(dropped_languages.items())),
            )

        total_parsed = sum(len(v) for v in parsed_spans.values())
        # Count snippets that produced at least one AST span (non-empty parse).
        non_empty_parsed = sum(
            1
            for spans_list in parsed_spans.values()
            for spans_dict in spans_list
            if any(spans_dict.get(cat) for cat in self._CATEGORIES)
        )
        for lang in sorted(parsed_spans):
            lang_total = len(parsed_spans[lang])
            lang_non_empty = sum(
                1 for sd in parsed_spans[lang]
                if any(sd.get(cat) for cat in self._CATEGORIES)
            )
            logger.info(
                "  %s: %d snippet(s) loaded, %d with AST spans "
                "(%.0f%% non-empty)",
                lang, lang_total, lang_non_empty,
                100 * lang_non_empty / lang_total if lang_total else 0,
            )
        logger.info(
            "Phase 1 complete: parsed %d snippet(s) across %d language(s) "
            "(%d with AST spans). Starting Phase 2 (tokenizer encoding + alignment).",
            total_parsed, len(parsed_spans), non_empty_parsed,
        )

        # ----- Phase 2: encode with tokenizers and measure alignment -----
        # Loop order: (language → snippet → tokenizer) so that
        # tokenizer-independent per-snippet work (byte_to_char, indentation,
        # byte→char span conversion) is computed ONCE and reused across all
        # tokenizers.

        # Pre-filter encodable tokenizers (avoid repeated can_encode() checks).
        active_tokenizers: List[Tuple[str, Any]] = []
        for tok_name in self.tokenizer_names:
            tokenizer = self.input_provider.get_tokenizer(tok_name)
            if tokenizer.can_encode():
                active_tokenizers.append((tok_name, tokenizer))
            else:
                logger.info(
                    "Tokenizer %s cannot encode text; skipping AST metrics",
                    tok_name,
                )

        shared_encodings = self._shared_code_encodings(active_tokenizers)

        for code_lang, spans_list in parsed_spans.items():
            snippets = code_snippets[code_lang]
            is_ws_significant = code_lang in _WHITESPACE_SIGNIFICANT_LANGS

            for si, categorized_spans in enumerate(spans_list):
                snippet = snippets[si]

                # Tokenizer-independent pre-computation (ONCE per snippet)
                source_bytes = snippet.encode("utf-8")
                byte_to_char = self._byte_to_char_offsets(source_bytes)

                indentation = (
                    self._extract_line_indentation(snippet)
                    if is_ws_significant
                    else None
                )
                indent_unit = (
                    self._infer_indent_unit(indentation)
                    if indentation is not None
                    else None
                )

                # Pre-convert ALL byte spans to char spans.
                char_spans_by_category: Dict[str, List[Tuple[int, int]]] = {}
                for category, spans in categorized_spans.items():
                    if category == ERROR_SPANS_KEY:
                        # A region the grammar could not parse. Scoring it as a
                        # sixth AST category put unparsed source into
                        # full_alignment_rate: 4645 of gpt2's 1594200 spans on
                        # the benchmark corpus, 0.2914%. Counted here and
                        # published under parse_error_spans instead.
                        parse_error_acc[code_lang] += len(spans)
                        continue
                    if category not in self._CATEGORIES:
                        raise ValueError(
                            f"tree-sitter worker returned span category "
                            f"{category!r} for {code_lang}, which is neither one "
                            f"of {self._CATEGORIES} nor {ERROR_SPANS_KEY!r}. "
                            f"Scoring it would put it in the alignment rates; "
                            f"add it to CATEGORIES in _treesitter_worker.py or "
                            f"handle it here."
                        )
                    char_spans: List[Tuple[int, int]] = []
                    for byte_start, byte_end in spans:
                        if byte_start >= len(byte_to_char) or byte_end >= len(byte_to_char):
                            continue
                        char_spans.append(
                            (byte_to_char[byte_start], byte_to_char[byte_end])
                        )
                    char_spans_by_category[category] = char_spans

                # Pre-extract identifier texts.
                ident_char_spans = char_spans_by_category.get("identifier", [])
                ident_texts = [
                    snippet[c_start:c_end]
                    for c_start, c_end in ident_char_spans
                ]

                logger.debug(
                    "Phase 2 pre-computed: lang=%s snippet=%d/%d",
                    code_lang, si + 1, len(spans_list),
                )

                # Per-tokenizer work
                for tok_name, tokenizer in active_tokenizers:
                    if shared_encodings is None:
                        # No corpus was registered, so this metric loaded its
                        # own code and there is nothing to share. It encodes
                        # here, as it always has. scripts/run_ast_only.py,
                        # per_example.py and most of the test suite reach this
                        # branch.
                        try:
                            token_ids, enc_offsets = tokenizer.encode_with_offsets(
                                snippet
                            )
                        except Exception as e:
                            logger.debug(
                                "Encoding failed for %s on %s snippet: %s",
                                tok_name, code_lang, e,
                            )
                            continue
                    else:
                        record = shared_encodings[tok_name].get(
                            (code_lang, snippet)
                        )
                        if record is None:
                            raise ValueError(
                                f"The shared {CODE_CORPUS!r} corpus holds no "
                                f"encoding of {code_lang} snippet {si + 1} of "
                                f"{len(spans_list)} for tokenizer {tok_name!r}. "
                                "The snippet this metric parsed and the texts "
                                "the provider encoded are supposed to be the "
                                "same strings; they are not. The cause is "
                                "InputProvider._encode_corpus dropping a text "
                                "that is empty or whitespace-only. This is not "
                                "reachable through --max-code-file-chars: "
                                "CodeDataLoader drops a snippet its truncation "
                                "leaves whitespace-only, so look at the corpus "
                                "the provider was given. Scoring this snippet "
                                "against "
                                "another snippet's tokens is what keying the "
                                "lookup by text instead of by position "
                                "prevents, so it fails here rather than "
                                "reporting a number measured against the wrong "
                                "source. Snippet starts: "
                                f"{snippet[:60]!r}"
                            )
                        token_ids, enc_offsets = record.tokens, record.offsets
                    # A record whose tokens are empty lands here too, which is
                    # the case this skip already covered when the encoding
                    # happened above.
                    if not token_ids:
                        continue

                    token_strings = self._convert_ids_to_tokens(
                        tokenizer, token_ids
                    )

                    # Source character -> token index, straight from the
                    # encoding offsets. Built ONCE per (snippet, tokenizer) and
                    # shared by the alignment, identifier and indentation paths.
                    source_char_to_token = self._build_source_char_to_token_map(
                        snippet, token_strings,
                        offsets=enc_offsets, tokenizer_name=tok_name,
                    )
                    s2t_arr = np.fromiter(
                        (-1 if t is None else t for t in source_char_to_token),
                        dtype=np.int64,
                        count=len(source_char_to_token),
                    )

                    for category, char_spans in char_spans_by_category.items():
                        for span_idx, (c_start, c_end) in enumerate(char_spans):
                            alignment = self._check_boundary_alignment_offsets(
                                c_start, c_end, s2t_arr
                            )
                            if alignment is None:
                                unmappable_acc[tok_name][code_lang][category] += 1
                            else:
                                acc[tok_name][code_lang][category].append(alignment)

                            # Identifier fragmentation tracking
                            if category == "identifier":
                                num_tokens = self._count_identifier_tokens_offsets(
                                    c_start, c_end, s2t_arr
                                )
                                # num_tokens is None when no token covers any
                                # character of the identifier. That is a
                                # measurement failure, not a fragmented
                                # identifier, so it is recorded as unmappable and
                                # excluded from both rates rather than counted as
                                # fragmented with a -1 token count. Averaging
                                # that sentinel produced a negative
                                # avg_tokens_per_identifier for C# in every
                                # tokenizer.
                                ident_acc[tok_name][code_lang].append({
                                    "text": ident_texts[span_idx],
                                    "num_tokens": num_tokens,
                                    "fragmented": (
                                        None if num_tokens is None else num_tokens > 1
                                    ),
                                })

                    # Indentation consistency (whitespace-significant languages).
                    # Reuses the source_char_to_token map built above.
                    if indentation is not None:
                        for ws_string, line_start, ws_end in indentation:
                            if not ws_string:
                                continue
                            token_indices: List[int] = []
                            for pos in range(line_start, ws_end):
                                if pos < len(source_char_to_token):
                                    tidx = source_char_to_token[pos]
                                    if tidx is not None and (
                                        not token_indices
                                        or token_indices[-1] != tidx
                                    ):
                                        token_indices.append(tidx)
                            # Count only tokens whose surface is entirely
                            # whitespace. Selecting every token that *overlaps*
                            # the leading-whitespace range also caught the first
                            # code token whenever that token covered the last
                            # indent space too, which happens when the
                            # pre-tokenizer groups a leading space with the
                            # following word and a merge for the pair was
                            # learned. That token is code, not indentation.
                            ws_token_indices = [
                                ti for ti in token_indices
                                if self._is_whitespace_token(token_strings[ti])
                            ]
                            ws_width = len(ws_string.expandtabs())
                            depth = ws_width // indent_unit if indent_unit else ws_width
                            num_ws_tokens = len(ws_token_indices)
                            indent_acc[tok_name][code_lang].append({
                                "depth": depth,
                                "num_ws_tokens": num_ws_tokens,
                                "ws_width": ws_width,
                            })

        # Log Phase 2 summary: how many AST nodes contributed per tokenizer
        for tok_name in self.tokenizer_names:
            tok_node_count = sum(
                len(items)
                for lang_cats in acc.get(tok_name, {}).values()
                for items in lang_cats.values()
            )
            tok_lang_count = len(acc.get(tok_name, {}))
            tok_unmappable = sum(
                n
                for lang_cats in unmappable_acc.get(tok_name, {}).values()
                for n in lang_cats.values()
            )
            logger.info(
                "Phase 2 complete for %s: %d AST nodes aligned across %d language(s).",
                tok_name, tok_node_count, tok_lang_count,
            )
            if tok_unmappable:
                logger.warning(
                    "%s: %d AST span(s) had no token covering any of their "
                    "characters and were not measured. They are reported under "
                    "'unmappable' and left out of the alignment rates. The "
                    "encoding offsets did not cover part of the source.",
                    tok_name, tok_unmappable,
                )

        total_parse_errors = sum(parse_error_acc.values())
        if total_parse_errors:
            logger.warning(
                "tree-sitter emitted %d ERROR span(s) across %d language(s): %s. "
                "These are regions the grammar could not parse, so they are not "
                "AST nodes and are not scored against token boundaries. They are "
                "reported under 'parse_error_spans', which is a parse failure, "
                "not the token-side failure 'unmappable' counts.",
                total_parse_errors, len(parse_error_acc),
                ", ".join(
                    f"{lang}={n}" for lang, n in sorted(parse_error_acc.items())
                ),
            )

        return {
            "ast_boundary_alignment": self._build_results(
                acc, unmappable_acc, parse_error_acc
            ),
            "identifier_fragmentation": self._build_identifier_fragmentation_results(ident_acc),
            "indentation_consistency": self._build_indentation_consistency_results(indent_acc),
        }

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_results(
        self,
        acc: Dict[str, Dict[str, Dict[str, List[Dict]]]],
        unmappable_acc: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
        parse_error_acc: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Aggregate the per-span alignment records into the published results.

        *unmappable_acc* holds the spans no token covered, counted by tokenizer,
        language and category. They are not in *acc* because they were not
        measured. Every level of the output carries the count next to ``count``
        so a reader can see how much of the corpus each rate is computed over;
        it is 0 whenever the encoding offsets cover the whole source, which is
        the normal case.

        *parse_error_acc* holds the tree-sitter ERROR spans per language: source
        the grammar could not parse. It is published next to ``unmappable`` and
        deliberately named differently, because the two count different
        failures. ``unmappable`` is token-side (the encoding covered no
        character of a span that did parse) and differs by tokenizer;
        ``parse_error_spans`` is parse-side and is the same for every tokenizer.
        Neither is in the alignment rates.
        """
        unmappable_acc = unmappable_acc or {}
        parse_error_acc = parse_error_acc or {}
        total_parse_errors = sum(parse_error_acc.values())
        results: Dict[str, Any] = {
            "per_tokenizer": {},
            "summary": {},
            "metadata": {
                "description": (
                    "How often a token boundary coincides with an AST node "
                    "boundary, over the nodes of every programming language."
                ),
                "aggregation": AGGREGATION_MICRO_POOLED,
                "count_unit": "AST nodes",
                "parse_error_spans": (
                    "tree-sitter ERROR spans, meaning source the grammar could "
                    "not parse. They are not AST nodes, so they are excluded "
                    "from every rate here and counted instead. The count is a "
                    "property of the corpus and the grammar, so it is identical "
                    "for every tokenizer, unlike 'unmappable'."
                ),
            },
        }

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {
                "by_category": {},
                "by_language": {},
                "overall": {},
            }

            tok_acc = acc.get(tok_name, {})
            tok_unmappable = unmappable_acc.get(tok_name, {})

            all_full: List[float] = []
            all_start: List[float] = []
            all_end: List[float] = []
            all_cross: List[float] = []
            total_count = 0
            total_unmappable = 0
            languages_seen: set = set()

            for code_lang in sorted(
                set(tok_acc) | set(tok_unmappable) | set(parse_error_acc)
            ):
                lang_full: List[float] = []
                lang_start: List[float] = []
                lang_end: List[float] = []
                lang_cross: List[float] = []
                lang_unmappable = 0
                lang_parse_errors = int(parse_error_acc.get(code_lang, 0))

                lang_acc = tok_acc.get(code_lang, {})
                lang_unmappable_by_cat = tok_unmappable.get(code_lang, {})

                for category in sorted(set(lang_acc) | set(lang_unmappable_by_cat)):
                    items = lang_acc.get(category, [])
                    cat_unmappable = int(lang_unmappable_by_cat.get(category, 0))
                    if not items and not cat_unmappable:
                        continue

                    lang_unmappable += cat_unmappable

                    if category not in tok_data["by_category"]:
                        tok_data["by_category"][category] = {}

                    if not items:
                        # Every span of this category was unmappable, so there is
                        # no rate to report. The rates are null rather than 0.0
                        # so they cannot be read as "nothing aligned".
                        tok_data["by_category"][category][code_lang] = {
                            "start_alignment_rate": None,
                            "end_alignment_rate": None,
                            "full_alignment_rate": None,
                            "cross_boundary_rate": None,
                            "count": 0,
                            "unmappable": cat_unmappable,
                        }
                        continue

                    s_rates = [1.0 if it["start_aligned"] else 0.0 for it in items]
                    e_rates = [1.0 if it["end_aligned"] else 0.0 for it in items]
                    f_rates = [1.0 if it["fully_aligned"] else 0.0 for it in items]
                    c_rates = [1.0 if it["cross_boundary"] else 0.0 for it in items]

                    tok_data["by_category"][category][code_lang] = {
                        "start_alignment_rate": float(np.mean(s_rates)),
                        "end_alignment_rate": float(np.mean(e_rates)),
                        "full_alignment_rate": float(np.mean(f_rates)),
                        "cross_boundary_rate": float(np.mean(c_rates)),
                        "count": len(items),
                        "unmappable": cat_unmappable,
                    }

                    lang_full.extend(f_rates)
                    lang_start.extend(s_rates)
                    lang_end.extend(e_rates)
                    lang_cross.extend(c_rates)

                total_unmappable += lang_unmappable

                if lang_full:
                    tok_data["by_language"][code_lang] = {
                        "overall_full_alignment_rate": float(np.mean(lang_full)),
                        "overall_start_alignment_rate": float(np.mean(lang_start)),
                        "overall_end_alignment_rate": float(np.mean(lang_end)),
                        "overall_cross_boundary_rate": float(np.mean(lang_cross)),
                        "count": len(lang_full),
                        "unmappable": lang_unmappable,
                        "parse_error_spans": lang_parse_errors,
                    }
                    all_full.extend(lang_full)
                    all_start.extend(lang_start)
                    all_end.extend(lang_end)
                    all_cross.extend(lang_cross)
                    total_count += len(lang_full)
                    languages_seen.add(code_lang)
                elif lang_unmappable or lang_parse_errors:
                    # Nothing was measured for this language. It still gets an
                    # entry, because a language that produced only unparsable or
                    # uncovered spans is not the same as a language that was
                    # never loaded, and dropping it here would make the two look
                    # identical in the results.
                    tok_data["by_language"][code_lang] = {
                        "overall_full_alignment_rate": None,
                        "overall_start_alignment_rate": None,
                        "overall_end_alignment_rate": None,
                        "overall_cross_boundary_rate": None,
                        "count": 0,
                        "unmappable": lang_unmappable,
                        "parse_error_spans": lang_parse_errors,
                    }

            if all_full:
                tok_data["overall"] = {
                    "full_alignment_rate": float(np.mean(all_full)),
                    "start_alignment_rate": float(np.mean(all_start)),
                    "end_alignment_rate": float(np.mean(all_end)),
                    "cross_boundary_rate": float(np.mean(all_cross)),
                    "count": total_count,
                    "unmappable": total_unmappable,
                    "parse_error_spans": total_parse_errors,
                }

            results["per_tokenizer"][tok_name] = tok_data

            if all_full:
                results["summary"][tok_name] = {
                    "avg_full_alignment_rate": float(np.mean(all_full)),
                    "avg_start_alignment_rate": float(np.mean(all_start)),
                    "avg_end_alignment_rate": float(np.mean(all_end)),
                    "avg_cross_boundary_rate": float(np.mean(all_cross)),
                    "total_nodes_analyzed": total_count,
                    "total_nodes_unmappable": total_unmappable,
                    "total_parse_error_spans": total_parse_errors,
                    "languages_analyzed": len(languages_seen),
                }

        return results

    def _build_identifier_fragmentation_results(
        self, ident_acc: Dict[str, Dict[str, List[Dict]]]
    ) -> Dict[str, Any]:
        """Build identifier fragmentation results from accumulated data."""
        results: Dict[str, Any] = {
            "per_tokenizer": {},
            "summary": {},
            "metadata": {
                "description": (
                    "How often an identifier is split into more than one "
                    "token, over the identifier occurrences of every "
                    "programming language."
                ),
                "aggregation": AGGREGATION_MICRO_POOLED,
                "count_unit": "identifier occurrences",
            },
        }

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {"by_language": {}, "overall": {}}
            all_items: List[Dict] = []
            languages_seen: set = set()

            for code_lang in sorted(ident_acc.get(tok_name, {})):
                items = ident_acc[tok_name][code_lang]
                if not items:
                    continue

                stats = _identifier_stats(items)
                if stats is None:
                    continue
                tok_data["by_language"][code_lang] = stats
                all_items.extend(items)
                languages_seen.add(code_lang)

            overall_stats = _identifier_stats(all_items) if all_items else None
            if overall_stats is not None:
                tok_data["overall"] = overall_stats
                overall_frag = overall_stats["fragmentation_rate"]
                overall_avg = overall_stats["avg_tokens_per_identifier"]

            results["per_tokenizer"][tok_name] = tok_data

            if overall_stats is not None:
                results["summary"][tok_name] = {
                    "fragmentation_rate": float(overall_frag),
                    "avg_tokens_per_identifier": float(overall_avg),
                    "identifiers_analyzed": overall_stats["count"],
                    "identifiers_unmappable": overall_stats["unmappable"],
                    "languages_analyzed": len(languages_seen),
                }

        return results

    @staticmethod
    def _is_whitespace_token(token_string: str) -> bool:
        """Is this token's surface entirely whitespace?

        Decodes the byte-level and SentencePiece whitespace markers first, so
        'GGG' (three spaces under a ByteLevel vocabulary) counts and 'Greturn'
        does not. A token that carries any non-whitespace character is code, not
        indentation, however much whitespace it also carries.
        """
        if not token_string:
            return False
        decoded = token_string
        for marker, actual in BaseMetrics._DEFAULT_CHAR_DECODE.items():
            decoded = decoded.replace(marker, actual)
        return decoded.strip() == ""

    @staticmethod
    def _spearman_correlation(x: List[float], y: List[float]) -> float:
        """Compute Spearman rank correlation between two lists.

        Uses scipy.stats.spearmanr if available, otherwise falls back to a
        pure-Python implementation.

        Returns NaN when the correlation is undefined, which happens for fewer
        than two points and for constant input.  NaN rather than 0.0, because
        the callers already treat NaN as "not computed" and publish None for
        it, while 0.0 is a real measurement meaning depth and whitespace-token
        count are unrelated.  Publishing 0.0 for a constant input stated that
        finding without having measured it, and fed a spurious zero into
        avg_depth_proportionality_correlation, pulling the mean toward zero for
        every language whose correlation could not be computed.
        """
        n = len(x)
        if n < 2:
            return float("nan")
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(x, y)
            return float(rho)
        except ImportError:
            pass

        # Pure-Python rank correlation
        def _rank(vals):
            indexed = sorted(range(n), key=lambda i: vals[i])
            ranks = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    ranks[indexed[k]] = avg_rank
                i = j + 1
            return ranks

        rx = _rank(x)
        ry = _rank(y)
        d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
        denom = n * (n * n - 1)
        if denom == 0:
            return float("nan")
        rho = 1.0 - 6.0 * d_sq / denom
        return rho

    def _build_indentation_consistency_results(
        self, indent_acc: Dict[str, Dict[str, List[Dict]]]
    ) -> Dict[str, Any]:
        """Build indentation consistency results from accumulated data.

        Computes one metric per language per tokenizer:

        - ``depth_proportionality_correlation``: Spearman rho between
          indentation depth and the number of whitespace tokens. Spearman is
          rank-based, so this rewards the token count increasing *monotonically*
          with depth, not increasing in proportion to it. The name says
          proportionality; the statistic measures monotonicity.

        A ``pattern_stability_rate`` was removed before release. Once only
        whitespace-only tokens were counted, it was 1.0 for 11 of 12 tokenizers
        measured and took two distinct values in total, which is what theory
        predicts: a deterministic tokenizer encodes a fixed whitespace string
        one fixed way, so the rate can only drop when two different indent
        widths map to the same depth, a property of the source rather than of
        the tokenizer. The spread it showed beforehand came from the bug of
        counting the first code token, which made it measure which word each
        line started with.

        Depth is the line's leading-whitespace width divided by an indent unit
        inferred per snippet as the GCD of its non-zero indent widths. It does
        not come from the parse tree.
        """
        results: Dict[str, Any] = {
            "per_tokenizer": {},
            "summary": {},
            "metadata": {
                "depth_source": "leading whitespace width // per-snippet GCD indent unit",
                "depth_proportionality_correlation": (
                    "Spearman rho, so it measures monotonic increase of "
                    "whitespace-token count with depth, not proportionality"
                ),
                "counted_tokens": "whitespace-only tokens",
                "languages": sorted(_WHITESPACE_SIGNIFICANT_LANGS),
                "aggregation": AGGREGATION_MICRO_POOLED,
                "count_unit": "indented lines",
            },
        }

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {"by_language": {}}
            lang_correlations: List[float] = []
            languages_seen: set = set()
            # Pooled (depth, num_ws_tokens) pairs across every programming
            # language, for the micro-averaged "overall" block built below.
            pooled_depths: List[float] = []
            pooled_num_ws_tokens: List[float] = []

            for code_lang in sorted(indent_acc.get(tok_name, {})):
                records = indent_acc[tok_name][code_lang]
                if not records:
                    continue

                total_indented_lines = len(records)
                depths = [r["depth"] for r in records]
                num_ws_tokens = [r["num_ws_tokens"] for r in records]
                distinct_depths = len(set(depths))

                # Depth-proportionality correlation (Spearman ρ)
                if distinct_depths >= 3:
                    corr = self._spearman_correlation(
                        [float(d) for d in depths],
                        [float(t) for t in num_ws_tokens],
                    )
                else:
                    corr = float("nan")

                tok_data["by_language"][code_lang] = {
                    "depth_proportionality_correlation": float(corr) if not math.isnan(corr) else None,
                    "num_depth_levels": distinct_depths,
                    "total_indented_lines": total_indented_lines,
                    # count is the schema-wide name for how many items of the
                    # metric's count_unit this entry covers. The unit here is
                    # indented lines, so it repeats total_indented_lines.
                    "count": total_indented_lines,
                }

                if not math.isnan(corr):
                    lang_correlations.append(corr)
                languages_seen.add(code_lang)

                pooled_depths.extend(float(d) for d in depths)
                pooled_num_ws_tokens.extend(float(t) for t in num_ws_tokens)

            results["per_tokenizer"][tok_name] = tok_data

            if languages_seen:
                # Micro-averaged global: one Spearman correlation computed over
                # the pooled (depth, num_ws_tokens) pairs of every language
                # together, rather than the mean of the per-language
                # correlations in ``summary`` below. Indentation conventions
                # differ by language (Python uses 4 spaces, Go uses tabs,
                # Haskell aligns to an arbitrary column), so this pooled value
                # depends on the language mix of the code corpus. It is a
                # deliberate choice, not a defect; ``by_language`` above is
                # where a reader sees each language on its own.
                pooled_distinct_depths = len(set(pooled_depths))
                if pooled_distinct_depths >= 3:
                    pooled_corr = self._spearman_correlation(
                        pooled_depths, pooled_num_ws_tokens
                    )
                else:
                    pooled_corr = float("nan")

                tok_data["overall"] = {
                    "depth_proportionality_correlation": (
                        float(pooled_corr) if not math.isnan(pooled_corr) else None
                    ),
                    "num_depth_levels": pooled_distinct_depths,
                    "total_indented_lines": len(pooled_depths),
                }

                summary: Dict[str, Any] = {
                    "languages_analyzed": len(languages_seen),
                }
                if lang_correlations:
                    summary["avg_depth_proportionality_correlation"] = float(
                        np.mean(lang_correlations)
                    )
                results["summary"][tok_name] = summary

        return results

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print AST boundary alignment results."""
        ast = results.get("ast_boundary_alignment")
        if not ast:
            return

        if "error" in ast:
            print(f"\nAST BOUNDARY ALIGNMENT: {ast['error']}")
            return

        print("\n" + "=" * 60)
        print("AST BOUNDARY ALIGNMENT RESULTS")
        print("=" * 60)

        # Summary
        if "summary" in ast:
            print("\nSUMMARY STATISTICS")
            print("-" * 40)
            for tok_name in self.tokenizer_names:
                if tok_name in ast["summary"]:
                    s = ast["summary"][tok_name]
                    print(f"{tok_name}:")
                    print(f"  {'Full Alignment':25}: {s['avg_full_alignment_rate']:.3f}")
                    print(f"  {'Start Alignment':25}: {s['avg_start_alignment_rate']:.3f}")
                    print(f"  {'End Alignment':25}: {s['avg_end_alignment_rate']:.3f}")
                    print(f"  {'Cross-Boundary Rate':25}: {s['avg_cross_boundary_rate']:.3f}")
                    print(f"  {'Nodes Analyzed':25}: {s['total_nodes_analyzed']:,}")
                    print(f"  {'Nodes Unmappable':25}: {s.get('total_nodes_unmappable', 0):,}")
                    print(f"  {'Parse Error Spans':25}: {s.get('total_parse_error_spans', 0):,}")
                    print(f"  {'Languages':25}: {s['languages_analyzed']}")

        # By category
        if "per_tokenizer" in ast:
            print("\nBY CATEGORY")
            print("-" * 60)
            for tok_name in self.tokenizer_names:
                tok = ast["per_tokenizer"].get(tok_name, {})
                by_cat = tok.get("by_category", {})
                if not by_cat:
                    continue
                print(f"\n{tok_name}:")
                for category in self._CATEGORIES:
                    if category not in by_cat:
                        continue
                    lang_data = by_cat[category]
                    # A language whose spans were all unmappable has a null rate
                    # and count 0; it contributes only to the unmappable total.
                    measured = [
                        d for d in lang_data.values()
                        if d["count"] and d["full_alignment_rate"] is not None
                    ]
                    total_items = sum(d["count"] for d in measured)
                    unmappable = sum(d.get("unmappable", 0) for d in lang_data.values())
                    if total_items == 0:
                        continue
                    weighted_full = sum(
                        d["full_alignment_rate"] * d["count"] for d in measured
                    ) / total_items
                    print(
                        f"  {category:15}: "
                        f"full_align={weighted_full:.3f}  "
                        f"n={total_items}  "
                        f"unmappable={unmappable}"
                    )

            # By language
            print("\nBY LANGUAGE")
            print("-" * 60)
            for tok_name in self.tokenizer_names:
                tok = ast["per_tokenizer"].get(tok_name, {})
                by_lang = tok.get("by_language", {})
                if not by_lang:
                    continue
                print(f"\n{tok_name}:")
                for lang in sorted(by_lang):
                    d = by_lang[lang]
                    unmappable = d.get("unmappable", 0)
                    parse_errors = d.get("parse_error_spans", 0)
                    if d["overall_full_alignment_rate"] is None:
                        print(
                            f"  {lang:15}: not measured, "
                            f"unmappable={unmappable}  "
                            f"parse_errors={parse_errors}"
                        )
                        continue
                    print(
                        f"  {lang:15}: "
                        f"full_align={d['overall_full_alignment_rate']:.3f}  "
                        f"start={d['overall_start_alignment_rate']:.3f}  "
                        f"end={d['overall_end_alignment_rate']:.3f}  "
                        f"n={d['count']}  "
                        f"unmappable={unmappable}  "
                        f"parse_errors={parse_errors}"
                    )

        # --- Identifier Fragmentation ---
        ident = results.get("identifier_fragmentation")
        if ident and "summary" in ident and ident["summary"]:
            print("\nIDENTIFIER FRAGMENTATION")
            print("-" * 60)
            for tok_name in self.tokenizer_names:
                if tok_name in ident.get("summary", {}):
                    s = ident["summary"][tok_name]
                    print(f"{tok_name}:")
                    print(f"  {'Fragmentation Rate':25}: {s['fragmentation_rate']:.3f}")
                    print(f"  {'Avg Tokens/Identifier':25}: {s['avg_tokens_per_identifier']:.2f}")
                    print(f"  {'Identifiers Analyzed':25}: {s['identifiers_analyzed']:,}")
                    print(f"  {'Languages':25}: {s['languages_analyzed']}")

                    # By language breakdown
                    tok_detail = ident.get("per_tokenizer", {}).get(tok_name, {})
                    by_lang = tok_detail.get("by_language", {})
                    if by_lang:
                        for lang in sorted(by_lang):
                            d = by_lang[lang]
                            print(
                                f"    {lang:13}: "
                                f"frag_rate={d['fragmentation_rate']:.3f}  "
                                f"avg_tok={d['avg_tokens_per_identifier']:.2f}  "
                                f"n={d['count']}"
                            )

        # --- Indentation Consistency ---
        indent = results.get("indentation_consistency")
        if indent and "summary" in indent and indent["summary"]:
            print("\nINDENTATION CONSISTENCY")
            print("-" * 60)
            for tok_name in self.tokenizer_names:
                if tok_name in indent.get("summary", {}):
                    s = indent["summary"][tok_name]
                    print(f"{tok_name}:")
                    if "avg_depth_proportionality_correlation" in s:
                        print(f"  {'Avg Depth Correlation':25}: {s['avg_depth_proportionality_correlation']:.3f}")
                    print(f"  {'Languages':25}: {s['languages_analyzed']}")

                    tok_detail = indent.get("per_tokenizer", {}).get(tok_name, {})
                    by_lang = tok_detail.get("by_language", {})
                    if by_lang:
                        for lang in sorted(by_lang):
                            d = by_lang[lang]
                            corr = d.get("depth_proportionality_correlation")
                            print(
                                f"    {lang:13}: "
                                f"depth_corr={format_optional(corr)}  "
                                f"levels={d['num_depth_levels']}  "
                                f"lines={d['total_indented_lines']}"
                            )

        print("\n" + "=" * 60)

    # ------------------------------------------------------------------
    # Per-text entry point (independent of compute()'s aggregator pipeline)
    # ------------------------------------------------------------------

    def compute_per_text(
        self,
        tokenizer_obj: Any,
        source_code: str,
        language: str = "python",
    ) -> Dict[str, Any]:
        """Compute AST boundary alignment metrics for ONE source-code snippet
        under ONE tokenizer.

        Reuses :func:`parse_snippets_fenced` for parsing, then routes
        ``_check_boundary_alignment_offsets``,
        ``_count_identifier_tokens_offsets`` and
        ``_build_source_char_to_token_map`` at single-snippet granularity, which
        is the same chain ``compute()`` uses. The standard ``compute()``
        workflow is unaffected: this method snapshots and restores
        ``self._char_decode_table``, ``self._special_tokens`` and
        ``self._subword_markers`` and does not touch the aggregator state.

        *tokenizer_obj* has to report character offsets, either through a
        wrapper's ``encode_with_offsets`` or through a ``tokenizers.Encoding``
        with an ``offsets`` attribute. One that does not raises ``ValueError``
        naming it, rather than returning a row: the shortfall applies to every
        text the caller would pass, so it is not a per-row condition.

        Parsing goes through the same subprocess fence ``compute()`` uses. An
        earlier version parsed in-process on the theory that a single snippet
        does not accumulate heap pressure, which was wrong: a grammar can
        corrupt the heap on its own input (the Haskell grammar in
        tree-sitter-language-pack 0.13.0 aborts on the bundled synthetic
        sample), and that kills the calling process outright. The fence costs
        one subprocess spawn per call, which is the price of not taking down a
        caller iterating over a corpus.

        If a parse fails (crash, timeout, exception, or no matching grammar)
        all alignment scalars come back NaN with ``n_ast_nodes = 0`` and
        ``parse_status`` naming the reason, so the caller can drop the row.

        Returns a dict with keys: ``n_ast_nodes`` (spans scored, across all
        categories), ``n_unmappable_nodes`` (spans no token covered, which are
        left out of the rates), ``n_parse_error_spans`` (tree-sitter ERROR
        spans, source the grammar could not parse, also left out of the rates),
        ``full_alignment_rate``,
        ``start_alignment_rate``, ``end_alignment_rate``, plus per-category
        rates (``identifier_full_alignment_rate``, etc.),
        ``identifier_fragmentation_rate`` (fraction of identifiers split
        into more than one token), ``n_identifiers`` (identifiers the rate is
        computed over), ``n_identifiers_unmappable``, and ``n_tokens``.
        """
        from ..loaders.code_data import CodeDataLoader

        empty = {
            "n_ast_nodes": 0,
            "n_unmappable_nodes": 0,
            "n_parse_error_spans": 0,
            "full_alignment_rate": float("nan"),
            "start_alignment_rate": float("nan"),
            "end_alignment_rate": float("nan"),
            "identifier_full_alignment_rate": float("nan"),
            "keyword_full_alignment_rate": float("nan"),
            "operator_full_alignment_rate": float("nan"),
            "literal_full_alignment_rate": float("nan"),
            "delimiter_full_alignment_rate": float("nan"),
            "identifier_fragmentation_rate": float("nan"),
            "n_identifiers": 0,
            "n_identifiers_unmappable": 0,
            "n_tokens": 0,
            "parse_status": "empty",
        }

        if not source_code or not source_code.strip():
            return empty

        # Tree-sitter availability check (cheap if already set).
        if not self._ensure_treesitter():
            out = dict(empty)
            out["parse_status"] = "treesitter_unavailable"
            return out

        ts_name = CodeDataLoader._LANG_TO_TREESITTER.get(language)
        if ts_name is None:
            out = dict(empty)
            out["parse_status"] = f"no_grammar_for_{language}"
            return out

        # Parse through the same subprocess fence compute() uses, so a grammar
        # that crashes on this snippet cannot take the caller's process down.
        parsed, dropped = parse_snippets_fenced(
            {language: [source_code]},
            CodeDataLoader._LANG_TO_TREESITTER,
            timeout_s=self._PER_LANG_TIMEOUT,
        )
        if language in dropped:
            out = dict(empty)
            out["parse_status"] = dropped[language]
            return out
        spans_list = parsed.get(language, [])
        if not spans_list:
            out = dict(empty)
            out["parse_status"] = "parse_empty"
            return out
        categorized_spans = spans_list[0]
        if not any(categorized_spans.get(cat) for cat in self._CATEGORIES):
            # Parse succeeded but no AST nodes extracted (e.g. comment-only / empty)
            out = dict(empty)
            out["parse_status"] = "no_ast_nodes"
            try:
                # Still report token count for context.
                try:
                    ids_raw = tokenizer_obj.encode(source_code, add_special_tokens=False)
                except TypeError:
                    ids_raw = tokenizer_obj.encode(source_code)
                ids = list(ids_raw.ids) if hasattr(ids_raw, "ids") else list(ids_raw)
                out["n_tokens"] = len(ids)
            except Exception as exc:
                logger.debug("could not count tokens for an unparsed snippet "
                             "(%s: %s); n_tokens stays 0",
                             type(exc).__name__, exc)
            return out

        # Byte→char map for span conversion (mirrors compute() body).
        source_bytes = source_code.encode("utf-8")
        byte_to_char = self._byte_to_char_offsets(source_bytes)

        # ERROR spans are source the grammar could not parse. They are counted
        # and reported, not scored: including them in the rates would measure
        # token boundaries against a region that has no syntax structure. Same
        # rule as compute().
        n_parse_error_spans = len(categorized_spans.get(ERROR_SPANS_KEY, ()))

        char_spans_by_category: Dict[str, List[Tuple[int, int]]] = {}
        for category, spans in categorized_spans.items():
            if category == ERROR_SPANS_KEY:
                continue
            if category not in self._CATEGORIES:
                raise ValueError(
                    f"tree-sitter worker returned span category {category!r} "
                    f"for {language}, which is neither one of "
                    f"{self._CATEGORIES} nor {ERROR_SPANS_KEY!r}."
                )
            char_spans: List[Tuple[int, int]] = []
            for byte_start, byte_end in spans:
                if byte_start >= len(byte_to_char) or byte_end >= len(byte_to_char):
                    continue
                char_spans.append((byte_to_char[byte_start], byte_to_char[byte_end]))
            char_spans_by_category[category] = char_spans

        # Encode source. The offsets are what the alignment is measured
        # against, so they are required rather than optional here.
        try:
            enc_offsets: Optional[List[Tuple[int, int]]]
            if hasattr(tokenizer_obj, "encode_with_offsets"):
                token_ids, enc_offsets = tokenizer_obj.encode_with_offsets(source_code)
            else:
                try:
                    ids_raw = tokenizer_obj.encode(source_code, add_special_tokens=False)
                except TypeError:
                    ids_raw = tokenizer_obj.encode(source_code)
                token_ids = list(ids_raw.ids) if hasattr(ids_raw, "ids") else list(ids_raw)
                enc_offsets = (
                    [tuple(pair) for pair in ids_raw.offsets]
                    if hasattr(ids_raw, "offsets") else None
                )
        except Exception as e:
            out = dict(empty)
            out["parse_status"] = f"encode_error:{type(e).__name__}"
            return out

        if not token_ids:
            out = dict(empty)
            out["parse_status"] = "empty_encode"
            return out

        token_strings = self._convert_ids_to_tokens(tokenizer_obj, token_ids)
        source_char_to_token = self._build_source_char_to_token_map(
            source_code, token_strings,
            offsets=enc_offsets,
            tokenizer_name=(
                tokenizer_obj.get_name()
                if hasattr(tokenizer_obj, "get_name")
                else getattr(tokenizer_obj, "name_or_path", None)
                or type(tokenizer_obj).__name__
            ),
        )
        s2t_arr = np.fromiter(
            (-1 if t is None else t for t in source_char_to_token),
            dtype=np.int64,
            count=len(source_char_to_token),
        )

        # Per-category alignment counters (one record per AST span).
        per_cat_full: Dict[str, List[float]] = defaultdict(list)
        per_cat_start: Dict[str, List[float]] = defaultdict(list)
        per_cat_end: Dict[str, List[float]] = defaultdict(list)

        n_unmappable = 0
        n_idents = 0
        n_idents_unmappable = 0
        n_idents_fragmented = 0

        for category, char_spans in char_spans_by_category.items():
            for c_start, c_end in char_spans:
                alignment = self._check_boundary_alignment_offsets(
                    c_start, c_end, s2t_arr
                )
                if alignment is None:
                    # No token covers any character of this span, so it was
                    # not measured. Counting it as misaligned, which this
                    # used to do, reports a failure to measure as a property
                    # of the tokenizer.
                    n_unmappable += 1
                else:
                    per_cat_full[category].append(1.0 if alignment["fully_aligned"] else 0.0)
                    per_cat_start[category].append(1.0 if alignment["start_aligned"] else 0.0)
                    per_cat_end[category].append(1.0 if alignment["end_aligned"] else 0.0)

                if category == "identifier":
                    n_tok = self._count_identifier_tokens_offsets(
                        c_start, c_end, s2t_arr
                    )
                    if n_tok is None:
                        n_idents_unmappable += 1
                    else:
                        n_idents += 1
                        if n_tok > 1:
                            n_idents_fragmented += 1

        # Aggregate
        all_full = [v for vals in per_cat_full.values() for v in vals]
        all_start = [v for vals in per_cat_start.values() for v in vals]
        all_end = [v for vals in per_cat_end.values() for v in vals]
        n_total = len(all_full)

        def _rate(values: List[float]) -> float:
            return float(np.mean(values)) if values else float("nan")

        result: Dict[str, Any] = {
            "n_ast_nodes": n_total,
            "n_unmappable_nodes": n_unmappable,
            "n_parse_error_spans": n_parse_error_spans,
            "full_alignment_rate": _rate(all_full),
            "start_alignment_rate": _rate(all_start),
            "end_alignment_rate": _rate(all_end),
            "n_identifiers": n_idents,
            "n_identifiers_unmappable": n_idents_unmappable,
            "identifier_fragmentation_rate": (
                (n_idents_fragmented / n_idents) if n_idents else float("nan")
            ),
            "n_tokens": len(token_ids),
            "parse_status": "ok",
        }
        for cat in self._CATEGORIES:
            result[f"{cat}_full_alignment_rate"] = _rate(per_cat_full.get(cat, []))
        return result
