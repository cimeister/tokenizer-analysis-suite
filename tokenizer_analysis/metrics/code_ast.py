"""
AST Boundary Alignment metrics for code tokenization evaluation.

Parses source code into an AST using tree-sitter, then measures the fraction
of AST node boundaries (identifiers, keywords, operators, literals,
delimiters) that coincide with token boundaries produced by the tokenizer.

Five AST node categories are tracked independently:

1. **Identifiers** -- variable names, function names, class names, etc.
2. **Keywords** -- language keywords (``if``, ``for``, ``return``, ...).
3. **Operators** -- ``+``, ``==``, ``&&``, etc.
4. **Literals** -- string, numeric, and boolean literals.
5. **Delimiters** -- ``(``, ``)``, ``{``, ``}``, ``[``, ``]``, ``;``, ``,``.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import logging

from .base import BaseMetrics
from ..core.input_providers import InputProvider

logger = logging.getLogger(__name__)

# ======================================================================
# AST node category constants
# ======================================================================

_IDENTIFIER_TYPES: Set[str] = {
    "identifier",
    "type_identifier",
    "field_identifier",
    "property_identifier",
    "shorthand_property_identifier",
    "shorthand_property_identifier_pattern",
    "variable_name",
    "name",
    "attribute",
    "word",  # bash
}

_LITERAL_TYPES: Set[str] = {
    "string",
    "string_literal",
    "string_content",
    "string_fragment",
    "raw_string_literal",
    "char_literal",
    "character_literal",
    "template_string",
    "heredoc_body",
    "integer",
    "integer_literal",
    "decimal_integer_literal",
    "hex_integer_literal",
    "octal_integer_literal",
    "binary_integer_literal",
    "float",
    "float_literal",
    "decimal_floating_point_literal",
    "number",
    "number_literal",
    "true",
    "false",
    "none",
    "null",
    "nil",
    "boolean",
    "boolean_literal",
    "null_literal",
    "regex",
    "regex_pattern",
}

_DELIMITER_CHARS: Set[str] = set("(){}[];,")

# Punctuation that is syntactically not an operator in most languages.
_NON_OPERATOR_PUNCTUATION: Set[str] = {
    ".", ":", "::", "@", "#", "...", "\\",
}

# Keywords that may appear as named nodes in some grammars
_KNOWN_KEYWORDS: Set[str] = {
    "if", "else", "elif", "for", "while", "do", "switch", "case",
    "break", "continue", "return", "yield", "class", "struct",
    "enum", "interface", "trait", "impl", "def", "func", "fn",
    "function", "var", "let", "const", "static", "public", "private",
    "protected", "import", "from", "export", "module", "package",
    "try", "catch", "except", "finally", "throw", "throws", "raise",
    "async", "await", "match", "where", "in", "is", "as", "not",
    "and", "or", "with", "lambda", "new", "delete", "typeof",
    "instanceof", "void", "self", "this", "super",
    "then", "end", "local", "elsif", "unless", "until",
    "begin", "rescue", "ensure", "when", "val", "object", "extends",
    "override", "abstract", "final", "sealed", "lazy",
    "mut", "ref", "pub", "crate", "mod", "use", "extern",
    "type", "foreach", "namespace", "using",
}

_WHITESPACE_SIGNIFICANT_LANGS: Set[str] = {"python", "haskell"}


class ASTBoundaryMetrics(BaseMetrics):
    """AST boundary alignment metrics for code tokenization.

    This metric loads its own code data (from config paths or synthetic
    samples) and encodes it with each tokenizer.  It does **not** use the
    ``tokenized_data`` parameter passed to :meth:`compute`.
    """

    _CATEGORIES = ("identifier", "keyword", "operator", "literal", "delimiter")

    def __init__(
        self,
        input_provider: InputProvider,
        code_config: Optional[Dict[str, str]] = None,
        max_snippets_per_lang: int = 100,
    ):
        super().__init__(input_provider)

        # Tree-sitter availability (lazy)
        self._treesitter_available: Optional[bool] = None
        self._parser_cache: Dict[str, Any] = {}

        # Load code data
        from ..loaders.code_data import CodeDataLoader

        self.code_loader = CodeDataLoader(
            code_config, max_snippets_per_lang=max_snippets_per_lang
        )
        self.max_snippets_per_lang = max_snippets_per_lang

        if code_config:
            self.code_loader.load_all()

        # If no data was loaded from config, use synthetic samples
        if not self.code_loader.code_snippets:
            synthetic = CodeDataLoader.generate_synthetic_samples()
            for lang, snippets in synthetic.items():
                self.code_loader.code_snippets.setdefault(lang, []).extend(snippets)

    # ------------------------------------------------------------------
    # Tree-sitter helpers
    # ------------------------------------------------------------------

    def _ensure_treesitter(self) -> bool:
        """Lazily import tree-sitter.  Returns ``True`` if available."""
        if self._treesitter_available is not None:
            return self._treesitter_available
        try:
            import tree_sitter_language_pack as _ts_pack  # noqa: F401
            self._ts_pack = _ts_pack
            self._treesitter_available = True
        except ImportError:
            logger.warning(
                "tree-sitter-language-pack not installed. "
                "AST boundary metrics disabled. "
                "Install with: pip install tree-sitter-language-pack"
            )
            self._treesitter_available = False
        return self._treesitter_available

    def _get_parser(self, lang: str):
        """Get or create a cached tree-sitter parser for *lang*."""
        if lang in self._parser_cache:
            return self._parser_cache[lang]

        from ..loaders.code_data import CodeDataLoader

        ts_name = CodeDataLoader._LANG_TO_TREESITTER.get(lang)
        if ts_name is None:
            return None

        try:
            parser = self._ts_pack.get_parser(ts_name)
            self._parser_cache[lang] = parser
            return parser
        except Exception as e:
            logger.warning("Failed to create tree-sitter parser for %s: %s", lang, e)
            return None

    # ------------------------------------------------------------------
    # AST node classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_node(node) -> Optional[str]:
        """Classify a tree-sitter leaf node into one of the five categories.

        Returns ``'identifier'``, ``'keyword'``, ``'operator'``,
        ``'literal'``, ``'delimiter'``, or ``None``.
        """
        node_type = node.type
        text = node.text.decode("utf-8") if isinstance(node.text, bytes) else node.text

        if not text:
            return None

        # Delimiters
        if text in _DELIMITER_CHARS and len(text) == 1:
            return "delimiter"

        # Identifiers — checked before keywords intentionally: if the grammar
        # assigns an identifier type, we trust that classification even when
        # the text appears in _KNOWN_KEYWORDS (e.g. ``self`` in Python is
        # a conventional identifier, not a reserved keyword).
        if node_type in _IDENTIFIER_TYPES:
            return "identifier"

        # Literals
        if node_type in _LITERAL_TYPES:
            return "literal"

        # Keywords: anonymous nodes whose type equals their text and is
        # alphabetic (tree-sitter represents keyword tokens this way).
        if not node.is_named and node_type == text and text.isalpha():
            return "keyword"

        # Some grammars expose keywords as named nodes
        if node.is_named and text in _KNOWN_KEYWORDS and text.isalpha():
            return "keyword"

        # Operators: anonymous non-alphanumeric leaf nodes that aren't
        # delimiters, whitespace, or structural punctuation
        if not node.is_named and not text.isalnum() and text not in _DELIMITER_CHARS:
            stripped = text.strip()
            if stripped and not stripped.isspace() and stripped not in _NON_OPERATOR_PUNCTUATION:
                return "operator"

        return None

    # ------------------------------------------------------------------
    # AST span extraction
    # ------------------------------------------------------------------

    def _extract_leaf_spans(
        self, tree
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Extract categorised leaf-node spans from a parse tree.

        Returns a dict mapping category names to lists of
        ``(start_byte, end_byte)`` spans.
        """
        categorized: Dict[str, List[Tuple[int, int]]] = {
            cat: [] for cat in self._CATEGORIES
        }

        def _walk(node):
            # Skip error nodes
            if node.type == "ERROR":
                return
            if node.child_count == 0:
                cat = self._classify_node(node)
                if cat is not None:
                    start = node.start_byte
                    end = node.end_byte
                    if start < end:
                        categorized[cat].append((start, end))
            else:
                for child in node.children:
                    _walk(child)

        _walk(tree.root_node)
        return categorized

    # ------------------------------------------------------------------
    # Byte ↔ character offset conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _byte_to_char_offsets(source_bytes: bytes) -> List[int]:
        """Map each byte offset to a character offset.

        For pure-ASCII text this is the identity mapping.  Returns a list
        of length ``len(source_bytes) + 1`` so that ``end_byte`` lookups
        (exclusive) work without special-casing.
        """
        byte_to_char: List[int] = []
        source_str = source_bytes.decode("utf-8")
        char_idx = 0
        for ch in source_str:
            n_bytes = len(ch.encode("utf-8"))
            for _ in range(n_bytes):
                byte_to_char.append(char_idx)
            char_idx += 1
        # Sentinel for exclusive end positions
        byte_to_char.append(char_idx)
        return byte_to_char

    # ------------------------------------------------------------------
    # Source → reconstructed-text coordinate mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _build_source_to_recon_map(
        source_code: str, recon_text: str
    ) -> List[Optional[int]]:
        """Map each source-code character position to the corresponding
        position in the reconstructed text.

        Uses a greedy forward scan.  Prefers exact (case-sensitive) matches;
        falls back to case-insensitive only when no exact match is available.
        Characters dropped during reconstruction (whitespace consumed by
        subword prefixes) get ``None``.

        **Assumption:** The reconstructed text preserves the identity and
        order of non-whitespace characters from the source.  If a tokenizer
        performs case-folding or character-dropping (e.g. ``"aAa"`` →
        ``"aaa"``), the greedy scan may pair source characters with
        incorrect reconstructed positions, producing a positive (non-None)
        but wrong mapping rather than failing gracefully.  All major code
        tokenizers preserve character identity, so this is acceptable.
        """
        source_to_recon: List[Optional[int]] = [None] * len(source_code)
        recon_idx = 0
        for src_idx in range(len(source_code)):
            if recon_idx >= len(recon_text):
                break
            if source_code[src_idx] == recon_text[recon_idx]:
                source_to_recon[src_idx] = recon_idx
                recon_idx += 1
            elif source_code[src_idx].lower() == recon_text[recon_idx].lower():
                source_to_recon[src_idx] = recon_idx
                recon_idx += 1
        return source_to_recon

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

        Uses the existing ``source_to_recon`` + ``char_to_token`` coordinate
        chain.  Returns ``None`` if the span cannot be mapped.
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
        """
        return self._process_token(raw_token, preserve_space=True)

    # ------------------------------------------------------------------
    # Source char → token map (whitespace-inclusive)
    # ------------------------------------------------------------------

    def _build_source_char_to_token_map(
        self, source_code: str, token_strings: List[str],
    ) -> List[Optional[int]]:
        """Map each source character (including whitespace) to a token index.

        This exists alongside the inherited :meth:`_build_char_to_token_map`
        because the base-class method strips whitespace (via ``_clean_token``)
        — suitable for identifier/keyword/operator alignment — whereas
        indentation measurement needs whitespace characters mapped to their
        producing tokens.

        Walks through *token_strings*, decodes each via
        :meth:`_decode_raw_token`, and greedily aligns decoded characters
        against *source_code*.

        Returns a list of length ``len(source_code)`` where entry *i* is the
        token index covering source char *i*, or ``None``.
        """
        result: List[Optional[int]] = [None] * len(source_code)
        src_idx = 0

        for tok_idx, raw_token in enumerate(token_strings):
            decoded = self._decode_raw_token(raw_token)
            if decoded is None:
                continue
            for ch in decoded:
                if src_idx >= len(source_code):
                    break
                if source_code[src_idx] == ch:
                    result[src_idx] = tok_idx
                    src_idx += 1
                elif source_code[src_idx].lower() == ch.lower():
                    result[src_idx] = tok_idx
                    src_idx += 1
                # else: skip character in decoded token (mismatch)

        return result

    # ------------------------------------------------------------------
    # Line indentation extraction
    # ------------------------------------------------------------------

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

        * *char_start* — inclusive start in source-code character coordinates
        * *char_end* — exclusive end in source-code character coordinates

        Positions are translated to the reconstructed-text coordinate space
        via *source_to_recon* before checking against *char_to_token*.

        Returns ``None`` if the span cannot be mapped.
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
    # Main compute
    # ------------------------------------------------------------------

    def compute(
        self, tokenized_data=None
    ) -> Dict[str, Any]:
        """Compute AST boundary alignment metrics.

        .. note::

           *tokenized_data* is **not used** — the metric loads its own code
           snippets and encodes them with each tokenizer.
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

        # ident_acc: tok -> lang -> [{text, num_tokens, fragmented}]
        ident_acc: Dict[str, Dict[str, List[Dict]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # indent_acc: tok -> lang -> ws_string -> [pattern_tuples]
        indent_acc: Dict[str, Dict[str, Dict[str, List[Tuple]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for tok_name in self.tokenizer_names:
            tokenizer = self.input_provider.get_tokenizer(tok_name)

            if not tokenizer.can_encode():
                logger.info("Tokenizer %s cannot encode text; skipping AST metrics", tok_name)
                continue

            for code_lang in self.code_loader.get_languages():
                parser = self._get_parser(code_lang)
                if parser is None:
                    continue

                snippets = self.code_loader.get_code_snippets(code_lang)

                for snippet in snippets[: self.max_snippets_per_lang]:
                    source_bytes = snippet.encode("utf-8")
                    tree = parser.parse(source_bytes)

                    categorized_spans = self._extract_leaf_spans(tree)

                    byte_to_char = self._byte_to_char_offsets(source_bytes)

                    # Tokenize with the current tokenizer
                    try:
                        token_ids = tokenizer.encode(snippet)
                    except Exception as e:
                        logger.debug("Encoding failed for %s on %s snippet: %s", tok_name, code_lang, e)
                        continue

                    if not token_ids:
                        continue

                    token_strings = self._convert_ids_to_tokens(tokenizer, token_ids)
                    recon_text, char_to_token = self._build_char_to_token_map(token_strings)

                    if not char_to_token:
                        continue

                    source_to_recon = self._build_source_to_recon_map(snippet, recon_text)

                    for category, spans in categorized_spans.items():
                        for byte_start, byte_end in spans:
                            # byte_end is exclusive (tree-sitter convention).
                            # byte_to_char has a sentinel at len(source_bytes) so
                            # byte_to_char[byte_end] yields an exclusive char offset.
                            if byte_start >= len(byte_to_char) or byte_end >= len(byte_to_char):
                                continue

                            c_start = byte_to_char[byte_start]   # inclusive
                            c_end = byte_to_char[byte_end]       # exclusive

                            alignment = self._check_boundary_alignment(
                                c_start, c_end, source_to_recon, char_to_token
                            )
                            if alignment is not None:
                                acc[tok_name][code_lang][category].append(alignment)

                            # Identifier fragmentation tracking
                            if category == "identifier":
                                num_tokens = self._count_identifier_tokens(
                                    c_start, c_end, source_to_recon, char_to_token
                                )
                                if num_tokens is not None:
                                    ident_acc[tok_name][code_lang].append({
                                        "text": snippet[c_start:c_end],
                                        "num_tokens": num_tokens,
                                        "fragmented": num_tokens > 1,
                                    })

                    # Indentation consistency (whitespace-significant languages)
                    if code_lang in _WHITESPACE_SIGNIFICANT_LANGS:
                        source_char_to_token = self._build_source_char_to_token_map(
                            snippet, token_strings
                        )
                        for ws_string, line_start, ws_end in self._extract_line_indentation(snippet):
                            if not ws_string:
                                continue
                            # Collect ordered unique token indices in [line_start, ws_end)
                            token_indices: List[int] = []
                            for pos in range(line_start, ws_end):
                                if pos < len(source_char_to_token):
                                    tidx = source_char_to_token[pos]
                                    if tidx is not None and (
                                        not token_indices or token_indices[-1] != tidx
                                    ):
                                        token_indices.append(tidx)
                            pattern = tuple(token_strings[ti] for ti in token_indices)
                            indent_acc[tok_name][code_lang][ws_string].append(pattern)

        return {
            "ast_boundary_alignment": self._build_results(acc),
            "identifier_fragmentation": self._build_identifier_fragmentation_results(ident_acc),
            "indentation_consistency": self._build_indentation_consistency_results(indent_acc),
        }

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_results(
        self, acc: Dict[str, Dict[str, Dict[str, List[Dict]]]]
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {"per_tokenizer": {}, "summary": {}}

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {
                "by_category": {},
                "by_language": {},
                "overall": {},
            }

            all_full: List[float] = []
            all_start: List[float] = []
            all_end: List[float] = []
            all_cross: List[float] = []
            total_count = 0
            languages_seen: set = set()

            for code_lang in sorted(acc.get(tok_name, {})):
                lang_full: List[float] = []
                lang_start: List[float] = []
                lang_end: List[float] = []
                lang_cross: List[float] = []

                for category in sorted(acc[tok_name][code_lang]):
                    items = acc[tok_name][code_lang][category]
                    if not items:
                        continue

                    s_rates = [1.0 if it["start_aligned"] else 0.0 for it in items]
                    e_rates = [1.0 if it["end_aligned"] else 0.0 for it in items]
                    f_rates = [1.0 if it["fully_aligned"] else 0.0 for it in items]
                    c_rates = [1.0 if it["cross_boundary"] else 0.0 for it in items]

                    if category not in tok_data["by_category"]:
                        tok_data["by_category"][category] = {}

                    tok_data["by_category"][category][code_lang] = {
                        "start_alignment_rate": float(np.mean(s_rates)),
                        "end_alignment_rate": float(np.mean(e_rates)),
                        "full_alignment_rate": float(np.mean(f_rates)),
                        "cross_boundary_rate": float(np.mean(c_rates)),
                        "count": len(items),
                    }

                    lang_full.extend(f_rates)
                    lang_start.extend(s_rates)
                    lang_end.extend(e_rates)
                    lang_cross.extend(c_rates)

                if lang_full:
                    tok_data["by_language"][code_lang] = {
                        "overall_full_alignment_rate": float(np.mean(lang_full)),
                        "overall_start_alignment_rate": float(np.mean(lang_start)),
                        "overall_end_alignment_rate": float(np.mean(lang_end)),
                        "overall_cross_boundary_rate": float(np.mean(lang_cross)),
                        "count": len(lang_full),
                    }
                    all_full.extend(lang_full)
                    all_start.extend(lang_start)
                    all_end.extend(lang_end)
                    all_cross.extend(lang_cross)
                    total_count += len(lang_full)
                    languages_seen.add(code_lang)

            if all_full:
                tok_data["overall"] = {
                    "full_alignment_rate": float(np.mean(all_full)),
                    "start_alignment_rate": float(np.mean(all_start)),
                    "end_alignment_rate": float(np.mean(all_end)),
                    "cross_boundary_rate": float(np.mean(all_cross)),
                    "count": total_count,
                }

            results["per_tokenizer"][tok_name] = tok_data

            if all_full:
                results["summary"][tok_name] = {
                    "avg_full_alignment_rate": float(np.mean(all_full)),
                    "avg_start_alignment_rate": float(np.mean(all_start)),
                    "avg_end_alignment_rate": float(np.mean(all_end)),
                    "avg_cross_boundary_rate": float(np.mean(all_cross)),
                    "total_nodes_analyzed": total_count,
                    "languages_analyzed": len(languages_seen),
                }

        return results

    def _build_identifier_fragmentation_results(
        self, ident_acc: Dict[str, Dict[str, List[Dict]]]
    ) -> Dict[str, Any]:
        """Build identifier fragmentation results from accumulated data."""
        results: Dict[str, Any] = {"per_tokenizer": {}, "summary": {}}

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {"by_language": {}, "overall": {}}
            all_items: List[Dict] = []
            languages_seen: set = set()

            for code_lang in sorted(ident_acc.get(tok_name, {})):
                items = ident_acc[tok_name][code_lang]
                if not items:
                    continue

                frag_rate = sum(1 for it in items if it["fragmented"]) / len(items)
                avg_tokens = sum(it["num_tokens"] for it in items) / len(items)

                tok_data["by_language"][code_lang] = {
                    "fragmentation_rate": float(frag_rate),
                    "avg_tokens_per_identifier": float(avg_tokens),
                    "count": len(items),
                }
                all_items.extend(items)
                languages_seen.add(code_lang)

            if all_items:
                overall_frag = sum(1 for it in all_items if it["fragmented"]) / len(all_items)
                overall_avg = sum(it["num_tokens"] for it in all_items) / len(all_items)
                tok_data["overall"] = {
                    "fragmentation_rate": float(overall_frag),
                    "avg_tokens_per_identifier": float(overall_avg),
                    "count": len(all_items),
                }

            results["per_tokenizer"][tok_name] = tok_data

            if all_items:
                results["summary"][tok_name] = {
                    "fragmentation_rate": float(overall_frag),
                    "avg_tokens_per_identifier": float(overall_avg),
                    "identifiers_analyzed": len(all_items),
                    "languages_analyzed": len(languages_seen),
                }

        return results

    def _build_indentation_consistency_results(
        self, indent_acc: Dict[str, Dict[str, Dict[str, List[Tuple]]]]
    ) -> Dict[str, Any]:
        """Build indentation consistency results from accumulated data."""
        results: Dict[str, Any] = {"per_tokenizer": {}, "summary": {}}

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {"by_language": {}}
            lang_consistency_rates: List[float] = []
            lang_weighted_rates: List[float] = []
            languages_seen: set = set()

            for code_lang in sorted(indent_acc.get(tok_name, {})):
                ws_groups = indent_acc[tok_name][code_lang]
                if not ws_groups:
                    continue

                # Filter to non-empty ws_strings (actual indentation)
                indent_levels = {
                    ws: patterns
                    for ws, patterns in ws_groups.items()
                    if ws  # skip empty-string (no indentation)
                }

                if not indent_levels:
                    continue

                consistent_levels = 0
                total_lines = 0
                dominant_matches = 0

                for ws_string, patterns in indent_levels.items():
                    if not patterns:
                        continue
                    counter = Counter(patterns)
                    most_common_count = counter.most_common(1)[0][1]
                    total_lines += len(patterns)
                    dominant_matches += most_common_count

                    # A level is consistent if all patterns are the same
                    if len(counter) == 1:
                        consistent_levels += 1

                num_levels = len(indent_levels)
                consistency_rate = consistent_levels / num_levels if num_levels else 0.0
                weighted_consistency = dominant_matches / total_lines if total_lines else 0.0

                tok_data["by_language"][code_lang] = {
                    "consistency_rate": float(consistency_rate),
                    "weighted_consistency": float(weighted_consistency),
                    "num_indent_levels": num_levels,
                    "total_lines": total_lines,
                }
                lang_consistency_rates.append(consistency_rate)
                lang_weighted_rates.append(weighted_consistency)
                languages_seen.add(code_lang)

            results["per_tokenizer"][tok_name] = tok_data

            if lang_consistency_rates:
                results["summary"][tok_name] = {
                    "avg_consistency_rate": float(np.mean(lang_consistency_rates)),
                    "avg_weighted_consistency": float(np.mean(lang_weighted_rates)),
                    "languages_analyzed": len(languages_seen),
                }

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
                    total_items = sum(d["count"] for d in lang_data.values())
                    if total_items == 0:
                        continue
                    weighted_full = sum(
                        d["full_alignment_rate"] * d["count"]
                        for d in lang_data.values()
                    ) / total_items
                    print(
                        f"  {category:15}: "
                        f"full_align={weighted_full:.3f}  "
                        f"n={total_items}"
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
                    print(
                        f"  {lang:15}: "
                        f"full_align={d['overall_full_alignment_rate']:.3f}  "
                        f"start={d['overall_start_alignment_rate']:.3f}  "
                        f"end={d['overall_end_alignment_rate']:.3f}  "
                        f"n={d['count']}"
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
                    print(f"  {'Avg Consistency Rate':25}: {s['avg_consistency_rate']:.3f}")
                    print(f"  {'Avg Weighted Consistency':25}: {s['avg_weighted_consistency']:.3f}")
                    print(f"  {'Languages':25}: {s['languages_analyzed']}")

                    tok_detail = indent.get("per_tokenizer", {}).get(tok_name, {})
                    by_lang = tok_detail.get("by_language", {})
                    if by_lang:
                        for lang in sorted(by_lang):
                            d = by_lang[lang]
                            print(
                                f"    {lang:13}: "
                                f"consistency={d['consistency_rate']:.3f}  "
                                f"weighted={d['weighted_consistency']:.3f}  "
                                f"levels={d['num_indent_levels']}  "
                                f"lines={d['total_lines']}"
                            )

        print("\n" + "=" * 60)
