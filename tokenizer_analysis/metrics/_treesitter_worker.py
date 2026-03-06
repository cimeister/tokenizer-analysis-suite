"""
Lightweight tree-sitter parsing worker.

This module is designed to be run as a subprocess to isolate tree-sitter's
C heap from the tokenizer's Rust/C backend.  It intentionally avoids
importing anything from the tokenizer_analysis package (no BaseMetrics,
InputProvider, numpy, etc.) so that the subprocess starts quickly and
operates on a clean heap.

The subprocess returns ONLY what tree-sitter produces: categorized
byte-offset spans.  Everything else (byte-to-char mapping, indentation
extraction, snippet text) stays in the main process.
"""

import pickle
import sys
from typing import Dict, List, Optional, Set, Tuple

# ======================================================================
# AST node category constants
# ======================================================================

CATEGORIES = ("identifier", "keyword", "operator", "literal", "delimiter")

IDENTIFIER_TYPES: Set[str] = {
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

LITERAL_TYPES: Set[str] = {
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

DELIMITER_CHARS: Set[str] = set("(){}[];,")

# Punctuation that is syntactically not an operator in most languages.
NON_OPERATOR_PUNCTUATION: Set[str] = {
    ".", ":", "::", "@", "#", "...", "\\",
}

# Keywords that may appear as named nodes in some grammars
KNOWN_KEYWORDS: Set[str] = {
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


# ======================================================================
# Parsing helpers
# ======================================================================


def classify_node(node) -> Optional[str]:
    """Classify a tree-sitter leaf node into one of the five categories.

    Returns ``'identifier'``, ``'keyword'``, ``'operator'``,
    ``'literal'``, ``'delimiter'``, or ``None``.
    """
    node_type = node.type
    text = node.text.decode("utf-8") if isinstance(node.text, bytes) else node.text

    if not text:
        return None

    # Delimiters
    if text in DELIMITER_CHARS and len(text) == 1:
        return "delimiter"

    # Identifiers — checked before keywords intentionally: if the grammar
    # assigns an identifier type, we trust that classification even when
    # the text appears in KNOWN_KEYWORDS (e.g. ``self`` in Python is
    # a conventional identifier, not a reserved keyword).
    if node_type in IDENTIFIER_TYPES:
        return "identifier"

    # Literals
    if node_type in LITERAL_TYPES:
        return "literal"

    # Keywords: anonymous nodes whose type equals their text and is
    # alphabetic (tree-sitter represents keyword tokens this way).
    if not node.is_named and node_type == text and text.isalpha():
        return "keyword"

    # Some grammars expose keywords as named nodes
    if node.is_named and text in KNOWN_KEYWORDS and text.isalpha():
        return "keyword"

    # Operators: anonymous non-alphanumeric leaf nodes that aren't
    # delimiters, whitespace, or structural punctuation
    if not node.is_named and not text.isalnum() and text not in DELIMITER_CHARS:
        stripped = text.strip()
        if stripped and not stripped.isspace() and stripped not in NON_OPERATOR_PUNCTUATION:
            return "operator"

    return None


def extract_leaf_spans(tree) -> Dict[str, List[Tuple[int, int]]]:
    """Extract categorised leaf-node spans from a parse tree.

    Returns a dict mapping category names to lists of
    ``(start_byte, end_byte)`` spans.
    """
    categorized: Dict[str, List[Tuple[int, int]]] = {
        cat: [] for cat in CATEGORIES
    }

    def _walk(node):
        if node.type == "ERROR":
            return
        if node.child_count == 0:
            cat = classify_node(node)
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


def _parse_one_snippet(parser, snippet, timeout):
    """Parse a single snippet with a thread-based timeout.

    tree-sitter's C parser may hold the GIL, making ``signal.alarm``
    ineffective.  Instead we run the parse in a daemon thread and join
    with a timeout.  If the thread is still alive after *timeout*
    seconds the snippet is considered stuck and we return ``None``.
    The daemon thread will be cleaned up when the (subprocess) process
    exits.
    """
    import threading

    result = [None]
    error = [None]

    def _do_parse():
        try:
            source_bytes = snippet.encode("utf-8")
            tree = parser.parse(source_bytes)
            result[0] = extract_leaf_spans(tree)
        except Exception as exc:
            error[0] = exc

    t = threading.Thread(target=_do_parse, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        return None  # timed out — thread will die with the process
    if error[0] is not None:
        return None
    return result[0]


def parse_snippets(
    code_snippets: Dict[str, List[str]],
    lang_to_treesitter: Dict[str, str],
    per_snippet_timeout: float = 10.0,
) -> Dict[str, List[Dict[str, List[Tuple[int, int]]]]]:
    """Parse all code snippets with tree-sitter.

    This is the main entry point for both in-process and subprocess usage.
    All inputs and outputs are pure-Python primitives (picklable).

    Parameters
    ----------
    per_snippet_timeout:
        Maximum wall-clock seconds to spend parsing a single snippet.
        Snippets exceeding this limit are skipped (an empty spans dict
        is emitted so indexing stays aligned with the input list).

    Returns ``parsed[lang] = [categorized_spans_dict, ...]`` where each
    ``categorized_spans_dict`` maps category names to lists of
    ``(start_byte, end_byte)`` spans.
    """
    import tree_sitter_language_pack as ts_pack

    parsed: Dict[str, List[Dict[str, List[Tuple[int, int]]]]] = {}
    skipped = 0
    empty_spans = {cat: [] for cat in CATEGORIES}

    for lang, snippets in code_snippets.items():
        ts_name = lang_to_treesitter.get(lang)
        if ts_name is None:
            continue

        try:
            parser = ts_pack.get_parser(ts_name)
        except Exception:
            continue

        lang_parsed = []
        for snippet in snippets:
            result = _parse_one_snippet(parser, snippet, per_snippet_timeout)
            if result is not None:
                lang_parsed.append(result)
            else:
                skipped += 1
                lang_parsed.append({cat: list(v) for cat, v in empty_spans.items()})

        parsed[lang] = lang_parsed

    if skipped:
        # Print to stderr so the main process can see it (stdout is for data)
        print(
            f"tree-sitter worker: skipped {skipped} snippet(s) due to "
            f"per-snippet timeout ({per_snippet_timeout}s)",
            file=sys.stderr,
        )

    return parsed


# ======================================================================
# Subprocess entry point
# ======================================================================

if __name__ == "__main__":
    # When run as a script, Python prepends the script's directory to
    # sys.path.  That directory (tokenizer_analysis/metrics/) contains
    # math.py which shadows the stdlib ``math`` module and causes a
    # circular-import crash in tree-sitter-language-pack → tempfile →
    # random → math.  Strip it so only the stdlib math is found.
    #
    # Use both abspath and realpath to handle symlinks / bind-mounts.
    import os as _os

    _script_dir_abs = _os.path.abspath(_os.path.dirname(__file__))
    _script_dir_real = _os.path.realpath(_os.path.dirname(__file__))
    _blocked = {_script_dir_abs, _script_dir_real}
    _original_path = list(sys.path)
    sys.path = [
        p for p in sys.path
        if _os.path.abspath(p) not in _blocked
        and _os.path.realpath(p) not in _blocked
    ]

    _removed = set(_original_path) - set(sys.path)
    if _removed:
        print(
            f"tree-sitter worker: stripped sys.path entries to avoid "
            f"math.py shadow: {_removed}",
            file=sys.stderr,
        )

    # Protocol: two CLI arguments — input pickle path, output pickle path.
    # The input file contains (code_snippets, lang_to_treesitter).
    # The output file receives the parsed result dict.
    if len(sys.argv) != 3:
        print("Usage: _treesitter_worker.py <input_pickle> <output_pickle>",
              file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        with open(input_path, "rb") as f:
            code_snippets, lang_to_treesitter = pickle.load(f)
    except Exception as exc:
        print(f"tree-sitter worker: failed to load input pickle: {exc}",
              file=sys.stderr)
        sys.exit(2)

    try:
        result = parse_snippets(code_snippets, lang_to_treesitter)
    except Exception as exc:
        print(f"tree-sitter worker: parse_snippets failed: {exc}",
              file=sys.stderr)
        sys.exit(3)

    try:
        with open(output_path, "wb") as f:
            pickle.dump(result, f)
    except Exception as exc:
        print(f"tree-sitter worker: failed to write output pickle: {exc}",
              file=sys.stderr)
        sys.exit(4)
