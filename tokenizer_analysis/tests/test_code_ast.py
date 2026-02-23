"""Tests for tokenizer_analysis.metrics.code_ast (ASTBoundaryMetrics)."""

import pytest

from tokenizer_analysis.metrics.code_ast import (
    ASTBoundaryMetrics,
    _IDENTIFIER_TYPES,
    _LITERAL_TYPES,
    _DELIMITER_CHARS,
    _KNOWN_KEYWORDS,
    _NON_OPERATOR_PUNCTUATION,
    _WHITESPACE_SIGNIFICANT_LANGS,
)
from tokenizer_analysis.loaders.code_data import CodeDataLoader
from tokenizer_analysis.core.input_types import TokenizedData


# ======================================================================
# Helpers
# ======================================================================

_EPS = 1e-9


def _make_instance():
    """Return a bare ASTBoundaryMetrics without a live InputProvider.

    Only usable for calling static / class methods and helpers that
    don't touch ``self.input_provider``.
    """
    inst = object.__new__(ASTBoundaryMetrics)
    inst._tokenizer_vocab_cache = {}
    inst._warned_tokenizers = set()
    return inst


# ======================================================================
# _build_source_to_recon_map
# ======================================================================

class TestSourceToReconMap:
    """Verify source-code to reconstructed-text coordinate mapping."""

    def test_no_whitespace(self):
        source = "abc"
        recon = "abc"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        assert result == [0, 1, 2]

    def test_leading_whitespace_dropped(self):
        source = "  abc"
        recon = "abc"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        # first two chars are spaces -> None; then abc maps to 0,1,2
        assert result == [None, None, 0, 1, 2]

    def test_internal_whitespace_dropped(self):
        source = "a b"
        recon = "ab"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        assert result == [0, None, 1]

    def test_newlines_dropped(self):
        source = "a\nb"
        recon = "ab"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        assert result == [0, None, 1]

    def test_tab_dropped(self):
        source = "a\tb"
        recon = "ab"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        assert result == [0, None, 1]

    def test_case_insensitive_matching(self):
        source = "ABC"
        recon = "abc"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        assert result == [0, 1, 2]

    def test_identical_source_and_recon(self):
        source = "def fibonacci(n):"
        recon = "deffibonacci(n):"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        # 'd','e','f' match; space->None; then rest matches
        assert result[0] == 0  # 'd'
        assert result[1] == 1  # 'e'
        assert result[2] == 2  # 'f'
        assert result[3] is None  # ' '
        assert result[4] == 3  # 'f' (of fibonacci)

    def test_case_sensitive_preferred(self):
        """Exact match should bind before case-insensitive fallback."""
        source = "aA"
        recon = "aA"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        # 'a' matches recon[0] exactly, 'A' matches recon[1] exactly
        assert result == [0, 1]

    def test_dropped_character(self):
        """A character dropped in recon leaves None; subsequent chars align."""
        source = "a b"
        recon = "ab"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        assert result == [0, None, 1]


# ======================================================================
# _byte_to_char_offsets
# ======================================================================

class TestByteToCharOffsets:

    def test_ascii(self):
        source = "abc".encode("utf-8")
        offsets = ASTBoundaryMetrics._byte_to_char_offsets(source)
        # 3 bytes + sentinel = 4 entries
        assert offsets == [0, 1, 2, 3]

    def test_multibyte_utf8(self):
        source = "aé".encode("utf-8")  # 'a' = 1 byte, 'é' = 2 bytes
        offsets = ASTBoundaryMetrics._byte_to_char_offsets(source)
        # byte 0 -> char 0 ('a')
        # bytes 1,2 -> char 1 ('é')
        # sentinel -> char 2
        assert offsets == [0, 1, 1, 2]

    def test_empty(self):
        offsets = ASTBoundaryMetrics._byte_to_char_offsets(b"")
        assert offsets == [0]  # just sentinel

    def test_cjk_character(self):
        # Chinese character: 3 bytes in UTF-8
        source = "x中".encode("utf-8")
        offsets = ASTBoundaryMetrics._byte_to_char_offsets(source)
        # x: 1 byte -> char 0
        # 中: 3 bytes -> char 1,1,1
        # sentinel -> char 2
        assert offsets == [0, 1, 1, 1, 2]


# ======================================================================
# _check_boundary_alignment
# ======================================================================

class TestBoundaryAlignment:
    """Test boundary alignment check logic."""

    def test_perfect_alignment_at_start(self):
        # Tokens: [0,0,0, 1,1,1] representing "abcdef"
        # Node spans chars 0..3 (abc) -> all token 0
        source_to_recon = [0, 1, 2, 3, 4, 5]
        char_to_token = [0, 0, 0, 1, 1, 1]
        result = ASTBoundaryMetrics._check_boundary_alignment(
            0, 3, source_to_recon, char_to_token
        )
        assert result is not None
        assert result["start_aligned"] is True
        assert result["end_aligned"] is True
        assert result["fully_aligned"] is True
        assert result["cross_boundary"] is False

    def test_perfect_alignment_in_middle(self):
        # Tokens: [0,0, 1,1, 2,2]
        # Node spans chars 2..4 (token 1) -> start changes from 0 to 1, end changes from 1 to 2
        source_to_recon = [0, 1, 2, 3, 4, 5]
        char_to_token = [0, 0, 1, 1, 2, 2]
        result = ASTBoundaryMetrics._check_boundary_alignment(
            2, 4, source_to_recon, char_to_token
        )
        assert result is not None
        assert result["fully_aligned"] is True

    def test_cross_boundary_start(self):
        # Tokens: [0,0,0, 1,1,1]
        # Node spans chars 1..4 -> starts mid-token-0, ends mid-token-1
        source_to_recon = [0, 1, 2, 3, 4, 5]
        char_to_token = [0, 0, 0, 1, 1, 1]
        result = ASTBoundaryMetrics._check_boundary_alignment(
            1, 4, source_to_recon, char_to_token
        )
        assert result is not None
        assert result["start_aligned"] is False
        assert result["fully_aligned"] is False
        assert result["cross_boundary"] is True

    def test_unmappable_span_returns_none(self):
        # All whitespace in source -> all None in source_to_recon
        source_to_recon = [None, None, None]
        char_to_token = [0, 0, 0]
        result = ASTBoundaryMetrics._check_boundary_alignment(
            0, 3, source_to_recon, char_to_token
        )
        assert result is None

    def test_node_at_end_of_text(self):
        # Node spans to end of text -> end_aligned should be True
        source_to_recon = [0, 1, 2]
        char_to_token = [0, 0, 1]
        result = ASTBoundaryMetrics._check_boundary_alignment(
            2, 3, source_to_recon, char_to_token
        )
        assert result is not None
        assert result["end_aligned"] is True

    def test_with_whitespace_in_source(self):
        # Source: "a b" -> recon: "ab"
        # source_to_recon: [0, None, 1]
        # char_to_token: [0, 1]  (tokens for "ab")
        # Node spans source chars 2..3 ('b') -> recon pos 1..2
        source_to_recon = [0, None, 1]
        char_to_token = [0, 1]
        result = ASTBoundaryMetrics._check_boundary_alignment(
            2, 3, source_to_recon, char_to_token
        )
        assert result is not None
        # Token at recon pos 1 differs from token at recon pos 0 -> start aligned
        assert result["start_aligned"] is True
        # recon_end=2 >= len(char_to_token) -> end aligned
        assert result["end_aligned"] is True
        assert result["fully_aligned"] is True


# ======================================================================
# _classify_node (requires tree-sitter)
# ======================================================================

class _MockNode:
    """Minimal tree-sitter node substitute for classification tests."""

    def __init__(self, node_type, text, is_named=True, child_count=0):
        self.type = node_type
        self.text = text.encode("utf-8") if isinstance(text, str) else text
        self.is_named = is_named
        self.child_count = child_count


class TestClassifyNode:
    """Test AST node classification without tree-sitter."""

    def test_delimiter_paren(self):
        node = _MockNode("(", "(", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "delimiter"

    def test_delimiter_brace(self):
        node = _MockNode("{", "{", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "delimiter"

    def test_delimiter_semicolon(self):
        node = _MockNode(";", ";", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "delimiter"

    def test_identifier(self):
        node = _MockNode("identifier", "fibonacci", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "identifier"

    def test_type_identifier(self):
        node = _MockNode("type_identifier", "Int", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "identifier"

    def test_field_identifier(self):
        node = _MockNode("field_identifier", "name", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "identifier"

    def test_string_literal(self):
        node = _MockNode("string_literal", '"hello"', is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "literal"

    def test_integer(self):
        node = _MockNode("integer", "42", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "literal"

    def test_float_literal(self):
        node = _MockNode("float_literal", "3.14", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "literal"

    def test_true_literal(self):
        node = _MockNode("true", "true", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "literal"

    def test_false_literal(self):
        node = _MockNode("false", "false", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "literal"

    def test_null_literal(self):
        node = _MockNode("null", "null", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "literal"

    def test_keyword_anonymous_if(self):
        # Tree-sitter represents keywords as anonymous nodes whose type == text
        node = _MockNode("if", "if", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "keyword"

    def test_keyword_anonymous_return(self):
        node = _MockNode("return", "return", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "keyword"

    def test_keyword_anonymous_class(self):
        node = _MockNode("class", "class", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "keyword"

    def test_keyword_def_in_known_set(self):
        # Named node with text in _KNOWN_KEYWORDS
        node = _MockNode("keyword", "def", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "keyword"

    def test_operator_plus(self):
        node = _MockNode("+", "+", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "operator"

    def test_operator_equals_equals(self):
        node = _MockNode("==", "==", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "operator"

    def test_operator_ampersand_ampersand(self):
        node = _MockNode("&&", "&&", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "operator"

    def test_operator_arrow(self):
        node = _MockNode("->", "->", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "operator"

    def test_empty_text_returns_none(self):
        node = _MockNode("", "", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_non_leaf_generic_node_returns_none(self):
        # A named node with non-matching type
        node = _MockNode("expression_statement", "x + 1", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) is None

    # -- Punctuation exclusion from operator category --

    def test_colon_not_operator(self):
        node = _MockNode(":", ":", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_dot_not_operator(self):
        node = _MockNode(".", ".", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_double_colon_not_operator(self):
        node = _MockNode("::", "::", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_at_sign_not_operator(self):
        node = _MockNode("@", "@", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_hash_not_operator(self):
        node = _MockNode("#", "#", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_spread_not_operator(self):
        node = _MockNode("...", "...", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_arrow_still_operator(self):
        node = _MockNode("->", "->", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "operator"


# ======================================================================
# CodeDataLoader
# ======================================================================

class TestCodeDataLoader:
    """Tests for the code data loader."""

    def test_synthetic_samples_all_languages(self):
        samples = CodeDataLoader.generate_synthetic_samples()
        expected_langs = set(CodeDataLoader._LANG_TO_TREESITTER.keys())
        assert set(samples.keys()) == expected_langs

    def test_synthetic_samples_nonempty(self):
        samples = CodeDataLoader.generate_synthetic_samples()
        for lang, snippets in samples.items():
            assert len(snippets) > 0, f"No snippets for {lang}"
            for snippet in snippets:
                assert len(snippet.strip()) > 0, f"Empty snippet for {lang}"

    def test_get_languages_empty(self):
        loader = CodeDataLoader()
        assert loader.get_languages() == []

    def test_get_languages_after_synthetic(self):
        loader = CodeDataLoader()
        synthetic = CodeDataLoader.generate_synthetic_samples()
        for lang, snippets in synthetic.items():
            loader.code_snippets[lang] = snippets
        assert len(loader.get_languages()) == len(CodeDataLoader._LANG_TO_TREESITTER)

    def test_get_code_snippets_missing_lang(self):
        loader = CodeDataLoader()
        assert loader.get_code_snippets("nonexistent") == []

    def test_default_max_snippets_per_lang(self):
        loader = CodeDataLoader()
        assert loader.max_snippets_per_lang == CodeDataLoader.DEFAULT_MAX_SNIPPETS_PER_LANG

    def test_custom_max_snippets_per_lang(self):
        loader = CodeDataLoader(max_snippets_per_lang=10)
        assert loader.max_snippets_per_lang == 10

    def test_cap_zero_disables_limit(self):
        loader = CodeDataLoader(max_snippets_per_lang=0)
        assert loader.max_snippets_per_lang == 0

    def test_cap_limits_loaded_snippets(self, tmp_path):
        """Loading from a directory with many files respects the cap."""
        lang_dir = tmp_path / "python"
        lang_dir.mkdir()
        for i in range(10):
            (lang_dir / f"f{i}.py").write_text(f"x = {i}\n")

        loader = CodeDataLoader(
            {"python": str(lang_dir)}, max_snippets_per_lang=3
        )
        loader.load_all()
        assert len(loader.get_code_snippets("python")) == 3

    def test_cap_limits_parquet_snippets(self, tmp_path):
        """Parquet loading also respects the cap."""
        import pandas as pd
        df = pd.DataFrame({"content": [f"x = {i}" for i in range(20)]})
        path = tmp_path / "code.parquet"
        df.to_parquet(path)

        loader = CodeDataLoader(
            {"python": str(path)}, max_snippets_per_lang=5
        )
        loader.load_all()
        assert len(loader.get_code_snippets("python")) == 5

    def test_lang_to_treesitter_consistency(self):
        """All extension-mapped languages should have a tree-sitter grammar mapping."""
        for lang in CodeDataLoader._LANG_EXTENSIONS:
            assert lang in CodeDataLoader._LANG_TO_TREESITTER, (
                f"Language {lang} has extensions but no tree-sitter mapping"
            )


# ======================================================================
# StarCoder metadata stripping
# ======================================================================

class TestStarCoderMetadataStripping:
    """Verify that StarCoder metadata prefixes are stripped from content."""

    def test_all_three_tags(self):
        raw = "<reponame>user/repo<filename>main.py<gh_stars>1-10\nimport os\n"
        assert CodeDataLoader._strip_starcoder_metadata(raw) == "import os\n"

    def test_filename_only(self):
        raw = "<filename>setup.py\nimport setuptools\n"
        assert CodeDataLoader._strip_starcoder_metadata(raw) == "import setuptools\n"

    def test_gh_stars_only(self):
        raw = "<gh_stars>1-10\nfn main() {}\n"
        assert CodeDataLoader._strip_starcoder_metadata(raw) == "fn main() {}\n"

    def test_no_tags(self):
        raw = "import os\nprint('hello')\n"
        assert CodeDataLoader._strip_starcoder_metadata(raw) == raw

    def test_tags_not_at_start_preserved(self):
        raw = "# comment\n<filename>not_a_prefix.py\n"
        assert CodeDataLoader._strip_starcoder_metadata(raw) == raw


# ======================================================================
# Parquet loading
# ======================================================================

class TestParquetLoading:
    """Verify that CodeDataLoader can read parquet files."""

    @pytest.fixture()
    def parquet_file(self, tmp_path):
        """Create a small parquet file with a content column."""
        import pandas as pd
        df = pd.DataFrame({
            "content": [
                "<reponame>u/r<filename>a.py<gh_stars>0\ndef foo(): pass\n",
                "class Bar:\n    x = 1\n",
                "",       # empty — should be skipped
                "  \t  ", # whitespace-only — should be skipped
            ]
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path)
        return str(path)

    def test_read_parquet_strips_metadata(self, parquet_file):
        snippets = CodeDataLoader._read_parquet(parquet_file)
        assert len(snippets) == 2
        assert snippets[0] == "def foo(): pass"
        assert snippets[1] == "class Bar:\n    x = 1"

    def test_load_language_parquet(self, parquet_file):
        loader = CodeDataLoader({"python": parquet_file})
        loader.load_all()
        assert "python" in loader.get_languages()
        assert len(loader.get_code_snippets("python")) == 2

    def test_read_parquet_missing_column(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({"code": ["print(1)"]})
        path = tmp_path / "no_content.parquet"
        df.to_parquet(path)
        snippets = CodeDataLoader._read_parquet(str(path))
        assert snippets == []


# ======================================================================
# Synthetic samples parse without tree-sitter errors
# ======================================================================

class TestSyntheticSamplesParsing:
    """Verify that all synthetic code snippets parse without tree-sitter errors.

    This test is skipped if tree-sitter-language-pack is not installed.
    """

    @pytest.fixture(scope="class")
    def ts_pack(self):
        try:
            import tree_sitter_language_pack
            return tree_sitter_language_pack
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

    def test_all_snippets_parse(self, ts_pack):
        samples = CodeDataLoader.generate_synthetic_samples()
        for lang, snippets in samples.items():
            ts_name = CodeDataLoader._LANG_TO_TREESITTER.get(lang)
            if ts_name is None:
                continue
            try:
                parser = ts_pack.get_parser(ts_name)
            except Exception:
                pytest.skip(f"No tree-sitter grammar for {ts_name}")

            for i, snippet in enumerate(snippets):
                tree = parser.parse(snippet.encode("utf-8"))
                root = tree.root_node

                # Count ERROR nodes
                errors = []
                def _find_errors(node):
                    if node.type == "ERROR":
                        errors.append(
                            f"ERROR at {node.start_point}-{node.end_point}: "
                            f"'{snippet[node.start_byte:node.end_byte][:50]}'"
                        )
                    for child in node.children:
                        _find_errors(child)

                _find_errors(root)
                assert len(errors) == 0, (
                    f"Parse errors in {lang} snippet #{i}: {errors}"
                )

    def test_extract_leaf_spans_nonempty(self, ts_pack):
        """Verify that extracted spans cover all 5 categories for key languages."""
        inst = _make_instance()
        inst._ts_pack = ts_pack
        inst._treesitter_available = True

        # Test a subset of languages likely to cover all categories
        test_langs = ["python", "javascript", "java", "rust", "go"]
        samples = CodeDataLoader.generate_synthetic_samples()

        for lang in test_langs:
            ts_name = CodeDataLoader._LANG_TO_TREESITTER.get(lang)
            if ts_name is None:
                continue

            parser = ts_pack.get_parser(ts_name)
            snippet = samples[lang][0]
            tree = parser.parse(snippet.encode("utf-8"))
            spans = inst._extract_leaf_spans(tree)

            # Each language snippet should produce at least identifiers,
            # keywords, operators, and delimiters
            for cat in ("identifier", "keyword", "operator", "delimiter"):
                assert len(spans[cat]) > 0, (
                    f"No {cat} spans extracted from {lang} snippet"
                )


from .conftest import MockTokenizer, MockProvider as _MockProvider


# ======================================================================
# Mock infrastructure for end-to-end compute() tests
# ======================================================================

class _MockTokenizer(MockTokenizer):
    """Extends MockTokenizer with encode/can_encode for AST tests."""

    def can_encode(self):
        return True

    def encode(self, text):
        """Character-level encoding: one token per character."""
        return list(range(len(text)))


class _CharTokenizer:
    """Simple character-level tokenizer for testing."""

    def __init__(self):
        pass

    def convert_ids_to_tokens(self, ids):
        # The IDs are character ordinals
        return [chr(i) for i in ids]

    def can_encode(self):
        return True

    def encode(self, text):
        return [ord(c) for c in text]

    def get_vocab(self):
        return {}


class _PerfectTokenizer:
    """Tokenizer that returns pre-defined tokens for specific snippets."""

    def __init__(self, snippet_to_tokens):
        """
        Args:
            snippet_to_tokens: Dict mapping snippet text -> list of token strings
        """
        self._snippet_map = snippet_to_tokens
        self._build_vocab()

    def _build_vocab(self):
        self._token_to_id = {}
        self._id_to_token = {}
        next_id = 0
        for tokens in self._snippet_map.values():
            for t in tokens:
                if t not in self._token_to_id:
                    self._token_to_id[t] = next_id
                    self._id_to_token[next_id] = t
                    next_id += 1

    def can_encode(self):
        return True

    def encode(self, text):
        tokens = self._snippet_map.get(text, [])
        return [self._token_to_id[t] for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self._id_to_token[i] for i in ids]

    def get_vocab(self):
        return dict(self._token_to_id)


# ======================================================================
# Graceful degradation when tree-sitter is unavailable
# ======================================================================

class TestGracefulDegradation:
    """Verify compute() returns error dict when tree-sitter is missing."""

    def test_returns_error_when_unavailable(self):
        """When tree-sitter import fails, compute returns an error dict."""
        inst = _make_instance()
        inst._treesitter_available = False
        inst.input_provider = _MockProvider("test", _CharTokenizer())
        inst.tokenizer_names = ["test"]

        loader = CodeDataLoader()
        synthetic = CodeDataLoader.generate_synthetic_samples()
        for lang, snippets in synthetic.items():
            loader.code_snippets[lang] = snippets
        inst.code_loader = loader
        inst.max_snippets_per_lang = 1

        result = inst.compute()
        assert "ast_boundary_alignment" in result
        assert "error" in result["ast_boundary_alignment"]


# ======================================================================
# End-to-end compute() test with tree-sitter
# ======================================================================

class TestEndToEnd:
    """Full pipeline test with real tree-sitter parsing."""

    @pytest.fixture(scope="class")
    def ts_pack(self):
        try:
            import tree_sitter_language_pack
            return tree_sitter_language_pack
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

    def test_compute_with_char_tokenizer(self, ts_pack):
        """Character-level tokenizer should produce non-zero alignment rates.

        A character-level tokenizer splits every character into its own token,
        so every AST node boundary aligns perfectly with token boundaries.
        """
        char_tok = _CharTokenizer()
        provider = _MockProvider("char_tok", char_tok)

        # Build metrics with a single Python snippet
        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["char_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [
            'def add(a, b):\n'
            '    return a + b\n'
        ]}
        inst.code_loader = loader

        result = inst.compute()

        assert "ast_boundary_alignment" in result
        ast = result["ast_boundary_alignment"]
        assert "error" not in ast
        assert "per_tokenizer" in ast
        assert "char_tok" in ast["per_tokenizer"]

        tok_data = ast["per_tokenizer"]["char_tok"]
        assert "overall" in tok_data
        # Character-level tokenizer: every boundary aligns
        assert tok_data["overall"]["full_alignment_rate"] == pytest.approx(1.0)
        assert tok_data["overall"]["count"] > 0

    def test_compute_summary_structure(self, ts_pack):
        """Verify the summary structure is populated."""
        char_tok = _CharTokenizer()
        provider = _MockProvider("test_tok", char_tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["test_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [
            'x = 1 + 2\n'
        ]}
        inst.code_loader = loader

        result = inst.compute()
        ast = result["ast_boundary_alignment"]

        # Check summary
        assert "summary" in ast
        assert "test_tok" in ast["summary"]
        s = ast["summary"]["test_tok"]
        assert "avg_full_alignment_rate" in s
        assert "total_nodes_analyzed" in s
        assert s["total_nodes_analyzed"] > 0
        assert "languages_analyzed" in s
        assert s["languages_analyzed"] == 1

    def test_compute_by_category(self, ts_pack):
        """Verify by_category results contain known categories."""
        char_tok = _CharTokenizer()
        provider = _MockProvider("test_tok", char_tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["test_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [CodeDataLoader.generate_synthetic_samples()["python"][0]]}
        inst.code_loader = loader

        result = inst.compute()
        ast = result["ast_boundary_alignment"]
        by_cat = ast["per_tokenizer"]["test_tok"]["by_category"]

        # At least identifiers, keywords, operators, delimiters should be present
        for cat in ("identifier", "keyword", "operator", "delimiter"):
            assert cat in by_cat, f"Missing category {cat}"
            assert "python" in by_cat[cat], f"Missing python in {cat}"
            assert by_cat[cat]["python"]["count"] > 0

    def test_compute_by_language(self, ts_pack):
        """Verify by_language results appear for each loaded language."""
        char_tok = _CharTokenizer()
        provider = _MockProvider("test_tok", char_tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["test_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        samples = CodeDataLoader.generate_synthetic_samples()
        # Test with 3 languages
        for lang in ["python", "javascript", "go"]:
            loader.code_snippets[lang] = samples[lang]
        inst.code_loader = loader

        result = inst.compute()
        ast = result["ast_boundary_alignment"]
        by_lang = ast["per_tokenizer"]["test_tok"]["by_language"]

        for lang in ["python", "javascript", "go"]:
            assert lang in by_lang, f"Missing language {lang}"
            assert by_lang[lang]["count"] > 0

    def test_perfect_tokenizer_high_alignment(self, ts_pack):
        """A tokenizer that preserves AST boundaries should score well.

        We use a manually crafted tokenizer that splits a simple Python
        snippet exactly at keyword/identifier/operator/delimiter boundaries.
        """
        snippet = 'x = 1 + 2'
        # Tokens: "x" " " "=" " " "1" " " "+" " " "2"
        # This tokenizer keeps each meaningful AST node as its own token
        tokens = ["x", "Ġ=", "Ġ1", "Ġ+", "Ġ2"]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("perfect", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["perfect"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        ast = result["ast_boundary_alignment"]
        overall = ast["per_tokenizer"]["perfect"]["overall"]

        # Should have very high alignment since tokens match AST boundaries
        assert overall["full_alignment_rate"] > 0.5
        assert overall["count"] > 0


# ======================================================================
# print_results smoke test
# ======================================================================

class TestPrintResults:

    def test_print_error(self, capsys):
        inst = _make_instance()
        inst.tokenizer_names = ["test"]
        inst.print_results({"ast_boundary_alignment": {"error": "no tree-sitter"}})
        captured = capsys.readouterr()
        assert "no tree-sitter" in captured.out

    def test_print_empty(self, capsys):
        inst = _make_instance()
        inst.tokenizer_names = ["test"]
        inst.print_results({})
        captured = capsys.readouterr()
        # Should produce no output
        assert captured.out == ""

    def test_print_results_with_data(self, capsys):
        inst = _make_instance()
        inst.tokenizer_names = ["test_tok"]
        results = {
            "ast_boundary_alignment": {
                "per_tokenizer": {
                    "test_tok": {
                        "by_category": {
                            "identifier": {
                                "python": {
                                    "full_alignment_rate": 0.9,
                                    "start_alignment_rate": 0.95,
                                    "end_alignment_rate": 0.92,
                                    "cross_boundary_rate": 0.1,
                                    "count": 50,
                                }
                            }
                        },
                        "by_language": {
                            "python": {
                                "overall_full_alignment_rate": 0.85,
                                "overall_start_alignment_rate": 0.90,
                                "overall_end_alignment_rate": 0.88,
                                "overall_cross_boundary_rate": 0.15,
                                "count": 100,
                            }
                        },
                        "overall": {
                            "full_alignment_rate": 0.85,
                            "start_alignment_rate": 0.90,
                            "end_alignment_rate": 0.88,
                            "cross_boundary_rate": 0.15,
                            "count": 100,
                        }
                    }
                },
                "summary": {
                    "test_tok": {
                        "avg_full_alignment_rate": 0.85,
                        "avg_start_alignment_rate": 0.90,
                        "avg_end_alignment_rate": 0.88,
                        "avg_cross_boundary_rate": 0.15,
                        "total_nodes_analyzed": 100,
                        "languages_analyzed": 1,
                    }
                }
            }
        }
        inst.print_results(results)
        captured = capsys.readouterr()
        assert "AST BOUNDARY ALIGNMENT" in captured.out
        assert "SUMMARY STATISTICS" in captured.out
        assert "test_tok" in captured.out
        assert "identifier" in captured.out
        assert "python" in captured.out


# ======================================================================
# _decode_raw_token
# ======================================================================

class TestDecodeRawToken:
    """Verify whitespace-preserving token decoding."""

    def setup_method(self):
        self.inst = _make_instance()

    def test_g_prefix_to_space(self):
        assert self.inst._decode_raw_token("Ġdef") == " def"

    def test_underscore_prefix_to_space(self):
        assert self.inst._decode_raw_token("▁def") == " def"

    def test_space_prefix_to_space(self):
        assert self.inst._decode_raw_token(" def") == " def"

    def test_continuation_stripped(self):
        assert self.inst._decode_raw_token("##ing") == "ing"

    def test_end_word_stripped(self):
        assert self.inst._decode_raw_token("word</w>") == "word"

    def test_continuation_end_stripped(self):
        assert self.inst._decode_raw_token("word@@") == "word"

    def test_special_token_returns_none(self):
        assert self.inst._decode_raw_token("<|endoftext|>") is None

    def test_special_token_bracket(self):
        assert self.inst._decode_raw_token("[CLS]") is None

    def test_plain_unchanged(self):
        assert self.inst._decode_raw_token("abc") == "abc"


# ======================================================================
# _process_token (shared helper in BaseMetrics)
# ======================================================================

class TestProcessToken:
    """Verify the shared _process_token helper produces results consistent
    with both _clean_token (preserve_space=False) and _decode_raw_token
    (preserve_space=True)."""

    def setup_method(self):
        self.inst = _make_instance()

    # -- preserve_space=False (mirrors _clean_token) --

    def test_strip_g_prefix(self):
        assert self.inst._process_token("Ġdef", preserve_space=False) == "def"

    def test_strip_underscore_prefix(self):
        assert self.inst._process_token("▁def", preserve_space=False) == "def"

    def test_strip_continuation(self):
        assert self.inst._process_token("##ing", preserve_space=False) == "ing"

    def test_strip_end_word(self):
        assert self.inst._process_token("word</w>", preserve_space=False) == "word"

    def test_strip_continuation_end(self):
        assert self.inst._process_token("word@@", preserve_space=False) == "word"

    def test_special_returns_none(self):
        assert self.inst._process_token("<|endoftext|>", preserve_space=False) is None

    def test_plain_unchanged(self):
        assert self.inst._process_token("abc", preserve_space=False) == "abc"

    # -- preserve_space=True (mirrors _decode_raw_token) --

    def test_preserve_g_prefix(self):
        assert self.inst._process_token("Ġdef", preserve_space=True) == " def"

    def test_preserve_underscore_prefix(self):
        assert self.inst._process_token("▁def", preserve_space=True) == " def"

    def test_preserve_continuation(self):
        assert self.inst._process_token("##ing", preserve_space=True) == "ing"

    def test_preserve_end_word(self):
        assert self.inst._process_token("word</w>", preserve_space=True) == "word"

    def test_preserve_continuation_end(self):
        assert self.inst._process_token("word@@", preserve_space=True) == "word"

    def test_preserve_special_returns_none(self):
        assert self.inst._process_token("<|endoftext|>", preserve_space=True) is None

    def test_preserve_plain_unchanged(self):
        assert self.inst._process_token("abc", preserve_space=True) == "abc"

    # -- Consistency: _clean_token and _decode_raw_token delegate correctly --

    def test_clean_token_matches_process_token(self):
        tokens = ["Ġdef", "▁x", "##ing", "word</w>", "tok@@", "<|pad|>", "abc"]
        for t in tokens:
            assert self.inst._clean_token(t) == self.inst._process_token(t, preserve_space=False)

    def test_decode_raw_token_matches_process_token(self):
        tokens = ["Ġdef", "▁x", "##ing", "word</w>", "tok@@", "<|pad|>", "abc"]
        for t in tokens:
            assert self.inst._decode_raw_token(t) == self.inst._process_token(t, preserve_space=True)


# ======================================================================
# _build_source_char_to_token_map
# ======================================================================

class TestSourceCharToTokenMap:
    """Verify whitespace-inclusive source → token mapping."""

    def setup_method(self):
        self.inst = _make_instance()

    def test_simple_tokens(self):
        source = "abc"
        tokens = ["a", "b", "c"]
        result = self.inst._build_source_char_to_token_map(source, tokens)
        assert result == [0, 1, 2]

    def test_g_prefix_tokens_with_whitespace(self):
        source = "a b"
        tokens = ["a", "Ġb"]
        result = self.inst._build_source_char_to_token_map(source, tokens)
        assert result == [0, 1, 1]  # space is part of token 1

    def test_special_tokens_skipped(self):
        source = "ab"
        tokens = ["<|bos|>", "a", "b"]
        result = self.inst._build_source_char_to_token_map(source, tokens)
        assert result == [1, 2]

    def test_indentation_mapping(self):
        source = "    x"
        # Four spaces as a single token, then 'x'
        tokens = ["Ġ   ", "x"]  # Ġ decodes to space, so " " + "   " = "    "
        result = self.inst._build_source_char_to_token_map(source, tokens)
        assert result == [0, 0, 0, 0, 1]

    def test_longer_than_source(self):
        """Tokens with more characters than source should stop cleanly."""
        source = "ab"
        tokens = ["abc"]
        result = self.inst._build_source_char_to_token_map(source, tokens)
        assert result == [0, 0]


# ======================================================================
# _extract_line_indentation
# ======================================================================

class TestExtractLineIndentation:
    """Verify line indentation extraction."""

    def test_python_style(self):
        source = "def f():\n    return 1"
        result = ASTBoundaryMetrics._extract_line_indentation(source)
        assert len(result) == 2
        ws0, start0, end0 = result[0]
        assert ws0 == ""  # no indentation on first line
        ws1, start1, end1 = result[1]
        assert ws1 == "    "
        assert source[start1:end1] == "    "

    def test_blank_lines_excluded(self):
        source = "a\n\nb"
        result = ASTBoundaryMetrics._extract_line_indentation(source)
        assert len(result) == 2  # blank line excluded

    def test_tab_indentation(self):
        source = "\tif True:\n\t\tpass"
        result = ASTBoundaryMetrics._extract_line_indentation(source)
        assert result[0][0] == "\t"
        assert result[1][0] == "\t\t"

    def test_no_indentation(self):
        source = "x = 1\ny = 2"
        result = ASTBoundaryMetrics._extract_line_indentation(source)
        assert all(ws == "" for ws, _, _ in result)


# ======================================================================
# _count_identifier_tokens
# ======================================================================

class TestCountIdentifierTokens:
    """Verify identifier token counting."""

    def test_single_token_identifier(self):
        # source: "abc" => recon: "abc" => all token 0
        source_to_recon = [0, 1, 2]
        char_to_token = [0, 0, 0]
        result = ASTBoundaryMetrics._count_identifier_tokens(
            0, 3, source_to_recon, char_to_token
        )
        assert result == 1

    def test_multi_token_identifier(self):
        # source: "abc" => recon: "abc" => tokens [0,0,1]
        source_to_recon = [0, 1, 2]
        char_to_token = [0, 0, 1]
        result = ASTBoundaryMetrics._count_identifier_tokens(
            0, 3, source_to_recon, char_to_token
        )
        assert result == 2

    def test_unmappable_returns_none(self):
        source_to_recon = [None, None, None]
        char_to_token = [0, 0, 0]
        result = ASTBoundaryMetrics._count_identifier_tokens(
            0, 3, source_to_recon, char_to_token
        )
        assert result is None


# ======================================================================
# End-to-end: Identifier Fragmentation
# ======================================================================

class TestIdentifierFragmentationE2E:
    """End-to-end tests for identifier fragmentation metric."""

    @pytest.fixture(scope="class")
    def ts_pack(self):
        try:
            import tree_sitter_language_pack
            return tree_sitter_language_pack
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

    def test_char_tokenizer_high_fragmentation(self, ts_pack):
        """A character-level tokenizer should fragment most identifiers."""
        char_tok = _CharTokenizer()
        provider = _MockProvider("char_tok", char_tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["char_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [
            'def fibonacci(n):\n    return n\n'
        ]}
        inst.code_loader = loader

        result = inst.compute()
        ident = result["identifier_fragmentation"]
        assert "per_tokenizer" in ident
        assert "char_tok" in ident["per_tokenizer"]

        overall = ident["per_tokenizer"]["char_tok"]["overall"]
        # "fibonacci" is 9 chars -> 9 tokens -> fragmented
        # "n" is 1 char -> 1 token -> not fragmented
        assert overall["fragmentation_rate"] > 0.0
        assert overall["count"] > 0

    def test_perfect_tokenizer_zero_fragmentation(self, ts_pack):
        """A tokenizer that keeps identifiers whole should have zero fragmentation."""
        snippet = 'x = 1'
        tokens = ["x", "Ġ=", "Ġ1"]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("perfect", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["perfect"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        ident = result["identifier_fragmentation"]
        overall = ident["per_tokenizer"]["perfect"]["overall"]
        assert overall["fragmentation_rate"] == pytest.approx(0.0)


# ======================================================================
# End-to-end: Indentation Consistency
# ======================================================================

class TestIndentationConsistencyE2E:
    """End-to-end tests for indentation consistency metric."""

    @pytest.fixture(scope="class")
    def ts_pack(self):
        try:
            import tree_sitter_language_pack
            return tree_sitter_language_pack
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

    def test_python_present(self, ts_pack):
        """Python snippets should produce indentation consistency data."""
        char_tok = _CharTokenizer()
        provider = _MockProvider("char_tok", char_tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["char_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [
            'def f():\n    return 1\n    x = 2\n'
        ]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        assert "per_tokenizer" in indent
        tok_data = indent["per_tokenizer"]["char_tok"]
        assert "python" in tok_data["by_language"]
        py = tok_data["by_language"]["python"]
        assert py["total_lines"] > 0
        assert py["num_indent_levels"] > 0

    def test_non_ws_lang_excluded(self, ts_pack):
        """Non-whitespace-significant languages should not appear."""
        char_tok = _CharTokenizer()
        provider = _MockProvider("char_tok", char_tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["char_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"javascript": [
            'function f() {\n    return 1;\n}\n'
        ]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        tok_data = indent["per_tokenizer"]["char_tok"]
        # javascript is not whitespace-significant, so no by_language data
        assert "javascript" not in tok_data.get("by_language", {})

    def test_consistent_indentation_scores_high(self, ts_pack):
        """When indentation is uniform, consistency should be 1.0."""
        snippet = 'if True:\n    a = 1\n    b = 2\n    c = 3\n'
        # Each "    " (4 spaces) should produce the same token pattern
        tokens = [
            "if", "ĠTrue", "Ġ:", "\n",
            "Ġ   ", "a", "Ġ=", "Ġ1", "\n",
            "Ġ   ", "b", "Ġ=", "Ġ2", "\n",
            "Ġ   ", "c", "Ġ=", "Ġ3", "\n",
        ]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("perf", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["perf"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        py = indent["per_tokenizer"]["perf"]["by_language"]["python"]
        assert py["consistency_rate"] == pytest.approx(1.0)
        assert py["weighted_consistency"] == pytest.approx(1.0)


# ======================================================================
# print_results for new metrics
# ======================================================================

class TestPrintNewMetrics:

    def test_print_fragmentation(self, capsys):
        inst = _make_instance()
        inst.tokenizer_names = ["test_tok"]
        results = {
            "ast_boundary_alignment": {
                "per_tokenizer": {
                    "test_tok": {
                        "by_category": {},
                        "by_language": {},
                        "overall": {},
                    }
                },
                "summary": {},
            },
            "identifier_fragmentation": {
                "per_tokenizer": {
                    "test_tok": {
                        "by_language": {
                            "python": {
                                "fragmentation_rate": 0.75,
                                "avg_tokens_per_identifier": 3.2,
                                "count": 100,
                            }
                        },
                        "overall": {
                            "fragmentation_rate": 0.75,
                            "avg_tokens_per_identifier": 3.2,
                            "count": 100,
                        },
                    }
                },
                "summary": {
                    "test_tok": {
                        "fragmentation_rate": 0.75,
                        "avg_tokens_per_identifier": 3.2,
                        "identifiers_analyzed": 100,
                        "languages_analyzed": 1,
                    }
                },
            },
            "indentation_consistency": {"per_tokenizer": {}, "summary": {}},
        }
        inst.print_results(results)
        captured = capsys.readouterr()
        assert "IDENTIFIER FRAGMENTATION" in captured.out
        assert "0.750" in captured.out
        assert "3.20" in captured.out
        assert "python" in captured.out

    def test_print_indentation(self, capsys):
        inst = _make_instance()
        inst.tokenizer_names = ["test_tok"]
        results = {
            "ast_boundary_alignment": {
                "per_tokenizer": {
                    "test_tok": {
                        "by_category": {},
                        "by_language": {},
                        "overall": {},
                    }
                },
                "summary": {},
            },
            "identifier_fragmentation": {"per_tokenizer": {}, "summary": {}},
            "indentation_consistency": {
                "per_tokenizer": {
                    "test_tok": {
                        "by_language": {
                            "python": {
                                "consistency_rate": 0.9,
                                "weighted_consistency": 0.95,
                                "num_indent_levels": 3,
                                "total_lines": 20,
                            }
                        },
                    }
                },
                "summary": {
                    "test_tok": {
                        "avg_consistency_rate": 0.9,
                        "avg_weighted_consistency": 0.95,
                        "languages_analyzed": 1,
                    }
                },
            },
        }
        inst.print_results(results)
        captured = capsys.readouterr()
        assert "INDENTATION CONSISTENCY" in captured.out
        assert "0.900" in captured.out
        assert "0.950" in captured.out
        assert "python" in captured.out
