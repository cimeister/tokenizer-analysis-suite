"""Tests for tokenizer_analysis.loaders.code_data (CodeDataLoader).

Covers the max_snippet_chars truncation path in _load_language, in
particular that a snippet whose truncated form is whitespace-only is dropped
rather than entering the corpus as a blank string, and that truncation does
not otherwise alter the text it keeps.
"""

from tokenizer_analysis.loaders.code_data import CodeDataLoader


class TestMaxSnippetCharsTruncationDropsWhitespaceOnlyResult:
    """Tests for the whitespace-only-after-truncation guard in _load_language."""

    def test_truncation_that_leaves_only_whitespace_drops_the_snippet(
        self, tmp_path, caplog,
    ):
        """A file whose first max_snippet_chars characters are all whitespace
        must not enter the corpus as a blank snippet.

        Before this guard, ``s[:char_cap]`` on a file with 500 leading blank
        lines and a char_cap of 400 produced a whitespace-only string that
        went straight into code_snippets, matching neither the "no cap"
        default nor the whitespace-only check _read_file and _read_parquet
        already apply on their own paths.
        """
        lang_dir = tmp_path / "python"
        lang_dir.mkdir()
        (lang_dir / "a.py").write_text("\n" * 500 + "def g(y):\n    return y * 2\n")

        loader = CodeDataLoader({"python": str(lang_dir)}, max_snippet_chars=400)
        with caplog.at_level("WARNING"):
            loader.load_all()

        assert loader.get_code_snippets("python") == []
        assert loader.dropped_whitespace_only_counts == {"python": 1}
        assert any(
            "whitespace-only" in r.message for r in caplog.records
        ), [r.message for r in caplog.records]

    def test_truncation_that_leaves_real_content_keeps_and_truncates_it(
        self, tmp_path,
    ):
        """The ordinary case is unaffected: truncation still cuts the tail of
        an oversized snippet and keeps the (non-blank) result.
        """
        lang_dir = tmp_path / "python"
        lang_dir.mkdir()
        (lang_dir / "b.py").write_text("x" * 100)

        loader = CodeDataLoader({"python": str(lang_dir)}, max_snippet_chars=50)
        loader.load_all()

        assert loader.get_code_snippets("python") == ["x" * 50]
        assert loader.truncated_char_counts == {"python": 50}
        assert loader.dropped_whitespace_only_counts == {}

    def test_no_char_cap_leaves_a_whitespace_heavy_file_untouched(
        self, tmp_path,
    ):
        """max_snippet_chars=0 (the default) disables truncation entirely, so
        the whitespace-only guard, which only runs inside the truncation
        branch, must not affect this path: the file is kept in full, leading
        blank lines and all.
        """
        lang_dir = tmp_path / "python"
        lang_dir.mkdir()
        source = "\n" * 500 + "def g(y):\n    return y * 2\n"
        (lang_dir / "a.py").write_text(source)

        loader = CodeDataLoader({"python": str(lang_dir)})
        assert loader.max_snippet_chars == 0

        loader.load_all()
        snippets = loader.get_code_snippets("python")
        assert len(snippets) == 1
        assert snippets[0] == source.rstrip()
        assert loader.dropped_whitespace_only_counts == {}
        assert loader.truncated_char_counts == {}

    def test_truncation_keeps_trailing_whitespace_in_the_text_it_returns(
        self, tmp_path,
    ):
        """Truncation cuts the text and changes nothing else about it.

        The guard above drops a snippet that truncation reduces to whitespace.
        It is a separate question what happens to a snippet that survives, and
        the answer has to be "exactly ``s[:max_snippet_chars]``". Rstripping
        the kept prefix instead changes 4 of the 23 snippets the benchmark
        corpus truncates at 400 characters, so it would move measured values
        rather than only dropping empty ones. The other truncation test cuts
        ``"x" * 100`` at 50, which lands on a non-space character and so passes
        either way; this one lands the cut on whitespace.
        """
        lang_dir = tmp_path / "python"
        lang_dir.mkdir()
        (lang_dir / "c.py").write_text("x = 1" + " " * 20 + "y = 2")

        loader = CodeDataLoader({"python": str(lang_dir)}, max_snippet_chars=10)
        loader.load_all()

        assert loader.get_code_snippets("python") == ["x = 1     "]
        assert loader.dropped_whitespace_only_counts == {}
