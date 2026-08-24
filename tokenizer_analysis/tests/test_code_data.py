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


class TestTheFileCapCountsSnippetsThatSurviveTruncation:
    """The file cap is applied to what survives, not to what was opened.

    A file that ``max_snippet_chars`` truncates to whitespace is dropped, so
    it must not also occupy one of the ``max_snippets_per_lang`` slots while
    a readable candidate goes unread. Before this, the walk stopped as soon
    as it had read *cap* files and the whitespace-only drop ran afterward, so
    a language could finish under its cap with candidates still on disk.
    """

    def test_a_file_truncated_to_whitespace_does_not_consume_a_cap_slot(
        self, tmp_path,
    ):
        lang_dir = tmp_path / "python"
        lang_dir.mkdir()
        # Sorted order matters: the walk reads a.py first.
        (lang_dir / "a.py").write_text("\n" * 500 + "def g(y):\n    return y\n")
        (lang_dir / "b.py").write_text("def h(z):\n    return z + 1\n")

        loader = CodeDataLoader(
            {"python": str(lang_dir)}, max_snippets_per_lang=1, max_snippet_chars=400,
        )
        loader.load_all()

        assert loader.get_code_snippets("python") == ["def h(z):\n    return z + 1"]
        assert loader.dropped_whitespace_only_counts == {"python": 1}

    def test_the_cap_still_stops_the_walk_once_enough_snippets_survive(
        self, tmp_path,
    ):
        """The fix must not turn the cap into "read everything".

        Two readable files under a cap of 1 still yield one snippet, and the
        second is counted as dropped rather than read.
        """
        lang_dir = tmp_path / "python"
        lang_dir.mkdir()
        (lang_dir / "a.py").write_text("def g(y):\n    return y\n")
        (lang_dir / "b.py").write_text("def h(z):\n    return z + 1\n")

        loader = CodeDataLoader({"python": str(lang_dir)}, max_snippets_per_lang=1)
        loader.load_all()

        assert loader.get_code_snippets("python") == ["def g(y):\n    return y"]
        assert loader.dropped_file_counts == {"python": 1}


class TestTheResolvedCorpusCarriesWhatTheCapsRemoved:
    """resolve_code_corpus builds a loader and discards it.

    The counters that record what the caps did lived only on that loader, so
    nothing downstream could reach them: ASTBoundaryMetrics.max_snippet_chars
    reported the loader default rather than the value its corpus was actually
    truncated with, and dropped_file_counts was always empty in a pipeline
    run whatever the caps did.
    """

    def test_the_caps_and_their_counters_survive_on_the_corpus(self, tmp_path):
        from tokenizer_analysis.loaders.corpora import resolve_code_corpus

        lang_dir = tmp_path / "python"
        lang_dir.mkdir()
        (lang_dir / "a.py").write_text("def a():\n    return " + "1" * 500 + "\n")
        (lang_dir / "b.py").write_text("def b():\n    return 2\n")

        corpus = resolve_code_corpus(
            {"python": str(lang_dir)}, max_snippets_per_lang=1, max_snippet_chars=40,
        )

        assert corpus.caps is not None
        assert corpus.caps.max_snippet_chars == 40
        assert corpus.caps.max_snippets_per_lang == 1
        assert corpus.caps.truncated_char_counts == {"python": 480}
        assert corpus.caps.dropped_file_counts == {"python": 1}

    def test_a_corpus_built_with_no_cap_records_that_too(self, tmp_path):
        """None would be indistinguishable from "nobody recorded it"."""
        from tokenizer_analysis.loaders.corpora import resolve_code_corpus

        lang_dir = tmp_path / "python"
        lang_dir.mkdir()
        (lang_dir / "a.py").write_text("def a():\n    return 1\n")

        corpus = resolve_code_corpus({"python": str(lang_dir)})

        assert corpus.caps is not None
        assert corpus.caps.max_snippet_chars == 0
        assert corpus.caps.truncated_char_counts == {}


class TestTheBundledCorpusIsCappedForEveryMetricThatReadsIt:
    """The two code metrics scored different text sets under one corpus name.

    With --max-code-files-per-lang 2 and no --code-ast-config, the operator
    and digit metrics read the registered corpus at 3 snippets per language
    while the AST metrics re-applied the cap through get_code_snippets and
    scored 2, both published as "the code corpus" with the same source.
    """

    def test_the_count_cap_reaches_the_bundled_samples(self):
        from tokenizer_analysis.loaders.corpora import (
            resolve_code_corpus, synthetic_code_corpus,
        )

        uncapped = synthetic_code_corpus()
        assert all(len(v) == 3 for v in uncapped.texts.values())

        capped = resolve_code_corpus(None, max_snippets_per_lang=2)
        assert capped.texts, "the bundled samples must still be there"
        assert all(len(v) == 2 for v in capped.texts.values())
        assert set(capped.texts) == set(uncapped.texts)
        assert capped.caps is not None
        assert capped.caps.max_snippets_per_lang == 2

    def test_both_code_metrics_then_see_the_same_texts(self):
        """The property issue 1 is about, asserted directly."""
        from tokenizer_analysis.loaders.corpora import resolve_code_corpus
        from tokenizer_analysis.loaders.code_data import CodeDataLoader

        corpus = resolve_code_corpus(None, max_snippets_per_lang=2)
        loader = CodeDataLoader(None, max_snippets_per_lang=2)
        loader.code_snippets = {k: list(v) for k, v in corpus.texts.items()}

        registered = sum(len(v) for v in corpus.texts.values())
        scored = sum(len(loader.get_code_snippets(l)) for l in corpus.texts)
        assert registered == scored

    def test_the_character_cap_is_refused_on_the_bundled_samples(self, caplog):
        """Truncating source code corrupts its syntax, so this path says so.

        The 57 bundled samples parse with zero tree-sitter errors at full
        length. Cutting them at 400 characters produces 19 ERROR or missing
        nodes, and cutting at the last line boundary before 400 still produces
        15, so every AST alignment rate would move for a reason that has
        nothing to do with any tokenizer. The whole corpus is 48715
        characters, so there is no I/O for the cap to bound here.
        """
        from tokenizer_analysis.loaders.corpora import resolve_code_corpus

        with caplog.at_level("WARNING"):
            corpus = resolve_code_corpus(None, max_snippet_chars=400)

        assert max(len(t) for v in corpus.texts.values() for t in v) > 400
        assert any("max_snippet_chars" in r.message for r in caplog.records), \
            [r.message for r in caplog.records]
        assert corpus.caps.max_snippet_chars == 0
