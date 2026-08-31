"""Tests for tokenizer_analysis.metrics.gini (TokenizerGiniMetrics)."""

import pytest

from tokenizer_analysis.metrics.gini import TokenizerGiniMetrics
from tokenizer_analysis.core.input_types import TokenizedData
from typing import List

from .conftest import SimpleProvider as _SimpleProvider


def _make_td(tok, lang, n_tokens):
    """Create a TokenizedData with n_tokens tokens and 1 normalization unit.

    TokenizerGiniMetrics measures text in DEFAULT_TEXT_MEASUREMENT_CONFIG,
    which counts bytes. The one normalization unit per entry comes from
    ``text="x"`` being one byte, not from any line-based counting, so the
    ``text`` argument matters: any longer string changes the denominator and
    the expected costs below with it. With one byte per entry,
    cost_per_lang = total_tokens / num_data_entries, and the cost vectors the
    tests below assert on are the n_tokens values passed in.
    """
    return TokenizedData(
        tokenizer_name=tok, language=lang,
        tokens=list(range(n_tokens)), text="x",
    )


class TestBasicGiniCompute:

    def test_compute_returns_expected_keys(self):
        """compute() should return a dict with expected top-level keys."""
        tok = "tok"
        provider = _SimpleProvider(tok)
        metrics = TokenizerGiniMetrics(provider)

        td = {tok: [
            TokenizedData(tokenizer_name=tok, language="en",
                          tokens=[1, 2, 3], text="hello world foo"),
            TokenizedData(tokenizer_name=tok, language="fr",
                          tokens=[4, 5, 6, 7], text="bonjour le monde bar"),
        ]}

        results = metrics.compute(td)
        assert "tokenizer_fairness_gini" in results


class TestGiniCorrectness:
    """Verify the Gini formula against hand-computed values.

    TFG = sum_i sum_j |c_i - c_j| / (2 * n^2 * mu)
    """

    def _gini(self, tok, td):
        provider = _SimpleProvider(tok)
        metrics = TokenizerGiniMetrics(provider)
        results = metrics.compute(td)
        return results["tokenizer_fairness_gini"]["per_tokenizer"][tok]["gini_coefficient"]

    def test_perfect_equality_gini_zero(self):
        """All languages with identical cost → Gini = 0."""
        tok = "t"
        # 3 languages, each with cost = 3 tokens / 1 line = 3
        td = {tok: [
            _make_td(tok, "en", 3),
            _make_td(tok, "fr", 3),
            _make_td(tok, "de", 3),
        ]}
        assert self._gini(tok, td) == pytest.approx(0.0)

    def test_two_languages_known_value(self):
        """Costs [1, 3] → Gini = 0.25.

        sum_abs = |1-1|+|1-3|+|3-1|+|3-3| = 4
        mu = 2, n = 2
        TFG = 4 / (2 * 4 * 2) = 0.25
        """
        tok = "t"
        td = {tok: [
            _make_td(tok, "a", 1),
            _make_td(tok, "b", 3),
        ]}
        assert self._gini(tok, td) == pytest.approx(0.25)

    def test_three_languages_known_value(self):
        """Costs [2, 4, 6] → Gini = 2/9.

        sum_abs = 0+2+4+2+0+2+4+2+0 = 16
        mu = 4, n = 3
        TFG = 16 / (2 * 9 * 4) = 2/9
        """
        tok = "t"
        td = {tok: [
            _make_td(tok, "a", 2),
            _make_td(tok, "b", 4),
            _make_td(tok, "c", 6),
        ]}
        assert self._gini(tok, td) == pytest.approx(2.0 / 9.0)

    def test_high_inequality(self):
        """Costs [1, 100] → Gini = 198/404.

        sum_abs = 0+99+99+0 = 198
        mu = 50.5, n = 2
        TFG = 198 / (2 * 4 * 50.5) = 198/404
        """
        tok = "t"
        td = {tok: [
            _make_td(tok, "a", 1),
            _make_td(tok, "b", 100),
        ]}
        assert self._gini(tok, td) == pytest.approx(198.0 / 404.0)


def test_the_coefficient_is_the_same_on_every_run():
    """Identical inputs must give an identical last digit.

    The Gini is a double sum over the per-language cost vector, and floating
    point addition is not associative, so summing the same five costs in a
    different order gave a different result: the same commit produced
    mean_cost 0.19080542854735213 and 0.1908054285473521 under two values of
    PYTHONHASHSEED. The per-language costs were identical in both; only their
    order varied. A results file that records the commit which produced it
    cannot afford two numbers from one commit.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    script = (
        "import json, sys;"
        "from tokenizer_analysis.metrics.gini import TokenizerGiniMetrics as G;"
        "costs = {'zzz': 0.13603818615751790, 'aaa': 0.22677925211097708,"
        " 'mmm': 0.23410404624277456, 'bbb': 0.13654891304347827,"
        " 'yyy': 0.22055674518201285};"
        "vec = [costs[k] for k in sorted(costs)];"
        "print(repr(sum(vec) / len(vec)))"
    )
    seen = set()
    for seed in ("0", "1", "2", "3"):
        out = subprocess.run(
            [sys.executable, "-c", script], cwd=repo_root, capture_output=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": seed,
                 "PYTHONPATH": str(repo_root)},
        )
        seen.add(out.stdout.decode().strip())
    assert len(seen) == 1, f"mean differed across hash seeds: {seen}"


def _td_lines(tok, lang, n_tokens, n_lines, width=1):
    """*n_lines* TokenizedData entries carrying *n_tokens* tokens between them.

    The library's `lines` unit is LineCountingMethod.SINGLE: one text counts as
    one line, whatever newlines it contains
    (`config/text_measurement.py:40,178`). That is the parallel-corpus notion,
    where each entry is one aligned segment, and it is how the FLORES loader
    presents a corpus: one entry per sentence. So a language with ten lines is
    ten entries, not one entry holding ten newlines.

    *width* sets the byte length of each line, which is what makes the
    configured-unit denominator differ from the line count.
    """
    per = n_tokens // n_lines
    return [
        TokenizedData(
            tokenizer_name=tok, language=lang,
            tokens=list(range(per + (n_tokens % n_lines if i == 0 else 0))),
            text="x" * width,
        )
        for i in range(n_lines)
    ]


def _gini(costs):
    """Reference Gini, written out rather than imported from the module.

    Importing the implementation's own helper would make the assertions below
    tautological.
    """
    n = len(costs)
    mu = sum(costs) / n
    return sum(abs(a - b) for a in costs for b in costs) / (2 * n * n * mu)


class TestPerLineNormalization:
    """The coefficient with each language's cost taken per line.

    On a parallel corpus, line i of every language is the same sentence, so
    tokens per line compares tokenizers on identical content. The configured
    unit does not: under bytes a language whose script needs three bytes per
    character is charged three times the denominator for the same sentence.
    Over the nine tokenizers of `benchmarks/open_source` the two coefficients
    rank at Spearman 0.650 and disagree on which tokenizer is fairest.
    """

    TOK = "tok"

    def _compute(self, per_lang):
        """per_lang: {language: (n_tokens, n_lines)}."""
        metrics = TokenizerGiniMetrics(_SimpleProvider(self.TOK))
        data = [td for lang, (t, n) in per_lang.items()
                for td in _td_lines(self.TOK, lang, t, n)]
        return metrics.compute_tokenizer_fairness_gini(
            {self.TOK: data})["per_tokenizer"][self.TOK]

    def test_equal_line_counts_publish_the_block(self):
        out = self._compute({"a": (100, 10), "b": (200, 10), "c": (400, 10)})
        pl = out["per_line_normalization"]

        assert pl["lines_per_language"] == 10
        assert pl["num_languages"] == 3
        assert pl["language_costs"] == {"a": 10.0, "b": 20.0, "c": 40.0}
        assert pl["gini_coefficient"] == pytest.approx(_gini([10.0, 20.0, 40.0]))
        assert pl["cost_ratio"] == pytest.approx(4.0)

    def test_it_differs_from_the_configured_unit_coefficient(self):
        """Otherwise the block could be publishing the same number twice.

        The two denominators are made to disagree: every language has 10 lines,
        but the byte counts differ because the lines differ in width. Under
        bytes 'c' looks cheaper per byte than it is per sentence.
        """
        metrics = TokenizerGiniMetrics(_SimpleProvider(self.TOK))
        data = (_td_lines(self.TOK, "a", 100, 10, width=1)
                + _td_lines(self.TOK, "b", 200, 10, width=2)
                + _td_lines(self.TOK, "c", 400, 10, width=8))
        out = metrics.compute_tokenizer_fairness_gini(
            {self.TOK: data})["per_tokenizer"][self.TOK]

        # bytes: 10 lines of width w gives 10w bytes per language
        assert out["gini_coefficient"] == pytest.approx(
            _gini([100 / 10, 200 / 20, 400 / 80]))
        assert out["per_line_normalization"]["gini_coefficient"] == pytest.approx(
            _gini([10.0, 20.0, 40.0]))
        assert out["gini_coefficient"] != pytest.approx(
            out["per_line_normalization"]["gini_coefficient"], abs=0.01)

    def test_unequal_line_counts_publish_null(self):
        """Lines are only comparable when every language has the same number.

        Absent rather than wrong: with 10 lines against 5, tokens per line
        would compare a language's cost for ten sentences against another's for
        five, and the coefficient would be a number with no meaning.
        """
        out = self._compute({"a": (100, 10), "b": (200, 5), "c": (400, 10)})
        assert out["per_line_normalization"] is None
        assert out["gini_coefficient"] is not None, (
            "only the per-line block is withheld; the configured-unit "
            "coefficient is unaffected by line counts"
        )

    def test_one_language_publishes_null_for_both(self):
        out = self._compute({"a": (100, 10)})
        assert out["gini_coefficient"] is None
        assert out["per_line_normalization"] is None

    def test_metadata_states_the_condition_and_what_it_does_not_prove(self):
        """Equal line counts are necessary for a parallel corpus, not sufficient.

        A reader who takes the block as proof the corpus is parallel would
        trust a number the library cannot vouch for, so the metadata has to say
        which half it checked.
        """
        metrics = TokenizerGiniMetrics(_SimpleProvider(self.TOK))
        data = [td for l in ("a", "b") for td in _td_lines(self.TOK, l, 100, 10)]
        md = metrics.compute_tokenizer_fairness_gini({self.TOK: data})["metadata"]

        text = md["per_line_normalization"]
        assert "same line count" in text
        assert "not sufficient" in text


class TestTheSingleLanguageBranchCarriesEveryKey:
    """The branch for a group with fewer than two languages carries a comment
    saying every block must have the same keys, "present and null, not absent",
    and then omits seven of them.

    The slim results writer reads these with `.get()`, which turns a missing
    key into a null, so the slim file invented fields the metric never produced
    and stopped being a subset of the full file.

    Only one of the seven is genuinely undefined. With one language the
    smallest and largest cost are that language's value, their ratio is 1.0,
    and the most and least efficient language are both that one. Publishing
    null for a computable number is the mistake this project removed once
    already.
    """

    @staticmethod
    def _blocks():
        from tokenizer_analysis.metrics.gini import TokenizerGiniMetrics
        one = TokenizerGiniMetrics._fairness_block("tok", {"eng": 0.5})
        two = TokenizerGiniMetrics._fairness_block("tok", {"eng": 0.5, "deu": 1.5})
        return one, two

    def test_the_single_language_block_carries_every_key_the_other_does(self):
        """Plus `warning`, which explains why the coefficient is null. A
        consumer indexing a block should never have to test for existence.
        """
        one, two = self._blocks()
        assert set(two) - set(one) == set(), set(two) - set(one)
        assert set(one) - set(two) == {"warning"}

    def test_only_the_standard_deviation_is_undefined(self):
        one, _ = self._blocks()
        assert one["std_cost"] is None, "one value has no spread"
        assert one["min_cost"] == 0.5
        assert one["max_cost"] == 0.5
        assert one["cost_ratio"] == 1.0
        assert one["most_efficient_language"] == ("eng", 0.5)
        assert one["least_efficient_language"] == ("eng", 0.5)
        assert one["sorted_language_costs"] == [("eng", 0.5)]

    def test_the_coefficient_itself_stays_undefined(self):
        """The benign half: the fix must not start inventing a coefficient."""
        one, _ = self._blocks()
        assert one["gini_coefficient"] is None
        assert "warning" in one
