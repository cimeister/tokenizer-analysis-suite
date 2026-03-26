"""Tests for tokenizer_analysis.metrics.gini (TokenizerGiniMetrics)."""

import pytest

from tokenizer_analysis.metrics.gini import TokenizerGiniMetrics
from tokenizer_analysis.core.input_types import TokenizedData
from typing import List

from .conftest import SimpleProvider as _SimpleProvider


def _make_td(tok, lang, n_tokens):
    """Create a TokenizedData with n_tokens tokens and 1 normalization unit.

    Gini uses DEFAULT_LINE_MEASUREMENT_CONFIG (LineCountingMethod.SINGLE),
    so each TokenizedData counts as 1 line regardless of text content.
    This means cost_per_lang = total_tokens / num_data_entries.
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
