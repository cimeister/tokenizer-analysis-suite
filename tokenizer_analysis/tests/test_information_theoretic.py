"""Tests for tokenizer_analysis.metrics.information_theoretic (compression rate)."""

import pytest

from tokenizer_analysis.metrics.information_theoretic import InformationTheoreticMetrics
from tokenizer_analysis.core.input_types import TokenizedData
from tokenizer_analysis.config import TextMeasurementConfig, NormalizationMethod
from typing import List

from .conftest import SimpleProvider as _SimpleProvider


def _make_td(tok_name: str, text: str, n_tokens: int, lang: str = "en") -> TokenizedData:
    """Create a TokenizedData with *n_tokens* dummy token IDs."""
    return TokenizedData(
        tokenizer_name=tok_name,
        language=lang,
        tokens=list(range(n_tokens)),
        text=text,
    )


# ======================================================================
# T3: Compression rate uses ratio-of-means
# ======================================================================

class TestCompressionRateRatioOfMeans:

    def _make_metrics(self, tok_name: str) -> InformationTheoreticMetrics:
        provider = _SimpleProvider(tok_name)
        # Use bytes normalization for predictable unit counts
        config = TextMeasurementConfig(method=NormalizationMethod.BYTES)
        return InformationTheoreticMetrics(provider, measurement_config=config)

    def test_single_sample(self):
        """Single sample: ratio-of-means == per-sample ratio."""
        tok = "tok"
        m = self._make_metrics(tok)
        text = "hello"  # 5 bytes
        td = {tok: [_make_td(tok, text, 2)]}
        results = m.compute_compression_rate(td)
        rate = results["per_tokenizer"][tok]["global"]["compression_rate"]
        assert rate == pytest.approx(5.0 / 2.0)

    def test_ratio_of_means_not_mean_of_ratios(self):
        """Two samples with different sizes: ratio-of-means != mean-of-ratios.

        Sample 1: 10 bytes, 5 tokens  -> per-sample ratio = 2.0
        Sample 2: 2 bytes,  1 token   -> per-sample ratio = 2.0
        Mean-of-ratios = 2.0
        Ratio-of-means = 12 / 6 = 2.0  (same in this case)

        Now skew it:
        Sample 1: 10 bytes, 2 tokens  -> per-sample ratio = 5.0
        Sample 2: 2 bytes,  4 tokens  -> per-sample ratio = 0.5
        Mean-of-ratios = 2.75
        Ratio-of-means = 12 / 6 = 2.0
        """
        tok = "tok"
        m = self._make_metrics(tok)
        # "helloworld" = 10 bytes, "hi" = 2 bytes
        td = {tok: [
            _make_td(tok, "helloworld", 2),  # 10 bytes / 2 tokens = 5.0
            _make_td(tok, "hi", 4),           # 2 bytes / 4 tokens = 0.5
        ]}
        results = m.compute_compression_rate(td)
        rate = results["per_tokenizer"][tok]["global"]["compression_rate"]
        # Ratio-of-means: (10 + 2) / (2 + 4) = 12 / 6 = 2.0
        assert rate == pytest.approx(2.0)
        # Mean-of-ratios would give (5.0 + 0.5) / 2 = 2.75 — verify it's NOT that
        assert rate != pytest.approx(2.75)

    def test_totals_reported(self):
        """Global dict should include total_units and total_tokens."""
        tok = "tok"
        m = self._make_metrics(tok)
        td = {tok: [_make_td(tok, "abc", 3)]}
        results = m.compute_compression_rate(td)
        g = results["per_tokenizer"][tok]["global"]
        assert g["total_units"] == 3   # 3 ASCII bytes
        assert g["total_tokens"] == 3

    def test_per_language(self):
        """Per-language rates should also be ratio-of-means."""
        tok = "tok"
        m = self._make_metrics(tok)
        td = {tok: [
            _make_td(tok, "hello", 2, lang="en"),       # 5 bytes / 2 tokens
            _make_td(tok, "world!", 3, lang="en"),       # 6 bytes / 3 tokens
        ]}
        results = m.compute_compression_rate(td)
        en_rate = results["per_tokenizer"][tok]["per_language"]["en"]
        # (5 + 6) / (2 + 3) = 11 / 5 = 2.2
        assert en_rate == pytest.approx(11.0 / 5.0)
