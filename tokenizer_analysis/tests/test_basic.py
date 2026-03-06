"""Tests for tokenizer_analysis.metrics.basic (BasicTokenizationMetrics)."""

import pytest

from tokenizer_analysis.metrics.basic import BasicTokenizationMetrics
from tokenizer_analysis.core.input_types import TokenizedData
from typing import List

from .conftest import SimpleProvider as _SimpleProvider


def _make_td(tok_name: str, text: str, tokens: List[int], lang: str = "en") -> TokenizedData:
    return TokenizedData(
        tokenizer_name=tok_name,
        language=lang,
        tokens=tokens,
        text=text,
    )


# ======================================================================
# T5: Blank-line exclusion in avg_tokens_per_line
# ======================================================================

class TestBlankLineExclusion:

    def test_blank_lines_not_counted(self):
        """Blank lines should be excluded from line count."""
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        # Text with 2 non-blank lines and 2 blank lines
        text = "hello world\n\ngoodbye world\n\n"
        td = {tok_name: [_make_td(tok_name, text, [1, 2, 3, 4])]}

        results = metrics.compute_avg_tokens_per_line_analysis(td)
        tpl_data = results["avg_tokens_per_line"]["per_tokenizer"][tok_name]
        # 4 tokens / 2 non-blank lines = 2.0
        assert tpl_data["global_avg"] == pytest.approx(2.0)

    def test_all_blank_lines(self):
        """Text with only blank lines should produce 0 tokens per line."""
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        text = "\n\n\n"
        td = {tok_name: [_make_td(tok_name, text, [1])]}

        results = metrics.compute_avg_tokens_per_line_analysis(td)
        tpl_data = results["avg_tokens_per_line"]["per_tokenizer"][tok_name]
        assert tpl_data["global_avg"] == 0.0


# ======================================================================
# T6: Fertility skip when text is None
# ======================================================================

class TestFertilitySkip:

    def test_no_text_skipped(self):
        """Samples without text should be skipped, not use a fallback."""
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [
            # Sample WITH text
            _make_td(tok_name, "hello world", [1, 2]),
            # Sample WITHOUT text
            TokenizedData(tokenizer_name=tok_name, language="en", tokens=[3, 4, 5]),
        ]}

        results = metrics.compute(td)
        fertility_data = results["fertility"]["per_tokenizer"][tok_name]["global"]
        # Only the first sample (2 tokens / 2 words = 1.0) should be counted
        assert fertility_data["count"] == 1

    def test_whitespace_only_text_skipped(self):
        """Whitespace-only texts should be skipped."""
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [
            _make_td(tok_name, "   \n\t  ", [1, 2]),
            _make_td(tok_name, "actual text", [3, 4]),
        ]}

        results = metrics.compute(td)
        fertility_data = results["fertility"]["per_tokenizer"][tok_name]["global"]
        assert fertility_data["count"] == 1


# ======================================================================
# T7: Bytes-per-token metric
# ======================================================================

class TestBytesPerToken:

    def test_ascii_text(self):
        """For ASCII text, bytes_per_token == chars_per_token."""
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        text = "hello"  # 5 ASCII chars = 5 bytes
        td = {tok_name: [_make_td(tok_name, text, [1, 2])]}

        results = metrics.compute_token_length_analysis(td)
        tok_data = results["token_length"]["per_tokenizer"][tok_name]
        assert "byte_length" in tok_data
        char_mean = tok_data["character_length"]["mean"]
        byte_mean = tok_data["byte_length"]["mean"]
        assert char_mean == pytest.approx(byte_mean)  # ASCII: same
        assert char_mean == pytest.approx(2.5)  # 5 chars / 2 tokens

    def test_multibyte_text(self):
        """For multi-byte UTF-8, bytes_per_token > chars_per_token."""
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        text = "\u00e9\u00e9"  # 2 chars, each 2 bytes in UTF-8 = 4 bytes total
        td = {tok_name: [_make_td(tok_name, text, [1, 2])]}

        results = metrics.compute_token_length_analysis(td)
        tok_data = results["token_length"]["per_tokenizer"][tok_name]
        char_mean = tok_data["character_length"]["mean"]
        byte_mean = tok_data["byte_length"]["mean"]
        assert char_mean == pytest.approx(1.0)   # 2 chars / 2 tokens
        assert byte_mean == pytest.approx(2.0)   # 4 bytes / 2 tokens
