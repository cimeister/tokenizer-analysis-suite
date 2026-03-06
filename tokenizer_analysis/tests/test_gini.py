"""Tests for tokenizer_analysis.metrics.gini (TokenizerGiniMetrics)."""

import pytest

from tokenizer_analysis.metrics.gini import TokenizerGiniMetrics
from tokenizer_analysis.core.input_types import TokenizedData
from typing import List

from .conftest import SimpleProvider as _SimpleProvider


# ======================================================================
# T4: Dead methods removed, basic compute works
# ======================================================================

class TestDeadMethodsRemoved:

    def test_no_compute_by_script_family(self):
        """compute_by_script_family should no longer exist."""
        assert not hasattr(TokenizerGiniMetrics, "compute_by_script_family")

    def test_no_compute_by_resource_level(self):
        """compute_by_resource_level should no longer exist."""
        assert not hasattr(TokenizerGiniMetrics, "compute_by_resource_level")


class TestBasicGiniCompute:

    def test_compute_returns_expected_keys(self):
        """compute() should return a dict with expected top-level keys."""
        tok = "tok"
        provider = _SimpleProvider(tok)
        metrics = TokenizerGiniMetrics(provider)

        # Provide data for 2+ languages (needed for Gini)
        td = {tok: [
            TokenizedData(tokenizer_name=tok, language="en",
                          tokens=[1, 2, 3], text="hello world foo"),
            TokenizedData(tokenizer_name=tok, language="fr",
                          tokens=[4, 5, 6, 7], text="bonjour le monde bar"),
        ]}

        results = metrics.compute(td)
        assert "tokenizer_fairness_gini" in results
