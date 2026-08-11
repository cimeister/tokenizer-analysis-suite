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
        # Mean-of-ratios would give (5.0 + 0.5) / 2 = 2.75. Verify it is not that.
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
        """Per-language rates should also be ratio-of-means, with their count."""
        tok = "tok"
        m = self._make_metrics(tok)
        td = {tok: [
            _make_td(tok, "hello", 2, lang="en"),       # 5 bytes / 2 tokens
            _make_td(tok, "world!", 3, lang="en"),       # 6 bytes / 3 tokens
        ]}
        results = m.compute_compression_rate(td)
        en = results["per_tokenizer"][tok]["per_language"]["en"]
        # (5 + 6) / (2 + 3) = 11 / 5 = 2.2
        assert en["compression_rate"] == pytest.approx(11.0 / 5.0)
        # count is the measurement units the rate was computed over, so the
        # pooled rate stays derivable from the per-language block alone.
        assert en["count"] == 11
        assert en["total_tokens"] == 5


# ======================================================================
# TestBigramEntropy
# ======================================================================

def _make_td_tokens(tok_name: str, tokens: list, lang: str = "en") -> TokenizedData:
    """Create a TokenizedData with explicit token IDs (no text needed)."""
    return TokenizedData(
        tokenizer_name=tok_name,
        language=lang,
        tokens=tokens,
        text="dummy",
    )


class TestBigramEntropy:

    def _make_metrics(self, tok_name: str) -> InformationTheoreticMetrics:
        provider = _SimpleProvider(tok_name)
        return InformationTheoreticMetrics(provider)

    def test_uniform_successors(self):
        """Token 1 followed equally by 2,3,4,5,6 (5 times each) → η ≈ 1.0.

        Use separate 2-token documents so successor tokens never appear as
        left elements of bigrams (tests document boundary handling too).
        """
        tok = "tok"
        m = self._make_metrics(tok)
        docs = []
        for _ in range(5):
            for succ in [2, 3, 4, 5, 6]:
                docs.append(_make_td_tokens(tok, [1, succ]))
        td = {tok: docs}
        results = m.compute_bigram_entropy(td)
        eta = results['per_tokenizer'][tok]['global_bigram_entropy']
        assert eta == pytest.approx(1.0, abs=0.01)

    def test_dominated_successor_exact(self):
        """Token 1 followed by 2 (20x) and 3 (5x) → exact η value.

        Use separate 2-token documents to isolate token 1 as the only
        left-element type, making the exact value computable.

        H = -(20/25)*log2(20/25) - (5/25)*log2(5/25)
        H_max = log2(2)
        η = H / H_max
        """
        import math
        tok = "tok"
        m = self._make_metrics(tok)
        docs = []
        for _ in range(20):
            docs.append(_make_td_tokens(tok, [1, 2]))
        for _ in range(5):
            docs.append(_make_td_tokens(tok, [1, 3]))
        td = {tok: docs}
        results = m.compute_bigram_entropy(td)
        eta = results['per_tokenizer'][tok]['global_bigram_entropy']

        p1 = 20 / 25
        p2 = 5 / 25
        h = -(p1 * math.log2(p1) + p2 * math.log2(p2))
        expected_eta = h / math.log2(2)
        assert eta == pytest.approx(expected_eta)

    def test_single_successor(self):
        """[1,2,1,2,...] (>=5 bigrams) → only one successor for type 1, η = 0."""
        tok = "tok"
        m = self._make_metrics(tok)
        # 10 repetitions of [1,2] → token 1 always followed by 2
        seq = [1, 2] * 10
        td = {tok: [_make_td_tokens(tok, seq)]}
        results = m.compute_bigram_entropy(td)
        eta = results['per_tokenizer'][tok]['global_bigram_entropy']
        assert eta == pytest.approx(0.0)

    def test_below_threshold(self):
        """[1,2,3] has 2 bigrams, both types have <3 occurrences, so all are
        filtered and nothing is measured.  The pooled value is None, not 0.0:
        an entropy of 0.0 is a real measurement meaning every context has
        exactly one successor.
        """
        tok = "tok"
        m = self._make_metrics(tok)
        td = {tok: [_make_td_tokens(tok, [1, 2, 3])]}
        results = m.compute_bigram_entropy(td)
        r = results['per_tokenizer'][tok]
        assert r['global_bigram_entropy'] is None
        assert r['global_types_evaluated'] == 0

    def test_per_language_separation(self):
        """Uniform lang should have higher η than skewed lang."""
        tok = "tok"
        m = self._make_metrics(tok)
        # Uniform language: token 1 → {2,3,4,5,6} each 5 times
        uniform_seq = []
        for _ in range(5):
            for succ in [2, 3, 4, 5, 6]:
                uniform_seq.extend([1, succ])
        # Skewed language: token 1 → 2 (20x), 3 (5x)
        skewed_seq = []
        for _ in range(20):
            skewed_seq.extend([1, 2])
        for _ in range(5):
            skewed_seq.extend([1, 3])

        td = {tok: [
            _make_td_tokens(tok, uniform_seq, lang="uniform"),
            _make_td_tokens(tok, skewed_seq, lang="skewed"),
        ]}
        results = m.compute_bigram_entropy(td)
        uniform_eta = results['per_tokenizer'][tok]['per_language']['uniform']['bigram_entropy']
        skewed_eta = results['per_tokenizer'][tok]['per_language']['skewed']['bigram_entropy']
        assert uniform_eta > skewed_eta

    def test_schema_keys_present(self):
        """All expected keys should exist in the result."""
        tok = "tok"
        m = self._make_metrics(tok)
        seq = []
        for _ in range(5):
            for succ in [2, 3, 4, 5, 6]:
                seq.extend([1, succ])
        td = {tok: [_make_td_tokens(tok, seq)]}
        results = m.compute_bigram_entropy(td)

        assert 'per_tokenizer' in results
        assert 'per_language' in results
        assert 'metadata' in results

        tok_r = results['per_tokenizer'][tok]
        assert 'global_bigram_entropy' in tok_r
        assert 'global_total_bigrams' in tok_r
        assert 'global_types_evaluated' in tok_r
        assert 'global_types_excluded' in tok_r
        assert 'per_language' in tok_r

    def test_no_bigrams_single_token_docs(self):
        """Single-token documents produce no bigrams, so the pooled value is
        None rather than 0.0."""
        tok = "tok"
        m = self._make_metrics(tok)
        td = {tok: [
            _make_td_tokens(tok, [1]),
            _make_td_tokens(tok, [2]),
        ]}
        results = m.compute_bigram_entropy(td)
        r = results['per_tokenizer'][tok]
        assert r['global_bigram_entropy'] is None
        assert r['global_total_bigrams'] == 0

    def test_bigram_entropy_in_compute(self):
        """compute() should include bigram_entropy in its results."""
        tok = "tok"
        provider = _SimpleProvider(tok)
        m = InformationTheoreticMetrics(provider)
        docs = []
        for _ in range(5):
            for succ in [2, 3, 4, 5, 6]:
                docs.append(_make_td_tokens(tok, [1, succ]))
        td = {tok: docs}
        results = m.compute(td)
        assert 'bigram_entropy' in results
        assert 'per_tokenizer' in results['bigram_entropy']
        assert tok in results['bigram_entropy']['per_tokenizer']


# ======================================================================
# TestTrigramEntropy
# ======================================================================


class TestTrigramEntropy:

    def _make_metrics(self, tok_name: str, min_trigram_occurrences: int = 3) -> InformationTheoreticMetrics:
        provider = _SimpleProvider(tok_name)
        return InformationTheoreticMetrics(
            provider, min_trigram_occurrences=min_trigram_occurrences,
        )

    def test_uniform_successors(self):
        """Context (1,2) followed equally by 3,4,5,6,7 (5 times each) → η ≈ 1.0.

        Use separate 3-token documents so context (1,2) is the only trigram
        context and successor tokens never form new contexts.
        """
        tok = "tok"
        m = self._make_metrics(tok)
        docs = []
        for _ in range(5):
            for succ in [3, 4, 5, 6, 7]:
                docs.append(_make_td_tokens(tok, [1, 2, succ]))
        td = {tok: docs}
        results = m.compute_trigram_entropy(td)
        eta = results['per_tokenizer'][tok]['global_trigram_entropy']
        assert eta == pytest.approx(1.0, abs=0.01)

    def test_dominated_successor_exact(self):
        """Context (1,2) followed by 3 (20x) and 4 (5x) → exact η value.

        H = -(20/25)*log2(20/25) - (5/25)*log2(5/25)
        H_max = log2(2)
        η = H / H_max
        """
        import math
        tok = "tok"
        m = self._make_metrics(tok)
        docs = []
        for _ in range(20):
            docs.append(_make_td_tokens(tok, [1, 2, 3]))
        for _ in range(5):
            docs.append(_make_td_tokens(tok, [1, 2, 4]))
        td = {tok: docs}
        results = m.compute_trigram_entropy(td)
        eta = results['per_tokenizer'][tok]['global_trigram_entropy']

        p1 = 20 / 25
        p2 = 5 / 25
        h = -(p1 * math.log2(p1) + p2 * math.log2(p2))
        expected_eta = h / math.log2(2)
        assert eta == pytest.approx(expected_eta)

    def test_single_successor(self):
        """Context (1,2) always followed by 3 → η = 0."""
        tok = "tok"
        m = self._make_metrics(tok)
        docs = []
        for _ in range(10):
            docs.append(_make_td_tokens(tok, [1, 2, 3]))
        td = {tok: docs}
        results = m.compute_trigram_entropy(td)
        eta = results['per_tokenizer'][tok]['global_trigram_entropy']
        assert eta == pytest.approx(0.0)

    def test_below_threshold(self):
        """[1,2,3,4] has 2 trigrams, both contexts have <3 occurrences, so all
        are filtered and the pooled value is None rather than 0.0."""
        tok = "tok"
        m = self._make_metrics(tok)
        td = {tok: [_make_td_tokens(tok, [1, 2, 3, 4])]}
        results = m.compute_trigram_entropy(td)
        r = results['per_tokenizer'][tok]
        assert r['global_trigram_entropy'] is None
        assert r['global_types_evaluated'] == 0

    def test_per_language_separation(self):
        """Uniform lang should have higher η than skewed lang."""
        tok = "tok"
        m = self._make_metrics(tok)
        # Uniform: context (1,2) → {3,4,5,6,7} each 5 times
        uniform_docs = []
        for _ in range(5):
            for succ in [3, 4, 5, 6, 7]:
                uniform_docs.append(_make_td_tokens(tok, [1, 2, succ], lang="uniform"))
        # Skewed: context (1,2) → 3 (20x), 4 (5x)
        skewed_docs = []
        for _ in range(20):
            skewed_docs.append(_make_td_tokens(tok, [1, 2, 3], lang="skewed"))
        for _ in range(5):
            skewed_docs.append(_make_td_tokens(tok, [1, 2, 4], lang="skewed"))

        td = {tok: uniform_docs + skewed_docs}
        results = m.compute_trigram_entropy(td)
        uniform_eta = results['per_tokenizer'][tok]['per_language']['uniform']['trigram_entropy']
        skewed_eta = results['per_tokenizer'][tok]['per_language']['skewed']['trigram_entropy']
        assert uniform_eta > skewed_eta

    def test_no_trigrams_short_docs(self):
        """Documents with <=2 tokens produce no trigrams, so the pooled value
        is None rather than 0.0."""
        tok = "tok"
        m = self._make_metrics(tok)
        td = {tok: [
            _make_td_tokens(tok, [1, 2]),
            _make_td_tokens(tok, [3]),
        ]}
        results = m.compute_trigram_entropy(td)
        r = results['per_tokenizer'][tok]
        assert r['global_trigram_entropy'] is None
        assert r['global_total_trigrams'] == 0

    def test_schema_keys_present(self):
        """All expected keys should exist in the result."""
        tok = "tok"
        m = self._make_metrics(tok)
        docs = []
        for _ in range(5):
            for succ in [3, 4, 5, 6, 7]:
                docs.append(_make_td_tokens(tok, [1, 2, succ]))
        td = {tok: docs}
        results = m.compute_trigram_entropy(td)

        assert 'per_tokenizer' in results
        assert 'per_language' in results
        assert 'metadata' in results

        tok_r = results['per_tokenizer'][tok]
        assert 'global_trigram_entropy' in tok_r
        assert 'global_total_trigrams' in tok_r
        assert 'global_types_evaluated' in tok_r
        assert 'global_types_excluded' in tok_r
        assert 'per_language' in tok_r

    def test_trigram_entropy_in_compute(self):
        """compute() should include trigram_entropy in its results."""
        tok = "tok"
        provider = _SimpleProvider(tok)
        m = InformationTheoreticMetrics(provider)
        docs = []
        for _ in range(5):
            for succ in [3, 4, 5, 6, 7]:
                docs.append(_make_td_tokens(tok, [1, 2, succ]))
        td = {tok: docs}
        results = m.compute(td)
        assert 'trigram_entropy' in results
        assert 'per_tokenizer' in results['trigram_entropy']
        assert tok in results['trigram_entropy']['per_tokenizer']

    def test_separate_threshold(self):
        """Trigram threshold should be independent of bigram threshold."""
        tok = "tok"
        # Set bigram threshold high, trigram threshold low
        provider = _SimpleProvider(tok)
        m = InformationTheoreticMetrics(
            provider, min_bigram_occurrences=100, min_trigram_occurrences=2,
        )
        # 3 occurrences of context (1,2) → should pass trigram threshold (2)
        docs = []
        for succ in [3, 4, 3]:
            docs.append(_make_td_tokens(tok, [1, 2, succ]))
        td = {tok: docs}

        tri_results = m.compute_trigram_entropy(td)
        tri_r = tri_results['per_tokenizer'][tok]
        assert tri_r['global_types_evaluated'] > 0

        bi_results = m.compute_bigram_entropy(td)
        bi_r = bi_results['per_tokenizer'][tok]
        # Bigram threshold is 100, so all types should be excluded
        assert bi_r['global_types_evaluated'] == 0


class TestReferenceDefinitionPerLanguage:
    """The reference-normalizer block, corpus level and per language.

    ``bigram_entropy`` and ``trigram_entropy`` publish this library's own eta,
    normalized by each context's own successor count. The
    ``reference_definition`` block reports the same corpus under Poelman et
    al.'s normalizer, ``log2(min(accessor domain, context count))``, and their
    unweighted aggregation. Nothing covered that block before.

    The per-language entries treat each language as its own corpus. That is a
    choice with a consequence worth pinning: the divisor differs by language, so
    two per-language values are not comparable to each other.
    """

    TOK = "tok"

    def _metrics(self):
        return InformationTheoreticMetrics(_SimpleProvider(self.TOK))

    def _two_language_corpus(self):
        """Two languages whose accessor domains differ by construction.

        'narrow' has one context with 2 distinct successors. 'wide' has one
        context with 8. Both contexts are perfectly uniform, so this library's
        eta is 1.0 for each and only the reference normalizer separates them.
        """
        docs = []
        for _ in range(6):
            for succ in (2, 3):
                docs.append(_make_td_tokens(self.TOK, [1, succ], lang="narrow"))
        for _ in range(6):
            for succ in range(10, 18):
                docs.append(_make_td_tokens(self.TOK, [9, succ], lang="wide"))
        return {self.TOK: docs}

    def test_per_language_entries_exist_for_every_language(self):
        results = self._metrics().compute_bigram_entropy(self._two_language_corpus())
        ref = results['reference_definition']['per_tokenizer'][self.TOK]
        assert set(ref['per_language']) == {"narrow", "wide"}
        for lang, entry in ref['per_language'].items():
            assert set(entry) == {
                'bigram_entropy', 'accessor_domain_size', 'types_evaluated', 'count'
            }, lang

    def test_each_language_is_normalized_by_its_own_accessor_domain(self):
        """The divisor is the language's own domain, not the corpus-wide one."""
        results = self._metrics().compute_bigram_entropy(self._two_language_corpus())
        ref = results['reference_definition']['per_tokenizer'][self.TOK]

        assert ref['per_language']['narrow']['accessor_domain_size'] == 2
        assert ref['per_language']['wide']['accessor_domain_size'] == 8
        # The corpus-wide domain is the union, which is neither language's.
        assert ref['accessor_domain_size'] == 10

    def test_this_librarys_eta_cannot_tell_the_two_languages_apart(self):
        """Which is the whole reason the reference block exists.

        Both contexts are uniform over their own support, so dividing by the
        context's own successor count gives 1.0 for both. The reference
        normalizer separates them, because 'narrow' could only ever have
        branched two ways.
        """
        results = self._metrics().compute_bigram_entropy(self._two_language_corpus())
        own = results['per_tokenizer'][self.TOK]['per_language']
        ref = results['reference_definition']['per_tokenizer'][self.TOK]['per_language']

        assert own['narrow']['bigram_entropy'] == pytest.approx(1.0)
        assert own['wide']['bigram_entropy'] == pytest.approx(1.0)
        assert ref['narrow']['bigram_entropy'] == pytest.approx(1.0)
        # 3 bits of observed entropy against log2(min(8, 12)) = 3 bits.
        assert ref['wide']['bigram_entropy'] == pytest.approx(1.0)

    def test_a_context_that_could_have_branched_wider_scores_below_one(self):
        """Separates the reference normalizer from this library's.

        One language, one context, 2 equally likely successors, but an accessor
        domain of 4 because two other contexts contribute successors this one
        never used. Own-successor-count normalization gives 1.0; the reference
        normalizer gives 1 bit over log2(min(4, 12)) = 0.5.
        """
        tok = self.TOK
        docs = []
        for _ in range(6):
            for succ in (2, 3):
                docs.append(_make_td_tokens(tok, [1, succ], lang="en"))
        # Two more contexts, each seen min_occ times, adding successors 4 and 5
        # to the domain without touching context 1.
        for _ in range(3):
            docs.append(_make_td_tokens(tok, [6, 4], lang="en"))
            docs.append(_make_td_tokens(tok, [7, 5], lang="en"))

        results = self._metrics().compute_bigram_entropy({tok: docs})
        own = results['per_tokenizer'][tok]['per_language']['en']['bigram_entropy']
        ref = results['reference_definition']['per_tokenizer'][tok]['per_language']['en']

        assert ref['accessor_domain_size'] == 4
        assert own > ref['bigram_entropy'], (
            "this library's normalizer is never larger than the reference's, so "
            "its eta is never smaller"
        )
        assert ref['bigram_entropy'] < 1.0

    def test_per_language_aggregation_is_unweighted(self):
        """The reference takes an unweighted mean over context types.

        This library's own eta is frequency-weighted, so if the per-language
        reference block inherited that weighting it would not be the reference
        definition. Two contexts of unequal frequency are needed to see the
        difference; with one context per language the two agree and the test
        proves nothing.

        Context 1: 20 bigrams, 2 uniform successors. Context 9: 4 bigrams, 4
        uniform successors. Accessor domain 6.
          eta(1) = log2(2) / log2(min(6, 20)) = 1 / log2(6)
          eta(9) = log2(4) / log2(min(6, 4))  = 2 / 2 = 1
        Unweighted mean 0.6935; frequency-weighted mean 0.4891.
        """
        import math
        tok = self.TOK
        docs = []
        for _ in range(10):
            for succ in (2, 3):
                docs.append(_make_td_tokens(tok, [1, succ], lang="en"))
        for succ in (10, 11, 12, 13):
            docs.append(_make_td_tokens(tok, [9, succ], lang="en"))

        results = self._metrics().compute_bigram_entropy({tok: docs})
        ref = results['reference_definition']['per_tokenizer'][tok]['per_language']['en']
        assert ref['accessor_domain_size'] == 6

        eta_1 = 1.0 / math.log2(6)
        eta_9 = 1.0
        unweighted = (eta_1 + eta_9) / 2
        weighted = (20 * eta_1 + 4 * eta_9) / 24
        assert unweighted != pytest.approx(weighted, abs=0.05), (
            "the fixture must separate the two aggregations"
        )
        assert ref['bigram_entropy'] == pytest.approx(unweighted, abs=1e-6)

    def test_trigram_publishes_the_same_shape(self):
        tok = self.TOK
        docs = []
        for _ in range(6):
            for succ in (3, 4):
                docs.append(_make_td_tokens(tok, [1, 2, succ], lang="en"))
        results = self._metrics().compute_trigram_entropy({tok: docs})
        ref = results['reference_definition']['per_tokenizer'][tok]
        assert set(ref['per_language']) == {"en"}
        assert set(ref['per_language']['en']) == {
            'trigram_entropy', 'accessor_domain_size', 'types_evaluated', 'count'
        }

    def test_metadata_says_the_per_language_divisor_differs(self):
        """The block's own metadata has to disclose the scale caveat.

        Without it a reader compares two per-language values that were divided
        by different numbers.
        """
        results = self._metrics().compute_bigram_entropy(self._two_language_corpus())
        md = results['reference_definition']['metadata']
        assert 'accessor_domain_scope' in md
        assert 'not on a common scale' in md['accessor_domain_scope']
        assert 'corpus-wide' not in md['normalizer'], (
            "the normalizer string describes both scopes now, so it must not "
            "claim the domain is always corpus-wide"
        )
