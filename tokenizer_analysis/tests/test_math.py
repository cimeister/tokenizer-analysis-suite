"""Tests for tokenizer_analysis.metrics.math (DigitBoundaryMetrics)."""

import math
import re
import pytest

from tokenizer_analysis.metrics.math import DigitBoundaryMetrics
from tokenizer_analysis.core.input_types import TokenizedData


# ======================================================================
# Helpers
# ======================================================================

# Tolerance for floating-point comparisons
_EPS = 1e-9


def _make_instance():
    """Return a bare DigitBoundaryMetrics without a live InputProvider.

    Only usable for calling static / class methods and methods that don't
    touch ``self.input_provider``.
    """
    inst = object.__new__(DigitBoundaryMetrics)
    inst._tokenizer_vocab_cache = {}
    return inst


# ======================================================================
# _ideal_boundaries
# ======================================================================

class TestIdealBoundaries:
    """Right-aligned grouping at positions L-3, L-6, ... from the left."""

    def test_single_digit(self):
        assert DigitBoundaryMetrics._ideal_boundaries(1) == set()

    def test_two_digits(self):
        assert DigitBoundaryMetrics._ideal_boundaries(2) == set()

    def test_three_digits(self):
        # 3 digits fit in one group -> no internal boundary
        assert DigitBoundaryMetrics._ideal_boundaries(3) == set()

    def test_four_digits(self):
        # "X|XXX"  -> boundary at position 1
        assert DigitBoundaryMetrics._ideal_boundaries(4) == {1}

    def test_six_digits(self):
        # "XXX|XXX" -> boundary at position 3
        assert DigitBoundaryMetrics._ideal_boundaries(6) == {3}

    def test_seven_digits(self):
        # "X|XXX|XXX" -> boundaries at 1 and 4
        assert DigitBoundaryMetrics._ideal_boundaries(7) == {1, 4}

    def test_nine_digits(self):
        # "XXX|XXX|XXX" -> boundaries at 3 and 6
        assert DigitBoundaryMetrics._ideal_boundaries(9) == {3, 6}

    def test_ten_digits(self):
        # "X|XXX|XXX|XXX" -> 1, 4, 7
        assert DigitBoundaryMetrics._ideal_boundaries(10) == {1, 4, 7}

    def test_twelve_digits(self):
        # "XXX|XXX|XXX|XXX" -> 3, 6, 9
        assert DigitBoundaryMetrics._ideal_boundaries(12) == {3, 6, 9}


# ======================================================================
# _score_boundaries — vacuous cases
# ======================================================================

class TestScoreBoundariesVacuous:
    """The four vacuous-case rows from the docstring table."""

    def test_both_empty(self):
        # Short number, single token — perfect.
        result = DigitBoundaryMetrics._score_boundaries(set(), set())
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_actual_nonempty_ideal_empty(self):
        # Short number needlessly split — all boundaries spurious.
        result = DigitBoundaryMetrics._score_boundaries({1}, set())
        assert result == {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    def test_actual_nonempty_ideal_empty_multiple(self):
        result = DigitBoundaryMetrics._score_boundaries({1, 2}, set())
        assert result == {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    def test_actual_empty_ideal_nonempty(self):
        # Long number kept as single token — no wrong boundaries but ideal missed.
        result = DigitBoundaryMetrics._score_boundaries(set(), {1, 4})
        assert result == {"precision": 1.0, "recall": 0.0, "f1": 0.0}


# ======================================================================
# _score_boundaries — normal cases
# ======================================================================

class TestScoreBoundariesNormal:
    """Non-vacuous cases from the class docstring worked examples."""

    def test_perfect_match(self):
        # "1234567" tokenized as "1" "234" "567" -> actual {1,4}, ideal {1,4}
        result = DigitBoundaryMetrics._score_boundaries({1, 4}, {1, 4})
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_recall(self):
        # "1234567" tokenized as "1234" "567" -> actual {4}, ideal {1,4}
        result = DigitBoundaryMetrics._score_boundaries({4}, {1, 4})
        assert result["precision"] == 1.0
        assert result["recall"] == pytest.approx(0.5)
        assert result["f1"] == pytest.approx(2 / 3)

    def test_complete_miss(self):
        # "1234567" tokenized as "12" "345" "67" -> actual {2,5}, ideal {1,4}
        result = DigitBoundaryMetrics._score_boundaries({2, 5}, {1, 4})
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_partial_overlap(self):
        # One of two ideal boundaries hit, one spurious.
        # actual {1,3}, ideal {1,4} -> TP=1, FP=1, FN=1
        result = DigitBoundaryMetrics._score_boundaries({1, 3}, {1, 4})
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(0.5)
        assert result["f1"] == pytest.approx(0.5)


# ======================================================================
# _clean_token
# ======================================================================

class TestCleanToken:
    """Subword marker stripping."""

    @pytest.fixture()
    def inst(self):
        return _make_instance()

    def test_sentencepiece_space_prefix(self, inst):
        assert inst._clean_token("▁hello") == "hello"

    def test_gpt_space_prefix(self, inst):
        assert inst._clean_token("Ġworld") == "world"

    def test_literal_space_prefix(self, inst):
        assert inst._clean_token(" foo") == "foo"

    def test_bert_continuation(self, inst):
        assert inst._clean_token("##bar") == "bar"

    def test_bpe_end_of_word(self, inst):
        assert inst._clean_token("baz</w>") == "baz"

    def test_bpe_continuation_suffix(self, inst):
        assert inst._clean_token("qux@@") == "qux"

    def test_special_token_angle(self, inst):
        assert inst._clean_token("<|endoftext|>") is None

    def test_special_token_bracket(self, inst):
        assert inst._clean_token("[CLS]") is None

    def test_plain_token(self, inst):
        assert inst._clean_token("hello") == "hello"

    def test_digit_token(self, inst):
        assert inst._clean_token("1234") == "1234"


# ======================================================================
# _build_char_to_token_map
# ======================================================================

class TestBuildCharToTokenMap:

    @pytest.fixture()
    def inst(self):
        return _make_instance()

    def test_plain_tokens(self, inst):
        text, mapping = inst._build_char_to_token_map(["abc", "de"])
        assert text == "abcde"
        assert mapping == [0, 0, 0, 1, 1]

    def test_strips_space_prefix(self, inst):
        text, mapping = inst._build_char_to_token_map(["Ġhello", "Ġworld"])
        assert text == "helloworld"
        assert mapping == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    def test_skips_special_tokens(self, inst):
        text, mapping = inst._build_char_to_token_map(
            ["<|start|>", "abc", "<|end|>"]
        )
        assert text == "abc"
        assert mapping == [1, 1, 1]

    def test_empty_input(self, inst):
        text, mapping = inst._build_char_to_token_map([])
        assert text == ""
        assert mapping == []

    def test_mixed_markers(self, inst):
        tokens = ["hello", "##123", "Ġ456"]
        text, mapping = inst._build_char_to_token_map(tokens)
        assert text == "hello123456"
        assert mapping == [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]


# ======================================================================
# _find_number_spans
# ======================================================================

class TestFindNumberSpans:

    def test_no_digits(self):
        assert DigitBoundaryMetrics._find_number_spans("no digits here") == []

    def test_single_number(self):
        spans = DigitBoundaryMetrics._find_number_spans("abc123def")
        assert spans == [(3, 6, "123")]

    def test_multiple_numbers(self):
        spans = DigitBoundaryMetrics._find_number_spans("a12b345c")
        assert spans == [(1, 3, "12"), (4, 7, "345")]

    def test_adjacent_to_text(self):
        spans = DigitBoundaryMetrics._find_number_spans("2024year")
        assert spans == [(0, 4, "2024")]

    def test_duplicate_numbers(self):
        spans = DigitBoundaryMetrics._find_number_spans("x2024y2024z")
        assert len(spans) == 2
        assert spans[0] == (1, 5, "2024")
        assert spans[1] == (6, 10, "2024")


# ======================================================================
# _get_digit_span_boundaries
# ======================================================================

class TestGetDigitSpanBoundaries:

    def test_single_token_number(self):
        # "1234" all mapped to token 0 -> no boundaries
        char_to_token = [0, 0, 0, 0]
        result = DigitBoundaryMetrics._get_digit_span_boundaries(
            char_to_token, 0, 4
        )
        assert result == []

    def test_two_token_split(self):
        # "12|34" -> token 0 for first 2, token 1 for last 2
        char_to_token = [0, 0, 1, 1]
        result = DigitBoundaryMetrics._get_digit_span_boundaries(
            char_to_token, 0, 4
        )
        assert result == [2]

    def test_three_token_split(self):
        # "1|234|567" -> boundaries at 1 and 4
        char_to_token = [0, 1, 1, 1, 2, 2, 2]
        result = DigitBoundaryMetrics._get_digit_span_boundaries(
            char_to_token, 0, 7
        )
        assert result == [1, 4]

    def test_offset_span(self):
        # Digit span embedded in larger text: "abc1234def"
        # char_to_token indices:               0,0,0,1,1,2,2,3,3,3
        # Span is positions 3..7 ("1234"), tokens 1,1,2,2 -> boundary at 2
        char_to_token = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]
        result = DigitBoundaryMetrics._get_digit_span_boundaries(
            char_to_token, 3, 7
        )
        assert result == [2]

    def test_span_exceeds_map(self):
        char_to_token = [0, 0]
        result = DigitBoundaryMetrics._get_digit_span_boundaries(
            char_to_token, 0, 5
        )
        assert result is None

    def test_duplicate_digit_strings_get_own_boundaries(self):
        # "x2024y2024z" where first "2024" is split as "20|24" (tok 1,2)
        # and second "2024" is a single token (tok 4).
        #        x     2   0     2   4     y     2   0   2   4     z
        c2t = [  0,    1,  1,    2,  2,    3,    4,  4,  4,  4,    5]
        spans = DigitBoundaryMetrics._find_number_spans("x2024y2024z")
        b0 = DigitBoundaryMetrics._get_digit_span_boundaries(
            c2t, spans[0][0], spans[0][1]
        )
        b1 = DigitBoundaryMetrics._get_digit_span_boundaries(
            c2t, spans[1][0], spans[1][1]
        )
        assert b0 == [2], "first occurrence should split at position 2"
        assert b1 == [], "second occurrence should be single-token"

    def test_every_digit_separate_token(self):
        # "1234" with each digit a separate token
        char_to_token = [0, 1, 2, 3]
        result = DigitBoundaryMetrics._get_digit_span_boundaries(
            char_to_token, 0, 4
        )
        assert result == [1, 2, 3]


# ======================================================================
# _digit_length_bucket / _is_short_bucket
# ======================================================================

class TestBucketHelpers:

    @pytest.mark.parametrize("length,expected", [
        (1, "1"), (5, "5"), (9, "9"), (10, "10+"), (15, "10+"), (100, "10+"),
    ])
    def test_digit_length_bucket(self, length, expected):
        assert DigitBoundaryMetrics._digit_length_bucket(length) == expected

    @pytest.mark.parametrize("bucket,expected", [
        ("1", True), ("2", True), ("3", True),
        ("4", False), ("9", False), ("10+", False),
    ])
    def test_is_short_bucket(self, bucket, expected):
        assert DigitBoundaryMetrics._is_short_bucket(bucket) is expected


# ======================================================================
# _compute_pattern_entropy
# ======================================================================

class TestComputePatternEntropy:

    def test_empty_list(self):
        result = DigitBoundaryMetrics._compute_pattern_entropy([])
        assert result["entropy"] == 0.0
        assert result["normalized_entropy"] == 0.0
        assert result["num_patterns"] == 0
        assert result["count"] == 0

    def test_single_pattern_repeated(self):
        # All identical -> H = 0
        result = DigitBoundaryMetrics._compute_pattern_entropy([(1,)] * 10)
        assert result["entropy"] == 0.0
        assert result["normalized_entropy"] == 0.0
        assert result["num_patterns"] == 1
        assert result["dominant_pattern"] == (1,)
        assert result["dominant_pattern_freq"] == 1.0
        assert result["count"] == 10

    def test_two_patterns_equal_frequency(self):
        # 50/50 split of 2 patterns -> H = 1.0 bit, normalized = 1.0
        pats = [(1,)] * 5 + [(2,)] * 5
        result = DigitBoundaryMetrics._compute_pattern_entropy(pats)
        assert result["entropy"] == pytest.approx(1.0)
        assert result["normalized_entropy"] == pytest.approx(1.0)
        assert result["num_patterns"] == 2
        assert result["count"] == 10

    def test_three_patterns_uniform(self):
        # Uniform distribution over 3 patterns -> H = log2(3) ≈ 1.585
        pats = [(1,)] * 4 + [(2,)] * 4 + [(3,)] * 4
        result = DigitBoundaryMetrics._compute_pattern_entropy(pats)
        assert result["entropy"] == pytest.approx(math.log2(3))
        assert result["normalized_entropy"] == pytest.approx(1.0)
        assert result["num_patterns"] == 3
        assert result["count"] == 12

    def test_skewed_distribution(self):
        # 80/20 split -> H = -(0.8*log2(0.8) + 0.2*log2(0.2))
        pats = [(1,)] * 8 + [(2,)] * 2
        result = DigitBoundaryMetrics._compute_pattern_entropy(pats)
        expected_h = -(0.8 * math.log2(0.8) + 0.2 * math.log2(0.2))
        assert result["entropy"] == pytest.approx(expected_h)
        assert result["normalized_entropy"] == pytest.approx(
            expected_h / math.log2(2)
        )
        assert result["dominant_pattern"] == (1,)
        assert result["dominant_pattern_freq"] == pytest.approx(0.8)

    def test_single_observation(self):
        result = DigitBoundaryMetrics._compute_pattern_entropy([(3, 6)])
        assert result["entropy"] == 0.0
        assert result["num_patterns"] == 1
        assert result["count"] == 1

    def test_empty_pattern_tuples(self):
        # Numbers tokenized as single tokens -> pattern ()
        result = DigitBoundaryMetrics._compute_pattern_entropy([(), (), ()])
        assert result["entropy"] == 0.0
        assert result["dominant_pattern"] == ()

    def test_normalized_entropy_bounded(self):
        # Normalized entropy must be in [0, 1] for any distribution.
        pats = [(1,)] * 7 + [(2,)] * 2 + [(1, 3)] * 1
        result = DigitBoundaryMetrics._compute_pattern_entropy(pats)
        assert 0.0 <= result["normalized_entropy"] <= 1.0


# ======================================================================
# _score_boundaries + _ideal_boundaries end-to-end
#
# These follow the worked examples from the class docstring verbatim.
# ======================================================================

class TestDocstringWorkedExamples:
    """Verify every worked example from the class docstring."""

    # -- "1234567" (L=7), ideal = {1, 4} --

    def test_1234567_perfect(self):
        # "1" "234" "567" -> actual {1, 4}
        ideal = DigitBoundaryMetrics._ideal_boundaries(7)
        assert ideal == {1, 4}
        r = DigitBoundaryMetrics._score_boundaries({1, 4}, ideal)
        assert r["f1"] == pytest.approx(1.0)

    def test_1234567_partial(self):
        # "1234" "567" -> actual {4}
        r = DigitBoundaryMetrics._score_boundaries({4}, {1, 4})
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(0.5)
        assert r["f1"] == pytest.approx(2 / 3, abs=0.01)

    def test_1234567_wrong(self):
        # "12" "345" "67" -> actual {2, 5}
        r = DigitBoundaryMetrics._score_boundaries({2, 5}, {1, 4})
        assert r["f1"] == pytest.approx(0.0)

    def test_1234567_single_token(self):
        # "1234567" -> actual {}
        r = DigitBoundaryMetrics._score_boundaries(set(), {1, 4})
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(0.0)
        assert r["f1"] == pytest.approx(0.0)

    # -- "42" (L=2), ideal = {} --

    def test_42_single_token(self):
        # "42" -> actual {}
        ideal = DigitBoundaryMetrics._ideal_boundaries(2)
        assert ideal == set()
        r = DigitBoundaryMetrics._score_boundaries(set(), ideal)
        assert r["f1"] == pytest.approx(1.0)

    def test_42_needlessly_split(self):
        # "4" "2" -> actual {1}
        r = DigitBoundaryMetrics._score_boundaries({1}, set())
        assert r["precision"] == pytest.approx(0.0)
        assert r["f1"] == pytest.approx(0.0)


# ======================================================================
# End-to-end: _build_char_to_token_map -> _find_number_spans
#             -> _get_digit_span_boundaries -> _score_boundaries
#
# Simulate what compute() does for a single text, without needing a
# real tokenizer or InputProvider.
# ======================================================================

class TestEndToEndBoundaryPipeline:
    """Full pipeline from token strings to alignment scores."""

    @pytest.fixture()
    def inst(self):
        return _make_instance()

    def _run_pipeline(self, inst, token_strings):
        """Run the boundary pipeline on a list of token strings.

        Returns a list of ``(digit_str, boundaries, scores)`` tuples.
        """
        recon_text, c2t = inst._build_char_to_token_map(token_strings)
        spans = DigitBoundaryMetrics._find_number_spans(recon_text)
        out = []
        for start, end, digit_str in spans:
            boundaries = DigitBoundaryMetrics._get_digit_span_boundaries(
                c2t, start, end
            )
            if boundaries is None:
                continue
            actual = set(boundaries)
            ideal = DigitBoundaryMetrics._ideal_boundaries(len(digit_str))
            scores = DigitBoundaryMetrics._score_boundaries(actual, ideal)
            out.append((digit_str, boundaries, scores))
        return out

    def test_number_in_sentence(self, inst):
        # "the year 2024 was" tokenized as: "the" "Ġyear" "Ġ2024" "Ġwas"
        # "2024" is a single token -> ideal {1}, actual {} -> P=1, R=0, F1=0
        results = self._run_pipeline(
            inst, ["the", "Ġyear", "Ġ2024", "Ġwas"]
        )
        assert len(results) == 1
        digit_str, boundaries, scores = results[0]
        assert digit_str == "2024"
        assert boundaries == []
        assert scores["f1"] == pytest.approx(0.0)

    def test_number_split_correctly(self, inst):
        # "1234567" split as "1" "234" "567" (ideal right-aligned)
        results = self._run_pipeline(inst, ["1", "234", "567"])
        assert len(results) == 1
        digit_str, boundaries, scores = results[0]
        assert digit_str == "1234567"
        assert sorted(boundaries) == [1, 4]
        assert scores["f1"] == pytest.approx(1.0)

    def test_short_number_single_token(self, inst):
        # "42" as single token -> both sets empty -> perfect
        results = self._run_pipeline(inst, ["Ġ42"])
        assert len(results) == 1
        _, boundaries, scores = results[0]
        assert boundaries == []
        assert scores["f1"] == pytest.approx(1.0)

    def test_short_number_needlessly_split(self, inst):
        # "42" split into "4" "2" -> actual {1}, ideal {} -> F1=0
        results = self._run_pipeline(inst, ["4", "2"])
        assert len(results) == 1
        _, boundaries, scores = results[0]
        assert boundaries == [1]
        assert scores["f1"] == pytest.approx(0.0)

    def test_multiple_numbers_in_text(self, inst):
        # "from 2024 to 2025" -> two 4-digit numbers
        tokens = ["from", "Ġ2024", "Ġto", "Ġ2025"]
        results = self._run_pipeline(inst, tokens)
        assert len(results) == 2
        assert results[0][0] == "2024"
        assert results[1][0] == "2025"

    def test_duplicate_numbers_independent(self, inst):
        # "2024 and 2024" with different tokenizations:
        # first "2024" as single token, second "2024" as "20" "24"
        tokens = ["2024", "Ġand", "Ġ20", "24"]
        results = self._run_pipeline(inst, tokens)
        assert len(results) == 2
        # First "2024": single token -> no boundaries
        assert results[0][1] == []
        # Second "2024": split at position 2
        assert results[1][1] == [2]

    def test_no_numbers(self, inst):
        tokens = ["hello", "Ġworld"]
        results = self._run_pipeline(inst, tokens)
        assert results == []

    def test_bert_style_tokens(self, inst):
        # BERT-style: "12345" as "123" "##45"
        # -> recon "12345", boundaries at 3
        # ideal for L=5: {2} (5-3=2)
        # actual: {3} -> TP=0, FP=1, FN=1 -> F1=0
        results = self._run_pipeline(inst, ["123", "##45"])
        assert len(results) == 1
        _, boundaries, scores = results[0]
        assert boundaries == [3]
        assert scores["f1"] == pytest.approx(0.0)

    def test_uniform_chunk_via_pipeline(self, inst):
        # "123456" as "123" "456" -> two chunks of 3 -> uniform
        recon, c2t = inst._build_char_to_token_map(["123", "456"])
        spans = DigitBoundaryMetrics._find_number_spans(recon)
        assert len(spans) == 1
        start, end, digit_str = spans[0]
        boundaries = DigitBoundaryMetrics._get_digit_span_boundaries(
            c2t, start, end
        )
        bnd_list = sorted(boundaries)
        chunk_lengths = []
        prev = 0
        for b in bnd_list:
            chunk_lengths.append(b - prev)
            prev = b
        chunk_lengths.append(len(digit_str) - prev)
        assert len(set(chunk_lengths)) == 1  # uniform

    def test_non_uniform_chunk_via_pipeline(self, inst):
        # "12345" as "12" "345" -> chunks of 2 and 3 -> not uniform
        recon, c2t = inst._build_char_to_token_map(["12", "345"])
        spans = DigitBoundaryMetrics._find_number_spans(recon)
        start, end, digit_str = spans[0]
        boundaries = DigitBoundaryMetrics._get_digit_span_boundaries(
            c2t, start, end
        )
        bnd_list = sorted(boundaries)
        chunk_lengths = []
        prev = 0
        for b in bnd_list:
            chunk_lengths.append(b - prev)
            prev = b
        chunk_lengths.append(len(digit_str) - prev)
        assert len(set(chunk_lengths)) > 1  # not uniform


# ======================================================================
# Entropy: pooling vs averaging
#
# Verify that the summary computes entropy from the *pooled* pattern
# distribution, not by averaging pre-computed per-group entropies.
# ======================================================================

class TestPooledEntropy:
    """The summary must pool patterns before computing entropy."""

    def test_pooling_reveals_cross_group_variation(self):
        """Two groups each perfectly consistent but with different patterns.

        Per-group entropy is 0 for each, but the pooled distribution has
        two distinct patterns at 50/50 -> H = 1.0 bit.
        """
        pats_a = [(1,)] * 5
        pats_b = [(2,)] * 5

        ha = DigitBoundaryMetrics._compute_pattern_entropy(pats_a)
        hb = DigitBoundaryMetrics._compute_pattern_entropy(pats_b)
        pooled = DigitBoundaryMetrics._compute_pattern_entropy(pats_a + pats_b)

        # Per-group: zero
        assert ha["entropy"] == pytest.approx(0.0)
        assert hb["entropy"] == pytest.approx(0.0)
        # Average of zeros is zero (the old, wrong approach)
        assert (ha["entropy"] + hb["entropy"]) / 2 == pytest.approx(0.0)
        # Pooled: 1.0 bit (the correct approach)
        assert pooled["entropy"] == pytest.approx(1.0)

    def test_pooling_identical_groups(self):
        """When both groups have the same distribution, pooled == per-group."""
        pats = [(1,)] * 3 + [(2,)] * 3
        per_group = DigitBoundaryMetrics._compute_pattern_entropy(pats)
        pooled = DigitBoundaryMetrics._compute_pattern_entropy(pats + pats)
        assert pooled["entropy"] == pytest.approx(per_group["entropy"])

    def test_pooling_subset_strictly_lower_entropy(self):
        """A homogeneous subset has entropy <= the pooled set."""
        pats_homogeneous = [(1,)] * 10
        pats_mixed = [(1,)] * 10 + [(2,)] * 10
        h_homo = DigitBoundaryMetrics._compute_pattern_entropy(pats_homogeneous)
        h_mixed = DigitBoundaryMetrics._compute_pattern_entropy(pats_mixed)
        assert h_homo["entropy"] < h_mixed["entropy"]


# ======================================================================
# _compute_fertility_scaling
# ======================================================================

class TestComputeFertilityScaling:
    """Scaling statistics for numeric magnitude consistency."""

    def test_empty_input(self):
        result = DigitBoundaryMetrics._compute_fertility_scaling({})
        assert result["per_bucket"] == {}
        assert result["spearman_rho"] is None
        assert result["cv_of_mean_fertility"] == 0.0
        assert result["linear_fit"] is None

    def test_single_bucket(self):
        result = DigitBoundaryMetrics._compute_fertility_scaling(
            {"4": [0.5, 0.5, 0.5]}
        )
        assert "4" in result["per_bucket"]
        assert result["per_bucket"]["4"]["mean_fertility"] == pytest.approx(0.5)
        assert result["per_bucket"]["4"]["count"] == 3
        # Only one bucket => no correlation possible
        assert result["spearman_rho"] is None
        assert result["linear_fit"] is None

    def test_two_buckets(self):
        result = DigitBoundaryMetrics._compute_fertility_scaling(
            {"1": [1.0, 1.0], "4": [0.5, 0.5]}
        )
        assert result["spearman_rho"] is not None
        assert result["linear_fit"] is not None
        assert "slope" in result["linear_fit"]
        assert "r_squared" in result["linear_fit"]

    def test_constant_fertility_zero_cv(self):
        # All buckets have identical mean fertility => CV = 0
        result = DigitBoundaryMetrics._compute_fertility_scaling(
            {"1": [1.0], "2": [1.0], "3": [1.0], "4": [1.0]}
        )
        assert result["cv_of_mean_fertility"] == pytest.approx(0.0)

    def test_increasing_fertility_high_rho(self):
        # Fertility increases with digit length => positive rho
        result = DigitBoundaryMetrics._compute_fertility_scaling(
            {"1": [0.2], "2": [0.4], "3": [0.6], "4": [0.8]}
        )
        assert result["spearman_rho"] is not None
        assert result["spearman_rho"] > 0.9

    def test_ten_plus_bucket_uses_10(self):
        # The "10+" bucket is treated as digit length 10 for scaling
        result = DigitBoundaryMetrics._compute_fertility_scaling(
            {"1": [1.0], "10+": [0.5]}
        )
        assert result["spearman_rho"] is not None
        # Fertility decreases with length => negative rho
        assert result["spearman_rho"] < 0


# ======================================================================
# Operator regex and categories
# ======================================================================

class TestOperatorRegex:
    """Operator span regex and category mapping."""

    @pytest.mark.parametrize("op", ["+", "-", "*", "/", "=", "<", ">", "!", "&", "|", "^", "~", "%"])
    def test_single_char_operators(self, op):
        m = DigitBoundaryMetrics._OPERATOR_SPAN.search(op)
        assert m is not None
        assert m.group() == op

    @pytest.mark.parametrize("op", ["**", "<=", ">=", "==", "!=", "&&", "||", "<<", ">>"])
    def test_multi_char_operators(self, op):
        m = DigitBoundaryMetrics._OPERATOR_SPAN.search(op)
        assert m is not None
        assert m.group() == op

    def test_multi_char_longest_match(self):
        # "**" should match as compound, not two "*"
        matches = list(DigitBoundaryMetrics._OPERATOR_SPAN.finditer("**"))
        assert len(matches) == 1
        assert matches[0].group() == "**"

    def test_no_operators_in_text(self):
        assert DigitBoundaryMetrics._OPERATOR_SPAN.search("hello world 42") is None

    def test_category_lookup(self):
        assert DigitBoundaryMetrics._OPERATOR_TO_CATEGORY["+"] == "arithmetic"
        assert DigitBoundaryMetrics._OPERATOR_TO_CATEGORY["**"] == "arithmetic"
        assert DigitBoundaryMetrics._OPERATOR_TO_CATEGORY["<="] == "comparison"
        assert DigitBoundaryMetrics._OPERATOR_TO_CATEGORY["="] == "assignment"
        assert DigitBoundaryMetrics._OPERATOR_TO_CATEGORY["&&"] == "logical_bitwise"
        assert DigitBoundaryMetrics._OPERATOR_TO_CATEGORY["<<"] == "shift"

    def test_operator_embedded_in_text(self):
        matches = list(DigitBoundaryMetrics._OPERATOR_SPAN.finditer("a+b<=c"))
        ops = [m.group() for m in matches]
        assert "+" in ops
        assert "<=" in ops


# ======================================================================
# Operator isolation logic
# ======================================================================

class TestOperatorIsolation:
    """Operator isolation checks using the char_to_token / token_to_chars approach."""

    @pytest.fixture()
    def inst(self):
        return _make_instance()

    def _check_isolation(self, inst, token_strings):
        """Run operator isolation check on token strings.

        Returns a list of ``(op_str, isolated, compound_preserved)`` tuples.
        """
        from collections import defaultdict

        recon_text, char_to_token = inst._build_char_to_token_map(token_strings)
        token_to_chars = defaultdict(set)
        for ci, ti in enumerate(char_to_token):
            token_to_chars[ti].add(ci)

        out = []
        for m in DigitBoundaryMetrics._OPERATOR_SPAN.finditer(recon_text):
            op_str = m.group()
            op_start = m.start()
            op_end = m.end()

            op_token_indices = set(
                char_to_token[i] for i in range(op_start, op_end) if i < len(char_to_token)
            )
            if not op_token_indices:
                continue

            op_char_set = set(range(op_start, op_end))
            all_token_chars = set()
            for ti in op_token_indices:
                all_token_chars |= token_to_chars[ti]
            isolated = all_token_chars.issubset(op_char_set)

            compound_preserved = None
            if len(op_str) > 1:
                compound_preserved = len(op_token_indices) == 1

            out.append((op_str, isolated, compound_preserved))
        return out

    def test_isolated_single_char(self, inst):
        # "a" "+" "b" => "+" is isolated
        results = self._check_isolation(inst, ["a", "+", "b"])
        assert len(results) == 1
        assert results[0] == ("+", True, None)  # single-char, no compound check

    def test_merged_with_adjacent(self, inst):
        # "a+" is a single token => "+" is NOT isolated (token also covers "a")
        results = self._check_isolation(inst, ["a+", "b"])
        assert len(results) == 1
        assert results[0][0] == "+"
        assert results[0][1] is False  # not isolated

    def test_compound_preserved(self, inst):
        # "<=" as a single token => isolated and compound preserved
        results = self._check_isolation(inst, ["a", "<=", "b"])
        assert len(results) == 1
        assert results[0] == ("<=", True, True)

    def test_compound_split(self, inst):
        # "<=" split as "<" "=" => two tokens, compound NOT preserved
        results = self._check_isolation(inst, ["a", "<", "=", "b"])
        # The regex will match "<=" starting at the "<" position
        # Actually: "a<=" => regex finds "<=" at position 1
        # But "<" is token 1, "=" is token 2 => 2 tokens => compound not preserved
        le_results = [r for r in results if r[0] == "<="]
        assert len(le_results) == 1
        assert le_results[0][2] is False  # compound not preserved

    def test_space_prefixed_token(self, inst):
        # "Ġ+" should clean to "+" and be isolated
        results = self._check_isolation(inst, ["a", "Ġ+", "Ġb"])
        assert len(results) == 1
        assert results[0] == ("+", True, None)

    def test_multiple_operators(self, inst):
        # "a + b * c"
        results = self._check_isolation(inst, ["a", "Ġ+", "Ġb", "Ġ*", "Ġc"])
        ops = [r[0] for r in results]
        assert "+" in ops
        assert "*" in ops
        assert all(r[1] for r in results)  # all isolated

    def test_no_operators(self, inst):
        results = self._check_isolation(inst, ["hello", "Ġworld"])
        assert results == []

    def test_double_star_isolated(self, inst):
        # "**" as single token
        results = self._check_isolation(inst, ["x", "**", "2"])
        star_results = [r for r in results if r[0] == "**"]
        assert len(star_results) == 1
        assert star_results[0] == ("**", True, True)


# ======================================================================
# Magnitude pipeline (fertility per digit)
# ======================================================================

class TestMagnitudePipeline:
    """Fertility-per-digit computation for magnitude consistency."""

    @pytest.fixture()
    def inst(self):
        return _make_instance()

    def test_single_token_number_fertility(self, inst):
        # "2024" as a single token => 1 token / 4 digits = 0.25
        recon, c2t = inst._build_char_to_token_map(["2024"])
        spans = DigitBoundaryMetrics._find_number_spans(recon)
        assert len(spans) == 1
        start, end, digit_str = spans[0]
        token_indices = set(c2t[i] for i in range(start, end))
        fertility = len(token_indices) / len(digit_str)
        assert fertility == pytest.approx(0.25)

    def test_split_number_fertility(self, inst):
        # "2024" split as "20" "24" => 2 tokens / 4 digits = 0.5
        recon, c2t = inst._build_char_to_token_map(["20", "24"])
        spans = DigitBoundaryMetrics._find_number_spans(recon)
        assert len(spans) == 1
        start, end, digit_str = spans[0]
        token_indices = set(c2t[i] for i in range(start, end))
        fertility = len(token_indices) / len(digit_str)
        assert fertility == pytest.approx(0.5)

    def test_per_digit_separate_fertility(self, inst):
        # "1234" with each digit a separate token => 4 / 4 = 1.0
        recon, c2t = inst._build_char_to_token_map(["1", "2", "3", "4"])
        spans = DigitBoundaryMetrics._find_number_spans(recon)
        assert len(spans) == 1
        start, end, digit_str = spans[0]
        token_indices = set(c2t[i] for i in range(start, end))
        fertility = len(token_indices) / len(digit_str)
        assert fertility == pytest.approx(1.0)

    @pytest.mark.parametrize("token_strings,expected_tokens", [
        (["1234567"], 1),        # single token
        (["1", "234", "567"], 3),  # right-aligned split
        (["12", "34", "56", "7"], 4),  # every-2 split + remainder
        (["1", "2", "3", "4"], 4),  # every digit separate
    ])
    def test_boundary_count_equals_unique_tokens(self, inst, token_strings, expected_tokens):
        """Verify len(set(c2t[i]...)) == len(boundaries) + 1.

        This equivalence is why the separate magnitude_acc accumulator
        was removed: fertility can be derived from boundary data already
        computed for alignment.
        """
        recon, c2t = inst._build_char_to_token_map(token_strings)
        spans = DigitBoundaryMetrics._find_number_spans(recon)
        assert len(spans) == 1
        start, end, digit_str = spans[0]

        # Method 1: unique token indices (old magnitude_acc approach)
        unique_tokens = len(set(c2t[i] for i in range(start, end)))

        # Method 2: boundary count + 1 (new approach via alignment data)
        boundaries = DigitBoundaryMetrics._get_digit_span_boundaries(c2t, start, end)
        tokens_from_boundaries = len(boundaries) + 1

        assert unique_tokens == expected_tokens
        assert tokens_from_boundaries == expected_tokens
        assert unique_tokens == tokens_from_boundaries


from .conftest import MockTokenizer as _MockTokenizer, MockProvider as _MockProvider


# ======================================================================
# TestGoodVsBadTokenizer — end-to-end compute() demonstration
# ======================================================================

class TestGoodVsBadTokenizer:
    """Axis-specific dummy tokenizers demonstrating metric independence.

    Four data sets exercise the four metric axes independently:

    - ``_GOOD_DATA``: perfect on all axes (12 numbers, 4 operators).
    - ``_BAD_BOUNDARY_DATA``: wrong digit splits, perfect operators (6 numbers).
    - ``_BAD_ISOLATION_DATA``: perfect digits, operators merged with neighbours.
    - ``_BAD_COMPOUND_DATA``: perfect digits, single-char ops isolated but
      compound ops (``<=``, ``**``) split across tokens.

    Operator-only texts use digit-free strings (``"a + b = c"`` instead of
    ``"3 + 5 = 8"``) so that bad digit tokenization cannot contaminate
    operator metrics and vice versa.
    """

    # Eight texts exercising all four metric axes.
    _GOOD_DATA = [
        ("result is 42",      ["result", "Ġis", "Ġ42"]),
        ("year 2024",         ["year", "Ġ2", "024"]),
        ("count 1234567",     ["count", "Ġ1", "234", "567"]),
        ("total 12345",       ["total", "Ġ12", "345"]),
        ("3 + 5 = 8",         ["3", "Ġ+", "Ġ5", "Ġ=", "Ġ8"]),
        ("x <= 100",          ["x", "Ġ<=", "Ġ100"]),
        ("2 ** 8",            ["2", "Ġ**", "Ġ8"]),
        ("from 5678 to 9012", ["from", "Ġ5", "678", "Ġto", "Ġ9", "012"]),
    ]

    # Bad digit splits, good operators.  Digit-free operator texts prevent
    # cross-contamination.  6 numbers, all F1=0.
    _BAD_BOUNDARY_DATA = [
        ("result is 42",      ["result", "Ġis", "Ġ4", "2"]),
        ("year 2024",         ["year", "Ġ20", "24"]),
        ("count 1234567",     ["count", "Ġ12", "345", "67"]),
        ("total 12345",       ["total", "Ġ123", "45"]),
        ("a + b = c",         ["a", "Ġ+", "Ġb", "Ġ=", "Ġc"]),
        ("x <= y",            ["x", "Ġ<=", "Ġy"]),
        ("x ** y",            ["x", "Ġ**", "Ġy"]),
        ("from 5678 to 9012", ["from", "Ġ56", "78", "Ġto", "Ġ901", "2"]),
    ]

    # Perfect digit boundaries, operators merged with adjacent characters.
    # 6 numbers, all F1=1.0.  Isolation and compound rates both 0.
    _BAD_ISOLATION_DATA = [
        ("result is 42",      ["result", "Ġis", "Ġ42"]),
        ("year 2024",         ["year", "Ġ2", "024"]),
        ("count 1234567",     ["count", "Ġ1", "234", "567"]),
        ("total 12345",       ["total", "Ġ12", "345"]),
        ("a + b = c",         ["a+", "b=", "c"]),
        ("x <= y",            ["x<", "=y"]),
        ("x ** y",            ["x*", "*y"]),
        ("from 5678 to 9012", ["from", "Ġ5", "678", "Ġto", "Ġ9", "012"]),
    ]

    # Perfect digit boundaries, single-char operators isolated, but compound
    # operators (<= , **) split into individual characters.
    # 6 numbers, all F1=1.0.  Isolation=1.0, compound=0.0.
    _BAD_COMPOUND_DATA = [
        ("result is 42",      ["result", "Ġis", "Ġ42"]),
        ("year 2024",         ["year", "Ġ2", "024"]),
        ("count 1234567",     ["count", "Ġ1", "234", "567"]),
        ("total 12345",       ["total", "Ġ12", "345"]),
        ("a + b = c",         ["a", "Ġ+", "Ġb", "Ġ=", "Ġc"]),
        ("x <= y",            ["x", "Ġ<", "=", "Ġy"]),
        ("x ** y",            ["x", "Ġ*", "*", "Ġy"]),
        ("from 5678 to 9012", ["from", "Ġ5", "678", "Ġto", "Ġ9", "012"]),
    ]

    @staticmethod
    def _build(tok_name, samples):
        """Construct a DigitBoundaryMetrics instance and tokenized_data dict."""
        token_to_id = {}
        next_id = 0
        for _text, toks in samples:
            for t in toks:
                if t not in token_to_id:
                    token_to_id[t] = next_id
                    next_id += 1

        id_to_token = {v: k for k, v in token_to_id.items()}
        provider = _MockProvider(tok_name, _MockTokenizer(id_to_token))
        metrics = DigitBoundaryMetrics(provider)

        data_list = [
            TokenizedData(
                tokenizer_name=tok_name,
                language="en",
                tokens=[token_to_id[t] for t in toks],
                text=text,
            )
            for text, toks in samples
        ]
        return metrics, {tok_name: data_list}

    @pytest.fixture()
    def good_results(self):
        m, td = self._build("good_tok", self._GOOD_DATA)
        return m.compute(td)

    @pytest.fixture()
    def bad_boundary_results(self):
        m, td = self._build("bad_bnd", self._BAD_BOUNDARY_DATA)
        return m.compute(td)

    @pytest.fixture()
    def bad_isolation_results(self):
        m, td = self._build("bad_iso", self._BAD_ISOLATION_DATA)
        return m.compute(td)

    @pytest.fixture()
    def bad_compound_results(self):
        m, td = self._build("bad_cmp", self._BAD_COMPOUND_DATA)
        return m.compute(td)

    # -- Three-Digit Boundary Alignment --

    def test_good_alignment_perfect_f1(self, good_results):
        """Good tokenizer: all 12 numbers get F1=1.0."""
        summary = good_results["three_digit_boundary_alignment"]["summary"]["good_tok"]
        assert summary["avg_f1"] == pytest.approx(1.0)

    def test_bad_boundary_alignment_all_zero_f1(self, bad_boundary_results):
        """Bad-boundary tokenizer: all 6 multi-digit numbers get F1=0.0."""
        summary = bad_boundary_results["three_digit_boundary_alignment"]["summary"]["bad_bnd"]
        assert summary["avg_f1"] == pytest.approx(0.0)

    def test_bad_operator_does_not_affect_alignment(
        self, bad_isolation_results, bad_compound_results,
    ):
        """Merging/splitting operators must not change digit alignment scores."""
        iso = bad_isolation_results["three_digit_boundary_alignment"]["summary"]["bad_iso"]
        cmp = bad_compound_results["three_digit_boundary_alignment"]["summary"]["bad_cmp"]
        assert iso["avg_f1"] == pytest.approx(1.0)
        assert cmp["avg_f1"] == pytest.approx(1.0)

    # -- Cross-Number Boundary Entropy --

    def test_good_consistent_patterns_zero_entropy(self, good_results):
        """Good tokenizer: three 4-digit numbers all share pattern (1,)."""
        by_dl = good_results["cross_number_boundary_entropy"][
            "per_tokenizer"]["good_tok"]["by_digit_length"]
        assert by_dl["4"]["en"]["entropy"] == pytest.approx(0.0)

    def test_bad_boundary_inconsistent_patterns_nonzero_entropy(
        self, bad_boundary_results,
    ):
        """Bad-boundary tokenizer: 4-digit patterns (2,), (2,), (3,) diverge."""
        by_dl = bad_boundary_results["cross_number_boundary_entropy"][
            "per_tokenizer"]["bad_bnd"]["by_digit_length"]
        assert by_dl["4"]["en"]["entropy"] > 0.0

    # -- Operator Isolation --

    def test_good_operator_isolation_perfect(self, good_results):
        """Good tokenizer: all operators isolated and compounds preserved."""
        summary = good_results["operator_isolation_rate"]["summary"]["good_tok"]
        assert summary["overall_isolation_rate"] == pytest.approx(1.0)
        assert summary["overall_compound_preservation_rate"] == pytest.approx(1.0)

    def test_bad_boundary_operator_still_perfect(self, bad_boundary_results):
        """Bad digit splits must not affect operator isolation (independence)."""
        summary = bad_boundary_results["operator_isolation_rate"]["summary"]["bad_bnd"]
        assert summary["overall_isolation_rate"] == pytest.approx(1.0)
        assert summary["overall_compound_preservation_rate"] == pytest.approx(1.0)

    def test_bad_isolation_operator_rates_zero(self, bad_isolation_results):
        """Operators merged with neighbours: both rates drop to 0."""
        summary = bad_isolation_results["operator_isolation_rate"]["summary"]["bad_iso"]
        assert summary["overall_isolation_rate"] == pytest.approx(0.0)
        assert summary["overall_compound_preservation_rate"] == pytest.approx(0.0)

    def test_bad_compound_isolation_perfect_compound_zero(
        self, bad_compound_results,
    ):
        """Single-char ops isolated but compounds split: isolation=1, compound=0."""
        summary = bad_compound_results["operator_isolation_rate"]["summary"]["bad_cmp"]
        assert summary["overall_isolation_rate"] == pytest.approx(1.0)
        assert summary["overall_compound_preservation_rate"] == pytest.approx(0.0)

    # -- Magnitude Consistency --

    def test_magnitude_results_present(
        self, good_results, bad_boundary_results,
        bad_isolation_results, bad_compound_results,
    ):
        """All four variants produce the expected number of analysed numbers."""
        for results, tok, expected in [
            (good_results, "good_tok", 12),
            (bad_boundary_results, "bad_bnd", 6),
            (bad_isolation_results, "bad_iso", 6),
            (bad_compound_results, "bad_cmp", 6),
        ]:
            summary = results["numeric_magnitude_consistency"]["summary"][tok]
            assert summary["numbers_analyzed"] == expected
            assert summary["avg_fertility"] > 0.0

    def test_magnitude_counts_match_alignment(
        self, good_results, bad_boundary_results,
    ):
        """Alignment and magnitude must report the same number counts."""
        for results, tok in [
            (good_results, "good_tok"),
            (bad_boundary_results, "bad_bnd"),
        ]:
            align_n = results["three_digit_boundary_alignment"]["summary"][tok]["numbers_analyzed"]
            mag_n = results["numeric_magnitude_consistency"]["summary"][tok]["numbers_analyzed"]
            assert align_n == mag_n

    def test_ten_plus_bucket(self):
        """An 11-digit number lands in the '10+' bucket with F1=1.0."""
        data = [
            ("big 12345678901", ["big", "Ġ12", "345", "678", "901"]),
        ]
        m, td = self._build("tenplus", data)
        results = m.compute(td)
        by_dl = results["three_digit_boundary_alignment"][
            "per_tokenizer"]["tenplus"]["by_digit_length"]
        assert "10+" in by_dl
        assert by_dl["10+"]["en"]["mean_f1"] == pytest.approx(1.0)
