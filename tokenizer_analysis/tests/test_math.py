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
    inst._char_decode_table = None
    inst._special_tokens = None
    inst._special_token_cache = {}
    inst._subword_markers = None
    inst._subword_marker_cache = {}
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
# _score_boundaries: vacuous cases
# ======================================================================

class TestScoreBoundariesVacuous:
    """The four vacuous-case rows from the docstring table."""

    def test_both_empty(self):
        # Short number, single token: perfect.
        result = DigitBoundaryMetrics._score_boundaries(set(), set())
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_actual_nonempty_ideal_empty(self):
        # Short number needlessly split: all boundaries spurious.
        result = DigitBoundaryMetrics._score_boundaries({1}, set())
        assert result == {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    def test_actual_nonempty_ideal_empty_multiple(self):
        result = DigitBoundaryMetrics._score_boundaries({1, 2}, set())
        assert result == {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    def test_actual_empty_ideal_nonempty(self):
        # Long number kept as single token: no wrong boundaries but ideal missed.
        result = DigitBoundaryMetrics._score_boundaries(set(), {1, 4})
        assert result == {"precision": 1.0, "recall": 0.0, "f1": 0.0}


# ======================================================================
# _score_boundaries: normal cases
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
    """Subword marker stripping.

    A marker is stripped only when inst._subword_markers says the tokenizer
    being processed actually uses it (BaseMetrics._detect_subword_markers).
    The marker-stripping tests below set that attribute explicitly, the same
    way TestBuildCharToTokenMap.test_skips_special_tokens sets
    inst._special_tokens: a bare _make_instance() never resolved a real
    tokenizer, so without this the marker set defaults to empty and nothing
    is stripped.
    """

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
        inst._subword_markers = {"##"}
        assert inst._clean_token("##bar") == "bar"

    def test_bpe_end_of_word(self, inst):
        inst._subword_markers = {"</w>"}
        assert inst._clean_token("baz</w>") == "baz"

    def test_bpe_continuation_suffix(self, inst):
        inst._subword_markers = {"@@"}
        assert inst._clean_token("qux@@") == "qux"

    def test_special_token_angle(self, inst):
        assert inst._clean_token("<|endoftext|>") is None

    def test_special_token_bracket(self, inst):
        assert inst._clean_token("[CLS]") is None

    def test_plain_token(self, inst):
        assert inst._clean_token("hello") == "hello"

    def test_digit_token(self, inst):
        assert inst._clean_token("1234") == "1234"

    # Marker gating: the actual defect (unconditional stripping)

    def test_continuation_not_stripped_when_marker_unresolved(self, inst):
        """A token matching '##...' is left alone when no tokenizer has been
        shown to use the WordPiece prefix (inst._subword_markers is None,
        _make_instance()'s default). This is the reported defect: '###' (a
        markdown heading in a byte-level BPE vocabulary) must not become '#'."""
        assert inst._clean_token("###") == "###"

    def test_continuation_not_stripped_when_marker_absent_from_set(self, inst):
        """A tokenizer resolved to use '</w>' does not also strip '##': the
        two markers are independent, gated separately."""
        inst._subword_markers = {"</w>"}
        assert inst._clean_token("##bar") == "##bar"

    def test_end_word_not_stripped_when_marker_unresolved(self, inst):
        assert inst._clean_token("baz</w>") == "baz</w>"

    def test_continuation_suffix_not_stripped_when_marker_unresolved(self, inst):
        assert inst._clean_token("@@") == "@@"

    def test_continuation_suffix_not_stripped_when_marker_empty_set(self, inst):
        """An explicitly empty marker set (a tokenizer resolved to use none of
        the three) behaves the same as the unresolved None default."""
        inst._subword_markers = set()
        assert inst._clean_token("qux@@") == "qux@@"


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
        # Skipping is driven by what the tokenizer declares, so the declaration
        # has to be set. This test used to leave it implicit and pass on a
        # surface pattern that matched any <|...|> or [...] token.
        inst._special_tokens = {"<|start|>", "<|end|>"}
        text, mapping = inst._build_char_to_token_map(
            ["<|start|>", "abc", "<|end|>"]
        )
        assert text == "abc"
        assert mapping == [1, 1, 1]

    def test_keeps_undeclared_bracket_tokens(self, inst):
        """Bracket-form tokens the tokenizer does not declare are content.

        '[...]' is a real token in the bundled tokenizers/bpe.json vocabulary and
        '[]' is one in apertus and llama3. The surface pattern that used to decide
        this deleted them from the reconstruction.
        """
        inst._special_tokens = {"<|start|>"}
        text, mapping = inst._build_char_to_token_map(["a", "[...]", "[]"])
        assert text == "a[...][]"
        assert mapping == [0, 1, 1, 1, 1, 1, 2, 2]

    def test_empty_input(self, inst):
        text, mapping = inst._build_char_to_token_map([])
        assert text == ""
        assert mapping == []

    def test_mixed_markers(self, inst):
        inst._subword_markers = {"##"}
        tokens = ["hello", "##123", "Ġ456"]
        text, mapping = inst._build_char_to_token_map(tokens)
        assert text == "hello123456"
        assert mapping == [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]


# ======================================================================
# _detect_subword_markers / _resolve_subword_markers
# ======================================================================

class _FakeModel:
    """Duck-typed stand-in for a tokenizers.models.{BPE,WordPiece} model."""

    def __init__(self, continuing_subword_prefix=None, end_of_word_suffix=None):
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix


class _FakeBackend:
    """Duck-typed stand-in for a raw tokenizers.Tokenizer: exposes .model."""

    def __init__(self, model):
        self.model = model


class _FakeWrapper:
    """Duck-typed stand-in for a TokenizerWrapper wrapping _FakeBackend."""

    def __init__(self, backend):
        self._backend = backend

    def get_underlying_tokenizer(self):
        return self._backend


class _BehavioralOnlyTokenizer:
    """No .model at all (like a raw tiktoken.Encoding or a stub): the
    declared channel must find nothing, forcing the behavioral probe."""

    def __init__(self, piece_map):
        """piece_map: {probe_text: [piece_strings]} for the encode() calls
        _detect_subword_markers will make."""
        self._piece_map = piece_map

    def encode(self, text):
        return list(range(len(self._piece_map.get(text, []))))

    def convert_ids_to_tokens(self, ids):
        # ids are just range(n); use the length to look up the one registered probe.
        for pieces in self._piece_map.values():
            if len(pieces) == len(ids):
                return pieces
        return []


class TestDetectSubwordMarkers:
    """Declared and behavioral detection channels, checked independently."""

    # Declared channel

    def test_declared_wordpiece_prefix(self):
        backend = _FakeBackend(_FakeModel(continuing_subword_prefix="##"))
        assert DigitBoundaryMetrics._detect_subword_markers(backend) == {"##"}

    def test_declared_bpe_end_of_word_suffix(self):
        backend = _FakeBackend(_FakeModel(end_of_word_suffix="</w>"))
        assert DigitBoundaryMetrics._detect_subword_markers(backend) == {"</w>"}

    def test_declared_both_markers(self):
        backend = _FakeBackend(_FakeModel(continuing_subword_prefix="##",
                                          end_of_word_suffix="</w>"))
        assert DigitBoundaryMetrics._detect_subword_markers(backend) == {"##", "</w>"}

    def test_declared_empty_fields_strip_nothing(self):
        """The bpe.json / cl100k_base / o200k_base case: both fields declared
        but empty ('' or None), same as the real tokenizers.models.BPE default."""
        backend = _FakeBackend(_FakeModel(continuing_subword_prefix="",
                                          end_of_word_suffix=""))
        assert DigitBoundaryMetrics._detect_subword_markers(backend) == set()

    def test_declared_wordpiece_binding_gap_defaults_to_hash_hash(self):
        """A model of type WordPiece whose binding leaves
        continuing_subword_prefix unpopulated still defaults to '##': every
        tokenizers-library WordPiece does, even with the field readable, so
        this only covers a binding that does not populate it."""
        class WordPiece:
            continuing_subword_prefix = None
        backend = _FakeBackend(WordPiece())
        assert DigitBoundaryMetrics._detect_subword_markers(backend) == {"##"}

    def test_declared_non_standard_prefix_not_guessed(self):
        """A declared but non-'##' prefix is not translated into a strip
        rule: _process_token only knows the three canonical forms."""
        backend = _FakeBackend(_FakeModel(continuing_subword_prefix="**"))
        assert DigitBoundaryMetrics._detect_subword_markers(backend) == set()

    def test_declared_unwraps_wrapper(self):
        """A wrapper exposing get_underlying_tokenizer() is unwrapped to reach
        the backend's .model, the same path _has_bytelevel_component uses."""
        backend = _FakeBackend(_FakeModel(continuing_subword_prefix="##"))
        wrapper = _FakeWrapper(backend)
        assert DigitBoundaryMetrics._detect_subword_markers(wrapper) == {"##"}

    def test_no_model_falls_through_to_behavioral(self):
        """No .model at all and no .encode(): empty set, no crash."""
        assert DigitBoundaryMetrics._detect_subword_markers(object()) == set()

    # Behavioral channel (declared channel silent: no .model)

    def test_behavioral_wordpiece_prefix(self):
        probe = "supercalifragilisticexpialidocious"
        tok = _BehavioralOnlyTokenizer({probe: ["super", "##cal", "##ious"]})
        assert DigitBoundaryMetrics._detect_subword_markers(tok) == {"##"}

    def test_behavioral_end_of_word_suffix(self):
        probe = "supercalifragilisticexpialidocious"
        tok = _BehavioralOnlyTokenizer({probe: ["su", "per", "cali", "ous</w>"]})
        assert DigitBoundaryMetrics._detect_subword_markers(tok) == {"</w>"}

    def test_behavioral_continuation_suffix(self):
        probe = "supercalifragilisticexpialidocious"
        tok = _BehavioralOnlyTokenizer({probe: ["su@@", "per@@", "cious"]})
        assert DigitBoundaryMetrics._detect_subword_markers(tok) == {"@@"}

    def test_behavioral_single_piece_inconclusive(self):
        """A probe that does not fragment (whole word as one piece) gives no
        evidence either way, so the marker set stays empty."""
        probe = "supercalifragilisticexpialidocious"
        tok = _BehavioralOnlyTokenizer({probe: ["supercalifragilisticexpialidocious"]})
        assert DigitBoundaryMetrics._detect_subword_markers(tok) == set()

    def test_behavioral_no_markers_in_pieces(self):
        probe = "supercalifragilisticexpialidocious"
        tok = _BehavioralOnlyTokenizer({probe: ["su", "per", "cal", "ious"]})
        assert DigitBoundaryMetrics._detect_subword_markers(tok) == set()

    # Memoization (mirrors _resolve_special_tokens's identity-cache test)

    def test_resolve_memoizes_per_tokenizer_object(self):
        inst = _make_instance()
        backend = _FakeBackend(_FakeModel(continuing_subword_prefix="##"))
        first = inst._resolve_subword_markers(backend)
        # Mutate the backend after the first call; a cached second call must
        # still return the first result, proving the cache (not a fresh
        # detection) answered.
        backend.model.continuing_subword_prefix = None
        second = inst._resolve_subword_markers(backend)
        assert first == second == {"##"}


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
        assert "normalized_entropy" not in result
        assert result["num_patterns"] == 0
        assert result["count"] == 0

    def test_single_pattern_repeated(self):
        # All identical -> H = 0
        result = DigitBoundaryMetrics._compute_pattern_entropy([(1,)] * 10)
        assert result["entropy"] == 0.0
        assert "normalized_entropy" not in result
        assert result["num_patterns"] == 1
        assert result["dominant_pattern"] == (1,)
        assert result["dominant_pattern_freq"] == 1.0
        assert result["count"] == 10

    def test_two_patterns_equal_frequency(self):
        # 50/50 split of 2 patterns -> H = 1.0 bit
        pats = [(1,)] * 5 + [(2,)] * 5
        result = DigitBoundaryMetrics._compute_pattern_entropy(pats)
        assert result["entropy"] == pytest.approx(1.0)
        assert "normalized_entropy" not in result
        assert result["num_patterns"] == 2
        assert result["count"] == 10

    def test_three_patterns_uniform(self):
        # Uniform distribution over 3 patterns -> H = log2(3) ≈ 1.585 bits
        pats = [(1,)] * 4 + [(2,)] * 4 + [(3,)] * 4
        result = DigitBoundaryMetrics._compute_pattern_entropy(pats)
        assert result["entropy"] == pytest.approx(math.log2(3))
        assert "normalized_entropy" not in result
        assert result["num_patterns"] == 3
        assert result["count"] == 12

    def test_skewed_distribution(self):
        # 80/20 split -> H = -(0.8*log2(0.8) + 0.2*log2(0.2))
        pats = [(1,)] * 8 + [(2,)] * 2
        result = DigitBoundaryMetrics._compute_pattern_entropy(pats)
        expected_h = -(0.8 * math.log2(0.8) + 0.2 * math.log2(0.2))
        assert result["entropy"] == pytest.approx(expected_h)
        assert "normalized_entropy" not in result
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

    def test_raw_entropy_not_normalized(self):
        # Verify entropy is raw Shannon entropy (bits), not normalized.
        # 4 equiprobable patterns -> H = log2(4) = 2.0 bits (not 1.0).
        pats = [(1,)] * 3 + [(2,)] * 3 + [(3,)] * 3 + [(4,)] * 3
        result = DigitBoundaryMetrics._compute_pattern_entropy(pats)
        assert result["entropy"] == pytest.approx(2.0)
        assert "normalized_entropy" not in result


# ======================================================================
# _score_boundaries + _ideal_boundaries end-to-end
#
# These follow the worked examples from the class docstring verbatim.
# ======================================================================

class TestDocstringWorkedExamples:
    """Verify every worked example from the class docstring."""

    # "1234567" (L=7), ideal = {1, 4}

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

    # "42" (L=2), ideal = {}

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
        inst._subword_markers = {"##"}
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
    """Scaling statistics for numeric magnitude consistency.

    ``_compute_fertility_scaling`` takes per-number records
    (``{"fertility_per_digit": ..., "num_digits": ...}``), not bare fertility
    floats, so that the linear fit can use each bucket's true mean digit
    length instead of a fixed stand-in for the open-ended '10+' bucket.
    """

    @staticmethod
    def _records(fertility_and_digits):
        """Build [{"fertility_per_digit": f, "num_digits": d}, ...] from (f, d) pairs."""
        return [
            {"fertility_per_digit": f, "num_digits": d}
            for f, d in fertility_and_digits
        ]

    def test_empty_input(self):
        """Nothing to measure, so every scaling field is null.

        cv_of_mean_fertility used to be 0.0 here while its three siblings in
        the same branch were None, which said the buckets have no dispersion
        when there were no buckets.
        """
        result = DigitBoundaryMetrics._compute_fertility_scaling({})
        assert result["per_bucket"] == {}
        assert result["spearman_rho"] is None
        assert result["cv_of_mean_fertility"] is None
        assert result["linear_fit"] is None

    def test_single_bucket(self):
        result = DigitBoundaryMetrics._compute_fertility_scaling(
            {"4": self._records([(0.5, 4), (0.5, 4), (0.5, 4)])}
        )
        assert "4" in result["per_bucket"]
        assert result["per_bucket"]["4"]["mean_fertility"] == pytest.approx(0.5)
        assert result["per_bucket"]["4"]["count"] == 3
        assert result["per_bucket"]["4"]["mean_digit_length"] == pytest.approx(4.0)
        # Only one bucket => no correlation possible
        assert result["spearman_rho"] is None
        assert result["linear_fit"] is None

    def test_two_buckets(self):
        result = DigitBoundaryMetrics._compute_fertility_scaling(
            {
                "1": self._records([(1.0, 1), (1.0, 1)]),
                "4": self._records([(0.5, 4), (0.5, 4)]),
            }
        )
        assert result["spearman_rho"] is not None
        assert result["linear_fit"] is not None
        assert "slope" in result["linear_fit"]
        assert "r_squared" in result["linear_fit"]

    def test_constant_fertility_zero_cv(self):
        # All buckets have identical mean fertility => CV = 0
        result = DigitBoundaryMetrics._compute_fertility_scaling(
            {
                "1": self._records([(1.0, 1)]),
                "2": self._records([(1.0, 2)]),
                "3": self._records([(1.0, 3)]),
                "4": self._records([(1.0, 4)]),
            }
        )
        assert result["cv_of_mean_fertility"] == pytest.approx(0.0)

    def test_increasing_fertility_high_rho(self):
        # Fertility increases with digit length => positive rho
        result = DigitBoundaryMetrics._compute_fertility_scaling(
            {
                "1": self._records([(0.2, 1)]),
                "2": self._records([(0.4, 2)]),
                "3": self._records([(0.6, 3)]),
                "4": self._records([(0.8, 4)]),
            }
        )
        assert result["spearman_rho"] is not None
        assert result["spearman_rho"] > 0.9

    def test_ten_plus_bucket_spearman_uses_pinned_10(self):
        """spearman_rho keeps the pre-fix bucket-order representation.

        This test previously asserted that the '10+' bucket "is treated as
        digit length 10 for scaling" and checked only spearman_rho. That
        claim was the defect: the same pinned-at-10 value was also used as
        the x-coordinate of the linear fit, which is wrong when the '10+'
        bucket's numbers are not all exactly 10 digits long (see
        test_linear_fit_uses_true_mean_digit_length_for_open_bucket below).
        spearman_rho itself is a rank correlation over bucket order, not a
        fit to a specific x value, so pinning '10+' at 10 for that one
        statistic was left unchanged deliberately; this test now says so.
        """
        result = DigitBoundaryMetrics._compute_fertility_scaling(
            {
                "1": self._records([(1.0, 1)]),
                "10+": self._records([(0.5, 12)]),
            }
        )
        assert result["spearman_rho"] is not None
        # Fertility decreases with length => negative rho
        assert result["spearman_rho"] < 0

    def test_linear_fit_uses_true_mean_digit_length_for_open_bucket(self):
        """Regression test for the '10+' bucket linear-fit defect.

        Every number here is generated from the exact line
        num_tokens = 0.5 * num_digits + 1.0, including two numbers of
        different lengths (12 and 20 digits) inside the open '10+' bucket.
        Because the generating relationship is linear, the true mean token
        count of the '10+' bucket equals the line evaluated at its true mean
        digit length (16.0), so a fit that uses each bucket's own mean
        digit length and mean token count recovers the exact line over all
        five buckets: slope 0.5, intercept 1.0, R-squared 1.0.

        The pre-fix implementation instead pinned the '10+' bucket's x-value
        at a fixed 10.0 and its y-value at mean_fertility * 10.0 (here
        0.566667 * 10 = 5.666667, against the true 9.0 at digit length 16),
        which does not sit on this line and pulls the fit off it.
        """
        bucket_records = {
            "2": self._records([(1.0, 2)]),                       # 2 digits -> 2.0 tokens
            "4": self._records([(0.75, 4)]),                      # 4 digits -> 3.0 tokens
            "6": self._records([(4 / 6, 6)]),                     # 6 digits -> 4.0 tokens
            "8": self._records([(0.625, 8)]),                     # 8 digits -> 5.0 tokens
            "10+": self._records([(7 / 12, 12), (11 / 20, 20)]),  # 7.0 and 11.0 tokens
        }
        result = DigitBoundaryMetrics._compute_fertility_scaling(bucket_records)

        bucket_10p = result["per_bucket"]["10+"]
        assert bucket_10p["mean_digit_length"] == pytest.approx(16.0)

        fit = result["linear_fit"]
        assert fit["slope"] == pytest.approx(0.5, abs=1e-9)
        assert fit["intercept"] == pytest.approx(1.0, abs=1e-9)
        assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)


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
# TestGoodVsBadTokenizer: end-to-end compute() demonstration
# ======================================================================

def _offsets_for(text: str, token_strings) -> list:
    """Character spans for hand-written token strings, for the synthetic fixtures.

    Real wrappers get offsets from the encoder. Here they are derived by walking
    each token's surface through the source one character at a time, skipping
    source whitespace that the surface omits. That matters because several
    fixtures deliberately model a tokenizer that merges an operator with its
    operand across a space: the token "a+" against the source "a + b" covers
    characters 0 to 2, so the token carries the operand "a" and the operator is
    correctly counted as not isolated. Concatenating surfaces instead would put
    the "+" at index 1, inside a space, and the operator would look isolated.

    Operator isolation resolves operators to tokens through offsets, so a
    fixture without them is skipped rather than measured.
    """
    spans = []
    pos = 0
    for tok in token_strings:
        surface = (
            tok.replace("\u0120", " ")
               .replace("\u2581", " ")
               .replace("\u010a", "\n")
        )
        if not surface:
            spans.append((pos, pos))
            continue
        first = None
        i = pos
        for ch in surface:
            if ch.isspace():
                # A space in the surface matches a space in the source if one is
                # there, otherwise it contributed nothing.
                if i < len(text) and text[i].isspace():
                    if first is None:
                        first = i
                    i += 1
                continue
            while i < len(text) and text[i] != ch and text[i].isspace():
                i += 1
            if i < len(text) and text[i] == ch:
                if first is None:
                    first = i
                i += 1
        if first is None:
            spans.append((pos, pos))
        else:
            spans.append((first, i))
            pos = i
    return spans


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
                offsets=_offsets_for(text, toks),
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

    # Three-Digit Boundary Alignment

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

    # Digit Split Variability

    def test_good_consistent_patterns_zero_entropy(self, good_results):
        """Good tokenizer: three 4-digit numbers all share pattern (1,)."""
        by_dl = good_results["digit_split_variability"][
            "per_tokenizer"]["good_tok"]["by_digit_length"]
        assert by_dl["4"]["en"]["entropy"] == pytest.approx(0.0)

    def test_bad_boundary_inconsistent_patterns_nonzero_entropy(
        self, bad_boundary_results,
    ):
        """Bad-boundary tokenizer: 4-digit patterns (2,), (2,), (3,) diverge."""
        by_dl = bad_boundary_results["digit_split_variability"][
            "per_tokenizer"]["bad_bnd"]["by_digit_length"]
        assert by_dl["4"]["en"]["entropy"] > 0.0

    # Operator Isolation

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

    # Magnitude Consistency

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


# ======================================================================
# T1: Regression test for C1 bug: adjacent numbers must not merge
# ======================================================================

class TestAdjacentNumbersNotMerged:
    """Verify that numbers separated by whitespace are treated as separate spans.

    Before the C1 fix, _find_number_spans was called on the whitespace-stripped
    reconstructed text, which merged "1234 5678" into "12345678".
    """

    @staticmethod
    def _build(tok_name, samples):
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
                offsets=_offsets_for(text, toks),
            )
            for text, toks in samples
        ]
        return metrics, {tok_name: data_list}

    def test_adjacent_four_digit_numbers(self):
        """'1234 5678' must produce two 4-digit spans, not one 8-digit span."""
        data = [
            # Two 4-digit numbers separated by space, each a single token
            ("1234 5678", ["1234", "Ġ5678"]),
        ]
        m, td = self._build("adj_tok", data)
        results = m.compute(td)
        by_dl = results["three_digit_boundary_alignment"][
            "per_tokenizer"]["adj_tok"]["by_digit_length"]
        # Both numbers are 4 digits -> bucket "4"
        assert "4" in by_dl
        assert by_dl["4"]["en"]["count"] == 2
        # No 8-digit bucket should exist
        assert "8" not in by_dl

    def test_three_adjacent_numbers(self):
        """'12 345 6789' must produce three separate spans."""
        data = [
            ("12 345 6789", ["12", "Ġ345", "Ġ6789"]),
        ]
        m, td = self._build("adj3_tok", data)
        results = m.compute(td)
        by_dl = results["three_digit_boundary_alignment"][
            "per_tokenizer"]["adj3_tok"]["by_digit_length"]
        # 2-digit, 3-digit, and 4-digit spans
        assert "2" in by_dl
        assert "3" in by_dl
        assert "4" in by_dl


class TestNumberAfterMultiSpaceToken:
    """A number on an indented line is measured, with the boundaries the offsets report.

    The digit metrics used to map source positions into a text rebuilt from
    cleaned token strings. ``BaseMetrics._process_token`` removes one leading
    space from a token, not all of them, so a token whose surface is three
    spaces (``ĠĠĠ``, which Llama 3, OLMo 2, Qwen 2.5, Mistral NeMo and the
    bundled tokenizers/bpe.json all have) left two spaces in the reconstruction
    that the source has no counterpart for, and
    ``_build_source_to_recon_map`` resynchronized onto one of them.

    On the snippet below the bundled BPE splits ``1234567`` into ``Ġ123``,
    ``45`` and ``67``, whose offsets are (28,32), (32,34) and (34,36), so the
    boundaries are at digit 3 and digit 5. Under the reconstruction path not one
    of the seven digits mapped, so the number was dropped and the snippet's only
    number was reported as zero numbers analyzed.

    Measured on a four-snippet indented corpus, the reconstruction path measured
    1 of 14 numbers for meta-llama/Meta-Llama-3-8B and 1 of 14 for
    Qwen/Qwen2.5-7B, and got both of those wrong: for ``12345678`` it returned
    boundaries [1, 3, 4, 7] under Llama 3 where the offsets give [3, 6], and
    under Qwen 2.5 it returned a boundary at digit 8 of an 8-digit number, which
    only a 9-character span can produce.
    """

    SOURCE = "def totals():\n    subtotal = 1234567\n    return subtotal\n"

    def test_indented_number_measured_with_offset_boundaries(self):
        from tokenizer_analysis.core.tokenizer_wrapper import create_tokenizer_wrapper

        tokenizer = create_tokenizer_wrapper(
            "bundled-bpe", {"class": "huggingface", "path": "tokenizers/bpe.json"}
        )
        token_ids, offsets = tokenizer.encode_with_offsets(self.SOURCE)
        assert offsets is not None, "the bundled BPE reports offsets"
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)
        assert "ĠĠĠ" in token_strings, (
            "this test needs a tokenizer that emits a multi-space token; "
            f"tokenizers/bpe.json produced {token_strings}"
        )

        metrics = DigitBoundaryMetrics(_MockProvider("bundled-bpe", tokenizer))
        results = metrics.compute({
            "bundled-bpe": [
                TokenizedData(
                    tokenizer_name="bundled-bpe",
                    language="code",
                    tokens=list(token_ids),
                    text=self.SOURCE,
                    offsets=offsets,
                )
            ]
        })

        alignment = results["three_digit_boundary_alignment"][
            "per_tokenizer"]["bundled-bpe"]["by_digit_length"]
        assert alignment["7"]["code"]["count"] == 1, (
            "the seven-digit number sits after a multi-space token and has to "
            "be measured, not dropped"
        )
        variability = results["digit_split_variability"][
            "per_tokenizer"]["bundled-bpe"]["by_digit_length"]
        assert variability["7"]["code"]["dominant_pattern"] == (3, 5), (
            "the boundaries are the ones the encoding offsets report for "
            "'Ġ123' + '45' + '67'"
        )


class _CharTokenizer:
    """Char-level tokenizer: every character is its own token.

    Every operator is therefore perfectly isolated, which makes the per-domain
    expectations deterministic.
    """

    def __init__(self):
        self._vocab = {}
        self._rev = {}

    def _id(self, ch):
        if ch not in self._vocab:
            idx = len(self._vocab)
            self._vocab[ch] = idx
            self._rev[idx] = ch
        return self._vocab[ch]

    def encode(self, text):
        return [self._id(c) for c in text]

    def encode_with_offsets(self, text):
        """One token per character, so each span is exactly one character.

        Present because operator isolation resolves operators to tokens through
        offsets; a tokenizer that reports none is skipped rather than measured.
        """
        return [self._id(c) for c in text], [(i, i + 1) for i in range(len(text))]

    def convert_ids_to_tokens(self, ids):
        return [self._rev[i] for i in ids]


class TestOperatorIsolationDomains:
    """Operator isolation is reported separately for prose, code and math."""

    TOK = "dom_tok"
    PROSE = "The total is 12 + 34 = 46 today."

    def _prose_data(self, tokenizer):
        ids, offsets = tokenizer.encode_with_offsets(self.PROSE)
        return {
            self.TOK: [
                TokenizedData(
                    tokenizer_name=self.TOK,
                    language="eng_Latn",
                    tokens=ids,
                    text=self.PROSE,
                    offsets=offsets,
                )
            ]
        }

    def test_reports_prose_code_and_math_each_with_its_corpus(self):
        """All three domains appear, each recording the corpus it used."""
        tok = _CharTokenizer()
        metrics = DigitBoundaryMetrics(_MockProvider(self.TOK, tok))
        ops = metrics.compute(self._prose_data(tok))["operator_isolation_rate"]

        assert set(ops["by_domain"]) == {"prose", "code", "math"}
        for domain in ("prose", "code", "math"):
            assert ops["by_domain"][domain]["source"], f"{domain} records no corpus"
            # a char-level tokenizer isolates every operator
            assert ops["by_domain"][domain]["summary"][self.TOK][
                "overall_isolation_rate"] == pytest.approx(1.0)
        # code and math fall back to the bundled samples, and say so
        assert ops["by_domain"]["code"]["source"].endswith("code_samples.json")
        assert ops["by_domain"]["math"]["source"].endswith("math_samples.json")

    def test_pooled_summary_is_the_sum_of_the_domains(self):
        """The top-level summary pools prose+code+math rather than prose only."""
        tok = _CharTokenizer()
        metrics = DigitBoundaryMetrics(_MockProvider(self.TOK, tok))
        ops = metrics.compute(self._prose_data(tok))["operator_isolation_rate"]

        pooled = ops["summary"][self.TOK]["total_operators"]
        per_domain = sum(
            ops["by_domain"][d]["summary"][self.TOK]["total_operators"]
            for d in ("prose", "code", "math")
        )
        assert pooled == per_domain
        # the code/math samples contribute operators, so pooling is not a no-op
        assert pooled > ops["by_domain"]["prose"]["summary"][self.TOK]["total_operators"]

    def test_pretokenized_tokenizer_is_excluded_and_does_not_crash(self):
        """A real pre-tokenized provider yields no operator domains, and must not crash.

        The code and math domains are derived corpora that have to be encoded.
        ``PreTokenizedDataTokenizer`` *defines* ``encode`` and raises from it, so
        the guard has to test ``can_encode()``; a ``hasattr(tok, "encode")``
        check would sail past this and blow up. This test uses the real wrapper
        rather than a mock precisely so that mistake cannot pass.

        Prose is excluded too, which changed in 1.0. Operator isolation resolves
        an operator to its covering tokens through character offsets, and a
        pre-tokenized provider supplies ids and text but no offsets, so the
        correspondence cannot be established. It used to be guessed by matching
        token surfaces against a reconstruction, which is what produced wrong
        isolation rates for any tokenizer whose special tokens the surface
        pattern did not recognise. Declining to measure is the intended
        behaviour; the metric is simply unavailable for this input.
        """
        from tokenizer_analysis.core.tokenizer_wrapper import PreTokenizedDataTokenizer

        tokens = ["1", " ", "+", " ", "2", " ", "=", " ", "3"]
        token_to_id = {t: i for i, t in enumerate(dict.fromkeys(tokens))}
        tok_name = "pretok_only"
        tokenizer = PreTokenizedDataTokenizer(
            tok_name, vocab_size=len(token_to_id), vocab_dict=token_to_id
        )
        assert not tokenizer.can_encode()
        assert callable(getattr(tokenizer, "encode", None))  # it exists, and raises

        metrics = DigitBoundaryMetrics(_MockProvider(tok_name, tokenizer))
        data = {
            tok_name: [
                TokenizedData(
                    tokenizer_name=tok_name,
                    language="eng_Latn",
                    tokens=[token_to_id[t] for t in tokens],
                    text="1 + 2 = 3",
                )
            ]
        }
        ops = metrics.compute(data)["operator_isolation_rate"]

        # No domain can be measured without offsets, and nothing crashes.
        for domain in ("prose", "code", "math"):
            assert tok_name not in ops["by_domain"][domain]["summary"], (
                f"{domain} should be unavailable for a pre-tokenized provider"
            )
        # Absent, not zero: a zero here would read as "no operator was isolated".
        assert tok_name not in ops["summary"]

    def test_pooled_by_language_namespaces_code_and_math(self):
        """Code/math rows must not silently merge into the prose language namespace."""
        tok = _CharTokenizer()
        metrics = DigitBoundaryMetrics(_MockProvider(self.TOK, tok))
        ops = metrics.compute(self._prose_data(tok))["operator_isolation_rate"]

        langs = ops["per_tokenizer"][self.TOK]["by_language"]
        assert "eng_Latn" in langs                       # prose stays a bare FLORES code
        assert any(k.startswith("code:") for k in langs)  # code is marked
        assert "math:math" in langs                       # math is marked
        # no bare code language leaked into the prose namespace
        assert "python" not in langs

    def test_domain_operator_counts_expose_the_pooled_weighting(self):
        """The pooled summary is a micro-average, so its denominators must be visible."""
        tok = _CharTokenizer()
        metrics = DigitBoundaryMetrics(_MockProvider(self.TOK, tok))
        ops = metrics.compute(self._prose_data(tok))["operator_isolation_rate"]

        counts = ops["domain_operator_counts"][self.TOK]
        assert set(counts) == {"prose", "code", "math"}
        assert sum(counts.values()) == ops["summary"][self.TOK]["total_operators"]
        # the bundled code corpus really does dominate the pool
        assert counts["code"] > counts["prose"] + counts["math"]

    def test_each_domain_records_the_corpus_it_measured(self):
        """Provenance: the pooled number is corpus-weighted, so each domain's size is reported."""
        tok = _CharTokenizer()
        code = {"python": ["x = a + b\n"], "javascript": ["const z = p !== q;\n"]}
        metrics = DigitBoundaryMetrics(_MockProvider(self.TOK, tok), code_texts=code)
        ops = metrics.compute(self._prose_data(tok))["operator_isolation_rate"]

        code_corpus = ops["by_domain"]["code"]["corpus"]
        assert code_corpus["n_languages"] == 2
        assert code_corpus["texts_per_language"] == {"javascript": 1, "python": 1}
        assert code_corpus["n_chars"] > 0
        # the caller's dataset was used, not the bundled samples
        assert not ops["by_domain"]["code"]["source"].endswith("code_samples.json")
        for domain in ("prose", "code", "math"):
            assert ops["by_domain"][domain]["corpus"]["n_texts"] > 0

    def test_derived_corpora_are_encoded_once_across_compute_calls(self):
        """compute() runs once per language group; the code/math corpora must not be re-encoded.

        Counts ``encode_with_offsets``, which is what _build_derived_corpora
        actually calls.  This test previously counted ``encode`` and passed for
        the wrong reason: the only caller of ``encode`` on this path was the
        four-probe character-decode-table build inside _set_tokenizer_context,
        whose result no metric read.  Removing that dead call dropped the count
        to zero and exposed the mismatch.
        """
        tok = _CharTokenizer()
        metrics = DigitBoundaryMetrics(_MockProvider(self.TOK, tok))
        # build the prose corpus once: _prose_data() itself encodes, and we only
        # want to count the encodes that compute() does on the derived corpora
        prose = self._prose_data(tok)

        calls = {"n": 0}
        original = tok.encode_with_offsets

        def counting(text):
            calls["n"] += 1
            return original(text)

        tok.encode_with_offsets = counting
        metrics.compute(prose)
        after_first = calls["n"]
        assert after_first > 0, "the code/math corpora should have been encoded once"
        metrics.compute(prose)
        assert calls["n"] == after_first, "derived corpora were re-encoded on the second call"


class TestOperatorOverlapKeepsLaterToken:
    """A character two tokens claim belongs to the later of the two.

    A SentencePiece vocabulary emits the word-start marker as its own token, and
    HuggingFace reports a range for it that covers the first character of the
    following word: xlm-roberta-base encodes '1234567' as '▁', '1234', '567'
    with offsets (0,1), (0,4) and (4,7), so two tokens claim character 0. The
    marker produces no source character, so the character belongs to the content
    token, which is the rule ``ASTBoundaryMetrics._map_from_offsets`` and the
    digit metrics both apply.

    The fixture below has the same shape: the marker's range (0,1) and the token
    'x=' range (0,2) both claim the 'x'. Reading that character as the marker's
    leaves the token covering the operator looking as though it covers nothing
    else, and the text is scored 1.0 isolated instead of 0.0.
    """

    TOK = "marker_tok"
    TEXT = "x=y"
    TOKENS = ["▁", "x=", "y"]
    OFFSETS = [(0, 1), (0, 2), (2, 3)]

    def test_operator_glued_to_its_operand_is_not_isolated(self):
        id_to_token = dict(enumerate(self.TOKENS))
        metrics = DigitBoundaryMetrics(
            _MockProvider(self.TOK, _MockTokenizer(id_to_token))
        )
        results = metrics.compute({
            self.TOK: [
                TokenizedData(
                    tokenizer_name=self.TOK,
                    language="eng_Latn",
                    tokens=list(id_to_token),
                    text=self.TEXT,
                    offsets=list(self.OFFSETS),
                )
            ]
        })

        prose = results["operator_isolation_rate"]["by_domain"]["prose"][
            "summary"][self.TOK]
        assert prose["total_operators"] == 1
        assert prose["overall_isolation_rate"] == pytest.approx(0.0), (
            "the token covering '=' also covers the operand 'x', which is only "
            "visible if the overlapping character is read as the content "
            "token's rather than the word-start marker's"
        )


class TestPerTextOperatorsMatchComputeOperators:
    """``compute_per_text`` and ``compute()`` score one text's operators alike.

    ``compute_per_text`` used to run the operator regex over a text rebuilt by
    concatenating cleaned token strings, while ``compute()`` reads the encoder's
    character offsets, so the two disagreed on the same text. The reconstruction
    drops the space in '! =', and the regex then reads a '!=' there that the
    source does not contain: with tokenizers/bpe.json the per-document path
    reported 3 compound operators against the 1 the corpus path reports, and a
    compound preservation rate over that wrong denominator.
    """

    TOK = "bundled-bpe"
    TEXT = "0! = 1, 5! = 120, and 20 >= 3."

    def test_both_paths_report_the_same_operator_numbers(self):
        from tokenizer_analysis.core.tokenizer_wrapper import create_tokenizer_wrapper

        tokenizer = create_tokenizer_wrapper(
            self.TOK, {"class": "huggingface", "path": "tokenizers/bpe.json"}
        )
        token_ids, offsets = tokenizer.encode_with_offsets(self.TEXT)
        assert offsets is not None, "the bundled BPE reports offsets"

        metrics = DigitBoundaryMetrics(_MockProvider(self.TOK, tokenizer))
        per_text = metrics.compute_per_text(tokenizer, self.TEXT)
        corpus = metrics.compute({
            self.TOK: [
                TokenizedData(
                    tokenizer_name=self.TOK,
                    language="eng_Latn",
                    tokens=list(token_ids),
                    text=self.TEXT,
                    offsets=offsets,
                )
            ]
        })["operator_isolation_rate"]["by_domain"]["prose"]["summary"][self.TOK]

        assert per_text["n_operators"] == corpus["total_operators"]
        assert per_text["operator_isolation_rate"] == pytest.approx(
            corpus["overall_isolation_rate"]
        )
        assert per_text["n_compound_operators"] == corpus["total_compound_operators"]
        assert per_text["compound_operator_preserved_rate"] == pytest.approx(
            corpus["overall_compound_preservation_rate"]
        )
        # The text carries the one compound operator it is written to carry,
        # '>=', so a change that made both paths count the '! =' as a compound
        # operator would not pass by agreeing with itself.
        assert per_text["n_compound_operators"] == 1
