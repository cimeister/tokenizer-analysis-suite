"""Tests for tokenizer_analysis.metrics.basic (BasicTokenizationMetrics)."""

import pytest

from tokenizer_analysis.metrics.basic import BasicTokenizationMetrics
from tokenizer_analysis.core.input_types import TokenizedData
from typing import Dict, List, Optional, Tuple

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

    def test_blank_lines_are_counted_because_their_tokens_are(self):
        """Every line counts, since the numerator is the whole text's tokens.

        These two tests previously asserted that blank lines were dropped from
        the denominator. That made the numerator and the denominator describe
        different text: a blank line contributes a newline token, which stays
        in the token count and cannot be taken out of it, so the reported rate
        came out higher than the text supports.
        """
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        # splitlines() gives 4: 'hello world', '', 'goodbye world', ''. The
        # final newline ends the fourth line rather than starting a fifth.
        text = "hello world\n\ngoodbye world\n\n"
        td = {tok_name: [_make_td(tok_name, text, [1, 2, 3, 4])]}

        results = metrics.compute_avg_tokens_per_line_analysis(td)
        tpl_data = results["avg_tokens_per_line"]["per_tokenizer"][tok_name]
        assert tpl_data["global_avg"] == pytest.approx(1.0)

    def test_a_whitespace_only_text_is_not_measured(self):
        """A text with no content is skipped, so the result is null.

        The loop guard requires text.strip(), so a whitespace-only text
        contributes nothing. Reporting 0.0 said the tokenizer produced no
        tokens per line, which is a measurement that was never taken.
        """
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        text = "\n\n\n"
        td = {tok_name: [_make_td(tok_name, text, [1])]}

        results = metrics.compute_avg_tokens_per_line_analysis(td)
        tpl_data = results["avg_tokens_per_line"]["per_tokenizer"][tok_name]
        assert tpl_data["global_avg"] is None
        assert tpl_data["total_lines"] == 0


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


# ======================================================================
# Mock decodable tokenizer and provider
# ======================================================================

class _MockDecodableTokenizer:
    """Minimal tokenizer wrapper with configurable encode/decode for tests."""

    def __init__(self, encode_fn=None, decode_fn=None, unk_id=None):
        self._encode_fn = encode_fn or (lambda t: list(range(len(t.split()))))
        self._decode_fn = decode_fn  # None means decode not supported
        self._unk_id = unk_id

    def get_name(self) -> str:
        return "mock_tok"

    def get_vocab_size(self) -> int:
        return 100

    def get_vocab(self) -> Optional[Dict[str, int]]:
        return None

    def can_encode(self) -> bool:
        return True

    def encode(self, text: str) -> List[int]:
        return self._encode_fn(text)

    def can_pretokenize(self) -> bool:
        return False

    def pretokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    def can_decode(self) -> bool:
        return self._decode_fn is not None

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> Optional[str]:
        if self._decode_fn is None:
            return None
        try:
            return self._decode_fn(token_ids)
        except Exception:
            return None

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        return self.encode(text), None

    def get_unk_token_id(self) -> Optional[int]:
        return self._unk_id

    def has_unk_token(self) -> bool:
        return self._unk_id is not None

    @classmethod
    def from_config(cls, name, config):
        return cls()


class _MockDecodableProvider(_SimpleProvider):
    """Provider that wraps a _MockDecodableTokenizer."""

    def __init__(self, tok_name: str, tokenizer: _MockDecodableTokenizer):
        super().__init__(tok_name)
        self._tokenizer = tokenizer

    def get_tokenizer(self, name: str):
        return self._tokenizer


# ======================================================================
# T8: Reconstruction fidelity
# ======================================================================

class TestReconstructionFidelity:

    def test_perfect_roundtrip(self):
        """Perfect round-trip -> exact_match=1.0, CER=0.0."""
        tok_name = "mock_tok"
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, 2, 3],
            decode_fn=lambda ids: "hello world",
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, "hello world", [1, 2, 3])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)

        summary = results["reconstruction_fidelity"]["summary"][tok_name]
        assert summary["exact_match_rate"] == pytest.approx(1.0)
        assert summary["mean_cer"] == pytest.approx(0.0)

    def test_lossy_roundtrip(self):
        """Lossy round-trip -> exact_match=0.0, CER>0."""
        tok_name = "mock_tok"
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, 2],
            decode_fn=lambda ids: "helo world",  # missing 'l'
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, "hello world", [1, 2])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)

        summary = results["reconstruction_fidelity"]["summary"][tok_name]
        assert summary["exact_match_rate"] == pytest.approx(0.0)
        assert summary["mean_cer"] > 0.0

    def test_unk_counting(self):
        """UNK tokens should be counted correctly."""
        tok_name = "mock_tok"
        unk_id = 99
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, unk_id, 2, unk_id],  # 2 UNKs out of 4
            decode_fn=lambda ids: "test text",
            unk_id=unk_id,
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, "test text", [1, unk_id, 2, unk_id])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)

        summary = results["reconstruction_fidelity"]["summary"][tok_name]
        assert summary["unk_token_rate"] == pytest.approx(0.5)

    def test_no_unk_id_defined(self):
        """When no UNK ID is defined, UNK rate should be 0.0."""
        tok_name = "mock_tok"
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, 2, 3],
            decode_fn=lambda ids: "hello",
            unk_id=None,
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, "hello", [1, 2, 3])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)

        summary = results["reconstruction_fidelity"]["summary"][tok_name]
        assert summary["unk_token_rate"] == pytest.approx(0.0)

    def test_whitespace_preserved(self):
        """All whitespace preserved -> fidelity=1.0."""
        tok_name = "mock_tok"
        text = "a b\tc"
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, 2, 3],
            decode_fn=lambda ids: text,  # perfect decode
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, text, [1, 2, 3])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)

        summary = results["reconstruction_fidelity"]["summary"][tok_name]
        assert summary["whitespace_fidelity"] == pytest.approx(1.0)

    def test_non_decodable_tokenizer_skipped(self):
        """Non-decodable tokenizer should be silently skipped."""
        tok_name = "mock_tok"
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, 2],
            decode_fn=None,  # can't decode
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, "hello", [1, 2])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)

        assert tok_name not in results["reconstruction_fidelity"]["summary"]


# ======================================================================
# T8b: Reconstruction fidelity, the code/math branch of
# BasicTokenizationMetrics.compute_reconstruction_fidelity_analysis
# ======================================================================

class TestReconstructionFidelityCodeMath:
    """Pins compute_reconstruction_fidelity_analysis's code_texts/math_texts
    branch, which no other test exercises: nothing else in this file passes
    code_texts= to BasicTokenizationMetrics or a math corpus, so the
    code_<lang> and math per_domain entries, and the encode()/decode() calls
    that build them, run under no test but this one.
    """

    @staticmethod
    def _char_roundtrip_tokenizer():
        """A tokenizer whose encode/decode is exactly reversible per
        character: encode(text) is ord() of each character, decode(ids) is
        chr() of each id. Round-tripping is lossless by construction, so
        exact_match_rate is 1.0 regardless of the snippet content, and the
        token count for a snippet is exactly its character count.
        """
        return _MockDecodableTokenizer(
            encode_fn=lambda t: [ord(c) for c in t],
            decode_fn=lambda ids: "".join(chr(i) for i in ids),
        )

    @pytest.fixture
    def code_math_setup(self, tmp_path):
        tok_name = "mock_tok"
        tok = self._char_roundtrip_tokenizer()
        provider = _MockDecodableProvider(tok_name, tok)

        math_text = "2 + 2 = 4"
        math_file = tmp_path / "math.txt"
        math_file.write_text(math_text + "\n", encoding="utf-8")

        code_texts = {"python": ["print(1)"], "cpp": ["int x = 1;"]}
        metrics = BasicTokenizationMetrics(
            provider,
            code_texts=code_texts,
            math_data_path=str(math_file),
        )
        return tok_name, metrics, code_texts, math_text

    def test_per_domain_keys_are_code_lang_and_math(self, code_math_setup):
        """by_domain has exactly one code_<lang> entry per language passed to
        code_texts=, plus one "math" entry.

        If the code changed the label it writes -- dropping the "code_"
        prefix, using the raw language name, or lumping code and math into a
        single "code_math" bucket -- the key set here would no longer match,
        and a report reader could no longer tell which language a code
        reconstruction number belongs to.
        """
        tok_name, metrics, code_texts, _math_text = code_math_setup
        results = metrics.compute_reconstruction_fidelity_analysis({tok_name: []})

        by_domain = results["reconstruction_fidelity"]["per_tokenizer"][tok_name]["by_domain"]
        expected_keys = {f"code_{lang}" for lang in code_texts} | {"math"}
        assert set(by_domain.keys()) == expected_keys

    def test_lossless_tokenizer_reconstructs_every_snippet(self, code_math_setup):
        """A lossless tokenizer's exact_match_rate is 1.0 in every code/math
        domain and overall, since decode(encode(text)) == text for all of
        them by construction of the fixture tokenizer.

        If the code/math loop stopped calling decode() on these texts, or
        compared against the wrong reference string, this would report
        something less than 1.0 even though nothing was actually lost.
        """
        tok_name, metrics, code_texts, _math_text = code_math_setup
        results = metrics.compute_reconstruction_fidelity_analysis({tok_name: []})

        by_domain = results["reconstruction_fidelity"]["per_tokenizer"][tok_name]["by_domain"]
        for lang in code_texts:
            assert by_domain[f"code_{lang}"]["exact_match_rate"] == pytest.approx(1.0)
        assert by_domain["math"]["exact_match_rate"] == pytest.approx(1.0)

        summary = results["reconstruction_fidelity"]["summary"][tok_name]
        assert summary["exact_match_rate"] == pytest.approx(1.0)

    def test_math_token_count_is_the_encoded_id_count(self, code_math_setup):
        """math's total_tokens is len(tokenizer.encode(math_text)): with the
        char-roundtrip fixture tokenizer that is exactly len(math_text),
        one id per character.

        This pins that total_tokens is read from the ids tokenizer.encode()
        actually returns rather than, say, a word or character count computed
        independently of the tokenizer. If encode() ever returned a different
        id sequence for the same text, this count would move with it.
        """
        tok_name, metrics, _code_texts, math_text = code_math_setup
        results = metrics.compute_reconstruction_fidelity_analysis({tok_name: []})

        math_domain = results["reconstruction_fidelity"]["per_tokenizer"][tok_name]["by_domain"]["math"]
        assert math_domain["count"] == 1
        assert math_domain["total_tokens"] == len(math_text)

    def test_code_domain_counts_and_tokens_per_language(self, code_math_setup):
        """Each code_<lang> domain's count is the number of snippets supplied
        for that language, and total_tokens sums each snippet's encoded id
        count -- here, its character count, one id per character.

        This pins that snippets are grouped by their code_texts= key rather
        than merged into one shared "code" bucket, and that each language's
        token total is specific to that language's snippets.
        """
        tok_name, metrics, code_texts, _math_text = code_math_setup
        results = metrics.compute_reconstruction_fidelity_analysis({tok_name: []})

        by_domain = results["reconstruction_fidelity"]["per_tokenizer"][tok_name]["by_domain"]
        for lang, snippets in code_texts.items():
            domain = by_domain[f"code_{lang}"]
            assert domain["count"] == len(snippets)
            assert domain["total_tokens"] == sum(len(s) for s in snippets)


# ======================================================================
# Reconstruction fidelity over the registered code and math corpora
# ======================================================================

class _CountingCharTokenizer(_MockDecodableTokenizer):
    """Char-level round-trip tokenizer that records every text it encodes.

    encode_with_offsets() on the base class calls encode(), so the record
    covers the provider's encoding of a registered corpus as well as any
    encoding the metric does itself.
    """

    def __init__(self):
        super().__init__(
            encode_fn=lambda t: [ord(c) for c in t],
            decode_fn=lambda ids: "".join(chr(i) for i in ids),
        )
        self.encoded_texts: List[str] = []

    def encode(self, text: str) -> List[int]:
        self.encoded_texts.append(text)
        return super().encode(text)


def _provider_with_corpora(tokenizer, code_texts, math_texts, tok_name="mock_tok"):
    from tokenizer_analysis.core.input_types import (
        CODE_CORPUS, MATH_CORPUS, Corpus,
    )

    provider = _MockDecodableProvider(tok_name, tokenizer)
    provider.add_corpus(Corpus(
        name=CODE_CORPUS, texts=code_texts,
        source="test code", synthetic=False,
    ))
    provider.add_corpus(Corpus(
        name=MATH_CORPUS, texts={MATH_CORPUS: math_texts},
        source="test math", synthetic=False,
    ))
    return provider


class TestTheCodeAndMathCorporaAreEncodedOncePerRun:
    """Reconstruction fidelity reads the ids the provider already made.

    The provider encodes a registered corpus once and memoizes it. This metric
    used to call encode() on every code and math text on top of that, so a run
    with the AST or digit metrics active encoded each of those texts twice.
    """

    def test_each_code_and_math_text_is_encoded_once(self):
        from collections import Counter

        code_texts = {"python": ["print(1)\n", "total = 12 + 345\n"]}
        math_texts = ["12 + 345 = 357"]
        tokenizer = _CountingCharTokenizer()
        metrics = BasicTokenizationMetrics(
            _provider_with_corpora(tokenizer, code_texts, math_texts)
        )

        results = metrics.compute_reconstruction_fidelity_analysis({"mock_tok": []})

        counts = Counter(tokenizer.encoded_texts)
        assert [counts[text] for text in code_texts["python"] + math_texts] == [1, 1, 1]

        # Each count is one because the metric read the provider's encoding,
        # not because it measured nothing: with the char tokenizer a text's
        # token count is its character count.
        by_domain = results["reconstruction_fidelity"]["per_tokenizer"]["mock_tok"]["by_domain"]
        assert by_domain["code_python"]["count"] == 2
        assert by_domain["code_python"]["total_tokens"] == sum(
            len(text) for text in code_texts["python"]
        )
        assert by_domain["math"]["count"] == 1
        assert by_domain["math"]["total_tokens"] == len(math_texts[0])

    def test_a_whitespace_only_text_is_dropped_on_both_sides(self):
        """The provider's encode keeps only texts satisfying text.strip(), and
        so does this metric, so a whitespace-only text is absent from both.

        If the two filters ever disagree, a text this metric measures has no
        entry in the provider's encoding and compute_reconstruction_fidelity_
        analysis raises naming it, rather than pairing the surrounding texts
        with each other's ids. max_snippet_chars produces such a text by
        truncating an indented file down to its leading whitespace.
        """
        code_texts = {"python": ["print(1)\n", "   \n  ", "print(22)\n"]}
        tokenizer = _CountingCharTokenizer()
        metrics = BasicTokenizationMetrics(
            _provider_with_corpora(tokenizer, code_texts, ["1 + 1 = 2"])
        )

        results = metrics.compute_reconstruction_fidelity_analysis({"mock_tok": []})

        assert "   \n  " not in tokenizer.encoded_texts
        domain = results["reconstruction_fidelity"]["per_tokenizer"]["mock_tok"]["by_domain"]["code_python"]
        assert domain["count"] == 2
        assert domain["total_tokens"] == len("print(1)\n") + len("print(22)\n")
        assert domain["exact_match_rate"] == pytest.approx(1.0)


class TestATokenizerThatCannotEncodeRawTextGetsNoCodeOrMathDomain:
    """Decided 2026-08-19, and the one deliberate behaviour change of this
    refactor.

    This metric selects tokenizers on can_decode() alone and then encodes the
    code and math texts, so a tokenizer that decodes but cannot encode raw text
    used to raise out of the whole analysis. It now loses the code and math
    domains with a warning naming it, which is what
    InputProvider._encode_corpus already does with the same tokenizer.

    Only those domains. An earlier version of this dropped the tokenizer from
    the results entirely, which took its prose numbers with it even though
    those are computed from the ids already in the TokenizedData and need no
    encoder.
    """

    @staticmethod
    def _decode_only_tokenizer():
        class _DecodeOnly(_MockDecodableTokenizer):
            def can_encode(self) -> bool:
                return False

            def encode(self, text: str) -> List[int]:
                raise NotImplementedError("cannot encode raw text")

        return _DecodeOnly(decode_fn=lambda ids: "".join(chr(i) for i in ids))

    @staticmethod
    def _provider(tokenizers):
        from tokenizer_analysis.core.input_types import CODE_CORPUS, Corpus

        class _MultiTokenizerProvider(_SimpleProvider):
            def __init__(self):
                super().__init__(next(iter(tokenizers)))

            def get_tokenizer_names(self) -> List[str]:
                return list(tokenizers)

            def get_tokenizer(self, name: str):
                return tokenizers[name]

        provider = _MultiTokenizerProvider()
        provider.add_corpus(Corpus(
            name=CODE_CORPUS, texts={"python": ["print(1)\n"]},
            source="test code", synthetic=False,
        ))
        return provider

    def test_the_run_completes_and_the_other_tokenizer_still_reports(self, caplog):
        import logging

        encoder = _CountingCharTokenizer()
        provider = self._provider({
            "encoder": encoder, "ids_only": self._decode_only_tokenizer(),
        })
        metrics = BasicTokenizationMetrics(provider)

        with caplog.at_level(logging.WARNING):
            results = metrics.compute_reconstruction_fidelity_analysis({})

        per_tokenizer = results["reconstruction_fidelity"]["per_tokenizer"]
        assert set(per_tokenizer) == {"encoder"}
        assert per_tokenizer["encoder"]["by_domain"]["code_python"]["count"] == 1
        assert any(
            "ids_only" in record.getMessage()
            and "cannot encode raw text" in record.getMessage()
            for record in caplog.records
        ), [record.getMessage() for record in caplog.records]

    def test_it_keeps_its_prose_domains(self):
        """The prose numbers need no encoder, so losing them was avoidable."""
        tokenizer = self._decode_only_tokenizer()
        provider = self._provider({"ids_only": tokenizer})
        metrics = BasicTokenizationMetrics(provider)

        prose = {"ids_only": [_make_td("ids_only", "hi", [ord("h"), ord("i")])]}
        results = metrics.compute_reconstruction_fidelity_analysis(prose)

        by_domain = results["reconstruction_fidelity"]["per_tokenizer"]["ids_only"]["by_domain"]
        assert by_domain["en"]["exact_match_rate"] == pytest.approx(1.0)
        assert not [d for d in by_domain if d.startswith("code_") or d == "math"], (
            f"the code corpus needs an encoder this tokenizer lacks, got {sorted(by_domain)}"
        )

    def test_it_is_not_skipped_when_there_is_no_code_or_math_text(self):
        """With nothing to encode the loop never calls encode(), so such a
        tokenizer's prose numbers are measurable and still reported.

        A can_encode() check applied unconditionally would drop them.
        """
        tokenizer = self._decode_only_tokenizer()
        provider = _MockDecodableProvider("ids_only", tokenizer)
        metrics = BasicTokenizationMetrics(provider)

        prose = {"ids_only": [_make_td("ids_only", "hi", [ord("h"), ord("i")])]}
        results = metrics.compute_reconstruction_fidelity_analysis(prose)

        per_tokenizer = results["reconstruction_fidelity"]["per_tokenizer"]
        assert per_tokenizer["ids_only"]["by_domain"]["en"]["exact_match_rate"] == pytest.approx(1.0)


# ======================================================================
# CER time-budget projection counts code/math texts before the filter
# ======================================================================

class TestTheCERBudgetProjectionCountsCodeAndMathTextsBeforeTheStripFilter:
    """total_code_math_texts, which feeds the CER time-budget projection,
    counts every code/math text passed to this metric, not just the ones
    that survive the ``text.strip()`` filter the code/math loop applies.

    The projection extrapolates the CER work still ahead from
    ``total_all_texts - texts_processed``, the count of texts left. If that
    count were taken after the filter, a corpus holding whitespace-only or
    empty snippets (what max_snippet_chars produces truncating an indented
    file down to its leading whitespace) would under-project the remaining
    CER work, and cer_skipped could come out False where the correct count
    would have set it True -- moving a number this repo publishes.
    """

    def test_whitespace_only_snippets_push_the_projection_over_budget(self):
        from unittest.mock import patch
        import itertools

        from tokenizer_analysis.metrics.basic import _CER_WARMUP

        # The budget check fires unconditionally on the _CER_WARMUP-th CER
        # call (see the "n_cer_calls == _CER_WARMUP" branch), so using exactly
        # that many real snippets makes the projection point deterministic
        # regardless of the budget value.
        n_real = _CER_WARMUP
        # 20 extra whitespace-only snippets: enough that the projection they
        # add (n_dropped * time_per_cer = 0.20s below) separates the two
        # budget outcomes with room either side, not a value load-bearing on
        # its own.
        n_dropped = 20

        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, 2, 3],
            # Never matches the source text, so every text takes the CER path.
            decode_fn=lambda ids: "MISMATCH",
        )
        provider = _MockDecodableProvider("mock_tok", tok)

        code_texts = {
            "python": (
                [f"snippet_{i}\n" for i in range(n_real)]
                + ["   \n"] * n_dropped
            ),
        }
        metrics = BasicTokenizationMetrics(provider, code_texts=code_texts)

        # Two time.monotonic() reads per CER call (one before, one after
        # _character_error_rate), 0.01s apart by construction: cer_elapsed
        # after the _CER_WARMUP-th call is exactly 0.01 * n_real = 0.5s, and
        # each call's own cost (time_per_cer) is exactly 0.01s.
        with patch(
            "tokenizer_analysis.metrics.basic.time.monotonic",
            side_effect=itertools.count(0.0, 0.01),
        ):
            # Strictly between the two projections this budget could compare
            # against: counting only the n_real texts that survive the filter
            # gives 0.5 + 0.01 * 0 = 0.5s; counting all n_real + n_dropped, as
            # this metric does, gives 0.5 + 0.01 * n_dropped = 0.7s.
            results = metrics.compute_reconstruction_fidelity_analysis(
                {"mock_tok": []}, cer_time_budget_s=0.6,
            )

        summary = results["reconstruction_fidelity"]["summary"]["mock_tok"]
        assert summary.get("cer_skipped") is True, (
            "the projection must include the dropped whitespace-only "
            "snippets, which pushes it to 0.7s and over the 0.6s budget; "
            "counting only the texts that survive the filter would leave "
            "the projection at 0.5s and cer_skipped unset"
        )


# ======================================================================
# T9: _character_error_rate edge cases
# ======================================================================

class TestCharacterErrorRate:

    def test_identical_strings(self):
        assert BasicTokenizationMetrics._character_error_rate("abc", "abc") == pytest.approx(0.0)

    def test_single_char_missing(self):
        # "abc" vs "ac": Levenshtein distance 1 (1 deletion), CER = 1/3
        assert BasicTokenizationMetrics._character_error_rate("abc", "ac") == pytest.approx(1.0 / 3.0)

    def test_empty_reference(self):
        # Empty reference -> 0.0 (nothing to measure against)
        assert BasicTokenizationMetrics._character_error_rate("", "abc") == pytest.approx(0.0)

    def test_both_empty(self):
        assert BasicTokenizationMetrics._character_error_rate("", "") == pytest.approx(0.0)

    def test_empty_hypothesis(self):
        # "abc" -> "": 3 deletions / 3 chars -> 1.0
        assert BasicTokenizationMetrics._character_error_rate("abc", "") == pytest.approx(1.0)

    def test_hypothesis_longer_than_reference(self):
        # "ab" vs "aXbY": Levenshtein distance 2 (2 insertions), CER = 2/2 = 1.0
        assert BasicTokenizationMetrics._character_error_rate("ab", "aXbY") == pytest.approx(1.0)

    def test_always_non_negative(self):
        # CER is always >= 0.0 but can exceed 1.0
        assert BasicTokenizationMetrics._character_error_rate("a", "bcdefg") >= 0.0
        assert BasicTokenizationMetrics._character_error_rate("abcdef", "x") >= 0.0

    def test_cer_can_exceed_one(self):
        # "a" vs "abcde": Levenshtein distance 4 (4 insertions), CER = 4/1 = 4.0
        assert BasicTokenizationMetrics._character_error_rate("a", "abcde") == pytest.approx(4.0)

    def test_common_prefix_suffix(self):
        # Shared prefix "hello world, goodby" and suffix "!"; differ by 1 char
        # "hello world, goodbye!" vs "hello world, goodby!" -> distance 1, len 21
        assert BasicTokenizationMetrics._character_error_rate(
            "hello world, goodbye!", "hello world, goodby!"
        ) == pytest.approx(1.0 / 21.0)

    def test_differ_only_in_middle(self):
        # 100 A's + "XYZ" + 100 B's vs same with "X_Z" -> 1 substitution, len 203
        ref = "A" * 100 + "XYZ" + "B" * 100
        hyp = "A" * 100 + "X_Z" + "B" * 100
        assert BasicTokenizationMetrics._character_error_rate(ref, hyp) == pytest.approx(1.0 / 203.0)


# ======================================================================
# T10: Whitespace fidelity
# ======================================================================

class TestWhitespaceFidelity:

    def test_whitespace_stripped(self):
        """All whitespace stripped -> 0 preserved."""
        original = "a b c"
        decoded = "abc"
        preserved, total = BasicTokenizationMetrics._whitespace_fidelity(
            original, decoded
        )
        assert total == 2
        assert preserved == 0

    def test_partial_whitespace_loss(self):
        """Two spaces in, one in the decode, so one is lost -> 1/2.

        Which of the two is "the lost one" is not a question this metric
        answers and the assertion does not claim it does: the rule counts how
        many of the original's whitespace characters an alignment can match,
        and one is all it can match here.
        """
        original = "a b c"
        decoded = "ab c"  # first space lost
        preserved, total = BasicTokenizationMetrics._whitespace_fidelity(
            original, decoded
        )
        assert total == 2
        assert preserved == 1

    def test_no_whitespace(self):
        """Text with no whitespace -> (0, 0)."""
        original = "abc"
        decoded = "abc"
        preserved, total = BasicTokenizationMetrics._whitespace_fidelity(
            original, decoded
        )
        assert total == 0
        assert preserved == 0

    def test_unicode_zs_separators_count_as_whitespace(self):
        """whitespace_fidelity counts Unicode Zs separators (NBSP / thin /
        ideographic), not just ASCII.  NBSP->space is a real loss."""
        from tokenizer_analysis.metrics.basic import _is_ws
        for ch in (" ", "\t", "\n", "\r", " ", " ", "　"):
            assert _is_ws(ch), repr(ch)
        # NBSP folded to a regular space = the non-breaking property lost
        assert BasicTokenizationMetrics._whitespace_fidelity(
            "a b", "a b") == (0, 1)
        assert BasicTokenizationMetrics._whitespace_fidelity(
            "a　b", "a　b") == (1, 1)

    def test_zwsp_cf_excluded_from_whitespace(self):
        """ZWSP (U+200B, category Cf) is deliberately NOT whitespace -- its
        loss is captured by exact_match_rate / CER, not whitespace_fidelity."""
        from tokenizer_analysis.metrics.basic import _is_ws
        assert _is_ws("​") is False
        # ZWSP not counted -> total_ws stays 0 here
        assert BasicTokenizationMetrics._whitespace_fidelity(
            "a​b", "ab") == (0, 0)

    def test_a_substituted_character_does_not_lose_the_whitespace_after_it(self):
        """The scan defect this rule replaced (RELEASE_AUDIT Q35.2 R7).

        A greedy forward scan left its pointer behind on a character it could
        not match, so every later whitespace was compared at the wrong index.
        Every space here survives at its own index.
        """
        assert BasicTokenizationMetrics._whitespace_fidelity(
            "El ni\u00f1o est\u00e1 aqu\u00ed", "el nino esta aqui") == (3, 3)
        assert BasicTokenizationMetrics._whitespace_fidelity(
            "Caf\u00e9 \u00c9lan", "cafe elan") == (1, 1)

    def test_whitespace_moved_elsewhere_is_not_credited(self):
        """The one assertion separating this rule from a context-blind one.

        Aligning the two whitespace streams alone scores this 1 of 1: the
        original holds one space and so does the decode. But the word boundary
        was deleted and a space appended somewhere else, so nothing was
        preserved. Every other case in this class scores the same either way,
        which is why this test is the one that pins the definition.
        """
        assert BasicTokenizationMetrics._whitespace_fidelity(
            "hello world", "helloworld ") == (0, 1)

    def test_the_rule_matches_an_independently_enumerated_oracle(self):
        """Differential test against a brute force, not against itself.

        The oracle enumerates subsequences with itertools.combinations and
        takes the most whitespace among the longest common ones. It shares no
        code with the implementation, which is a dynamic program. An earlier
        oracle for this metric restated the implementation's own definition and
        reported zero disagreements against a rule that was wrong.

        The alphabet must hold at least two distinct whitespace characters. With
        one, greedy matching and optimal matching coincide, and the same sweep
        returns a false all-clear.
        """
        from itertools import combinations, product

        def oracle(o, d):
            def subs(t):
                out = set()
                for r in range(len(t) + 1):
                    for idx in combinations(range(len(t)), r):
                        out.add("".join(t[i] for i in idx))
                return out
            total = sum(1 for c in o if c.isspace())
            common = subs(o) & subs(d)
            longest = max(len(x) for x in common)
            best = max(sum(1 for c in x if c.isspace())
                       for x in common if len(x) == longest)
            return (best, total)

        alphabet = "a \t"
        checked = 0
        for lo in range(0, 6):
            for ld in range(0, 5):
                for o in map("".join, product(alphabet, repeat=lo)):
                    for d in map("".join, product(alphabet, repeat=ld)):
                        checked += 1
                        assert BasicTokenizationMetrics._whitespace_fidelity(
                            o, d) == oracle(o, d), (o, d)
        assert checked == 44044, checked

    def test_it_stays_within_the_budget_it_shares_with_the_error_rate(self):
        """--cer-time-budget bounds both, so the cost has to be comparable.

        Without the common prefix and suffix trimmed, the alignment ran 921 ms
        against 1.29 ms for the character error rate on this input, and the
        budget could not see it because the call sat outside the timed region.
        The intention is the ratio, not an absolute time, so a slow machine
        does not fail it.
        """
        from time import perf_counter

        line = "        result = compute_value(alpha, beta) + offset  # note\n"
        original = line * 200
        decoded = original[:7100] + "X" + original[7101:]

        t0 = perf_counter()
        BasicTokenizationMetrics._character_error_rate(original, decoded)
        cer = perf_counter() - t0
        # Both calls the loop makes, not just the alignment: the indentation
        # sub-rate runs its own table and shares the same budget.
        t0 = perf_counter()
        BasicTokenizationMetrics._whitespace_matches(original, decoded)
        BasicTokenizationMetrics._indentation_fidelity(original, decoded)
        ws = perf_counter() - t0

        assert ws <= 2 * cer, (
            f"whitespace work {ws * 1000:.1f}ms against character error rate "
            f"{cer * 1000:.1f}ms; they share one time budget"
        )


class TestReconstructionRatesArePublishedNullWhenTheyHaveNoDenominator:
    """docs/OUTPUT.md: a value that could not be computed is null, never a
    stand-in. Four rates in this block carried one: exact_match_rate and
    mean_cer 0.0, unk_token_rate 0.0, whitespace_fidelity 1.0. No golden
    configuration reaches a zero denominator, so this is the only guard.
    """

    def test_a_text_with_no_whitespace_publishes_null_not_one(self):
        """1.0 said "every whitespace character survived" about a text that
        had none, and the two cases were indistinguishable from the field.
        """
        tok_name = "mock_tok"
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, 2, 3],
            decode_fn=lambda ids: "abc",
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, "abc", [1, 2, 3])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)
        overall = results["reconstruction_fidelity"]["per_tokenizer"][tok_name]["overall"]

        assert overall["whitespace_fidelity"] is None
        assert overall["indentation_fidelity"] is None
        assert overall["tab_fidelity"] is None
        assert overall["newline_fidelity"] is None
        # The rates that do have a denominator are unaffected.
        assert overall["exact_match_rate"] == pytest.approx(1.0)

    def test_whitespace_present_still_publishes_a_number(self):
        """The benign half: null must mean "no denominator", not "always null"."""
        tok_name = "mock_tok"
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, 2, 3],
            decode_fn=lambda ids: "a b",
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, "a b", [1, 2, 3])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)
        overall = results["reconstruction_fidelity"]["per_tokenizer"][tok_name]["overall"]

        assert overall["whitespace_fidelity"] == pytest.approx(1.0)


class TestTheStructuralWhitespaceSubRates:
    """Indentation, newlines and tabs, reported beside the roll-up.

    The roll-up weights every whitespace character equally, so it cannot
    separate damage that breaks code from damage that changes nothing. Each
    test pairs a harmful case with a benign one: a sub-rate that always
    returned 0 would satisfy the harmful half alone.
    """

    def test_indentation_is_run_exact(self):
        """Four spaces arriving as three is broken code, not 0.75 preserved."""
        src = "def f():\n    return 1\n"
        assert BasicTokenizationMetrics._indentation_fidelity(
            src, "def f():\n   return 1\n") == (0, 1)
        assert BasicTokenizationMetrics._indentation_fidelity(
            src, "def f():\n    return 1\n") == (1, 1)

    def test_indentation_survives_a_line_being_dropped(self):
        """Matched as a sequence, so a lost line does not shift the rest."""
        src = "if a:\n    x = 1\n    y = 2\n"
        preserved, total = BasicTokenizationMetrics._indentation_fidelity(
            src, "if a:\n    y = 2\n")
        assert (preserved, total) == (1, 2)

    def test_inner_spaces_collapsing_leaves_indentation_alone(self):
        """The benign half: harmless damage must not read as structural."""
        src = "def f():\n    return  a  +  b\n"
        assert BasicTokenizationMetrics._indentation_fidelity(
            src, "def f():\n    return a + b\n") == (1, 1)

    def test_a_tab_replaced_by_spaces_is_a_tab_loss(self):
        """A Makefile recipe line, and the case no bundled corpus exercises."""
        _, _, tabs, _ = BasicTokenizationMetrics._whitespace_matches(
            "t:\n\tgcc -o x\n", "t:\n    gcc -o x\n")
        assert tabs == 0
        _, _, tabs_kept, _ = BasicTokenizationMetrics._whitespace_matches(
            "t:\n\tgcc -o x\n", "t:\n\tgcc -o y\n")
        assert tabs_kept == 1

    def test_newlines_lost_and_kept(self):
        _, newlines, _, _ = BasicTokenizationMetrics._whitespace_matches(
            "l1\nl2\nl3", "l1 l2 l3")
        assert newlines == 0
        _, newlines_kept, _, _ = BasicTokenizationMetrics._whitespace_matches(
            "l1\nl2\nl3", "L1\nL2\nL3")
        assert newlines_kept == 2

    def test_the_sub_counts_never_exceed_the_roll_up(self):
        """They are partitions of one alignment, not separate computations."""
        pairs = [("a\tb\nc", "a b\nc"), ("  x\n\ty", "x\n  y"),
                 ("a b\tc\nd", "abcd"), ("\t\n ", " \n\t")]
        for original, decoded in pairs:
            matched, nl, tab, total = BasicTokenizationMetrics._whitespace_matches(
                original, decoded)
            assert nl + tab <= matched <= total, (original, decoded)


class TestWhitespaceFidelityMetadataDefinition:

    def test_reconstruction_metadata_self_describes_definition(self):
        """The widened definition is traceable in result metadata."""
        from tokenizer_analysis.metrics.basic import WHITESPACE_DEFINITION
        assert WHITESPACE_DEFINITION == "ascii(space,tab,nl,cr)+unicode_Zs"


# ======================================================================
# Vocabulary-utilization cross-language dispersion (per_language_std / cov)
# ======================================================================

class TestVocabUtilDispersion:

    def test_dispersion_undefined_when_one_language(self):
        """One language: SD and CoV are both None, the mean is the one value.

        A dispersion over a single value is undefined.  Publishing 0.0 for the
        SD read as "the same utilization in every language", which is the
        defect tokenizer_fairness_gini already avoids by returning None below
        MIN_LANGUAGES_FOR_GINI.  The --input single-corpus route makes this the
        ordinary case rather than an edge one.
        """
        tok = "t"
        provider = _SimpleProvider(tok, vocab_size=100)
        metrics = BasicTokenizationMetrics(provider)
        td = {tok: [_make_td(tok, "x", [1, 2, 3, 4, 5], lang="eng_Latn")]}
        out = metrics.compute_vocabulary_utilization_analysis(td)
        per_tok = out["vocabulary_utilization"]["per_tokenizer"][tok]
        assert per_tok["per_language_std"] is None
        assert per_tok["per_language_cov"] is None
        assert per_tok["per_language_mean"] == pytest.approx(0.05)  # 5/100

    def test_dispersion_known_value(self):
        """Two langs with utilizations [0.2, 0.4] → mean 0.3, sd≈0.1414, cov≈0.4714."""
        tok = "t"
        provider = _SimpleProvider(tok, vocab_size=10)
        metrics = BasicTokenizationMetrics(provider)
        # eng uses 2 unique tokens out of 10 -> 0.2; fra uses 4 unique out of 10 -> 0.4
        td = {tok: [
            _make_td(tok, "x", [1, 2], lang="eng_Latn"),
            _make_td(tok, "y", [3, 4, 5, 6], lang="fra_Latn"),
        ]}
        out = metrics.compute_vocabulary_utilization_analysis(td)
        per_tok = out["vocabulary_utilization"]["per_tokenizer"][tok]
        assert per_tok["per_language_mean"] == pytest.approx(0.3)
        # Sample SD with ddof=1 of [0.2, 0.4]:
        #   variance = ((0.2-0.3)^2 + (0.4-0.3)^2) / (2-1) = 0.02
        #   sd = sqrt(0.02) ≈ 0.14142135
        assert per_tok["per_language_std"] == pytest.approx(0.14142135, abs=1e-7)
        assert per_tok["per_language_cov"] == pytest.approx(0.14142135 / 0.3, abs=1e-7)

    def test_dispersion_uses_ratio_not_absolute_count(self):
        """Two tokenizers with the same per-language ratios but different vocab
        sizes must produce identical dispersion."""
        small_tok, big_tok = "small", "big"

        class TwoTokProvider(_SimpleProvider):
            def get_tokenizer_names(self):
                return [small_tok, big_tok]
            def get_vocab_size(self, name):
                return 10 if name == small_tok else 100
            def get_languages(self, tokenizer_name=None):
                return ["eng_Latn", "fra_Latn"]
        provider = TwoTokProvider("ignored")
        metrics = BasicTokenizationMetrics(provider)

        td = {
            # small_tok: 2/10 and 4/10  → ratios [0.2, 0.4]
            small_tok: [
                _make_td(small_tok, "x", [1, 2], lang="eng_Latn"),
                _make_td(small_tok, "y", [3, 4, 5, 6], lang="fra_Latn"),
            ],
            # big_tok: 20/100 and 40/100 → ratios [0.2, 0.4]
            big_tok: [
                _make_td(big_tok, "x", list(range(20)), lang="eng_Latn"),
                _make_td(big_tok, "y", list(range(20, 60)), lang="fra_Latn"),
            ],
        }
        out = metrics.compute_vocabulary_utilization_analysis(td)
        small = out["vocabulary_utilization"]["per_tokenizer"][small_tok]
        big = out["vocabulary_utilization"]["per_tokenizer"][big_tok]
        assert small["per_language_std"] == pytest.approx(big["per_language_std"], abs=1e-9)
        assert small["per_language_cov"] == pytest.approx(big["per_language_cov"], abs=1e-9)


class TestSharedCorpusIdsArePairedByTextNotByRecordOrder:
    """Reconstruction fidelity finds a text's ids by ``(label, text)``.

    Within this package the pairing could also be done by position: the
    ``text and text.strip()`` filter this metric applies to the code and math
    corpora is the same one ``InputProvider._encode_corpus`` applies, so the
    records and the texts come out in the same order and the same number. That
    agreement is a property of the two filters matching, not something the
    lookup relies on, and ``get_corpus_data`` is a public method a caller can
    implement. A provider that returns the same records in a different order is
    a correct provider, and pairing by position would silently score each text
    against another text's ids.

    Reversing the records is the smallest way to make the two disagree.
    """

    def test_records_returned_in_reverse_order_are_still_matched_correctly(self):
        from tokenizer_analysis.core.input_types import CODE_CORPUS, Corpus

        # Each snippet decodes back to itself, so correct pairing round-trips
        # exactly and a mispairing decodes to the other snippet.
        ids_for = {"aaa": [1], "bbb": [2]}
        text_for = {1: "aaa", 2: "bbb"}

        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: ids_for[t],
            decode_fn=lambda ids: text_for[ids[0]],
        )

        class _ReversedOrderProvider(_MockDecodableProvider):
            """Returns the corpus records in the opposite order to the texts."""

            def get_corpus_data(self, name):
                corpus = self.get_corpus(name)
                records = [
                    TokenizedData(
                        tokenizer_name="mock_tok", language=label,
                        tokens=ids_for[text], text=text,
                    )
                    for label, texts in corpus.texts.items()
                    for text in texts
                ]
                return {"mock_tok": list(reversed(records))}

        provider = _ReversedOrderProvider("mock_tok", tok)
        provider.add_corpus(Corpus(
            name=CODE_CORPUS, texts={"python": ["aaa", "bbb"]},
            source="test", synthetic=False,
        ))
        metrics = BasicTokenizationMetrics(provider)

        results = metrics.compute_reconstruction_fidelity_analysis({"mock_tok": []})
        by_domain = results["reconstruction_fidelity"]["per_tokenizer"]["mock_tok"]["by_domain"]

        assert by_domain["code_python"]["exact_match_rate"] == 1.0, (
            "each snippet must be scored against its own ids; pairing by "
            "position scores 'aaa' against the ids of 'bbb'"
        )


class TestAGroupedRunReportsNoCodeOrMathDomain:
    """A language group contains prose languages, so it reports prose only.

    ``UnifiedTokenizerAnalyzer.run_grouped_analysis`` selects the prose
    TokenizedData for a group's languages and calls this metric with it, but
    the code and math loop ran unconditionally off the constructor's corpora.
    Every group therefore reported the whole code and math corpus. Measured on
    the bundled demo before this was gated: the Arabic script family reported
    321 texts in its reconstruction ``global`` of which 6 were Arabic, so 315
    of the 321 were the same code and math texts that appeared in every other
    group. The same texts also entered each group's CER budget.
    """

    def _metrics_with_code(self):
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1],
            decode_fn=lambda ids: "whatever",
        )
        provider = _MockDecodableProvider("mock_tok", tok)
        return BasicTokenizationMetrics(
            provider, code_texts={"python": ["a = 1", "b = 2"]},
        )

    def test_include_code_math_false_drops_the_code_domains(self):
        metrics = self._metrics_with_code()
        td = {"mock_tok": [_make_td("mock_tok", "hello", [1])]}

        results = metrics.compute_reconstruction_fidelity_analysis(
            td, include_code_math=False,
        )
        by_domain = results["reconstruction_fidelity"]["per_tokenizer"]["mock_tok"]["by_domain"]

        assert not [d for d in by_domain if d.startswith("code_") or d == "math"], (
            f"a grouped run must report prose only, got {sorted(by_domain)}"
        )

    def test_the_default_still_reports_them(self):
        """The ungrouped run is unchanged, which is what keeps published values."""
        metrics = self._metrics_with_code()
        td = {"mock_tok": [_make_td("mock_tok", "hello", [1])]}

        results = metrics.compute_reconstruction_fidelity_analysis(td)
        by_domain = results["reconstruction_fidelity"]["per_tokenizer"]["mock_tok"]["by_domain"]

        assert "code_python" in by_domain
        assert by_domain["code_python"]["count"] == 2

    def test_the_group_global_counts_only_the_group_texts(self):
        """The defect was in `global`, not only in the per-domain listing."""
        metrics = self._metrics_with_code()
        td = {"mock_tok": [_make_td("mock_tok", "hello", [1])]}

        grouped = metrics.compute_reconstruction_fidelity_analysis(
            td, include_code_math=False,
        )["reconstruction_fidelity"]["per_tokenizer"]["mock_tok"]["overall"]
        ungrouped = metrics.compute_reconstruction_fidelity_analysis(
            td,
        )["reconstruction_fidelity"]["per_tokenizer"]["mock_tok"]["overall"]

        assert grouped["count"] == 1, "the one prose text"
        assert ungrouped["count"] == 3, "the prose text plus the two snippets"


class TestConstructionOrderDoesNotChangeWhatIsMeasured:
    """Building another metric afterwards must not redirect this one.

    DigitBoundaryMetrics registers the corpora it builds on the shared input
    provider. A BasicTokenizationMetrics built before it, holding the caller's
    own code_texts, used to re-read the registry at compute time, find the
    corpus the other constructor had just registered, and look its own texts up
    in an encoding of a different corpus. It failed loudly, but several frames
    from the constructor that caused it and only once compute() ran.
    """

    def test_a_corpus_registered_afterwards_does_not_redirect_this_metric(self):
        from tokenizer_analysis.metrics.math import DigitBoundaryMetrics

        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1],
            decode_fn=lambda ids: "a = 1",
        )
        provider = _MockDecodableProvider("mock_tok", tok)

        basic = BasicTokenizationMetrics(provider, code_texts={"python": ["a = 1"]})
        # Registers the bundled synthetic code and math corpora on the provider.
        DigitBoundaryMetrics(provider)

        results = basic.compute_reconstruction_fidelity_analysis(
            {"mock_tok": [_make_td("mock_tok", "a = 1", [1])]},
        )
        by_domain = results["reconstruction_fidelity"]["per_tokenizer"]["mock_tok"]["by_domain"]

        assert by_domain["code_python"]["count"] == 1, (
            "the one snippet this metric was constructed with, not the "
            "synthetic corpus the other constructor registered"
        )


class TestTheAggregationLabelMatchesTheComputation:
    """fertility and token_length declared micro_pooled, which constants.py
    defines as one ratio from summed counts, while both average per-document
    ratios. The two coincide on a balanced corpus, which is why it survived.
    """

    @staticmethod
    def _unequal_corpus(tok_name):
        """One language with four short documents, one with a single long one."""
        texts = [("aaa_Latn", "a b"), ("aaa_Latn", "a b"), ("aaa_Latn", "a b"),
                 ("aaa_Latn", "a b"), ("bbb_Latn", "a b c d e f g h i j")]
        return [_make_td(tok_name, t, list(range(len(t))), lang=l)
                for l, t in texts]

    def test_fertility_global_is_the_mean_of_ratios_not_the_pooled_ratio(self):
        from tokenizer_analysis.constants import AGGREGATION_MEAN_OF_RATIOS

        tok_name = "mock_tok"
        tok = _MockDecodableTokenizer(encode_fn=lambda t: [1], decode_fn=lambda i: "")
        metrics = BasicTokenizationMetrics(_MockDecodableProvider(tok_name, tok))
        rows = self._unequal_corpus(tok_name)
        results = metrics.compute_fertility_analysis({tok_name: rows})

        assert results["fertility"]["metadata"]["aggregation"] == AGGREGATION_MEAN_OF_RATIOS

        ratios = [len(r.tokens) / len(r.text.split()) for r in rows]
        mean_of_ratios = sum(ratios) / len(ratios)
        pooled = (sum(len(r.tokens) for r in rows)
                  / sum(len(r.text.split()) for r in rows))
        assert mean_of_ratios != pytest.approx(pooled), (
            "corpus is balanced, so this cannot tell the two rules apart"
        )
        glob = results["fertility"]["per_tokenizer"][tok_name]["global"]
        assert glob["mean"] == pytest.approx(mean_of_ratios)
        assert glob["mean"] != pytest.approx(pooled)

    def test_token_length_counts_documents_not_tokens(self):
        """count_unit said "tokens" beside a count of documents: 3250 in the
        committed benchmark against 109014 to 271337 actual tokens.
        """
        from tokenizer_analysis.constants import AGGREGATION_MEAN_OF_RATIOS

        tok_name = "mock_tok"
        tok = _MockDecodableTokenizer(encode_fn=lambda t: [1], decode_fn=lambda i: "")
        metrics = BasicTokenizationMetrics(_MockDecodableProvider(tok_name, tok))
        rows = self._unequal_corpus(tok_name)
        results = metrics.compute_token_length_analysis({tok_name: rows})

        meta = results["token_length"]["metadata"]
        assert meta["count_unit"] == "documents"
        assert meta["aggregation"] == AGGREGATION_MEAN_OF_RATIOS
        glob = results["token_length"]["per_tokenizer"][tok_name]["global"]
        assert glob["count"] == len(rows)


class TestAZeroTokenDocumentIsExcludedOnlyWhereItsRatioIsUndefined:
    """A text the tokenizer erases entirely.

    Reachable through --input: a text of C0 control characters is non-blank to
    str.strip() but encodes to nothing under a normalizer with
    clean_text=True. The five metrics that read one disagreed about it, and
    the disagreement was accidental rather than reasoned.

    fertility is tokens per unit, so an erased document has a defined value of
    0, and including it silently drags the mean toward zero with nothing in
    the output saying why. It is excluded and counted instead. token_length,
    avg_tokens_per_line and compression_rate already exclude it, two of them
    because the ratio would divide by zero.

    The Gini blocks and reconstruction_fidelity deliberately keep it. A
    zero-cost language is how a fairness metric reports that a tokenizer
    erased a language, and a total round-trip failure is what reconstruction
    fidelity exists to measure. Excluding there would hide the finding.
    """

    @staticmethod
    def _rows(tok_name):
        return [
            _make_td(tok_name, "a b", [1, 2], lang="aaa_Latn"),
            _make_td(tok_name, "c d", [3, 4], lang="aaa_Latn"),
            _make_td(tok_name, "\x01\x02", [], lang="aaa_Latn"),
        ]

    def test_fertility_excludes_it_and_publishes_the_count(self):
        tok = _MockDecodableTokenizer(encode_fn=lambda t: [1], decode_fn=lambda i: "")
        metrics = BasicTokenizationMetrics(_MockDecodableProvider("mock_tok", tok))
        rows = self._rows("mock_tok")
        results = metrics.compute_fertility_analysis({"mock_tok": rows})

        per_lang = results["fertility"]["per_tokenizer"]["mock_tok"]["per_language"]
        block = per_lang["aaa_Latn"]
        # Two documents at 1.0 each; the erased one is not averaged in.
        assert block["mean"] == pytest.approx(1.0)
        assert block["count"] == 2
        assert block["zero_token_documents"] == 1
        glob = results["fertility"]["per_tokenizer"]["mock_tok"]["global"]
        assert glob["zero_token_documents"] == 1

    def test_a_corpus_with_none_reports_zero_not_a_missing_key(self):
        """The benign half: the field must be present and 0, so a reader can
        tell "none were erased" from "this run did not look".
        """
        tok = _MockDecodableTokenizer(encode_fn=lambda t: [1], decode_fn=lambda i: "")
        metrics = BasicTokenizationMetrics(_MockDecodableProvider("mock_tok", tok))
        rows = self._rows("mock_tok")[:2]
        results = metrics.compute_fertility_analysis({"mock_tok": rows})

        glob = results["fertility"]["per_tokenizer"]["mock_tok"]["global"]
        assert glob["zero_token_documents"] == 0
        assert glob["count"] == 2

    def test_constructing_one_logs_a_line_naming_the_tokenizer(self, caplog):
        """The justification for dropping the constructor check said the
        package filters blank text upstream so this cannot happen. It can:
        blank-to-strip and empty-after-encoding are different properties.
        """
        with caplog.at_level("WARNING"):
            _make_td("mock_tok", "\x01\x02", [], lang="aaa_Latn")
        messages = [r.message for r in caplog.records]
        assert any("mock_tok" in m and "aaa_Latn" in m for m in messages), messages

    def test_an_ordinary_record_logs_nothing(self):
        import logging
        logger = logging.getLogger("tokenizer_analysis.core.input_types")
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logger.addHandler(handler)
        try:
            _make_td("mock_tok", "a b", [1, 2], lang="aaa_Latn")
        finally:
            logger.removeHandler(handler)
        assert records == []
