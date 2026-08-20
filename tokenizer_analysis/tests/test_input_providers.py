"""Tests for the corpus surface on ``InputProvider``.

Prose reaches the metrics through an ``InputProvider``; the code and math
corpora are loaded and encoded by the metric classes that consume them. These
tests cover the shared surface that lets all three arrive the same way: a
``Corpus`` value, a registry on the provider, and ``get_tokenized_data(corpus)``.

The prose path must be unaffected, so several of these check that the
no-argument call still returns exactly what it did.
"""
import pytest

from tokenizer_analysis.core.input_providers import (
    PreTokenizedProvider, RawTokenizationProvider, create_input_provider,
)
from tokenizer_analysis.core.input_types import (
    Corpus, InputProvider, InputSpecification, TokenizedData,
)


class _CharTokenizer:
    """Char-level tokenizer that records how it was asked to encode."""

    def __init__(self, batch_raises=False, drop_from_batch=0):
        self._batch_raises = batch_raises
        self._drop = drop_from_batch
        self.batch_calls = []
        self.single_calls = []

    def can_encode(self):
        return True

    def encode(self, text):
        return [ord(c) for c in text]

    def encode_with_offsets(self, text):
        self.single_calls.append(text)
        return self.encode(text), [(i, i + 1) for i in range(len(text))]

    def encode_batch_with_offsets(self, texts):
        self.batch_calls.append(list(texts))
        if self._batch_raises:
            raise RuntimeError("no batch API here")
        out = [(self.encode(t), [(i, i + 1) for i in range(len(t))]) for t in texts]
        return out[: len(out) - self._drop] if self._drop else out

    def get_vocab_size(self):
        return 1000


class _IdsOnlyTokenizer:
    """What a pre-tokenized provider carries: ids, and no way to encode text."""

    def can_encode(self):
        return False

    def encode(self, text):
        raise RuntimeError("pre-tokenized input carries ids, not text")

    def get_vocab_size(self):
        return 1000


PROSE = {"eng_Latn": ["one two", "three four"], "deu_Latn": ["eins zwei"]}

#: The labels are deliberately out of alphabetical order, so that a test can
#: tell insertion order from sorted order.
CODE = Corpus(
    name="code",
    texts={"rust": ["let x = 1;"], "python": ["a = 1", "b = 2", "   ", "c = a + b"]},
    source="bundled samples",
    synthetic=True,
)


def _raw_provider(tokenizer):
    return RawTokenizationProvider({
        "tok": InputSpecification(tokenizer=tokenizer, texts=dict(PROSE))
    })


def _pretokenized_provider(tokenizer):
    return PreTokenizedProvider({
        "tok": InputSpecification(
            tokenizer=tokenizer,
            tokenized_data=[
                TokenizedData(tokenizer_name="tok", language="eng_Latn",
                              tokens=[1, 2, 3], text="one two"),
            ],
        )
    })


class TestTheCorpusRegistry:

    def test_a_corpus_round_trips_through_the_registry(self):
        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(CODE)

        assert provider.get_corpus("code") is CODE
        assert provider.get_corpus("code").source == "bundled samples"
        assert provider.get_corpus("code").synthetic is True

    def test_corpus_names_lists_what_was_added_in_registration_order(self):
        provider = _raw_provider(_CharTokenizer())
        assert provider.corpus_names() == []

        math_corpus = Corpus(name="math", texts={"math": ["1 + 1 = 2"]},
                             source="bundled math", synthetic=True)
        provider.add_corpus(math_corpus)
        provider.add_corpus(CODE)
        assert provider.corpus_names() == ["math", "code"], (
            "registration order, not sorted: the report names the corpora in "
            "the order the run loaded them"
        )

    def test_an_unregistered_name_raises_naming_it_and_the_registered_ones(self):
        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(CODE)

        with pytest.raises(ValueError) as excinfo:
            provider.get_corpus("maths")
        message = str(excinfo.value)
        assert "'maths'" in message and "'code'" in message

    def test_registering_a_name_twice_is_refused_rather_than_overwriting(self):
        """Whichever loader ran second would otherwise redefine the corpus.

        The metrics that already read the first registration would then report
        numbers measured on the second, with nothing in the output saying so.
        """
        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(CODE)

        with pytest.raises(ValueError) as excinfo:
            provider.add_corpus(Corpus(name="code", texts={"go": ["x := 1"]},
                                       source="starcoder parquet", synthetic=False))
        message = str(excinfo.value)
        assert "bundled samples" in message and "starcoder parquet" in message
        assert provider.get_corpus("code") is CODE

    def test_two_providers_do_not_share_a_registry(self):
        """The registry is built on first use, so it must not land on the class."""
        first = _raw_provider(_CharTokenizer())
        second = _raw_provider(_CharTokenizer())
        first.add_corpus(CODE)

        assert second.corpus_names() == []

    def test_corpus_stats_reports_the_size_of_what_was_measured(self):
        """Equal to DigitBoundaryMetrics._corpus_stats, which publishes it today.

        The published field is `by_domain.<domain>.size`; this pins the shape so
        that moving the metric onto Corpus.stats() cannot change it.
        """
        from tokenizer_analysis.metrics.math import DigitBoundaryMetrics

        assert CODE.stats() == DigitBoundaryMetrics._corpus_stats(CODE.texts)
        stats = CODE.stats()
        assert stats == {
            "n_texts": 5,
            "n_chars": len("a = 1") + len("b = 2") + 3 + len("c = a + b")
                       + len("let x = 1;"),
            "n_languages": 2,
            "texts_per_language": {"python": 4, "rust": 1},
        }
        assert list(stats["texts_per_language"]) == ["python", "rust"], (
            "sorted by label, so two runs over the same corpus print the same "
            "block whatever order the loader happened to read the labels in"
        )


class TestTheProseCorpusIsUnaffected:

    def test_the_no_argument_call_still_returns_the_prose_data(self):
        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(CODE)

        data = provider.get_tokenized_data()
        assert [d.text for d in data["tok"]] == ["one two", "three four", "eins zwei"]
        assert [d.language for d in data["tok"]] == [
            "eng_Latn", "eng_Latn", "deu_Latn"
        ]
        assert all(d.metadata["source"] == "raw_tokenization" for d in data["tok"])

    def test_a_corpus_registered_as_prose_does_not_replace_the_prose_data(self):
        """The prose corpus is served from the provider's own specifications.

        Registering one under the name "prose" records its texts and its source
        for reporting; routing the default call to it instead would change every
        prose number in the output.
        """
        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(Corpus(name="prose", texts={"eng_Latn": ["something else"]},
                                   source="flores", synthetic=False))

        data = provider.get_tokenized_data()
        assert [d.text for d in data["tok"]] == ["one two", "three four", "eins zwei"]

    def test_the_prose_call_still_records_encode_times(self):
        """encoding_speed is published from these, and only the prose loop keeps them."""
        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(CODE)
        provider.get_tokenized_data()
        provider.get_tokenized_data("code")

        assert len(provider.encode_times["tok"]) == 3, (
            "one per prose text, and the code corpus must not enter the count"
        )


class TestEncodingARegisteredCorpus:

    def test_a_registered_corpus_is_encoded_one_batch_per_label(self):
        tokenizer = _CharTokenizer()
        provider = _raw_provider(tokenizer)
        provider.add_corpus(CODE)

        data = provider.get_tokenized_data("code")

        assert [d.text for d in data["tok"]] == [
            "let x = 1;", "a = 1", "b = 2", "c = a + b"
        ], "whitespace-only texts are dropped before the batch call"
        assert [d.language for d in data["tok"]] == [
            "rust", "python", "python", "python"
        ]
        assert all(d.offsets for d in data["tok"])
        assert tokenizer.batch_calls == [
            ["let x = 1;"], ["a = 1", "b = 2", "c = a + b"]
        ]

    def test_a_second_call_reuses_the_first_encoding(self):
        """compute() runs once per language group; the corpus does not change.

        Re-encoding it per group would multiply the cost of every derived
        corpus by the number of groups.
        """
        tokenizer = _CharTokenizer()
        provider = _raw_provider(tokenizer)
        provider.add_corpus(CODE)

        first = provider.get_tokenized_data("code")
        calls_after_first = len(tokenizer.batch_calls)
        second = provider.get_tokenized_data("code")

        assert second is first
        assert len(tokenizer.batch_calls) == calls_after_first

    def test_a_failed_batch_call_falls_back_to_one_call_per_text(self):
        tokenizer = _CharTokenizer(batch_raises=True)
        provider = _raw_provider(tokenizer)
        provider.add_corpus(CODE)

        data = provider.get_tokenized_data("code")

        assert tokenizer.single_calls == [
            "let x = 1;", "a = 1", "b = 2", "c = a + b"
        ]
        assert [d.text for d in data["tok"]] == tokenizer.single_calls

    def test_a_short_batch_raises_naming_the_counts_and_the_corpus(self):
        """Pairing by position would attach one text's offsets to another."""
        provider = _raw_provider(_CharTokenizer(drop_from_batch=1))
        # One label of three texts, so the message reports 2 against 3 rather
        # than the degenerate 0 against 1.
        provider.add_corpus(Corpus(
            name="code", texts={"python": ["a = 1", "b = 2", "c = a + b"]},
            source="bundled samples", synthetic=True,
        ))

        with pytest.raises(ValueError) as excinfo:
            provider.get_tokenized_data("code")
        message = str(excinfo.value)
        assert "2 encodings" in message and "3 'python' texts" in message
        assert "code corpus" in message

    def test_a_tokenizer_that_cannot_encode_raw_text_is_left_out(self):
        """It gets no code or math domain rather than crashing the metric.

        A pre-tokenized provider carries ids somebody else produced, so there is
        no way to encode a corpus it never saw.
        """
        provider = _pretokenized_provider(_IdsOnlyTokenizer())
        provider.add_corpus(CODE)

        assert provider.get_tokenized_data("code") == {}
        assert [d.text for d in provider.get_tokenized_data()["tok"]] == ["one two"]

    def test_an_unregistered_corpus_name_raises_from_get_tokenized_data(self):
        provider = _raw_provider(_CharTokenizer())

        with pytest.raises(ValueError, match="No corpus named 'code'"):
            provider.get_tokenized_data("code")


class TestTheDeclaredTokenizerAccessor:

    def test_a_provider_without_one_says_so_and_names_itself(self):
        """Eight call sites already required it while the ABC never named it."""

        class _NoTokenizers(InputProvider):
            def get_tokenized_data(self, corpus="prose"):
                return {}

            def get_tokenizer_names(self):
                return ["tok"]

            def get_vocab_size(self, tokenizer_name):
                return 0

            def get_languages(self, tokenizer_name=None):
                return []

        with pytest.raises(NotImplementedError, match="_NoTokenizers"):
            _NoTokenizers().get_tokenizer("tok")

    def test_such_a_provider_gets_an_empty_corpus_rather_than_an_exception(self):
        """The skip that keeps a missing tokenizer from crashing the whole metric."""

        class _NoTokenizers(InputProvider):
            def get_tokenized_data(self, corpus="prose"):
                if corpus != "prose":
                    return self._tokenized_corpus(corpus)
                return {}

            def get_tokenizer_names(self):
                return ["tok"]

            def get_vocab_size(self, tokenizer_name):
                return 0

            def get_languages(self, tokenizer_name=None):
                return []

        provider = _NoTokenizers()
        provider.add_corpus(CODE)
        assert provider.get_tokenized_data("code") == {}


class TestMixedSpecificationsAreRefused:
    """One run analyses one mode.

    The provider that combined them was never constructed: the CLI selects raw
    or pre-tokenized for the whole run. A run that did reach it would mix
    numbers measured by encoding text with numbers measured from ids somebody
    else produced, with nothing in the output saying which tokenizer came from
    which.
    """

    def test_the_factory_names_both_sides(self):
        specs = {
            "raw_tok": InputSpecification(
                tokenizer=_CharTokenizer(), texts={"eng_Latn": ["one two"]},
            ),
            "pre_tok": InputSpecification(
                tokenizer=_IdsOnlyTokenizer(),
                tokenized_data=[TokenizedData(tokenizer_name="pre_tok",
                                              language="eng_Latn", tokens=[1])],
            ),
        }

        with pytest.raises(ValueError) as excinfo:
            create_input_provider(specs)
        message = str(excinfo.value)
        assert "'raw_tok'" in message and "'pre_tok'" in message


class TestAZeroTokenEncodingIsAMeasurement:
    """TokenizedData used to refuse an empty token list.

    A tokenizer that encodes a text to zero tokens has measured something, and
    refusing to construct the item turns that into a crash rather than a
    number. Nothing depended on the check: every library construction site
    filters blank text upstream, so it never fired on real input.
    """

    def test_an_item_with_no_tokens_constructs_and_counts_zero(self):
        item = TokenizedData(tokenizer_name="tok", language="eng_Latn",
                             tokens=[], text="")
        assert item.token_count == 0
        assert item.unique_tokens == set()

    def test_the_other_two_checks_still_fire(self):
        with pytest.raises(ValueError, match="tokenizer_name cannot be empty"):
            TokenizedData(tokenizer_name="", language="eng_Latn", tokens=[1])
        with pytest.raises(ValueError, match="language cannot be empty"):
            TokenizedData(tokenizer_name="tok", language="", tokens=[1])
