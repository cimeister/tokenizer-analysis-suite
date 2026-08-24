"""Tests for the corpus surface on ``InputProvider``.

Prose reaches the metrics through an ``InputProvider``; the code and math
corpora are loaded and encoded by the metric classes that consume them. These
tests cover the shared surface that lets all three arrive the same way: a
``Corpus`` value, a registry on the provider, and ``get_corpus_data(name)``.

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


class _NoBatchTokenizer:
    """Offsets one text at a time, which is the only API some wrappers expose."""

    def can_encode(self):
        return True

    def encode(self, text):
        return [ord(c) for c in text]

    def encode_with_offsets(self, text):
        return self.encode(text), [(i, i + 1) for i in range(len(text))]

    def get_vocab_size(self):
        return 1000


class _NoOffsetsTokenizer:
    """Ids and neither offset method, so operator isolation skips it and says so.

    The alternative would be to guess which token covers an operator, which is
    what the reconstruction the metrics moved off did.
    """

    def can_encode(self):
        return True

    def encode(self, text):
        return [ord(c) for c in text]

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
        """The shape operator isolation publishes as `by_domain.<domain>.corpus`.

        It used to be DigitBoundaryMetrics._corpus_stats, whose output this
        asserted against until that method was deleted and the metric moved onto
        Corpus.stats().

        The CODE fixture holds a whitespace-only python text. It used to be
        counted here, which is what made this block disagree with the metrics:
        every scored path filters on ``text and text.strip()``, so the block
        whose purpose is to say what a number was measured on reported one text
        more than any metric read. It is excluded now, and the expected values
        below are three python texts rather than four.
        """
        stats = CODE.stats()
        assert stats == {
            "n_texts": 4,
            "n_chars": len("a = 1") + len("b = 2") + len("c = a + b")
                       + len("let x = 1;"),
            "n_languages": 2,
            "texts_per_language": {"python": 3, "rust": 1},
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

    def test_registering_a_corpus_named_prose_is_refused(self):
        """One way to reach the prose texts, not two that could disagree.

        The prose corpus is served from the provider's own specifications, so a
        corpus registered under that name would have recorded a source and a
        set of texts that nothing reads, while every prose number came from
        somewhere else. It used to be accepted and then ignored.
        """
        provider = _raw_provider(_CharTokenizer())
        with pytest.raises(ValueError, match="cannot be registered"):
            provider.add_corpus(
                Corpus(name="prose", texts={"eng_Latn": ["something else"]},
                       source="flores", synthetic=False)
            )

        data = provider.get_tokenized_data()
        assert [d.text for d in data["tok"]] == ["one two", "three four", "eins zwei"]

    def test_asking_get_corpus_data_for_prose_is_refused(self):
        """Prose does not come from the registry, so this is a caller error."""
        provider = _raw_provider(_CharTokenizer())
        with pytest.raises(ValueError, match="not part of the corpus registry"):
            provider.get_corpus_data("prose")

    def test_the_prose_call_still_records_encode_times(self):
        """encoding_speed is published from these, and only the prose loop keeps them."""
        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(CODE)
        provider.get_tokenized_data()
        provider.get_corpus_data("code")

        assert len(provider.encode_times["tok"]) == 3, (
            "one per prose text, and the code corpus must not enter the count"
        )


class TestEncodingARegisteredCorpus:

    def test_a_registered_corpus_is_encoded_one_batch_per_label(self):
        tokenizer = _CharTokenizer()
        provider = _raw_provider(tokenizer)
        provider.add_corpus(CODE)

        data = provider.get_corpus_data("code")

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
        assert tokenizer.single_calls == [], "the per-text path is the fallback"

    def test_a_second_call_reuses_the_first_encoding(self):
        """compute() runs once per language group; the corpus does not change.

        Re-encoding it per group would multiply the cost of every derived
        corpus by the number of groups.
        """
        tokenizer = _CharTokenizer()
        provider = _raw_provider(tokenizer)
        provider.add_corpus(CODE)

        first = provider.get_corpus_data("code")
        calls_after_first = len(tokenizer.batch_calls)
        second = provider.get_corpus_data("code")

        assert second is first
        assert len(tokenizer.batch_calls) == calls_after_first

    def test_a_failed_batch_call_aborts_and_does_not_re_encode(self):
        """This used to fall back to the per-text path under a warning saying
        the ids and offsets were the same either way.

        Nothing verified that claim, and a wrapper whose two methods disagree
        falsifies it: the run would publish numbers measured through a
        different encode path from the one it reported using. The abort names
        the tokenizer, the corpus and the label, and no single-text encode is
        attempted.
        """
        tokenizer = _CharTokenizer(batch_raises=True)
        provider = _raw_provider(tokenizer)
        provider.add_corpus(CODE)

        with pytest.raises(RuntimeError) as exc:
            provider.get_corpus_data("code")

        message = str(exc.value)
        assert "tok" in message and "code" in message, message
        assert tokenizer.single_calls == [], (
            "the corpus must not be re-encoded one text at a time"
        )

    def test_a_tokenizer_without_the_batch_api_is_encoded_one_text_at_a_time(self):
        provider = _raw_provider(_NoBatchTokenizer())
        provider.add_corpus(CODE)

        data = provider.get_corpus_data("code")

        assert [d.text for d in data["tok"]] == [
            "let x = 1;", "a = 1", "b = 2", "c = a + b"
        ]
        assert all(d.offsets for d in data["tok"])

    def test_a_tokenizer_with_neither_offset_method_yields_items_without_offsets(self):
        """Ids but no offsets, so operator isolation skips it and says so."""
        provider = _raw_provider(_NoOffsetsTokenizer())
        provider.add_corpus(CODE)

        data = provider.get_corpus_data("code")

        assert [d.text for d in data["tok"]] == [
            "let x = 1;", "a = 1", "b = 2", "c = a + b"
        ]
        assert all(d.offsets is None for d in data["tok"])
        assert all(d.tokens for d in data["tok"])

    def test_a_malformed_batch_return_aborts_rather_than_re_encoding(self):
        """A backend returning something other than (ids, offsets) pairs.

        The unpacking sits inside the try, so this used to be caught and the
        per-text path run instead. That is the same silent substitution as a
        raising batch method, and it aborts for the same reason: nothing here
        can check that the two paths agree.
        """
        class _BadBatch(_CharTokenizer):
            def encode_batch_with_offsets(self, texts):
                self.batch_calls.append(list(texts))
                return [(self.encode(t), [], "extra") for t in texts]

        tokenizer = _BadBatch()
        provider = _raw_provider(tokenizer)
        provider.add_corpus(CODE)

        with pytest.raises(RuntimeError):
            provider.get_corpus_data("code")
        assert tokenizer.single_calls == []

    def test_a_real_tokenizer_encodes_a_batch_as_it_encodes_each_text(self):
        """Encoding a corpus in batches rests on this, and nothing checked it.

        `C8` in the sanity checker compares the two, but on ids only, over 50
        prose probes, and reports WARN rather than failing. This compares ids
        and offsets on the corpus shape the batch call is used for.
        """
        from tokenizer_analysis.core.tokenizer_wrapper import create_tokenizer_wrapper

        tokenizer = create_tokenizer_wrapper(
            "bundled-bpe", {"class": "huggingface", "path": "tokenizers/bpe.json"}
        )
        texts = [t for texts in CODE.texts.values() for t in texts if t.strip()]
        texts += [
            "def f(a, b): return a <= b and a != 0",
            "résumé = naïve  # 字符 >= 10",
            "x" * 500 + " == " + "y" * 500,
        ]
        batch = tokenizer.encode_batch_with_offsets(texts)
        loop = [tokenizer.encode_with_offsets(t) for t in texts]

        assert [ids for ids, _ in batch] == [ids for ids, _ in loop]
        assert [list(offs) for _, offs in batch] == [list(offs) for _, offs in loop]
        assert all(offs for _, offs in batch), "the bundled BPE reports offsets"

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
            provider.get_corpus_data("code")
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

        assert provider.get_corpus_data("code") == {}
        assert [d.text for d in provider.get_tokenized_data()["tok"]] == ["one two"]

    def test_an_unregistered_corpus_name_raises_from_get_corpus_data(self):
        provider = _raw_provider(_CharTokenizer())

        with pytest.raises(ValueError, match="No corpus named 'code'"):
            provider.get_corpus_data("code")


class TestTheDeclaredTokenizerAccessor:

    def test_a_provider_without_one_says_so_and_names_itself(self):
        """Eight call sites already required it while the ABC never named it."""

        class _NoTokenizers(InputProvider):
            def get_tokenized_data(self):
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
            def get_tokenized_data(self):
                return {}

            def get_tokenizer_names(self):
                return ["tok"]

            def get_vocab_size(self, tokenizer_name):
                return 0

            def get_languages(self, tokenizer_name=None):
                return []

        provider = _NoTokenizers()
        provider.add_corpus(CODE)
        assert provider.get_corpus_data("code") == {}


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


class TestAMetricRefusesArgumentsARegisteredCorpusWouldOverride:
    """A metric takes its corpus from the registry or from its own arguments.

    When both are supplied the registry used to win and the arguments were
    dropped without a word, so a caller who passed ``math_data_path`` could be
    handed numbers measured on a corpus they never named. Both classes below
    accept those arguments because they can be built directly, without
    ``UnifiedTokenizerAnalyzer``, which registers the corpora instead.
    """

    def test_basic_metrics_refuses_code_texts_when_a_code_corpus_is_registered(self):
        from tokenizer_analysis.metrics.basic import BasicTokenizationMetrics

        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(CODE)

        with pytest.raises(ValueError, match="already registered"):
            BasicTokenizationMetrics(provider, code_texts={"python": ["z = 9"]})

    def test_digit_metrics_refuses_a_math_path_when_a_math_corpus_is_registered(self):
        from tokenizer_analysis.metrics.math import DigitBoundaryMetrics

        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(
            Corpus(name="math", texts={"math": ["1 + 1 = 2"]},
                   source="bundled math", synthetic=True)
        )

        with pytest.raises(ValueError, match="already registered"):
            DigitBoundaryMetrics(provider, math_data_path="/some/other/math.txt")

    def test_an_empty_code_config_is_a_request_and_is_refused(self):
        """`{}` means something for code_config specifically.

        cli/run_analysis returns `{}` for --code-ast-config to mean "use the
        bundled samples" and None to mean "disabled", so an empty dict there is
        an explicit request and the registered corpus must not override it
        silently. Each call site states its own reading, because the empty
        value does not mean the same thing for every argument: see the next
        test.
        """
        from tokenizer_analysis.metrics.code_ast import ASTBoundaryMetrics

        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(CODE)

        with pytest.raises(ValueError, match="already registered"):
            ASTBoundaryMetrics(provider, code_config={})

    def test_an_empty_code_texts_asks_for_nothing_and_is_not_refused(self):
        """Unlike code_config, an empty code_texts is not a request.

        It has always meant "this metric reports no code domain". Treating
        every empty value as a request turned a documented default into an
        error, which is why the reading is per argument.
        """
        from tokenizer_analysis.metrics.basic import BasicTokenizationMetrics

        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(CODE)

        BasicTokenizationMetrics(provider, code_texts={})

    def test_a_zero_char_cap_asks_for_nothing_either(self):
        """max_snippet_chars=0 is documented as "keep each file in full"."""
        from tokenizer_analysis.metrics.code_ast import ASTBoundaryMetrics

        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(CODE)

        ASTBoundaryMetrics(provider, max_snippet_chars=0)

    def test_a_false_boolean_is_not_a_request(self):
        """use_builtin_math_data=False is the default, not an explicit ask."""
        from tokenizer_analysis.metrics.basic import BasicTokenizationMetrics

        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(
            Corpus(name="math", texts={"math": ["1 + 1"]},
                   source="bundled math", synthetic=True)
        )

        BasicTokenizationMetrics(provider, use_builtin_math_data=False)

    def test_the_registry_is_still_used_when_no_argument_is_passed(self):
        """The check must not disturb the path the run itself takes."""
        from tokenizer_analysis.metrics.basic import BasicTokenizationMetrics

        provider = _raw_provider(_CharTokenizer())
        provider.add_corpus(CODE)

        metrics = BasicTokenizationMetrics(provider)
        assert metrics._registered_corpus("code") is CODE


class TestAWrapperThatRaisesFromEncodeWithOffsets:
    """An exception out of encode_with_offsets is a defect, not "no offsets".

    ``TokenizerWrapper.encode_with_offsets`` returns ``(ids, None)`` when a
    tokenizer has none, so raising is a wrapper defect. The encode used to
    catch it, log at debug level, and encode the text with ``encode()``
    instead, which measures that one text through a different path from the
    rest of its corpus and says so nowhere a default log level shows. The AST
    metric then failed further down reporting that the wrapper had "returned
    none", which is not what happened.
    """

    class _RaisesOnOffsets:
        def can_encode(self): return True
        def encode(self, text): return [ord(c) for c in text]
        def encode_with_offsets(self, text):
            raise RuntimeError("backend exploded")
        def get_vocab_size(self): return 1000

    def test_it_raises_naming_the_tokenizer_the_corpus_and_the_text(self):
        provider = _raw_provider(self._RaisesOnOffsets())
        provider.add_corpus(Corpus(
            name="code", texts={"python": ["a = 1"]},
            source="test", synthetic=False,
        ))

        with pytest.raises(RuntimeError) as excinfo:
            provider.get_corpus_data("code")
        message = str(excinfo.value)
        assert "'tok'" in message and "python" in message and "code" in message
        assert "backend exploded" in message

    def test_a_wrapper_that_returns_no_offsets_is_still_fine(self):
        """The legitimate case must not be caught by the same net."""
        provider = _raw_provider(_NoOffsetsTokenizer())
        provider.add_corpus(Corpus(
            name="code", texts={"python": ["a = 1"]},
            source="test", synthetic=False,
        ))

        data = provider.get_corpus_data("code")
        assert [d.offsets for d in data["tok"]] == [None]
        assert [d.tokens for d in data["tok"]] == [[ord(c) for c in "a = 1"]]


class TestAnalyzerRegistrationRollsBackOnFailure:
    """A refusal on the second add_corpus leaves the provider untouched.

    UnifiedTokenizerAnalyzer registers the code and math corpora together; a
    refusal on the second must not leave the first behind, because a later
    construction over the same provider would silently reuse it. Pinned
    because removing main.py's rollback passed the whole suite
    (RELEASE_AUDIT Q35.4).
    """

    def test_a_refused_second_registration_leaves_the_registry_empty(self, tmp_path):
        from tokenizer_analysis.main import UnifiedTokenizerAnalyzer

        class QuotaProvider(InputProvider):
            """Accepts one corpus, refuses the second."""

            def get_tokenized_data(self):
                return {"tok": [TokenizedData(
                    tokenizer_name="tok", language="en",
                    tokens=[1, 2, 3], text="abc")]}

            def get_tokenizer_names(self):
                return ["tok"]

            def get_vocab_size(self, tokenizer_name):
                return 100

            def get_languages(self, tokenizer_name=None):
                return ["en"]

            def add_corpus(self, corpus):
                if len(self._corpus_registry()) >= 1:
                    raise ValueError(
                        f"quota reached, refusing {corpus.name!r}")
                super().add_corpus(corpus)

        provider = QuotaProvider()
        with pytest.raises(ValueError, match="quota reached"):
            UnifiedTokenizerAnalyzer(
                provider, plot_save_dir=str(tmp_path),
                code_ast_config=None, use_builtin_math_data=True,
            )
        assert provider.corpus_names() == []


class TestAForeignRecordIsRefusedRatherThanRelabelled:
    """PreTokenizedProvider copied a record whose tokenizer_name disagreed
    with its dict key under the key's name, at warning level, and every metric
    then scored one tokenizer's ids under another's name.

    Both name-agreement validators read the post-correction output, so nothing
    downstream could catch it; only an id above the declared vocab size tripped
    anything. Reachable through --tokenized-data-file.
    """

    @staticmethod
    def _spec(rows):
        from tokenizer_analysis.core.input_types import InputSpecification

        class _Tok:
            def can_encode(self): return False
            def get_vocab_size(self): return 100

        return InputSpecification(tokenizer=_Tok(), tokenized_data=rows)

    def test_a_mismatched_name_aborts_and_names_both(self):
        from tokenizer_analysis.core.input_providers import PreTokenizedProvider
        from tokenizer_analysis.core.input_types import TokenizedData

        rows = [TokenizedData(tokenizer_name="other_tok", language="eng_Latn",
                              tokens=[1, 2], text="hi")]
        provider = PreTokenizedProvider({"declared_tok": self._spec(rows)})

        with pytest.raises(ValueError) as exc:
            provider.get_tokenized_data()
        message = str(exc.value)
        assert "declared_tok" in message and "other_tok" in message, message

    def test_matching_records_are_untouched(self):
        """The benign half: the ordinary path must not become an error."""
        from tokenizer_analysis.core.input_providers import PreTokenizedProvider
        from tokenizer_analysis.core.input_types import TokenizedData

        rows = [TokenizedData(tokenizer_name="declared_tok", language="eng_Latn",
                              tokens=[1, 2], text="hi")]
        provider = PreTokenizedProvider({"declared_tok": self._spec(rows)})

        out = provider.get_tokenized_data()
        assert [r.tokenizer_name for r in out["declared_tok"]] == ["declared_tok"]


class TestEncodeCorpusAbortsRatherThanReEncodingSilently:
    """Two silent re-encode paths inside _encode_corpus.

    Above the per-text loop, any exception from encode_batch_with_offsets fell
    back to per-text encoding under a warning asserting "The ids and offsets
    are the same either way; only the speed changes", which nothing verified
    and which a divergent wrapper falsifies. Below it, a wrapper returning
    (None, offsets) had its ids taken from encode() with no log at all.

    The prose loop in RawTokenizationProvider has neither defect and is not
    touched: it calls the batch method with no try/except and raises on a
    malformed return. Replicating anything from here into it would put a
    silent re-encode where there is none.
    """

    @staticmethod
    def _provider(tokenizer):
        from tokenizer_analysis.core.input_providers import RawTokenizationProvider
        from tokenizer_analysis.core.input_types import InputSpecification
        return RawTokenizationProvider({
            "tok": InputSpecification(tokenizer=tokenizer, texts={"eng_Latn": ["hi"]}),
        })

    def test_a_failing_batch_method_aborts_naming_the_tokenizer(self):
        class _Tok:
            def can_encode(self): return True
            def encode(self, t): return [1, 2]
            def encode_with_offsets(self, t): return [1, 2], [(0, 1), (1, 2)]
            def encode_batch_with_offsets(self, texts):
                raise RuntimeError("backend exploded")
            def get_vocab_size(self): return 100

        provider = self._provider(_Tok())
        with pytest.raises(RuntimeError) as exc:
            provider.add_corpus(CODE)
            provider.get_corpus_data("code")
        assert "tok" in str(exc.value), str(exc.value)

    def test_none_ids_from_the_offsets_method_abort(self):
        class _Tok:
            def can_encode(self): return True
            def encode(self, t): return [1, 2]
            def encode_with_offsets(self, t): return None, [(0, 1)]
            def get_vocab_size(self): return 100

        provider = self._provider(_Tok())
        provider.add_corpus(CODE)
        with pytest.raises(RuntimeError, match="ids"):
            provider.get_corpus_data("code")

    def test_a_tokenizer_without_the_offsets_method_still_encodes(self):
        """The benign half. `ids is None` had two causes and only one was a
        fallback: with no encode_with_offsets at all, the plain encode below
        is the primary path, and removing it outright would hand
        TokenizedData a None and name neither the tokenizer nor the method.
        """
        class _Tok:
            def can_encode(self): return True
            def encode(self, t): return [1, 2]
            def get_vocab_size(self): return 100

        provider = self._provider(_Tok())
        provider.add_corpus(CODE)
        data = provider.get_corpus_data("code")
        assert data["tok"], "a tokenizer with no offsets method must still encode"
        assert all(d.tokens == [1, 2] for d in data["tok"])
