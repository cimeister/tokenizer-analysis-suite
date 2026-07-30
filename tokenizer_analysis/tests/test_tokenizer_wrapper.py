"""Tests for tokenizer wrapper consistency across wrapper types.

Verifies that HuggingFaceTokenizer, SentencePieceTokenizer,
CustomBPETokenizer, and the script_bpe wrappers (ScriptBPETokenizer,
MinGramTokenizer) behave consistently for the operations that intrinsic
metrics depend on: encode, encode_with_offsets, decode,
convert_ids_to_tokens, get_vocab, and get_vocab_size.
"""

import os
import pytest
import tempfile

from tokenizer_analysis.core.tokenizer_wrapper import (
    HuggingFaceTokenizer,
    SentencePieceTokenizer,
    CustomBPETokenizer,
    ScriptBPETokenizer,
    MinGramTokenizer,
    _ScriptTokTokenizer,
    create_tokenizer_wrapper,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# Shared corpus for training tiny tokenizers
_CORPUS = [
    "The theory of relativity revolutionized our understanding of space.",
    "Die Relativitätstheorie revolutionierte unser Verständnis von Raum.",
    "La théorie de la relativité a révolutionné notre compréhension.",
    "Hello world, this is a simple test sentence.",
    "import os; print(os.getcwd())",
    "def foo(x): return x + 1",
    "The quick brown fox jumps over the lazy dog.",
    "Les mathématiques sont belles et utiles.",
    "Spaces   and\ttabs\nand newlines are whitespace.",
    "1234567890 numbers and symbols: @#$%^&*()",
]

_TEST_TEXT = "Hello world, this is a test."
_TEST_TEXT_MULTI = "Die Relativitätstheorie ist wichtig."


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_hf_tokenizer():
    """Train a minimal BPE tokenizer using the tokenizers library."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    tok = Tokenizer(BPE(unk_token="<unk>"))
    # add_prefix_space=False so ByteLevel does not prepend a space to the
    # first token. This ensures decode(encode(text)) == text exactly,
    # without needing whitespace stripping in roundtrip assertions.
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=300,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    )
    tok.train_from_iterator(_CORPUS, trainer=trainer)
    return tok


@pytest.fixture(scope="module")
def hf_wrapper(trained_hf_tokenizer):
    return HuggingFaceTokenizer("test-hf", trained_hf_tokenizer, {})


@pytest.fixture(scope="module")
def custom_bpe_wrapper(trained_hf_tokenizer):
    return CustomBPETokenizer("test-cbpe", trained_hf_tokenizer, {})


@pytest.fixture(scope="module")
def sp_model_path():
    """Train a tiny SentencePiece model and return the model file path."""
    spm = pytest.importorskip("sentencepiece")
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for line in _CORPUS:
                f.write(line + "\n")

        prefix = os.path.join(tmpdir, "test_sp")
        # Disable SentencePiece's default NFKC normalization, dummy prefix,
        # and extra-whitespace removal so that decode(encode(text)) == text
        # exactly, without needing whitespace stripping in roundtrip assertions.
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=prefix,
            vocab_size=200,
            model_type="bpe",
            character_coverage=1.0,
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
        )
        model_path = prefix + ".model"
        # Read model bytes so the fixture outlives the tmpdir
        with open(model_path, "rb") as f:
            model_bytes = f.read()

        # Write to a persistent temp file (cleaned up at process exit)
        persistent = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
        persistent.write(model_bytes)
        persistent.close()
        yield persistent.name
        os.unlink(persistent.name)


@pytest.fixture(scope="module")
def sp_processor(sp_model_path):
    spm = pytest.importorskip("sentencepiece")
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    return sp


@pytest.fixture(scope="module")
def sp_wrapper(sp_processor):
    return SentencePieceTokenizer("test-sp", sp_processor, {})


@pytest.fixture(scope="module")
def sp_wrapper_with_bos_eos(sp_processor):
    return SentencePieceTokenizer(
        "test-sp-bos-eos", sp_processor,
        {"add_bos": True, "add_eos": True},
    )


@pytest.fixture(scope="module")
def script_bpe_wrapper():
    """Load the committed tiny SCRIPT BPE fixture through the registry.

    Skips when ``script_bpe`` is not installed (it is an optional, non-PyPI
    dependency; the wrapper imports it lazily in from_config).
    """
    pytest.importorskip("script_bpe")
    path = os.path.join(FIXTURE_DIR, "scriptbpe_tiny.json.gz")
    return create_tokenizer_wrapper(
        "test-scriptbpe", {"class": "script_bpe", "path": path}
    )


@pytest.fixture(scope="module")
def mingram_wrapper():
    """Load the committed tiny MinGram fixture through the registry."""
    pytest.importorskip("script_bpe")
    path = os.path.join(FIXTURE_DIR, "mingram_tiny.json.gz")
    return create_tokenizer_wrapper(
        "test-mingram", {"class": "mingram", "path": path}
    )


ALL_WRAPPER_FIXTURES = [
    "hf_wrapper", "sp_wrapper", "custom_bpe_wrapper",
    "script_bpe_wrapper", "mingram_wrapper",
]


# ── Tests ─────────────────────────────────────────────────────────────


class TestEncodeConsistency:
    """encode() and encode_with_offsets() must return the same token IDs."""

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_ids_match(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        for text in [_TEST_TEXT, _TEST_TEXT_MULTI]:
            ids_plain = wrapper.encode(text)
            ids_offsets, _ = wrapper.encode_with_offsets(text)
            assert ids_plain == ids_offsets, (
                f"{wrapper.get_name()}: encode() and encode_with_offsets() "
                f"returned different IDs for {text!r}"
            )


class TestNoSpecialTokensInEncode:
    """encode() must not include BOS/EOS unless explicitly configured."""

    def test_hf_no_special_tokens(self, hf_wrapper, trained_hf_tokenizer):
        """HF wrapper with add_special_tokens=False should not add BOS/EOS."""
        text = _TEST_TEXT
        ids = hf_wrapper.encode(text)
        # Encode with special tokens enabled for comparison
        ids_with_special = trained_hf_tokenizer.encode(
            text, add_special_tokens=True
        ).ids
        ids_without_special = trained_hf_tokenizer.encode(
            text, add_special_tokens=False
        ).ids
        assert ids == ids_without_special
        # If the tokenizer has a post-processor that adds specials,
        # the wrapper's output should be shorter
        if len(ids_with_special) > len(ids_without_special):
            assert len(ids) == len(ids_without_special)

    def test_sp_default_no_bos_eos(self, sp_wrapper):
        """SP wrapper without add_bos/add_eos config should not add them."""
        text = _TEST_TEXT
        ids = sp_wrapper.encode(text)
        sp = sp_wrapper._sp
        bos_id = sp.bos_id()
        eos_id = sp.eos_id()
        if bos_id >= 0 and len(ids) > 0:
            assert ids[0] != bos_id, "BOS found but add_bos not configured"
        if eos_id >= 0 and len(ids) > 0:
            assert ids[-1] != eos_id, "EOS found but add_eos not configured"

    def test_sp_with_bos_eos_adds_them(self, sp_wrapper_with_bos_eos):
        """SP wrapper with add_bos=True, add_eos=True should add them."""
        text = _TEST_TEXT
        ids = sp_wrapper_with_bos_eos.encode(text)
        sp = sp_wrapper_with_bos_eos._sp
        bos_id = sp.bos_id()
        eos_id = sp.eos_id()
        if bos_id >= 0:
            assert ids[0] == bos_id, "BOS not found despite add_bos=True"
        if eos_id >= 0:
            assert ids[-1] == eos_id, "EOS not found despite add_eos=True"

    def test_sp_bos_eos_increases_length(self, sp_wrapper, sp_wrapper_with_bos_eos):
        """Adding BOS+EOS should produce exactly 2 more tokens."""
        text = _TEST_TEXT
        ids_plain = sp_wrapper.encode(text)
        ids_bos_eos = sp_wrapper_with_bos_eos.encode(text)
        sp = sp_wrapper._sp
        expected_extra = 0
        if sp.bos_id() >= 0:
            expected_extra += 1
        if sp.eos_id() >= 0:
            expected_extra += 1
        assert len(ids_bos_eos) == len(ids_plain) + expected_extra


class TestSentencePieceSpecialTokenReporting:
    """get_special_token_strings/get_special_token_ids must distinguish "cannot
    report" from "declares none", and must not silently inherit the base
    class's empty-set default for ids.
    """

    def test_special_token_strings_none_when_every_probe_fails(self):
        """A processor whose bos/eos/unk/pad probes all raise, and that exposes
        no IsControl/IsUnknown, cannot be asked for its special tokens at all.
        get_special_token_strings() must return None (the "cannot report"
        signal that makes resolve_special_token_strings warn and fall back to
        GENERIC_SPECIAL_TOKENS), not an empty set, which would assert this
        tokenizer genuinely declares none.
        """
        class UnreadableSPProcessor:
            def bos_id(self):
                raise RuntimeError("simulated failure")

            def eos_id(self):
                raise RuntimeError("simulated failure")

            def unk_id(self):
                raise RuntimeError("simulated failure")

            def pad_id(self):
                raise RuntimeError("simulated failure")

            def get_piece_size(self):
                return 0

        wrapper = SentencePieceTokenizer("unreadable", UnreadableSPProcessor(), {})
        assert wrapper.get_special_token_strings() is None

    def test_special_token_ids_read_declared_roles(self, sp_wrapper, sp_processor):
        """get_special_token_ids() must read the model's own bos/eos/unk/pad ids
        instead of inheriting TokenizerWrapper's default empty set, and must
        skip roles the model reports unset (sentencepiece signals unset with a
        negative id).
        """
        expected = {
            i for i in (sp_processor.bos_id(), sp_processor.eos_id(),
                        sp_processor.unk_id(), sp_processor.pad_id())
            if i >= 0
        }
        assert expected, "fixture model should declare at least one of bos/eos/unk/pad"
        assert sp_wrapper.get_special_token_ids() == expected


class TestConvertIdsRoundtrip:
    """convert_ids_to_tokens(encode(text)) should produce valid token strings."""

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_all_tokens_valid(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        ids = wrapper.encode(_TEST_TEXT)
        tokens = wrapper.convert_ids_to_tokens(ids)
        assert len(tokens) == len(ids)
        for i, tok_str in enumerate(tokens):
            assert isinstance(tok_str, str), (
                f"Token {i} is {type(tok_str)}, expected str"
            )
            assert not tok_str.startswith("<UNK_"), (
                f"Token ID {ids[i]} from encode() mapped to {tok_str!r}, "
                f"all IDs from encode() should be in the vocabulary"
            )


class TestDecodeRoundtrip:
    """decode(encode(text)) must exactly recover the original text.

    The test fixtures are deliberately configured to avoid any
    preprocessing that would alter the text before encoding:
    - HF ByteLevel: add_prefix_space=False (no leading space injection)
    - SentencePiece: normalization_rule_name="identity", add_dummy_prefix=False,
      remove_extra_whitespaces=False (no NFKC normalization, no dummy prefix)

    This ensures the roundtrip test is strict: any mismatch is a real bug,
    not an artifact of pretokenization normalization.
    """

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_roundtrip(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        if not wrapper.can_decode():
            pytest.skip(f"{wrapper.get_name()} does not support decode")
        text = _TEST_TEXT
        ids = wrapper.encode(text)
        decoded = wrapper.decode(ids)
        assert decoded is not None, (
            f"{wrapper.get_name()}: decode returned None"
        )
        assert decoded == text, (
            f"{wrapper.get_name()}: roundtrip mismatch: "
            f"{decoded!r} != {text!r}"
        )


class TestVocabConsistency:
    """get_vocab_size() bounds the number of distinct tokens.

    For gapless id spaces (HF/SP/custom_bpe) it equals len(get_vocab()).
    The script_bpe wrappers reserve id 0 (SCRIPT BPE) or leave gaps from
    MinGram's pruning, so vocab_size (the id-space bound) exceeds the token
    count; get_vocab() still holds one entry per real token.
    """

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_size_matches_dict(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        vocab = wrapper.get_vocab()
        if vocab is not None:
            assert wrapper.get_vocab_size() >= len(vocab), (
                f"{wrapper.get_name()}: get_vocab_size()={wrapper.get_vocab_size()} "
                f"< len(get_vocab())={len(vocab)}"
            )
            if not isinstance(wrapper, _ScriptTokTokenizer):
                assert wrapper.get_vocab_size() == len(vocab), (
                    f"{wrapper.get_name()}: get_vocab_size()={wrapper.get_vocab_size()} "
                    f"!= len(get_vocab())={len(vocab)}"
                )


class TestOffsetsCoverText:
    """Offsets from encode_with_offsets should span the input text."""

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_offsets_span_text(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        text = _TEST_TEXT
        ids, offsets = wrapper.encode_with_offsets(text)
        if offsets is None:
            pytest.skip(f"{wrapper.get_name()} does not provide offsets")
        assert len(offsets) == len(ids), (
            f"Offset count {len(offsets)} != token count {len(ids)}"
        )
        for i, (s, e) in enumerate(offsets):
            assert 0 <= s <= len(text), (
                f"Token {i}: start offset {s} out of range [0, {len(text)}]"
            )
            assert 0 <= e <= len(text), (
                f"Token {i}: end offset {e} out of range [0, {len(text)}]"
            )
            assert s <= e, (
                f"Token {i}: start {s} > end {e}"
            )

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_offsets_cover_all_characters(self, wrapper_name, request):
        """Every character in the text should be owned by at least one token."""
        wrapper = request.getfixturevalue(wrapper_name)
        text = _TEST_TEXT
        ids, offsets = wrapper.encode_with_offsets(text)
        if offsets is None:
            pytest.skip(f"{wrapper.get_name()} does not provide offsets")
        covered = set()
        for s, e in offsets:
            covered.update(range(s, e))
        for i in range(len(text)):
            assert i in covered, (
                f"Character {i} ({text[i]!r}) not covered by any token offset"
            )


class TestBatchEncoding:
    """encode_batch_with_offsets must match per-sample encode_with_offsets."""

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_batch_matches_single(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        texts = [_TEST_TEXT, _TEST_TEXT_MULTI]
        batch_results = wrapper.encode_batch_with_offsets(texts)
        assert len(batch_results) == len(texts)
        for text, (batch_ids, batch_offsets) in zip(texts, batch_results):
            single_ids, single_offsets = wrapper.encode_with_offsets(text)
            assert batch_ids == single_ids, (
                f"{wrapper.get_name()}: batch IDs differ from single for {text!r}"
            )
            assert batch_offsets == single_offsets, (
                f"{wrapper.get_name()}: batch offsets differ from single for {text!r}"
            )

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_empty_batch(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        assert wrapper.encode_batch_with_offsets([]) == []


# ── script_bpe wrappers (SCRIPT BPE + MinGram) ────────────────────────

_SCRIPT_TOK_FIXTURES = ["script_bpe_wrapper", "mingram_wrapper"]


class TestScriptTokWrappers:
    """Behaviour specific to the script_bpe wrappers.

    These tokenizers have no special tokens, expose no HuggingFace-style
    underlying object, and (under the SCRIPT pretokenizer) normalize text, so
    their contract differs from the HF/SP wrappers in ways the shared
    parametrized tests do not cover.
    """

    def test_registry_dispatches_to_correct_class(
        self, script_bpe_wrapper, mingram_wrapper
    ):
        assert isinstance(script_bpe_wrapper, ScriptBPETokenizer)
        assert isinstance(mingram_wrapper, MinGramTokenizer)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_no_special_or_unk_tokens(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        assert wrapper.get_special_token_ids() == set()
        assert wrapper.get_unk_token_id() is None
        assert wrapper.has_unk_token() is False

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_underlying_tokenizer_is_none(self, wrapper_name, request):
        # None so MorphScore and the HF-internal sanity checks skip cleanly.
        wrapper = request.getfixturevalue(wrapper_name)
        assert wrapper.get_underlying_tokenizer() is None

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_encode_returns_plain_int_list(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        ids = wrapper.encode(_TEST_TEXT)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_no_offsets_provided(self, wrapper_name, request):
        # The bundled pretokenizer normalizes, so true source offsets are not
        # recoverable; the wrapper reports None rather than guessing.
        wrapper = request.getfixturevalue(wrapper_name)
        ids, offsets = wrapper.encode_with_offsets(_TEST_TEXT)
        assert offsets is None
        assert ids == wrapper.encode(_TEST_TEXT)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_pretokenize_returns_strings(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        pretokens = wrapper.pretokenize("Hello 12 世界")
        assert isinstance(pretokens, list)
        assert pretokens
        assert all(isinstance(p, str) for p in pretokens)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_get_vocab_is_faithful(self, wrapper_name, request):
        # tokens_repr is distinct per token for these fixtures, so get_vocab
        # holds exactly one entry per real token (no collisions collapse it).
        wrapper = request.getfixturevalue(wrapper_name)
        vocab = wrapper.get_vocab()
        assert len(vocab) == len(wrapper._backend.tokens)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_id_space_is_compacted(self, wrapper_name, request):
        # The wrapper remaps the back end's (possibly sparse) ids to a gap-free
        # range, so vocab_size is the token count (plus one for a reserved id 0),
        # never the sparse max-native-id+1. This is what keeps the
        # vocabulary-utilization denominator from being inflated by MinGram's
        # pruning gaps. (On the tiny MinGram fixture the sparse bound is 2031 for
        # 1993 tokens; compaction must bring it to 1994.)
        wrapper = request.getfixturevalue(wrapper_name)
        n_tokens = len(wrapper._backend.tokens)
        assert n_tokens <= wrapper.get_vocab_size() <= n_tokens + 1
        # The compacted id space is contiguous (no holes) and vocab_size bounds it.
        dense_ids = sorted(wrapper._dense_to_native)
        assert dense_ids == list(range(dense_ids[0], dense_ids[0] + len(dense_ids)))
        assert wrapper.get_vocab_size() == dense_ids[-1] + 1
        # Every id encode can emit stays below the reported vocab size.
        ids = wrapper.encode("The quick brown fox 42 café 世界 x >= 10")
        assert all(0 <= i < wrapper.get_vocab_size() for i in ids)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_decode_reencode_is_stable(self, wrapper_name, request):
        # decode(encode(text)) yields the normalized form, which re-encodes to
        # the same ids (a normalization-robust round-trip invariant).
        wrapper = request.getfixturevalue(wrapper_name)
        for text in [_TEST_TEXT, "世界 123 café", "def f(): return 1 + 2"]:
            ids = wrapper.encode(text)
            decoded = wrapper.decode(ids)
            assert isinstance(decoded, str)
            assert wrapper.encode(decoded) == ids

    def test_missing_path_raises_value_error(self):
        pytest.importorskip("script_bpe")
        with pytest.raises(ValueError):
            create_tokenizer_wrapper("bad", {"class": "script_bpe"})

    def test_missing_file_raises_file_not_found(self):
        pytest.importorskip("script_bpe")
        with pytest.raises(FileNotFoundError):
            create_tokenizer_wrapper(
                "bad", {"class": "mingram", "path": "/no/such/file.json.gz"}
            )

    def test_unknown_class_lists_new_keys(self):
        # The factory's error enumerates available classes, now including ours.
        with pytest.raises(ValueError) as exc:
            create_tokenizer_wrapper("x", {"class": "does_not_exist", "path": "p"})
        msg = str(exc.value)
        assert "script_bpe" in msg and "mingram" in msg


class TestScriptTokEdgeCases:
    """Edge cases that distinguish these wrappers from the dense HF/SP ones.

    The interesting properties: a non-contiguous id space (1-based, with
    reserved/pruned gaps), a bundled pretokenizer whose normalization makes
    the decode round-trip non-identity, token strings that come from
    ``tokens_repr`` (not per-id decode), the ``reindex`` load knob, and
    loading that must fail loudly when the class does not match the file.
    """

    # ── id space ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_encoded_ids_are_within_vocab_size(self, wrapper_name, request):
        # The input validator and id-indexed metrics require id < vocab_size.
        # These tokenizers have a sparse/1-based id space, so vocab_size must be
        # the id-space bound (max id + 1), not the token count; this guards that.
        wrapper = request.getfixturevalue(wrapper_name)
        vocab_size = wrapper.get_vocab_size()
        text = ("The quick brown fox 42 café Zürich Πολύγλωσσο 世界 "
                "x >= 10 and 3 + 4 * 5 = 23; def f(): return 1 + 2")
        ids = wrapper.encode(text)
        assert ids, "expected a non-empty encoding for a rich sample"
        assert all(0 <= i < vocab_size for i in ids), (
            f"{wrapper.get_name()}: encoded ids outside [0, {vocab_size}): "
            f"{[i for i in ids if not (0 <= i < vocab_size)][:5]}"
        )

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_reserved_and_out_of_range_ids_map_to_unk(self, wrapper_name, request):
        # Id 0 is reserved (absent from the token table) and vocab_size is one
        # past the end; convert must fall back to a sentinel rather than raise
        # or return a real token string.
        wrapper = request.getfixturevalue(wrapper_name)
        vocab_size = wrapper.get_vocab_size()
        assert wrapper.convert_ids_to_tokens([0]) == ["<UNK_0>"]
        assert wrapper.convert_ids_to_tokens([vocab_size]) == [f"<UNK_{vocab_size}>"]

    # ── normalization / round-trip ────────────────────────────────────

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_normalization_is_not_identity(self, wrapper_name, request):
        # The bundled pretokenizer applies NFC, so a decomposed (NFD) input
        # comes back composed (NFC). This is why reconstruction exact-match is
        # below 1.0 for normalizing inputs; it is expected, not a decode bug.
        import unicodedata
        wrapper = request.getfixturevalue(wrapper_name)
        nfd = "café"                      # 'e' + combining acute
        nfc = unicodedata.normalize("NFC", nfd)  # 'é'
        assert nfd != nfc
        decoded = wrapper.decode(wrapper.encode(nfd))
        assert decoded == nfc
        assert decoded != nfd

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_empty_and_whitespace_inputs(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        assert wrapper.encode("") == []
        assert wrapper.decode([]) == ""
        # Whitespace is not normalized away, so it round-trips exactly.
        ws = "  \t \n  "
        assert wrapper.decode(wrapper.encode(ws)) == ws

    # ── vocab views agree ─────────────────────────────────────────────

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_vocab_and_convert_match_the_backend(self, wrapper_name, request):
        # Independent check against the back end: for real tokens, the compacted
        # id must render (via convert_ids_to_tokens and via get_vocab) to the back
        # end's own tokens_repr string, and the native->compacted remap must land
        # on that id. This catches a wrong remap or wrong rendering, which a plain
        # get_vocab-vs-convert comparison cannot (both share one code path).
        wrapper = request.getfixturevalue(wrapper_name)
        vocab = wrapper.get_vocab()
        backend = wrapper._backend
        native_ids = sorted(backend.tokens)
        mid = len(native_ids) // 2
        # A spread of base atomic tokens (low ids) and merged tokens (high ids).
        sample = native_ids[:3] + native_ids[mid:mid + 3] + native_ids[-3:]
        for nid in sample:
            expected = backend.pretokenizer.tokens_repr(backend.tokens[nid].atomic_tokens)
            dense = wrapper._native_to_dense[nid]
            assert wrapper.convert_ids_to_tokens([dense]) == [expected]
            assert vocab[expected] == dense

    # ── load knobs and failure modes ──────────────────────────────────

    def test_mingram_reindex_changes_origin_not_density(self):
        # The wrapper compacts gaps either way, so both loads give a dense id
        # space; reindex only changes the origin. Reindex packs ids 0-based, so
        # vocab_size == token count; the default keeps the reserved id 0, so
        # vocab_size == token count + 1. Neither is inflated by MinGram's gaps.
        pytest.importorskip("script_bpe")
        path = os.path.join(FIXTURE_DIR, "mingram_tiny.json.gz")
        default = create_tokenizer_wrapper("mg", {"class": "mingram", "path": path})
        reindexed = create_tokenizer_wrapper(
            "mg-ri", {"class": "mingram", "path": path, "reindex": True}
        )
        n = len(default._backend.tokens)
        assert len(reindexed._backend.tokens) == n
        assert reindexed.get_vocab_size() == n          # 0-based dense
        assert default.get_vocab_size() == n + 1        # 1-based dense (reserved id 0)
        # Renumbering ids does not change what either tokenizer represents.
        assert reindexed.decode(reindexed.encode("Hello world 42")) == "Hello world 42"
        assert default.decode(default.encode("Hello world 42")) == "Hello world 42"

    def test_wrong_class_for_file_fails_loudly(self):
        # Pointing the wrong class at a file must raise, not silently build a
        # broken tokenizer (the on-disk formats are class-specific).
        pytest.importorskip("script_bpe")
        bpe_file = os.path.join(FIXTURE_DIR, "scriptbpe_tiny.json.gz")
        mingram_file = os.path.join(FIXTURE_DIR, "mingram_tiny.json.gz")
        with pytest.raises(Exception):
            create_tokenizer_wrapper("x", {"class": "mingram", "path": bpe_file})
        with pytest.raises(Exception):
            create_tokenizer_wrapper("x", {"class": "script_bpe", "path": mingram_file})
