"""Tests for the tokenizer-visualize CLI.

The tokenizer is trained at test time (same pattern as
tests/test_tokenizer_wrapper.py) so the tests do not depend on any tokenizer
file on disk.
"""

import json
import re
from collections import Counter

import pytest

from tokenizer_analysis.cli.visualize_tokenization import (
    _get_offsets,
    main,
    visualize_tokens,
)
from tokenizer_analysis.core.tokenizer_wrapper import HuggingFaceTokenizer

_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "def count(path):\n    total = 0\n    return total",
    "hello world, this is a simple test sentence.",
    "import os; print(os.getcwd())",
]


@pytest.fixture(scope="module")
def byte_bpe_wrapper():
    """A byte-level BPE with the full 256-byte alphabet and few merges.

    ``initial_alphabet=ByteLevel.alphabet()`` puts every byte token in the
    vocabulary, so CJK text encodes as a run of byte tokens. Without it the
    same text encodes entirely as <unk> and the sub-character split reporting
    under test is never reached.

    ``add_prefix_space=False`` keeps offsets aligned to the source text, so the
    reconstruction test compares against the input unchanged.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    tok.train_from_iterator(_CORPUS, trainer=BpeTrainer(
        vocab_size=300,
        special_tokens=["<unk>"],
        initial_alphabet=ByteLevel.alphabet(),
    ))
    return HuggingFaceTokenizer("tiny-byte-bpe", tok, {})


# ── Output parsing helpers ───────────────────────────────────────────────
# These invert the no-colour rendering in visualize_tokens.

_GLYPH_TO_CHAR = {"\u00b7": " ", "\u2192": "\t", "\u21b5": "\n", "\u240d": "\r"}
_SPLIT_MARK = re.compile(r"\(\d+\)")
_SEPARATOR = "\u2500" * 72


def _rendered_source_lines(output: str, n_lines: int) -> list:
    """Return the per-source-line renderings, which follow the header rule."""
    lines = output.split("\n")
    start = lines.index(_SEPARATOR) + 1
    return lines[start:start + n_lines]


def _unrender(rendered: str, line_no: int) -> str:
    """Recover one source line from its no-colour rendering."""
    prefix = f"{line_no:3d} "
    assert rendered.startswith(prefix), (
        f"line {line_no} rendering {rendered!r} lacks the {prefix!r} prefix"
    )
    body = rendered[len(prefix):]
    body = _SPLIT_MARK.sub("", body)   # drop the "(3)" split-count marks
    body = body.replace("|", "")       # drop the token boundary marks
    return "".join(_GLYPH_TO_CHAR.get(ch, ch) for ch in body)


def _stats_line(output: str, needle: str) -> str:
    return next(line for line in output.split("\n") if needle in line)


# ── Tests ────────────────────────────────────────────────────────────────


def test_exits_nonzero_when_no_tokenizer_loads(tmp_path, capsys):
    """A run where every tokenizer fails to load must fail.

    Measured before the fix: the command printed the source text, no
    tokenization, and exited 0, with no statement of how many of the requested
    tokenizers had loaded. A wrong config path was therefore indistinguishable
    from a successful run.
    """
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world\n", encoding="utf-8")
    config = tmp_path / "tokenizers.json"
    config.write_text(json.dumps({
        "broken_a": {"class": "no_such_class", "path": "/nonexistent"},
        "broken_b": {"class": "no_such_class", "path": "/nonexistent"},
    }), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["--tokenizer-config", str(config),
              "--input", str(sample), "--no-color"])

    assert exc.value.code != 0
    captured = capsys.readouterr()
    assert "Loaded 0 of 2 requested tokenizer(s)." in captured.err
    assert captured.out == "", (
        "the source text was printed even though nothing could tokenize it"
    )


@pytest.mark.parametrize("text", [
    pytest.param("def f():\n    total = 0\n        return total\n", id="indentation"),
    pytest.param("a\tb\n\nc\n", id="tab-and-blank-line"),
    pytest.param("\u76f8\u5bfe\u6027\u7406\u8ad6\n\u3053\u3093\u306b\u3061\u306f\n",
                 id="multibyte"),
    pytest.param("no trailing newline", id="no-trailing-newline"),
])
def test_no_colour_rendering_reconstructs_the_source(byte_bpe_wrapper, text):
    """The rendered view must be the source text plus removable marks.

    The no-colour view inserts a line-number prefix, '|' at token boundaries,
    '(N)' after a character split across N byte-tokens, and visible glyphs for
    whitespace. Removing exactly those must give the source back. A dropped
    character, a duplicated one, or a line-offset error in the pos bookkeeping
    (pos = line_end + 1) all break this and nothing else in the output would
    show it.
    """
    output = visualize_tokens("tiny-byte-bpe", text, byte_bpe_wrapper,
                              use_color=False, label="sample")
    expected = text.split("\n")
    rendered = _rendered_source_lines(output, len(expected))
    assert len(rendered) == len(expected)
    reconstructed = [_unrender(r, i + 1) for i, r in enumerate(rendered)]
    assert reconstructed == expected


def test_byte_continuations_are_not_counted_as_special_tokens(byte_bpe_wrapper):
    """Sub-character continuation spans belong to the split report, not 'Special'.

    _fill_offsets clamps the 2nd..Nth byte-token of a split multi-byte
    character to a zero-length span, because all of them carry the same source
    offset. Counting zero-length spans as special tokens therefore reported
    'Special: 10' for '相対性理論\\n', a text with zero special tokens, while
    the next output line reported the same 10 tokens as '10 hidden token(s)'.
    """
    text = "\u76f8\u5bfe\u6027\u7406\u8ad6\n"
    ids = byte_bpe_wrapper.encode(text)
    assert not set(ids) & byte_bpe_wrapper.get_special_token_ids(), (
        "fixture precondition: this text must encode with no special tokens"
    )

    output = visualize_tokens("tiny-byte-bpe", text, byte_bpe_wrapper,
                              use_color=False)
    split_line = _stats_line(output, "Sub-character splits")
    assert "5 char(s) split" in split_line
    assert "10 hidden token(s)" in split_line
    assert "Special:" not in output


def test_indent_patterns_match_the_source_indent_widths(byte_bpe_wrapper):
    """Each reported indent pattern must sum to a real leading-whitespace width.

    The pattern is a tuple of per-token space counts over one line's leading
    whitespace, so the tuple must sum to that line's indent width, and the
    tuples over all indented lines must account for every indented line exactly
    once. Comparing against widths read straight from the source text makes
    this independent of the char_color bookkeeping that produces the tuples.
    """
    text = (
        "def f():\n"
        "    a = 1\n"
        "    b = 2\n"
        "        c = 3\n"
        "if x:\n"
        "  y = 1\n"
    )
    output = visualize_tokens("tiny-byte-bpe", text, byte_bpe_wrapper,
                              use_color=False)
    line = _stats_line(output, "Indent patterns")

    reported = Counter()
    for body, count in re.findall(r"\(([^)]*)\) x(\d+)", line):
        widths = [int(part) for part in body.split(",") if part.strip()]
        reported[sum(widths)] += int(count)

    expected = Counter(
        len(src) - len(src.lstrip())
        for src in text.split("\n")
        if len(src) - len(src.lstrip()) > 0
    )
    assert reported == expected


def test_get_offsets_rejects_an_offset_count_mismatch():
    """A tokenizer whose offsets do not match its ids must fail by name.

    Every downstream index (char ownership, split counts, indent patterns) is
    computed by zipping ids against offsets, so a mismatch silently truncates
    the view rather than erroring. The guard names the tokenizer because a run
    covers several at once.
    """
    class MismatchedWrapper:
        def get_name(self):
            return "mismatched-tok"

        def encode_with_offsets(self, text):
            return [1, 2, 3], [(0, 1), (1, 2)]

    with pytest.raises(ValueError) as exc:
        _get_offsets(MismatchedWrapper(), "abc", [1, 2, 3])
    message = str(exc.value)
    assert "mismatched-tok" in message
    assert "3 tokens" in message and "2 offsets" in message
