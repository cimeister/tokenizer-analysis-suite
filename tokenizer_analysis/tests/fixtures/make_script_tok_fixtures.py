"""Generate tiny SCRIPT BPE and MinGram fixtures for the wrapper tests.

These fixtures back the ``ScriptBPETokenizer`` / ``MinGramTokenizer`` tests in
``test_tokenizer_wrapper.py``. They are committed so the test suite does not
train anything; this script only needs to be re-run if the fixtures must be
regenerated.

Requires the ``script_bpe`` package (from https://github.com/... /script_tok),
which is not a dependency of this library. Run it from an environment that has
``script_bpe`` installed, for example:

    /path/to/script_tok/.venv/bin/python \
        tokenizer_analysis/tests/fixtures/make_script_tok_fixtures.py

The trainers use multiprocessing, so the work must run under the
``if __name__ == "__main__"`` guard below (importing at top level would spawn
workers during module import and fail).

Both fixtures use the ``scriptenc`` pretokenizer (the SCRIPT encoding), matching
the real use case and exercising the wrapper's block-marker token strings. The
files are saved gzip-compressed (~0.4 MB each); the ``.json.gz`` suffix is what
tells ``BPETokenizer.load`` / ``MinGramModel.load`` to gunzip on read.
"""

import os
import tempfile

# A short multi-script sample: Latin (with diacritics), Greek, digits, code,
# and math operators, so the trained vocab has some variety.
SAMPLE = (
    "The quick brown fox jumps over 12 lazy dogs.\n"
    "Zürich café naïve coöperate façade.\n"
    "Πολύγλωσσο κείμενο 3.14159 και αριθμοί.\n"
    "def add(a, b):\n    return a + b  # 2+2==4\n"
    "3 + 4 * 5 - 6 / 2 = 17 and x >= 10\n"
) * 40

ADDITIONAL_VOCAB_SIZE = 80  # merges on top of the scriptenc base atomic table
FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    from script_bpe.corpus import PretokenizedCorpus
    from script_bpe.pretokenize import get_pretokenizer
    from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
    from script_bpe.tokenizers.mingram.trainer import (
        MinGramTrainer,
        MinGramTrainerConfig,
    )

    tmp = tempfile.mkdtemp()

    pretok = get_pretokenizer("scriptenc")
    bpe_corpus = PretokenizedCorpus.from_texts(
        "fixture_scriptbpe", texts=[SAMPLE], pretokenizer=pretok, base_path=tmp
    )
    bpe = BPETrainer(
        pretok, bpe_corpus,
        BPETrainerConfig(additional_vocab_size=ADDITIONAL_VOCAB_SIZE,
                         num_workers=1, verbose=False),
    ).train()
    bpe_path = os.path.join(FIXTURE_DIR, "scriptbpe_tiny.json.gz")
    bpe.save(bpe_path)
    print(f"wrote {bpe_path} ({os.path.getsize(bpe_path):,} bytes, "
          f"{len(bpe.tokens)} tokens)")

    mg_pretok = get_pretokenizer("scriptenc")
    mg_corpus = PretokenizedCorpus.from_texts(
        "fixture_mingram", texts=[SAMPLE], pretokenizer=mg_pretok, base_path=tmp
    )
    mingram = MinGramTrainer(
        mg_pretok, mg_corpus,
        MinGramTrainerConfig(additional_vocab_size=ADDITIONAL_VOCAB_SIZE,
                             overshoot_factor=1.5, num_workers=1, verbose=False),
    ).train()
    mg_path = os.path.join(FIXTURE_DIR, "mingram_tiny.json.gz")
    mingram.save(mg_path)
    print(f"wrote {mg_path} ({os.path.getsize(mg_path):,} bytes, "
          f"{len(mingram.tokens)} tokens)")


if __name__ == "__main__":
    main()
