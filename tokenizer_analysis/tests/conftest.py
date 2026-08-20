"""Shared test infrastructure for tokenizer_analysis.metrics tests."""

from pathlib import Path
from typing import Dict, List, Optional

import pytest
from tokenizer_analysis.core.input_types import (
    PROSE_CORPUS, InputProvider, TokenizedData,
)


class MockTokenizer:
    """Minimal tokenizer: maps integer IDs back to token strings."""

    def __init__(self, id_to_token):
        self._map = id_to_token

    def convert_ids_to_tokens(self, ids):
        return [self._map[i] for i in ids]


class MockProvider(InputProvider):
    """Minimal InputProvider stand-in for end-to-end compute() tests.

    It subclasses InputProvider for the corpus registry. A metric that builds
    its own code or math corpus registers it on the provider, which is where a
    corpus is encoded and memoized, so a stand-in without the registry cannot
    construct DigitBoundaryMetrics or ASTBoundaryMetrics at all.

    The prose corpus stays empty: every test using this class passes prose to
    compute() directly.
    """

    def __init__(self, tok_name, tokenizer):
        self._name = tok_name
        self._tok = tokenizer

    def get_tokenizer_names(self):
        return [self._name]

    def get_tokenizer(self, name):
        return self._tok

    def get_tokenized_data(self, corpus=PROSE_CORPUS):
        if corpus != PROSE_CORPUS:
            return self._tokenized_corpus(corpus)
        return {}

    def get_vocab_size(self, tokenizer_name):
        return 0

    def get_languages(self, tokenizer_name=None):
        return []


class SimpleProvider(InputProvider):
    """Minimal InputProvider for unit-testing metric classes."""

    def __init__(self, tok_name: str, vocab_size: int = 100):
        self._tok_name = tok_name
        self._vocab_size = vocab_size

    def get_tokenized_data(self, corpus: str = PROSE_CORPUS) -> Dict[str, List[TokenizedData]]:
        if corpus != PROSE_CORPUS:
            return self._tokenized_corpus(corpus)
        return {}

    def get_tokenizer_names(self) -> List[str]:
        return [self._tok_name]

    def get_vocab_size(self, tokenizer_name: str) -> int:
        return self._vocab_size

    def get_languages(self, tokenizer_name: Optional[str] = None) -> List[str]:
        return ["en"]


# --------------------------------------------------------------------------
# FLORES+ corpus availability
# --------------------------------------------------------------------------

_FLORES_DIR = Path(__file__).resolve().parents[2] / "parallel"

#: Skip a test that needs the FLORES+ corpus, which this repository does not
#: redistribute (CC-BY-SA 4.0). Anything driving `--use-sample-data` or a config
#: from `configs/` needs it. Without the marker those tests fail on a fresh
#: checkout with a corpus error, which looks like a defect in the code rather
#: than a missing download.
requires_flores = pytest.mark.skipif(
    not (_FLORES_DIR / "eng_Latn.txt").exists(),
    reason=(
        "FLORES+ corpus not present in parallel/. Fetch it with "
        "`uv run python scripts/fetch_flores.py`."
    ),
)
