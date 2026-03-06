"""Shared test infrastructure for tokenizer_analysis.metrics tests."""

from typing import Dict, List, Optional
from tokenizer_analysis.core.input_types import TokenizedData, InputProvider


class MockTokenizer:
    """Minimal tokenizer: maps integer IDs back to token strings."""

    def __init__(self, id_to_token):
        self._map = id_to_token

    def convert_ids_to_tokens(self, ids):
        return [self._map[i] for i in ids]


class MockProvider:
    """Minimal InputProvider stand-in for end-to-end compute() tests."""

    def __init__(self, tok_name, tokenizer):
        self._name = tok_name
        self._tok = tokenizer

    def get_tokenizer_names(self):
        return [self._name]

    def get_tokenizer(self, name):
        return self._tok


class SimpleProvider(InputProvider):
    """Minimal InputProvider for unit-testing metric classes."""

    def __init__(self, tok_name: str, vocab_size: int = 100):
        self._tok_name = tok_name
        self._vocab_size = vocab_size

    def get_tokenized_data(self) -> Dict[str, List[TokenizedData]]:
        return {}

    def get_tokenizer_names(self) -> List[str]:
        return [self._tok_name]

    def get_vocab_size(self, tokenizer_name: str) -> int:
        return self._vocab_size

    def get_languages(self, tokenizer_name: Optional[str] = None) -> List[str]:
        return ["en"]
