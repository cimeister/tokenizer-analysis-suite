"""Shared test infrastructure for tokenizer_analysis.metrics tests."""


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
