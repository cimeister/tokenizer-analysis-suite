# Extending

This page describes the Python API, the package layout, and how to add a new
tokenizer class or a new metric. See [../README.md](../README.md) for the
command-line interface.

## Python API

The CLI is a wrapper around `UnifiedTokenizerAnalyzer`.
`create_analyzer_from_raw_inputs` builds one from a tokenizer config dict and a
language-to-texts dict, and `run_analysis` returns the metrics as a dict:

```python
from tokenizer_analysis import create_analyzer_from_raw_inputs

analyzer = create_analyzer_from_raw_inputs(
    tokenizer_configs={
        "bpe": {"class": "huggingface", "path": "tokenizers/bpe.json"},
    },
    language_texts={
        "eng_Latn": ["The value 1234567 was measured.", "3 + 5 >= 8 holds."],
        "deu_Latn": ["Der Wert 1234567 wurde gemessen.", "3 + 5 >= 8 gilt."],
    },
    plot_save_dir="api_results",
)

results = analyzer.run_analysis(
    save_plots=False,
    include_morphscore=False,
    verbose=False,
)

print(results["compression_rate"]["per_tokenizer"]["bpe"]["global"]["compression_rate"])
print(results["fertility"]["per_tokenizer"]["bpe"]["global"]["mean"])
```

`run_analysis` also takes `include_digit_boundary`, `include_code_ast`,
`include_utf8_integrity`, `include_reconstruction`, `cer_time_budget_s`,
`save_tokenized_data` and `tokenized_data_path`, matching the corresponding CLI
flags, and taking the same defaults.
`create_analyzer_from_tokenized_data` is the equivalent factory for
pre-tokenized input.

The returned dict is the full result. The CLI always writes a slimmed projection
of it to `analysis_results.json`, and writes the full dict to
`analysis_results_full.json` only with `--save-full-results`, so a path read off
this dict is a full-results path (see
[JSON results schema](OUTPUT.md#json-results-schema)).

## Repository layout

The `tokenizer_analysis` package holds the toolkit. `main.py` defines
`UnifiedTokenizerAnalyzer`, which owns the run: it builds the metric objects,
calls them, merges the redundant ones and passes the result to the plotters.
`cli/` holds the three console-script entry points (`run_analysis.py`,
`visualize_tokenization.py`, `sanity_check.py`), and `cli/run_analysis.py` also
holds the argument parser, the run-metadata builder and the slimming step that
produces `analysis_results.json`.

`metrics/` holds one module per metric family (`basic`, `information_theoretic`,
`math`, `code_ast`, `utf8_integrity`, `morphscore`, `gini`), all subclassing
`BaseMetrics` in `metrics/base.py`, plus `redundancy.py` for the merges
described under
[Metrics reported under another metric](METRICS.md#metrics-reported-under-another-metric)
and `_treesitter_worker.py` for the parser subprocess.

`core/` holds the data structures the metrics consume: `TokenizedData` and the
`InputProvider` implementations that produce it, and `tokenizer_wrapper.py` with
the `TokenizerWrapper` base class and the class registry. `loaders/` reads
corpora (`multilingual_data.py`) and code (`code_data.py`); `config/` parses the
language and measurement configs; `visualization/` holds the plotters and the
LaTeX table generator; `diagnostics/` holds the `tokenizer-sanity-check` logic.

At package level, `constants.py` holds the shared constants, `per_example.py`
exposes the same computations at single-document granularity for joining with
LM-eval sample files, `_migration.py` raises named errors for API names removed
before 1.0.0, and `py.typed` marks the package as typed.

`sample_data/` and the top-level `parallel/` and `tokenizers/` directories hold
the bundled demo. `tests/` holds the test suite and is excluded from the built
wheel. `scripts/` holds `fetch_flores.py`, which downloads the evaluation
corpus, and nothing else.

## Adding new tokenizers

Subclass `TokenizerWrapper` from `tokenizer_analysis.core.tokenizer_wrapper` and
implement the required abstract methods. Then register it so the config system
can instantiate it by name.

### Required methods (abstract)

| Method | Purpose |
|--------|---------|
| `get_name() -> str` | Return the tokenizer's display name. |
| `get_vocab_size() -> int` | Return the total vocabulary size. |
| `get_vocab() -> Dict[str, int]` | Return the `{token_string: id}` mapping. Used for vocabulary utilization metrics and as a fallback for `convert_ids_to_tokens`. Return `None` if unavailable, which disables the vocab-dependent metrics. |
| `can_encode() -> bool` | Return `True` if `encode()` works. Return `False` for pre-tokenized-only wrappers, which skips all encoding-dependent metrics (AST, math, UTF-8, indentation). |
| `encode(text: str) -> List[int]` | Encode text to token ids. Only called when `can_encode()` is `True`. |
| `can_pretokenize() -> bool` | Whether `pretokenize()` is available. Return `False` if not applicable. |
| `pretokenize(text: str) -> List[str]` | Split text into subword pieces (strings). Only called when `can_pretokenize()` is `True`. |
| `get_special_token_strings() -> Optional[Set[str]]` | Return the surface strings the tokenizer declares special, read from its own metadata. Return an empty set if it genuinely has none, or `None` if it cannot report them, in which case the toolkit warns and falls back to `GENERIC_SPECIAL_TOKENS`. Never pattern-match on token surfaces here: the shapes overlap with ordinary content such as `[0]` and `[...]`. |
| `from_config(cls, name, config) -> TokenizerWrapper` | Class-method factory. Receives the tokenizer name and the config dict from the JSON file. |

### Optional overrides

These have working defaults but can be overridden for better results:

| Method | Default behaviour | Why override |
|--------|------------------|--------------|
| `convert_ids_to_tokens(ids) -> List[str]` | Reverses `get_vocab()`. | Faster or more accurate when the underlying library has a direct lookup (for example `id_to_token`). |
| `encode_with_offsets(text) -> (List[int], Optional[List[Tuple[int,int]]])` | Returns `(self.encode(text), None)`. | Provide `(start_char, end_char)` offsets per token for exact source-to-token mapping. Without this, the code metrics raise rather than score: the greedy character alignment they used to fall back to mismapped most non-ASCII text without reporting anything, and was removed in 1.0.2. The digit metrics raise for a tokenizer that encodes raw text and reports no offsets, and skip the document for pre-tokenized input, which carries no offsets by construction. HuggingFace `tokenizers` and SentencePiece both expose offsets natively. |
| `get_underlying_tokenizer()` | Returns `None`. | Expose the raw HuggingFace tokenizer object for consumers like MorphScore, which only works with HF tokenizers. |
| `get_unk_token_id() -> Optional[int]` | Returns `None`. | Enables `unk_token_rate`. |

### Minimal example

```python
from tokenizer_analysis.core.tokenizer_wrapper import TokenizerWrapper, register_tokenizer_class

class MyTokenizer(TokenizerWrapper):
    def __init__(self, name, tok):
        self._name, self._tok = name, tok

    def get_name(self): return self._name
    def get_vocab_size(self): return self._tok.vocab_size
    def get_vocab(self): return self._tok.get_vocab()
    def can_encode(self): return True
    def encode(self, text): return self._tok.encode(text)
    def can_pretokenize(self): return False
    def pretokenize(self, text): raise NotImplementedError

    def get_special_token_strings(self):
        # Read from the tokenizer's own metadata. Return None if it cannot
        # report them, and the toolkit warns and uses a generic list.
        return set(self._tok.all_special_tokens)

    @classmethod
    def from_config(cls, name, config):
        tok = load_my_tokenizer(config['path'])  # your loading logic
        return cls(name, tok)

register_tokenizer_class('my_class', MyTokenizer)
```

Then reference `"class": "my_class"` in your tokenizer config.

## Supplying input: providers and corpora

`InputProvider` in `tokenizer_analysis.core.input_types` is the interface the
metrics read texts and encodings from. The two implementations in `input_providers.py`
cover the shipped modes: `RawTokenizationProvider` encodes raw texts, and
`PreTokenizedProvider` serves ids that were produced elsewhere. Subclass
`InputProvider` only to supply data the two do not.

Four abstract methods have to be implemented: `get_tokenized_data()`,
`get_tokenizer_names()`, `get_vocab_size(tokenizer_name)` and
`get_languages(tokenizer_name=None)`.

`get_tokenized_data()` takes no arguments. It returns the provider's own prose
texts, keyed by tokenizer name.

The code and math corpora are separate. A run resolves each one, registers it
with `add_corpus(Corpus(...))`, and the metrics read it back through
`get_corpus_data(name)`. Both methods are concrete on `InputProvider`, so a
subclass inherits them and does not implement either. `get_corpus_data` encodes
a registered corpus once per tokenizer and memoizes the result, which is what
stops the three metric classes that consume code from encoding it three times.

Two names are refused rather than resolved. `add_corpus` rejects a corpus named
`prose`, and `get_corpus_data("prose")` raises. Prose comes from
`get_tokenized_data()`, so allowing it in the registry would create a second
place to put prose texts that no metric reads.

A metric that can also be constructed on its own accepts the corpus as a
constructor argument instead: `BasicTokenizationMetrics` and
`DigitBoundaryMetrics` both take `code_texts`, `math_data_path` and
`use_builtin_math_data`. Passing one of those while the same corpus is
registered on the provider raises. The registered corpus and the argument can
name different texts, and reporting the first under a request for the second is
the failure that check exists to prevent.

Subclasses should also implement `get_tokenizer(name)`, which returns the
`TokenizerWrapper` for a name. `InputProvider` defines it, but the definition
raises `NotImplementedError`. The method is concrete, so a provider that omits
it still constructs; the failure comes at the first call, with a message naming
the provider class rather than an `AttributeError` about a missing attribute.
Callers do not all treat that the same way.
`_encode_corpus` leaves the tokenizer out of the encoded corpus and
reconstruction fidelity skips it, both with a logged warning; MorphScore records
an `error` for that tokenizer in its results; and the digit, AST and UTF-8
metrics do not catch it at all, so the run fails once one of them is reached.
Implement `get_tokenizer` unless every metric that needs a tokenizer object is
turned off.

## Adding new metrics

1. Inherit from `BaseMetrics` in `tokenizer_analysis/metrics/base.py`.
2. Implement the `compute()` method.
3. Register it in `main.py`.
4. Add a row to the slimming step in `cli/run_analysis.py`, or the metric will
   be written as an empty object in `analysis_results.json`.
