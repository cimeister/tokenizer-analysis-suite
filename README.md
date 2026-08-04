# TokEval

A toolkit for computing intrinsic quality metrics for tokenizers across natural
language, code, and math.

## Install

Python 3.10 or newer. Install from a git checkout:

```bash
git clone https://github.com/cimeister/tokenizer-intrinsic-evals.git
cd tokenizer-intrinsic-evals
uv sync
```

That is enough for the [Quick Start](#quick-start), which runs on a corpus you
supply. The bundled demo and the configs in `configs/` additionally need the
FLORES+ corpus, which is fetched rather than shipped; see
[The evaluation corpus](#the-evaluation-corpus).

This project is not published to PyPI, so there is nothing to `pip install` by
name. To install it as a dependency rather than working in the checkout, use
`pip install git+https://github.com/cimeister/tokenizer-intrinsic-evals.git`.
The import name is `tokenizer_analysis`. The demo data and the example configs
live in the checkout and are not part of an installable package, so a checkout
is required for those in any case.

### Optional extras

Order matters here. Install every extra in one `uv sync`, then do the editable
MorphScore install last:

```bash
# 1. extras, all named in one command
#    parquet:       reading corpora and code corpora from .parquet files
#    sentencepiece: loading SentencePiece model files (the `sentencepiece` class)
uv sync --extra parquet --extra sentencepiece

# 2. MorphScore morphological analysis, last
git submodule update --init --recursive
uv pip install -e ./morphscore
```

`uv sync` performs an exact sync by default: it removes packages that the
lockfile and the named extras do not account for. So `uv sync --extra parquet`
run after `uv pip install -e ./morphscore` uninstalls MorphScore, and
`uv sync --extra sentencepiece` run after `uv sync --extra parquet` uninstalls
pyarrow. Each step undoes the one before it. Pass `uv sync --inexact` if you
need a different order.

**MorphScore note**: the data is a separate download. The files are on the Hub
in the layout the code reads, one CSV per language named for the language in
English:

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('catherinearnett/morphscore', repo_type='dataset',
                  local_dir='morphscore_data', allow_patterns=['*.csv'])
"
```

Point `--morphscore-data-dir` elsewhere if you keep them somewhere else. Only
`<ISO 639-3>_<Script>` language codes are mapped to those files automatically;
see [the MorphScore repository](https://github.com/cimeister/morphscore) for the
code that produces them.

### The evaluation corpus

The configs in `configs/` and the `--use-sample-data` demo read
`parallel/<iso639-3>_<Script>.txt`. Those files are not in this repository:
they come from FLORES+, which is CC-BY-SA 4.0, and this repository does not
redistribute it. Fetch them:

```bash
uv pip install datasets            # or: uv sync --extra flores
hf auth login                      # FLORES+ is gated, approval is automatic
uv run python scripts/fetch_flores.py                              # 13 languages
uv run python scripts/fetch_flores.py --config configs/flores60_lang_config.json
uv run python scripts/fetch_flores.py --all                        # every language
```

A run that names a file which is not there aborts and repeats that command; it
never proceeds on a smaller corpus than the config asked for. Nothing in the
library requires FLORES+: `--input` and `--language-config` take your own
corpus, and only the demo and the shipped configs point at `parallel/`.
See NOTICE for the attribution terms.

## Quick Start

Compare tokenizers on one corpus. The corpus is written locally; `gpt2` and
`xlm-roberta-base` are downloaded from the Hugging Face Hub the first time this
runs (both are public models, no login needed).

```bash
# 1. a corpus: one plain-text file, one document per line
cat > corpus.txt <<'CORPUS'
The quick brown fox jumps over the lazy dog.
Tokenizers differ most on text they were not trained on.
def compute_total(items): return sum(i.price for i in items)
The invoice totalled 1234567 euros on 2024-03-15.
CORPUS

# 2. the tokenizers to compare
cat > my_tokenizers.json <<'TOKENIZERS'
{
  "gpt2":     {"class": "huggingface", "path": "gpt2"},
  "xlm-r":    {"class": "huggingface", "path": "xlm-roberta-base"}
}
TOKENIZERS

# 3. run
uv run tokenizer-analysis --tokenizer-config my_tokenizers.json --input corpus.txt

# 4. read one number: bytes per token, higher meaning fewer tokens
uv run python -c "
import json
d = json.load(open('results/analysis_results.json'))
for tok, block in d['compression_rate']['per_tokenizer'].items():
    print(tok, round(block['global']['compression_rate'], 3))
"
```

That prints `gpt2 3.8` and `xlm-r 3.119` on those four lines. `uv run` is what
puts the console script on the path after `uv sync`; activate the venv instead
if you prefer, and drop the prefix.

`--input` takes a single file (`.txt`, `.json`, `.jsonl`, `.parquet`) or a
directory of them. A `.txt` file is one document per line, which is what the
example above writes; see [Data Configuration](#data-configuration) for the
other three. A tokenizer `path` is either a Hub model id, as here, or a local
`tokenizer.json`.

Results land in `results/analysis_results.json`, with plots beside it. Every
metric is defined in [METRICS.md](METRICS.md), and
[the worked comparison](benchmarks/open_source/REPORT.md) runs the same command
shape over nine tokenizers.

For several corpora at once, and for the cross-lingual metrics, use
`--language-config` instead (see [Data Configuration](#data-configuration)).

This command leaves three things at their defaults, which is fine for a first
look and not for a number you intend to publish: the code metrics run on
built-in synthetic samples, the digit metrics run on the prose corpus, and CER
is dropped for any tokenizer that exceeds the 10-second budget. See
[Full evaluation](#full-evaluation).

### Full evaluation

```bash
tokenizer-analysis \
    --tokenizer-config my_tokenizers.json \
    --language-config my_languages.json \
    --code-ast-config my_code.json \
    --use-builtin-math-data \
    --cer-time-budget 0
```

- `--code-ast-config` supplies real source code. Without it the code metrics are
  computed on built-in synthetic samples, and the run prints a warning to that
  effect. Those numbers are not comparable across the two cases.
- `--use-builtin-math-data` points the three digit metrics at the 285 bundled
  math texts (`tokenizer_analysis/sample_data/math_samples.json`).
  `--math-data FILE` points them at your own instead. Without either, those three
  are computed on whatever numbers the prose corpus happens to contain: on the
  bundled five-language FLORES+ corpus at the default `--samples-per-lang` that
  is 1797 digit spans, of which 6 are longer than 4 digits, so the place-value
  boundaries at positions L-6 and L-9 are almost never exercised. Neither flag changes the `math` domain of `operator_isolation_rate`,
  which already reads the same bundled file.
- `--cer-time-budget 0` removes the cap on CER computation, so `mean_cer` and
  `whitespace_fidelity` are measured rather than reported as `null` for slow
  tokenizers.

### The bundled demo

The demo runs on FLORES+, which this repository does not redistribute, so fetch
the corpus first:

```bash
uv sync --extra flores
hf auth login                                  # FLORES+ is gated
uv run python scripts/fetch_flores.py          # writes parallel/
uv run tokenizer-analysis --use-sample-data
```

Two sample tokenizers over five FLORES+ languages. Plots are written to
`results/` as SVG; open `results/fertility_individual.svg` in a browser or an
image viewer. Without the corpus the run exits naming that fetch command; see
[The evaluation corpus](#the-evaluation-corpus).

The demo also needs a source checkout: it reads `parallel/` and `tokenizers/`,
neither of which is part of an installable package.
`--use-sample-data` supplies its own tokenizers, corpus and measurement
settings, so it cannot be combined with `--tokenizer-config`,
`--language-config`, `--input` or `--measurement-config`.

Two things about the demo output are expected rather than broken. First, the two
demo tokenizers fail `tokenizer-sanity-check` on purpose (`bpe.json` covers 94
of 256 byte values), so [that command](#tokenizer-sanity-check) has something to
report. Second, `unigramlm` decodes slowly enough to exceed the default
`--cer-time-budget 10`, so its `mean_cer` and `whitespace_fidelity` come back
`null` and the log names the projected time that triggered the skip. Add
`--cer-time-budget 0` to measure them.

## tokenizer-visualize

Renders token boundaries directly on source text, for inspecting how different
tokenizers split code, math and multilingual content.

```bash
# Built-in samples (Python code, LaTeX math, multilingual text)
uv run tokenizer-visualize \
    --tokenizer-config configs/sample_tokenizers.json

# Only specific tokenizers
uv run tokenizer-visualize \
    --tokenizer-config configs/sample_tokenizers.json \
    --tokenizers "bpe" "unigramlm"

# Every file in a directory. Within a file, samples are separated by a line
# containing only "---"; --samples-per-file controls how many are read.
uv run tokenizer-visualize \
    --tokenizer-config configs/sample_tokenizers.json \
    --input path/to/samples/ --samples-per-file 3
```

Each sample is shown as line-numbered source text followed by a colour-coded
token-boundary view for every tokenizer, plus whitespace and indentation
statistics. `--no-color` disables the ANSI colours for piping to a file.

## tokenizer-sanity-check

Runs a single-tokenizer health report: byte coverage, whitespace and digit
handling, special-token behaviour, determinism, Unicode normalization,
vocabulary integrity, and vocabulary reachability. Each check is flagged pass,
warn or fail, and a failing check sets a non-zero exit code, so the command can
gate a tokenizer before a full analysis.

```bash
# A single tokenizer (CLASS:PATH form)
uv run tokenizer-sanity-check huggingface:tokenizers/bpe.json

# Every tokenizer listed in a config
uv run tokenizer-sanity-check --tokenizer-config configs/sample_tokenizers.json

# One tokenizer from a config, by name
uv run tokenizer-sanity-check --tokenizer-config configs/sample_tokenizers.json --only bpe

# Add multilingual breadth (requires a language config)
uv run tokenizer-sanity-check huggingface:tokenizers/bpe.json \
    --use-sample-data --language-config configs/core_lang_config.json
```

`--exit-zero` always returns exit code 0, reporting without gating.
`--quiet` collapses passing checks in the text report. `--output-dir` writes
`sanity_results.json`.

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
[JSON results schema](#json-results-schema)).

## Configuration Files

### Tokenizer Configuration

Specify tokenizers via `--tokenizer-config`:

```json
{
  "tokenizer1": {
    "class": "huggingface",
    "path": "bert-base-uncased"
  },
  "tokenizer2": {
    "class": "huggingface",
    "path": "/path/to/local/tokenizer"
  },
  "custom_bpe": {
    "class": "custom_bpe",
    "path": "/path/to/bpe/directory"
  }
}
```

Available classes: `"huggingface"` (aliases `"hf"`, `"transformers"`, and the
deprecated `"standard"`, which warns), `"sentencepiece"`, `"custom_bpe"`
(requires `vocab.json` + `merges.txt`), `"unimixlm"`, `"pretokenized"` (for
pre-tokenized data), and `"script_bpe"` / `"mingram"` for the SCRIPT BPE and
MinGram tokenizers. You can add your own with `register_tokenizer_class()`
(see [Adding new tokenizers](#adding-new-tokenizers)).

The `"script_bpe"` and `"mingram"` classes require the external `script_bpe`
package, which is not a dependency of this toolkit (install it with
`pip install -e /path/to/script_tok` or put its repo on `PYTHONPATH`). Each
`path` is a single saved tokenizer file (`.json` or `.json.gz`); `"mingram"`
accepts an optional `reindex` flag (default `false`). These tokenizers have no
special tokens. Under the SCRIPT pretokenizer a token string is its rendered
form, with `<|BLOCK_...|>` / `<|SCRIPT_INDEX_...|>` markers for sub-character
pieces, and reflects NFC normalization and digit regrouping, so the token-string
metrics (byte coverage, junk and dead-vocab) describe that rendered form.
MorphScore is not available for these two classes.

### Data Configuration

Specify languages and analysis groupings via `--language-config`:

```json
{
  "languages": {
    "eng_Latn": {
      "name": "English",
      "iso_code": "en",
      "data_path": "/path/to/english/data"
    },
    "arb_Arab": {
      "name": "Arabic",
      "iso_code": "ar",
      "data_path": "/path/to/arabic/data"
    }
  },
  "analysis_groups": {
    "script_family": {
      "Latin": ["eng_Latn", "fra_Latn"],
      "Arabic": ["arb_Arab"]
    },
    "resource_level": {
      "high": ["eng_Latn"],
      "low": ["som_Latn"]
    }
  }
}
```

For simple setups, `"languages"` can map codes directly to file paths:
`{"en": "/path/to/data"}`.

#### Corpus file formats

A corpus file, whether named by `--input` or by a `data_path`, is read in one of
four formats, chosen by extension:

| Extension | Shape |
|---|---|
| `.txt` | Plain text. One document per line, or one document per blank-line-separated paragraph; see below |
| `.json` | Either a JSON array of objects each with a `text` key, or one such object on its own |
| `.jsonl` | JSON Lines: one object per line, each with a `text` key |
| `.parquet` | A DataFrame with a text column; see below |

#### What counts as one document in a `.txt` file

A blank line means the file is paragraph-separated, and each paragraph is one
document. With no blank line anywhere, each line is one document. That rule is
decided per file, and it is the unit every per-document metric divides by, so it
is worth being deliberate about: a file of prose paragraphs and a file of one
sentence per line are read differently on purpose.

Two filters apply either way. A document shorter than 5 characters is dropped,
and runs of whitespace inside a document are collapsed to single spaces, which
means a `.txt` corpus is not the right input for measuring indentation. Use
`--code-ast-config` for source code, which preserves it.

`--samples-per-lang` caps the number of documents kept per corpus, so on a
line-per-document file it is a line count and on a paragraph file it is a
paragraph count.

A `.json` file shaped `{"texts": [...]}` is **not** a corpus file and loads as
zero texts. That shape belongs to `--math-data` only, which accepts either
`{"texts": [...]}` or a bare JSON array of strings.

Parquet column names differ by loader:

- A **corpus** parquet is read by `loaders/multilingual_data.py`, which takes the
  first column present out of `text`, `content`, `sentence`, `document`,
  `passage`, and otherwise falls back to the first string column with a log line
  naming it.
- A **code** parquet named by `--code-ast-config` is read by
  `loaders/code_data.py`, which requires a `content` column and logs a warning
  naming the columns it did find if there is none.

Either way parquet needs the `parquet` extra. Without an engine installed the
run fails with an error naming the extra, rather than reading an empty corpus.

A directory passed as `--input` or as a `data_path` is scanned for `*.json`,
`*.parquet` and `*.txt` in that order, sorted by name. `*.jsonl` is not in that
list, so a `.jsonl` file is read when named directly and skipped when it sits in
a scanned directory. A directory holding only `.jsonl` files loads zero texts.

#### Metric families and the data they need

| Metric family | Computed on | Flag that supplies it | Without the flag |
|---|---|---|---|
| Compression, fertility, token length, vocabulary utilization, Rényi efficiency, bigram and trigram entropy, Gini, UTF-8 integrity, reconstruction fidelity, encoding speed | the main corpus | `--input`, `--language-config` or `--use-sample-data` | the run aborts with an error naming the three options. There is no fallback to demo data |
| The three digit metrics: `three_digit_boundary_alignment`, digit split variability, `numeric_magnitude_consistency` | dedicated math texts | `--math-data FILE` or `--use-builtin-math-data` | computed on the main corpus instead, and the run prints a warning naming all three. `--no-digit-boundary` turns them off |
| `operator_isolation_rate`, `math` domain | dedicated math texts | `--math-data FILE` | the bundled `sample_data/math_samples.json`, which is also what `--use-builtin-math-data` names. This domain never falls back to the main corpus |
| `operator_isolation_rate`, `prose` domain | the main corpus | `--input`, `--language-config` or `--use-sample-data` | see the first row |
| The three AST metrics: `ast_boundary_alignment`, `identifier_fragmentation`, `indentation_consistency` | dedicated source-code snippets | `--code-ast-config FILE` | computed on built-in synthetic code samples, and the run prints a warning naming all three. `--no-code-ast` turns them off |
| `operator_isolation_rate`, `code` domain | dedicated source-code snippets | `--code-ast-config FILE` | the bundled `sample_data/code_samples.json`. This domain runs under `--no-code-ast` as well; only `--no-digit-boundary`, which drops `operator_isolation_rate` entirely, turns it off |
| MorphScore | MorphScore datasets | `--morphscore` or `--morphscore-config`, plus `--morphscore-data-dir` | not computed |
| Cross-language metrics (`tokenizer_fairness_gini`, `per_language_cov`) | at least 2 languages | `--language-config` with 2 or more entries | computed as `null` with a stated reason |

`operator_isolation_rate` logs its three sources on one line each run:
`Operator isolation domains: prose=multilingual, math=..., code=...`.

#### How `data_path` is resolved

An absolute path is used as written. A relative path is resolved against the
package root, the directory that holds the `tokenizer_analysis` package, which
in a source checkout is the repository root. It is never resolved against the
directory you happen to run from, so the same config names the same corpus from
anywhere and a run is reproducible. A path that does not resolve is an error
naming it; the language is never dropped. Use an absolute path for data outside
the repository.

A relative `path` in a tokenizer config follows a different rule, because a Hub
model id such as `meta-llama/Meta-Llama-3-8B` is also a relative-looking string.
There, a path that exists relative to your working directory is used as is, a
path that exists only under the package root is rewritten with a log line
recording the rewrite, and anything else is passed to the loader unchanged.
`--input` is a command-line path and is always relative to your working
directory.

### Text Measurement Configuration

Control how text "length" is measured for metric normalization via
`--measurement-config`:

| Method | `method` value | Counting key and options | Default for |
|--------|----------------|--------------------------|-------------|
| Bytes | `"bytes"` | `byte_counting`: `"utf8"`, `"hf_bytelevel"` | Compression, Gini, the information-theoretic metrics |
| Characters | `"characters"` | (none) | (none) |
| Lines | `"lines"` | `line_counting`: `"single"`, `"newline_split"`, `"custom_regex"` | (none) |
| Words | `"words"` | `word_counting`: `"python_split"`, `"hf_whitespace"`, `"regex_whitespace"`, `"custom_regex"` | Fertility |

Every metric that normalizes by text length uses the unit set here, so
`--measurement-config` changes compression, Gini and the information-theoretic
metrics together. Fertility is the exception: it always counts words, because a
tokens-per-word figure is what the name means.

`include_empty_splits` (bool, default `false`) affects word and line counting.
`custom_regex` (string) is required when a counting method is set to
`"custom_regex"`. Unknown keys are rejected with an error.

Example:
```json
{
  "method": "lines",
  "line_counting": "newline_split",
  "include_empty_splits": false
}
```

### MorphScore Configuration

Specify via `--morphscore-config`:

```json
{
    "data_dir": "/path/to/morphscore/datasets",
    "by_split": false,
    "freq_scale": true,
    "exclude_single_tok": false
}
```

Requires languages in `<ISO 639-3>_<script>` format (for example `eng_Latn`).
Override with `"language_subset"` in the config to bypass code mapping.
Datasets come from the Hugging Face dataset `catherinearnett/morphscore`; see
[Optional extras](#optional-extras) above for the download command. The
MorphScore submodule holds the scoring code, not the data.

### Code AST Configuration

Specify source code paths for AST boundary analysis via `--code-ast-config`:

```json
{
  "python": "/path/to/python/files/",
  "javascript": "/path/to/js/files.parquet",
  "java": "/path/to/java/dir/"
}
```

A parquet file here needs a `content` column and the `parquet` extra. StarCoder
metadata prefixes are stripped automatically, as are a leading byte-order mark
and CRLF line endings. Without a config file, built-in synthetic code samples
are used and the run logs a warning. Which languages are measured, and why three
are not, is in [METRICS.md](METRICS.md#code-tokenization-metrics).

## Metrics

Definitions, worked examples and the caveats for each metric are in
**[METRICS.md](METRICS.md)**.
[Metric names and results keys](#metric-names-and-results-keys) below maps each
metric to where its value sits in the results file.

## Output Structure

A run with every output flag set writes:

```
results/
├── analysis_results.json            # the metrics, slimmed schema; always written
├── analysis_results_full.json       # every computed field (--save-full-results)
├── fertility_individual.svg         # one chart per metric, always written unless --no-plots
├── compression_rate_individual.svg
├── vocabulary_utilization_individual.svg
├── bigram_entropy_individual.svg
├── lorenz_curves_individual.svg
├── tokenizer_fairness_gini_individual.svg
├── vocab_util_cross_lingual_cov_individual.svg
├── morphscore_individual.svg         # only when MorphScore results are present (--morphscore or --morphscore-config)
├── utf8_integrity.svg
├── per-language/                    # one chart per metric per language (--per-language-plots)
├── faceted_plots/                   # one subplot per tokenizer, shared y-axis (--faceted-plots)
├── grouped_plots/                   # one chart per metric per language group (--run-grouped-analysis)
└── latex_tables/                    # LaTeX tables (--generate-latex-tables)
```

`--save-tokenized-data` writes the cache and its companion files next to the
pickle rather than into `--output-dir`, so `--tokenized-data-output-path` moves
them all together:

```
<pickle dir>/
├── tokenized_data.pkl               # the cached tokenization
├── tokenized_data_config.json       # the vocabulary map that accompanies it
├── tokenized_data_language_config.json
└── <tokenizer>_vocab.txt            # one per tokenizer, line-by-line vocabulary
```

`--latex-output-dir` defaults differently on the two LaTeX paths:
`--generate-latex-tables` writes to `<output-dir>/latex_tables`, while
`--custom-latex-config` writes to `<output-dir>` itself with no subdirectory.
Passing `--latex-output-dir` overrides both.

### Metric names and results keys

Every metric, the top-level key it is written under in `analysis_results.json`,
and the path to its headline value. `<tok>` is a tokenizer name from your
config; `<lang>` is a language code.

The identifiers in this table are the results keys. The LaTeX table generator
uses its own row ids (`three_digit_boundary_f1`, `operator_isolation`,
`ast_full_alignment`, `ident_fragmentation`, `indent_depth_corr`,
`utf8_boundary_crossing`, `avg_token_rank`, `utf8_char_split`), and
`vocab_util_cross_lingual_cov` is a plot filename. None of those strings appears
in the results file.

| Metric | Top-level key | Path to the headline value | Direction |
|---|---|---|---|
| Compression rate | `compression_rate` | `.per_tokenizer.<tok>.global.compression_rate` | higher |
| Tokens per line | (under `compression_rate`) | `.per_tokenizer.<tok>.tokens_per_line.global_avg` | lower |
| Fertility | `fertility` | `.per_tokenizer.<tok>.global.mean` | lower |
| Token length | `token_length` | `.per_tokenizer.<tok>.global.mean` | neither |
| Vocabulary utilization | `vocabulary_utilization` | `.per_tokenizer.<tok>.global.utilization` | higher |
| Type-token ratio | (under `vocabulary_utilization`) | `.per_tokenizer.<tok>.type_token_ratio.global_ttr` | neither |
| Cross-lingual vocabulary-utilization CoV | (under `vocabulary_utilization`) | `.per_tokenizer.<tok>.per_language_cov` | lower |
| Rényi efficiency | `renyi_efficiency` | `.per_tokenizer.<tok>.global.renyi_<alpha>`, one field per alpha in 1.0, 2.0, 2.5, 3.0 | higher |
| Average token rank | (under `renyi_efficiency`) | `.per_tokenizer.<tok>.unigram_distribution.global_avg_token_rank` | neither |
| Bigram entropy | `bigram_entropy` | `.per_tokenizer.<tok>.global.bigram_entropy` | higher |
| Trigram entropy | `trigram_entropy` | `.per_tokenizer.<tok>.global.trigram_entropy` | higher |
| MorphScore | `morphscore` | `.per_tokenizer.<tok>.global.avg_morphscore_recall` and `.avg_morphscore_precision` | higher |
| Three-digit boundary alignment | `three_digit_boundary_alignment` | `.per_tokenizer.<tok>.global.mean_f1` | higher |
| Digit split variability | (under `three_digit_boundary_alignment`) | `.per_tokenizer.<tok>.split_variability.by_digit_length.<n>.<lang>.entropy` | lower |
| Numeric magnitude consistency | `numeric_magnitude_consistency` | `.per_tokenizer.<tok>.global.mean_fertility` | lower |
| Operator isolation rate | `operator_isolation_rate` | `.per_tokenizer.<tok>.global.overall_isolation_rate` | higher |
| Compound operator preservation | (under `operator_isolation_rate`) | `.per_tokenizer.<tok>.global.overall_compound_preservation_rate` | higher |
| Round-trip exact match rate | `reconstruction_fidelity` | `.per_tokenizer.<tok>.global.exact_match_rate` | higher |
| Character error rate | (under `reconstruction_fidelity`) | `.per_tokenizer.<tok>.global.mean_cer` | lower |
| UNK token rate | (under `reconstruction_fidelity`) | `.per_tokenizer.<tok>.global.unk_token_rate` | lower |
| Whitespace fidelity | (under `reconstruction_fidelity`) | `.per_tokenizer.<tok>.global.whitespace_fidelity` | higher |
| Token UTF-8 completeness rate | `utf8_token_integrity` | `.per_tokenizer.<tok>.global.completeness_rate` | higher |
| Character boundary crossing rate | (under `utf8_token_integrity`) | `.per_tokenizer.<tok>.global.boundary_crossing_rate` | lower |
| Character boundary split rate | (under `utf8_token_integrity`) | `.per_tokenizer.<tok>.char_split.global.split_rate` | lower |
| AST leaf-node boundary alignment | `ast_boundary_alignment` | `.per_tokenizer.<tok>.global.full_alignment_rate` | higher |
| Identifier fragmentation | `identifier_fragmentation` | `.per_tokenizer.<tok>.global.fragmentation_rate` | lower |
| Indentation depth correlation | `indentation_consistency` | `.per_tokenizer.<tok>.global.depth_proportionality_correlation` | higher |
| Tokenizer fairness Gini | `tokenizer_fairness_gini` | `.per_tokenizer.<tok>.global.gini_coefficient` | lower |
| Lorenz curve | (under `tokenizer_fairness_gini`) | `.per_tokenizer.<tok>.lorenz_curve` | n/a, a curve |
| Encoding speed | `encoding_speed` | `.per_tokenizer.<tok>.global.mean_ms` | lower |

"Direction" gives the better direction for the quantity as defined.
"neither" marks a metric that describes a tokenizer without ranking it: a longer
mean token or a higher type-token ratio is not better or worse on its own. Plot
titles show an arrow for the subset of metrics listed in
`METRIC_BETTER_DIRECTION` (`tokenizer_analysis/visualization/plots.py`); metrics
absent from that map get no arrow.

Two more top-level keys are not metrics. `run_metadata` is described under
[Provenance](#provenance). `grouped_analysis` is written when
`--run-grouped-analysis` is passed, and holds the metrics recomputed within each
language group: `grouped_analysis.<group type>.<group name>.<metric>`. In
`analysis_results.json` it is written as an empty object, because the slimming
step recognizes only per-metric shapes; the populated version is in
`analysis_results_full.json`, so pass `--save-full-results` to read it.

### JSON Results Schema

`analysis_results.json` is always written with a slimmed schema. Pass
`--save-full-results` to also write `analysis_results_full.json` with all
computed data.

Three things hold for every metric, and the test suite asserts each of them
over the whole file rather than a sample:

- `per_tokenizer.<tok>.global` is the headline block. There are no exemptions.
  `token_length` and `encoding_speed` carry one that duplicates a block they
  already publish, because an exception in the schema costs a reader more than a
  duplicated number does.
- `metadata.aggregation` says which average `global` reports: `micro_pooled`,
  `macro_languages`, `ratio_of_sums` or `set_union`. This matters because
  `global` means different things in different metrics, and on a parallel corpus
  where every language holds the same number of lines micro and macro agree, so
  the difference only shows on an unequal corpus.
- Every `per_language` entry carries a `count`, in the unit named by
  `metadata.count_unit`, so a consumer can re-derive the other weighting.
  `tokenizer_fairness_gini` and `morphscore` are the exception: their unit is
  languages, so the count would be 1 for every entry, and `metadata` says so.

Most metrics follow this layout:

```json
{
  "<metric_name>": {
    "per_tokenizer": {
      "<tokenizer_name>": {
        "global": {},
        "per_language": {"<lang_code>": {}}
      }
    },
    "per_language": {"<lang_code>": {"<tokenizer_name>": "<value>"}},
    "metadata": {}
  }
}
```

- **`per_tokenizer.<tok>.global`**: the aggregate for this tokenizer. Stats dicts
  hold `mean`, `std`, `median`, `count`; structured dicts vary by metric.
- **`per_tokenizer.<tok>.per_language`**: the per-language breakdown for this
  tokenizer.
- **`per_language`** (top level): per-language values keyed by language then
  tokenizer, where the raw data has them.
- **`metadata`**: the metric's configuration and data provenance, where it has
  any.

Three metrics depart from that layout. All three still have `global` and
`metadata.aggregation`; what differs is the rest of the shape.

| Metric | How it differs |
|---|---|
| `token_length` | `global` holds `count`, `mean`, `median`, `std`. Three sibling blocks carry the same four stats separately: `character_length`, `byte_length`, `primary_length`. No `per_language` |
| `encoding_speed` | `global` holds `mean_ms`, `total_s`, `num_samples`, duplicating the same three fields already published directly under `per_tokenizer.<tok>`. No `per_language` |
| `reconstruction_fidelity` | `per_domain` in place of `per_language`, because it also runs on code and math |

`operator_isolation_rate` holds `global`, `per_language` **and** `by_domain`,
the last splitting the pooled global into `prose`, `code` and `math`. Its
`per_language` keys natural languages (`arb_Arab`) and programming languages
(`code:bash`) in one dict.

The slimmed file omits `pairwise_comparisons`, `summary`, `per_category`
breakdowns, and the derivable stat fields `sum`, `std_err`, `min` and `max`. The
full results file adds `per_category` for the metrics that have category
breakdowns (AST node types, operator types). `bigram_entropy` also holds a
top-level `reference_definition` block with the same measurement under the
reference normalizer and aggregation.

Since 1.0.2 the slim file is a strict projection of the full one: every leaf
sits at the same key path with the same value in both, and slimming only
deletes keys. Renaming happens before either file is written, so a path read
off one holds in the other. Earlier versions renamed while selecting, which
left the two files impossible to cross-reference.

#### `null` means not measured

A value that could not be computed is `null`, never `0.0`. This matters because
`0.0` is a legal value for most of these metrics, so a zero would be
indistinguishable from a real measurement. `count` and `sum` stay numeric.

A field is `null` when:

- A rate has no denominator: no operator, no multi-byte character, no digit span
  of that length in the evaluated text.
- A metric needs at least two languages and one was given. On a single corpus,
  `tokenizer_fairness_gini.per_tokenizer.<tok>.global.gini_coefficient` is
  `null` with the reason in a sibling `warning` field, and
  `vocabulary_utilization.per_tokenizer.<tok>.per_language_cov` is `null` and
  omitted from its plot.
- A domain holds nothing of the relevant kind.
- The CER time budget was exceeded. `reconstruction_fidelity`'s `mean_cer` and
  `whitespace_fidelity` are `null` for that tokenizer, and the log names it and
  the projected time. `--cer-time-budget 0` removes the cap.

Consumers doing arithmetic on these fields need a `None` check.

#### Provenance

Every results file holds a top-level `run_metadata` block: package version, a
UTC `timestamp_utc`, git commit, the config files read and their hashes, an
identifier per tokenizer, a digest of the corpus, sample count, the code-corpus
caps in force, which metric families were disabled, and under `arguments` every
command-line argument the caller changed from its default.

A tokenizer loaded from a local file is recorded with a `sha256_16` of that
file. One named by a Hugging Face Hub id has no local file to hash, so it is
recorded with `hub_revision`, the commit sha of the cached snapshot that was
loaded. Without it, a retokenization published upstream would move every number
in the file with nothing saying so. All nine tokenizers in
`benchmarks/open_source` are Hub-loaded and carry `hub_revision`.

`corpus.digest` gives each language's path, byte count and a short hash, plus
whatever the fetch script recorded about the dataset revision it read. The
config hashes say which corpus was requested; the digest says what the files
held, so two runs over different snapshots of one corpus under one config are
distinguishable after the fact. Paths under the working directory are recorded
relative to it, so a committed results file does not name whoever produced it. Arguments that only decide where
output goes are left out, so two runs writing to different directories do not
look like different measurements. When two results files disagree,
`run_metadata` identifies whether the tokenizer, the corpus, the settings or the
toolkit version changed.

`git_tree_modified` sits beside `git_commit`. It is `true` when the working tree
had uncommitted changes at run time, in which case the commit hash does not
describe the code that ran.

On the `--tokenized-data-file` path, `run_metadata.tokenizers` records the
vocabulary dumps named by the cache, each tagged
`"source": "vocabulary file from --tokenized-data-config"`, with no
`sha256_16`, because no tokenizer file is read on that path.

#### Metrics reported under another metric

Six metrics restate a measurement another metric already publishes, four of them
as exact algebraic identities. They are written as a field of the metric that
owns the measurement rather than as separate top-level keys, so the file does not
present one number twice as if it were two pieces of evidence. Each primary
records the merge and its evidence under `metadata.merged_metrics`.

| Reported under | Field | Relationship |
|---|---|---|
| `compression_rate` | `tokens_per_line` | product is exactly 1 |
| `vocabulary_utilization` | `type_token_ratio` | TTR is utilization rescaled by vocab size over token count |
| `renyi_efficiency` | `unigram_distribution` | unigram entropy is the unnormalized numerator of `renyi_1.0` |
| `utf8_token_integrity` | `char_split` | the same events counted from the character side |
| `tokenizer_fairness_gini` | `lorenz_curve` | `1 - 2*area(lorenz)` is the Gini coefficient |
| `three_digit_boundary_alignment` | `split_variability` | Spearman -0.992 between pooled `entropy_short` and `avg_f1` |

## Performance

### Encoding

Encoding is single-threaded: every combination of tokenizer, language and sample
is processed sequentially, so the number of encode calls scales as
O(N x L x M) for N tokenizers, L languages and M samples per language.

The results file reports what encoding actually cost on the run you just did:
`encoding_speed.per_tokenizer.<tok>.total_s` is the total seconds spent encoding
for that tokenizer and `mean_ms` the mean per sample. Read those from a small
run and scale by N, L and M to size a large one, rather than relying on a figure
measured on other hardware with other tokenizers.

`--samples-per-lang N` (default 2000) is the direct knob.
`--save-tokenized-data` and `--tokenized-data-file` remove the cost entirely for
repeat runs; see [Pre-tokenized data cache](#pre-tokenized-data-cache).

### Reconstruction fidelity

The reconstruction metrics decode every tokenized text back to a string and
compare it to the original. `exact_match_rate` and `unk_token_rate` are linear
in text length. `mean_cer` is not: it runs a Levenshtein dynamic program whose
cost is the product of the two lengths after the common prefix and suffix are
stripped, so a tokenizer that round-trips exactly is cheap and one that diverges
early is not.

CER is therefore capped per tokenizer by `--cer-time-budget` (default 10
seconds): after a warmup the total is extrapolated, and if the projection
exceeds the budget, `mean_cer` and `whitespace_fidelity` are skipped for that
tokenizer and reported as `null`. On the bundled demo this happens to
`unigramlm` and not to `bpe`.

`--no-reconstruction` skips the group outright. `--cer-time-budget 0` removes
the cap and measures CER however long it takes.

### Pre-tokenized data cache

Encode once, then iterate on metrics and plots without re-encoding.

```bash
# Step 1: encode and save (slow, once)
uv run tokenizer-analysis \
  --tokenizer-config tokenizers.json --language-config languages.json \
  --save-tokenized-data --tokenized-data-output-path results/tokenized_data.pkl

# Step 2: replay the cache (fast, repeat as needed)
uv run tokenizer-analysis \
  --tokenized-data-file results/tokenized_data.pkl \
  --tokenized-data-config results/tokenized_data_config.json \
  --language-config results/tokenized_data_language_config.json
```

Beside the pickle, step 1 writes a `<stem>_config.json` naming the per-tokenizer
vocabulary dumps, one `<tokenizer>_vocab.txt` dump per tokenizer, and a
`<stem>_language_config.json` copy of the language metadata. Step 2 needs all of
them. `--language-config` is required on the replay path and the run aborts
without it: the pickle holds token ids and language labels but no groupings, and
replaying against a different language config would relabel the data. Step 1
logs the exact replay command for the files it just wrote.

The replay path differs from a fresh run in two ways:

- `run_metadata.tokenizers` records the vocabulary dumps named by the cache,
  each tagged `"source": "vocabulary file from --tokenized-data-config"`, with no
  tokenizer-file hash, because no tokenizer file is read.
- The metrics that need raw `encode()` calls are absent from the results:
  `ast_boundary_alignment`, `identifier_fragmentation`,
  `indentation_consistency` and `encoding_speed`. `PreTokenizedDataTokenizer`
  reports `can_encode() == False`, which is what skips them. MorphScore is a
  separate case: it needs `get_underlying_tokenizer()`, which the pre-tokenized
  wrapper does not provide, so the `morphscore` key is written with an `error`
  field per tokenizer rather than a score.

For manually prepared pre-tokenized data, supply a pickle or JSON dict mapping
tokenizer names to lists of `TokenizedData` objects, a JSON config pointing to
vocabulary files, and line-by-line vocabulary text files.

### Quick-iteration recipe

For fast development iterations, minimize samples and disable the expensive
metric groups:

```bash
uv run tokenizer-analysis \
  --tokenizer-config tokenizers.json --language-config languages.json \
  --samples-per-lang 100 \
  --no-reconstruction --no-plots --no-code-ast --no-utf8-integrity --no-digit-boundary
```

## Troubleshooting

**`No module named 'morphscore'`**: initialize submodules, then install
MorphScore into the project environment:
`git submodule update --init --recursive && uv pip install -e ./morphscore`. If
it worked before and stopped working, a later `uv sync` removed it; see
[Optional extras](#optional-extras).

**`Unknown tokenizer class`**: the available classes are listed under
[Tokenizer Configuration](#tokenizer-configuration), plus any you register at
runtime with `register_tokenizer_class()`.

**`FileNotFoundError`**: a relative `data_path` in a language config is resolved
against the package root, not your working directory, while `--input` and a
relative tokenizer `path` are resolved against your working directory. See
[How `data_path` is resolved](#how-data_path-is-resolved).

**`Cannot read the parquet file`**: install the extra with
`uv sync --extra parquet`, naming any other extras in the same command.

**`_tkinter.TclError: no display name`**: set `export MPLBACKEND=Agg` before
running on a headless server.

**A code language is missing from the results**: three of the 19 configured
languages are excluded by design, and a grammar that crashes or exceeds
`TOKEVAL_PARSE_TIMEOUT_S` is reported as unmeasured and named in the log. See
[METRICS.md](METRICS.md#code-tokenization-metrics).

## CLI reference

Every flag `tokenizer-analysis` accepts.

### Input

| Flag | Description |
|------|-------------|
| `--tokenizer-config FILE` | JSON file with tokenizer configurations |
| `--input PATH` | Analyze a single corpus: one file or a directory. Exclusive with `--language-config` and with `--use-sample-data` |
| `--input-label NAME` | Name for the `--input` corpus in the results (default: the path stem) |
| `--language-config FILE` | JSON file with languages and analysis groups |
| `--measurement-config FILE` | JSON file with the text measurement method |
| `--use-sample-data` | Run the built-in demo. Exclusive with `--tokenizer-config`, `--language-config`, `--input` and `--measurement-config` |
| `--samples-per-lang N` | Text samples per language (default: 2000) |
| `--pairwise TOK1 TOK2` | Restrict the run to two named tokenizers |
| `--filter-script-family FAMILY` | Restrict to one script family from `analysis_groups` |
| `--filter-resource-level NAME` | Restrict to one resource level from `analysis_groups` |

### Metric selection

| Flag | Description |
|------|-------------|
| `--no-reconstruction` | Skip the decode round trip: `exact_match_rate`, `mean_cer`, `unk_token_rate`, `whitespace_fidelity` |
| `--cer-time-budget SECONDS` | Cap on CER computation per tokenizer; 0 disables the cap (default: 10) |
| `--no-digit-boundary` | Skip `three_digit_boundary_alignment`, digit split variability, `numeric_magnitude_consistency` and `operator_isolation_rate` |
| `--math-data FILE` | Math-rich text file (`.txt`/`.json`) for the digit metrics |
| `--use-builtin-math-data` | Use the bundled math corpus for the digit metrics. Ignored when `--math-data` is also given |
| `--no-code-ast` | Skip `ast_boundary_alignment`, `identifier_fragmentation` and `indentation_consistency`, including their synthetic-code fallback. The `code` domain of `operator_isolation_rate` still runs |
| `--code-ast-config FILE` | JSON mapping languages to code paths for AST analysis |
| `--max-code-files-per-lang N` | Cap on code files loaded per language from `--code-ast-config` paths (default: 0, no cap) |
| `--max-code-file-chars N` | Truncate each loaded code file to this many characters before it reaches the code metrics (default: 0, no cap) |
| `--no-utf8-integrity` | Skip `utf8_token_integrity` |
| `--morphscore` | Enable MorphScore with default settings |
| `--morphscore-config FILE` | Custom MorphScore configuration |
| `--morphscore-data-dir DIR` | Where MorphScore datasets live (default: `morphscore_data`) |
| `--no-plots` | Skip all matplotlib rendering |

### Output

| Flag | Description |
|------|-------------|
| `--output-dir DIR` | Output directory (default: `results/`) |
| `--save-full-results` | Also write `analysis_results_full.json` with every computed field |
| `--verbose` | Detailed console output |
| `--run-grouped-analysis` | Also compute metrics within each language group from `analysis_groups` |
| `--per-language-plots` | Per-language grouped bar charts |
| `--faceted-plots` | One subplot per tokenizer with shared y-axis |
| `--no-global-lines` | Hide global average reference lines in plots |
| `--generate-latex-tables` | Generate LaTeX tables into `<output-dir>/latex_tables` |
| `--latex-table-types ...` | Which LaTeX tables to emit: `basic`, `information`, `comprehensive` (default: `basic comprehensive`) |
| `--custom-latex-config FILE` | JSON config for a custom LaTeX table, written into `<output-dir>` with no subdirectory |
| `--latex-output-dir DIR` | Override the destination for both LaTeX paths |

### Tokenization cache

| Flag | Description |
|------|-------------|
| `--save-tokenized-data` | Cache the tokenization for reuse |
| `--tokenized-data-output-path PATH` | Where `--save-tokenized-data` writes (default: `<output-dir>/tokenized_data.pkl`) |
| `--tokenized-data-file FILE` | Replay a cached tokenization instead of encoding. Requires `--language-config` |
| `--tokenized-data-config FILE` | The vocabulary map that accompanies a cache |

## Contributing

### Repository layout

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
[Metrics reported under another metric](#metrics-reported-under-another-metric)
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

### Adding new tokenizers

Subclass `TokenizerWrapper` from `tokenizer_analysis.core.tokenizer_wrapper` and
implement the required abstract methods. Then register it so the config system
can instantiate it by name.

#### Required methods (abstract)

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

#### Optional overrides

These have working defaults but can be overridden for better results:

| Method | Default behaviour | Why override |
|--------|------------------|--------------|
| `convert_ids_to_tokens(ids) -> List[str]` | Reverses `get_vocab()`. | Faster or more accurate when the underlying library has a direct lookup (for example `id_to_token`). |
| `encode_with_offsets(text) -> (List[int], Optional[List[Tuple[int,int]]])` | Returns `(self.encode(text), None)`. | Provide `(start_char, end_char)` offsets per token for exact source-to-token mapping. Without this, the code metrics fall back to greedy character alignment, which fails for tokenizers that strip whitespace from tokens (for example custom BPE with a `Whitespace` pre-tokenizer), and the operator metric skips the document. HuggingFace `tokenizers` and SentencePiece both expose offsets natively. |
| `get_underlying_tokenizer()` | Returns `None`. | Expose the raw HuggingFace tokenizer object for consumers like MorphScore, which only works with HF tokenizers. |
| `get_unk_token_id() -> Optional[int]` | Returns `None`. | Enables `unk_token_rate`. |

#### Minimal example

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

### Adding new metrics

1. Inherit from `BaseMetrics` in `tokenizer_analysis/metrics/base.py`.
2. Implement the `compute()` method.
3. Register it in `main.py`.
4. Add a row to the slimming step in `cli/run_analysis.py`, or the metric will
   be written as an empty object in `analysis_results.json`.

### Submitting changes

1. Fork the repository.
2. Create a feature branch.
3. Ensure the existing tests pass.
4. Submit a pull request.

`SECURITY.md` describes how to report a vulnerability, and
`CODE_OF_CONDUCT.md` applies to participation in the project.

## A worked comparison

[benchmarks/open_source/REPORT.md](benchmarks/open_source/REPORT.md) measures
nine widely used tokenizers on 13 FLORES+ languages, 1500 source files across
15 programming languages, and the bundled math corpus. It is what a full run
looks like, and it is regenerated by one command:

```bash
bash benchmarks/open_source/run.sh
```

Everything it needs is in that directory: the tokenizer config, the code-corpus
fetch script, the committed `analysis_results.json`, and the script that renders
the report from it. No number in the report is typed by hand, so the tables
cannot drift from the results file, and the results file records the commit and
a hash of every input.

## Other documents

| File | What it holds |
|---|---|
| [METRICS.md](METRICS.md) | The definition, worked example and caveats for every metric |
| [benchmarks/open_source/REPORT.md](benchmarks/open_source/REPORT.md) | Nine open-source tokenizers measured on the full metric set |
| [MIGRATION.md](MIGRATION.md) | Breaking changes from `tokenizer-analysis-suite` to 1.0.0 and their replacements. The CLI's error messages point here |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [SECURITY.md](SECURITY.md) | How to report a vulnerability |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Expectations for participation |
| [CITATION.cff](CITATION.cff) | Machine-readable citation metadata |

## License

MIT. See [LICENSE](LICENSE), and [NOTICE](NOTICE) for third-party attributions.

## Citation

```bibtex
@software{meister_tokeval_2026,
  title   = {TokEval: intrinsic evaluation metrics for tokenizers},
  author  = {Meister, Clara},
  year    = {2026},
  version = {1.0.2},
  url     = {https://github.com/cimeister/tokenizer-intrinsic-evals}
}
```
