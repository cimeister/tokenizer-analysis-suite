# Configuration

This page describes the `tokenizer-analysis` command-line flags and the JSON
configuration files it reads: tokenizer configs, language configs,
measurement configs, MorphScore configs and code AST configs. See
[../README.md](../README.md) for installation, the quick start and the
bundled demo.

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
| `--operator-prose-domain` | Score the main corpus as a `prose` domain of `operator_isolation_rate`. Off by default: an operator is a code construct, and the pattern matches a hyphen, a slash and an exclamation mark |
| `--code-ast-config FILE` | JSON mapping languages to code paths for AST analysis |
| `--max-code-files-per-lang N` | Cap on code files loaded per language from `--code-ast-config` paths (default: 0, no cap) |
| `--max-code-file-chars N` | Truncate each loaded code file to this many characters before it reaches the code metrics (default: 0, no cap) |
| `--no-utf8-integrity` | Skip `utf8_token_integrity` |
| `--morphscore` | Enable MorphScore with default settings |
| `--morphscore-config FILE` | Custom MorphScore configuration |
| `--morphscore-data-dir DIR` | Where MorphScore datasets are stored (default: `morphscore_data`) |
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
(see [Adding new tokenizers](EXTENDING.md#adding-new-tokenizers)).

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
decided per file, and it is the unit every per-document metric divides by. A
file of prose paragraphs and a file of one sentence per line are read
differently on purpose.

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
list, so a `.jsonl` file is read when named directly and skipped when it is in
a scanned directory. A directory holding only `.jsonl` files loads zero texts.

#### Metric families and the data they need

| Metric family | Computed on | Flag that supplies it | Without the flag |
|---|---|---|---|
| Compression, fertility, token length, vocabulary utilization, Rényi efficiency, bigram and trigram entropy, Gini, UTF-8 integrity, reconstruction fidelity, encoding speed | the main corpus | `--input`, `--language-config` or `--use-sample-data` | the run aborts with an error naming the three options. There is no fallback to demo data |
| The three digit metrics: `three_digit_boundary_alignment`, digit split variability, `numeric_magnitude_consistency` | dedicated math texts | `--math-data FILE` or `--use-builtin-math-data` | computed on the main corpus instead, and the run prints a warning naming all three. `--no-digit-boundary` turns them off |
| `operator_isolation_rate`, `math` domain | dedicated math texts | `--math-data FILE` | the bundled `sample_data/math_samples.json`, which is also what `--use-builtin-math-data` names. This domain never falls back to the main corpus |
| `operator_isolation_rate`, `prose` domain | the main corpus | `--operator-prose-domain`, plus `--input`, `--language-config` or `--use-sample-data` | not scored at all. This domain is off by default |
| The three AST metrics: `ast_boundary_alignment`, `identifier_fragmentation`, `indentation_consistency` | dedicated source-code snippets | `--code-ast-config FILE` | computed on built-in synthetic code samples, and the run prints a warning naming all three. `--no-code-ast` turns them off |
| `operator_isolation_rate`, `code` domain | dedicated source-code snippets | `--code-ast-config FILE` | the bundled `sample_data/code_samples.json`. This domain runs under `--no-code-ast` as well; only `--no-digit-boundary`, which drops `operator_isolation_rate` entirely, turns it off |
| MorphScore | MorphScore datasets | `--morphscore` or `--morphscore-config`, plus `--morphscore-data-dir` | not computed |
| Cross-language metrics (`tokenizer_fairness_gini`, `per_language_cov`) | at least 2 languages | `--language-config` with 2 or more entries | computed as `null` with a stated reason |

`operator_isolation_rate` logs its three sources on one line each run:
`Operator isolation domains: math=..., code=...`, naming the domains that will
run.

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
[Optional extras](#optional-extras) below for the download command. The
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

## Optional extras

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

## Pre-tokenized data cache

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

## Quick-iteration recipe

For fast development iterations, minimize samples and disable the expensive
metric groups:

```bash
uv run tokenizer-analysis \
  --tokenizer-config tokenizers.json --language-config languages.json \
  --samples-per-lang 100 \
  --no-reconstruction --no-plots --no-code-ast --no-utf8-integrity --no-digit-boundary
```
