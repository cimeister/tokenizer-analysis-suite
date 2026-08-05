# TokEval

A toolkit for computing intrinsic quality metrics for tokenizers across natural
language, code, and math.

This supersedes `tokenizer-analysis-suite`. The install name changed to
`tokenizer-intrinsic-evals`; the import name is still `tokenizer_analysis`. The
prior suite is preserved on the `legacy-suite` branch and the
`legacy-suite-final` tag. Several metrics changed what they compute, so numbers
from the old suite are not comparable with numbers from this one:
[CHANGELOG.md](CHANGELOG.md) lists every such change with its measured effect.

## Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [What it measures](#what-it-measures)
- [Running it on your own data](#running-it-on-your-own-data)
- [Reading the output](#reading-the-output)
- [The other two commands](#the-other-two-commands)
- [A worked comparison](#a-worked-comparison)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Documents](#documents)

Reference material is in [docs/](docs/): [METRICS.md](docs/METRICS.md) defines
every metric, [CONFIGURATION.md](docs/CONFIGURATION.md) is the flag and config
reference, [OUTPUT.md](docs/OUTPUT.md) describes the results file, and
[EXTENDING.md](docs/EXTENDING.md) covers the Python API and adding your own
tokenizers and metrics.

## Install

Python 3.10 or newer, tested on Linux.

```bash
git clone https://github.com/cimeister/tokenizer-intrinsic-evals.git
cd tokenizer-intrinsic-evals
uv sync
```

`uv run` puts the console scripts on the path; activate the venv instead if you
prefer, and drop the prefix. That is enough for the Quick Start.

Not published to PyPI. To use it as a dependency rather than working in the
checkout, `pip install git+https://github.com/cimeister/tokenizer-intrinsic-evals.git`.
The demo data and the example configs are in the checkout and not in the
installable package, so a checkout is needed for those.

MorphScore, parquet input and the FLORES+ fetcher are optional extras with an
install order that matters: see
[Optional extras](docs/CONFIGURATION.md#optional-extras).

## Quick Start

Compare two tokenizers on one corpus. `gpt2` and `xlm-roberta-base` download
from the Hugging Face Hub on first use; both are public.

```bash
cat > corpus.txt <<'CORPUS'
The quick brown fox jumps over the lazy dog.
Tokenizers differ most on text they were not trained on.
def compute_total(items): return sum(i.price for i in items)
The invoice totalled 1234567 euros on 2024-03-15.
CORPUS

cat > my_tokenizers.json <<'TOKENIZERS'
{
  "gpt2":  {"class": "huggingface", "path": "gpt2"},
  "xlm-r": {"class": "huggingface", "path": "xlm-roberta-base"}
}
TOKENIZERS

uv run tokenizer-analysis --tokenizer-config my_tokenizers.json --input corpus.txt
```

Results are written to `results/analysis_results.json`, with plots beside it.

## What it measures

Eighteen metrics. One line each below; the definition, worked example and
caveats for each are in [docs/METRICS.md](docs/METRICS.md), and the exact key
path for each headline value is in
[Metric names and results keys](docs/METRICS.md#metric-names-and-results-keys).

| Metric | Measures | Better |
|---|---|---|
| `compression_rate` | text units per token | higher |
| `fertility` | tokens per word or character | lower |
| `token_length` | mean token size in bytes and characters | neither |
| `vocabulary_utilization` | share of the declared vocabulary the corpus used | higher |
| `renyi_efficiency` | how evenly token probability mass is spread | higher |
| `bigram_entropy` | how predictable each token's successor is | higher |
| `trigram_entropy` | the same over a two-token context | higher |
| `morphscore` | whether token boundaries fall on morpheme boundaries | higher |
| `three_digit_boundary_alignment` | whether numbers split on place-value boundaries | higher |
| `numeric_magnitude_consistency` | whether tokens per digit stay stable across magnitudes | lower |
| `operator_isolation_rate` | whether operators are their own tokens | higher |
| `reconstruction_fidelity` | whether decoding restores the input exactly | higher |
| `utf8_token_integrity` | whether tokens are complete UTF-8 sequences | higher |
| `ast_boundary_alignment` | whether token boundaries fall on syntax-tree leaves | higher |
| `identifier_fragmentation` | how many tokens an identifier costs | lower |
| `indentation_consistency` | whether whitespace tokens scale with indent depth | higher |
| `tokenizer_fairness_gini` | how unevenly encoding cost falls across languages | lower |
| `encoding_speed` | wall-clock cost of the run, not a quality measure | lower |

"Better" gives the direction for the quantity as defined. "neither" marks a
metric that describes a tokenizer without ranking it.

## Running it on your own data

`--input` takes one file (`.txt`, `.json`, `.jsonl`, `.parquet`) or a directory
of them. A `.txt` file is one document per line. For several corpora at once,
and for the cross-lingual metrics, use `--language-config` instead. Both are
described in [Data Configuration](docs/CONFIGURATION.md#data-configuration).

A tokenizer `path` is either a Hub model id or a local `tokenizer.json`. Ten
tokenizer classes are supported; see
[Tokenizer Configuration](docs/CONFIGURATION.md#tokenizer-configuration).

### Three defaults to change before publishing a number

Three things stay at their defaults in the Quick Start command. That is fine
for a first look and not for a result you intend to report.

```bash
uv run tokenizer-analysis \
    --tokenizer-config my_tokenizers.json \
    --language-config my_languages.json \
    --code-ast-config my_code.json \
    --use-builtin-math-data \
    --cer-time-budget 0
```

- **`--code-ast-config`** supplies real source code. Without it the code
  metrics run on built-in synthetic samples and the run warns. The two are not
  comparable.
- **`--use-builtin-math-data`**, or `--math-data FILE`, points the three digit
  metrics at a corpus with long numbers in it. Without either they run on
  whatever numbers the prose happens to contain, which on the bundled corpus is
  1797 digit spans of which 6 exceed 4 digits, so the place-value boundaries at
  L-6 and L-9 are almost never exercised.
- **`--cer-time-budget 0`** removes the cap on the character error rate, so
  `mean_cer` and `whitespace_fidelity` are measured rather than reported as
  `null` for slow tokenizers.

### The evaluation corpus

The configs in `configs/` and the `--use-sample-data` demo read
`parallel/<iso639-3>_<Script>.txt`. Those files are not in this repository.
They come from FLORES+, which is CC-BY-SA 4.0 and is fetched rather than
redistributed here. Cite NLLB Team et al., Nature 630 (2024) if you use it.

```bash
uv pip install datasets            # or: uv sync --extra flores
hf auth login                      # FLORES+ is gated, approval is automatic
uv run python scripts/fetch_flores.py                              # 13 languages
uv run python scripts/fetch_flores.py --all                        # every language
```

A run that names a missing file aborts and repeats that command. It never
proceeds on a smaller corpus than the config asked for. Nothing in the library
requires FLORES+: `--input` and `--language-config` take your own corpus.

## Reading the output

```
results/
├── analysis_results.json       # every metric, per tokenizer and per language
├── analysis_results_full.json  # with --save-full-results
└── *.svg                       # one chart per metric, unless --no-plots
```

Every metric is written as `<metric>.per_tokenizer.<tok>.global` for the
headline value and `.per_language.<lang>` for the breakdown, with
`metadata.aggregation` naming which average the `global` is. A value that could
not be computed is `null`, never `0.0`.

[docs/OUTPUT.md](docs/OUTPUT.md) describes the schema, the aggregation labels,
the null convention and the `run_metadata` provenance block.

## The other two commands

**`tokenizer-visualize`** renders token boundaries on source text, for
inspecting how tokenizers split code, math and multilingual content.

```bash
uv run tokenizer-visualize --tokenizer-config configs/sample_tokenizers.json
```

**`tokenizer-sanity-check`** runs 16 checks against one tokenizer: byte
coverage, whitespace and digit handling, special tokens, determinism, Unicode
normalization, vocabulary integrity and reachability. A failing check sets a
non-zero exit code, so it can gate a tokenizer before a full run.

```bash
uv run tokenizer-sanity-check huggingface:tokenizers/bpe.json
```

The bundled demo tokenizers fail this check on purpose: `bpe.json` covers 94 of
256 byte values. They exist so the check has something to report.

Both commands' flags are in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md#cli-reference).

## A worked comparison

[benchmarks/open_source/REPORT.md](benchmarks/open_source/REPORT.md) measures
nine widely used tokenizers on 13 FLORES+ languages, 1500 source files across
15 programming languages, and the bundled math corpus. One command regenerates
it:

```bash
bash benchmarks/open_source/run.sh
```

No number in that report is typed by hand, and its results file records the
commit and a hash of every input.

## Troubleshooting

**`No module named 'morphscore'`**: initialize submodules, then install
MorphScore into the project environment:
`git submodule update --init --recursive && uv pip install -e ./morphscore`. If
it worked before and stopped, a later `uv sync` removed it; see
[Optional extras](docs/CONFIGURATION.md#optional-extras).

**`Unknown tokenizer class`**: the available classes are listed under
[Tokenizer Configuration](docs/CONFIGURATION.md#tokenizer-configuration), plus
any you register with `register_tokenizer_class()`.

**`FileNotFoundError`**: a relative `data_path` in a language config resolves
against the package root, while `--input` and a relative tokenizer `path`
resolve against your working directory. See
[How `data_path` is resolved](docs/CONFIGURATION.md#how-data_path-is-resolved).

**`Cannot read the parquet file`**: `uv sync --extra parquet`, naming any other
extras in the same command.

**`_tkinter.TclError: no display name`**: `export MPLBACKEND=Agg` before
running on a headless server.

**A code language is missing from the results**: three of the 19 configured
languages are excluded by design, and a grammar that crashes or exceeds
`TOKEVAL_PARSE_TIMEOUT_S` is reported as unmeasured and named in the log. See
[docs/METRICS.md](docs/METRICS.md#code-tokenization-metrics).

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Ensure the existing tests pass: `uv run pytest tokenizer_analysis/tests -q`.
4. Submit a pull request.

Adding a tokenizer class, adding a metric, and the repository layout are
covered in [docs/EXTENDING.md](docs/EXTENDING.md).

## Documents

| File | What it holds |
|---|---|
| [docs/METRICS.md](docs/METRICS.md) | The definition, worked example and caveats for every metric |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Every CLI flag and every config file format |
| [docs/OUTPUT.md](docs/OUTPUT.md) | The results file: schema, aggregation, null convention, provenance |
| [docs/EXTENDING.md](docs/EXTENDING.md) | Python API, repository layout, adding tokenizers and metrics |
| [benchmarks/open_source/REPORT.md](benchmarks/open_source/REPORT.md) | Nine open-source tokenizers measured on the full metric set |
| [CHANGELOG.md](CHANGELOG.md) | Changes from `tokenizer-analysis-suite`, breaking ones first |
| [CITATION.cff](CITATION.cff) | Machine-readable citation metadata |

## License

MIT. See [LICENSE](LICENSE). The evaluation corpora and the tokenizers named in
the example configs are downloaded at run time under their own licenses and are
not redistributed here.

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
