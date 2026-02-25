# SwissAI TokEval
This is the library used by the Apertus tokenization team for intrinsic evaluation during tokenizer development


## Quick Start

Get up and running in 30 seconds:

```bash
# Clone and install
git clone https://github.com/swiss-ai/tokenizer-intrinsic-evals.git
cd tokenizer-intrinsic-evals
pip install -e .

# Run demo analysis with built-in sample data
python scripts/run_tokenizer_analysis.py --use-sample-data

# View results
open results/fertility.png  # Basic metric comparison chart
```

This will analyze two sample tokenizers (BPE and Unigram) across 5 languages and generate comparison plots.

## Adding Tokenizer Results
Use the following measurement config and language config for adding results to GitHub:

```bash
# Generate / update a local RESULTS.md
python scripts/run_tokenizer_analysis.py \
    --tokenizer-config configs/baseline_tokenizers.json \
    --language-config configs/core_lang_config.json \
    --measurement-config configs/text_measurement_config_lines.json \
    --verbose --run-grouped-analysis --per-language-plots --no-global-lines \
    --update-results-md --dataset flores_core

# Push results to GitHub
python scripts/update_remote.py
```

Specify the path to your tokenizer file in the JSON given to `--tokenizer-config` (see [Configuration Files](#configuration-files)).

## Setup

### Requirements
- Python 3.8+
- Git (for submodules)

### Full Installation
```bash
git clone https://github.com/swiss-ai/tokenizer-intrinsic-evals.git
cd tokenizer-intrinsic-evals
pip install -e .

# Optional: MorphScore morphological analysis
git submodule update --init --recursive
cd morphscore && pip install -e . && cd ..

# Optional: AST boundary analysis for code
pip install tree-sitter-language-pack
```

**MorphScore note**: Only `<ISO 639-3>_<script>` language codes are automatically mapped. Data files must be downloaded separately (see [MorphScore README](morphscore/README.md)) and placed in `morphscore_data/`.

## Usage

### Common CLI Options

| Flag | Description |
|------|-------------|
| `--tokenizer-config FILE` | JSON file with tokenizer configurations |
| `--language-config FILE` | JSON file with languages and analysis groups |
| `--measurement-config FILE` | JSON file with text measurement method |
| `--use-sample-data` | Use built-in demo data |
| `--output-dir DIR` | Output directory (default: `results/`) |
| `--verbose` | Detailed console output |
| `--no-plots` | Skip plot generation |
| `--save-full-results` | Save detailed JSON results |
| `--run-grouped-analysis` | Group analysis by script families / resource levels |
| `--per-language-plots` | Per-language grouped bar charts |
| `--faceted-plots` | One subplot per tokenizer with shared y-axis |
| `--filter-script-family FAMILY` | Filter languages by script family |
| `--morphscore` | Enable MorphScore analysis |
| `--morphscore-config FILE` | Custom MorphScore configuration |
| `--code-ast-config FILE` | JSON mapping languages to code paths for AST analysis |
| `--no-code-ast` | Skip AST boundary analysis |
| `--no-digit-boundary` | Skip math metrics (digit boundaries, operators) |
| `--no-utf8-integrity` | Skip UTF-8 character boundary integrity analysis |
| `--generate-latex-tables` | Generate LaTeX tables |
| `--update-results-md [PATH]` | Generate/update cumulative Markdown leaderboard |
| `--dataset NAME` | Dataset label for the results table |
| `--sort-results-by METRIC` | Sort results table by metric key |
| `--samples-per-lang N` | Text samples per language |
| `--save-tokenized-data` | Cache tokenized data for reuse |
| `--no-global-lines` | Hide global average lines in plots |

### Markdown Results Table

Generate a cumulative Markdown leaderboard that grows across successive runs. Each run merges new tokenizer rows into the existing table — previously evaluated tokenizers are preserved, re-evaluated ones are updated.

```bash
# Generate / update a local RESULTS.md
python scripts/run_tokenizer_analysis.py --use-sample-data --update-results-md --dataset flores

# Custom output path
python scripts/run_tokenizer_analysis.py --use-sample-data --update-results-md my_results.md
```

Each row is keyed by `tokenizer_name (user, dataset)` — different users or datasets produce separate rows, while re-running the same combination updates in place.

### Sharing Results via Git

Use `scripts/update_remote.py` to push results to a dedicated branch (default: `results`) without switching your branch or touching the working tree.

```bash
python scripts/update_remote.py                                # Push to origin/results
python scripts/update_remote.py --validate-local-results       # Validate format only
python scripts/update_remote.py --remove-my-results            # Remove your rows from remote
python scripts/update_remote.py --remove-my-results --all      # Remove from all RESULTS files
```

When multiple team members push, the remote file is fetched and merged first — rows from others are preserved.

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

Available classes: `"huggingface"`, `"custom_bpe"` (requires `vocab.json` + `merges.txt`), `"pretokenized"` (for pre-tokenized data).

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

For simple setups, `"languages"` can map codes directly to file paths: `{"en": "/path/to/data"}`.

### Text Measurement Configuration

Control how text "length" is measured for metric normalization via `--measurement-config`:

| Method | Key | Options | Default for |
|--------|-----|---------|-------------|
| Bytes | `"bytes"` | `byte_counting_method`: `"utf8"`, `"hf_tokenizer"` | Compression metrics |
| Characters | `"characters"` | — | — |
| Lines | `"lines"` | `line_counting_method`: `"python_split"`, `"regex"` | Gini metrics |
| Words | `"words"` | `word_counting_method`: `"whitespace"`, `"hf_whitespace"`, `"regex"` | Fertility |

Example:
```json
{
  "method": "lines",
  "line_counting_method": "python_split",
  "include_empty_lines": false
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

Requires languages in `<ISO 639-3>_<script>` format (e.g., `eng_Latn`). Override with `"language_subset"` in the config to bypass code mapping. Download datasets from [MorphScore README](morphscore/README.md).

### Code AST Configuration

Specify source code paths for AST boundary analysis via `--code-ast-config`:

```json
{
  "python": "/path/to/python/files/",
  "javascript": "/path/to/js/files.parquet",
  "java": "/path/to/java/dir/"
}
```

Supports 19 languages. Parquet files should have a `content` column; StarCoder metadata prefixes are stripped automatically. Without a config file, built-in synthetic code samples are used.

### Pre-tokenized Data

```bash
# Save tokenized data for reuse
python scripts/run_tokenizer_analysis.py --use-sample-data \
    --save-tokenized-data --tokenized-data-output-path my_data.pkl

# Reuse cached data (faster — no re-encoding)
python scripts/run_tokenizer_analysis.py \
    --tokenized-data-file my_data.pkl \
    --tokenized-data-config my_data_config.json
```

The save step auto-generates a config file and per-tokenizer vocabulary files. For manually prepared pre-tokenized data, provide a pickle/JSON dict mapping tokenizer names to lists of `TokenizedData` objects, a JSON config pointing to vocabulary files, and line-by-line vocabulary text files.

## Output Structure
```
results/
├── fertility_individual.svg         # Metric comparison charts
├── compression_rate_individual.svg
├── vocabulary_utilization_individual.svg
├── grouped_plots/                   # Cross-tokenizer comparisons
├── per-language/                    # Language-specific analysis
├── latex_tables/                    # Academic publication tables
├── RESULTS.md                       # Cumulative Markdown leaderboard
├── analysis_results.json            # Key metrics summary
├── analysis_results_full.json       # Detailed results (--save-full-results)
└── tokenized_data.pkl               # Cached data (--save-tokenized-data)
```

## Metrics

### Basic Tokenization Metrics
- **Compression Rate**: Text size (bytes/chars/lines) per token — measures encoding efficiency
- **Fertility**: Tokens per word/character — measures tokenization granularity
- **Token Length**: Average token size in bytes/characters
- **Type-Token Ratio**: Unique tokens / total tokens — measures vocabulary usage diversity

### Information-Theoretic Metrics
- **Renyi Entropy**: Information content at different alpha values — generalizes Shannon entropy
- **Vocabulary Utilization**: Fraction of vocabulary actually used
- **Average Token Rank**: Typical position of tokens within the frequency-ordered vocabulary

### Morphological Metrics
- **Boundary Precision/Recall**: How well tokens align with morpheme boundaries
- **MorphScore V2**: Advanced morphological evaluation ([Arnett et al. 2025](https://arxiv.org/abs/2507.06378))

### Mathematical Content Metrics

Evaluates tokenizer handling of mathematical expressions. Based on Singh & Strouse (2024, [arXiv:2402.14903](https://arxiv.org/abs/2402.14903)), who showed that right-to-left tokenization of numbers improved arithmetic accuracy by >22 percentage points. These metrics run on any text data containing numbers or operators. Disable with `--no-digit-boundary`.

#### Three-Digit Place-Value Boundary Alignment (F1)

Measures whether numbers are tokenized with right-aligned 3-digit groupings that match place-value structure (units, thousands, millions).

For each number, compares actual token boundaries against ideal boundaries at positions L-3, L-6, L-9 from the left. Reports precision, recall, and F1. Short numbers (<=3 digits) that remain single tokens score F1 = 1.0; short numbers needlessly split score F1 = 0.

**Example:** The number `1234567` has ideal boundaries at positions 1, 4 — yielding `1|234|567` (millions, thousands, units). A tokenizer producing `1|234|567` scores F1 = 1.0. One producing `12|345|67` scores F1 = 0.0 — it has three boundaries but none at the right positions. A short number like `42` kept as a single token scores F1 = 1.0 (no boundaries needed, none placed). But `42` split into `4|2` scores F1 = 0.0 — a boundary was placed where none was needed.

**Why it matters:** Singh & Strouse (2024) showed that right-to-left digit grouping improves arithmetic accuracy by ensuring corresponding digit positions across operands occupy consistent token positions.

#### Cross-Number Boundary Pattern Entropy

For numbers of the same digit length, measures Shannon entropy of the distribution of boundary patterns. Low entropy means the tokenizer uses a consistent splitting scheme; high entropy means chaotic splitting.

Entropy is computed on patterns pooled across languages, not averaged per-language. Reports normalized entropy, dominant pattern, and dominant frequency per digit-length bucket.

**Example:** A corpus contains three 5-digit numbers. If all are split as `XX|XXX` (pattern `{2}`), the entropy for the 5-digit bucket is 0.0 — perfectly consistent. If instead one is split `XX|XXX`, one as `X|XXXX`, and one as `XXX|XX`, there are three distinct patterns with equal frequency, giving normalized entropy of 1.0 — maximally chaotic. The first tokenizer has a learnable (if wrong) scheme; the second forces the model to handle every number as a special case.

**Why it matters:** A tokenizer with moderate F1 but low entropy has a consistent-but-wrong scheme (potentially fixable by retraining). Moderate F1 with high entropy indicates a deeper structural problem.

#### Numeric Magnitude Consistency

Tracks fertility-per-digit (tokens per digit) across digit lengths. Reports Spearman correlation, coefficient of variation, and linear fit (slope, R-squared) between digit length and mean token count.

**Example:** A tokenizer has memorized `0`-`999` as single vocabulary entries, so 1-digit numbers cost 1 token (1.0 tokens/digit), 2-digit numbers cost 1 token (0.5 tokens/digit), and 3-digit numbers cost 1 token (0.33 tokens/digit). Then at 4 digits, it fragments: `1234` -> `12|34` (0.5 tokens/digit). At 7 digits: `1234567` -> `123|45|67` (0.43 tokens/digit). The discontinuity between 3 and 4 digits — where fertility-per-digit jumps from 0.33 to 0.5 — shows up as a break in the linear fit and a low R-squared value. A smooth tokenizer would instead show a near-constant ratio across all digit lengths.

**Why it matters:** Tokenizers trained on natural language often have dense vocabulary coverage for small numbers (0-999 as single tokens) but fragment larger numbers unpredictably, creating representational discontinuities.

#### Operator Isolation Rate

Fraction of mathematical operators (`+`, `-`, `*`, `=`, `<=`, etc.) tokenized as standalone tokens rather than merged with adjacent content. Includes a compound preservation sub-metric measuring whether multi-character operators (`**`, `<=`, `!=`) are kept as single tokens vs. split.

**Example:** In the expression `3+5>=8`, a good tokenizer produces `3` | `+` | `5` | `>=` | `8` — isolation rate 1.0 and compound preservation 1.0. A bad tokenizer produces `3+` | `5` | `>` | `=` | `8` — the `+` is merged with `3` (isolation fails), and `>=` is split into `>` and `=` (compound preservation fails). Isolation rate: 1/3 (only `=` might be isolated depending on boundaries). Compound preservation: 0/1.

**Why it matters:** Merging an operator with its operand (e.g., `+3` as one token) forces the model to disentangle operation from value within a single embedding.

### UTF-8 Character Boundary Integrity

Evaluates whether byte-level tokenizers split multi-byte UTF-8 characters across token boundaries. Runs on any text data (no special config needed). Disable with `--no-utf8-integrity`.

#### Token Boundary Integrity Rate

Fraction of content tokens whose bytes form valid, complete UTF-8 sequences. A token containing an orphan continuation byte (e.g., `0xA9` alone) or a truncated leading byte (e.g., `0xC3` alone) is counted as invalid.

**Example:** The character `é` (U+00E9) is encoded as bytes `C3 A9`. A tokenizer that keeps `café` as `caf` | `é` produces two tokens, both valid UTF-8 — integrity rate 1.0. A byte-fallback tokenizer that produces `caf` | `<0xC3>` | `<0xA9>` has 3 content tokens, of which 2 (`<0xC3>` and `<0xA9>`) are individually invalid UTF-8 — integrity rate 1/3.

**Why it matters:** Invalid byte fragments force the model to reconstruct characters from meaningless pieces. Each orphan byte wastes a position in the context window while carrying no character-level information.

#### Character Boundary Split Count

Counts how many multi-byte characters in the source text have their constituent bytes spread across multiple tokens. Reports the split rate (splits / total multi-byte characters) and splits per 1k tokens.

**Example:** The Chinese text `你好` contains two 3-byte characters (`你` = `E4 BD A0`, `好` = `E5 A5 BD`). A tokenizer that keeps each character as a single token has 0 splits. A byte-fallback tokenizer that splits `你` into `<0xE4>` | `<0xBD>` | `<0xA0>` has 1 split (the character's bytes span 3 different tokens). The split rate would be 1/2 = 0.5 if `好` remains intact.

**Why it matters:** Split characters are the text-centric complement to the token-centric integrity metric. A tokenizer might have few invalid tokens overall (high integrity rate) but still split most multi-byte characters because each split produces multiple invalid tokens — the split count reveals the actual character-level damage.

### Code Tokenization Metrics

Evaluates tokenizer handling of source code by parsing it with tree-sitter and measuring alignment between AST node boundaries and token boundaries. Requires `pip install tree-sitter-language-pack`. Supports 19 languages (Python, JavaScript, Java, C, C++, Go, Rust, TypeScript, PHP, Ruby, C#, Scala, Swift, Kotlin, Lua, R, Perl, Haskell, Bash). Configure with `--code-ast-config`; disable with `--no-code-ast`.

#### AST Leaf-Node Boundary Alignment

Parses source code with tree-sitter, extracts leaf-node spans, and measures the fraction whose boundaries coincide with token boundaries. Tracks five categories independently: identifiers, keywords, operators, literals, and delimiters.

Reports start-alignment rate, end-alignment rate, full-alignment rate, and cross-boundary rate, broken down by category and language.

**Example:** For the Python snippet `return total`, tree-sitter identifies `return` (keyword, bytes 0-6) and `total` (identifier, bytes 7-12). If the tokenizer produces `return` | ` total` — both AST nodes fully align with token boundaries: full alignment = 1.0. If it produces `ret` | `urn total` — the keyword `return` has start-aligned = True (token changes at position 0) but end-aligned = False (positions 5 and 6 share a token with position 7), so fully_aligned = False. The identifier `total` has start-aligned = False (it shares a token with `urn`), so it also fails. Full alignment rate: 0/2 = 0.0.

**Why it matters:** Code has deterministic grammar, so AST node boundaries are objectively derivable with no manual annotation. A tokenizer that splits `return` into `ret` + `urn` fragments a syntactically atomic unit.

#### Identifier Fragmentation Rate

Fraction of programmer-defined identifiers split into multiple tokens, plus average tokens per identifier. Computed occurrence-weighted from the same AST extraction pass.

**Example:** A Python file contains identifiers `self` (x10 occurrences), `i` (x5), `process_data` (x3), and `MyAuthenticationFactory` (x1). The tokenizer keeps `self`, `i` as single tokens but splits `process_data` -> `process` | `_` | `data` (3 tokens) and `MyAuthenticationFactory` -> `My` | `Auth` | `entication` | `Factory` (4 tokens). Fragmentation rate: 4 fragmented occurrences out of 19 total = 0.21. Average tokens per identifier: (10x1 + 5x1 + 3x3 + 1x4) / 19 = 1.47. Note that the 10 occurrences of `self` dominate the metric and mask the fragmentation of the rarer, semantically richer identifiers.

**Why it matters:** Identifiers carry domain-specific semantics. Fragmenting `getUserName` into arbitrary sub-pieces destroys meaningful structure, though the current implementation does not yet distinguish semantically-aligned splits (at camelCase/snake_case boundaries) from arbitrary ones.

#### Indentation Depth Proportionality Correlation

Measures whether the number of whitespace tokens a tokenizer produces for leading indentation grows proportionally with nesting depth. Computes the Spearman rank correlation (ρ) between logical nesting depth (from tree-sitter) and the count of whitespace-only tokens in the leading indentation of each line. Only evaluated on whitespace-significant languages (Python, YAML). Requires at least 3 distinct depth levels per language; languages with fewer are skipped.

**Example:** A Python file has lines at depths 1, 2, 3, and 4. A proportional tokenizer encodes depth-1 indentation as 1 whitespace token, depth-2 as 2, depth-3 as 3, and depth-4 as 4 — perfect rank correlation, ρ = 1.0. A tokenizer that merges all indentation into a single token regardless of depth (1, 1, 1, 1 whitespace tokens) produces ρ ≈ 0.0. A tokenizer that uses *more* tokens for shallow depths than deep ones gives ρ < 0.

**Why it matters:** If indentation depth maps monotonically to whitespace token count, the model receives a natural positional signal for nesting structure without needing to learn it from context.

#### Indentation Pattern Stability Rate

Measures whether lines at the same nesting depth are tokenized with the same whitespace pattern. For each depth level, groups all indented lines and counts how many share the dominant tokenization pattern (the tuple of whitespace token lengths). The stability rate is the total number of lines matching their depth's dominant pattern divided by the total number of indented lines.

**Example:** A file has 10 lines at depth 2. Eight of them tokenize the leading whitespace as `(4, 4)` (two 4-space tokens) and two as `(8,)` (one 8-space token). The dominant pattern at depth 2 is `(4, 4)` with 8 matches. If all other depths similarly have high dominant-pattern counts, the overall stability rate approaches 1.0. A tokenizer that encodes the same 8-space indentation differently depending on what follows — sometimes `(4, 4)`, sometimes `(3, 5)`, sometimes `(8,)` — yields a low stability rate.

**Why it matters:** Consistent indentation tokenization means the model sees the same token pattern for the same structural level, reducing the number of surface forms it must learn to associate with a single syntactic meaning.

### Multilingual Fairness
- **Tokenizer Gini Coefficient**: Measures equitable treatment across languages, defined as:

* $`L = \{1, \dots, n\}`$ be the set of languages, each weighted equally.
* For every language $`\ell \in L`$, define the **token cost**
```math
  c_\ell \;=\;
  \frac{\text{number of tokens produced by the tokenizer on language }\ell}
       {\text{number of raw bytes (or lines for parallel ds) in the same text}}
```
  (lower $`c_\ell`$ means cheaper encoding, higher means more byte-hungry).

* Let the mean cost be
```math
  \mu \;=\; \frac{1}{n}\;\sum_{\ell=1}^{n} c_\ell.
```

Then the **Tokenizer Fairness Gini** with equal weights is

```math
\mathrm{TFG}
=\frac{\displaystyle\sum_{i=1}^{n}\sum_{j=1}^{n} \lvert c_i - c_j \rvert}
        {2\,n^2\,\mu}
```
* **Range:** $`0 \le \mathrm{TFG} \le 1`$
  * $`0`$: perfect parity (every language has identical byte-normalised token cost).
  * $`1`$: maximal unfairness.

## Data Format Requirements

The framework supports three input text formats:

- **Plain text** (`.txt`): One sentence per line recommended for parallel corpora
- **JSON**: Object with a `"texts"` array of strings
- **Parquet**: DataFrame with a `"text"` column

## Module Structure

```
tokenizer_analysis/
├── __init__.py                    # Main package exports
├── main.py                        # UnifiedTokenizerAnalyzer orchestration class
├── constants.py                   # Package-level constants
├── config/                        # Configuration modules
│   ├── language_metadata.py      # LanguageMetadata for grouping analysis
│   └── text_measurement.py       # Text measurement configuration
├── core/                          # Core data structures and providers
│   ├── input_providers.py        # InputProvider implementations
│   ├── input_types.py            # TokenizedData and core types
│   ├── input_utils.py            # Input loading and validation utilities
│   ├── tokenizer_wrapper.py      # Generic wrapper for tokenizer objects
│   └── validation.py             # Data validation functions
├── metrics/                       # Metrics computation modules
│   ├── base.py                   # BaseMetrics with common utilities
│   ├── basic.py                  # Basic tokenization metrics
│   ├── information_theoretic.py  # Information-theoretic metrics
│   ├── math.py                   # Mathematical content metrics (digit boundaries, operators)
│   ├── code_ast.py               # Code tokenization metrics (AST alignment, indentation)
│   ├── utf8_integrity.py         # UTF-8 character boundary integrity metrics
│   ├── morphological.py          # Morphological boundary alignment
│   ├── morphscore.py             # MorphScore neural evaluation
│   └── gini.py                   # Multilingual fairness metrics
├── loaders/                       # Data loading modules
│   ├── constants.py              # Language code mappings (ISO639-1 to FLORES)
│   ├── code_data.py              # Code snippet loader for AST metrics
│   ├── morphological.py          # Morphological dataset loader
│   └── multilingual_data.py      # Multilingual text dataset loader
├── utils/                         # Utility functions
│   ├── text_utils.py             # Text processing utilities
│   └── tokenizer_utils.py        # Tokenizer loading utilities
└── visualization/                 # Plotting and visualization
    ├── plotter.py                # TokenizerVisualizer main class
    ├── plots.py                  # Core plotting functions
    ├── data_extraction.py        # Data extraction for plotting
    ├── latex_tables.py           # LaTeX table generation
    ├── markdown_tables.py        # Markdown table generation and git push
    └── visualization_config.py   # Visualization configuration

scripts/
├── run_tokenizer_analysis.py     # Main CLI for analysis
└── update_remote.py              # Push RESULTS.md to a remote git branch
```

## Troubleshooting

**`No module named 'morphscore'`** — Initialize submodules: `git submodule update --init --recursive && cd morphscore && pip install -e . && cd ..`

**`Unknown tokenizer class`** — Available classes: `"huggingface"`, `"custom_bpe"`, `"pretokenized"`. Register custom classes with `register_tokenizer_class()` (see Contributing).

**`FileNotFoundError`** — Check that paths in config files are absolute or relative to the working directory.

**`_tkinter.TclError: no display name`** — Set `export MPLBACKEND=Agg` before running on headless servers.

## Contributing

### Adding New Tokenizers

Subclass `TokenizerWrapper` from `tokenizer_analysis.core.tokenizer_wrapper`:

```python
from tokenizer_analysis.core.tokenizer_wrapper import TokenizerWrapper, register_tokenizer_class

class MyTokenizer(TokenizerWrapper):
    def get_name(self) -> str: ...
    def get_vocab_size(self) -> int: ...
    def get_vocab(self) -> Dict[str, int]: ...
    def can_encode(self) -> bool: return True
    def encode(self, text: str) -> List[int]: ...

    @classmethod
    def from_config(cls, name, config):
        return cls(name, config['path'])

register_tokenizer_class('my_class', MyTokenizer)
```

Then reference `"class": "my_class"` in your tokenizer config.

### Adding New Metrics
1. Inherit from `BaseMetrics` in `tokenizer_analysis/metrics/base.py`
2. Implement `compute()` method
3. Register in `main.py`

### Submitting Changes
1. Fork the repository
2. Create a feature branch
3. Ensure all existing tests pass
4. Submit a pull request

## Citation

```bibtex
@software{meister_tokenizer_analysis_2025,
  title = {TokEval: A Tokenizer Analysis Suite},
  author = {Meister, Clara},
  year = {2025},
  url = {https://github.com/swiss-ai/tokenizer-intrinsic-evals}
}
```
