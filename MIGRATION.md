# Migrating from tokenizer-analysis-suite to 1.0.0

This project was consolidated into
`github.com/cimeister/tokenizer-intrinsic-evals` and released as 1.0.0. The
import name is unchanged (`import tokenizer_analysis`), but the install
(distribution) name changed. This guide lists the breaking changes and their
replacements.

## Repository and install

- The distribution name is now `tokenizer-intrinsic-evals` (was
  `tokenizer-analysis`), so `pip install --upgrade tokenizer-analysis` will not
  find this release. Nothing is published to PyPI under either name yet, so
  install from the repository:
  `pip install git+https://github.com/cimeister/tokenizer-intrinsic-evals.git`,
  or clone it and run `uv sync`. The console scripts are unchanged
  (`tokenizer-analysis`, `tokenizer-visualize`, `tokenizer-sanity-check`), as is
  `import tokenizer_analysis`.
- The old repository was renamed, so existing clones keep working: `git pull`
  fast-forwards, and the old URL redirects for both web and git. To point a
  remote at the new name explicitly:
  `git remote set-url origin https://github.com/cimeister/tokenizer-intrinsic-evals.git`.
- Minimum Python is now 3.10 (was 3.8).
- tree-sitter (code AST metrics) is a core dependency. Parquet reading and
  SentencePiece model files are optional extras:
  `uv sync --extra parquet --extra sentencepiece`. Install both in one command:
  `uv sync` performs an exact sync, so a second `uv sync` with a different
  `--extra` removes what the first one installed.

## Command-line changes

| Removed | Replacement |
|---------|-------------|
| `--morphological-config FILE` | `--morphscore` (defaults) or `--morphscore-config FILE` |
| `--latex-table-types morphological` | Use MorphScore; valid types are `basic`, `information`, `comprehensive` |
| `--update-results-md [PATH]` | Read `analysis_results.json`, or `--generate-latex-tables` |
| `--dataset NAME` | No replacement; it only labelled leaderboard rows |
| `--sort-results-by METRIC` | No replacement; sort when reading the JSON |

Both removed options now exit with a message pointing to MorphScore rather than
a generic argparse error.

## Results / output schema

### Absent values are `null`

Any rate whose denominator was zero now serializes as `null` instead of `0.0`.
A parser doing arithmetic on these fields needs a `None` check. The change is
deliberate: `0.0` made "nothing to measure" indistinguishable from "the measured
value is zero", which is the difference between a tokenizer that never emitted
an UNK and one that has no UNK token.

### Six metrics moved under the metric they restate

These are no longer top-level keys in `analysis_results.json`. Each is reported
as a field of the metric that owns the measurement, and the reason is recorded
in that metric's `metadata.merged_metrics`.

| Old top-level key | New location |
|---|---|
| `avg_tokens_per_line` | `compression_rate.per_tokenizer.<tok>.tokens_per_line` |
| `type_token_ratio` | `vocabulary_utilization.per_tokenizer.<tok>.type_token_ratio` |
| `unigram_distribution_metrics` | `renyi_efficiency.per_tokenizer.<tok>.unigram_distribution` |
| `utf8_char_split` | `utf8_token_integrity.per_tokenizer.<tok>.char_split` |
| `lorenz_curve_data` | `tokenizer_fairness_gini.per_tokenizer.<tok>.lorenz_curve` |
| `digit_split_variability` | `three_digit_boundary_alignment.per_tokenizer.<tok>.split_variability` |

Four of the six are exact identities rather than correlations, so nothing is
lost: `compression_rate(lines) * avg_tokens_per_line` is 1 for every tokenizer,
`1 - 2*area(lorenz)` is the Gini coefficient, `renyi_1.0 * log2(observed types)`
is the unigram entropy, and TTR is vocabulary utilization rescaled by vocabulary
size over token count. The other two rank tokenizers at Spearman -0.954 and
-0.992 against their primaries.

### Numbers that changed

Runs produced before this release are affected by three correctness fixes, so
their values are not comparable with new ones:

- Any `utf8_token_integrity` or `utf8_char_split` figure for a byte-level
  tokenizer whose vocabulary carries fewer than 50 of the 68 GPT-2 marker
  characters is invalid; it was reported as a perfect 1.0.
- `identifier_fragmentation.avg_tokens_per_identifier` was biased low, and
  negative for C#, by an unmappable-span sentinel.
- Digit-boundary metrics measured far fewer spans than the corpus contained for
  any language with non-ASCII text.


- The per-tokenizer compression key is now `compression_rate` (was
  `compression_ratio`). Update any downstream parser that reads the old key.
- The slim `analysis_results.json` is now organized as
  `{per_tokenizer: {global, per_language}}`. If you consumed the old flat
  layout (or its `summary` / `pairwise_comparisons` blocks), read the new
  structure, or pass `--save-full-results` for the detailed output.

## Python API

### `TokenizerWrapper` subclasses must implement `get_special_token_strings()`

It is a new abstract method, so a custom wrapper written against 0.x fails to
instantiate until it is added:

    TypeError: Can't instantiate abstract class MyTokenizer with abstract
    method get_special_token_strings

Return the surface strings your tokenizer declares special, read from its own
metadata. Return an empty set only if it genuinely has none, and `None` if it
cannot report them; the library then warns and falls back to
`GENERIC_SPECIAL_TOKENS`. Do not pattern-match on token surfaces: that is what
the removed `_SPECIAL_TOKEN` regex did, and it deleted ordinary content tokens
such as `[0]` and `[...]` while missing `<s>` and `</s>`.


- `MorphologicalMetrics` and `MorphologicalDataLoader` were removed from
  `tokenizer_analysis`. Use MorphScore instead.
- `MarkdownTableGenerator` and `results_filename` were removed from
  `tokenizer_analysis.visualization`, along with
  `UnifiedTokenizerAnalyzer.generate_markdown_table()`. The cumulative Markdown
  leaderboard they produced was built for one internal tokenizer project. Read
  `analysis_results.json`, or use `generate_latex_tables()`, which reads the
  same per-tokenizer aggregates.
- `UnifiedTokenizerAnalyzer(...)` no longer accepts `morphological_config`, and
  `run_analysis(...)` no longer accepts `include_morphological`.
- Constants moved from namespace classes to module-level names. Replace imports
  like `from tokenizer_analysis.constants import DataProcessing` (then
  `DataProcessing.DEFAULT_CHUNK_SIZE`) with the module-level constant
  (`from tokenizer_analysis.constants import DEFAULT_CHUNK_SIZE`).

## Configuration

- `text_measurement` configs now reject unknown keys. The correct keys are
  `method`, `byte_counting`, `word_counting`, `line_counting`, `custom_regex`,
  and `include_empty_splits`. Note earlier docs used names such as
  `line_counting_method` and `include_empty_lines`, which were never valid keys
  in the code; update those configs. See the README "Text Measurement
  Configuration" table for the valid values.

## Data and artifacts

- The bundled OpenAI tiktoken vocabulary JSONs were removed. Load GPT-4 /
  GPT-4o tokenizers via `tiktoken` (a core dependency) at run time instead.
- Apertus-specific reports and design docs were removed from the tracked tree.
  The complete prior suite state (including the `PA_BPE_tokenizers/` directory)
  is preserved on the `legacy-suite` branch and the `legacy-suite-final` tag.
