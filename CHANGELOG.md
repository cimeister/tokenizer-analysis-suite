# Changelog

This project supersedes `tokenizer-analysis-suite`. The section below is the
complete set of changes from that suite to the current release, ordered with
the breaking ones first. Per-release detail is in `git log`.

The install name is now `tokenizer-intrinsic-evals` (was `tokenizer-analysis`).
The import name is unchanged: `import tokenizer_analysis`. The prior suite,
including the `PA_BPE_tokenizers/` directory, is preserved on the
`legacy-suite` branch and the `legacy-suite-final` tag.

---

## Changes from tokenizer-analysis-suite

### Breaking: metric definitions changed, so old numbers are not comparable

These changed what a metric computes, not where it is written. A value from the
old suite cannot be compared with a value from this one for any field below.

| Field | What changed | Measured effect |
|---|---|---|
| `ast_boundary_alignment`, `identifier_fragmentation` | Source spans map to tokens through the tokenizer's own character offsets. They were previously matched against a string rebuilt from token surfaces, which lost synchronisation after any multi-space token. | Llama 3 `full_alignment_rate` 0.127 to 0.519; its `unmappable` count 119560 to 0 |
| `three_digit_boundary_alignment`, `numeric_magnitude_consistency`, `operator_isolation_rate` | Same change from reconstruction to offsets. | Far more spans measured for any language with non-ASCII text |
| `operator_isolation_rate` | A character claimed by two token ranges is assigned to the later token. | XLM-RoBERTa isolation 0.6948 to 0.6770, compound preservation 0.7222 to 0.8443, over 454693 operators |
| `identifier_fragmentation.avg_tokens_per_identifier` | Unmappable spans are excluded rather than counted with a `-1` sentinel. | Was biased low, and negative for C# |
| `utf8_token_integrity`, `utf8_char_split` | Byte-level detection no longer requires 50 of 68 GPT-2 marker characters in the vocabulary. | gpt4o-english-bpe `completeness_rate` 1.0000, best of 37 tokenizers, to 0.6688, worst of 37 |
| `reconstruction_fidelity.whitespace_fidelity` | Whitespace widened from ASCII space, tab, newline and carriage return to those plus every Unicode `Zs` separator. | Any corpus containing NBSP, thin space or ideographic space |
| `renyi_efficiency` | Normalizes by `log2` of the declared vocabulary size, following Zouhar et al. 2023. It previously divided by the number of token types observed in the corpus. | The two rank tokenizers at Spearman 0.678 over 37 tokenizers. The old normalization is still published under `renyi_efficiency.observed_normalization` |

A `null` where an old run had `0.0` reflects the convention change below, not a
change in what was measured.

### Breaking: output format

- Every metric publishes a per-tokenizer `global`. Five had none.
- Every metric's `metadata` includes `aggregation`, one of `micro_pooled`,
  `macro_languages`, `ratio_of_sums` or `set_union`, naming which average its
  `global` is. Every `per_language` entry includes a `count`, so the other
  weighting can be re-derived.
- A value that could not be measured is `null`, not `0.0`. `0.0` is a legal
  value for most of these metrics, so a zero was indistinguishable from a
  measurement. `count` and `sum` stay numeric.
- The per-tokenizer compression key is `compression_rate`, was
  `compression_ratio`.
- `analysis_results.json` is organized as
  `{metric: {per_tokenizer: {tok: {global, per_language}}, metadata}}`. The old
  flat layout and its `summary` and `pairwise_comparisons` blocks are gone.
- `analysis_results.json` is a strict projection of `analysis_results_full.json`:
  every leaf is at the same key path with the same value in both.
- Six metrics are no longer top-level keys. Each is a field of the metric that
  owns the measurement, with the reason in that metric's
  `metadata.merged_metrics`.

  | Old top-level key | New location |
  |---|---|
  | `avg_tokens_per_line` | `compression_rate.per_tokenizer.<tok>.tokens_per_line` |
  | `type_token_ratio` | `vocabulary_utilization.per_tokenizer.<tok>.type_token_ratio` |
  | `unigram_distribution_metrics` | `renyi_efficiency.per_tokenizer.<tok>.unigram_distribution` |
  | `utf8_char_split` | `utf8_token_integrity.per_tokenizer.<tok>.char_split` |
  | `lorenz_curve_data` | `tokenizer_fairness_gini.per_tokenizer.<tok>.lorenz_curve` |
  | `digit_split_variability` | `three_digit_boundary_alignment.per_tokenizer.<tok>.split_variability` |

  Four are exact identities, so nothing is lost: `compression_rate` under the
  `lines` measurement times `avg_tokens_per_line` is 1; `1 - 2*area(lorenz)` is
  the Gini coefficient; `renyi_1.0` times `log2` of the declared vocabulary
  size is the unigram entropy; and the type-token ratio is vocabulary
  utilization rescaled by vocabulary size over token count. The other two rank
  tokenizers at Spearman -0.954 and -0.992 against the metric they were folded
  into.

### Breaking: Python API

- `TokenizerWrapper` subclasses must implement `get_special_token_strings()`.
  It is abstract, so a wrapper written against 0.x raises `TypeError` until it
  is added. Return the surface strings the tokenizer declares special, read
  from its own metadata; an empty set only if it genuinely has none, and `None`
  if it cannot report them, after which the library warns and falls back to
  `GENERIC_SPECIAL_TOKENS`. Do not pattern-match on token surfaces. That is
  what the removed `_SPECIAL_TOKEN` regex did, and it deleted ordinary content
  tokens such as `[0]` and `[...]` while missing `<s>` and `</s>`.
- `MorphologicalMetrics` and `MorphologicalDataLoader` are removed. Use
  MorphScore, through `MorphScoreMetrics` or the `--morphscore` flags.
- `MarkdownTableGenerator`, `results_filename` and
  `UnifiedTokenizerAnalyzer.generate_markdown_table()` are removed. The
  cumulative Markdown leaderboard they produced was built for one internal
  project. Read `analysis_results.json`, or use `generate_latex_tables()`.
- `UnifiedTokenizerAnalyzer(...)` no longer accepts `morphological_config`, and
  `run_analysis(...)` no longer accepts `include_morphological`.
- Constants moved from namespace classes to module-level names. Replace
  `from tokenizer_analysis.constants import DataProcessing` and
  `DataProcessing.DEFAULT_CHUNK_SIZE` with
  `from tokenizer_analysis.constants import DEFAULT_CHUNK_SIZE`.

Accessing a removed name raises with its replacement named, rather than a bare
`ImportError`.

### Breaking: command line

| Removed | Use instead |
|---|---|
| `--morphological-config` | `--morphscore`, or `--morphscore-config <file>` |
| `--latex-table-types morphological` | any of the remaining table types |
| `--update-results-md`, `--dataset`, `--sort-results-by` | read `analysis_results.json`, or `--generate-latex-tables` |

`--use-sample-data` no longer overrides `--tokenizer-config`,
`--language-config` or `--measurement-config`. It previously replaced all three
without a warning, producing a complete results file for tokenizers and a
corpus the caller did not ask for. The combination is now an error naming the
conflicting flags.

### Added

- `--input PATH` for a single corpus, taking a file or a directory, with
  `--input-label` for the name it appears under. Omitting both `--input` and
  `--language-config` is an error naming both.
- A `run_metadata` block in every results file: package version, UTC timestamp,
  git commit and tree state, the config paths with hashes, a hash or Hub
  revision per tokenizer, a digest of each corpus file, the non-default
  arguments, and the code-corpus caps.
- `tokenizer-sanity-check`, which runs 16 checks against one tokenizer and
  exits non-zero on a failure.
- `benchmarks/open_source/`, nine widely used tokenizers measured on the full
  metric set, regenerated by one command.
- `--max-code-files-per-lang` and `--max-code-file-chars`, replacing silent
  caps of 100 files and 15000 characters.

### Fixed, where the fix changed a published number

- A missing corpus file, a missing MorphScore directory and a failed language
  load aborted rather than producing a complete-looking file with a language
  silently dropped or four zeros where nothing was evaluated.
- `tokenizer_fairness_gini` with fewer than two languages reports `null` rather
  than `0.0`, which read as perfect fairness. `cost_ratio` reports `null`
  rather than `float('inf')`, and `std_cost` uses `ddof=1`.
- The Gini coefficient is deterministic. The per-language cost vector was
  summed in a hash-dependent order, so one commit produced two values.
- Tree-sitter `ERROR` spans are excluded from `ast_boundary_alignment` and
  published as `parse_error_spans`. They were scored as AST leaves and scored
  above each tokenizer's average, so every rate was optimistic.
- `indentation_consistency.depth_proportionality_correlation` reports `null`
  when the whitespace-token count is constant across depths, where a
  correlation is undefined. It reported `0.0`, which is a real measurement
  meaning depth and whitespace-token count are unrelated.
- The results file is strict JSON on every corpus. It could contain `NaN`.
- A subword marker is stripped only from a tokenizer shown to use it. The
  WordPiece, CLIP-BPE and subword-nmt rules were applied to every tokenizer,
  truncating ordinary content such as Markdown headings and punctuation runs.
- Directory input is sorted, so the same corpus gives the same result on two
  machines.

---

## Releases

- **1.0.2** Fifteen fields that published a number where nothing was measured
  now publish `null`. `analysis_results.json` became a strict projection of the
  full file. `run_metadata` gained a timestamp, per-tokenizer Hub revisions and
  a corpus digest. Two sanity checks stopped reporting a passing score for
  checks that ran on nothing, and one could turn a FAIL into a PASS.
- **1.0.1** A LaTeX table had an empty column. Tree-sitter `ERROR` spans
  were scored as AST leaves. The verbose printers rendered a `null` as `0.000`
  or raised. A directory of `vocab.json` and `merges.txt` could not load.
- **1.0.0** First release of the consolidated repository.
