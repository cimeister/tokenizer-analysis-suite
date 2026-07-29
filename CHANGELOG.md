# Changelog

All notable changes to this project are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (output format, breaking)
- A value that could not be measured is now `null`, not `0.0`. This affects
  every rate the pipeline publishes, via `BaseMetrics.safe_divide` and
  `BaseMetrics.empty_stats()`. A tokenizer that never emitted an UNK and one
  with no UNK token at all both reported `unk_token_rate: 0.0`; a domain
  containing no whitespace reported `whitespace_fidelity: 1.0` beside domains
  that genuinely preserved it. `count` and `sum` stay numeric.
- Six metrics are folded into the metric that owns the measurement, so the
  results file no longer publishes one number twice. Four are algebraic
  identities that hold for every tokenizer, not empirical correlations.

  | Was a top-level metric | Now reported under | Field | Evidence |
  |---|---|---|---|
  | `avg_tokens_per_line` | `compression_rate` | `tokens_per_line` | product = 1.000000 |
  | `type_token_ratio` | `vocabulary_utilization` | `type_token_ratio` | identity = 1.000000 |
  | `unigram_distribution_metrics` | `renyi_efficiency` | `unigram_distribution` | identity, zero relative error |
  | `utf8_char_split` | `utf8_token_integrity` | `char_split` | Spearman -0.954 |
  | `lorenz_curve_data` | `tokenizer_fairness_gini` | `lorenz_curve` | identity to 1e-6 |
  | `digit_split_variability` | `three_digit_boundary_alignment` | `split_variability` | Spearman -0.992 |

  Measured across 37 tokenizers on 13 FLORES+ languages. Each primary records
  the merge and its evidence under `metadata.merged_metrics`.
- `tokenizer_fairness_gini` with fewer than two languages reports
  `gini_coefficient: null` and the real `mean_cost`, instead of `0.0` for both,
  which read as perfect fairness and zero cost.
- `cost_ratio` returns `null` rather than `float('inf')` when the minimum cost
  is zero. `json.dump` wrote that as the bare token `Infinity`, which is not
  valid JSON and is rejected by strict parsers.
- `tokenizer_fairness_gini.std_cost` uses `ddof=1`, matching
  `compute_basic_stats` and `vocabulary_utilization`. The population form
  understated it by 4.1% at 13 languages.
- `identifier_fragmentation` excludes identifier spans that could not be mapped
  into the reconstructed text, and reports them as `unmappable`, instead of
  counting them as fragmented with a token count of -1.

### Changed (metric definitions)
- `renyi_efficiency` now follows Zouhar et al. 2023: `H_alpha / log2(|V|)` with
  `|V|` the declared vocabulary size. It previously divided by `log2(number of
  token types observed in the corpus)`, which is corpus-dependent and gave each
  language a different divisor, so per-language values were not on a common
  scale. The two rank tokenizers at Spearman 0.678 over 37 tokenizers, with a
  maximum shift of 16 places, so they are not interchangeable. The old
  normalization is still published under
  `renyi_efficiency.observed_normalization`, so values from earlier runs remain
  reproducible and the two can be compared directly.

- `bigram_entropy` and `trigram_entropy` keep their current definition
  (frequency-weighted, occurrence-count filtering, no windowing) and now
  document how it differs from the cited Poelman et al. 2025. Each also
  publishes a `reference_definition` block computed with the paper's
  normalizer, `log2(min(corpus-wide accessor domain, context count))`, and its
  unweighted mean over types, so the two can be compared on one run. The
  paper's punctuation, digit and boundary-ratio type filters are not
  implemented and that is stated in the block.

  The normalizers measure different things. Dividing by a context's own
  successor count makes eta pure evenness: two equally likely successors and
  500 equally likely successors both score 1.0. Dividing by the corpus-wide
  accessor domain puts variety and evenness on one shared scale, so a context
  with few successors scores low however even they are. Since a context's
  successor count never exceeds its occurrence count, the current definition is
  systematically the higher of the two.

### Fixed (metric correctness)
- `_build_source_to_recon_map` advanced only on a match, so one character the
  reconstruction added (a byte-level vocabulary renders `é` as `Ã©`) left it
  stuck and unmapped the rest of the document. Consumers score an unmappable
  span as a miss, so this dropped digit spans and marked AST nodes misaligned.
  Digit spans measured on the demo corpus went from 358 to 456; English from 116
  to 143 of the 143 present.
- UTF-8 byte-level detection counted vocabulary marker characters against a
  threshold of 50 of 68. A byte-level tokenizer whose training corpus never
  exercised the control bytes fell below it, was read as not byte-level, and
  could then only report a completeness of 1.0. `gpt4o-english-bpe` reported
  1.0000, best of 37 tokenizers, against a true 0.6688, worst of 37. Detection
  now reads the tokenizer's own ByteLevel components.
- `CodeDataLoader` strips a leading byte-order mark and normalizes CRLF. Two of
  the three bundled C# samples carry a BOM, which a byte-level tokenizer
  re-encodes as three visible characters, making 63% of C# AST spans unmappable
  and giving C# 0.13 to 0.19 alignment against a 0.51 to 0.70 median for every
  tokenizer alike.
- `LanguageMetadata` accessors read `analysis_groups['script_families']`
  (plural) while every shipped config writes `script_family`, so
  `get_script_families()` returned `[]` and `get_script_family()` returned
  `'Unknown'` for every language. Both spellings now resolve.
- `LanguageMetadata` accepts the `{"en": "/path/to/data"}` short form the README
  has always documented; it previously raised `AttributeError`.
- Directory input globs are sorted. They feed a `--samples-per-lang`
  truncation, so filesystem order decided which texts were analyzed.
- `scipy.stats` is imported explicitly in `metrics/base.py`.
- `indentation_consistency.pattern_stability_rate` counted the first code token
  as indentation. A byte-level tokenizer folds the last indent space into the
  following word, so `'    return x'` tokenizes as `ĠĠĠ` plus `Ġreturn` and the
  second token overlapped the indent range. Lines with identical indentation but
  different code therefore counted as different patterns, and the rate measured
  which word each line started with. Only whitespace-only tokens now enter the
  pattern. On the demo corpus Python goes to 1.0, which is correct for uniformly
  indented code.

### Removed
- Swift, Kotlin and Perl are excluded from the code AST metrics. `classify_node`
  does not know their identifier node types (`simple_identifier` for the first
  two, `varname` and `function` for Perl), so the identifier share of classified
  leaves was 0.073, 0.058 and 0.000 against 0.19 to 0.37 for every supported
  language. They are skipped with a named warning rather than scored on a
  fraction of their code. Adding them means extending `IDENTIFIER_TYPES` in
  `_treesitter_worker.py`; open an issue if you want one of them.
- `tokenizer_analysis/core/validation.py`. Its `ValidationResult`,
  `TokenizedDataValidator`, `InputProviderValidator`,
  `InputSpecificationValidator` and `AnalysisValidator` had no callers anywhere
  and duplicated `core/input_utils.InputValidator`, which is the one `main.py`
  uses. Having never run, it was also unverified against the current data model.
- The cumulative Markdown leaderboard, along with the `--update-results-md`,
  `--dataset` and `--sort-results-by` flags, the generated `RESULTS.md` and its
  per-dataset plot directories, `tokenizer_analysis/visualization/markdown_tables.py`,
  and the `MarkdownTableGenerator` / `results_filename` exports from
  `tokenizer_analysis.visualization`. It was built for one internal tokenizer
  project rather than for general use. Read `analysis_results.json` directly, or
  use `--generate-latex-tables`, which draws on the same per-tokenizer
  aggregates.
- `UnifiedTokenizerAnalyzer.generate_markdown_table()`.

### Fixed
- The test suite aborted partway through with `malloc(): mismatching
  next->prev_size` (SIGABRT), so 86% of it never ran. All tree-sitter parsing,
  including `ASTBoundaryMetrics.compute_per_text()` and the tests, now goes
  through the same one-subprocess-per-language fence that `compute()` already
  used. A language whose grammar crashes the worker is reported as unmeasured
  with the grammar and pack version named, rather than being absent from the
  results with no explanation.
- `--update-results-md` without `--dataset` blocked on `input()` from stdin,
  which hung batch and SLURM runs. The flag is gone.

## [1.0.0] - 2026-07-04

First release under the consolidated repository
`github.com/cimeister/tokenizer-intrinsic-evals`. This version supersedes the
older `tokenizer-analysis-suite`. The install (distribution) name is now
`tokenizer-intrinsic-evals` (was `tokenizer-analysis`); the import name is
still `import tokenizer_analysis`. See MIGRATION.md for a step-by-step upgrade
guide.

### Added
- New metric families: bigram and trigram successor entropy
  (`bigram_entropy`, `trigram_entropy`); math digit-boundary metrics
  (three-digit alignment, digit-split variability, magnitude consistency,
  operator isolation); code AST-boundary alignment and identifier
  fragmentation (tree-sitter, 19 languages); UTF-8 token integrity and
  character-split metrics; reconstruction fidelity (exact match, CER,
  whitespace fidelity); cross-lingual vocabulary-utilization CoV
  (`vocab_util_cross_lingual_cov`); `avg_langs_per_token`; `avg_tokens_per_line`.
- New console script `tokenizer-visualize`: colour-coded token-boundary views
  over source text.
- New console script `tokenizer-sanity-check`: single-tokenizer health report
  (byte coverage, whitespace/digits, special tokens, determinism, Unicode
  normalization, vocabulary integrity and reachability), with pass/warn/fail
  severities and non-zero exit codes.
- Per-document metric outputs (`tokenizer_analysis/per_example.py`).
- Reporting: faceted plots (one subplot per tokenizer), a cumulative Markdown
  leaderboard (`--update-results-md`), and expanded LaTeX tables with
  direction arrows.
- Packaging: `LICENSE` (MIT), `NOTICE` (FLORES+ attribution), this changelog,
  and `MIGRATION.md`.

### Changed
- Install (distribution) name renamed from `tokenizer-analysis` to
  `tokenizer-intrinsic-evals`, matching the repository. The import name is
  unchanged (`import tokenizer_analysis`), as are the console scripts
  (`tokenizer-analysis`, `tokenizer-visualize`, `tokenizer-sanity-check`).
  Because the distribution name changed, `pip install --upgrade
  tokenizer-analysis` will not find this release; install
  `tokenizer-intrinsic-evals` instead.
- Minimum Python raised from 3.8 to 3.10.
- Results JSON: the per-tokenizer compression key is `compression_rate`
  (previously `compression_ratio`); the slim `analysis_results.json` is now
  organized as `{per_tokenizer: {global, per_language}}`.
- Constants moved from namespace classes (`TextProcessing`, `DataProcessing`,
  `Statistics`, `Validation`, ...) to module-level names in
  `tokenizer_analysis/constants.py`.
- `text_measurement` configs now reject unknown keys with a clear error naming
  the offending key, instead of raising an opaque `TypeError`.
- Packaging moved to `pyproject.toml` + `uv.lock` (hatchling). tree-sitter
  support is a core dependency; parquet reading is the optional `parquet` extra.

### Removed
- The standalone morphological boundary metric, its module
  (`metrics/morphological.py`), its loader (`loaders/morphological.py`), the
  `MorphologicalMetrics` / `MorphologicalDataLoader` exports, the
  `--morphological-config` flag, and the `morphological` LaTeX table type. Use
  MorphScore (`--morphscore` / `--morphscore-config`) instead. The removed flag
  and table type now fail with a message pointing to MorphScore.
- The results-branch publishing workflow (`scripts/update_remote.py`).
- Bundled OpenAI tiktoken vocabulary JSONs
  (`tokenizers/gpt_4_hf.json`, `tokenizers/gpt_4o_hf.json`); load those via
  `tiktoken` at run time instead.
- Apertus-specific research artifacts (`apertus_tokenizer_design.md`, the
  `results/` reports and figures) and configs referencing untracked cluster
  data. The prior suite state is preserved on the `legacy-suite` branch and the
  `legacy-suite-final` tag.

### Fixed
- README documented `text_measurement` keys that did not match the code (for
  example `line_counting_method`); the documented example now loads.
- README quick-start pointed at `results/fertility.png`; individual plots are
  written as `results/fertility_individual.svg`.
