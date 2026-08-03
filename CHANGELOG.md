# Changelog

All notable changes to this project are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] (unreleased)

Everything below ships in 1.0.0. Nothing has been published yet, so this
section and the one at the end of the file are the same release: the summary
there describes the consolidation from `tokenizer-analysis-suite`, and this
section records the vetting that followed it. Set the date at tag time.


### Changed (output format, breaking)
- Every metric publishes a per-tokenizer `global`. Five had none:
  `trigram_entropy` published the same four values as flat `global_*` siblings,
  so a parser written against `bigram_entropy` found nothing;
  `three_digit_boundary_alignment` and `numeric_magnitude_consistency` had no
  headline block at all; `token_length` and `encoding_speed` now carry one that
  duplicates an existing block, deliberately, because an exception in the schema
  costs a reader more than a duplicated number.
- Every metric's `metadata` carries `aggregation`, one of `micro_pooled`,
  `macro_languages`, `ratio_of_sums` or `set_union`, saying which average
  `global` reports. `global` meant a ratio of sums in one metric, a mean of
  per-document ratios in another, an unweighted mean across languages in a third
  and a set union in a fourth, with nothing recording which. On the bundled
  parallel corpus micro and macro agree, so the difference was invisible until
  an unequal corpus. `bigram_entropy` and `trigram_entropy` held free text
  there; that prose moved to `context_weighting` and the field now takes the
  label.
- Every `per_language` entry carries a `count`, and every metric's `metadata`
  names the unit it is in. `compression_rate.per_language.<lang>` therefore
  changes from a bare number to `{compression_rate, count, total_tokens}`; the
  rate is unchanged. `tokenizer_fairness_gini` and `morphscore` publish no
  per-language count, because their unit is languages and the count would be 1
  for every entry, and say so in `metadata.per_language_count` rather than being
  silent.

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

### Fixed (documentation of behaviour)
- The comment on the Gini and information-theoretic metrics said they default
  to lines, and both modules imported the line config and never used it. Both
  have always taken the byte config, which is the intended quantity: cost per
  byte is the one unit that means the same thing in every script. The comment,
  the unused import and the README's measurement table now say bytes, and the
  table says that `--measurement-config` moves compression, Gini and the
  information-theoretic metrics together while fertility always counts words.
- The README says what one document is in a `.txt` corpus: a blank line
  anywhere means the file is paragraph separated and each paragraph is a
  document, otherwise each line is. It also says that documents under 5
  characters are dropped and that whitespace runs are collapsed, so a `.txt`
  corpus is the wrong input for measuring indentation.
- The MorphScore data is one `snapshot_download` from
  `catherinearnett/morphscore`, which ships the per-language CSVs in the layout
  the code reads. The README pointed at the code repository instead and never
  named the dataset. Verified end to end: 5 languages, 63803 samples.

### Fixed (output format)
- The results file is strict JSON on every corpus.
  `numeric_magnitude_consistency.scaling.spearman_p` was written as `NaN`
  whenever the corpus spanned two digit-length buckets, because
  `scipy.stats.spearmanr` over two points returns a defined rho and an
  undefined p: a rank correlation over two points has no significance level.
  `json.dump` renders that as the bare token `NaN`, which no strict parser
  accepts, so the whole file became unreadable to any consumer that is not
  Python. It reports `null` now, and so does `spearman_rho` when scipy cannot
  compute it either.

  The serializer is a second layer: `_convert_for_json_public` turns any
  non-finite float into `null` wherever it appears, so no future metric can
  make the file invalid. The failure is silent at write time and total at read
  time, which is why it needed both. Caught by the first CI run on GitHub, on
  all three Python versions.

### Fixed (metric correctness)
- Operator isolation resolves overlapping token ranges to the later token, the
  rule the digit and AST paths already used, and `compute_per_text` scores
  operators on offsets rather than on a reconstruction. These were the last two
  sites still inferring which token covers which source character by rebuilding
  text from token strings.

  XLM-RoBERTa is the only one of the nine benchmark tokenizers that moves,
  because its word-start marker reports a range overlapping the first content
  character: pooled isolation 0.6948 to 0.6770 and compound preservation 0.7222
  to 0.8443. The other eight also report overlapping ranges, but only on
  multi-byte characters split across byte tokens, which no operator regex
  matches.

  `compute_per_text` disagreed with `compute()` because the reconstruction drops
  the space in `"! ="`, so the regex read one `"!="` where the source has two
  separate operators. Over the bundled math and synthetic code samples with nine
  tokenizers, 3078 rows, 66 changed.
- The three digit metrics map a source span to tokens through the tokenizer's
  own offsets, the same way the code metrics do since earlier in this release.
  They used the reconstruction path, which resynchronizes onto a residual space
  left by a multi-space token and stays wrong from there. The bundled math
  corpus and FLORES prose contain no runs of whitespace, so the published
  figures were not affected, but a corpus with indentation was: on four indented
  Python snippets holding 14 numbers, Llama 3 measured 1 of 14 and got that one
  wrong, returning boundaries [1, 3, 4, 7] for `12345678` where the tokens
  `123`, `456`, `78` put them at [3, 6]. After the change all 14 are measured and
  every boundary matches a direct read of the offsets.

  Eight of the nine benchmark tokenizers are byte-identical. `bert-base-uncased`
  moves, because it lowercases: three numbers whose digits never mapped are now
  measured, so `numbers_analyzed` goes from 624 to 627 and `avg_f1` from 0.6269
  to 0.6255.

  Where two token ranges overlap the digit path takes the later token. XLM-R
  encodes `1234567` as the word-start marker plus `1234` and `567`, with ranges
  (0,1), (0,4) and (4,7), so the marker claims the first digit and the
  earlier-token rule invented a boundary the tokenizer did not make, on 26 of
  627 numbers.
- A `.txt` or `.json` corpus is segmented one way, not several ways at once.
  `extract_texts_with_fallback_strategies` ran its strategies additively despite
  the name, so the same content came back twice under two segmentations and
  every per-document metric was computed over a corpus twice the size of the
  file. A file of 12 paragraphs of 3 lines returned 48 texts: the 12 paragraphs,
  then all 36 lines inside them. A file of fewer than 10 lines returned each
  line twice, once from the line split and once from the sentence split, which
  differ by a trailing period so the duplicate check missed them: the four-line
  corpus in the README Quick Start became 7 texts and every number was computed
  over it. A line-per-document file of 10 or more lines with no blank line was
  correct, which is why FLORES+ and the committed benchmark never showed it and
  their figures do not change. Precedence is now paragraphs when the file has
  blank lines, otherwise lines, then sentences, then fixed-size chunks.
- MorphScore reports `null` on its exception path too. The zero-languages branch
  was fixed to report null earlier in this release and the `except` handler 60
  lines below still wrote 0.0 to all four fields, recording a tokenizer whose
  evaluation raised as scoring worst on every axis.
- `generate_custom_latex_table` passes `caption` and `label` through. Both were
  accepted, documented, and commented out at the call site, so a caller
  following the docstring got neither.
- `ast_boundary_alignment` and `identifier_fragmentation` map a source span to
  tokens through the tokenizer's own offsets, which the code already fetched and
  passed to the indentation metric, instead of reconstructing the text from
  token strings and matching it back. The reconstruction path published wrong
  numbers for any tokenizer that emits a multi-space token.

  `_process_token` removes one leading space, not all of them, so `ĠĠĠ` cleans
  to two spaces that enter the reconstruction with no source counterpart.
  `_build_source_to_recon_map` then resynchronizes on a divergence by scanning
  up to 32 characters ahead, finds one of those residual spaces, and stays wrong
  from there. A tokenizer whose reconstruction holds no spaces at all has no
  jump target and stays correct, which is the whole difference between the two
  groups. Llama 3, OLMo 2, Qwen 2.5 and Mistral NeMo group indentation into one
  learned merge; GPT-2 emits separate single-space tokens; GPT-NeoX and Gemma
  hold their whitespace runs in `added_tokens_decoder`, where they are deleted.

  Measured over 629400 AST spans from 1500 real source files in 15 languages.
  `full_alignment_rate`: Llama 3 0.127 to 0.519, OLMo 2 0.127 to 0.519, Qwen 2.5
  0.128 to 0.519, Mistral NeMo 0.136 to 0.546, BERT 0.416 to 1.000, Gemma 2
  0.634 to 0.756, GPT-NeoX 0.742 to 0.768, GPT-2 0.768 to 0.787, XLM-RoBERTa
  0.893 to 0.893. `identifier_fragmentation.unmappable` of 204184 identifiers:
  119560 to 0 for Llama 3, 95937 to 0 for BERT, 4215 to 0 for GPT-2, and 0 to 0
  for XLM-RoBERTa, which was the one tokenizer the old path handled.

  An unmappable span was scored as a missed boundary rather than excluded, which
  is why `count` was identical for every tokenizer and the failure did not show.
  Unmappable spans are now counted and reported beside `count`, and a warning
  names the tokenizer. On this corpus 198 spans per tokenizer remain unmappable,
  0.03%, and every one is an AST literal whose source text is a single space.

- Offsets are normalized before use: one word-start space is dropped from each
  token's range, a whitespace-only token is kept whole, and a leading newline or
  tab is kept. Without this the metric would read `trim_offsets`, a ByteLevel
  post-processor flag that changes the reported offsets and not the
  tokenization. GPT-2 ships it false and GPT-NeoX true, which alone moved GPT-2
  from 0.433 to 0.770 with byte-identical token ids.

- Known limitation, not fixed: the digit metrics in `metrics/math.py` use the
  same reconstruction path. The bundled math corpus and FLORES prose contain no
  runs of whitespace, so no tokenizer produces a residual-space token on them
  and the measured values are unaffected. A math or prose corpus containing
  indentation would trigger the same defect.
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
- The two per-tokenizer caches in `metrics/base.py` keyed on `id(tokenizer)`
  without holding a reference, and CPython recycles the id of a freed object.
  Reachable through the public `per_example` API, whose module-level singletons
  outlive caller-supplied tokenizers: 4 of 40 calls read another tokenizer's
  special-token set and 20 of 40 read another's reverse vocabulary. Both now
  store the tokenizer alongside the value and check identity on read.
- A readable but empty `added_tokens` list was read as "this tokenizer declares
  no special tokens", so `tokenizers/unigramlm.json` resolved to an empty set
  although `<unk>`, `<s>`, `</s>` and `<pad>` sit at ids 0 to 3. The model's
  unknown token is now consulted as well, and an empty result falls back to
  `GENERIC_SPECIAL_TOKENS` with a warning.
- `--generate-latex-tables` and `--custom-latex-config` exit 1 when the table
  could not be written, naming the flag and the path. Both logged the error and
  then printed `Results saved to: ...` with exit 0. The results file is still
  written and still reported, so only the exit code and a closing error line
  change.
- A malformed `--code-ast-config` is rejected up front, naming the flag, the
  file and the expected shape. A JSON array was swallowed at one call site
  (`code_texts = {}`, leaving the operator-isolation code domain empty) and
  uncaught at the other (`AttributeError: 'list' object has no attribute
  'items'`, mentioning neither the flag nor the file).
- A code-data path that does not exist aborts instead of dropping that language,
  matching the natural-language loader. A code config that reads no snippet at
  all is an error rather than a silent switch to the bundled synthetic samples:
  measured 0.562 full AST alignment on synthetic against 0.493 on StarCoder for
  the same tokenizer.
- `RawTokenizationProvider` copies the `InputSpecification` objects it is given.
  It released the corpus by setting `texts = None` on the caller's own objects,
  which left each one neither raw nor pre-tokenized: `is_raw_mode` returned
  False afterwards and reusing the specification raised `ValueError:
  Specification for <name> is not in raw mode`.
- A missing parquet engine is reported once, as an environment problem, instead
  of being folded into the per-language load report. The CLI prints it without
  a traceback.
- `SentencePieceTokenizer.get_special_token_ids()` reads the model's own
  declared ids. It inherited the base `set()`, so the callers that exclude
  special ids (the unused-vocabulary statistic, the visualizer, the sanity
  checker) counted `<unk>`, `<s>` and `</s>` as ordinary vocabulary. The id and
  string forms now come from one scan. On a model trained with
  `--control_symbols <|im_start|> <|im_end|>`: `set()` before, six ids after.
- `SentencePieceTokenizer.get_special_token_strings()` returns `None` when every
  probe failed, so the caller warns and falls back to `GENERIC_SPECIAL_TOKENS`.
  It returned an empty set, which asserts that the model declares no special
  tokens.
- `include_empty_splits` is honoured by `newline_split` line counting, which
  counted blank lines whatever the flag said. Only configs selecting
  `newline_split` change value; none of the four in `configs/` does. Measured on
  a 5-line text with 2 blank lines: 3 with the flag off, 5 with it on.
  `word_counting: hf_whitespace` with `include_empty_splits: true` is now
  rejected at config load: the HuggingFace Whitespace pretokenizer does not emit
  empty pieces, so the flag was a no-op there.
- `--filter-script-family` and `--filter-resource-level` name the unknown group
  and list the ones the config defines. Both exited with `No valid language
  texts loaded`, which said neither. A language in the selected group with no
  `data_path` is now an error rather than a silent drop.
- `--language-config` pointed at a directory exits with a message saying a JSON
  file is expected, instead of an unhandled `IsADirectoryError`.
- The CLI prints config and corpus errors without a traceback. The message
  already names the flag, the file and what was expected; every other exception
  keeps its stack.
- A subword marker is stripped only from a tokenizer shown to use it. The
  WordPiece `##` prefix, the CLIP-style `</w>` suffix and the subword-nmt `@@`
  suffix were stripped from every tokenizer, so a byte-level BPE had ordinary
  content truncated: `###` became `#`, a 16-character `#` banner lost two
  characters, and `@@` became the empty string. Vocabulary entries matching a
  marker pattern: 35 in `cl100k_base`, 24 in `o200k_base`, 1 in the bundled
  `tokenizers/bpe.json`, none of which uses any of the three. The marker set is
  read from the backend model's `continuing_subword_prefix` and
  `end_of_word_suffix`, then from a probe encoding; when neither answers,
  nothing is stripped. On the demo run this moved 16 values in
  `ast_boundary_alignment` and `identifier_fragmentation`, all for the R sample,
  whose comments start with `##`.
- `avg_tokens_per_line` (now `compression_rate.tokens_per_line`) counts lines
  with `str.splitlines()`. Blank lines were dropped from the denominator while
  their newline tokens stayed in the numerator, so `"a\n\nb\n\n"` with 4 tokens
  reported 2.0 against the 1.0 its four lines give. Line-per-item corpora such as
  FLORES are unaffected; document corpora are not. A text with no content reports
  null rather than 0.0.
- `InputSpecification.get_vocab_size()` works. Both of its branches raised
  `AttributeError`: the raw one read `tokenizer.vocab_size`, which
  `TokenizerWrapper` does not define, and the pre-tokenized one read
  `self.vocabulary`, which is None for the `tokenizer + tokenized_data` shape the
  package itself builds. Unreachable from the CLI, reachable from the API.
- A run with no `--math-data` and no `--use-builtin-math-data` warns that the
  digit metrics are being computed on the prose corpus. On FLORES the observed
  digit lengths are 1 to 4, so the metric named for three-digit grouping never
  sees an ideal boundary and 74.2% of the sample falls in the vacuous
  length-3-or-under case. The warning matches the one `--code-ast-config`
  already prints.
- A corpus path that does not exist raises `FileNotFoundError` naming it,
  instead of logging a warning and returning an empty list. The caller reported
  "no texts read from <path>" for the empty result, which reads as an empty file
  rather than as a path that is not there.
- A relative `data_path` in a language config is resolved against the package
  root (the directory holding `tokenizer_analysis`, which is the repository root
  in a source checkout) rather than the process working directory. The same
  absolute `--language-config` used to load 5 languages from the repository root
  and 0 from `configs/`. A relative tokenizer `path` is rewritten only when it
  does not exist in the working directory and does exist under the package root,
  so a Hub model id and a local file both keep working. `--input` is unchanged:
  it is a command-line path and stays relative to the working directory.
  `--use-sample-data` now runs from any directory in a source checkout.
- `indentation_consistency` publishes a per-tokenizer `global`. It is the
  micro-averaged value: one Spearman correlation over the pooled (depth,
  whitespace-token count) pairs of every programming language, not the mean of
  the per-language correlations. On the demo the two differ: pooled 0.7598
  against a per-language mean of 0.8121 for `bpe`, and pooled -0.2427 against
  -0.2793 for `unigramlm`. Indent conventions differ by language, so the pooled
  value depends on the language mix of the code corpus; the per-language block is
  where each language is read separately.
- `operator_isolation_rate` publishes `global` and `by_domain` in
  `analysis_results.json`. Both existed in the full results and were dropped from
  the slim file, which carried only `per_language`. The global pools prose, code
  and math by operator instance, so the domain that supplies the most operators
  pulls it hardest. Measured with `tokenizer-analysis --use-sample-data`: pooled
  0.7938 over 3016 instances, against 0.6832 over 1932 for code, 0.9886 over 787
  for prose and 0.9966 over 297 for math. Code supplies 64% of the instances.
  The figures move with `--samples-per-lang`, which changes the prose corpus.
- `numeric_magnitude_consistency` fits each digit-length bucket at its own mean
  digit length and mean token count. The open `10+` bucket was fitted at exactly
  10 digits with a token count reconstructed as `mean_fertility * 10`, so a
  20-digit number costing 10 tokens entered the fit at (10, 5.0). On numbers
  lying exactly on `tokens = 0.5 * digits + 1.0` with lengths 2, 4, 6, 8, 12 and
  20, the fit returned slope 0.4667 and R-squared 0.9949; it now returns 0.5 and
  1.0. Each bucket publishes `mean_digit_length` so the fit is auditable. The fit
  rests on at most 10 points, so slope and R-squared stay coarse.
- Operator isolation rates with a zero denominator report `null` rather than 0.0,
  matching the rest of the pipeline. The prose domain of the demo has no compound
  operator, and reported a compound-preservation rate of 0.0.
- A LaTeX standard-error cell used `\pm` in a non-raw f-string, which Python
  3.12 and 3.13 report as `SyntaxWarning: invalid escape sequence '\p'` on
  import. The classifiers and the CI matrix now cover 3.12 and 3.13, both of
  which the package installs and imports on.
- `--run-grouped-analysis` writes its results to `analysis_results.json`. The
  slimming step keys on `per_tokenizer`, and grouped results are one level
  deeper, so the slim file published `"grouped_analysis": {}` and the whole
  grouped result was lost unless `--save-full-results` was also passed. Group
  blocks are also folded by `merge_redundant_metrics` now, so a group and the
  whole-corpus block carry the same metric keys; a group still published
  `type_token_ratio` and `avg_tokens_per_line` after they had been merged away
  everywhere else.
- The CER time budget is checked on elapsed time as well as on call count. It
  only fired after 50 non-exact texts, and one call is an edit distance over two
  long strings, so a lossy tokenizer could spend minutes inside a single call and
  never reach the 50th: measured on `bert-base-uncased` over 5035 texts, the
  warmup ran past 10 minutes with the budget set to 120 seconds, which reads as a
  hung run. The budget now also fires as soon as the warmup has spent it.
- The CER time budget has one default, `DEFAULT_CER_TIME_BUDGET_S = 10.0`. The
  CLI used 10.0 and `UnifiedTokenizerAnalyzer.run_analysis` used 30.0, so the
  same corpus gave a different answer depending on which entry point started it.
- `--use-builtin-math-data` help text said the bundled dataset holds about 100
  expressions. It holds 285.
- `--run-grouped-analysis` works again. It read `digit_split_variability` as a
  top-level key after the metric merge had nested it under
  `three_digit_boundary_alignment`, so the flag exited 1 with a bare `KeyError`.
  The nested block is also language-filtered now; passing it through untouched
  gave each language group numbers computed over every language in the run.
- The grouped-plot loop no longer wraps its calls in `except Exception`. It was
  the only guarded call among roughly fifteen in `generate_all_plots`, so a real
  failure there logged a warning and returned normally while the same failure in
  any other plot propagated.
- The C16 vocabulary-reachability check scans in token-id order, so its example
  list is the same on every run. It iterated `get_vocab()`, whose order varies
  between processes, so two reports on the same tokenizer showed different
  examples beside identical counts.
- `_fill_offsets` in the visualizer no longer takes an unused `text_len`
  argument.
- MorphScore reports `null` rather than `0.0` for a tokenizer it evaluated on
  zero languages, which happens whenever the data directory is missing or holds
  no CSV for any requested language. The run exits 0 and otherwise looks
  normal, so four zeros beside `languages_evaluated: 0` read as a tokenizer
  scoring worst on every axis.
- MorphScore's missing-library errors name the install steps (submodule init,
  editable install, separate dataset download) instead of only the package.
- A missing parquet engine raises `ParquetEngineMissing` naming
  `uv sync --extra parquet`, instead of every parquet read returning an empty
  list. Three layers of `except Exception` used to swallow it, so a user
  without the optional extra got a run that completed with zero data for every
  parquet source.
- `indentation_consistency` counts only whitespace-only tokens as indentation.
  It previously counted every token overlapping the leading-whitespace range,
  which pulled in the first code token whenever that token also covered the last
  indent space. That happens when the pre-tokenizer groups a leading space with
  the following word and a merge for the pair was learned; it depends on the
  pre-tokenizer and the merges, not on byte-level encoding as such.

### Added
- `run_metadata.arguments` records every command-line argument the caller
  changed from its default, except the ones that only decide where output goes.
  The block named a hand-maintained list before, and `--filter-script-family`,
  `--filter-resource-level` and `--use-builtin-math-data` were not on it: all
  three change the published values and all three left `run_metadata`
  byte-identical. Two runs on a 3-language config differing only by
  `--filter-script-family Latin` gave Gini 0.005152 and 0.005774 with the same
  provenance block. Diffing the parsed namespace against the parser's defaults
  covers any flag added later without anyone remembering to add it.
- `--max-code-files-per-lang` and `--max-code-file-chars`, both defaulting to 0,
  which means no cap. They were previously always on at 100 files per language
  and 15000 characters per file, on no flag and recorded nowhere: on the
  benchmark's own corpus that discarded 6537749 of 12753776 characters, 51.3%,
  while the run reported measuring 1500 files. The effective values are written
  to `run_metadata.code_corpus_caps`, and a cap that drops or truncates anything
  logs a warning naming the language and the counts.
- `scripts/fetch_flores.py`, which downloads the FLORES+ corpus the configs
  name. Needs the `flores` extra (`uv sync --extra flores`) and a Hugging Face
  login, since the dataset is gated.
- `TOKEVAL_PARSE_TIMEOUT_S` overrides the per-language tree-sitter subprocess
  timeout, default 120 seconds. On a loaded machine a grammar can exceed the
  default and be dropped from the run: the php grammar timed out at 120 seconds
  on 3 snippets during a concurrent test run.
- `sentencepiece` is declared as an optional extra
  (`uv sync --extra sentencepiece`). `SentencePieceTokenizer` is a first-class
  tokenizer class but the package was never listed, so it was installable only
  by accident and its tests skipped in a clean checkout.

### Changed (Python API, breaking)
- `TokenizerWrapper` subclasses must implement `get_special_token_strings()`.
  It is abstract, so a custom wrapper written against 0.x fails to instantiate
  until it is added. See MIGRATION.md.

### Removed
- The `morphscore_data` symlink, which was tracked and pointed at an absolute
  path on the author's cluster. It resolved only there, it published that
  directory layout, and it stopped a reader from creating the directory the
  README tells them to create.
- The FLORES+ corpus is no longer in the repository. `parallel/` is untracked
  and stripped from the history: FLORES+ is CC-BY-SA 4.0 and this project does
  not redistribute it. `scripts/fetch_flores.py` downloads it, the shipped
  configs and `--use-sample-data` are unchanged, and a run that names a file
  which is not there aborts repeating the fetch command rather than proceeding
  on a smaller corpus. Nothing in the library requires FLORES+: `--input` and
  `--language-config` take your own corpus.
- `--test`. It built the whole analyzer, logged `Test methods not yet updated
  for unified system`, and exited 0 without running anything.
- `indentation_consistency.pattern_stability_rate` and its
  `avg_pattern_stability_rate` summary. Once only whitespace-only tokens were
  counted, it was 1.0 for 11 of 12 tokenizers measured and took two distinct
  values in total. That is what theory predicts: a deterministic tokenizer
  encodes a fixed whitespace string one fixed way, so the rate can only drop
  when two different indent widths map to the same depth, which is a property of
  the source rather than of the tokenizer. The spread it showed beforehand came
  from counting the first code token, which made it measure which word each line
  started with. `depth_proportionality_correlation` is unaffected.
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

## [1.0.0] (unreleased), the consolidation itself

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
