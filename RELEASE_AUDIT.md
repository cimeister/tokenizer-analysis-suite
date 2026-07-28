# 1.0 release audit

Findings from the pre-open-source audit. Each is classified by failure mode,
because that decides urgency:

- **silent**: the run completes and publishes a number a reader cannot
  distinguish from a real measurement. Worst class.
- **loud but unhelpful**: exits non-zero, but the message does not identify the
  cause.
- **loud**: exits non-zero and names the artifact. Working as intended.

Status is `fixed`, `open`, or `wontfix` with a reason. Every item was reproduced
by execution before being listed.

---

## silent

### S1. `--use-sample-data` overrode the user's configs — **fixed**
`cli/run_analysis.py`. `--use-sample-data --tokenizer-config mine.json
--language-config mine.json` exited 0 and analyzed the two bundled demo
tokenizers over five FLORES+ languages. The results file was complete and
described different tokenizers on a different corpus than the ones named, with
no warning. Now an error naming the conflicting flags.

### S2. Omitting `--language-config` analyzed the demo corpus — **fixed**
`cli/run_analysis.py`. The raw-tokenizer path fell back to the bundled
five-language sample. Now an error listing `--input`, `--language-config` and
`--use-sample-data`.

### S3. A missing corpus file dropped the language — **open**
`loaders/multilingual_data.py:81-83`. A two-language config with one
nonexistent `data_path` exits 0, logs two warnings, and writes a
complete-looking results file containing only the surviving language. On a
13-language config, one typo yields a 12-language result that reads as
finished. Nothing in the JSON records that a language was requested and
dropped.

### S4. Missing MorphScore data scores 0.0 — **open**
`metrics/morphscore.py`. A nonexistent `--morphscore-data-dir` exits 0 and
writes `avg_morphscore_recall: 0.0`, `avg_morphscore_precision: 0.0`,
`avg_micro_f1: 0.0`, `avg_macro_f1: 0.0`. A tokenizer that was never evaluated
is recorded as scoring zero on morphological alignment. The only tell is
`languages_evaluated: 0` beside four zeros that read as scores.

### S5. A single corpus reports perfect fairness — **open**
`metrics/gini.py:125-133`. One corpus gives `{"gini_coefficient": 0.0,
"mean_cost": 0.0, "std_cost": null, "num_languages": null, "warning": "..."}`
while the sibling `per_language` block in the same object records a cost of
0.2339. `mean_cost: 0.0` is not merely undefined, it is contradicted by the
neighbouring field. `--input` makes this the common case rather than an edge
case.

### S6. `empty_stats()` and `safe_divide` return 0.0 for absent data — **open**
`metrics/base.py:324-327,351-363`. "No data" and "measured zero" are the same
value throughout the aggregate pipeline. `per_example.py` uses `float("nan")`
for the same condition and `utf8_integrity.py:703-709` returns `None`, so three
conventions coexist.

### S7. A malformed `--custom-latex-config` reports success — **open**
`cli/run_analysis.py:1148`. A file the user named explicitly, containing
invalid JSON, logs `ERROR - Error generating custom LaTeX tables: ...` and then
prints `Results saved to: ...`. Exit 0. Same shape at line 1099 for
`--generate-latex-tables`.

### S8. `tokenizer-visualize` exits 0 when every tokenizer fails — **open**
`cli/visualize_tokenization.py:586`. A config whose tokenizer paths do not
exist prints `Skipping <name>: ...` per tokenizer, then the source samples with
no tokenization at all, and exits 0. No count of how many of N tokenizers
loaded.

### S9. A list-shaped `--code-ast-config` silently drops code data — **open**
`main.py:99`. Catches `Exception`, logs `Could not load code data: 'list'
object has no attribute 'items'`, and continues with `code_texts = {}`. The
operator-isolation code domain then has zero samples with no further signal.
See L1 for the other half of this.

### S10. Grouped plots swallow failures, individual plots do not — **open**
`visualization/plots.py:579`. The only guarded call among roughly 15 in
`generate_all_plots`. A malformed grouped result logs `Failed to plot <metric>
for group type <type>` and returns normally; the same failure in any other plot
propagates. Same failure class, opposite handling.

### S11. Missing parquet engine yields an empty corpus — **open**
`loaders/code_data.py:225-229`, `loaders/multilingual_data.py:241,269`.
pandas raises an `ImportError` that names pyarrow and fastparquet, but both
call sites catch it, log, and return `[]`. A user without the `parquet` extra
gets a run that completes with silently zero data for every parquet source.

### S12. `include_empty_splits` is ignored for two counting methods — **open**
`config/text_measurement.py`. The flag is honoured by `python_split`,
`regex_whitespace` and `custom_regex` line counting, but not by
`hf_whitespace` word counting or `newline_split` line counting. Setting it
there is a silent no-op. Measured on `"  hello   world  "`: `regex_whitespace`
gives 2 then 4 as the flag flips; `hf_whitespace` gives 2 both times.

### S13. Config paths resolve against the process CWD — **open**
`config/language_metadata.py`, `loaders/multilingual_data.py`. `data_path` is
used verbatim, never joined to the config file's own directory. The same
absolute `--language-config` loads 5 languages from the repo root and 0 from
`configs/`, logged as warnings, exit 0. Tokenizer `path`, `code_ast_config`
values and `math_data_path` share the behaviour. This is also why
`--use-sample-data` only works from a source checkout.

### S14. CRLF and BOM are preserved into the AST metrics — **open**
`loaders/code_data.py:172-192`. Files are opened in binary and decoded per
line, so there is no universal-newline translation and `utf-8` does not strip a
BOM. A Windows-authored source file reaches tree-sitter with embedded `\r`, and
a BOM-prefixed file with a leading `﻿`, both feeding the AST-boundary and
indentation-consistency metrics as if they were part of the code's layout.

### S15. `RawTokenizationProvider` mutates the caller's specs — **open**
`core/input_providers.py:111-117`. `spec.texts = None` is applied to the dict
the caller passed in, not a copy. After one `get_tokenized_data()` call the
caller's `InputSpecification` is neither raw nor pre-tokenized, so its own
validator reports it invalid and `spec.get_languages()` raises `TypeError`.

---

## loud but unhelpful

### L1. A list-shaped `--code-ast-config` crashes with a raw `AttributeError` — **open**
`main.py:145-151`. The narrow `except (ImportError, ValueError)` does not catch
the `AttributeError` from `CodeDataLoader.load_all()`. Exit 1 with
`AttributeError: 'list' object has no attribute 'items'`: no mention of
`--code-ast-config`, the file path, or that an object was expected. The same
input is swallowed 50 lines earlier (S9). Validate the shape once, up front.

### L2. `--filter-script-family` with an unknown name — **open**
`cli/run_analysis.py`. `--filter-script-family Klingon` exits 1 with `No valid
language texts loaded`, which never says the group name was unknown or lists
the valid ones.

### L3. `--language-config` pointed at a directory — **open**
`config/language_metadata.py:39-47`. Raises an unhandled `IsADirectoryError`,
not one of the two errors `_load_config` handles. A comment at
`cli/run_analysis.py:1029` claims directories are supported; no code path does.

### L4. MorphScore's ImportError gives no install instruction — **open**
`metrics/morphscore.py:20,72`. Names the package but not how to get it, unlike
`sentencepiece` and `script_bpe`, which both name the exact `pip install`.

---

## loud (working as intended)

- Missing tokenizer file: exit 1, `Could not load tokenizer from <path>`.
- Malformed language config JSON: exit 1, naming the parse position.
- Unknown `text_measurement` key: exit 1, naming the key and listing valid ones.
  The best error message in the codebase; the model for the others.
- Removed CLI flags: exit 2, naming the replacement and MIGRATION.md.
- `--pairwise` with an unknown tokenizer: exit 1, naming it.
- Empty or whitespace-only corpus: exit 1.
- Vocabulary export failure: `RuntimeError` naming the tokenizer, with
  `raise ... from e` preserving the cause.

---

## Correctness and dead code

### C1. `metrics/base.py:9,317` uses `scipy.stats` without importing it — **open**
Bare `import scipy` at line 9, `scipy.stats.sem(...)` at line 317. This works
only because `metrics/__init__.py` imports `.math`, which does `from
scipy.stats import spearmanr` at module scope, before anything can call
`compute_basic_stats`. scipy gained lazy submodule loading in 1.9.0; the
declared floor is 1.7.0. Add the explicit import.

### C2. `core/validation.py` is entirely unreachable — **open**
`ValidationResult`, `TokenizedDataValidator`, `InputProviderValidator`,
`InputSpecificationValidator`, `AnalysisValidator` and `validate_and_report`
are defined, exported from nowhere, and imported by nothing. The
`InputValidator` that `main.py:68-73` actually uses is a different class in
`core/input_utils.py`. 472 lines of code that never runs, including the only
logic that handles the single-language case explicitly.

### C3. `InputSpecification.get_vocab_size()` crashes on its own documented shape — **open**
`core/input_types.py:177-182`. The pre-tokenized branch reads
`self.vocabulary.vocab_size`, which is `None` for the `tokenizer +
tokenized_data` shape that `main.py:create_analyzer_from_tokenized_data`
constructs. Unreachable through the CLI, reachable by any API caller.

### C4. Dead branches — **partly fixed**
`_build_indentation_consistency_results` never writes `overall`, so the slim
branch testing for it is dead (open, see the schema section). The fertility
zero-sample guard at `basic.py:171` cannot fire, because `"你好".split()`
returns one element (open). The plural-key branches in
`loaders/multilingual_data.py:47-50` are vestigial (open, harmless).

---

## Output schema

Measured on a demo run: 9 of 23 metrics have no `global` key under
`per_tokenizer.<tok>`, against a README that documents one for all of them.
They are `encoding_speed`, `token_length`, `trigram_entropy`,
`lorenz_curve_data`, `three_digit_boundary_alignment`,
`digit_split_variability`, `numeric_magnitude_consistency`,
`operator_isolation_rate` and `indentation_consistency`.

`bigram_entropy` nests its global under `global`; `trigram_entropy` emits flat
`global_trigram_entropy` siblings, because `slim_results_for_json` has a pivot
case for one and not the other.

`global` also means three different things with nothing recording which: a
ratio of sums (`compression_rate`), a mean of per-document ratios
(`fertility`), an unweighted mean across languages (`tokenizer_fairness_gini`),
and a set union (`vocabulary_utilization.global_utilization`).

`operator_isolation_rate.per_language` keys natural languages (`arb_Arab`,
`cmn_Hani`) and programming languages (`bash`, `c`, `cpp`) in one dict.

No results file records the package version, git commit, config paths, input
hashes or sample count, so a results file cannot be traced to what produced it.

---

## Metric validity and overlap

Rank correlations are Spearman across 37 tokenizers on a 13-language FLORES+
corpus, unless stated. A 9-tokenizer run was also examined and discarded for
this purpose: at n=9, 110 of 153 pairs exceed |rho| 0.95, which is noise.

### Exact identities, not correlations

- `compression_rate` (lines) x `avg_tokens_per_line` = 1.000000 for every
  tokenizer, rho = -1.000. One measurement published under two names.
- `type_token_ratio.types` and `vocabulary_utilization.used_tokens` are the
  same integer for every tokenizer.
- `(ttr / vocab_utilization) x (total_tokens / vocab_size)` = 1.000000 for
  every tokenizer: TTR is a reparameterization of vocabulary utilization.
- `renyi_1.0 x log2(observed types)` = `unigram_entropy` to zero relative error.
- `compression_rate` under `characters`, `bytes` and `lines` correlate at
  exactly 1.0000, Spearman and Pearson. On a fixed corpus the numerator does not
  depend on the tokenizer, so the measurement method is a unit change and does
  not affect within-run rankings. The README presents the four methods as
  substantive analysis choices.

### Validity gaps

- `renyi_efficiency` normalizes by `log2(observed token types)`, not
  `log2(|V|)` as in Zouhar et al. 2023. The two rank tokenizers differently
  (rho 0.678, maximum rank shift 16 of 37). Uncited and no formula in the
  README.
- `bigram_entropy` and `trigram_entropy` deviate from the cited Poelman et al.
  2025 in four ways: the normalizer is the per-context successor count rather
  than the accessor-domain size (making the `min()` dead code), aggregation is
  frequency-weighted rather than unweighted over types, filtering is by raw
  count rather than the paper's punctuation/digit/boundary-ratio rules, and
  there is no windowing. The numbers should not be compared to published ones.
- `trigram_entropy` has no README entry at all, and its ranking is unstable
  under its undocumented `min_trigram_occurrences=3`: rho 0.728 between
  thresholds 3 and 25, against 0.985 for bigram at the same settings. 89% to
  90% of context types and a median 70% of occurrences are discarded.
- `fertility` pools per-document ratios across languages under whitespace word
  counting. 96.2% of Japanese and 69.3% of Chinese FLORES lines contain exactly
  one whitespace-delimited token, so for those languages it is tokens per
  sentence under another name. Median per-language fertility runs 1.261 (eng)
  to 38.195 (jpn). The pooled value correlates with fertility over the other 11
  languages at rho 0.401.
- `vocabulary_utilization` divides by a vocabulary size that includes special
  and added tokens which encoding, called with `add_special_tokens=False`, can
  never emit. Tokenizer-dependent: 1000 of 131072 for one, 4 for most.
- `avg_token_rank` correlates with total token count at rho -0.977 and has no
  scale-free definition.
- `reconstruction_fidelity` nulls `mean_cer` and `whitespace_fidelity` from a
  wall-clock budget, which fired for 1 of 37 tokenizers. The same command on a
  faster machine populates a different set of fields.
- `unk_token_rate = 0.0` means either no UNK id exists or no UNK was emitted.
  `whitespace_fidelity = 1.0` means either no loss or no whitespace present.

---

## Packaging and release

- `pyproject.toml` has no `license` or `license-files` field, so the wheel
  carries no license text. `NOTICE`, which carries the CC-BY-SA 4.0 attribution
  the bundled FLORES+ data requires, sits outside the package.
- `packages = ["tokenizer_analysis"]` ships `tokenizer_analysis/tests/` in the
  wheel.
- The demo resolves `tokenizers/bpe.json` and `parallel/eng_Latn.txt` relative
  to the working directory, so `--use-sample-data` only works from a source
  checkout. A pip install cannot run the documented quickstart.
- No `py.typed`, so the annotations are invisible to downstream type checkers.
- No `.github/`: no CI, and the 3.10 floor in `requires-python` has apparently
  never been executed.
- `tests/test_sanity_check.py:435` loads `test_tokenizers/test_bpe_tok-gpt4.json`,
  which is gitignored and untracked, so that test cannot pass in a fresh clone.
- `fastparquet` is in the `parquet` extra; pandas 3.1 deprecates that engine and
  the project is retiring.
- Both bundled demo tokenizers fail the bundled `tokenizer-sanity-check` (exit
  2; `bpe.json` covers 94 of 256 required byte values). The README's first two
  commands therefore contradict each other.
- Three names for one project: distribution `tokenizer-intrinsic-evals`, import
  `tokenizer_analysis`, README title `TokEval`. The README never shows the pip
  install line for the published name.
