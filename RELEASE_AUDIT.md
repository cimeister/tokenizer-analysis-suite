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

### S1. `--use-sample-data` overrode the user's configs: **fixed**
`cli/run_analysis.py`. `--use-sample-data --tokenizer-config mine.json
--language-config mine.json` exited 0 and analyzed the two bundled demo
tokenizers over five FLORES+ languages. The results file was complete and
described different tokenizers on a different corpus than the ones named, with
no warning. Now an error naming the conflicting flags.

### S2. Omitting `--language-config` analyzed the demo corpus: **fixed**
`cli/run_analysis.py`. The raw-tokenizer path fell back to the bundled
five-language sample. Now an error listing `--input`, `--language-config` and
`--use-sample-data`.

### S3. A missing corpus file dropped the language: **fixed**
`loaders/multilingual_data.py:81-83`. A two-language config with one
nonexistent `data_path` exits 0, logs two warnings, and writes a
complete-looking results file containing only the surviving language. On a
13-language config, one typo yields a 12-language result that reads as
finished. Nothing in the JSON records that a language was requested and
dropped.

### S4. Missing MorphScore data scores 0.0: **fixed**
`metrics/morphscore.py`. A nonexistent `--morphscore-data-dir` exits 0 and
writes `avg_morphscore_recall: 0.0`, `avg_morphscore_precision: 0.0`,
`avg_micro_f1: 0.0`, `avg_macro_f1: 0.0`. A tokenizer that was never evaluated
is recorded as scoring zero on morphological alignment. The only tell is
`languages_evaluated: 0` beside four zeros that read as scores.

### S5. A single corpus reports perfect fairness: **fixed**
`metrics/gini.py:125-133`. One corpus gives `{"gini_coefficient": 0.0,
"mean_cost": 0.0, "std_cost": null, "num_languages": null, "warning": "..."}`
while the sibling `per_language` block in the same object records a cost of
0.2339. `mean_cost: 0.0` is not merely undefined, it is contradicted by the
neighbouring field. `--input` makes this the common case rather than an edge
case.

### S6. `empty_stats()` and `safe_divide` return 0.0 for absent data: **fixed**
Plot rendering was fixed first (a null was drawn as a zero-height bar, and the
grouped plotter substituted 0 on any extractor failure), so the metric-side
change can now land without breaking the figures.
`metrics/base.py:324-327,351-363`. "No data" and "measured zero" are the same
value throughout the aggregate pipeline. `per_example.py` uses `float("nan")`
for the same condition and `utf8_integrity.py:703-709` returns `None`, so three
conventions coexist.

### S7. A malformed `--custom-latex-config` reports success: **fixed**
`cli/run_analysis.py`. A file the user named explicitly, containing invalid
JSON, logged `ERROR - Error generating custom LaTeX tables: ...` and then
printed `Results saved to: ...`. Exit 0. Same shape for
`--generate-latex-tables`. Both now record the failure, name the flag and the
path, finish the run so the results file is still written and reported, and
exit 1 listing the outputs that were requested and not produced.

### S8. `tokenizer-visualize` exits 0 when every tokenizer fails: **fixed**
`cli/visualize_tokenization.py`. A config whose tokenizer paths do not exist
printed `Skipping <name>: ...` per tokenizer, then the source samples with no
tokenization at all, and exited 0. It now prints `Loaded N of M requested
tokenizer(s).` on every run and exits 1 when N is zero. Re-measured on a config
naming one nonexistent path: exit 1.

### S9. A list-shaped `--code-ast-config` silently drops code data: **fixed**
`main.py`. Caught `Exception`, logged `Could not load code data: 'list' object
has no attribute 'items'`, and continued with `code_texts = {}`. The
operator-isolation code domain then had zero samples with no further signal.
The shape is now checked once in `CodeDataLoader._validate_config`, called from
`_resolve_code_ast_config` where the flag and the path are both known, and the
`try/except` around the load is gone: a code config the caller named either
loads or aborts. Same fix closes L1.

### S10. Grouped plots swallow failures, individual plots do not: **fixed**
`visualization/plots.py`. The only guarded call among roughly 15 in
`generate_all_plots`. A malformed grouped result logged `Failed to plot <metric>
for group type <type>` and returned normally; the same failure in any other plot
propagated. The handler is gone. `plot_grouped_analysis` already returns early
and logs when a metric is absent from the grouped results, which is the case the
handler was covering.

### S11. Missing parquet engine yields an empty corpus: **fixed**
`loaders/code_data.py:225-229`, `loaders/multilingual_data.py:241,269`.
pandas raises an `ImportError` that names pyarrow and fastparquet, but both
call sites catch it, log, and return `[]`. A user without the `parquet` extra
gets a run that completes with silently zero data for every parquet source.

### S16. A missing code-data path dropped that language: **fixed**
`loaders/code_data.py`. `load_all` logged `Code data path not found for
<lang>: <path>` and continued, so the code metrics were computed over fewer
languages than the config named with nothing in the output recording the
difference. This is the same defect as S3 on the natural-language side and now
has the same resolution: the run aborts and names every configured path that
does not exist.

### S17. A code config that read no snippet fell back to synthetic samples: **fixed**
`metrics/code_ast.py`. `ASTBoundaryMetrics.__init__` substituted the bundled
synthetic snippets whenever `code_loader.code_snippets` came back empty, which
included the case where the caller passed a real config whose directories held
no matching file. Synthetic samples are the documented input for an empty
config only. Measured divergence between the two sources for one tokenizer:
0.562 full AST alignment on synthetic against 0.493 on StarCoder. A non-empty
config that yields no snippet is now an error naming the configured languages.

### S18. SentencePiece models reported zero special token ids: **fixed**
`core/tokenizer_wrapper.py`. `SentencePieceTokenizer` did not override
`get_special_token_ids()`, so it inherited the base `set()`. The three callers
that exclude special ids (the unused-vocabulary statistic in `metrics/basic.py`,
the visualizer, the sanity checker) counted `<unk>`, `<s>` and `</s>` as
ordinary vocabulary entries. Both the id and the string form now come from one
scan, `_declared_special_ids`, so they cannot disagree. Measured on a
SentencePiece model trained with `--control_symbols <|im_start|> <|im_end|>`:
ids `set()` before, `{0, 1, 2, 3, 4, 5}` after, matching the six strings the
string form reports.

### S20. Grouped results carried an unfiltered copy of a merged block: **fixed**
`main.py`. `_filter_digit_boundary_results` passes keys it does not recognize
through untouched. The 1.0 merge nested `digit_split_variability` under
`three_digit_boundary_alignment` as `split_variability`, which made it one of
those keys although it holds per-language numbers, so each language group would
have inherited values computed over every language in the run. Nested blocks of
the same shape are now filtered by the same rule. This was masked by L5 below,
which crashed first.

### S19. The C16 example list changed between identical runs: **fixed**
`diagnostics/sanity_check.py`. C16 iterated `get_vocab()`, a dict built from the
tokenizer library's own hash map, so its order varies between processes. The
bucket counts were stable but the `examples` list differed on every run.
Measured on `tokenizers/bpe.json` under three values of `PYTHONHASHSEED`: three
different example lists before, one identical entry after. The scan is now
ordered by token id.

### S12. `include_empty_splits` is ignored for two counting methods: **fixed**

`config/text_measurement.py`. The flag was honoured by `python_split`,
`regex_whitespace` and `custom_regex` line counting, but not by `hf_whitespace`
word counting or `newline_split` line counting. Setting it there was a silent
no-op. Measured on `"  hello   world  "`: `regex_whitespace` gives 2 then 4 as
the flag flips; `hf_whitespace` gave 2 both times. `newline_split` now honours
it (measured on a 5-line text with 2 blank lines: 3 with the flag off, 5 with
it on), and `hf_whitespace` with `include_empty_splits: true` is rejected at
config load, because the HuggingFace Whitespace pretokenizer does not emit
empty pieces and so cannot honour it. Only configs that select `newline_split`
change value; the four configs in `configs/` do not.

### S13. Config paths resolve against the process CWD: **fixed**
`config/language_metadata.py`, `core/tokenizer_wrapper.py`. `data_path` was
used verbatim, so the same absolute `--language-config` loaded 5 languages from
the repository root and 0 from `configs/`, logged as warnings, exit 0.

A relative `data_path` is now anchored at `PACKAGE_ROOT`, the directory holding
the `tokenizer_analysis` package, which in a source checkout is the repository
root. That is unconditional: a corpus path decides what the numbers describe,
so it has to name the same files from every working directory, and a path that
does not resolve is an error naming the anchor rather than a dropped language.

A tokenizer `path` cannot be anchored unconditionally, because a Hub model id
such as `meta-llama/Meta-Llama-3-8B` is also a relative-looking string that has
to reach the loader unchanged. The order there is: absolute paths and paths
that exist relative to the working directory are left alone, so a local file is
never overridden by a package-root file of the same name; a path that exists
only under the package root is rewritten and the log line says so; anything
else is passed through.

`--input` stays relative to the working directory, since it is a command-line
path rather than a config entry, and is made absolute before it reaches the
loader.

Measured: `--use-sample-data` and `--tokenizer-config <abs> --input
local_corpus.txt` both exit 0 from an unrelated working directory; before this
both exited 1 with `Could not load tokenizer from tokenizers/bpe.json`. The
demo still needs a source checkout, because `parallel/` and `tokenizers/` are
not shipped inside the wheel.

### S14. CRLF and BOM are preserved into the AST metrics: **fixed**
`loaders/code_data.py:172-192`. Files are opened in binary and decoded per
line, so there is no universal-newline translation and `utf-8` does not strip a
BOM. A Windows-authored source file reaches tree-sitter with embedded `\r`, and
a BOM-prefixed file with a leading `﻿`, both feeding the AST-boundary and
indentation-consistency metrics as if they were part of the code's layout.

### S15. `RawTokenizationProvider` mutates the caller's specs: **fixed**
`core/input_providers.py`. `spec.texts = None`, which releases the corpus after
encoding, was applied to the objects the caller passed in. Measured before the
fix: after one `get_tokenized_data()` call the caller's `InputSpecification`
reported `is_raw_mode` False, and constructing a second provider from it raised
`ValueError: Specification for bpe is not in raw mode`. The provider now holds
`dataclasses.replace` copies, so it releases its own reference and the caller's
object is unchanged. Re-measured: `is_raw_mode` True, `texts` intact.

---

## loud but unhelpful

### L1. A list-shaped `--code-ast-config` crashes with a raw `AttributeError`: **fixed**
`main.py`. The narrow `except (ImportError, ValueError)` did not catch the
`AttributeError` from `CodeDataLoader.load_all()`. Exit 1 with
`AttributeError: 'list' object has no attribute 'items'`: no mention of
`--code-ast-config`, the file path, or that an object was expected. Now exits 1
with `--code-ast-config <path>: code_config must be an object mapping a
language name to a file or directory path, got list`, and no traceback. The AST
init handler catches only `ImportError` (tree-sitter absent), so a bad config is
no longer reported as "AST boundary metrics disabled".

### L5. `--run-grouped-analysis` exited 1 with a bare `KeyError`: **fixed**
`main.py`. `run_analysis` folds six metrics into their primaries before
returning, and `run_grouped_analysis` then filtered those base results while
still reading the pre-merge top-level key: `KeyError:
'digit_split_variability'`, naming no flag. The flag was broken outright from
the merge onwards, and nothing in the suite exercised it. Now reads the merged
location, and a grouped run is covered by
`test_output_contract.py::test_grouped_analysis_runs_after_the_metric_merge`.

### L2. `--filter-script-family` with an unknown name: **fixed**
`loaders/multilingual_data.py`. `--filter-script-family Klingon` exited 1 with
`No valid language texts loaded`, which never said the group name was unknown
or listed the valid ones. It now exits 1 with `No language in the config has
script_family=Klingon. The config defines: Arabic, Cyrillic, Devanagari,
Han_Simplified, Japanese, Korean, Latin.` and no traceback. A language in the
selected group with no `data_path` is now an error too, on the same grounds as
S3.

### L3. `--language-config` pointed at a directory: **fixed**
`config/language_metadata.py`. Raised an unhandled `IsADirectoryError`, not one
of the two errors `_load_config` handles, and a comment in `cli/run_analysis.py`
claimed directories were supported although no code path read one. Now exits 1
with `Language metadata configuration must be a JSON file, but configs is a
directory. Name the file inside it.` and the comment is corrected. `main()`
prints this class of error without a traceback.

### L4. MorphScore's ImportError gives no install instruction: **fixed**
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

## Correctness

### X1. Alignment scan could not re-synchronize: **fixed, partly**
`metrics/base.py`. `_build_source_to_recon_map` advanced its reconstruction
pointer only on a match, so one character the reconstruction *adds* left it
stuck for the rest of the document. A byte-level vocabulary renders `é` as `Ã©`,
so the map died at the first non-ASCII character. Consumers score an unmappable
span as a miss rather than as unmeasured, so this dropped digit spans and marked
AST nodes misaligned: adding one accented comment to a Python snippet took its
alignment from 0.93 to 0.00.

Fixed with a bounded-window re-sync. Digit spans actually measured, demo corpus
at 200 lines per language: English 116 to 143 of 143 present, German 112 to 145,
Spanish 89 to 126, total 358 to 456.

**Superseded.** Tuning the aligner was the wrong fix. `encode_with_offsets`
returns exact character spans and `TokenizedData.offsets` was already populated
for the whole corpus; no metric was reading it. Every place that inferred a
source-to-token correspondence from token surfaces now uses offsets or raises.
See the offsets entries below.

### X11. Subword markers are stripped for every tokenizer family: **fixed**
`metrics/base.py` `_process_token`. The WordPiece and BPE marker rules (`##`
prefix, `</w>` suffix, `@@` suffix) were applied whatever the tokenizer, so for
a byte-level BPE they truncated real content: `_clean_token('###')` returned
`'#'`, `_clean_token('################')` returned a 14-character run, and
`_clean_token('@@')` returned the empty string. Those are Markdown headings,
comment banners and ordinary punctuation runs. Measured vocabulary entries
matching a marker pattern: 35 in `cl100k_base`, 24 in `o200k_base`, 1 in the
bundled `tokenizers/bpe.json`, none of which uses any of the three markers.

The marker set is now resolved per tokenizer, alongside the character decode
table and the special-token set, through one `_set_tokenizer_context` entry
point so a call site cannot update two of the three. Detection reads the
backend model's own `continuing_subword_prefix` and `end_of_word_suffix`
fields, then falls back to encoding a probe word certain to fragment and
inspecting the pieces. When neither channel finds a marker, nothing is
stripped: applying a WordPiece rule to a non-WordPiece vocabulary changes a
content token into a different one, while leaving a real marker in place shows
up as an unmappable span rather than as a wrong number.

Re-measured after the fix: 0 altered entries for all three vocabularies. A
WordPiece model still resolves to `{'##'}` and `##ing` still cleans to `ing`; a
BPE trained with `end_of_word_suffix='</w>'` resolves to `{'</w>'}`. On the
demo run, 23 of 2389 values in the results file moved: 16 in
`ast_boundary_alignment` and `identifier_fragmentation`, all for tokenizer
`bpe` on the R sample, whose comments start with `##`; 4 in `encoding_speed`
(wall-clock); 3 in `tokenizer_fairness_gini` at 1e-15 relative magnitude.

Found by the agent implementing the special-token accessor.

### X13. `id()`-keyed caches returned another tokenizer's data: **fixed**
`metrics/base.py`. `_special_token_cache` and the pre-existing
`_tokenizer_vocab_cache` keyed on `id(tokenizer)` without holding a reference,
and CPython recycles the id of a freed object. The module-level singletons in
`per_example.py` live for the whole process while caller-supplied tokenizers do
not, so this was reachable through the public API: measured 4 of 40 calls
getting the wrong special-token set, and 20 of 40 the wrong reverse vocabulary,
with only one vocab-cache entry ever created because every `Tokenizer.from_file`
landed on the same address. Metric consequence for one text: `n_digit_spans` 0
against 2, `mean_digit_f1` nan against 0.0. Both caches now store the tokenizer
alongside the value and check identity on read. Re-measured: 0 of 40 wrong.

### X14. An empty metadata channel was read as "declares no special tokens": **fixed**
`core/tokenizer_wrapper.py`. `hf_special_token_strings` set `answered = True` on
a readable-but-empty `added_tokens`, so the bundled `tokenizers/unigramlm.json`
resolved to an empty set even though `<unk>`, `<s>`, `</s>` and `<pad>` sit at
ids 0 to 3. That bypassed the generic fallback and let all four be treated as
content: 20479 of 37892 `_process_token` calls in one run used an empty set. The
model's own unknown token is now a third channel, and an empty result returns
None so the caller warns and uses `GENERIC_SPECIAL_TOKENS`.

### X12. `SentencePieceTokenizer.get_special_token_ids()` is not overridden: **fixed**
`core/tokenizer_wrapper.py`. See S18: both forms now come from one scan.
`sentencepiece` was also not installed in the working venv, so those paths were
covered only by stubs and the SP fixtures in `test_tokenizer_wrapper.py`
skipped. It is now declared as an optional extra and in the dev group, and the
SP tests run.

### X10. Reconstruction guessing, four further instances: **fixed**
A dedicated sweep found four more places inferring source positions from token
surfaces rather than offsets.

- `code_ast._map_from_greedy_decode`: a second greedy aligner with no re-sync,
  reached whenever a wrapper returns no offsets, which four wrappers never
  override. On a 45-character French snippet it agreed with offsets on 19
  characters and left 26 unmapped, silently. Deleted; missing offsets now raise.
- `math` operator isolation ran its regex over a reconstruction, so a
  Mistral-form `<s>` contributed a `<` and a `>` counted as operators. Apertus
  and Llama3 reported 8 operators at 0.750 isolation on a sentence with 6 at
  1.000. Now 6 at 1.000.
- Derived code and math corpora were built without offsets, so both domains were
  skipped entirely once isolation required them (112 and 262 documents dropped).
  They are now encoded with offsets.
- `sanity_check` C10 conservation compared byte-level surface lengths, so CJK's
  3x inflation absorbed real loss and produced a false PASS. Now measured with
  pretokenizer spans, via a new `pretokenize_with_spans` on the wrapper.

### X2. UTF-8 byte-level detection failed in the flattering direction: **fixed (twice)**
`metrics/utf8_integrity.py`. Detection counted single-character vocabulary
entries against 68 GPT-2 marker characters with a threshold of 50. A byte-level
tokenizer trained on a corpus that never exercises the control bytes carries
fewer than 50 and was read as not byte-level. Every token string is then
interpreted as literal text, which always encodes to valid UTF-8, so the metric
could only report 1.0. `gpt4o-english-bpe` has 37 markers and reported
completeness 1.0000, best of 37 tokenizers, against a true 0.6688, worst of 37.
Detection now reads the tokenizer's own ByteLevel pre-tokenizer or decoder and
falls back to the marker count only when the components cannot be introspected.
Any `utf8_token_integrity` or `utf8_char_split` number published for
`gpt4o-english-bpe` before this is invalid.

The first attempt did not work. It read `.decoder` off whatever object it was
handed, and the pipeline hands it a `TokenizerWrapper`, which exposes neither
component, so the check returned None for every tokenizer and the heuristic
decided anyway. It now unwraps through `get_underlying_tokenizer()` first.
Measured end to end after the second fix: `gpt4o-english-bpe` completeness
1.0000 to 0.6623, split rate 0.0000 to 0.2712.

### X3. `_crosses_character_boundary` counts continuation bytes as characters: **fixed**
`metrics/utf8_integrity.py:233-237`. The branch increments the character count
once per continuation byte, so a token holding only the tail of one character
satisfies "more than one character and incomplete" and is counted as crossing a
boundary. The README's own worked example gives 1/3; the code gives 2/3.
Measured inflation on FLORES with SuperBPE: Korean 632 crossings reported
against 584 true, Japanese 159 against 152.

### X4. `pattern_stability_rate` counts the first code token as indentation: **fixed (metric then removed)**
`metrics/code_ast.py:925-945`. Tokens are selected by overlap with the leading
whitespace range, and with ByteLevel offsets the first code token absorbs the
preceding space, so it is always included. Two lines at the same depth with
different code therefore get different patterns. A four-line snippet all
indented identically reports 0.25 where the correct value is 1.0.

### X5. `_SPECIAL_TOKEN` deletes ordinary bracket tokens: **fixed**
`metrics/base.py:33`. The pattern `^(<\||\[).*(\|>|\])$` matches `[]`, `[0]`,
`[i]` and `[...]`, not only `[CLS]`. Those tokens are dropped from the
reconstruction and from the UTF-8 content-token denominator. The marker
strippers also run for every tokenizer family, so a byte-level BPE token `##`
becomes empty and `a@@` becomes `a`. Both `##` and `[...]` are in the bundled
demo vocabulary.

### X6. `numeric_magnitude_consistency` treats the `10+` bucket as exactly 10 digits: **fixed**
`metrics/math.py`. The linear fit reconstructed a token count as
`mean(tokens/digits) * 10` for that bucket, so a 20-digit number costing 10
tokens was fitted at `(10, 5.0)`. Each number now carries its real digit count
through the accumulator, and each bucket enters the fit at its own mean digit
length against its own mean token count. Buckets `1` to `9` are unchanged, since
every number in them has the same length. On numbers lying exactly on
`tokens = 0.5 * digits + 1.0` with lengths 2, 4, 6, 8, 12 and 20, the fit
returned slope 0.4667, intercept 1.1333 and R-squared 0.9949 before, and returns
0.5, 1.0 and 1.0 after. `mean_digit_length` is published per bucket so the fit
is auditable from the results file.

`spearman_rho` is left as it was. It is a rank correlation over bucket order, so
the position pinned at 10 does not affect it. The fit still rests on at most 10
bucket points, and that is stated in the code.

### X7. `avg_tokens_per_line` mismatched numerator and denominator: **fixed**
`metrics/basic.py`. Blank lines were filtered from the denominator while the
numerator kept the tokens they produced. A 4-line text with 2 non-blank lines
and 8 tokens reported 4.0 rather than 2.0. Lines are now counted with
`str.splitlines()`, so blank lines count and a single trailing newline does not
add a phantom empty line. A blank line contributes its newline token to the
numerator and cannot be removed from it. No effect on line-per-item corpora
such as FLORES; it matters for document corpora. The unmeasured case reports
null rather than 0.0.

### X8. Digit metrics silently use the prose corpus: **fixed**
`metrics/math.py`. Without `--math-data` or `--use-builtin-math-data` the digit
metrics run on whatever corpus was loaded, and that branch had no log line. On
FLORES the observed digit lengths are 1 to 4 only, so the metric named for
three-digit grouping never exercises a single ideal boundary, and 74.2% of the
sample falls in the vacuous length-3-or-under case. `avg_recall` and
`avg_uniform_chunk` then equal the corpus short-number share, identically for
every non-splitting tokenizer. The branch now prints the same shape of warning
`_resolve_code_ast_config` prints for a missing `--code-ast-config`: it names
the three metrics affected, says the numbers are not comparable with a run that
supplied math data, and points at `--no-digit-boundary`. The substitution
itself is kept, because the prose corpus is a legitimate input for these
metrics and the alternative is to disable them by default.

### X9. Dead code

### C1. `metrics/base.py` used `scipy.stats` without importing it: **fixed**
Bare `import scipy` at line 9, `scipy.stats.sem(...)` at line 317. This works
only because `metrics/__init__.py` imports `.math`, which does `from
scipy.stats import spearmanr` at module scope, before anything can call
`compute_basic_stats`. scipy gained lazy submodule loading in 1.9.0; the
declared floor is 1.7.0. Explicit import added.

### C2. `core/validation.py` is entirely unreachable: **fixed (deleted)**
`ValidationResult`, `TokenizedDataValidator`, `InputProviderValidator`,
`InputSpecificationValidator`, `AnalysisValidator` and `validate_and_report`
are defined, exported from nowhere, and imported by nothing. The
`InputValidator` that `main.py:68-73` actually uses is a different class in
`core/input_utils.py`. 472 lines of code that never runs, including the only
logic that handles the single-language case explicitly.

### C3. `InputSpecification.get_vocab_size()` crashes on its own documented shape: **fixed**
`core/input_types.py`. Both branches were wrong, not just the one first found.
The pre-tokenized branch read `self.vocabulary.vocab_size`, which is None for
the `tokenizer + tokenized_data` shape that
`main.py:create_analyzer_from_tokenized_data` constructs, and the raw branch
read `tokenizer.vocab_size`, an attribute `TokenizerWrapper` does not have. So
the method raised `AttributeError` on every specification the package itself
builds. It now reads `tokenizer.get_vocab_size()` when a tokenizer is present
and falls back to the legacy `vocabulary` provider, whose protocol declares
`vocab_size` as a property. Measured on all three shapes: 128000, 128000, 4242.
Unreachable through the CLI, reachable by any API caller.

### C4. Dead branches: **partly fixed**
`_build_indentation_consistency_results` never writes `overall`, so the slim
branch testing for it is dead (open, see the schema section). The fertility
zero-sample guard at `basic.py:171` cannot fire, because `"你好".split()`
returns one element (open). The plural-key branches in
`loaders/multilingual_data.py:47-50` are vestigial (open, harmless).

---

## Output schema

Three decisions taken by the repo owner and implemented (see OUTPUT_CONTRACT.md
for the rest of this workstream):

- `indentation_consistency.global` is the micro-averaged correlation over pooled
  (depth, whitespace-token count) pairs, not the mean of the per-language
  correlations. Demo: pooled 0.7598 against a per-language mean of 0.8121 for
  `bpe`.
- `operator_isolation_rate` publishes its pooled `global` and its `by_domain`
  split in the slim results file. Both existed in the full results and were
  dropped from the slim one. Measured with `--use-sample-data`: pooled 0.7938
  over 3016 operator instances, code 0.6832 over 1932.
- A relative corpus path in a language config is anchored at the package root.
  See S13.

The rest of the section below is the state before that work.


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

Fixed in this pass: license metadata and `license-files` so the wheel carries
LICENSE and NOTICE; `py.typed`; tests excluded from the wheel; dependency
ceilings with the matplotlib and scipy floors raised; `fastparquet` dropped from
the `parquet` extra; the `Development Status` classifier; a `.github/workflows`
CI job covering Python 3.10 and 3.12, the quickstart, strict-JSON and provenance
assertions, three failure paths and the wheel contents; `CITATION.cff`,
`SECURITY.md` and `CODE_OF_CONDUCT.md`; and the gitignored fixture that made
`test_sanity_check.py` unpassable in a fresh clone.

Also fixed: several release commits used `git add -A` and swept 40 untracked
research-scaffolding files into the branch. They were untracked again with
`git rm --cached` and added to `.gitignore`.

Still open below.

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
