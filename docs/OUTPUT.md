# Output

This page describes the files `tokenizer-analysis` writes to `--output-dir`:
the directory layout, the JSON results schema, what a `null` value means, and
the run-metadata block in each results file. See [../README.md](../README.md)
for how to run an analysis, and [METRICS.md](METRICS.md) for what each metric
measures.

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

### JSON Results Schema

`analysis_results.json` is always written with a slimmed schema. Pass
`--save-full-results` to also write `analysis_results_full.json` with all
computed data.

Three things hold for every metric, and the test suite asserts each of them
over the whole file rather than a sample:

- `per_tokenizer.<tok>.global` is the headline block. There are no exemptions.
  `token_length` and `encoding_speed` have one that duplicates a block they
  already publish, because an exception in the schema costs a reader more than a
  duplicated number does.
- `metadata.aggregation` names which average `global` reports, one of four
  labels. See [Aggregation labels](#aggregation-labels) below.
- Every `per_language` entry includes a `count`, in the unit named by
  `metadata.count_unit`, so a consumer can re-derive the other weighting.
  `tokenizer_fairness_gini` and `morphscore` are the exception: their unit is
  languages, so the count would be 1 for every entry, and `metadata` records this.

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
| `token_length` | `global` holds `count`, `mean`, `median`, `std`. Three sibling blocks hold the same four stats separately: `character_length`, `byte_length`, `primary_length`. No `per_language` |
| `encoding_speed` | `global` holds `mean_ms`, `total_s`, `num_samples`, duplicating the same three fields already published directly under `per_tokenizer.<tok>`. No `per_language` |
| `reconstruction_fidelity` | `per_domain` in place of `per_language`, because it also runs on code and math |

`tokenizer_fairness_gini` holds a `per_line_normalization` block beside
`global`, with the same coefficient computed on tokens per line rather than per
the configured unit. It is `null` unless every language has the same line count.
On a parallel corpus it is the one to read.

`operator_isolation_rate` holds `global`, `per_language` **and** `by_domain`,
the last splitting the pooled global by corpus. `by_domain` has `code` and
`math`, plus `prose` when `--operator-prose-domain` was passed, and `global`
pools whichever ran. Its `per_language` keys natural languages (`arb_Arab`) and
programming languages (`code:bash`) in one dict.

The slimmed file omits `pairwise_comparisons`, `summary`, `per_category`
breakdowns, and the derivable stat fields `sum`, `std_err`, `min` and `max`. The
full results file adds `per_category` for the metrics that have category
breakdowns (AST node types, operator types). `bigram_entropy` and `trigram_entropy` also hold a
top-level `reference_definition` block with the same measurement under the
reference normalizer and aggregation, per tokenizer and, inside that, per
language. A per-language entry normalizes by that language's own accessor
domain, published beside it as `accessor_domain_size`, so two languages there
are not on a common scale.

Since 1.0.2 the slim file is a strict projection of the full one: every leaf
is at the same key path with the same value in both, and slimming only
deletes keys. Renaming happens before either file is written, so a path read
off one holds in the other. Earlier versions renamed while selecting, which
left the two files impossible to cross-reference.

#### Aggregation labels

`metadata.aggregation` names which average `global` reports: `micro_pooled`,
`macro_languages`, `ratio_of_sums` or `set_union`. This matters because
`global` means different things in different metrics, and on a parallel corpus
where every language holds the same number of lines micro and macro agree, so
the difference only shows on an unequal corpus.

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
in the file with no record of it. All nine tokenizers in
`benchmarks/open_source` are Hub-loaded and have a `hub_revision`.

`corpus.digest` gives each language's path, byte count and a short hash, plus
whatever the fetch script recorded about the dataset revision it read. The
config hashes name which corpus was requested; the digest records what the files
held, so two runs over different snapshots of one corpus under one config are
distinguishable after the fact. Paths under the working directory are recorded
relative to it, so a committed results file does not name whoever produced it. Arguments that only set where
output goes are left out, so two runs writing to different directories do not
look like different measurements. When two results files disagree,
`run_metadata` shows whether the tokenizer, the corpus, the settings or the
toolkit version changed.

`git_tree_modified` is written next to `git_commit`. It is `true` when the working tree
had uncommitted changes at run time, in which case the commit hash does not
describe the code that ran.

On the `--tokenized-data-file` path, `run_metadata.tokenizers` records the
vocabulary dumps named by the cache, each tagged
`"source": "vocabulary file from --tokenized-data-config"`, with no
`sha256_16`, because no tokenizer file is read on that path.
