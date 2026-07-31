# Results-file contract

One row per metric in `analysis_results.json`. This is the review artifact for
the remaining schema work (workstream 5). The two rows that needed a decision
are settled and implemented; the `aggregation` label and the `count` unit are
still not implemented for any row, and five metrics still have no `global`.

Measured on the bundled demo (`--use-sample-data`, 5 FLORES+ languages, 2
tokenizers). Every metric already publishes `per_tokenizer.<tok>`; the columns
below describe what sits under it.

## Aggregation labels

Four values, applied to whatever `global` reports:

- `micro_pooled`: one ratio computed from summed counts over every item in
  every language. A language with more items counts for more.
- `macro_languages`: the unweighted mean of the per-language values. Every
  language counts the same regardless of size.
- `ratio_of_sums`: total units divided by total tokens. Distinct from
  `micro_pooled` in that the numerator and denominator are different units, not
  a count of the same thing.
- `set_union`: not an average. The value is a property of the union of the
  per-language sets.

On the bundled parallel corpus all languages hold 997 lines, so micro and macro
agree there and the difference appears only on unequal corpora. That is why the
label has to be published rather than inferred.

## Current state

| Metric | `global` today | `aggregation` today |
|---|---|---|
| `encoding_speed` | no | no |
| `fertility` | yes | no |
| `token_length` | no | no |
| `vocabulary_utilization` | yes | no |
| `reconstruction_fidelity` | yes | no |
| `compression_rate` | yes | no |
| `renyi_efficiency` | yes | yes |
| `bigram_entropy` | yes | yes |
| `trigram_entropy` | flat `global_*` siblings | yes |
| `tokenizer_fairness_gini` | yes | yes |
| `three_digit_boundary_alignment` | no | no |
| `numeric_magnitude_consistency` | no | no |
| `operator_isolation_rate` | yes | no |
| `ast_boundary_alignment` | yes | no |
| `identifier_fragmentation` | yes | no |
| `indentation_consistency` | yes | no |
| `utf8_token_integrity` | yes | no |
| `morphscore` | `summary` | no |

## Proposed contract

| Metric | `global` fields | Formula | `aggregation` | `count` unit |
|---|---|---|---|---|
| `encoding_speed` | exempt, keeps `mean_ms`, `total_s`, `num_samples` | wall-clock, not a corpus statistic | `micro_pooled` | samples |
| `fertility` | `mean`, `median`, `std`, `count` | mean of per-document tokens/unit | `micro_pooled` | documents |
| `token_length` | exempt, keeps `character_length`, `byte_length`, `primary_length` | mean over every token emitted | `micro_pooled` | tokens |
| `vocabulary_utilization` | `utilization`, `used_tokens`, `vocab_size` | size of the union of used ids over vocab size | `set_union` | vocabulary entries |
| `reconstruction_fidelity` | `exact_match_rate`, `mean_cer`, `unk_token_rate`, `whitespace_fidelity`, `count`, `total_tokens` | pooled over documents | `micro_pooled` | documents |
| `compression_rate` | `compression_rate`, `total_units`, `total_tokens` | total units over total tokens | `ratio_of_sums` | measurement units |
| `renyi_efficiency` | `renyi_<alpha>` per alpha | H_alpha over log2(vocab size), pooled unigram distribution | `micro_pooled` | tokens |
| `bigram_entropy` | `bigram_entropy`, `total_bigrams`, `types_evaluated`, `types_excluded` | frequency-weighted mean over types | `micro_pooled` | bigrams |
| `trigram_entropy` | same four fields, nested under `global` instead of flat `global_*` siblings | as bigram | `micro_pooled` | trigrams |
| `tokenizer_fairness_gini` | `gini_coefficient`, `mean_cost`, `std_cost`, `cost_ratio`, `num_languages` | Gini over the per-language cost vector | `macro_languages` | languages |
| `three_digit_boundary_alignment` | `mean_f1`, `mean_precision`, `mean_recall`, `count` | pooled over every digit span in every language | `micro_pooled` | digit spans |
| `numeric_magnitude_consistency` | `mean_fertility`, `count` | pooled over every number in every language | `micro_pooled` | numbers |
| `operator_isolation_rate` | `isolation_rate`, `compound_preservation_rate`, `total`, `isolated`, `compound_total` | pooled over prose, code and math, with the split kept in `by_domain` | `micro_pooled` | operator occurrences |
| `ast_boundary_alignment` | `full_alignment_rate`, `start_alignment_rate`, `end_alignment_rate`, `cross_boundary_rate`, `count` | pooled over AST nodes in every programming language | `micro_pooled` | AST nodes |
| `identifier_fragmentation` | `fragmentation_rate`, `avg_tokens_per_identifier`, `count`, `unmappable` | pooled over identifier occurrences | `micro_pooled` | identifier occurrences |
| `indentation_consistency` | `depth_proportionality_correlation`, `num_depth_levels`, `total_indented_lines`, pooled over languages | one Spearman correlation over the pooled pairs | `micro_pooled` | indented lines |
| `utf8_token_integrity` | `completeness_rate`, plus the six count fields it already carries | pooled over content tokens | `micro_pooled` | content tokens |
| `morphscore` | keep `summary` as the global block, add `aggregation` | unweighted mean over the languages with data | `macro_languages` | languages |

## Rows decided and implemented

**`indentation_consistency`: micro-averaged.** `global` is one Spearman
correlation over the pooled `(depth, whitespace-token count)` pairs of every
programming language, with the pooled `num_depth_levels` and
`total_indented_lines` beside it. Not the mean of the per-language correlations.
Measured on the demo: pooled 0.7598 against a per-language mean of 0.8121 for
`bpe`, and pooled -0.2427 against -0.2793 for `unigramlm`. Indent conventions
differ by language (Python 4 spaces, Go tabs, Haskell alignment), so the pooled
value depends on the language mix of the code corpus. The per-language block is
where each language is read separately, and the README says so.

**`operator_isolation_rate`: pooled global published with `by_domain`.** Both
existed in the full results and were dropped from the slim file, which carried
only `per_language`. The global is weighted by operator instances, so with a
code corpus it sits close to the code rate: 0.7285 pooled against 0.6832 for
code, which supplies 1932 of 2258 instances on the demo. `by_domain` travels
with it, because that is what makes the pooled number readable.

## Also in this workstream, not yet done

- `count` on every `per_language` entry, in the unit named above, so a consumer
  can re-derive the other weighting. Present today on `fertility`,
  `three_digit_boundary_alignment` and `numeric_magnitude_consistency`; absent
  elsewhere.
- `operator_isolation_rate.per_language` keys natural languages (`arb_Arab`)
  and programming languages (`code:bash`) in one dict. The `code:` prefix
  namespaces them and `by_domain` now carries the split, so they stay where they
  are.
- `analysis_results.json` should be a strict projection of
  `analysis_results_full.json`. Not verified.

`run_metadata` is done: the results file records the package version, the git
commit, the config paths and their hashes, the tokenizer hashes,
`--samples-per-lang` and a timestamp.
