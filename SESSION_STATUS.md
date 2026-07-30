# Session Status

Pre-open-source vetting of the 1.0 release, on branch `release/1.0-vetting`.
Findings and their status are in `RELEASE_AUDIT.md`; output-format changes are in
`CHANGELOG.md` and `MIGRATION.md`. Suite at last run: 579 passed, 65 skipped.

## Ongoing experiments
- Independent audit of the last two change sets (special-token accessor, visualizer fixes and tests) was still running when the session ended; its findings were never received. Re-run it before trusting those two commits. Its scratch work is under `scratchpad/audit2/` in the session scratch directory, which may have been purged.
- Both change sets are committed and the suite is green, but they have had only my own spot checks, not the adversarial pass. My spot checks: the special-token inversion is correct (`[]`, `[0]`, `[i]`, `[...]` no longer deleted; `<s>`, `</s>` now recognized for bpe.json and apertus), no em-dashes were added, and the `ast_boundary_alignment` delimiter category moves 0.000 to 0.500 on `y = a[...]` as predicted.

## Open decisions
- Git history: settled as publish-as-is, so `legacy-suite` / `legacy-suite-final` still carry `PA_BPE_tokenizers/` and `MIGRATION.md:67-69` points readers at them. Deleting the remote `results` and `claude/complete-todo-item-...` branches and picking between `main` and `master` are yours to run; I have not touched remote refs.
- Whether to rename `depth_proportionality_correlation`: Spearman measures monotonicity, which is the intended semantics, but the name says proportionality. Documented rather than renamed, since renaming is another breaking output key.
- Whether the successor-entropy `reference_definition` blocks should become primary after you compare them. Currently the library's own definition stays primary and both are published, likewise `renyi_efficiency.observed_normalization`.

## Known open defects
Full list with reproducers in `RELEASE_AUDIT.md`. The ones I would fix next, in order:
- `_process_token` strips WordPiece markers for every tokenizer family, so a byte-level BPE gets `###` turned into `#` and `##` into empty. Found by the special-token agent, not fixed. Same shape as the special-token fix: it needs a per-tokenizer decision about whether the marker convention applies. Affects reconstructions and the UTF-8 denominator.
- Three silent fallbacks still open: missing MorphScore data reports `avg_micro_f1: 0.0`; a malformed `--custom-latex-config` logs an error then prints success and exits 0; a missing parquet engine yields an empty corpus.
- `_SPECIAL_TOKEN`-adjacent: `SentencePieceTokenizer` does not override `get_special_token_ids()`, so it returns the base empty set.
