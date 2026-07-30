# Session Status

## Ongoing experiments
- 1.0 release vetting on `release/1.0-vetting`: 35 of 37 audit findings fixed and verified by execution, all committed. `RELEASE_AUDIT.md` is the record. Suite 625 passed, 53 skipped; the demo, `--input`, multilingual with grouped analysis and LaTeX, the visualizer and the sanity check all exit 0.
- Workstream 5 (results-file schema) is the one workstream not finished. `OUTPUT_CONTRACT.md` holds the proposed contract table for review; nothing is implemented from it yet. Missing today: a `global` key on 7 of 18 metrics, an `aggregation` label on 13, and `count` on most `per_language` entries. `run_metadata` is done.

## Open decisions
- `OUTPUT_CONTRACT.md`: the whole table, and in particular the two rows flagged there. `indentation_consistency`, macro mean of per-language correlations (my recommendation) against one pooled correlation. `operator_isolation_rate`, whether to publish a global that pools prose, code and math, or to publish only the `by_domain` split.
- S13, config paths resolving against the process CWD rather than the config file's directory: changing it alters behaviour for existing configs.
- X6, `numeric_magnitude_consistency` fits the open `10+` bucket as exactly 10 digits (slope 0.607, R-squared 0.794 against a true 0.587 and 0.980). Use the bucket's own mean digit length, or drop the open bucket from the fit. Both change published values.
- `sentencepiece` was absent from `.venv` and is now declared as an optional extra plus a dev dependency. Confirm you want an extra rather than a core dependency.
