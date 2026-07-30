# Session Status

## Ongoing experiments
- 1.0 release vetting on `release/1.0-vetting`: 34 of 37 audit findings fixed and verified by execution. `RELEASE_AUDIT.md` is the live record.
- Em-dash and double-hyphen cleanup across the six metric and test files touched this session: agent running, suite baseline 625 passed / 53 skipped.
- Not yet committed: the subword-marker gating, the `avg_tokens_per_line` line-counting fix, `InputSpecification.get_vocab_size`, the grouped-analysis fix and the CLI error-path work.

## Open decisions
- S13, config paths resolving against the process CWD rather than the config file's directory: changing it alters behaviour for existing configs, so it awaits your call.
- X6, `numeric_magnitude_consistency` fits the open `10+` bucket as exactly 10 digits (slope 0.607, R-squared 0.794 against a true 0.587 and 0.980). Two candidate fixes: use the bucket's own mean digit length, or drop the open bucket from the fit. Both change published values.
- X8, the digit metrics run on the prose corpus when neither `--math-data` nor `--use-builtin-math-data` is given. I plan the same loud warning `--code-ast-config` already prints; say if you want it to abort instead.
- `sentencepiece` was absent from `.venv` and is now declared as an optional extra plus a dev dependency. Confirm you want an extra rather than a core dependency.
