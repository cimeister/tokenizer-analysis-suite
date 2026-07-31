# Session Status

## Ongoing experiments
- 1.0 release vetting on `release/1.0-vetting`: `RELEASE_AUDIT.md` has no open findings. Suite 629 passed, 53 skipped. All six documented CLI routes exit 0.
- README rewritten after an independent Opus review that ran every documented command: 26 defects found, all applied. Metric reference split into `METRICS.md`.
- Workstream 5 (results-file schema) partly done. `OUTPUT_CONTRACT.md` lists what remains: the `aggregation` label on every metric, `count` on `per_language` entries, and a `global` on five metrics.

## Open decisions
- Whether to publish to PyPI before release. `pip install tokenizer-intrinsic-evals` returns 404 today, so the README and MIGRATION.md document the git checkout as the install.
- `RELEASE_AUDIT.md`, `SESSION_STATUS.md` and `OUTPUT_CONTRACT.md` are tracked at the repo root and would ship publicly. `OUTPUT_CONTRACT.md` is worth keeping; the other two are working documents.
- `OUTPUT_CONTRACT.md`, the remaining rows: whether the proposed `aggregation` label and `count` unit per metric are right before they are implemented.
