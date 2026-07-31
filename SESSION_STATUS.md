# Session Status

## Ongoing experiments
- 1.0 release vetting on `release/1.0-vetting`: `RELEASE_AUDIT.md` has no open findings. Suite 628 passed, 53 skipped on an unloaded machine.
- README reorganized and fact-checked against real runs; an Opus agent is reviewing it for correctness, organization and house style. Its findings still to be applied.
- Workstream 5 (results-file schema) partly done: `indentation_consistency` and `operator_isolation_rate` now publish a `global`, per your decisions. `OUTPUT_CONTRACT.md` lists what remains: the `aggregation` label on every metric, `count` on `per_language` entries, and a `global` on five metrics.

## Open decisions
- `OUTPUT_CONTRACT.md`, the remaining rows: whether the proposed `aggregation` label and `count` unit per metric are right before they are implemented.
