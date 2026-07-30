# Session Status

Pre-open-source vetting of the 1.0 release, on branch `release/1.0-vetting`.
Findings and their status are in `RELEASE_AUDIT.md`; output-format changes are in
`CHANGELOG.md` and `MIGRATION.md`.

## Ongoing experiments
- Special-token accessor: agent implementing `get_special_token_strings()` on the wrapper interface, replacing the `_SPECIAL_TOKEN` surface regex that deletes ordinary tokens like `[...]`, `[0]`, `[i]`; awaiting result, then audit.
- Visualizer bugs and tests: agent fixing exit-0-when-nothing-loads, byte-continuation spans counted as special tokens, unstable colours across the two UTF-8 panels; adding up to 10 tests to `test_plots.py` and a new `test_visualize_tokenization.py`; awaiting result, then audit.

## Open decisions
- Git history: settled as publish-as-is, so `legacy-suite` / `legacy-suite-final` still carry `PA_BPE_tokenizers/` and `MIGRATION.md:67-69` points readers at them. Remote also holds a `results` branch and a `claude/complete-todo-item-...` branch, and both `main` and `master` exist. Deleting those and picking a default branch are yours to run; I have not touched remote refs.
- Whether to rename `depth_proportionality_correlation`: Spearman measures monotonicity, which is the intended semantics, but the name says proportionality. Documented rather than renamed, since renaming is another breaking output key.
- Whether the successor-entropy `reference_definition` blocks should become the primary definition after you compare them; currently the library's own definition stays primary and both are published.
