# Session Status

## Ongoing experiments
- 1.0 release vetting on `release/1.0-vetting`: audit findings being fixed in severity order; RELEASE_AUDIT.md is the live record of fixed vs open.
- Subword-marker gating (`##`, `</w>`, `@@` stripped for every tokenizer family): agent working across metrics/base.py, code_ast.py, math.py, diagnostics/sanity_check.py. Measured 35 cl100k_base and 24 o200k_base vocabulary entries truncated before the fix.

## Open decisions
- S13, config paths resolving against the process CWD rather than the config file's directory: changing it alters documented behaviour for existing configs, so it is left open pending your call.
- `sentencepiece` was installed into `.venv` during this session (it was absent) and is now declared as an optional extra; confirm you want the extra rather than a core dependency.
