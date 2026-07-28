#!/usr/bin/env python3
"""Copy `operator_isolation_rate.by_domain` from analysis_results_full.json into the slim
analysis_results.json of the same run.

Why this exists. The slimmer (`cli/run_analysis.py::_slim_results`) used to drop `by_domain`, so a
run's summary file carried only the POOLED operator-isolation number (prose+code+math merged), which
is not any of the three. Consumers that read the summary file therefore could not get the domain they
asked for. The slimmer is fixed, but runs written before the fix (and any run whose process had
already imported the old module) still have summaries without `by_domain`.

This does NOT recompute anything. It copies a value the run already produced into the file that lost
it, and it refuses to touch a run whose full results lack the split or whose code domain was computed
on synthetic samples.

    python3 scripts/backfill_operator_by_domain.py results/report_mc_full [results/... ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KEY = "operator_isolation_rate"


def backfill(run_dir: Path) -> str:
    full_p = run_dir / "analysis_results_full.json"
    slim_p = run_dir / "analysis_results.json"
    if not full_p.exists():
        return f"SKIP  {run_dir}: no analysis_results_full.json"
    if not slim_p.exists():
        return f"SKIP  {run_dir}: no analysis_results.json"

    full = json.load(open(full_p))
    bd = (full.get(KEY) or {}).get("by_domain")
    if bd is None:
        return (f"ABORT {run_dir}: full results have no {KEY}['by_domain']. This run predates the "
                f"per-domain split; re-run it, do not backfill.")

    code_src = str((bd.get("code") or {}).get("source", ""))
    if "code-ast dataset" not in code_src:
        return (f"ABORT {run_dir}: the CODE domain was computed on '{code_src}', not the code-AST "
                f"dataset. Re-run with --code-ast-config starcoder_ast_config.json.")

    slim = json.load(open(slim_p))
    if KEY not in slim:
        return f"ABORT {run_dir}: slim results have no {KEY}"

    slim[KEY]["by_domain"] = {
        domain: {"summary": d.get("summary", {}), "source": d.get("source")}
        for domain, d in bd.items()
    }
    json.dump(slim, open(slim_p, "w"), indent=2)
    doms = ", ".join(sorted(bd))
    n = len((bd.get("prose") or {}).get("summary", {}))
    return f"OK    {run_dir}: wrote by_domain [{doms}] for {n} tokenizers"


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]]
    if not dirs:
        sys.exit(__doc__)
    bad = False
    for d in dirs:
        msg = backfill(d)
        print(msg)
        bad |= msg.startswith("ABORT")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
