#!/usr/bin/env python
"""Standalone driver to recompute AST boundary alignment for a chosen subset of
tokenizers without re-running the full core analysis pipeline. Used to verify
the SentencePiece byte-fallback fix in `tokenizer_analysis/metrics/base.py`
without paying for the ~30+-minute full core re-run.

Loads tokenizer wrappers via the same factory the analysis pipeline uses,
constructs a minimal InputProvider that only implements the AST metric's
contract (`get_tokenizer_names`, `get_tokenizer`), and runs
`ASTBoundaryMetrics.compute()` against the existing StarCoder code config.

Prints per-tokenizer overall full-alignment-rate. Optionally merges the new
values into `results/report_core/analysis_results_full.json` so the report
build picks them up without re-running the entire core analysis.
"""
import argparse
import json
import sys
from typing import Dict, List

from tokenizer_analysis.core.input_types import InputProvider, TokenizedData
from tokenizer_analysis.core.tokenizer_wrapper import create_tokenizer_wrapper
from tokenizer_analysis.metrics.code_ast import ASTBoundaryMetrics


def load_config_from_file(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


class _AstOnlyProvider(InputProvider):
    """Minimal InputProvider exposing only what ASTBoundaryMetrics needs.

    AST alignment loads its own code corpus (via `CodeDataLoader`) and does
    not consume `get_tokenized_data`, so the other abstract methods can be
    stubs.
    """

    def __init__(self, tokenizer_configs: Dict[str, Dict]):
        self._wrappers = {
            name: create_tokenizer_wrapper(name, cfg)
            for name, cfg in tokenizer_configs.items()
        }

    def get_tokenizer_names(self) -> List[str]:
        return list(self._wrappers)

    def get_tokenizer(self, tokenizer_name: str):
        return self._wrappers[tokenizer_name]

    def get_tokenized_data(self) -> Dict[str, List[TokenizedData]]:
        return {n: [] for n in self._wrappers}

    def get_vocab_size(self, tokenizer_name: str) -> int:
        return self._wrappers[tokenizer_name].get_vocab_size()

    def get_languages(self, tokenizer_name=None) -> List[str]:
        return []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-config", required=True,
                    help="Tokenizer config JSON ({name: {class, path}}).")
    ap.add_argument("--code-ast-config",
                    default="starcoder_ast_config.json",
                    help="StarCoder per-language parquet config. Lives at the repo "
                         "root, not under configs/; the wrong path silently substitutes "
                         "synthetic code.")
    ap.add_argument("--tokenizers", nargs="+",
                    help="Subset of tokenizer names to recompute (default: all in config).")
    ap.add_argument("--merge-into",
                    help="If given, merge the new AST results into this analysis JSON "
                         "(typically results/report_core/analysis_results_full.json) "
                         "and write it back in-place. Only the listed tokenizers are touched.")
    ap.add_argument("--max-snippets-per-lang", type=int, default=None,
                    help="Override the default snippet cap (smaller = faster verify).")
    args = ap.parse_args(argv)

    all_cfg = load_config_from_file(args.tokenizer_config)
    code_cfg = load_config_from_file(args.code_ast_config)
    selected = args.tokenizers or list(all_cfg)
    missing = [n for n in selected if n not in all_cfg]
    if missing:
        print(f"Unknown tokenizer name(s) in --tokenizers: {missing}", file=sys.stderr)
        return 2
    subset_cfg = {n: all_cfg[n] for n in selected}

    print(f"Loading {len(subset_cfg)} tokenizer(s): {list(subset_cfg)}")
    provider = _AstOnlyProvider(subset_cfg)
    metric = ASTBoundaryMetrics(
        provider,
        code_config=code_cfg,
        max_snippets_per_lang=args.max_snippets_per_lang,
    )
    print(f"Running AST boundary alignment "
          f"(max_snippets_per_lang={args.max_snippets_per_lang})...")
    out = metric.compute()
    ast = out.get("ast_boundary_alignment", {})
    if "error" in ast:
        print(f"ERROR: {ast['error']}", file=sys.stderr)
        return 1

    per_tok = ast.get("per_tokenizer", {})
    print()
    print(f"{'tokenizer':<40} {'full_align':<12} {'start':<12} {'end':<12} {'count':<10}")
    print("-" * 86)
    for t in selected:
        ov = (per_tok.get(t) or {}).get("overall", {})
        print(f"{t:<40} {ov.get('full_alignment_rate', 0):<12.6f} "
              f"{ov.get('start_alignment_rate', 0):<12.6f} "
              f"{ov.get('end_alignment_rate', 0):<12.6f} "
              f"{str(ov.get('count', 0)):<10}")

    if args.merge_into:
        with open(args.merge_into) as f:
            R = json.load(f)
        existing = R.setdefault("ast_boundary_alignment", {"per_tokenizer": {},
                                                            "summary": {}})
        existing_per = existing.setdefault("per_tokenizer", {})
        existing_sum = existing.setdefault("summary", {})
        for t in selected:
            if t in per_tok:
                existing_per[t] = per_tok[t]
            if t in ast.get("summary", {}):
                existing_sum[t] = ast["summary"][t]
        with open(args.merge_into, "w") as f:
            json.dump(R, f)
        print()
        print(f"Merged AST results for {len(selected)} tokenizer(s) into "
              f"{args.merge_into}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
