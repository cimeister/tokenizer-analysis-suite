#!/usr/bin/env python
"""Standalone FineWeb-Edu English compression (bytes/token) per tokenizer.

Independent of the analysis pipeline. Loads a deterministic English snippet from
a local FineWeb-Edu shard and, for each tokenizer in a tokenizer-config JSON,
computes compression = total UTF-8 bytes / total tokens (the same bytes/token
definition the library uses in information_theoretic.py). Records full provenance.

Usage:
    python scripts/fineweb_edu_compression.py \
        --tokenizer-config configs/sample_tokenizers.json \
        --output results/report_finewebedu_compression.json \
        [--only NAME ...] [--max-rows 1000]
"""
import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from tokenizer_analysis.core.tokenizer_wrapper import create_tokenizer_wrapper

# Canonical FineWeb-Edu mirror (HuggingFaceFW/fineweb-edu). NOTE: the
# weborganizer-annotated/fineweb-edu-full copy is a single truncated shard
# (PAR1 header, no footer) and is unusable — do not use it.
DEFAULT_SHARD = ("/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/"
                 "fineweb-edu/data/CC-MAIN-2013-20/train-00000-of-00014.parquet")
DATASET_NAME = "FineWeb-Edu (HuggingFaceFW/fineweb-edu local mirror)"


def load_snippet(shard: str, max_rows: int):
    # Read only the first row group (shards are large), then take first N rows
    # — deterministic snippet.
    pf = pq.ParquetFile(shard)
    t = pf.read_row_group(0)
    col = "text" if "text" in t.column_names else next(
        (c for c in t.column_names if t.schema.field(c).type == "string"), None)
    if col is None:
        raise ValueError(f"no string/text column in {shard}; columns={t.column_names}")
    docs = [d for d in t.column(col).to_pylist()[:max_rows] if d]
    return docs, col


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", default=DEFAULT_SHARD)
    ap.add_argument("--max-rows", type=int, default=1000)
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these tokenizer names")
    args = ap.parse_args(argv)

    cfg = json.load(open(args.tokenizer_config))
    if args.only:
        cfg = {k: v for k, v in cfg.items() if k in args.only}

    docs, text_col = load_snippet(args.shard, args.max_rows)
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    provenance = {
        "dataset": DATASET_NAME,
        "shard": args.shard,
        "text_column": text_col,
        "n_docs": len(docs),
        "total_bytes": total_bytes,
        "compression_definition": "total_utf8_bytes / total_tokens (bytes per token; higher = better)",
        "deterministic": "first N rows of the shard",
    }

    per_tok = {}
    for name, tcfg in cfg.items():
        try:
            w = create_tokenizer_wrapper(name, tcfg)
            n_tokens = sum(len(w.encode(d)) for d in docs)
            comp = total_bytes / n_tokens if n_tokens else None
            per_tok[name] = {"compression_bytes_per_token": comp,
                             "n_tokens": n_tokens, "n_bytes": total_bytes}
            print(f"  {name:24s} comp={comp:.4f} bytes/tok  ({n_tokens} tokens)")
        except Exception as e:  # fail loudly per tokenizer, don't abort the batch
            per_tok[name] = {"error": str(e)}
            print(f"  {name:24s} ERROR: {e}", file=sys.stderr)

    out = {"finewebedu_eng_compression": {"provenance": provenance,
                                          "per_tokenizer": per_tok}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
