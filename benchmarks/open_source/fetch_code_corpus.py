#!/usr/bin/env python
"""Fetch the source-code corpus the open-source benchmark measures.

The code metrics need real source code. Without `--code-ast-config` they run on
the bundled synthetic samples, whose numbers are not comparable with a real run
(measured 0.562 full AST alignment on synthetic against 0.493 on StarCoder for
the same tokenizer), so the benchmark supplies a real corpus.

Source: `bigcode/the-stack-smol-xs`, which is ungated, so anyone can reproduce
the benchmark without an access request. It holds 100 files per language, drawn
from The Stack, whose files carry their original repository licenses.

    uv run python benchmarks/open_source/fetch_code_corpus.py

Writes `benchmarks/open_source/code_data/<language>/<n>.<ext>`, which is the
directory-per-language layout `code_ast_config.json` names and
`CodeDataLoader` reads by file extension.
"""

import argparse
import json
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BENCHMARK_DIR / "code_data"
DATASET = "bigcode/the-stack-smol-xs"

# The Stack's directory name for each language, against the name TokEval uses.
# Only languages present in both are listed; the-stack-smol-xs spells C# as
# "c-sharp" and shell as "shell", TokEval as "csharp" and "bash".
LANGUAGES = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "java": "java",
    "c": "c",
    "c-sharp": "csharp",
    "go": "go",
    "rust": "rust",
    "php": "php",
    "ruby": "ruby",
    "scala": "scala",
    "lua": "lua",
    "r": "r",
    "haskell": "haskell",
    "shell": "bash",
}

# File extension written per TokEval language, matching what
# CodeDataLoader._LANG_EXTENSIONS accepts.
EXTENSIONS = {
    "python": ".py", "javascript": ".js", "typescript": ".ts", "java": ".java",
    "c": ".c", "csharp": ".cs", "go": ".go", "rust": ".rs", "php": ".php",
    "ruby": ".rb", "scala": ".scala", "lua": ".lua", "r": ".r",
    "haskell": ".hs", "bash": ".sh",
}

DEFAULT_FILES_PER_LANGUAGE = 100


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--files-per-language", type=int,
                        default=DEFAULT_FILES_PER_LANGUAGE)
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required. It ships with transformers, so "
              "`uv sync` should already have it.", file=sys.stderr)
        return 1

    written = {}
    for stack_name, tokeval_name in LANGUAGES.items():
        try:
            path = hf_hub_download(
                DATASET, f"data/{stack_name}/data.json", repo_type="dataset",
            )
        except Exception as e:
            print(f"  {tokeval_name:12s} could not be fetched: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            return 1

        # The files are named .json but hold JSON Lines.
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        out_dir = args.output_dir / tokeval_name
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = EXTENSIONS[tokeval_name]
        count = 0
        for i, row in enumerate(rows[: args.files_per_language]):
            content = row.get("content")
            if not content:
                continue
            (out_dir / f"{i:04d}{ext}").write_text(content, encoding="utf-8")
            count += 1
        written[tokeval_name] = count
        print(f"  {tokeval_name:12s} {count:4d} files -> "
              f"{out_dir.relative_to(BENCHMARK_DIR.parent.parent)}")

    total = sum(written.values())
    print(f"\nWrote {total} files across {len(written)} languages to "
          f"{args.output_dir}.")
    print(f"Source: https://huggingface.co/datasets/{DATASET}. Each file keeps "
          "the license of the repository it came from; see that dataset card.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
