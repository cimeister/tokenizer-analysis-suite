#!/usr/bin/env python
"""Fetch the FLORES+ evaluation corpus into parallel/.

FLORES+ is CC-BY-SA 4.0. This repository does not redistribute it, so the
corpus that `--use-sample-data` and the configs under `configs/` name is
fetched here instead of being committed.

    uv run python scripts/fetch_flores.py                 # the 13 core languages
    uv run python scripts/fetch_flores.py --config configs/flores60_lang_config.json
    uv run python scripts/fetch_flores.py --all           # every language in the dataset

The Hugging Face dataset is gated (approval is automatic, a token is not).
Run `hf auth login` first, or set HF_TOKEN.

One file per language, `parallel/<iso639-3>_<Script>.txt`, one sentence per
line, from the `dev` split. That is the layout every config in `configs/`
expects.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "core_lang_config.json"
OUTPUT_DIR = REPO_ROOT / "parallel"
DATASET = "openlanguagedata/flores_plus"
SPLIT = "dev"


def languages_from_config(config_path: Path) -> list:
    """Language codes a TokEval language config names, in file order."""
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    languages = config.get("languages")
    if not languages:
        raise ValueError(
            f"{config_path} has no 'languages' block, so there is nothing to fetch."
        )
    return list(languages)


def _resolved_revision(dataset, requested):
    """The commit sha the loaded dataset came from, when it can be read.

    `datasets` records it on the dataset's info for Hub loads. Returns the
    requested value when it cannot be read, so the manifest never claims a
    revision that was not observed.
    """
    info = getattr(dataset, "info", None)
    if info is not None:
        for attr in ("dataset_revision", "version"):
            value = getattr(info, attr, None)
            if value:
                return str(value)
    return requested


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Language config naming the languages to fetch (default: {DEFAULT_CONFIG.name})",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Fetch every language in the dataset, ignoring --config",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Where the .txt files are written (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--split", default=SPLIT,
        help=f"FLORES+ split to write (default: {SPLIT}, 997 sentences)",
    )
    parser.add_argument(
        "--revision", default=None,
        help="Dataset revision to fetch (branch, tag or commit sha). Defaults "
             "to the Hub's current default branch. Whichever is used is "
             "recorded in the manifest this script writes.",
    )
    args = parser.parse_args(argv)

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "The `datasets` package is required to fetch FLORES+. Install it with "
            "`uv pip install datasets`, then rerun this script.",
            file=sys.stderr,
        )
        return 1

    if args.all:
        wanted = None
    else:
        wanted = languages_from_config(args.config)
        print(f"{len(wanted)} language(s) from {args.config}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # One streamed pass over the split, grouping by language, rather than one
    # request per language: the dataset ships a single table keyed by
    # iso_639_3 and iso_15924, and 101 separate loads is 101 downloads of the
    # same file.
    print(f"Loading {DATASET} split={args.split}. The dataset is gated, so this "
          "fails with a 401 unless you have accepted its terms and logged in.")
    dataset = load_dataset(DATASET, split=args.split, revision=args.revision)

    by_language = {}
    for row in dataset:
        code = f"{row['iso_639_3']}_{row['iso_15924']}"
        if wanted is not None and code not in wanted:
            continue
        by_language.setdefault(code, []).append(row["text"])

    if wanted is not None:
        missing = [code for code in wanted if code not in by_language]
        if missing:
            print(
                f"{len(missing)} language(s) named in {args.config} are not in "
                f"{DATASET} split={args.split}: {', '.join(sorted(missing))}. "
                "Nothing was written for them, so a run using that config will "
                "abort naming the missing file.",
                file=sys.stderr,
            )

    for code, lines in sorted(by_language.items()):
        out = args.output_dir / f"{code}.txt"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  {code:12s} {len(lines):5d} lines -> {out.relative_to(REPO_ROOT)}")

    # Record which revision produced these files. Without it a Hub-side update
    # to the dataset changes every number computed from this corpus and nothing
    # says so. The manifest is read by nothing at run time; it is the record a
    # later reader consults when two runs of the same commit disagree.
    manifest = {
        "dataset": DATASET,
        "split": args.split,
        "revision_requested": args.revision,
        "revision_resolved": _resolved_revision(dataset, args.revision),
        "n_languages": len(by_language),
    }
    manifest_path = args.output_dir / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nWrote {len(by_language)} language file(s) to {args.output_dir}.")
    print(f"Recorded the dataset revision in {manifest_path.relative_to(REPO_ROOT)}.")
    print("FLORES+ is CC-BY-SA 4.0. Cite NLLB Team et al., Nature 630 (2024).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
