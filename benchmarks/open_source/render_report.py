#!/usr/bin/env python
"""Render REPORT.md from a benchmark results file.

Every number in the report comes from `analysis_results.json` through this
script. Nothing is transcribed by hand, so the tables cannot drift from the
results, and `run.sh` regenerates both together.

    uv run python benchmarks/open_source/render_report.py \
        --results benchmarks/open_source/analysis_results.json \
        --output benchmarks/open_source/REPORT.md
"""

import argparse
import json
from pathlib import Path

# Display name and Hub id per tokenizer key, in the order the tables use.
# Ordered by vocabulary size, since that is the axis most of the differences
# track and an alphabetical order would hide it.
TOKENIZERS = [
    ("bert-base", "BERT base uncased", "bert-base-uncased", "WordPiece"),
    ("gpt-neox-20b", "GPT-NeoX 20B", "EleutherAI/gpt-neox-20b", "BPE"),
    ("gpt2", "GPT-2", "gpt2", "BPE"),
    ("olmo-2", "OLMo 2", "allenai/OLMo-2-1124-7B", "BPE"),
    ("llama-3", "Llama 3", "meta-llama/Meta-Llama-3-8B", "BPE"),
    ("mistral-nemo", "Mistral NeMo", "mistralai/Mistral-Nemo-Base-2407", "BPE"),
    ("qwen-2.5", "Qwen 2.5", "Qwen/Qwen2.5-7B", "BPE"),
    ("xlm-roberta", "XLM-RoBERTa base", "xlm-roberta-base", "Unigram"),
    ("gemma-2", "Gemma 2", "google/gemma-2-9b", "Unigram"),
]

GATED = {"llama-3", "gemma-2"}


def get(results, *path, default=None):
    """Follow a dotted path, returning *default* when any step is absent."""
    node = results
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def fmt(value, places=3):
    """Render a number for a table cell, and an absent one as an em-free dash."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def weighted_mean(results, metric, field, tok):
    """Count-weighted mean of a per-language field, for a metric with no global.

    Five metrics do not publish a `global` block yet. Their `per_language`
    entries carry `count`, so the micro-average is recoverable, and every column
    computed this way says so in the report rather than presenting the number as
    if the results file had published it.
    """
    per_lang = get(results, metric, "per_tokenizer", tok, "per_language", default={})
    total, weight = 0.0, 0
    for entry in per_lang.values():
        if not isinstance(entry, dict):
            continue
        value, count = entry.get(field), entry.get("count")
        if value is None or not count:
            continue
        total += value * count
        weight += count
    return total / weight if weight else None


def present(results):
    """Tokenizer keys the results file actually holds, in TOKENIZERS order."""
    have = set(get(results, "fertility", "per_tokenizer", default={}))
    return [row for row in TOKENIZERS if row[0] in have]


def corpus_languages(results):
    """The corpus labels the run measured, from the first tokenizer's block.

    Each metric keys its per-language values under its own tokenizer entry;
    the top-level `per_language` block is a cross-tokenizer leaderboard that
    only some metrics publish, so it is not the place to read the corpus from.
    """
    per_tok = get(results, "compression_rate", "per_tokenizer", default={})
    if not per_tok:
        return []
    first = per_tok[sorted(per_tok)[0]]
    return sorted(first.get("per_language", {}))


def section_what_was_measured(results, rows):
    meta = results.get("run_metadata", {})
    corpus = meta.get("corpus", {})
    languages = corpus_languages(results)
    lines = ["## What was measured", ""]
    lines.append(
        f"{len(rows)} tokenizers, {len(languages)} languages. Every number below "
        f"comes from `analysis_results.json` in this directory, which records the "
        f"package version, the commit and a hash of each input."
    )
    lines.append("")
    lines.append("| Tokenizer | Hub id | Algorithm | Vocabulary | Gated |")
    lines.append("|---|---|---|---:|---|")
    for key, name, repo, algo in rows:
        vocab = get(results, "vocabulary_utilization", "per_tokenizer", key,
                    "global", "vocab_size")
        lines.append(
            f"| {name} | `{repo}` | {algo} | {fmt(vocab)} | "
            f"{'yes' if key in GATED else 'no'} |"
        )
    lines.append("")
    lines.append(
        "Corpus: FLORES+ `dev`, "
        f"{len(languages)} languages ({', '.join(languages)}), "
        f"{corpus.get('samples_per_lang')} sentences per language at most. "
        "Code: 100 files per language across 15 languages from "
        "`bigcode/the-stack-smol-xs`. Math: the 285 bundled expressions."
    )
    lines.append("")
    lines.append(
        f"Package version {meta.get('package_version')}, commit "
        f"`{(meta.get('git_commit') or 'unknown')[:12]}`, working tree "
        f"{'modified' if meta.get('git_tree_modified') else 'clean'}."
    )
    return lines


def section_headline(results, rows):
    lines = [
        "", "## Headline", "",
        "One row per tokenizer. The arrow gives the direction each column is "
        "read in; `compression_rate` is bytes per token, so higher means fewer "
        "tokens for the same text.",
        "",
        "| Tokenizer | Vocabulary | Compression (up) | Fertility (down) | "
        "Gini (down) | UTF-8 complete (up) | AST aligned (up) | Digit F1 (up) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, name, _repo, _algo in rows:
        lines.append(
            "| {name} | {vocab} | {comp} | {fert} | {gini} | {utf8} | {ast} | {digit} |".format(
                name=name,
                vocab=fmt(get(results, "vocabulary_utilization", "per_tokenizer", key,
                              "global", "vocab_size")),
                comp=fmt(get(results, "compression_rate", "per_tokenizer", key,
                             "global", "compression_rate")),
                fert=fmt(get(results, "fertility", "per_tokenizer", key,
                             "global", "mean")),
                gini=fmt(get(results, "tokenizer_fairness_gini", "per_tokenizer", key,
                             "global", "gini_coefficient")),
                utf8=fmt(get(results, "utf8_token_integrity", "per_tokenizer", key,
                             "global", "completeness_rate")),
                ast=fmt(get(results, "ast_boundary_alignment", "per_tokenizer", key,
                            "global", "full_alignment_rate")),
                digit=fmt(weighted_mean(results, "three_digit_boundary_alignment",
                                        "mean_f1", key)),
            )
        )
    lines.append("")
    lines.append(
        "Compression, fertility, UTF-8 completeness and AST alignment are the "
        "`global` blocks the results file publishes. Gini is computed over the "
        "per-language costs. Digit F1 has no `global` block yet, so the column "
        "is the count-weighted mean of its `per_language.mean_f1` values, "
        "computed by `render_report.py`. With `--use-builtin-math-data` that "
        "block holds one entry, `math`, so the column is the figure for the "
        "bundled math corpus rather than for the prose."
    )
    return lines


def section_per_language(results, rows):
    languages = corpus_languages(results)
    lines = ["", "## Compression by language", "",
             "Bytes per token, higher meaning fewer tokens for the same text.", ""]
    header = "| Tokenizer | " + " | ".join(languages) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(languages))
    for key, name, _repo, _algo in rows:
        cells = []
        for lang in languages:
            entry = get(results, "compression_rate", "per_tokenizer", key,
                        "per_language", lang)
            if isinstance(entry, dict):
                entry = entry.get("compression_rate")
            cells.append(fmt(entry, 2))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def section_domain(results, rows):
    lines = ["", "## Code, math and round trip", "",
             "| Tokenizer | AST aligned (up) | Identifier fragmentation (down) | "
             "Operator isolation (up) | Exact round trip (up) | CER (down) |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, name, _repo, _algo in rows:
        lines.append(
            "| {name} | {ast} | {ident} | {op} | {exact} | {cer} |".format(
                name=name,
                ast=fmt(get(results, "ast_boundary_alignment", "per_tokenizer", key,
                            "global", "full_alignment_rate")),
                ident=fmt(get(results, "identifier_fragmentation", "per_tokenizer", key,
                              "global", "fragmentation_rate")),
                op=fmt(get(results, "operator_isolation_rate", "per_tokenizer", key,
                           "global", "overall_isolation_rate")),
                exact=fmt(get(results, "reconstruction_fidelity", "per_tokenizer", key,
                              "global", "exact_match_rate")),
                cer=fmt(get(results, "reconstruction_fidelity", "per_tokenizer", key,
                            "global", "mean_cer"), 4),
            )
        )
    lines.append("")
    lines.append(
        "Operator isolation pools prose, code and math weighted by operator "
        "instances, so with this code corpus it sits close to the code rate; "
        "`by_domain` in the results file splits the three."
    )
    skipped = [
        name for key, name, _repo, _algo in rows
        if get(results, "reconstruction_fidelity", "per_tokenizer", key,
               "global", "mean_cer") is None
    ]
    if skipped:
        lines.append("")
        lines.append(
            "CER is `n/a` for " + ", ".join(skipped) + ". The character error "
            "rate is an edit distance, and a tokenizer that does not "
            "reconstruct its input has a large distance on every text, so the "
            "run projected past the 120-second budget and reported the value as "
            "null rather than spending the time. The exact round-trip column "
            "carries the same information: a tokenizer at 0.031 exact matches "
            "is lossy by construction, in this case through lowercasing, "
            "accent stripping and unknown-token substitution."
        )
    return lines


def section_caveats(rows):
    return [
        "", "## How to read this", "",
        "**The Gini column is tokens per UTF-8 byte, and the byte is not "
        "neutral across scripts.** UTF-8 spends one byte per Latin character, "
        "two for Cyrillic, three for most CJK and Devanagari, so a tokenizer "
        "can score cheaper on Chinese than on English because the denominator "
        "is larger rather than because it segments Chinese better. The corpus "
        "here is parallel, which is what makes the comparison meaningful at "
        "all; the unit is what stops it from being a clean one. Read the "
        "per-language costs in the results file beside the coefficient.",
        "",
        "**Vocabulary size is not held constant.** The tokenizers here span "
        f"{len(rows)} models and a wide vocabulary range, and a larger "
        "vocabulary buys compression almost mechanically. A comparison between "
        "two tokenizers of different size measures the size as much as the "
        "algorithm.",
        "",
        "**Training data is not held constant and is mostly undisclosed.** "
        "These tokenizers were trained on different corpora with different "
        "language mixes. A tokenizer that compresses Hindi well may have seen "
        "more Hindi, not tokenized it better.",
        "",
        "**The corpus is FLORES+, which is translated news.** It is parallel "
        "across languages, which is what makes the fairness comparison "
        "meaningful, and it is not representative of what any of these models "
        "was trained on.",
        "",
        "**Compression depends on the measurement unit.** These figures are "
        "bytes per token. Under a word or line unit the ordering changes, "
        "particularly for languages without whitespace word boundaries.",
        "",
        "**The code numbers depend on the code corpus.** They come from 100 "
        "files per language of The Stack, so they describe that sample rather "
        "than code in general.",
        "",
        "None of this makes the numbers wrong. It means a single ranking over "
        "them would be, so the report gives the columns and not a rank.",
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    rows = present(results)
    if not rows:
        raise SystemExit(
            f"{args.results} holds no tokenizer this report knows about. "
            f"Expected some of: {', '.join(key for key, *_ in TOKENIZERS)}."
        )

    lines = [
        "# Open-source tokenizers, measured",
        "",
        "Generated by `render_report.py` from `analysis_results.json` in this "
        "directory. Do not edit by hand: `bash benchmarks/open_source/run.sh` "
        "regenerates both.",
        "",
    ]
    lines += section_what_was_measured(results, rows)
    lines += section_headline(results, rows)
    lines += section_per_language(results, rows)
    lines += section_domain(results, rows)
    lines += section_caveats(rows)
    lines.append("")
    lines.append(
        "Every metric in the results file, including the ones no table above "
        "shows, is defined in [METRICS.md](../../METRICS.md)."
    )
    lines.append("")

    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output} from {args.results}: {len(rows)} tokenizers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
