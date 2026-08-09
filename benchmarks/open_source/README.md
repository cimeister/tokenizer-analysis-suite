# Open-source tokenizer benchmark

Nine widely used tokenizers measured on the full metric set. The report is
[REPORT.md](REPORT.md); everything needed to regenerate it is here.

| File | What it is |
|---|---|
| `tokenizers.json` | The nine tokenizers, by Hugging Face Hub id |
| `code_ast_config.json` | Maps each programming language to its directory under `code_data/` |
| `fetch_code_corpus.py` | Downloads `code_data/` from `bigcode/the-stack-smol-xs` |
| `run.sh` | The two commands, then the report render |
| `analysis_results.json` | The metrics the report is rendered from, with `run_metadata` |
| `sanity_results.json` | The 16 health checks per tokenizer, behind the health matrix in the report. See [../../docs/SANITY_CHECKS.md](../../docs/SANITY_CHECKS.md) |
| `render_report.py` | Turns those two files into `REPORT.md` |
| `REPORT.md` | Generated. Do not edit by hand |

## Regenerating it

```bash
uv sync --extra flores
hf auth login                                    # FLORES+ and two of the nine
                                                 # tokenizers are gated
uv run python scripts/fetch_flores.py --config configs/core_lang_config.json
uv run python benchmarks/open_source/fetch_code_corpus.py
bash benchmarks/open_source/run.sh
```

`code_data/` is not committed. It holds 1500 files from The Stack, each under
the license of the repository it came from, so it is fetched rather than
redistributed. The FLORES+ corpus under `parallel/` is fetched for the same
reason. FLORES+ is CC-BY-SA 4.0.

Two of the nine tokenizers, `meta-llama/Meta-Llama-3-8B` and
`google/gemma-2-9b`, are gated on the Hub and need an accepted license. Drop
them from `tokenizers.json` to run without one: no other number changes, since
every metric here is computed per tokenizer.

`run.sh` runs `tokenizer-analysis` and then `tokenizer-sanity-check` over the
same nine tokenizers, the same 13 languages and the same math corpus. The
second takes about 50 seconds and produces `sanity_results.json`. It is passed
`--exit-zero` because the script runs under `set -e` and the checker exits
non-zero on any warning, which these nine produce; the verdicts belong in the
report rather than in the script's exit code.

## The committed results file predates the operator-isolation prose change

`analysis_results.json` here was generated when `operator_isolation_rate` scored
the main corpus as a `prose` domain. That domain is now off unless
`--operator-prose-domain` is passed, and `run.sh` does not pass it, so a fresh
run gives an `operator_isolation_rate` block with two domains where the
committed file has three, and a pooled figure computed without prose. Prose
supplied 568 of the 455558 operator occurrences behind the committed numbers,
0.12 percent, so the pooled rates move in the fourth decimal. Every other metric
is unaffected.

Regeneration was deferred rather than run for this one change. Until it happens,
read the `operator_isolation_rate` block here as a three-domain measurement and
`by_domain.code` and `by_domain.math` as the two that a fresh run reproduces.

## What the committed results file leaves out

`analysis_results.json` here is the output of `run.sh` with two fields removed:
`encoding_speed`, which is wall-clock, and `run_metadata.timestamp_utc`. Both
change on every run, so leaving them in meant every regeneration produced a
diff of roughly ninety lines in which a real change to a measured value was not
visible. Your own run writes both.

## Reading the CER column

Which tokenizers report `mean_cer` depends on how fast the machine is. The
character error rate is an edit distance, and the run abandons it for a
tokenizer once it projects past `--cer-time-budget` (120 seconds here). In this
benchmark `bert-base-uncased` exceeds it, because a tokenizer that lowercases,
strips accents and substitutes unknown tokens has a large distance on every
text, so its `mean_cer` is `null`. On faster hardware it might not be.

`cer_skipped` in the results file distinguishes a skipped value from a measured
one, and `exact_match_rate` carries the same information without the time cost:
`bert-base-uncased` reconstructs 0.031 of its inputs exactly.
