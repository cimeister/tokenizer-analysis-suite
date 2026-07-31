#!/usr/bin/env bash
# Regenerate the open-source tokenizer benchmark.
#
#   bash benchmarks/open_source/run.sh
#
# Prerequisites, both fetched rather than redistributed:
#   uv sync --extra flores
#   hf auth login                                        # FLORES+ is gated
#   uv run python scripts/fetch_flores.py --config configs/core_lang_config.json
#   uv run python benchmarks/open_source/fetch_code_corpus.py
#
# --run-grouped-analysis is deliberately absent. It recomputes every metric once
# per language group, 25 groups for this config, which multiplies the runtime by
# roughly five and produces nothing the report reads. Add it if you want the
# per-group breakdown in analysis_results.json.
#
# --samples-per-lang 250 rather than the whole 997-sentence split: the corpus is
# parallel, so 250 sentences per language is enough for stable compression and
# fertility figures, and the character error rate is the one metric whose cost
# grows with it.
#
# Two of the nine tokenizers are gated on the Hugging Face Hub (meta-llama and
# google/gemma-2). Accept their licenses first, or drop them from
# tokenizers.json and rerun; every other number is unaffected by their absence.
#
# Most of the run is encoding and the character error rate, which is an edit
# distance in Python. --cer-time-budget 120 is generous for the byte-level
# tokenizers, which reconstruct exactly and finish in seconds. A lossy tokenizer
# such as bert-base-uncased has a large edit distance on every text and exceeds
# any practical budget, so its mean_cer comes back null and the log names the
# projection. Its exact_match_rate still reports the lossiness.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
OUTPUT_DIR="${1:-${HERE}}"

cd "${REPO_ROOT}"

if [ ! -f parallel/eng_Latn.txt ]; then
    echo "The FLORES+ corpus is not in parallel/. Fetch it first:" >&2
    echo "  uv run python scripts/fetch_flores.py --config configs/core_lang_config.json" >&2
    exit 1
fi
if [ ! -d "${HERE}/code_data" ]; then
    echo "The code corpus is not in ${HERE}/code_data. Fetch it first:" >&2
    echo "  uv run python benchmarks/open_source/fetch_code_corpus.py" >&2
    exit 1
fi

uv run tokenizer-analysis \
    --tokenizer-config "${HERE}/tokenizers.json" \
    --language-config configs/core_lang_config.json \
    --code-ast-config "${HERE}/code_ast_config.json" \
    --samples-per-lang 250 \
    --use-builtin-math-data \
    --cer-time-budget 120 \
    --save-full-results \
    --output-dir "${OUTPUT_DIR}"

uv run python "${HERE}/render_report.py" \
    --results "${OUTPUT_DIR}/analysis_results.json" \
    --output "${HERE}/REPORT.md"

echo "Wrote ${OUTPUT_DIR}/analysis_results.json and ${HERE}/REPORT.md"
