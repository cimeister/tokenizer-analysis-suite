#!/bin/bash
# Run intrinsic evaluations for all downstream LM tokenizers.
# Phase 1: FLORES data (small, all tokenizers at once, with AST metrics)
# Phase 2: FineWeb 1k data (large, batched 3 tokenizers at a time)
#
# Usage: bash scripts/run_downstream_lm_evals.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export MPLBACKEND=Agg

TOK_CONFIG="configs/downstream_lm_tokenizers.json"
MEASUREMENT_CONFIG="configs/text_measurement_config_lines.json"
AST_CONFIG="starcoder_ast_config.json"   # NOT configs/ — moved to repo root in the 2026-07-04 v1.0.0 restructure

echo "============================================================"
echo "Phase 1: FLORES data (997 sentences/lang, all 18 tokenizers)"
echo "============================================================"

if [ -f "results/downstream_lm_flores/analysis_results.json" ]; then
    echo "SKIP: results/downstream_lm_flores/analysis_results.json already exists"
else
    uv run tokenizer-analysis \
        --tokenizer-config "${TOK_CONFIG}" \
        --language-config configs/downstream_lm_lang_config.json \
        --measurement-config "${MEASUREMENT_CONFIG}" \
        --code-ast-config "${AST_CONFIG}" \
        --use-builtin-math-data \
        --verbose \
        --save-full-results \
        --output-dir results/downstream_lm_flores \
        --dataset flores 
    echo "Phase 1 complete."
fi

echo "============================================================"
echo "All intrinsic evaluations complete."
echo "Results:"
echo "  FLORES: results/downstream_lm_flores/"
echo "============================================================"
