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
MEASUREMENT_CONFIG="configs/text_measurement_config_bytes.json"
echo "============================================================"

# Split tokenizers into batches of 3 for memory safety
python3 -c "
import json
with open('${TOK_CONFIG}') as f:
    toks = json.load(f)
names = list(toks.keys())
batch_size = 3
for i in range(0, len(names), batch_size):
    chunk = {k: toks[k] for k in names[i:i+batch_size]}
    path = f'configs/_batch_tok_{i//batch_size}.json'
    with open(path, 'w') as f:
        json.dump(chunk, f, indent=2)
    print(f'Batch {i//batch_size}: {list(chunk.keys())} -> {path}')
"

FINEWEB_LANG="configs/downstream_lm_fineweb_lang_config.json"
OUTPUT_BASE="results/downstream_lm_fineweb_1k"

NUM_BATCHES=$(ls configs/_batch_tok_*.json 2>/dev/null | wc -l)
echo "Created ${NUM_BATCHES} batches of 3 tokenizers"
echo ""

for batch in $(seq 0 $((NUM_BATCHES - 1))); do
    BATCH_CONFIG="configs/_batch_tok_${batch}.json"
    BATCH_DIR="${OUTPUT_BASE}/batch${batch}"

    if [ -f "${BATCH_DIR}/analysis_results.json" ]; then
        echo "SKIP batch ${batch}: ${BATCH_DIR}/analysis_results.json already exists"
        continue
    fi

    echo "--- Batch ${batch}: $(python3 -c "import json; print(list(json.load(open('${BATCH_CONFIG}')).keys()))") ---"

    uv run tokenizer-analysis \
        --tokenizer-config "${BATCH_CONFIG}" \
        --language-config "${FINEWEB_LANG}" \
        --measurement-config "${MEASUREMENT_CONFIG}" \
        --use-builtin-math-data \
	--no-code-ast \
        --no-digit-boundary \
        --no-reconstruction \
        --save-full-results \
        --samples-per-lang 1000 \
        --output-dir "${BATCH_DIR}" \
        --dataset fineweb_1k \
        --no-plots
    echo "Batch ${batch} complete."
    echo ""
done

# Merge batched results
echo "Merging FineWeb 1k batched results..."
python3 -c "
import json
from pathlib import Path

base = Path('${OUTPUT_BASE}')
merged = {}

for batch_dir in sorted(base.glob('batch*')):
    # Prefer full results for per-language data
    path = batch_dir / 'analysis_results_full.json'
    if not path.exists():
        path = batch_dir / 'analysis_results.json'
    if not path.exists():
        print(f'WARNING: no results in {batch_dir}')
        continue
    with open(path) as f:
        data = json.load(f)

    for metric_name, metric_data in data.items():
        if metric_name not in merged:
            merged[metric_name] = {'per_tokenizer': {}, 'per_language': {}, 'metadata': {}}

        if 'per_tokenizer' in metric_data:
            merged[metric_name]['per_tokenizer'].update(metric_data['per_tokenizer'])

        if 'per_language' in metric_data and isinstance(metric_data['per_language'], dict):
            for lang, lang_data in metric_data['per_language'].items():
                if lang not in merged[metric_name]['per_language']:
                    merged[metric_name]['per_language'][lang] = {}
                if isinstance(lang_data, dict):
                    merged[metric_name]['per_language'][lang].update(lang_data)

        if 'metadata' in metric_data and isinstance(metric_data['metadata'], dict):
            merged[metric_name]['metadata'].update(metric_data['metadata'])

out_path = base / 'analysis_results.json'
with open(out_path, 'w') as f:
    json.dump(merged, f, indent=2)

n_metrics = len(merged)
n_toks = max(len(v['per_tokenizer']) for v in merged.values()) if merged else 0
print(f'Merged {n_metrics} metrics across {n_toks} tokenizers -> {out_path}')
"

# Cleanup batch configs
rm -f configs/_batch_tok_*.json

echo ""
echo "============================================================"
echo "All intrinsic evaluations complete."
echo "Results:"
echo "  FLORES: results/downstream_lm_flores/"
echo "  FineWeb: results/downstream_lm_fineweb_1k/"
echo "============================================================"
