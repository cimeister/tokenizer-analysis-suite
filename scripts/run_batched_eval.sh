#!/bin/bash
# Run intrinsic evals in batches of 3 tokenizers to avoid OOM.
# Results are merged afterwards.
set -euo pipefail

cd "$(dirname "$0")/.."

LANG_CONFIG="configs/downstream_lm_fineweb_lang_config.json"
MEASUREMENT_CONFIG="configs/text_measurement_config_lines.json"
AST_CONFIG="configs/starcoder_ast_config.json"
SAMPLES=1000
OUTPUT_BASE="results/downstream_lm_fineweb_1k"

export MPLBACKEND=Agg

for batch in 0 1 2 3 4 5; do
    TOK_CONFIG="configs/downstream_lm_tokenizers_batch${batch}.json"
    BATCH_DIR="${OUTPUT_BASE}/batch${batch}"

    if [ -f "${BATCH_DIR}/analysis_results.json" ]; then
        echo "=== Batch ${batch}: already done, skipping ==="
        continue
    fi

    echo "=== Batch ${batch}: $(cat ${TOK_CONFIG} | python3 -c 'import json,sys; print(list(json.load(sys.stdin).keys()))') ==="

    uv run tokenizer-analysis \
        --tokenizer-config "${TOK_CONFIG}" \
        --language-config "${LANG_CONFIG}" \
        --measurement-config "${MEASUREMENT_CONFIG}" \
        --code-ast-config "${AST_CONFIG}" \
        --use-builtin-math-data \
        --no-reconstruction \
        --save-full-results \
        --samples-per-lang "${SAMPLES}" \
        --output-dir "${BATCH_DIR}" \
        --dataset "fineweb_1k" \
        --no-plots

    echo "=== Batch ${batch}: done ==="
    echo ""
done

echo "=== All batches complete. Merging results... ==="

python3 -c "
import json, sys
from pathlib import Path

base = Path('${OUTPUT_BASE}')
merged = {}

for batch_dir in sorted(base.glob('batch*')):
    path = batch_dir / 'analysis_results.json'
    if not path.exists():
        print(f'WARNING: {path} not found, skipping', file=sys.stderr)
        continue
    with open(path) as f:
        data = json.load(f)

    for metric_name, metric_data in data.items():
        if metric_name not in merged:
            merged[metric_name] = {'per_tokenizer': {}, 'per_language': {}, 'metadata': {}}

        # Merge per_tokenizer
        if 'per_tokenizer' in metric_data:
            merged[metric_name]['per_tokenizer'].update(metric_data['per_tokenizer'])

        # Merge per_language (cross-tokenizer leaderboard)
        if 'per_language' in metric_data and isinstance(metric_data['per_language'], dict):
            for lang, lang_data in metric_data['per_language'].items():
                if lang not in merged[metric_name]['per_language']:
                    merged[metric_name]['per_language'][lang] = {}
                if isinstance(lang_data, dict):
                    merged[metric_name]['per_language'][lang].update(lang_data)

        # Merge metadata
        if 'metadata' in metric_data and isinstance(metric_data['metadata'], dict):
            merged[metric_name]['metadata'].update(metric_data['metadata'])

out_path = base / 'analysis_results.json'
with open(out_path, 'w') as f:
    json.dump(merged, f, indent=2)

n_metrics = len(merged)
n_toks = len(next(iter(merged.values()))['per_tokenizer']) if merged else 0
print(f'Merged {n_metrics} metrics across {n_toks} tokenizers -> {out_path}')
"

echo "=== Done ==="
