#!/bin/bash
# Re-run PA-BPE paper analyses on 6 tokenizer sets × FLORES+ devtest.
# Skipped: 60lang @ 256k (tokenizers not available locally).
# All no-dev variants removed from configs. UnigramLM only in 128k-unbalanced-30lang.

set -euo pipefail

ROOT=/users/cmeister747/tokenizer-intrinsic-evals/configs/pabpe_configs
OUT=/users/cmeister747/pa_tokenizers_branch/experiments/paper/reproduction/
COMMON=(
    --measurement-config configs/text_measurement_config_lines.json
    --generate-latex-tables --per-language-plots --no-global-lines --no-reconstruction
    --save-full-results --faceted-plots --run-grouped-analysis --morphscore
    --cer-time-budget 300
)

# 30-lang sets (use 30-lang devtest language config)
for vocab in 128k 256k; do
  for bal in unbalanced balanced; do
    if [[ "$vocab" == "128k" ]]; then
      tok_cfg=${ROOT}/tokenizer_config_30languages_${bal}.json
    else
      tok_cfg=${ROOT}/tokenizer_config_30languages_${vocab}_${bal}.json
    fi
    out_dir=${OUT}/30lang_${vocab}_${bal}
    mkdir -p ${out_dir}
    echo "=== 30lang ${vocab} ${bal} -> ${out_dir} ==="
    uv run tokenizer-analysis \
      --tokenizer-config ${tok_cfg} \
      --language-config ${ROOT}/language_config_30languages_devtest.json \
      --output-dir ${out_dir} \
      --dataset flores_pabpe30_${vocab}_${bal} \
      "${COMMON[@]}"
  done
done

# 60-lang sets (only 128k available)
for bal in unbalanced balanced; do
  tok_cfg=${ROOT}/tokenizer_config_60languages_${bal}.json
  out_dir=${OUT}/60lang_128k_${bal}
  mkdir -p ${out_dir}
  echo "=== 60lang 128k ${bal} -> ${out_dir} ==="
  uv run tokenizer-analysis \
    --tokenizer-config ${tok_cfg} \
    --language-config ${ROOT}/language_config_60languages.json \
    --output-dir ${out_dir} \
    --dataset flores_pabpe60_128k_${bal} \
    "${COMMON[@]}"
done

echo "=== all 6 analyses complete ==="
