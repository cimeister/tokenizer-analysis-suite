#!/bin/bash
#SBATCH --job-name=tok_analysis
#SBATCH --output=tok_analysis_%j.out
#SBATCH --error=tok_analysis_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -A a139



/users/cmeister747/tok_training_env/bin/python scripts/run_tokenizer_analysis.py \
    --tokenizer-config configs/test_tokenizers_w_unimix.json \
    --language-config configs/flores+_lang_config.json \
    --measurement-config configs/text_measurement_config_lines.json \
    --code-ast-config configs/starcoder_ast_config.json \
    --per-language-plots \
    --no-global-lines \
    --update-results-md \
    --dataset flores+
