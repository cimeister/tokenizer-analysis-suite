#!/usr/bin/env python3
"""Rename tokenizer keys inside the paper-reproduction analysis results
(both full and slimmed JSONs) to match the display names in
`intrinsic_eval_tokenizers.json`, then regenerate plots + LaTeX tables
in-place using the existing library code.

Scope: only the 6 dirs under experiments/pabpe_paper_reproduction/.
Plotters expect the FULL schema (flat `global_utilization`, `global_bigram_entropy`,
etc.) so we load analysis_results_full.json for the regenerated artifacts.
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, '/users/cmeister747/tokenizer-intrinsic-evals')
from tokenizer_analysis.visualization.plots import generate_all_plots, setup_plot_style
from tokenizer_analysis.visualization.latex_tables import LaTeXTableGenerator

REPO_DIRS = [
    '/users/cmeister747/pa_tokenizers_branch/experiments/pabpe_paper_reproduction/30lang_128k_unbalanced',
    '/users/cmeister747/pa_tokenizers_branch/experiments/pabpe_paper_reproduction/30lang_128k_balanced',
    '/users/cmeister747/pa_tokenizers_branch/experiments/pabpe_paper_reproduction/30lang_256k_unbalanced',
    '/users/cmeister747/pa_tokenizers_branch/experiments/pabpe_paper_reproduction/30lang_256k_balanced',
    '/users/cmeister747/pa_tokenizers_branch/experiments/pabpe_paper_reproduction/60lang_128k_unbalanced',
    '/users/cmeister747/pa_tokenizers_branch/experiments/pabpe_paper_reproduction/60lang_128k_balanced',
]

def rename(name):
    if name == 'Classical':    return 'BPE'
    if name == 'Parity-aware': return 'PA-BPE'
    if name.startswith('Parity-aware '): return 'PA-BPE ' + name[len('Parity-aware '):]
    return name

def remap_recursive(obj):
    if isinstance(obj, dict):
        return {rename(k): remap_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap_recursive(x) for x in obj]
    if isinstance(obj, str):
        return rename(obj)
    return obj

setup_plot_style()

for d in REPO_DIRS:
    full_p = Path(d) / 'analysis_results.json'
    slim_p = Path(d) / 'analysis_results.json'  # slim is .json
    # actual file names:
    slim_p = Path(d) / 'analysis_results.json'
    full_p = Path(d) / 'analysis_results_full.json'
    if not full_p.exists() or not slim_p.exists():
        print(f'SKIP {d} (missing results)')
        continue

    full = remap_recursive(json.loads(full_p.read_text()))
    slim = remap_recursive(json.loads(slim_p.read_text()))
    full_p.write_text(json.dumps(full, indent=2))
    slim_p.write_text(json.dumps(slim, indent=2))

    tokenizer_names = list(full['compression_rate']['per_tokenizer'].keys())
    print(f'{os.path.basename(d)}: {tokenizer_names}')

    grouped = full.get('grouped_analysis')
    generate_all_plots(
        full, d, tokenizer_names,
        grouped_results=grouped,
        show_global_lines=False,
        per_language_plots=True,
        faceted_plots=True,
    )

    latex_dir = Path(d) / 'latex_tables'
    latex_dir.mkdir(exist_ok=True)
    gen = LaTeXTableGenerator(full, tokenizer_names)
    for table_type, method, caption, label in [
        ('basic', gen.generate_basic_metrics_table,
         'Basic Tokenization Metrics', 'tab:basic_metrics'),
        ('comprehensive', gen.generate_comprehensive_table,
         'Comprehensive Tokenizer Analysis', 'tab:comprehensive_metrics'),
    ]:
        content = method()
        if content:
            gen.save_table(content, str(latex_dir / f'{table_type}_metrics_table.tex'),
                           caption, label)

    print(f'  -> plots + latex tables regenerated')

print('\nall 6 reproduction dirs re-rendered.')
