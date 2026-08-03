"""
Simplified plotting functions for tokenizer analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List, Any, Optional
import logging 
logger = logging.getLogger(__name__)


# Configure matplotlib to use Times font family
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # Use STIX fonts for math (Times-compatible)

# Paul Tol's colorblind-friendly palette
TOL_COLORS = [
    '#EE7733',  # Orange
    '#0077BB',  # Blue  
    '#33BBEE',  # Light blue
    '#EE3377',  # Red
    '#CC3311',  # Dark red
    '#009988',  # Teal
    '#BBBBBB',  # Grey
    '#000000'   # Black
]


def _plottable(value: Any) -> float:
    """Map a metric value to something matplotlib can draw, without inventing one.

    Missing data is ``None`` throughout the metrics (see
    ``BaseMetrics.safe_divide`` and ``UTF8IntegrityMetrics._ratio``). Passing
    that to matplotlib raises, and substituting 0.0 draws a zero-height bar that
    a reader cannot tell from a measured zero. NaN is the one value matplotlib
    renders as absent, so that is what missing data becomes here.
    """
    if value is None:
        return float('nan')
    if isinstance(value, (int, float)):
        return float(value)
    return float('nan')


def _finite(values: List[float]) -> List[float]:
    """Keep only values that are real measurements, for means and guards."""
    return [v for v in values if v is not None and not np.isnan(v)]


def setup_plot_style():
    """Setup consistent plotting style."""
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'font.size': 18,
        'axes.labelsize': 22,
        'axes.titlesize': 26,
        'xtick.labelsize': 18,
        'ytick.labelsize': 20,
        'legend.fontsize': 14,
        'figure.dpi': 300
    })


def get_colors(n_items: int) -> List[str]:
    """Get colorblind-friendly colors."""
    if n_items <= len(TOL_COLORS):
        return TOL_COLORS[:n_items]
    elif n_items <= 12:
        # Use matplotlib's tab10 + tab20 which has good contrast
        return sns.color_palette("tab10", n_items)
    else:
        # For larger numbers, use matplotlib's tab20 which is reasonably colorblind friendly
        return plt.cm.tab20(np.linspace(0, 1, n_items))


# Centralized label and title generation functions
def get_metric_display_name(metric_key: str) -> str:
    """Get display name for a metric."""
    metric_names = {
        'fertility': 'Fertility',
        'compression_rate': 'Compression Rate',
        'vocabulary_utilization': 'Vocabulary Utilization',
        'vocab_util_cross_lingual_cov': 'Vocab. Utilization CoV',
        'tokenizer_fairness_gini': 'Gini Coefficient',
        'bigram_entropy': 'Bigram Entropy',
        'morphscore': 'MorphScore',
        'unk_percentage': 'UNK Percentage'
    }
    return metric_names.get(metric_key, metric_key.replace('_', ' ').title())


# Direction of "better" performance for each metric. Unicode arrows are
# appended to plot titles. Metrics absent from this map get no arrow.
METRIC_BETTER_DIRECTION = {
    'fertility': '↓',
    'compression_rate': '↑',
    'vocabulary_utilization': '↑',
    'vocab_util_cross_lingual_cov': '↓',
    'tokenizer_fairness_gini': '↓',
    'bigram_entropy': '↑',
    'trigram_entropy': '↑',
    'shannon_entropy': '↑',
    'renyi_efficiency': '↑',
    'morphscore_recall': '↑',
    'morphscore_precision': '↑',
    'unk_percentage': '↓',
    'utf8_token_integrity': '↑',
    'utf8_char_split': '↓',
}


def _norm_unit(metadata: Optional[Dict]) -> str:
    """Singular form of the normalization unit (e.g. 'byte', 'word')."""
    norm_method = 'unit'
    if metadata:
        norm_method = metadata.get('normalization_method', 'unit')
    return norm_method.rstrip('s')


def get_ylabel(metric_key: str, metadata: Optional[Dict] = None) -> str:
    """Get y-axis label for a metric, with units where applicable."""
    norm = _norm_unit(metadata)

    labels = {
        'fertility': f'Fertility (tokens / {norm})',
        'compression_rate': f'Compression ({norm}s / token)',
        'vocabulary_utilization': 'Vocabulary Utilization (%)',
        'vocab_util_cross_lingual_cov': 'Vocab. Utilization CoV (cross-lingual)',
        'tokenizer_fairness_gini': 'Gini Coefficient',
        'bigram_entropy': 'Bigram Entropy Efficiency (η)',
        'trigram_entropy': 'Trigram Entropy Efficiency (η)',
        'shannon_entropy': 'Shannon Entropy (bits / token)',
        'renyi_efficiency': 'Rényi Efficiency',
        'morphscore_recall': 'MorphScore Recall',
        'morphscore_precision': 'MorphScore Precision',
        'unk_percentage': 'UNK Percentage (%)',
        'utf8_token_integrity': 'Completeness Rate',
        'utf8_char_split': 'Splits per 1k Tokens',
    }
    return labels.get(metric_key, metric_key.replace('_', ' ').title())


def _arrow_suffix(metric_key: Optional[str]) -> str:
    arrow = METRIC_BETTER_DIRECTION.get(metric_key)
    return f' ({arrow})' if arrow else ''


def get_plot_title(plot_type: str, metric_key: str = None, context: str = None) -> str:
    """Get plot title with a direction-of-better arrow appended where applicable."""
    metric_display = get_metric_display_name(metric_key) if metric_key else ''
    arrow = _arrow_suffix(metric_key)

    # Special cases for specific plots
    if metric_key == 'lorenz_curves':
        return 'Lorenz Curves - Cross-Language Fairness'
    elif metric_key == 'morphscore_recall':
        return f'MorphScore Recall Comparison{arrow}'
    elif metric_key == 'morphscore_precision':
        return f'MorphScore Precision Comparison{arrow}'
    elif metric_key == 'tokenizer_fairness_gini' and plot_type == 'individual':
        return f'Cross-Language Fairness (Gini Coefficient){arrow}'

    titles = {
        'individual': f'{metric_display} Comparison{arrow}',
        'per_language': f'{metric_display} by Language{arrow}',
        'faceted': f'Faceted Analysis: {metric_display}{arrow}',
        'grouped': (f'{metric_display} by {context}{arrow}' if context
                    else f'{metric_display} Analysis{arrow}')
    }
    return titles.get(plot_type, f'{metric_display} Analysis{arrow}')

def format_language_labels(lang_code):
    return lang_code.split('_')[0]

def save_plot(fig, filepath: str):
    """Save plot to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_metric_bar_chart(results: Dict[str, Any], save_path: str, tokenizer_names: List[str],
                          metric_key: str, value_extractor, show_global_lines: bool = True,
                          ylim: Optional[tuple] = None, global_avg_fmt: str = '.2f',
                          label_metric_key: Optional[str] = None):
    """Plot a bar chart for any metric.

    Args:
        results: Full results dict.
        save_path: File path to save the plot.
        tokenizer_names: Ordered list of tokenizer names.
        metric_key: Top-level key in *results* (e.g. ``'fertility'``).
        value_extractor: Callable ``(tok_data_dict) -> (value, optional_std)``
            or ``(tok_data_dict) -> value``.  When the return is a scalar,
            no error bars are drawn.
        show_global_lines: Whether to draw a global average line.
        ylim: Optional (ymin, ymax) tuple for the y-axis.
        global_avg_fmt: Format string for the global average label.
        label_metric_key: Optional distinct metric identity used only for the
            title / y-axis label (when the results-index ``metric_key`` is a
            shared parent, e.g. a sub-field of ``vocabulary_utilization``).
            Defaults to ``metric_key``.
    """
    if metric_key not in results:
        return

    fig, ax = plt.subplots()
    per_tok = results[metric_key]['per_tokenizer']

    values = []
    stds = []
    labels = []

    for tok_name in tokenizer_names:
        if tok_name not in per_tok:
            continue
        extracted = value_extractor(per_tok[tok_name])
        if isinstance(extracted, tuple):
            val, std = extracted
        else:
            val, std = extracted, None
        # Skip tokenizers with a missing/None metric value (e.g. the
        # cross-lingual utilization CoV is None for <2-language
        # tokenizers). A visible omission, not a fabricated 0.0 or nan, which
        # matches how latex_tables renders a missing value as '---'.
        if val is None:
            continue
        stds.append(std)
        values.append(val)
        labels.append(tok_name)

    if values:
        colors = get_colors(len(values))
        yerr = stds if any(s is not None for s in stds) else None
        ax.bar(labels, values, yerr=yerr, capsize=5 if yerr else 0,
               color=colors, alpha=0.8)

        if show_global_lines:
            measured = _finite(values)
            global_mean = np.mean(measured) if measured else float('nan')
            ax.axhline(y=global_mean, color='red', linestyle='--', alpha=0.7,
                       label=f'Global Average: {global_mean:{global_avg_fmt}}')
            ax.legend()

        metadata = results[metric_key].get('metadata', {})
        lk = label_metric_key or metric_key
        ax.set_ylabel(get_ylabel(lk, metadata))
        ax.set_title(get_plot_title('individual', lk))
        if ylim is not None:
            ax.set_ylim(*ylim)
        plt.xticks(rotation=45)

    save_plot(fig, save_path)


def plot_fertility(results: Dict[str, Any], save_path: str, tokenizer_names: List[str], show_global_lines: bool = True):
    """Plot fertility metric comparison."""
    plot_metric_bar_chart(
        results, save_path, tokenizer_names, 'fertility',
        lambda td: (td['global']['mean'], td['global']['std']),
        show_global_lines=show_global_lines,
    )


def plot_vocabulary_utilization(results: Dict[str, Any], save_path: str, tokenizer_names: List[str], show_global_lines: bool = True):
    """Plot vocabulary utilization comparison."""
    plot_metric_bar_chart(
        results, save_path, tokenizer_names, 'vocabulary_utilization',
        lambda td: td['global_utilization'] * 100,
        show_global_lines=show_global_lines, global_avg_fmt='.1f',
    )


def plot_vocab_util_cross_lingual_cov(results: Dict[str, Any], save_path: str, tokenizer_names: List[str], show_global_lines: bool = True):
    """Plot the cross-lingual coefficient of variation of vocabulary utilization.

    Reads the already-computed ``per_language_cov`` (``None`` for
    <2-language tokenizers; those bars are skipped by
    ``plot_metric_bar_chart``).  CoV has no per-tokenizer error, so the
    extractor returns a bare scalar (no error bars).
    """
    plot_metric_bar_chart(
        results, save_path, tokenizer_names, 'vocabulary_utilization',
        lambda td: td['per_language_cov'],
        show_global_lines=show_global_lines, global_avg_fmt='.3f',
        label_metric_key='vocab_util_cross_lingual_cov',
    )


def plot_compression_rate(results: Dict[str, Any], save_path: str, tokenizer_names: List[str], show_global_lines: bool = True):
    """Plot compression rate comparison."""
    plot_metric_bar_chart(
        results, save_path, tokenizer_names, 'compression_rate',
        lambda td: td['global']['compression_rate'],
        show_global_lines=show_global_lines,
    )


def plot_bigram_entropy(results: Dict[str, Any], save_path: str, tokenizer_names: List[str], show_global_lines: bool = True):
    """Plot bigram entropy comparison."""
    plot_metric_bar_chart(
        results, save_path, tokenizer_names, 'bigram_entropy',
        lambda td: td['global_bigram_entropy'],
        show_global_lines=show_global_lines, ylim=(0, 1), global_avg_fmt='.3f',
    )


def plot_gini_coefficient(results: Dict[str, Any], save_path: str, tokenizer_names: List[str], show_global_lines: bool = True):
    """Plot Gini coefficient comparison for fairness."""
    plot_metric_bar_chart(
        results, save_path, tokenizer_names, 'tokenizer_fairness_gini',
        lambda td: td['gini_coefficient'],
        show_global_lines=show_global_lines, ylim=(0, 1), global_avg_fmt='.3f',
    )


def plot_lorenz_curves(results: Dict[str, Any], save_path: str, tokenizer_names: List[str]):
    """Plot Lorenz curves for fairness analysis.

    The curve data lives under ``tokenizer_fairness_gini`` since 1.0, because
    ``1 - 2*area(lorenz)`` equals the Gini coefficient exactly and the two were
    published as separate metrics. The old top-level key is still accepted so a
    results file written before the merge still plots.
    """
    lorenz_data = None
    if 'lorenz_curve_data' in results:
        lorenz_data = results['lorenz_curve_data'].get('per_tokenizer', {})
    elif 'tokenizer_fairness_gini' in results:
        gini_per_tok = results['tokenizer_fairness_gini'].get('per_tokenizer', {})
        lorenz_data = {
            tok: entry['lorenz_curve']
            for tok, entry in gini_per_tok.items()
            if isinstance(entry, dict) and 'lorenz_curve' in entry
        }
    if not lorenz_data:
        return

    fig, ax = plt.subplots()

    for tok_name in tokenizer_names:
        if tok_name in lorenz_data:
            data = lorenz_data[tok_name]
            if 'x_values' in data and 'y_values' in data:
                ax.plot(data['x_values'], data['y_values'], label=tok_name, linewidth=2)
    
    # Add equality line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality')
    ax.set_xlabel('Cumulative Proportion of Languages')
    ax.set_ylabel('Cumulative Proportion of Costs')
    title = get_plot_title('individual', 'lorenz_curves')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_plot(fig, save_path)


def plot_morphscore(results: Dict[str, Any], save_path: str, tokenizer_names: List[str]):
    """Plot MorphScore comparison."""
    if 'morphscore' not in results or 'per_tokenizer' not in results['morphscore']:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    morph_data = results['morphscore']['per_tokenizer']
    
    recall_values = []
    precision_values = []
    labels = []
    
    for tok_name in tokenizer_names:
        if tok_name in morph_data and 'summary' in morph_data[tok_name]:
            summary = morph_data[tok_name]['summary']
            # No 0 default: an absent score is absent, and drawing it at 0
            # is indistinguishable from a measured 0.
            recall_values.append(_plottable(summary.get('avg_morphscore_recall')))
            precision_values.append(_plottable(summary.get('avg_morphscore_precision')))
            labels.append(tok_name)
    
    if recall_values:
        colors = get_colors(len(labels))
        ax1.bar(labels, recall_values, color=colors, alpha=0.8)
        ylabel1 = get_ylabel('morphscore_recall')
        title1 = get_plot_title('individual', 'morphscore_recall')
        ax1.set_ylabel(ylabel1)
        ax1.set_title(title1)
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(labels, precision_values, color=colors, alpha=0.8)
        ylabel2 = get_ylabel('morphscore_precision')
        title2 = get_plot_title('individual', 'morphscore_precision')
        ax2.set_ylabel(ylabel2)
        ax2.set_title(title2)
        ax2.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    save_plot(fig, save_path)


def plot_utf8_integrity(results: Dict[str, Any], save_path: str, tokenizer_names: List[str]):
    """Plot UTF-8 integrity metrics: token integrity rate and char splits per 1k tokens."""
    integrity_block = results.get('utf8_token_integrity', {})
    integrity_summary = integrity_block.get('summary', {})
    integrity_per_tok = integrity_block.get('per_tokenizer', {})

    # Split data moved under utf8_token_integrity in 1.0 (metrics/redundancy.py),
    # so read the merged location first and fall back to the old top-level key
    # for results files written before the merge. Reading only the old key left
    # the right panel blank while the figure was still written under the same
    # filename, which is the failure mode this metric exists to catch.
    legacy_splits = results.get('utf8_char_split', {}).get('summary', {})

    def _split_value(tok_name):
        entry = integrity_per_tok.get(tok_name, {})
        merged = entry.get('char_split') if isinstance(entry, dict) else None
        if isinstance(merged, dict):
            source = merged.get('global', merged)
            if isinstance(source, dict) and 'splits_per_1k_tokens' in source:
                return source['splits_per_1k_tokens']
        if tok_name in legacy_splits:
            return legacy_splits[tok_name].get('splits_per_1k_tokens')
        return None

    def _integrity_value(tok_name):
        if tok_name in integrity_summary:
            # No default: a missing rate is missing, and 1.0 is the best
            # possible value, so defaulting to it drew an absent measurement as
            # a perfect one.
            return integrity_summary[tok_name].get('completeness_rate')
        entry = integrity_per_tok.get(tok_name, {})
        if isinstance(entry, dict) and isinstance(entry.get('global'), dict):
            return entry['global'].get('completeness_rate')
        return None

    integrity_values = [_integrity_value(t) for t in tokenizer_names]
    split_values = [_split_value(t) for t in tokenizer_names]
    labels = list(tokenizer_names)

    if not any(v is not None for v in integrity_values + split_values):
        return

    has_integrity = any(v is not None for v in integrity_values)
    has_splits = any(v is not None for v in split_values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Colour is keyed to the tokenizer, not to its position in a panel's
    # filtered list. Slicing colors[:len(filtered)] per panel meant a tokenizer
    # missing from one panel shifted every colour in the other: measured with
    # tokenizer A absent from the integrity panel, B was #EE7733 on the left and
    # #0077BB on the right of the same figure.
    colors = get_colors(len(labels))
    color_by_tokenizer = dict(zip(labels, colors))

    # Left panel: Token Boundary Integrity Rate
    valid_integrity = [(l, v) for l, v in zip(labels, integrity_values) if v is not None]
    if valid_integrity:
        vi_labels, vi_vals = zip(*valid_integrity)
        vi_colors = [color_by_tokenizer[l] for l in vi_labels]
        ax1.bar(vi_labels, vi_vals, color=vi_colors, alpha=0.8)
        ax1.set_ylabel(get_ylabel('utf8_token_integrity'))
        ax1.set_title(f'Token UTF-8 Completeness Rate{_arrow_suffix("utf8_token_integrity")}')
        ax1.set_ylim(0, 1.05)
        ax1.tick_params(axis='x', rotation=45)

    # Right panel: Character Splits per 1k Tokens
    valid_splits = [(l, v) for l, v in zip(labels, split_values) if v is not None]
    if valid_splits:
        vs_labels, vs_vals = zip(*valid_splits)
        vs_colors = [color_by_tokenizer[l] for l in vs_labels]
        ax2.bar(vs_labels, vs_vals, color=vs_colors, alpha=0.8)
        ax2.set_ylabel(get_ylabel('utf8_char_split'))
        ax2.set_title(f'Character Splits per 1k Tokens{_arrow_suffix("utf8_char_split")}')
        ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_plot(fig, save_path)


# Per-group scalar extractors for grouped plots. Each takes the
# per-tokenizer dict produced by the metric on a filtered (per-group)
# data subset and returns a single float for the bar height.
GROUPED_VALUE_EXTRACTORS = {
    'fertility': lambda td: td['global']['mean'],
    'compression_rate': lambda td: td['global']['compression_rate'],
    'vocabulary_utilization': lambda td: td['global_utilization'] * 100,
    'tokenizer_fairness_gini': lambda td: td['gini_coefficient'],
    'bigram_entropy': lambda td: td['global_bigram_entropy'],
    'morphscore': lambda td: td['summary']['avg_morphscore_recall'],
}


def plot_grouped_analysis(grouped_results: Dict[str, Dict[str, Any]], save_dir: str,
                         metric_name: str, group_type: str,
                         value_extractor: Optional[callable] = None,
                         tokenizer_names: Optional[List[str]] = None):
    """Plot grouped analysis results.

    Args:
        value_extractor: Callable ``(tok_data_dict) -> float`` returning the
            per-group scalar for a tokenizer. Defaults to the entry in
            ``GROUPED_VALUE_EXTRACTORS`` for ``metric_name``.
        tokenizer_names: Optional ordered subset of tokenizers to plot. When
            ``None``, every tokenizer present in the grouped results is shown.
    """
    if group_type not in grouped_results:
        return

    if value_extractor is None:
        value_extractor = GROUPED_VALUE_EXTRACTORS.get(metric_name)
        if value_extractor is None:
            logger.warning(f"No grouped-plot extractor registered for metric {metric_name}; skipping")
            return

    fig, ax = plt.subplots(figsize=(12, 8))
    group_data = grouped_results[group_type]

    # Extract data for plotting
    groups = list(group_data.keys())
    discovered = set()

    for group_results in group_data.values():
        if metric_name in group_results:
            discovered.update(group_results[metric_name]['per_tokenizer'].keys())

    if tokenizer_names is not None:
        # Preserve user-supplied order; drop names absent from results
        tokenizer_names = [t for t in tokenizer_names if t in discovered]
    else:
        tokenizer_names = sorted(discovered)

    # Check if we have tokenizers to plot
    if not tokenizer_names:
        logger.warning(f"No tokenizers found for metric {metric_name} in group type {group_type}")
        return

    # Plot data
    x_pos = np.arange(len(groups))
    width = 0.8 / len(tokenizer_names)

    colors = get_colors(len(tokenizer_names))
    for i, tok_name in enumerate(tokenizer_names):
        values = []
        for group_name in groups:
            if (metric_name in group_data[group_name] and
                tok_name in group_data[group_name][metric_name]['per_tokenizer']):
                tok_data = group_data[group_name][metric_name]['per_tokenizer'][tok_name]
                try:
                    values.append(_plottable(value_extractor(tok_data)))
                except (KeyError, TypeError) as e:
                    logger.warning(
                        f"Extractor failed for {metric_name}/{group_name}/{tok_name}: {e}"
                    )
                    values.append(float('nan'))
            else:
                values.append(float('nan'))

        ax.bar(x_pos + i * width, values, width, label=tok_name, color=colors[i], alpha=0.8)
    
    xlabel = group_type.replace('_', ' ').title()
    # Pull metadata from the first group that has it so units render correctly
    metadata = next(
        (g[metric_name].get('metadata', {})
         for g in group_data.values() if metric_name in g),
        {},
    )
    ylabel = get_ylabel(metric_name, metadata)
    title = get_plot_title('grouped', metric_name, group_type.replace('_', ' ').title())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_pos + width * (len(tokenizer_names) - 1) / 2)
    # Rotate tick labels when many groups so they don't overlap.
    tick_rotation = 45 if len(groups) > 5 else 0
    tick_ha = 'right' if tick_rotation else 'center'
    # Scale font up for sparse horizontal axes; keep 18 when rotated/dense.
    if tick_rotation:
        tick_fontsize = 18
    elif len(groups) <= 2:
        tick_fontsize = 28
    elif len(groups) == 3:
        tick_fontsize = 24
    else:
        tick_fontsize = 20
    ax.set_xticklabels(groups, rotation=tick_rotation, ha=tick_ha, fontsize=tick_fontsize)
    ax.legend()
    
    save_path = os.path.join(save_dir, f'{group_type}_{metric_name}_individual.png')
    save_plot(fig, save_path)


def generate_all_plots(results: Dict[str, Any], save_dir: str, tokenizer_names: List[str],
                      grouped_results: Optional[Dict[str, Dict[str, Any]]] = None,
                      show_global_lines: bool = True, per_language_plots: bool = False,
                      faceted_plots: bool = False):
    """Generate all standard plots."""
    setup_plot_style()
    
    # Basic metrics
    plot_fertility(results, os.path.join(save_dir, 'fertility_individual.svg'), tokenizer_names, show_global_lines)
    plot_vocabulary_utilization(results, os.path.join(save_dir, 'vocabulary_utilization_individual.svg'), tokenizer_names, show_global_lines)
    plot_vocab_util_cross_lingual_cov(results, os.path.join(save_dir, 'vocab_util_cross_lingual_cov_individual.svg'), tokenizer_names, show_global_lines)

    # Information theory
    plot_compression_rate(results, os.path.join(save_dir, 'compression_rate_individual.svg'), tokenizer_names, show_global_lines)
    plot_bigram_entropy(results, os.path.join(save_dir, 'bigram_entropy_individual.svg'), tokenizer_names, show_global_lines)

    # Fairness
    plot_gini_coefficient(results, os.path.join(save_dir, 'tokenizer_fairness_gini_individual.svg'), tokenizer_names, show_global_lines)
    plot_lorenz_curves(results, os.path.join(save_dir, 'lorenz_curves_individual.svg'), tokenizer_names)

    # Morphological
    plot_morphscore(results, os.path.join(save_dir, 'morphscore_individual.svg'), tokenizer_names)

    # UTF-8 integrity
    plot_utf8_integrity(results, os.path.join(save_dir, 'utf8_integrity.svg'), tokenizer_names)
    
    # Per-language plots
    if per_language_plots:
        _generate_per_language_plots(results, save_dir, tokenizer_names, show_global_lines)
    
    # Faceted plots
    if faceted_plots:
        _generate_faceted_plots(results, save_dir, tokenizer_names, show_global_lines)
    
    # Grouped analysis
    if grouped_results:
        grouped_dir = os.path.join(save_dir, 'grouped_plots')
        os.makedirs(grouped_dir, exist_ok=True)
        
        for group_type, group_data in grouped_results.items():
            if not group_data:  # Skip empty group data
                continue
            # Unguarded, like the other roughly fifteen plot calls in this
            # function. plot_grouped_analysis already returns early and logs
            # when a metric is absent from the grouped results, which is what
            # the handler here was covering; what it also covered was a real
            # failure, which returned normally and left the figure missing.
            for metric in ['fertility', 'vocabulary_utilization', 'compression_rate', 'morphscore']:
                plot_grouped_analysis(grouped_results, grouped_dir, metric, group_type,
                                      tokenizer_names=tokenizer_names)


def _generate_per_language_plots(results: Dict[str, Any], save_dir: str, 
                                tokenizer_names: List[str], show_global_lines: bool):
    """Generate per-language breakdown plots with grouped bars (languages on x-axis)."""
    # Create per-language subdirectory
    per_lang_dir = os.path.join(save_dir, 'per-language')
    os.makedirs(per_lang_dir, exist_ok=True)
    
    # Generate combined subplot layout
    _plot_per_language_combined_subplots(results, per_lang_dir, tokenizer_names, show_global_lines)
    
    # Also generate separate per-language plot for each metric (legacy)
    _plot_per_language_fertility(results, per_lang_dir, tokenizer_names, show_global_lines)
    _plot_per_language_compression_rate(results, per_lang_dir, tokenizer_names, show_global_lines)
    _plot_per_language_vocabulary_utilization(results, per_lang_dir, tokenizer_names, show_global_lines)
    _plot_per_language_gini_coefficient(results, per_lang_dir, tokenizer_names, show_global_lines)
    _plot_per_language_bigram_entropy(results, per_lang_dir, tokenizer_names, show_global_lines)


def _generate_faceted_plots(results: Dict[str, Any], save_dir: str,
                          tokenizer_names: List[str], show_global_lines: bool):
    """Generate faceted plots with subplots for each tokenizer."""
    facet_dir = os.path.join(save_dir, 'faceted_plots')
    os.makedirs(facet_dir, exist_ok=True)

    # Generate faceted plots for key metrics
    for metric_name in ['fertility', 'compression_rate', 'vocabulary_utilization', 'bigram_entropy']:
        if metric_name in results:
            _plot_faceted_metric(results, facet_dir, tokenizer_names, metric_name, show_global_lines)


def _plot_per_language_combined_subplots(results: Dict[str, Any], save_dir: str,
                                        tokenizer_names: List[str], show_global_lines: bool):
    """Create combined per-language subplots with tied y-axes."""
    # Collect metrics that have per-language data
    metrics_info = [
        ('fertility', get_metric_display_name('fertility')),
        ('compression_rate', get_metric_display_name('compression_rate')),
        ('vocabulary_utilization', get_metric_display_name('vocabulary_utilization')),
        ('tokenizer_fairness_gini', get_metric_display_name('tokenizer_fairness_gini')),
        ('bigram_entropy', get_metric_display_name('bigram_entropy')),
    ]
    
    metrics_data = {}
    for metric_key, display_name in metrics_info:
        if metric_key not in results:
            continue
            
        lang_data = {}
        for tok_name in tokenizer_names:
            if tok_name in results[metric_key].get('per_tokenizer', {}):
                tok_data = results[metric_key]['per_tokenizer'][tok_name]
                if 'per_language' in tok_data:
                    for lang, lang_stats in tok_data['per_language'].items():
                        if lang not in lang_data:
                            lang_data[lang] = {}
                        
                        # Handle different data structures based on your changes
                        if metric_key == 'vocabulary_utilization':
                            util = (lang_stats.get('utilization')
                                    if isinstance(lang_stats, dict) else None)
                            value = None if util is None else util * 100
                        elif metric_key == 'compression_rate':
                            value = _compression_rate_per_language(lang_stats)
                        elif metric_key == 'tokenizer_fairness_gini':
                            value = (lang_stats if isinstance(lang_stats, (int, float))
                                     else lang_stats.get('mean'))
                        elif metric_key == 'bigram_entropy':
                            value = (lang_stats.get('bigram_entropy')
                                     if isinstance(lang_stats, dict) else lang_stats)
                        else:
                            value = (lang_stats.get('mean')
                                     if isinstance(lang_stats, dict) else lang_stats)
                        
                        lang_data[lang][tok_name] = value
        
        if lang_data:
            # Get labels using centralized functions
            metadata = results[metric_key].get('metadata', {})
            ylabel = get_ylabel(metric_key, metadata)
            metrics_data[metric_key] = (lang_data, ylabel, display_name)
    
    if not metrics_data:
        return
    
    # Create subplot layout
    n_metrics = len(metrics_data)
    if n_metrics == 1:
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        axes = [axes]
    elif n_metrics <= 2:
        fig, axes = plt.subplots(1, n_metrics, figsize=(12 * n_metrics, 6), sharey=True)
        axes = axes if hasattr(axes, '__iter__') else [axes]
    else:
        rows = (n_metrics + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(24, 6 * rows), sharey=True)
        axes = axes.flatten() if rows > 1 else [axes[0], axes[1]]
    
    # Get all languages across all metrics for consistent x-axis
    all_languages = set()
    for lang_data, _, _ in metrics_data.values():
        all_languages.update(lang_data.keys())
    languages = sorted(list(all_languages))
    
    colors = get_colors(len(tokenizer_names))
    
    for i, (metric_key, (lang_data, ylabel, display_name)) in enumerate(metrics_data.items()):
        ax = axes[i]
        
        if not lang_data:
            ax.set_visible(False)
            continue
        
        # Create grouped bar plot
        x_pos = np.arange(len(languages))
        width = 0.8 / len(tokenizer_names)
        
        for j, tok_name in enumerate(tokenizer_names):
            values = [_plottable(lang_data.get(lang, {}).get(tok_name))
                      for lang in languages]
            ax.bar(x_pos + j * width, values, width, label=tok_name, color=colors[j], alpha=0.8)

            # Add global reference line if requested
            measured = _finite(values)
            if show_global_lines and measured:
                global_mean = np.mean(measured)
                ax.axhline(y=global_mean, color=colors[j], linestyle='--', alpha=0.6,
                          linewidth=1.5)
        
        title = get_plot_title('per_language', metric_key)
        ax.set_title(title)
        ax.set_xticks(x_pos + width * (len(tokenizer_names) - 1) / 2)
        ax.set_xticklabels([format_language_labels(lang) for lang in languages], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Only add ylabel to leftmost plots
        if i % 2 == 0 or n_metrics == 1:
            ax.set_ylabel(ylabel)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    # Add shared legend
    if metrics_data:
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_plot(fig, os.path.join(save_dir, 'per_language_combined_subplots.svg'))


def _compression_rate_per_language(entry) -> Optional[float]:
    """Read one language's compression rate out of its per-language entry.

    The entry is ``{'compression_rate': rate, 'count': units, ...}``. It was a
    bare float before the count was added, and a results file written by an
    earlier version still holds the float, so both are read here.
    """
    if isinstance(entry, dict):
        return entry.get('compression_rate')
    return entry


def _plot_per_language_grouped_bars(lang_data: Dict[str, Dict[str, float]],
                                  save_path: str, tokenizer_names: List[str],
                                  title: str, ylabel: str, show_global_lines: bool):
    """Plot grouped bars for per-language metrics with languages on x-axis."""
    languages = sorted(list(lang_data.keys()))
    if not languages:
        return
    
    fig, ax = plt.subplots(figsize=(max(10, len(languages) * 1.5), 6))
    
    # Create grouped bar data
    x_pos = np.arange(len(languages))
    width = 0.8 / len(tokenizer_names)
    
    colors = get_colors(len(tokenizer_names))

    for i, tok_name in enumerate(tokenizer_names):
        values = [_plottable(lang_data[lang].get(tok_name)) for lang in languages]
        ax.bar(x_pos + i * width, values, width, label=tok_name, color=colors[i], alpha=0.8)

        # Add global reference line if requested
        if show_global_lines and values:
            measured = _finite(values)
            global_mean = np.mean(measured) if measured else float('nan')
            ax.axhline(y=global_mean, color=colors[i], linestyle='--', alpha=0.6,
                      linewidth=1.5)
    
    ax.set_xlabel('Language')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_pos + width * (len(tokenizer_names) - 1) / 2)
    ax.set_xticklabels([format_language_labels(lang) for lang in languages], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, save_path)


def _plot_per_language_metric(results: Dict[str, Any], save_dir: str,
                              tokenizer_names: List[str], metric_key: str,
                              value_extractor, filename: str, show_global_lines: bool):
    """Plot per-language comparison with grouped bars for any metric.

    Args:
        results: Full results dict.
        save_dir: Directory to save the plot.
        tokenizer_names: Ordered list of tokenizer names.
        metric_key: Top-level key in *results*.
        value_extractor: Callable ``(lang_stats) -> float`` to extract the
            per-language value from the metric's per-language data structure.
        filename: Output filename (e.g. ``'fertility_per_language.svg'``).
        show_global_lines: Whether to draw global average lines.
    """
    if metric_key not in results:
        return

    lang_data = {}
    for tok_name in tokenizer_names:
        if tok_name in results[metric_key].get('per_tokenizer', {}):
            tok_data = results[metric_key]['per_tokenizer'][tok_name]
            if 'per_language' in tok_data:
                for lang, lang_stats in tok_data['per_language'].items():
                    if lang not in lang_data:
                        lang_data[lang] = {}
                    lang_data[lang][tok_name] = value_extractor(lang_stats)

    if lang_data:
        metadata = results[metric_key].get('metadata', {})
        ylabel = get_ylabel(metric_key, metadata)
        title = get_plot_title('per_language', metric_key)
        _plot_per_language_grouped_bars(
            lang_data, os.path.join(save_dir, filename),
            tokenizer_names, title, ylabel, show_global_lines
        )


def _plot_per_language_fertility(results, save_dir, tokenizer_names, show_global_lines):
    """Plot per-language fertility comparison with grouped bars."""
    _plot_per_language_metric(
        results, save_dir, tokenizer_names, 'fertility',
        lambda s: s.get('mean') if isinstance(s, dict) else s,
        'fertility_per_language.svg', show_global_lines,
    )


def _plot_per_language_compression_rate(results, save_dir, tokenizer_names, show_global_lines):
    """Plot per-language compression rate comparison with grouped bars."""
    _plot_per_language_metric(
        results, save_dir, tokenizer_names, 'compression_rate',
        _compression_rate_per_language,
        'compression_rate_per_language.svg', show_global_lines,
    )


def _plot_per_language_vocabulary_utilization(results, save_dir, tokenizer_names, show_global_lines):
    """Plot per-language vocabulary utilization comparison with grouped bars."""
    _plot_per_language_metric(
        results, save_dir, tokenizer_names, 'vocabulary_utilization',
        lambda s: (lambda u: None if u is None else u * 100)(s.get('utilization'))
        if isinstance(s, dict) else s,
        'vocabulary_utilization_per_language.svg', show_global_lines,
    )


def _plot_per_language_gini_coefficient(results, save_dir, tokenizer_names, show_global_lines):
    """Plot per-language Gini coefficient comparison with grouped bars."""
    _plot_per_language_metric(
        results, save_dir, tokenizer_names, 'tokenizer_fairness_gini',
        lambda s: s if isinstance(s, (int, float)) else s.get('mean'),
        'tokenizer_fairness_gini_per_language.svg', show_global_lines,
    )


def _plot_per_language_bigram_entropy(results, save_dir, tokenizer_names, show_global_lines):
    """Plot per-language bigram entropy comparison with grouped bars."""
    _plot_per_language_metric(
        results, save_dir, tokenizer_names, 'bigram_entropy',
        lambda s: s.get('bigram_entropy', 0.0) if isinstance(s, dict) else s,
        'bigram_entropy_per_language.svg', show_global_lines,
    )


def _plot_faceted_metric(results: Dict[str, Any], save_dir: str,
                        tokenizer_names: List[str], metric_name: str, show_global_lines: bool):
    """Generate faceted plot for a specific metric."""
    from .visualization_config import PlotConfig
    
    if metric_name not in results:
        return
        
    n_tokenizers = len(tokenizer_names)
    cols = min(3, n_tokenizers)
    rows = (n_tokenizers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3*rows), sharey=True)
    if n_tokenizers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__iter__') else [axes]
    else:
        axes = axes.flatten()
    
    metric_data = results[metric_name]['per_tokenizer']
    
    # Use a single consistent color for all bars
    single_color = get_colors(1)[0]
    
    for i, tok_name in enumerate(tokenizer_names):
        ax = axes[i]
        
        if tok_name not in metric_data:
            ax.set_visible(False)
            continue
            
        tok_data = metric_data[tok_name]
        
        # Plot per-language data if available
        if 'per_language' in tok_data:
            languages = list(tok_data['per_language'].keys())
            values = []
            
            for lang in languages:
                lang_data = tok_data['per_language'][lang]
                if metric_name == 'vocabulary_utilization':
                    # Per-language entry is {'utilization': float, ...};
                    # render as percentage to match the y-label.
                    util = (lang_data.get('utilization')
                            if isinstance(lang_data, dict) else None)
                    values.append(float('nan') if util is None
                                  else _plottable(util) * 100)
                elif metric_name == 'compression_rate':
                    values.append(_plottable(
                        _compression_rate_per_language(lang_data)))
                elif isinstance(lang_data, dict) and 'mean' in lang_data:
                    values.append(_plottable(lang_data['mean']))
                elif isinstance(lang_data, dict) and 'bigram_entropy' in lang_data:
                    values.append(_plottable(lang_data['bigram_entropy']))
                elif isinstance(lang_data, (int, float)):
                    values.append(_plottable(lang_data))
                else:
                    values.append(float('nan'))
            
            if values:
                # Use single consistent color for all bars
                ax.bar(range(len(languages)), values, color=single_color, alpha=0.8)
                measured = _finite(values)
                if show_global_lines and measured:
                    global_mean = np.mean(measured)
                    ax.axhline(y=global_mean, color='red', linestyle='--', alpha=0.7)
                ax.set_xticks(range(len(languages)))
                # Use smaller font size for faceted plots
                ax.set_xticklabels([format_language_labels(lang) for lang in languages], 
                                  rotation=45, fontsize=PlotConfig.FACETED_XTICK_SIZE)
        
        ax.set_title(tok_name)
    
    # Hide unused subplots
    for i in range(n_tokenizers, len(axes)):
        axes[i].set_visible(False)
    
    # Add single y-axis label using supylabel instead of individual labels
    metadata = results[metric_name].get('metadata', {})
    ylabel = get_ylabel(metric_name, metadata)
    fig.supylabel(ylabel, x=0.02)
    
    plt.tight_layout()
    save_plot(fig, os.path.join(save_dir, f'{metric_name}_faceted.svg'))