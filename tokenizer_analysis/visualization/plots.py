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
        'compression_ratio': 'Compression Rate', 
        'vocabulary_utilization': 'Vocabulary Utilization',
        'tokenizer_fairness_gini': 'Gini Coefficient',
        'morphscore': 'MorphScore'
    }
    return metric_names.get(metric_key, metric_key.replace('_', ' ').title())


def get_ylabel(metric_key: str, metadata: Optional[Dict] = None) -> str:
    """Get y-axis label for a metric."""
    norm_method = 'units'
    if metadata:
        norm_method = metadata.get('normalization_method', 'units')
    
    labels = {
        'fertility': f'Fertility (tokens/{norm_method.rstrip("s")})',
        'compression_ratio': f'Per {norm_method.rstrip("s").title()} Compression Rate',
        'vocabulary_utilization': 'Vocabulary Utilization (%)',
        'tokenizer_fairness_gini': 'Gini Coefficient',
        'morphscore_recall': 'MorphScore Recall',
        'morphscore_precision': 'MorphScore Precision'
    }
    return labels.get(metric_key, metric_key.replace('_', ' ').title())


def get_plot_title(plot_type: str, metric_key: str = None, context: str = None) -> str:
    """Get plot title."""
    metric_display = get_metric_display_name(metric_key) if metric_key else ''
    
    # Special cases for specific plots
    if metric_key == 'lorenz_curves':
        return 'Lorenz Curves - Cross-Language Fairness'
    elif metric_key == 'morphscore_recall':
        return 'MorphScore Recall Comparison'  
    elif metric_key == 'morphscore_precision':
        return 'MorphScore Precision Comparison'
    elif metric_key == 'tokenizer_fairness_gini' and plot_type == 'individual':
        return 'Cross-Language Fairness (Gini Coefficient)'
    
    titles = {
        'individual': f'{metric_display} Comparison',
        'per_language': f'{metric_display} by Language', 
        'faceted': f'Faceted Analysis: {metric_display}',
        'grouped': f'{metric_display} by {context}' if context else f'{metric_display} Analysis'
    }
    return titles.get(plot_type, f'{metric_display} Analysis')

def format_language_labels(lang_code):
    return lang_code.split('_')[0]

def save_plot(fig, filepath: str):
    """Save plot to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_fertility(results: Dict[str, Any], save_path: str, tokenizer_names: List[str], show_global_lines: bool = True):
    """Plot fertility metric comparison."""
    if 'fertility' not in results:
        return
    
    fig, ax = plt.subplots()
    fertility_data = results['fertility']['per_tokenizer']
    
    values = []
    labels = []
    
    for tok_name in tokenizer_names:
        if tok_name in fertility_data:
            mean_val = fertility_data[tok_name]['global']['mean']
            std_val = fertility_data[tok_name]['global']['std']
            values.append((mean_val, std_val))
            labels.append(tok_name)
    
    if values:
        means, stds = zip(*values)
        colors = get_colors(len(values))
        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        
        # Add global reference line if requested
        if show_global_lines:
            global_mean = np.mean(means)
            ax.axhline(y=global_mean, color='red', linestyle='--', alpha=0.7, label=f'Global Average: {global_mean:.2f}')
            ax.legend()
        
        # Get labels using centralized functions
        metadata = results['fertility'].get('metadata', {})
        ylabel = get_ylabel('fertility', metadata)
        title = get_plot_title('individual', 'fertility')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.xticks(rotation=45)
        
    save_plot(fig, save_path)


def plot_vocabulary_utilization(results: Dict[str, Any], save_path: str, tokenizer_names: List[str], show_global_lines: bool = True):
    """Plot vocabulary utilization comparison."""
    if 'vocabulary_utilization' not in results:
        return
        
    fig, ax = plt.subplots()
    util_data = results['vocabulary_utilization']['per_tokenizer']
    
    values = []
    labels = []
    
    for tok_name in tokenizer_names:
        if tok_name in util_data:
            util = util_data[tok_name]['global_utilization']
            values.append(util * 100)
            labels.append(tok_name)
    
    if values:
        colors = get_colors(len(values))
        bars = ax.bar(labels, values, color=colors, alpha=0.8)
        
        # Add global reference line if requested
        if show_global_lines:
            global_mean = np.mean(values)
            ax.axhline(y=global_mean, color='red', linestyle='--', alpha=0.7, label=f'Global Average: {global_mean:.1f}%')
            ax.legend()
        
        ylabel = get_ylabel('vocabulary_utilization')
        title = get_plot_title('individual', 'vocabulary_utilization')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.xticks(rotation=45)
        
    save_plot(fig, save_path)


def plot_compression_rate(results: Dict[str, Any], save_path: str, tokenizer_names: List[str], show_global_lines: bool = True):
    """Plot compression rate comparison."""
    if 'compression_ratio' not in results:
        return
        
    fig, ax = plt.subplots()
    comp_data = results['compression_ratio']['per_tokenizer']
    
    values = []
    labels = []
    
    for tok_name in tokenizer_names:
        if tok_name in comp_data:
            ratio = comp_data[tok_name]['global']['mean']
            values.append(ratio)
            labels.append(tok_name)
    
    if values:
        colors = get_colors(len(values))
        bars = ax.bar(labels, values, color=colors, alpha=0.8)
        
        # Add global reference line if requested
        if show_global_lines:
            global_mean = np.mean(values)
            ax.axhline(y=global_mean, color='red', linestyle='--', alpha=0.7, label=f'Global Average: {global_mean:.2f}')
            ax.legend()
        
        # Get labels using centralized functions
        metadata = results['compression_ratio'].get('metadata', {})
        ylabel = get_ylabel('compression_ratio', metadata)
        title = get_plot_title('individual', 'compression_ratio')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.xticks(rotation=45)
        
    save_plot(fig, save_path)


def plot_gini_coefficient(results: Dict[str, Any], save_path: str, tokenizer_names: List[str], show_global_lines: bool = True):
    """Plot Gini coefficient comparison for fairness."""
    if 'tokenizer_fairness_gini' not in results:
        return
        
    fig, ax = plt.subplots()
    gini_data = results['tokenizer_fairness_gini']['per_tokenizer']
    
    values = []
    labels = []
    
    for tok_name in tokenizer_names:
        if tok_name in gini_data:
            gini = gini_data[tok_name]['gini_coefficient']
            values.append(gini)
            labels.append(tok_name)
    
    if values:
        colors = get_colors(len(values))
        bars = ax.bar(labels, values, color=colors, alpha=0.8)
        
        # Add global reference line if requested
        if show_global_lines:
            global_mean = np.mean(values)
            ax.axhline(y=global_mean, color='red', linestyle='--', alpha=0.7, label=f'Global Average: {global_mean:.3f}')
            ax.legend()
        
        ylabel = get_ylabel('tokenizer_fairness_gini')
        title = get_plot_title('individual', 'tokenizer_fairness_gini')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        
    save_plot(fig, save_path)


def plot_lorenz_curves(results: Dict[str, Any], save_path: str, tokenizer_names: List[str]):
    """Plot Lorenz curves for fairness analysis."""
    if 'lorenz_curve_data' not in results:
        return
        
    fig, ax = plt.subplots()
    lorenz_data = results['lorenz_curve_data']['per_tokenizer']
    
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
            recall_values.append(summary.get('avg_morphscore_recall', 0))
            precision_values.append(summary.get('avg_morphscore_precision', 0))
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


def plot_grouped_analysis(grouped_results: Dict[str, Dict[str, Any]], save_dir: str, 
                         metric_name: str, group_type: str):
    """Plot grouped analysis results."""
    if group_type not in grouped_results:
        return
        
    fig, ax = plt.subplots(figsize=(12, 8))
    group_data = grouped_results[group_type]
    
    # Extract data for plotting
    groups = list(group_data.keys())
    tokenizer_names = set()
    
    for group_results in group_data.values():
        if metric_name in group_results:
            tokenizer_names.update(group_results[metric_name]['per_tokenizer'].keys())
    
    tokenizer_names = sorted(list(tokenizer_names))
    
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
                if 'global' in tok_data:
                    values.append(tok_data['global']['mean'])
                else:
                    values.append(0)
            else:
                values.append(0)
        
        ax.bar(x_pos + i * width, values, width, label=tok_name, color=colors[i], alpha=0.8)
    
    xlabel = group_type.replace('_', ' ').title()
    ylabel = get_metric_display_name(metric_name) 
    title = get_plot_title('grouped', metric_name, group_type.replace('_', ' ').title())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_pos + width * (len(tokenizer_names) - 1) / 2)
    ax.set_xticklabels(groups, rotation=45)
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
    
    # Information theory
    plot_compression_rate(results, os.path.join(save_dir, 'compression_rate_individual.svg'), tokenizer_names, show_global_lines)
    
    # Fairness
    plot_gini_coefficient(results, os.path.join(save_dir, 'tokenizer_fairness_gini_individual.svg'), tokenizer_names, show_global_lines)
    plot_lorenz_curves(results, os.path.join(save_dir, 'lorenz_curves_individual.svg'), tokenizer_names)
    
    # Morphological
    plot_morphscore(results, os.path.join(save_dir, 'morphscore_individual.svg'), tokenizer_names)
    
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
            for metric in ['fertility', 'vocabulary_utilization', 'compression_ratio', 'morphscore']:
                try:
                    plot_grouped_analysis(grouped_results, grouped_dir, metric, group_type)
                except Exception as e:
                    logger.warning(f"Failed to plot {metric} for group type {group_type}: {e}")


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


def _generate_faceted_plots(results: Dict[str, Any], save_dir: str,
                          tokenizer_names: List[str], show_global_lines: bool):
    """Generate faceted plots with subplots for each tokenizer."""
    facet_dir = os.path.join(save_dir, 'faceted_plots')
    os.makedirs(facet_dir, exist_ok=True)
    
    # Generate faceted plots for key metrics
    for metric_name in ['fertility', 'compression_ratio', 'vocabulary_utilization']:
        if metric_name in results:
            _plot_faceted_metric(results, facet_dir, tokenizer_names, metric_name, show_global_lines)


def _plot_per_language_combined_subplots(results: Dict[str, Any], save_dir: str,
                                        tokenizer_names: List[str], show_global_lines: bool):
    """Create combined per-language subplots with tied y-axes."""
    # Collect metrics that have per-language data
    metrics_info = [
        ('fertility', get_metric_display_name('fertility')),
        ('compression_ratio', get_metric_display_name('compression_ratio')), 
        ('vocabulary_utilization', get_metric_display_name('vocabulary_utilization')),
        ('tokenizer_fairness_gini', get_metric_display_name('tokenizer_fairness_gini'))
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
                            value = lang_stats.get('utilization', 0.0) * 100
                        elif metric_key == 'compression_ratio':
                            # Use your scalar value structure
                            value = lang_stats if isinstance(lang_stats, (int, float)) else lang_stats.get('mean', 0.0)
                        elif metric_key == 'tokenizer_fairness_gini':
                            value = lang_stats if isinstance(lang_stats, (int, float)) else lang_stats.get('mean', 0.0)
                        else:
                            value = lang_stats.get('mean', 0.0) if isinstance(lang_stats, dict) else lang_stats
                        
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
            values = [lang_data.get(lang, {}).get(tok_name, 0) for lang in languages]
            bars = ax.bar(x_pos + j * width, values, width, label=tok_name, color=colors[j], alpha=0.8)
            
            # Add global reference line if requested
            if show_global_lines and values and any(v > 0 for v in values):
                global_mean = np.mean([v for v in values if v > 0])
                ax.axhline(y=global_mean, color='red', linestyle='--', alpha=0.3)
        
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
    
    for i, tok_name in enumerate(tokenizer_names):
        values = [lang_data[lang].get(tok_name, 0) for lang in languages]
        bars = ax.bar(x_pos + i * width, values, width, label=tok_name)
        
        # Add global reference line if requested
        if show_global_lines and values:
            global_mean = np.mean(values)
            ax.axhline(y=global_mean, color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Language')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_pos + width * (len(tokenizer_names) - 1) / 2)
    ax.set_xticklabels([format_language_labels(lang) for lang in languages], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, save_path)


def _plot_per_language_fertility(results: Dict[str, Any], save_dir: str,
                               tokenizer_names: List[str], show_global_lines: bool):
    """Plot per-language fertility comparison with grouped bars."""
    if 'fertility' not in results:
        return
    
    # Extract per-language data
    lang_data = {}
    for tok_name in tokenizer_names:
        if tok_name in results['fertility'].get('per_tokenizer', {}):
            tok_data = results['fertility']['per_tokenizer'][tok_name]
            if 'per_language' in tok_data:
                for lang, lang_stats in tok_data['per_language'].items():
                    if lang not in lang_data:
                        lang_data[lang] = {}
                    fertility_value = lang_stats.get('mean', 0.0)
                    lang_data[lang][tok_name] = fertility_value
    
    if lang_data:
        # Get labels using centralized functions
        metadata = results['fertility'].get('metadata', {})
        ylabel = get_ylabel('fertility', metadata)
        title = get_plot_title('per_language', 'fertility')
        
        _plot_per_language_grouped_bars(
            lang_data, os.path.join(save_dir, 'fertility_per_language.svg'),
            tokenizer_names, title, ylabel, show_global_lines
        )


def _plot_per_language_compression_rate(results: Dict[str, Any], save_dir: str,
                                       tokenizer_names: List[str], show_global_lines: bool):
    """Plot per-language compression ratio comparison with grouped bars."""
    if 'compression_ratio' not in results:
        return
    
    # Extract per-language data
    lang_data = {}
    for tok_name in tokenizer_names:
        if tok_name in results['compression_ratio'].get('per_tokenizer', {}):
            tok_data = results['compression_ratio']['per_tokenizer'][tok_name]
            if 'per_language' in tok_data:
                for lang, ratio_value in tok_data['per_language'].items():
                    if lang not in lang_data:
                        lang_data[lang] = {}
                    lang_data[lang][tok_name] = ratio_value
    
    if lang_data:
        # Get labels using centralized functions
        metadata = results['compression_ratio'].get('metadata', {})
        ylabel = get_ylabel('compression_ratio', metadata)
        title = get_plot_title('per_language', 'compression_ratio')
        
        _plot_per_language_grouped_bars(
            lang_data, os.path.join(save_dir, 'compression_rate_per_language.svg'),
            tokenizer_names, title, ylabel, show_global_lines
        )


def _plot_per_language_vocabulary_utilization(results: Dict[str, Any], save_dir: str,
                                            tokenizer_names: List[str], show_global_lines: bool):
    """Plot per-language vocabulary utilization comparison with grouped bars."""
    if 'vocabulary_utilization' not in results:
        return
    
    # Extract per-language data
    lang_data = {}
    for tok_name in tokenizer_names:
        if tok_name in results['vocabulary_utilization'].get('per_tokenizer', {}):
            tok_data = results['vocabulary_utilization']['per_tokenizer'][tok_name]
            if 'per_language' in tok_data:
                for lang, lang_stats in tok_data['per_language'].items():
                    if lang not in lang_data:
                        lang_data[lang] = {}
                    util_value = lang_stats.get('utilization', 0.0) * 100
                    lang_data[lang][tok_name] = util_value
    
    if lang_data:
        ylabel = get_ylabel('vocabulary_utilization')
        title = get_plot_title('per_language', 'vocabulary_utilization')
        _plot_per_language_grouped_bars(
            lang_data, os.path.join(save_dir, 'vocabulary_utilization_per_language.svg'),
            tokenizer_names, title, ylabel, show_global_lines
        )


def _plot_per_language_gini_coefficient(results: Dict[str, Any], save_dir: str,
                                      tokenizer_names: List[str], show_global_lines: bool):
    """Plot per-language Gini coefficient comparison with grouped bars."""
    if 'tokenizer_fairness_gini' not in results:
        return
    
    # Extract per-language data
    lang_data = {}
    for tok_name in tokenizer_names:
        if tok_name in results['tokenizer_fairness_gini'].get('per_tokenizer', {}):
            tok_data = results['tokenizer_fairness_gini']['per_tokenizer'][tok_name]
            if 'per_language' in tok_data:
                for lang, gini_value in tok_data['per_language'].items():
                    if lang not in lang_data:
                        lang_data[lang] = {}
                    lang_data[lang][tok_name] = gini_value
    
    if lang_data:
        ylabel = get_ylabel('tokenizer_fairness_gini')
        title = get_plot_title('per_language', 'tokenizer_fairness_gini')
        _plot_per_language_grouped_bars(
            lang_data, os.path.join(save_dir, 'tokenizer_fairness_gini_per_language.svg'),
            tokenizer_names, title, ylabel, show_global_lines
        )


def _plot_faceted_metric(results: Dict[str, Any], save_dir: str,
                        tokenizer_names: List[str], metric_name: str, show_global_lines: bool):
    """Generate faceted plot for a specific metric."""
    if metric_name not in results:
        return
        
    n_tokenizers = len(tokenizer_names)
    cols = min(3, n_tokenizers)
    rows = (n_tokenizers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), sharey=True)
    if n_tokenizers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__iter__') else [axes]
    else:
        axes = axes.flatten()
    
    # Get title using centralized function
    title = get_plot_title('faceted', metric_name)
    fig.suptitle(title, fontsize=16)
    
    metric_data = results[metric_name]['per_tokenizer']
    
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
                if isinstance(lang_data, dict) and 'mean' in lang_data:
                    values.append(lang_data['mean'])
                elif isinstance(lang_data, (int, float)):
                    values.append(lang_data)
                else:
                    values.append(0)
            
            if values:
                colors = get_colors(len(languages))
                bars = ax.bar(range(len(languages)), values, color=colors, alpha=0.8)
                if show_global_lines:
                    global_mean = np.mean(values)
                    ax.axhline(y=global_mean, color='red', linestyle='--', alpha=0.7)
                ax.set_xticks(range(len(languages)))
                ax.set_xticklabels([format_language_labels(lang) for lang in languages], rotation=45)
        
        ax.set_title(tok_name)
        # Only add ylabel to leftmost plots
        if i % cols == 0:
            metadata = results[metric_name].get('metadata', {})
            ylabel = get_ylabel(metric_name, metadata)
            ax.set_ylabel(ylabel)
    
    # Hide unused subplots
    for i in range(n_tokenizers, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    save_plot(fig, os.path.join(save_dir, f'{metric_name}_faceted.svg'))