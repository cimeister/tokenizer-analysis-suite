"""
Renderer for fairness metrics (Gini coefficients, Lorenz curves).
"""

from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import logging

from .base import PlotRenderer
from ..plot_config import PlotConfig, MetricConfig
from ..data_extraction import DataExtractor

logger = logging.getLogger(__name__)


class FairnessRenderer(PlotRenderer):
    """Renderer for fairness metrics."""
    
    def render(self, results: Dict[str, Any]) -> None:
        """Render fairness plots."""
        try:
            self._plot_tokenizer_fairness_gini(results)
            self._plot_lorenz_curves(results)
            self._plot_morphological_metrics(results)
            logger.info("Fairness plots rendered successfully")
        except Exception as e:
            logger.error(f"Error rendering fairness metrics: {e}")
            raise
    
    def _plot_tokenizer_fairness_gini(self, results: Dict[str, Any]) -> None:
        """Plot Tokenizer Fairness Gini coefficient and related metrics."""
        if 'tokenizer_fairness_gini' not in results:
            return
        
        gini_data = DataExtractor.extract_gini_data(results, self.tokenizer_names)
        if not gini_data.get('gini_coefficients'):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Tokenizer Fairness Analysis', fontsize=PlotConfig.TITLE_FONTSIZE)
        
        # 1. Gini Coefficients Comparison
        self._plot_gini_coefficients(gini_data, axes[0, 0])
        
        # 2. Cost Ratio (Max/Min cost)
        self._plot_cost_ratios(gini_data, axes[0, 1])
        
        # 3. Token Costs by Language
        self._plot_language_costs(gini_data, axes[1, 0])
        
        # 4. Summary statistics
        self._plot_fairness_summary(gini_data, axes[1, 1])
        
        plt.tight_layout()
        self._save_plot('tokenizer_fairness_gini.png')
    
    def _plot_gini_coefficients(self, gini_data: Dict[str, Any], ax: plt.Axes) -> None:
        """Plot Gini coefficients comparison."""
        gini_values = list(gini_data['gini_coefficients'].values())
        labels = list(gini_data['gini_coefficients'].keys())
        
        if not gini_values:
            ax.set_visible(False)
            return
        
        bars = ax.bar(labels, gini_values, color='lightcoral', alpha=PlotConfig.ALPHA_BARS)
        self._setup_subplot(ax, 'Tokenizer Fairness Gini Coefficient', 'Tokenizer', 
                          'TFG (lower = more fair)')
        
        # Add value labels on bars
        self._add_value_labels(ax, bars, gini_values, '{:.4f}')
    
    def _plot_cost_ratios(self, gini_data: Dict[str, Any], ax: plt.Axes) -> None:
        """Plot cost ratios comparison."""
        # Filter out infinite values
        cost_ratios = []
        labels = []
        
        for name, ratio in gini_data['cost_ratios'].items():
            if ratio != float('inf'):
                cost_ratios.append(ratio)
                labels.append(name)
        
        if not cost_ratios:
            ax.set_visible(False)
            return
        
        bars = ax.bar(labels, cost_ratios, color='skyblue', alpha=PlotConfig.ALPHA_BARS)
        
        # Use dynamic label based on normalization method
        cost_unit = gini_data['metadata'].get('cost_unit', 'tokens per unit')
        ylabel = f'Ratio ({cost_unit}) - lower = more equitable'
        
        self._setup_subplot(ax, 'Cost Ratio (Max/Min)', 'Tokenizer', ylabel)
        
        # Add value labels on bars
        self._add_value_labels(ax, bars, cost_ratios, '{:.2f}')
    
    def _plot_language_costs(self, gini_data: Dict[str, Any], ax: plt.Axes) -> None:
        """Plot token costs by language for all tokenizers."""
        if not gini_data.get('language_costs'):
            ax.set_visible(False)
            return
        
        # Get all languages and sort by average cost across tokenizers
        all_languages = set()
        for costs in gini_data['language_costs'].values():
            all_languages.update(costs.keys())
        
        if not all_languages:
            ax.set_visible(False)
            return
        
        # Calculate average costs for sorting
        lang_avg_costs = {}
        for lang in all_languages:
            costs = [gini_data['language_costs'][tok][lang] 
                    for tok in gini_data['language_costs'].keys() 
                    if lang in gini_data['language_costs'][tok]]
            if costs:
                lang_avg_costs[lang] = np.mean(costs)
        
        # Sort languages by average cost (most efficient first)
        sorted_languages = sorted(lang_avg_costs.keys(), key=lambda x: lang_avg_costs[x])
        
        # Create grouped bar data
        plot_data = {}
        for tok_name in self.tokenizer_names:
            if tok_name in gini_data['language_costs']:
                plot_data[tok_name] = [
                    gini_data['language_costs'][tok_name].get(lang, 0) 
                    for lang in sorted_languages
                ]
        
        if plot_data:
            # Get cost unit from metadata
            cost_unit = gini_data['metadata'].get('cost_unit', 'tokens per unit')
            
            self._create_grouped_bar_plot(
                ax, plot_data, sorted_languages,
                'Token Costs by Language (All Tokenizers)',
                f'Cost ({cost_unit.title()})'
            )
            
            # Set xlabel for sorting info
            ax.set_xlabel('Languages (sorted by efficiency)')
    
    def _plot_fairness_summary(self, gini_data: Dict[str, Any], ax: plt.Axes) -> None:
        """Plot summary statistics as text."""
        ax.axis('off')  # Turn off axis for text summary
        
        summary_text = "Summary Statistics:\n\n"
        
        for tok_name in self.tokenizer_names:
            if tok_name in gini_data['summary_stats']:
                stats = gini_data['summary_stats'][tok_name]
                summary_text += f"{tok_name}:\n"
                summary_text += f"  TFG: {gini_data['gini_coefficients'][tok_name]:.4f}\n"
                summary_text += f"  Mean cost: {stats['mean_cost']:.4f}\n"
                summary_text += f"  Cost range: {stats['min_cost']:.4f} - {stats['max_cost']:.4f}\n"
                summary_text += f"  Most efficient: {stats['most_efficient'][0]} ({stats['most_efficient'][1]:.4f})\n"
                summary_text += f"  Least efficient: {stats['least_efficient'][0]} ({stats['least_efficient'][1]:.4f})\n\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=PlotConfig.TICK_LABEL_FONTSIZE, verticalalignment='top', 
               fontfamily='monospace')
    
    def _plot_lorenz_curves(self, results: Dict[str, Any]) -> None:
        """Plot Lorenz curves for tokenizer fairness visualization."""
        if 'lorenz_curve_data' not in results:
            return
        
        lorenz_data = DataExtractor.extract_lorenz_data(results, self.tokenizer_names)
        if not lorenz_data:
            return
        
        # Determine subplot layout
        n_tokenizers = len(lorenz_data)
        nrows, ncols = PlotConfig.get_subplot_layout(n_tokenizers)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
        fig.suptitle('Lorenz Curves - Tokenizer Fairness Across Languages', 
                    fontsize=PlotConfig.TITLE_FONTSIZE)
        
        # Handle single subplot case
        if n_tokenizers == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        plot_idx = 0
        for tok_name in self.tokenizer_names:
            if tok_name in lorenz_data and plot_idx < len(axes):
                self._plot_single_lorenz_curve(lorenz_data[tok_name], axes[plot_idx], tok_name)
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        self._save_plot('lorenz_curves.png')
    
    def _plot_single_lorenz_curve(self, data: Dict[str, Any], ax: plt.Axes, tok_name: str) -> None:
        """Plot a single Lorenz curve."""
        x_values = data['x_values']
        y_values = data['y_values']
        equality_line = data['equality_line']
        
        # Plot Lorenz curve
        ax.plot(x_values, y_values, 'b-', linewidth=2, label=f'{tok_name} (actual)')
        
        # Plot perfect equality line
        ax.plot(equality_line, equality_line, 'r--', linewidth=1, label='Perfect equality')
        
        # Fill area between curves to show inequality
        ax.fill_between(x_values, y_values, equality_line, 
                       alpha=PlotConfig.ALPHA_FILL, color='lightblue', label='Inequality area')
        
        self._setup_subplot(ax, f'{tok_name}\n({data["n_languages"]} languages)',
                          'Cumulative Proportion of Languages\n(sorted by efficiency)',
                          'Cumulative Proportion of Token Cost')
        
        ax.legend(fontsize=PlotConfig.LEGEND_FONTSIZE)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add language labels for small number of languages
        if data['n_languages'] <= 10:
            for i, lang in enumerate(data['sorted_languages']):
                x = (i + 1) / data['n_languages']
                y = y_values[i + 1] if i + 1 < len(y_values) else y_values[-1]
                ax.annotate(lang, (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=PlotConfig.ANNOTATION_FONTSIZE, alpha=0.7)
    
    def _plot_morphological_metrics(self, results: Dict[str, Any]) -> None:
        """Plot morphological alignment metrics."""
        if 'morphological_alignment' not in results:
            return
        
        morph_results = results['morphological_alignment']
        
        # Check if we have morphological data
        if ('message' in morph_results or 
            not any(any(morph_results.get('per_tokenizer', {}).get(name, {}).get('boundary_f1', {}).values()) 
                   for name in self.tokenizer_names)):
            logger.info("No morphological data available for plotting")
            return
        
        metrics_to_plot = ['boundary_f1', 'morpheme_preservation', 'over_segmentation']
        available_metrics = [m for m in metrics_to_plot if m in morph_results.get('per_tokenizer', {}).get(self.tokenizer_names[0], {})]
        
        if not available_metrics:
            return
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 6))
        fig.suptitle('Morphological Alignment Metrics', fontsize=PlotConfig.TITLE_FONTSIZE)
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            self._plot_morphological_heatmap(morph_results, metric, axes[idx])
        
        plt.tight_layout()
        self._save_plot('morphological_metrics.png')
    
    def _plot_morphological_heatmap(self, morph_results: Dict[str, Any], 
                                   metric: str, ax: plt.Axes) -> None:
        """Plot morphological metric as heatmap."""
        # Collect all languages for this metric
        languages = set()
        for tok_name in self.tokenizer_names:
            if tok_name in morph_results.get('per_tokenizer', {}):
                languages.update(morph_results['per_tokenizer'][tok_name].get(metric, {}).keys())
        
        languages = sorted(list(languages))
        if not languages:
            ax.set_visible(False)
            return
        
        # Create data matrix
        data_matrix = np.zeros((len(self.tokenizer_names), len(languages)))
        
        for i, tok_name in enumerate(self.tokenizer_names):
            for j, lang in enumerate(languages):
                if (tok_name in morph_results.get('per_tokenizer', {}) and 
                    lang in morph_results['per_tokenizer'][tok_name].get(metric, {}) and
                    morph_results['per_tokenizer'][tok_name][metric][lang].get('count', 0) > 0):
                    data_matrix[i, j] = morph_results['per_tokenizer'][tok_name][metric][lang]['mean']
        
        # Create heatmap
        title_map = {
            'boundary_f1': 'Boundary Detection F1 Score',
            'morpheme_preservation': 'Morpheme Preservation Rate',
            'over_segmentation': 'Over-Segmentation Score'
        }
        title = title_map.get(metric, metric.replace("_", " ").title())
        
        self._create_heatmap(ax, data_matrix, self.tokenizer_names, languages, 
                           title, 'RdYlBu_r', add_colorbar=True, add_annotations=True)