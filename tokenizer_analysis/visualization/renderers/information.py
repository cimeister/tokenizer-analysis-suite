"""
Renderer for information-theoretic metrics.
"""

from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import logging

from .base import PlotRenderer
from ..plot_config import PlotConfig, MetricConfig
from ..data_extraction import DataExtractor

logger = logging.getLogger(__name__)


class InformationRenderer(PlotRenderer):
    """Renderer for information-theoretic metrics."""
    
    def render(self, results: Dict[str, Any]) -> None:
        """Render information-theoretic plots."""
        try:
            self._plot_information_theoretic_metrics(results)
            self._plot_unigram_distribution_metrics(results)
            self._plot_renyi_entropy_curves(results)
            logger.info("Information-theoretic plots rendered successfully")
        except Exception as e:
            logger.error(f"Error rendering information-theoretic metrics: {e}")
            raise
    
    def _plot_information_theoretic_metrics(self, results: Dict[str, Any]) -> None:
        """Plot information-theoretic metrics."""
        fig, axes = plt.subplots(2, 2, figsize=PlotConfig.BASIC_FIGURE_SIZE)
        fig.suptitle('Information-Theoretic Metrics', fontsize=PlotConfig.TITLE_FONTSIZE)
        
        # Type-Token Ratio
        self._plot_type_token_ratio(results, axes[0, 0])
        
        # Vocabulary Utilization
        self._plot_vocabulary_utilization(results, axes[0, 1])
        
        # Shannon Entropy
        self._plot_shannon_entropy(results, axes[1, 0])
        
        # Average tokens per line
        self._plot_avg_tokens_per_line(results, axes[1, 1])
        
        plt.tight_layout()
        self._save_plot('information_theoretic_metrics.png')
    
    def _plot_type_token_ratio(self, results: Dict[str, Any], ax: plt.Axes) -> None:
        """Plot Type-Token Ratio."""
        if 'type_token_ratio' not in results:
            ax.set_visible(False)
            return
        
        data = DataExtractor.extract_metric_data(results, 'type_token_ratio', self.tokenizer_names)
        if not self._has_valid_data(data):
            ax.set_visible(False)
            return
        
        # Extract TTR values
        ttr_values = []
        labels = []
        
        for name in self.tokenizer_names:
            if name in results['type_token_ratio']['per_tokenizer']:
                ttr_values.append(results['type_token_ratio']['per_tokenizer'][name]['global_ttr'])
                labels.append(name)
        
        if ttr_values:
            bars = ax.bar(labels, ttr_values, alpha=PlotConfig.ALPHA_BARS)
            self._setup_subplot(ax, 'Global Type-Token Ratio', 'Tokenizer', 'TTR')
            ax.set_ylim(PlotConfig.get_dynamic_ylim(ttr_values))
    
    def _plot_vocabulary_utilization(self, results: Dict[str, Any], ax: plt.Axes) -> None:
        """Plot Vocabulary Utilization."""
        if 'vocabulary_utilization' not in results:
            ax.set_visible(False)
            return
        
        data = DataExtractor.extract_metric_data(results, 'vocabulary_utilization', self.tokenizer_names)
        if not self._has_valid_data(data):
            ax.set_visible(False)
            return
        
        # Extract utilization values
        utilizations = []
        labels = []
        
        for name in self.tokenizer_names:
            if name in results['vocabulary_utilization']['per_tokenizer']:
                utilizations.append(results['vocabulary_utilization']['per_tokenizer'][name]['global_utilization'])
                labels.append(name)
        
        if utilizations:
            bars = ax.bar(labels, utilizations, alpha=PlotConfig.ALPHA_BARS)
            self._setup_subplot(ax, 'Global Vocabulary Utilization', 'Tokenizer', 
                              'Proportion of Vocabulary Used')
            ax.set_ylim(PlotConfig.get_dynamic_ylim(utilizations))
    
    def _plot_shannon_entropy(self, results: Dict[str, Any], ax: plt.Axes) -> None:
        """Plot Shannon Entropy."""
        if 'renyi_efficiency' not in results:
            ax.set_visible(False)
            return
        
        # Extract Shannon entropy (Rényi entropy with α=1.0)
        shannon_entropies = []
        labels = []
        
        for name in self.tokenizer_names:
            if (name in results['renyi_efficiency']['per_tokenizer'] and
                'renyi_1.0' in results['renyi_efficiency']['per_tokenizer'][name]):
                shannon_entropies.append(
                    results['renyi_efficiency']['per_tokenizer'][name]['renyi_1.0']['overall']
                )
                labels.append(name)
        
        if shannon_entropies:
            bars = ax.bar(labels, shannon_entropies, alpha=PlotConfig.ALPHA_BARS)
            self._setup_subplot(ax, 'Global Shannon Entropy', 'Tokenizer', 'Entropy (bits)')
            ax.set_ylim(PlotConfig.get_dynamic_ylim(shannon_entropies))
    
    def _plot_avg_tokens_per_line(self, results: Dict[str, Any], ax: plt.Axes) -> None:
        """Plot Average Tokens per Line."""
        if 'avg_tokens_per_line' not in results:
            ax.set_visible(False)
            return
        
        data = DataExtractor.extract_metric_data(results, 'avg_tokens_per_line', self.tokenizer_names)
        if not self._has_valid_data(data):
            ax.set_visible(False)
            return
        
        # Extract average tokens per line
        avg_tokens = []
        labels = []
        
        for name in self.tokenizer_names:
            if name in results['avg_tokens_per_line']['per_tokenizer']:
                avg_tokens.append(results['avg_tokens_per_line']['per_tokenizer'][name]['global_avg'])
                labels.append(name)
        
        if avg_tokens:
            bars = ax.bar(labels, avg_tokens, alpha=PlotConfig.ALPHA_BARS)
            self._setup_subplot(ax, 'Global Average Tokens per Line', 'Tokenizer', '# Tokens')
            ax.set_ylim(PlotConfig.get_dynamic_ylim(avg_tokens))
    
    def _plot_unigram_distribution_metrics(self, results: Dict[str, Any]) -> None:
        """Plot unigram distribution metrics."""
        if 'unigram_distribution_metrics' not in results:
            return
        
        unigram_results = results['unigram_distribution_metrics']
        
        # Dynamic figure width based on number of languages
        fig_width = 15
        if 'per_language' in unigram_results and 'unigram_entropy' in unigram_results['per_language']:
            num_languages = len(unigram_results['per_language']['unigram_entropy'])
            fig_width = max(15, min(25, 8 + num_languages * 0.8))
        
        fig, axes = plt.subplots(2, 2, figsize=(fig_width, 12))
        fig.suptitle('Unigram Distribution Metrics', fontsize=PlotConfig.TITLE_FONTSIZE)
        
        # Global metrics
        self._plot_global_unigram_metrics(unigram_results, axes[0, 0], axes[0, 1])
        
        # Per-language metrics
        self._plot_per_language_unigram_metrics(unigram_results, axes[1, 0], axes[1, 1])
        
        plt.tight_layout()
        self._save_plot('unigram_distribution_metrics.png')
    
    def _plot_global_unigram_metrics(self, unigram_results: Dict[str, Any], 
                                   ax_entropy: plt.Axes, ax_rank: plt.Axes) -> None:
        """Plot global unigram metrics."""
        if 'per_tokenizer' not in unigram_results:
            ax_entropy.set_visible(False)
            ax_rank.set_visible(False)
            return
        
        entropies = []
        avg_ranks = []
        labels = []
        
        for name in self.tokenizer_names:
            if name in unigram_results['per_tokenizer']:
                stats = unigram_results['per_tokenizer'][name]
                entropies.append(stats.get('global_unigram_entropy', 0))
                avg_ranks.append(stats.get('global_avg_token_rank', 0))
                labels.append(name)
        
        if entropies:
            bars = ax_entropy.bar(labels, entropies, alpha=PlotConfig.ALPHA_BARS)
            self._setup_subplot(ax_entropy, 'Global Unigram Entropy', 'Tokenizer', 'Entropy (bits)')
            ax_entropy.set_ylim(PlotConfig.get_dynamic_ylim(entropies))
        
        if avg_ranks:
            bars = ax_rank.bar(labels, avg_ranks, alpha=PlotConfig.ALPHA_BARS)
            self._setup_subplot(ax_rank, 'Global Average Token Rank', 'Tokenizer', 'Average Rank')
            ax_rank.set_ylim(PlotConfig.get_dynamic_ylim(avg_ranks))
    
    def _plot_per_language_unigram_metrics(self, unigram_results: Dict[str, Any],
                                         ax_entropy: plt.Axes, ax_rank: plt.Axes) -> None:
        """Plot per-language unigram metrics."""
        if 'per_language' not in unigram_results:
            ax_entropy.set_visible(False)
            ax_rank.set_visible(False)
            return
        
        # Plot per-language unigram entropy
        if 'unigram_entropy' in unigram_results['per_language']:
            self._plot_per_language_grouped_bars(
                unigram_results['per_language']['unigram_entropy'],
                ax_entropy, 'Unigram Entropy by Language', 'Entropy (bits)'
            )
        
        # Plot per-language average token rank
        if 'avg_token_rank' in unigram_results['per_language']:
            self._plot_per_language_grouped_bars(
                unigram_results['per_language']['avg_token_rank'],
                ax_rank, 'Average Token Rank by Language', 'Average Rank'
            )
    
    def _plot_per_language_grouped_bars(self, lang_data: Dict[str, Dict[str, float]],
                                      ax: plt.Axes, title: str, ylabel: str) -> None:
        """Plot grouped bars for per-language metrics."""
        languages = sorted(list(lang_data.keys()))
        if not languages:
            ax.set_visible(False)
            return
        
        # Create grouped bar data
        plot_data = {}
        for tok_name in self.tokenizer_names:
            plot_data[tok_name] = [lang_data[lang].get(tok_name, 0) for lang in languages]
        
        self._create_grouped_bar_plot(ax, plot_data, languages, title, ylabel)
    
    def _plot_renyi_entropy_curves(self, results: Dict[str, Any]) -> None:
        """Plot Rényi entropy curves for different alpha values."""
        if 'renyi_efficiency' not in results:
            return
        
        renyi_data = DataExtractor.extract_renyi_data(results, self.tokenizer_names)
        if not renyi_data.get('alphas'):
            return
        
        fig, axes = plt.subplots(1, 2, figsize=PlotConfig.WIDE_FIGURE_SIZE)
        fig.suptitle('Rényi Entropy Analysis', fontsize=PlotConfig.TITLE_FONTSIZE)
        
        # Plot 1: Rényi entropy curves
        self._plot_renyi_curves(renyi_data, axes[0])
        
        # Plot 2: Per-language entropy heatmap (for α=2.0)
        self._plot_renyi_heatmap(renyi_data, axes[1])
        
        plt.tight_layout()
        self._save_plot('renyi_entropy_analysis.png')
    
    def _plot_renyi_curves(self, renyi_data: Dict[str, Any], ax: plt.Axes) -> None:
        """Plot Rényi entropy curves."""
        alphas = renyi_data['alphas']
        entropy_curves = renyi_data['entropy_curves']
        
        for tok_name in self.tokenizer_names:
            if entropy_curves[tok_name]:
                ax.plot(alphas, entropy_curves[tok_name], marker='o', 
                       label=tok_name, linewidth=2)
        
        self._setup_subplot(ax, 'Rényi Entropy vs Alpha', 'Alpha (α)', 'Rényi Entropy (bits)')
        ax.legend(fontsize=PlotConfig.LEGEND_FONTSIZE)
    
    def _plot_renyi_heatmap(self, renyi_data: Dict[str, Any], ax: plt.Axes) -> None:
        """Plot per-language Rényi entropy heatmap."""
        if 'per_language_data' not in renyi_data or not renyi_data['per_language_data']:
            ax.set_visible(False)
            return
        
        per_lang_data = renyi_data['per_language_data']
        languages = per_lang_data['languages']
        data_matrix = np.array(per_lang_data['data_matrix'])
        
        if languages and data_matrix.size > 0:
            self._create_heatmap(
                ax, data_matrix, self.tokenizer_names, languages,
                'Per-Language Rényi Entropy (α=2.0)', 'viridis'
            )
            
            # Add colorbar label
            im = ax.get_images()[0]
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Entropy (bits)', rotation=270, labelpad=20)