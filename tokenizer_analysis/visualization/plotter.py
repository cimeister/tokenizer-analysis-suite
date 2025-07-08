"""
Visualization utilities for tokenizer analysis results.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TokenizerVisualizer:
    """Handles plotting and visualization of tokenizer analysis results."""
    
    def __init__(self, tokenizer_names: List[str], save_dir: str = "tokenizer_analysis_plots"):
        """
        Initialize visualizer.
        
        Args:
            tokenizer_names: List of tokenizer names
            save_dir: Directory to save plots
        """
        self.tokenizer_names = tokenizer_names
        self.save_dir = save_dir
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")

    def _get_dynamic_ylim(self, data: List[float], buffer_ratio: float = 0.1, lower_bound: Optional[float] = 0.0) -> tuple:
        """
        Calculates dynamic y-axis limits to zoom in on the data, especially when
        values are close together and far from zero.
        
        Args:
            data: List of data points for the y-axis.
            buffer_ratio: The percentage of the data range to add as padding.
            
        Returns:
            A tuple (min_limit, max_limit) for the y-axis.
        """
        if not data or len(data) == 0:
            return (0, 1)  # Default fallback

        min_val = min(data)
        max_val = max(data)

        if min_val == max_val:
            # If all values are the same, create a small, sensible range around it
            padding = abs(min_val * buffer_ratio) if min_val != 0 else 0.1
            return (min_val - padding, max_val + padding)

        data_range = max_val - min_val
        padding = data_range * buffer_ratio

        # Add padding to the min and max to create the limits
        lower_bound = min_val - padding if lower_bound is None else max(lower_bound, min_val - padding)
        upper_bound = max_val + padding
        
        return (lower_bound, upper_bound)
    
    def _extract_per_language_value(self, metric_name: str, value: Any) -> float:
        """
        Extract the appropriate numeric value for per-language plotting based on known metric structures.
        
        Args:
            metric_name: Name of the metric (e.g., 'fertility', 'compression_ratio', etc.)
            value: The value from the per_language data structure
            
        Returns:
            float: The numeric value to plot
        """
        # Handle scalar values (like compression_ratio per_language values)
        if isinstance(value, (int, float)):
            return value
        
        # Handle dictionary values based on known metric structures
        if isinstance(value, dict):
            if metric_name == 'fertility':
                # Fertility per_language structure: {'mean': float, 'std': float, 'median': float}
                return value.get('mean', 0.0)
            
            elif metric_name == 'vocabulary_utilization':
                # Vocabulary utilization per_language structure: {'utilization': float, 'used_tokens': int, 'vocab_size': int}
                return value.get('utilization', 0.0)
            
            elif metric_name == 'type_token_ratio':
                # TTR per_language structure: {'ttr': float, 'types': int, 'tokens': int}
                return value.get('ttr', 0.0)
            
            elif metric_name == 'avg_tokens_per_line':
                # Avg tokens per line per_language structure: {'avg_tokens_per_line': float, 'std_tokens_per_line': float, 'total_lines': int}
                return value.get('avg_tokens_per_line', 0.0)
            
            elif metric_name in ['unigram_entropy', 'avg_token_rank']:
                # Unigram distribution metrics per_language structure: {'unigram_entropy': float, 'avg_token_rank': float, 'total_tokens': int, 'unique_tokens': int}
                return value.get(metric_name, 0.0)
            
            elif 'mean' in value:
                # Fallback for any structure with 'mean' key
                return value['mean']
            
            else:
                raise ValueError(f"Unknown per_language data structure for metric '{metric_name}': {value}")
        
        # Fallback for unexpected types
        raise ValueError(f"Unexpected value type for metric '{metric_name}': {type(value)} - {value}")
    
    def _get_per_language_labels(self, metric_name: str, results: Dict[str, Any]) -> tuple:
        """
        Get appropriate ylabel and title for per-language plots based on metric and normalization.
        
        Args:
            metric_name: Name of the metric
            results: Full results dictionary containing metadata
            
        Returns:
            tuple: (ylabel, title) for the plot
        """
        # Default labels
        ylabel = metric_name.replace('_', ' ').title()
        title = f'{metric_name.replace("_", " ").title()} by Language'
        
        # Handle metrics with normalization metadata
        if metric_name == 'fertility' and 'metadata' in results.get(metric_name, {}):
            metadata = results[metric_name]['metadata']
            norm_method = metadata.get('normalization_method', 'tokens')
            ylabel = "# Tokens per " + norm_method.title() 
            title = f"Fertility ({norm_method.title()}) by Language"
        
        elif metric_name == 'compression_ratio' and 'metadata' in results.get(metric_name, {}):
            metadata = results[metric_name]['metadata']
            ylabel = "# " + metadata.get('unit', 'units').title() + "/ # Tokens"
            norm_method = metadata.get('normalization_method', 'units')
            title = f"Compression Ratio ({norm_method.title()}) by Language"
        
        # Handle specific metric types
        elif metric_name == 'vocabulary_utilization':
            ylabel = 'Vocabulary Utilization'
            title = 'Vocabulary Utilization by Language'
        
        elif metric_name == 'type_token_ratio':
            ylabel = 'Type-Token Ratio'
            title = 'Type-Token Ratio by Language'
        
        elif metric_name == 'avg_tokens_per_line':
            ylabel = 'Tokens per Line'
            title = 'Average Tokens per Line by Language'
        
        # Handle unigram distribution metrics
        elif metric_name in ['unigram_entropy', 'avg_token_rank']:
            if metric_name == 'unigram_entropy':
                ylabel = 'Unigram Entropy (bits)'
                title = 'Unigram Distribution Entropy by Language'
            else:
                ylabel = 'Average Token Rank'
                title = 'Average Token Rank by Language'
        
        # Handle TFG language costs
        elif metric_name == 'language_costs':
            # Get cost unit from TFG metadata if available
            if 'tokenizer_fairness_gini' in results:
                metadata = results['tokenizer_fairness_gini'].get('metadata', {})
                cost_unit = metadata.get('cost_unit', 'tokens per unit')
                ylabel = f'Cost ({cost_unit.title()})'
                title = 'Token Costs by Language'
            else:
                ylabel = 'Token Cost'
                title = 'Token Costs by Language'
        
        return ylabel, title
    
    def plot_basic_metrics_comparison(self, results: Dict[str, Any]) -> None:
        """Plot basic tokenization metrics comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Basic Tokenization Metrics Comparison', fontsize=16)

        
        # Token length comparison
        if 'token_length' in results:
            primary_lengths = []
            labels = []
            unit = 'bytes'
            if 'metadata' in results['token_length']:
                unit = results['token_length']['metadata'].get('primary_unit', 'bytes')
            
            for name, stats in results['token_length']['per_tokenizer'].items():
                if 'primary_length' in stats:
                    primary_lengths.append(stats['primary_length']['mean'])
                elif 'byte_length' in stats:
                    primary_lengths.append(stats['byte_length']['mean'])
                elif 'character_length' in stats:
                    primary_lengths.append(stats['character_length']['mean'])
                    unit = 'characters'
                else:
                    primary_lengths.append(stats.get('mean', 0))
                    unit = 'characters'
                labels.append(name)
            
            axes[0, 0].bar(labels, primary_lengths)
            axes[0, 0].set_title('Global Average Token Length')
            axes[0, 0].set_ylabel(f'{unit.capitalize()} per Token')
            axes[0, 0].tick_params(axis='x', rotation=45)
            if primary_lengths:
                axes[0, 0].set_ylim(self._get_dynamic_ylim(primary_lengths))
        
        # Compression ratios
        if 'compression_ratio' in results:
            compression_ratios = [stats['global'] for stats in results['compression_ratio']['per_tokenizer'].values()]
            labels = list(results['compression_ratio']['per_tokenizer'].keys())
            axes[1, 0].bar(labels, compression_ratios)
            axes[1, 0].set_title('Global Compression Ratio')
            
            # Use dynamic ylabel based on normalization method
            metadata = results['compression_ratio'].get('metadata', {})
            ylabel = metadata.get('unit', 'units')
            axes[1, 0].set_ylabel('# ' + ylabel.title() + ' / # Token (higher = more compressed)')
            
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
            if compression_ratios:
                axes[1, 0].set_ylim(self._get_dynamic_ylim(compression_ratios))

        # Fertility comparison (configurable normalization)
        if 'fertility' in results:
            fertilities = [stats['global']['mean'] for stats in results['fertility']['per_tokenizer'].values() if stats['global']]
            labels = [name for name, stats in results['fertility']['per_tokenizer'].items() if stats['global']]
            if fertilities:
                axes[1, 1].bar(labels, fertilities, color='skyblue', alpha=0.7)
                
                # Use metadata to set dynamic titles and labels
                metadata = results['fertility'].get('metadata', {})
                title = f"Global Fertility ({metadata.get('normalization_method', 'tokens')})"
                norm_method = metadata.get('normalization_method', 'tokens')
                ylabel = "# Tokens per " + norm_method.title()
                
                axes[1, 1].set_title(title)
                axes[1, 1].set_ylabel(ylabel.title())
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].set_ylim(self._get_dynamic_ylim(fertilities))

        # Handle legacy fertility results for backward compatibility
        elif 'whitespace_fertility' in results:
            ws_fertilities = [stats['global']['mean'] for stats in results['whitespace_fertility']['per_tokenizer'].values()]
            labels = list(results['whitespace_fertility']['per_tokenizer'].keys())
            axes[1, 1].bar(labels, ws_fertilities, color='skyblue', alpha=0.7)
            axes[1, 1].set_title('Global Whitespace-Delimited Fertility')
            axes[1, 1].set_ylabel('Tokens per Word')
            axes[1, 1].tick_params(axis='x', rotation=45)
            if ws_fertilities:
                axes[1, 1].set_ylim(self._get_dynamic_ylim(ws_fertilities))

        if 'character_fertility' in results:
            char_fertilities = [stats['global']['mean'] for stats in results['character_fertility']['per_tokenizer'].values()]
            labels = list(results['character_fertility']['per_tokenizer'].keys())
            axes[1, 2].bar(labels, char_fertilities, color='lightcoral', alpha=0.7)
            axes[1, 2].set_title('Global Character-Based Fertility')
            axes[1, 2].set_ylabel('Tokens per Character')
            axes[1, 2].tick_params(axis='x', rotation=45)
            if char_fertilities:
                axes[1, 2].set_ylim(self._get_dynamic_ylim(char_fertilities))
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/basic_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_information_theoretic_metrics(self, results: Dict[str, Any]) -> None:
        """Plot information-theoretic metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Information-Theoretic Metrics', fontsize=16)
        
        # Type-Token Ratio
        if 'type_token_ratio' in results:
            ttrs = [stats['global_ttr'] for stats in results['type_token_ratio']['per_tokenizer'].values()]
            labels = list(results['type_token_ratio']['per_tokenizer'].keys())
            axes[0, 0].bar(labels, ttrs)
            axes[0, 0].set_title('Global Type-Token Ratio')
            axes[0, 0].set_ylabel('TTR')
            axes[0, 0].tick_params(axis='x', rotation=45)
            if ttrs:
                axes[0, 0].set_ylim(self._get_dynamic_ylim(ttrs))

        # Vocabulary Utilization
        if 'vocabulary_utilization' in results:
            utilizations = [stats['global_utilization'] for stats in results['vocabulary_utilization']['per_tokenizer'].values()]
            labels = list(results['vocabulary_utilization']['per_tokenizer'].keys())
            axes[0, 1].bar(labels, utilizations)
            axes[0, 1].set_title('Global Vocabulary Utilization')
            axes[0, 1].set_ylabel('Proportion of Vocabulary Used')
            axes[0, 1].tick_params(axis='x', rotation=45)
            if utilizations:
                axes[0, 1].set_ylim(self._get_dynamic_ylim(utilizations))

        # Shannon Entropy
        if 'renyi_efficiency' in results:
            shannon_entropies = [stats['renyi_1.0']['overall'] for stats in results['renyi_efficiency']['per_tokenizer'].values() if 'renyi_1.0' in stats]
            labels = [name for name, stats in results['renyi_efficiency']['per_tokenizer'].items() if 'renyi_1.0' in stats]
            if shannon_entropies:
                axes[1, 0].bar(labels, shannon_entropies)
                axes[1, 0].set_title('Global Shannon Entropy')
                axes[1, 0].set_ylabel('Entropy (bits)')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].set_ylim(self._get_dynamic_ylim(shannon_entropies))
        
        # Average tokens per line
        if 'avg_tokens_per_line' in results:
            avg_tokens = [stats['global_avg'] for stats in results['avg_tokens_per_line']['per_tokenizer'].values()]
            labels = list(results['avg_tokens_per_line']['per_tokenizer'].keys())
            axes[1, 1].bar(labels, avg_tokens)
            axes[1, 1].set_title('Global Average Tokens per Line')
            axes[1, 1].set_ylabel('# Tokens')
            axes[1, 1].tick_params(axis='x', rotation=45)
            if avg_tokens:
                axes[1, 1].set_ylim(self._get_dynamic_ylim(avg_tokens))
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/information_theoretic_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_unigram_distribution_metrics(self, results: Dict[str, Any]) -> None:
        """Plot unigram distribution metrics."""
        if 'unigram_distribution_metrics' not in results:
            logger.info("No unigram distribution metrics found for plotting")
            return
            
        unigram_results = results['unigram_distribution_metrics']
        fig_width = 15
        if 'per_language' in unigram_results and 'unigram_entropy' in unigram_results['per_language']:
            num_languages = len(unigram_results['per_language']['unigram_entropy'])
            fig_width = max(15, min(25, 8 + num_languages * 0.8))
        
        fig, axes = plt.subplots(2, 2, figsize=(fig_width, 12))
        fig.suptitle('Unigram Distribution Metrics', fontsize=16)
        
        # Global Metrics
        if 'per_tokenizer' in unigram_results:
            entropies = [stats.get('global_unigram_entropy', 0) for stats in unigram_results['per_tokenizer'].values()]
            avg_ranks = [stats.get('global_avg_token_rank', 0) for stats in unigram_results['per_tokenizer'].values()]
            labels = list(unigram_results['per_tokenizer'].keys())
            
            if entropies:
                axes[0, 0].bar(labels, entropies)
                axes[0, 0].set_title('Global Unigram Entropy')
                axes[0, 0].set_ylabel('Entropy (bits)')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].set_ylim(self._get_dynamic_ylim(entropies))
            
            if avg_ranks:
                axes[0, 1].bar(labels, avg_ranks)
                axes[0, 1].set_title('Global Average Token Rank')
                axes[0, 1].set_ylabel('Average Rank')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].set_ylim(self._get_dynamic_ylim(avg_ranks))
        
        # Per-language Unigram Entropy
        if 'per_language' in unigram_results and 'unigram_entropy' in unigram_results['per_language']:
            lang_data = unigram_results['per_language']['unigram_entropy']
            languages = sorted(list(lang_data.keys()))
            if languages:
                tokenizer_names = list(unigram_results['per_tokenizer'].keys())
                n_groups = len(languages)
                n_tokenizers = len(tokenizer_names)
                group_width, bar_width = 0.8, 0.8 / n_tokenizers
                indices = np.arange(n_groups)
                all_values = []

                for i, tok_name in enumerate(tokenizer_names):
                    values = [lang_data[lang].get(tok_name, 0) for lang in languages]
                    all_values.extend(values)
                    offset = (i - n_tokenizers/2) * bar_width + bar_width/2
                    axes[1, 0].bar(indices + offset, values, bar_width, label=tok_name, alpha=0.8)
                
                axes[1, 0].set_title('Unigram Entropy by Language')
                axes[1, 0].set_ylabel('Entropy (bits)')
                axes[1, 0].set_xticks(indices)
                axes[1, 0].set_xticklabels(languages, rotation=45)
                axes[1, 0].legend()
                if all_values:
                    axes[1, 0].set_ylim(self._get_dynamic_ylim(all_values))
        
        # Per-language Average Token Rank
        if 'per_language' in unigram_results and 'avg_token_rank' in unigram_results['per_language']:
            lang_data = unigram_results['per_language']['avg_token_rank']
            languages = sorted(list(lang_data.keys()))
            if languages:
                tokenizer_names = list(unigram_results['per_tokenizer'].keys())
                n_groups = len(languages)
                n_tokenizers = len(tokenizer_names)
                group_width, bar_width = 0.8, 0.8 / n_tokenizers
                indices = np.arange(n_groups)
                all_values = []

                for i, tok_name in enumerate(tokenizer_names):
                    values = [lang_data[lang].get(tok_name, 0) for lang in languages]
                    all_values.extend(values)
                    offset = (i - n_tokenizers/2) * bar_width + bar_width/2
                    axes[1, 1].bar(indices + offset, values, bar_width, label=tok_name, alpha=0.8)
                
                axes[1, 1].set_title('Average Token Rank by Language')
                axes[1, 1].set_ylabel('Average Rank')
                axes[1, 1].set_xticks(indices)
                axes[1, 1].set_xticklabels(languages, rotation=45)
                axes[1, 1].legend()
                if all_values:
                    axes[1, 1].set_ylim(self._get_dynamic_ylim(all_values))

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/unigram_distribution_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_renyi_entropy_curves(self, results: Dict[str, Any]) -> None:
        """Plot Rényi entropy curves for different alpha values."""
        if 'renyi_efficiency' not in results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract alpha values
        alphas = []
        entropy_data = {}
        
        for name in self.tokenizer_names:
            entropy_data[name] = []
            if name in results['renyi_efficiency']['per_tokenizer']:
                tok_results = results['renyi_efficiency']['per_tokenizer'][name]
                for key in tok_results:
                    if key.startswith('renyi_'):
                        alpha = float(key.split('_')[1])
                        if not alphas or alpha not in alphas:
                            alphas.append(alpha)
        
        alphas.sort()
        
        # Collect entropy values for each tokenizer
        for name in self.tokenizer_names:
            entropy_values = []
            if name in results['renyi_efficiency']['per_tokenizer']:
                tok_results = results['renyi_efficiency']['per_tokenizer'][name]
                for alpha in alphas:
                    key = f'renyi_{alpha}'
                    if key in tok_results:
                        entropy_values.append(tok_results[key]['overall'])
                    else:
                        entropy_values.append(0)
                entropy_data[name] = entropy_values
        
        # Plot 1: Rényi entropy curves
        for name in self.tokenizer_names:
            if entropy_data[name]:
                ax1.plot(alphas, entropy_data[name], marker='o', label=name, linewidth=2)
        
        ax1.set_xlabel('Alpha (α)')
        ax1.set_ylabel('Rényi Entropy (bits)')
        ax1.set_title('Rényi Entropy vs Alpha')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Per-language entropy heatmap (for α=2.0 if available)
        if any('renyi_2.0' in results['renyi_efficiency']['per_tokenizer'][name] 
               for name in self.tokenizer_names):
            
            languages = set()
            for name in self.tokenizer_names:
                if name in results['renyi_efficiency']['per_tokenizer']:
                    tok_results = results['renyi_efficiency']['per_tokenizer'][name]
                    if 'renyi_2.0' in tok_results:
                        languages.update(k for k in tok_results['renyi_2.0'].keys() if k != 'overall')
            
            languages = sorted(list(languages))
            
            if languages:
                entropy_matrix = np.zeros((len(self.tokenizer_names), len(languages)))
                
                for i, name in enumerate(self.tokenizer_names):
                    if name in results['renyi_efficiency']['per_tokenizer']:
                        tok_results = results['renyi_efficiency']['per_tokenizer'][name]
                        if 'renyi_2.0' in tok_results:
                            for j, lang in enumerate(languages):
                                entropy_matrix[i, j] = tok_results['renyi_2.0'].get(lang, 0)
                
                im = ax2.imshow(entropy_matrix, aspect='auto', cmap='viridis')
                ax2.set_xticks(range(len(languages)))
                ax2.set_xticklabels(languages, rotation=45)
                ax2.set_yticks(range(len(self.tokenizer_names)))
                ax2.set_yticklabels(self.tokenizer_names)
                ax2.set_title('Per-Language Rényi Entropy (α=2.0)')
                
                for i in range(len(self.tokenizer_names)):
                    for j in range(len(languages)):
                        ax2.text(j, i, f'{entropy_matrix[i, j]:.2f}', 
                                ha='center', va='center', color='white', fontsize=8)
                
                plt.colorbar(im, ax=ax2, label='Entropy (bits)')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/renyi_entropy_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_morphological_metrics(self, results: Dict[str, Any]) -> None:
        """Plot morphological alignment metrics."""
        if 'morphological_alignment' not in results:
            return
        
        morph_results = results['morphological_alignment']
        if 'message' in morph_results:
            logger.info("No morphological data available for plotting")
            return
        
        if not any(any(morph_results['per_tokenizer'][name]['boundary_f1'].values()) 
                  for name in self.tokenizer_names):
            logger.info("No morphological results to plot")
            return
        
        metrics_to_plot = ['boundary_f1', 'morpheme_preservation', 'over_segmentation']
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 6))
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics_to_plot):
            languages = set()
            for name in self.tokenizer_names:
                if name in morph_results['per_tokenizer']:
                    languages.update(morph_results['per_tokenizer'][name][metric].keys())
            languages = sorted(list(languages))
            if not languages: continue
            
            data_matrix = np.zeros((len(self.tokenizer_names), len(languages)))
            for i, name in enumerate(self.tokenizer_names):
                for j, lang in enumerate(languages):
                    if (name in morph_results['per_tokenizer'] and 
                        lang in morph_results['per_tokenizer'][name][metric] and
                        morph_results['per_tokenizer'][name][metric][lang]['count'] > 0):
                        data_matrix[i, j] = morph_results['per_tokenizer'][name][metric][lang]['mean']
            
            im = axes[idx].imshow(data_matrix, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[idx].set_xticks(range(len(languages)))
            axes[idx].set_xticklabels(languages, rotation=45)
            axes[idx].set_yticks(range(len(self.tokenizer_names)))
            axes[idx].set_yticklabels(self.tokenizer_names)
            title_map = {
                'boundary_f1': 'Boundary Detection F1 Score',
                'morpheme_preservation': 'Morpheme Preservation Rate',
                'over_segmentation': 'Over-Segmentation Score'
            }
            title = title_map.get(metric, metric.replace("_", " ").title())
            axes[idx].set_title(title)
            
            for i in range(len(self.tokenizer_names)):
                for j in range(len(languages)):
                    if data_matrix[i, j] > 0:
                        axes[idx].text(j, i, f'{data_matrix[i, j]:.3f}', 
                                     ha='center', va='center', 
                                     color='white' if data_matrix[i, j] < 0.5 else 'black',
                                     fontsize=8)
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/morphological_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pairwise_comparisons(self, results: Dict[str, Any]) -> None:
        """Plot pairwise comparison matrices for key metrics."""
        if len(self.tokenizer_names) < 2:
            return
        
        # Collect metrics with pairwise comparisons
        pairwise_metrics = []
        for metric_group in results.values():
            if isinstance(metric_group, dict) and 'pairwise_comparisons' in metric_group:
                pairwise_metrics.append(metric_group)
        
        if not pairwise_metrics:
            return
        
        # Create a summary plot of key ratios
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Collect ratio data
        ratio_data = {}
        comparison_names = []
        
        for metric_group in pairwise_metrics:
            comparisons = metric_group['pairwise_comparisons']
            for pair_name, comparison in comparisons.items():
                if 'ratio' in comparison:
                    if pair_name not in comparison_names:
                        comparison_names.append(pair_name)
                    
                    metric_name = "Unknown"  # Would need to track metric names better
                    if pair_name not in ratio_data:
                        ratio_data[pair_name] = []
                    ratio_data[pair_name].append(comparison['ratio'])
        
        if ratio_data:
            # Plot ratios
            x_pos = np.arange(len(comparison_names))
            
            for i, metric_name in enumerate(['Metric1', 'Metric2', 'Metric3']):  # Simplified
                if i < len(pairwise_metrics):
                    ratios = [ratio_data[name][i] if i < len(ratio_data[name]) else 1.0 
                             for name in comparison_names]
                    ax.bar(x_pos + i*0.25, ratios, width=0.25, label=metric_name, alpha=0.8)
            
            ax.set_xlabel('Tokenizer Comparisons')
            ax.set_ylabel('Ratio')
            ax.set_title('Pairwise Tokenizer Comparison Ratios')
            ax.set_xticks(x_pos + 0.25)
            ax.set_xticklabels(comparison_names, rotation=45)
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/pairwise_comparisons.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_per_language_analysis(self, results: Dict[str, Any]) -> None:
        """Plot per-language performance analysis."""
        all_languages = set()
        for metric_group in results.values():
            if isinstance(metric_group, dict) and 'per_language' in metric_group:
                all_languages.update(metric_group['per_language'].keys())
        if not all_languages: return
        all_languages = sorted(list(all_languages))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Per-Language Performance Analysis', fontsize=16)
        
        # Updated to handle new unified fertility structure
        metrics_to_plot = [
            ('compression_ratio', 'per_tokenizer', 'per_language'),
            ('fertility', 'per_tokenizer', 'per_language'),
            ('type_token_ratio', 'per_language', None),
            ('vocabulary_utilization', 'per_tokenizer', 'per_language')
        ]
        
        for idx, (metric_name, path1, path2) in enumerate(metrics_to_plot):
            if idx >= 4: break
            ax = axes[idx // 2, idx % 2]
            if metric_name not in results: continue

            lang_data = {}
            if path1 == 'per_tokenizer' and path2:
                for tok_name in self.tokenizer_names:
                    if tok_name in results[metric_name][path1] and path2 in results[metric_name][path1][tok_name]:
                        for lang, value in results[metric_name][path1][tok_name][path2].items():
                            if lang not in lang_data: lang_data[lang] = {}
                            
                            # Extract value based on known metric data structures
                            extracted_value = self._extract_per_language_value(metric_name, value)
                            lang_data[lang][tok_name] = extracted_value
            elif path1 == 'per_language':
                # Handle direct per_language structure (like type_token_ratio)
                if metric_name in results and path1 in results[metric_name]:
                    for lang, tok_values in results[metric_name][path1].items():
                        if lang not in lang_data: lang_data[lang] = {}
                        for tok_name, value in tok_values.items():
                            lang_data[lang][tok_name] = value
            
            if lang_data:
                languages = sorted([lang for lang in lang_data.keys() if lang in all_languages])
                n_groups, n_tokenizers = len(languages), len(self.tokenizer_names)
                group_width, bar_width = 0.8, 0.8 / n_tokenizers
                x_pos = np.arange(n_groups)
                all_values = []
                
                for i, tok_name in enumerate(self.tokenizer_names):
                    values = [lang_data[lang].get(tok_name, 0) for lang in languages]
                    all_values.extend(values)
                    offset = (i - n_tokenizers/2) * bar_width + bar_width/2
                    ax.bar(x_pos + offset, values, width=bar_width, label=tok_name, alpha=0.8)
                
                ax.set_xlabel('Languages')
                # Use metadata for better labels when available
                ylabel, title = self._get_per_language_labels(metric_name, results)
                
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(languages, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                if all_values:
                    ax.set_ylim(self._get_dynamic_ylim(all_values))
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/per_language_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_fertility_comparison(self, results: Dict[str, Any]) -> None:
        """Create a dedicated plot for fertility metric."""
        # Handle new unified fertility structure
        if 'fertility' in results:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            # Get metadata for dynamic labeling
            metadata = results['fertility'].get('metadata', {})
            norm_method = metadata.get('normalization_method', 'tokens')
            description = metadata.get('description', 'Fertility')
            short_desc = metadata.get('short_description', 'tokens/unit')
            
            fig.suptitle(f'Global Fertility Analysis ({norm_method.title()})', fontsize=16)
            
            fertilities = []
            labels = []
            for name in self.tokenizer_names:
                if name in results['fertility']['per_tokenizer']:
                    fertilities.append(results['fertility']['per_tokenizer'][name]['global']['mean'])
                    labels.append(name)
            
            if fertilities:
                x_pos = np.arange(len(labels))
                bars = ax.bar(x_pos, fertilities, color='skyblue', alpha=0.8)
                ax.set_title(f"Global {description}")
                ax.set_ylabel("# Tokens per " + norm_method.title())
                ax.set_xlabel('Tokenizer')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=45)
                ax.set_ylim(self._get_dynamic_ylim(fertilities))
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, fertilities)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/fertility_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Handle legacy dual fertility structure for backward compatibility
        elif 'whitespace_fertility' in results and 'character_fertility' in results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Global Fertility Metrics Comparison (Legacy)', fontsize=16)
            
            ws_fertilities = []
            char_fertilities = []
            labels = []
            for name in self.tokenizer_names:
                if (name in results['whitespace_fertility']['per_tokenizer'] and
                    name in results['character_fertility']['per_tokenizer']):
                    ws_fertilities.append(results['whitespace_fertility']['per_tokenizer'][name]['global']['mean'])
                    char_fertilities.append(results['character_fertility']['per_tokenizer'][name]['global']['mean'])
                    labels.append(name)
            
            x_pos = np.arange(len(labels))
            bars1 = ax1.bar(x_pos, ws_fertilities, color='skyblue', alpha=0.8, label='Whitespace-Delimited')
            ax1.set_title('Global Whitespace-Delimited Fertility')
            ax1.set_ylabel('Tokens per Word')
            ax1.set_xlabel('Tokenizer')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(labels, rotation=45)
            if ws_fertilities:
                ax1.set_ylim(self._get_dynamic_ylim(ws_fertilities))
            
            bars2 = ax2.bar(x_pos, char_fertilities, color='lightcoral', alpha=0.8, label='Character-Based')
            ax2.set_title('Global Character-Based Fertility')
            ax2.set_ylabel('Tokens per Character')
            ax2.set_xlabel('Tokenizer')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(labels, rotation=45)
            if char_fertilities:
                ax2.set_ylim(self._get_dynamic_ylim(char_fertilities))

            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/fertility_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_tokenizer_fairness_gini(self, results: Dict[str, Any]) -> None:
        """Plot Tokenizer Fairness Gini coefficient and related metrics."""
        if 'tokenizer_fairness_gini' not in results:
            return
        
        gini_results = results['tokenizer_fairness_gini']['per_tokenizer']
        
        if not gini_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Tokenizer Fairness Analysis', fontsize=16)
        
        # 1. Gini Coefficients Comparison
        gini_values = []
        labels = []
        for name, data in gini_results.items():
            if 'warning' not in data:
                gini_values.append(data['gini_coefficient'])
                labels.append(name)
        
        if gini_values:
            bars = axes[0, 0].bar(labels, gini_values, color='lightcoral', alpha=0.7)
            axes[0, 0].set_title('Tokenizer Fairness Gini Coefficient')
            axes[0, 0].set_ylabel('TFG (lower = more fair)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, gini_values):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                               f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Cost Ratio (Max/Min cost)
        cost_ratios = []
        labels = []
        for name, data in gini_results.items():
            if 'warning' not in data and data['cost_ratio'] != float('inf'):
                cost_ratios.append(data['cost_ratio'])
                labels.append(name)
        
        if cost_ratios:
            bars = axes[0, 1].bar(labels, cost_ratios, color='skyblue', alpha=0.7)
            axes[0, 1].set_title('Cost Ratio (Max/Min)')
            
            # Use dynamic label based on normalization method
            metadata = results.get('tokenizer_fairness_gini', {}).get('metadata', {})
            cost_unit = metadata.get('cost_unit', 'tokens per unit')
            axes[0, 1].set_ylabel(f'Ratio ({cost_unit}) - lower = more equitable')
            
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, cost_ratios):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Token Costs by Language (All Tokenizers Comparison)
        if gini_results:
            # Extract language costs for all tokenizers
            all_language_costs = {}
            valid_tokenizers = []
            
            for tok_name, data in gini_results.items():
                if 'warning' not in data and 'language_costs' in data:
                    all_language_costs[tok_name] = data['language_costs']
                    valid_tokenizers.append(tok_name)
            
            if valid_tokenizers and all_language_costs:
                # Get all languages and sort by average cost across tokenizers
                all_languages = set()
                for costs in all_language_costs.values():
                    all_languages.update(costs.keys())
                
                lang_avg_costs = {}
                for lang in all_languages:
                    costs = [all_language_costs[tok][lang] for tok in valid_tokenizers 
                            if lang in all_language_costs[tok]]
                    if costs:
                        lang_avg_costs[lang] = np.mean(costs)
                
                # Sort languages by average cost (most efficient first)
                sorted_languages = sorted(lang_avg_costs.keys(), key=lambda x: lang_avg_costs[x])
                
                # Create grouped bar chart
                n_langs = len(sorted_languages)
                n_toks = len(valid_tokenizers)
                bar_width = 0.8 / n_toks
                x_pos = np.arange(n_langs)
                
                colors = plt.cm.Set3(np.linspace(0, 1, n_toks))
                
                for i, tok_name in enumerate(valid_tokenizers):
                    costs = [all_language_costs[tok_name].get(lang, 0) for lang in sorted_languages]
                    offset = (i - n_toks/2) * bar_width + bar_width/2
                    axes[1, 0].bar(x_pos + offset, costs, width=bar_width, 
                                  label=tok_name, alpha=0.8, color=colors[i])
                
                # Get cost unit from metadata
                metadata = results.get('tokenizer_fairness_gini', {}).get('metadata', {})
                cost_unit = metadata.get('cost_unit', 'tokens per unit')
                
                axes[1, 0].set_title('Token Costs by Language (All Tokenizers)')
                axes[1, 0].set_ylabel(f'Cost ({cost_unit.title()})')
                axes[1, 0].set_xlabel('Languages (sorted by efficiency)')
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels(sorted_languages, rotation=45, ha='right')
                axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                
                # Set dynamic y-limits
                all_costs = [cost for costs in all_language_costs.values() for cost in costs.values()]
                if all_costs:
                    axes[1, 0].set_ylim(self._get_dynamic_ylim(all_costs))
        
        # 4. Summary statistics
        axes[1, 1].axis('off')  # Turn off axis for text summary
        summary_text = "Summary Statistics:\n\n"
        
        for name, data in gini_results.items():
            if 'warning' not in data:
                summary_text += f"{name}:\n"
                summary_text += f"  TFG: {data['gini_coefficient']:.4f}\n"
                summary_text += f"  Mean cost: {data['mean_cost']:.4f}\n"
                summary_text += f"  Cost range: {data['min_cost']:.4f} - {data['max_cost']:.4f}\n"
                summary_text += f"  Most efficient: {data['most_efficient_language'][0]} ({data['most_efficient_language'][1]:.4f})\n"
                summary_text += f"  Least efficient: {data['least_efficient_language'][0]} ({data['least_efficient_language'][1]:.4f})\n\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/tokenizer_fairness_gini.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_lorenz_curves(self, results: Dict[str, Any]) -> None:
        """Plot Lorenz curves for tokenizer fairness visualization."""
        if 'lorenz_curve_data' not in results:
            return
        
        lorenz_data = results['lorenz_curve_data']['per_tokenizer']
        
        if not lorenz_data:
            return
        
        # Create a subplot for each tokenizer
        n_tokenizers = len([name for name, data in lorenz_data.items() if 'warning' not in data])
        if n_tokenizers == 0:
            return
        
        # Determine subplot layout
        if n_tokenizers == 1:
            nrows, ncols = 1, 1
        elif n_tokenizers == 2:
            nrows, ncols = 1, 2
        elif n_tokenizers <= 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 2, 3
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
        fig.suptitle('Lorenz Curves - Tokenizer Fairness Across Languages', fontsize=16)
        
        # Handle single subplot case
        if n_tokenizers == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        for tok_name, data in lorenz_data.items():
            if 'warning' in data or plot_idx >= len(axes):
                continue
            
            ax = axes[plot_idx]
            
            # Plot Lorenz curve
            ax.plot(data['x_values'], data['y_values'], 'b-', linewidth=2, 
                   label=f'{tok_name} (actual)')
            
            # Plot perfect equality line
            ax.plot(data['equality_line'], data['equality_line'], 'r--', linewidth=1,
                   label='Perfect equality')
            
            # Fill area between curves to show inequality
            ax.fill_between(data['x_values'], data['y_values'], data['equality_line'],
                           alpha=0.3, color='lightblue', label='Inequality area')
            
            ax.set_xlabel('Cumulative Proportion of Languages\n(sorted by efficiency)')
            ax.set_ylabel('Cumulative Proportion of Token Cost')
            ax.set_title(f'{tok_name}\n({data["n_languages"]} languages)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Add language labels on the curve
            if len(data['sorted_languages']) <= 10:  # Only for small number of languages
                for i, lang in enumerate(data['sorted_languages']):
                    x = (i + 1) / data['n_languages']
                    y = data['y_values'][i + 1]
                    ax.annotate(lang, (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/lorenz_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_plots(self, results: Dict[str, Any], print_pairwise: bool = False) -> None:
        """Generate all available plots for the results."""
        logger.info(f"Generating plots in {self.save_dir}")
        
        try:
            self.plot_basic_metrics_comparison(results)
            self.plot_fertility_comparison(results)
            self.plot_tokenizer_fairness_gini(results)  # New Gini fairness analysis
            self.plot_lorenz_curves(results)  # New Lorenz curves
            self.plot_information_theoretic_metrics(results)
            self.plot_renyi_entropy_curves(results)
            self.plot_unigram_distribution_metrics(results)
            self.plot_morphological_metrics(results)
            self.plot_per_language_analysis(results)
            if print_pairwise:
                self.plot_pairwise_comparisons(results)
            
            logger.info(f"All plots saved to {self.save_dir}")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            raise
