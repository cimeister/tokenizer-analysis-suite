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
    
    def plot_basic_metrics_comparison(self, results: Dict[str, Any]) -> None:
        """Plot basic tokenization metrics comparison."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Basic Tokenization Metrics Comparison', fontsize=16)
        
        # Vocabulary sizes - starting at 0 is important here
        if 'vocabulary_overlap' in results:
            vocab_sizes = results['vocabulary_overlap']['vocabulary_sizes']
            axes[0, 0].bar(vocab_sizes.keys(), vocab_sizes.values())
            axes[0, 0].set_title('Vocabulary Sizes')
            axes[0, 0].set_ylabel('Number of Tokens')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
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
            
            axes[0, 1].bar(labels, primary_lengths)
            axes[0, 1].set_title('Average Token Length')
            axes[0, 1].set_ylabel(f'{unit.capitalize()} per Token')
            axes[0, 1].tick_params(axis='x', rotation=45)
            if primary_lengths:
                axes[0, 1].set_ylim(self._get_dynamic_ylim(primary_lengths))
        
        # Compression ratios
        if 'compression_ratio' in results:
            compression_ratios = [stats['global'] for stats in results['compression_ratio']['per_tokenizer'].values()]
            labels = list(results['compression_ratio']['per_tokenizer'].keys())
            axes[1, 0].bar(labels, compression_ratios)
            axes[1, 0].set_title('Compression Ratio')
            axes[1, 0].set_ylabel('Bytes per Token')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
            if compression_ratios:
                axes[1, 0].set_ylim(self._get_dynamic_ylim(compression_ratios))

        # Fertility comparisons
        if 'whitespace_fertility' in results:
            ws_fertilities = [stats['global']['mean'] for stats in results['whitespace_fertility']['per_tokenizer'].values()]
            labels = list(results['whitespace_fertility']['per_tokenizer'].keys())
            axes[1, 1].bar(labels, ws_fertilities, color='skyblue', alpha=0.7)
            axes[1, 1].set_title('Whitespace-Delimited Fertility')
            axes[1, 1].set_ylabel('Tokens per Word')
            axes[1, 1].tick_params(axis='x', rotation=45)
            if ws_fertilities:
                axes[1, 1].set_ylim(self._get_dynamic_ylim(ws_fertilities))

        if 'character_fertility' in results:
            char_fertilities = [stats['global']['mean'] for stats in results['character_fertility']['per_tokenizer'].values()]
            labels = list(results['character_fertility']['per_tokenizer'].keys())
            axes[1, 2].bar(labels, char_fertilities, color='lightcoral', alpha=0.7)
            axes[1, 2].set_title('Character-Based Fertility')
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
            axes[0, 1].set_title('Vocabulary Utilization')
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
                axes[1, 0].set_title('Shannon Entropy')
                axes[1, 0].set_ylabel('Entropy (bits)')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].set_ylim(self._get_dynamic_ylim(shannon_entropies))
        
        # Average tokens per line
        if 'avg_tokens_per_line' in results:
            avg_tokens = [stats['global_avg'] for stats in results['avg_tokens_per_line']['per_tokenizer'].values()]
            labels = list(results['avg_tokens_per_line']['per_tokenizer'].keys())
            axes[1, 1].bar(labels, avg_tokens)
            axes[1, 1].set_title('Average Tokens per Line')
            axes[1, 1].set_ylabel('Tokens')
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
        
        metrics_to_plot = [
            ('compression_ratio', 'per_tokenizer', 'per_language'),
            ('whitespace_fertility', 'per_tokenizer', 'per_language'),
            ('character_fertility', 'per_tokenizer', 'per_language'),
            ('type_token_ratio', 'per_language', None)
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
                            lang_data[lang][tok_name] = value['mean'] if isinstance(value, dict) and 'mean' in value else value
            elif path1 == 'per_language':
                lang_data = results[metric_name][path1]
            
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
                ax.set_ylabel(metric_name.replace('_', ' ').title())
                ax.set_title(f'{metric_name.replace("_", " ").title()} by Language')
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
        """Create a dedicated plot comparing both fertility metrics."""
        if 'whitespace_fertility' not in results or 'character_fertility' not in results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Fertility Metrics Comparison', fontsize=16)
        
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
        ax1.set_title('Whitespace-Delimited Fertility')
        ax1.set_ylabel('Tokens per Word')
        ax1.set_xlabel('Tokenizer')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45)
        if ws_fertilities:
            ax1.set_ylim(self._get_dynamic_ylim(ws_fertilities))
        
        bars2 = ax2.bar(x_pos, char_fertilities, color='lightcoral', alpha=0.8, label='Character-Based')
        ax2.set_title('Character-Based Fertility')
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
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
            axes[0, 1].set_ylabel('Ratio (lower = more equitable)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, cost_ratios):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Language Cost Distribution (for first tokenizer as example)
        first_tokenizer = list(gini_results.keys())[0]
        if 'warning' not in gini_results[first_tokenizer]:
            data = gini_results[first_tokenizer]
            sorted_langs = data['sorted_language_costs']
            
            if sorted_langs:
                languages = [item[0] for item in sorted_langs]
                costs = [item[1] for item in sorted_langs]
                
                bars = axes[1, 0].bar(range(len(languages)), costs, color='lightgreen', alpha=0.7)
                axes[1, 0].set_title(f'Token Costs by Language ({first_tokenizer})')
                axes[1, 0].set_ylabel('Token Cost (tokens/byte)')
                axes[1, 0].set_xlabel('Languages (sorted by efficiency)')
                axes[1, 0].set_xticks(range(len(languages)))
                axes[1, 0].set_xticklabels(languages, rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
                
                # Highlight most and least efficient
                if len(costs) > 1:
                    bars[0].set_color('green')  # Most efficient
                    bars[-1].set_color('red')   # Least efficient
        
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
