"""
Renderer for basic tokenization metrics.
"""

from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

from .base import PlotRenderer
from ..plot_config import PlotConfig, MetricConfig
from ..data_extraction import DataExtractor

logger = logging.getLogger(__name__)


class BasicMetricsRenderer(PlotRenderer):
    """Renderer for basic tokenization metrics."""
    
    def render(self, results: Dict[str, Any]) -> None:
        """Render basic metrics plots."""
        try:
            self._plot_basic_metrics_comparison(results)
            self._plot_fertility_comparison(results)
            self._plot_per_language_analysis(results)
            logger.info("Basic metrics plots rendered successfully")
        except Exception as e:
            logger.error(f"Error rendering basic metrics: {e}")
            raise
    
    def _plot_basic_metrics_comparison(self, results: Dict[str, Any]) -> None:
        """Plot basic tokenization metrics comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Basic Tokenization Metrics Comparison', fontsize=PlotConfig.TITLE_FONTSIZE, y=0.98)
        
        # 1. Token length comparison (top-left)
        ax = axes[0, 0]
        if 'token_length' in results:
            primary_lengths = []
            labels = []
            unit = 'bytes'
            
            if 'metadata' in results['token_length']:
                unit = results['token_length']['metadata'].get('primary_unit', 'bytes')
            
            for name in self.tokenizer_names:
                if name in results['token_length'].get('per_tokenizer', {}):
                    stats = results['token_length']['per_tokenizer'][name]
                    
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
            
            if primary_lengths and labels:
                bars = ax.bar(labels, primary_lengths, alpha=PlotConfig.ALPHA_BARS)
                ax.set_title('Global Average Token Length', fontsize=PlotConfig.TITLE_FONTSIZE)
                ax.set_xlabel('Tokenizer', fontsize=PlotConfig.AXIS_LABEL_FONTSIZE)
                ax.set_ylabel(f'{unit.capitalize()} per Token', fontsize=PlotConfig.AXIS_LABEL_FONTSIZE)
                ax.tick_params(axis='x', rotation=45, labelsize=PlotConfig.TICK_LABEL_FONTSIZE)
                ax.grid(True, alpha=PlotConfig.ALPHA_GRID)
                ax.set_ylim(PlotConfig.get_dynamic_ylim(primary_lengths))
            else:
                ax.set_visible(False)
        else:
            ax.set_visible(False)
        
        # 2. Vocabulary utilization (top-right)
        ax = axes[0, 1]
        if 'vocabulary_utilization' in results:
            utilizations = []
            labels = []
            
            for name in self.tokenizer_names:
                if name in results['vocabulary_utilization'].get('per_tokenizer', {}):
                    utilizations.append(results['vocabulary_utilization']['per_tokenizer'][name]['global_utilization'])
                    labels.append(name)
            
            if utilizations and labels:
                bars = ax.bar(labels, utilizations, alpha=PlotConfig.ALPHA_BARS)
                ax.set_title('Global Vocabulary Utilization', fontsize=PlotConfig.TITLE_FONTSIZE)
                ax.set_xlabel('Tokenizer', fontsize=PlotConfig.AXIS_LABEL_FONTSIZE)
                ax.set_ylabel('Proportion of Vocabulary Used', fontsize=PlotConfig.AXIS_LABEL_FONTSIZE)
                ax.tick_params(axis='x', rotation=45, labelsize=PlotConfig.TICK_LABEL_FONTSIZE)
                ax.grid(True, alpha=PlotConfig.ALPHA_GRID)
                ax.set_ylim(PlotConfig.get_dynamic_ylim(utilizations))
            else:
                ax.set_visible(False)
        else:
            ax.set_visible(False)
        
        # 3. Compression ratios (bottom-left)
        ax = axes[1, 0]
        if 'compression_ratio' in results:
            data = DataExtractor.extract_metric_data(results, 'compression_ratio', self.tokenizer_names)
            if self._has_valid_data(data):
                compression_ratios = list(data['global_values'].values())
                labels = list(data['global_values'].keys())
                
                if compression_ratios and labels:
                    bars = ax.bar(labels, compression_ratios, alpha=PlotConfig.ALPHA_BARS)
                    ylabel = MetricConfig.get_metric_ylabel('compression_ratio', data['metadata'])
                    ax.set_title('Global Compression Ratio', fontsize=PlotConfig.TITLE_FONTSIZE)
                    ax.set_xlabel('Tokenizer', fontsize=PlotConfig.AXIS_LABEL_FONTSIZE)
                    ax.set_ylabel(ylabel, fontsize=PlotConfig.AXIS_LABEL_FONTSIZE)
                    ax.tick_params(axis='x', rotation=45, labelsize=PlotConfig.TICK_LABEL_FONTSIZE)
                    ax.grid(True, alpha=PlotConfig.ALPHA_GRID)
                    
                    # Add reference line
                    ref_value = MetricConfig.REFERENCE_LINES.get('compression_ratio')
                    if ref_value:
                        ax.axhline(y=ref_value, color='red', linestyle='--', alpha=0.7)
                    
                    ax.set_ylim(PlotConfig.get_dynamic_ylim(compression_ratios))
                else:
                    ax.set_visible(False)
            else:
                ax.set_visible(False)
        else:
            ax.set_visible(False)
        
        # 4. Fertility comparison (bottom-right)
        ax = axes[1, 1]
        if 'fertility' in results:
            data = DataExtractor.extract_metric_data(results, 'fertility', self.tokenizer_names)
            if self._has_valid_data(data):
                fertilities = list(data['global_values'].values())
                labels = list(data['global_values'].keys())
                
                if fertilities and labels:
                    bars = ax.bar(labels, fertilities, color='skyblue', alpha=PlotConfig.ALPHA_BARS)
                    title = MetricConfig.get_metric_title('fertility', data['metadata'])
                    ylabel = MetricConfig.get_metric_ylabel('fertility', data['metadata'])
                    ax.set_title(title, fontsize=PlotConfig.TITLE_FONTSIZE)
                    ax.set_xlabel('Tokenizer', fontsize=PlotConfig.AXIS_LABEL_FONTSIZE)
                    ax.set_ylabel(ylabel, fontsize=PlotConfig.AXIS_LABEL_FONTSIZE)
                    ax.tick_params(axis='x', rotation=45, labelsize=PlotConfig.TICK_LABEL_FONTSIZE)
                    ax.grid(True, alpha=PlotConfig.ALPHA_GRID)
                    ax.set_ylim(PlotConfig.get_dynamic_ylim(fertilities))
                else:
                    ax.set_visible(False)
            else:
                ax.set_visible(False)
        else:
            ax.set_visible(False)
        
        # Use subplots_adjust on the specific figure
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.90, wspace=0.3, hspace=0.4)
        
        # Save the specific figure
        filepath = os.path.join(self.save_dir, 'basic_metrics_comparison.png')
        fig.savefig(filepath, dpi=PlotConfig.DEFAULT_DPI, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
    
    def _plot_fertility_comparison(self, results: Dict[str, Any]) -> None:
        """Create dedicated fertility comparison plot."""
        if 'fertility' not in results:
            return
        
        data = DataExtractor.extract_metric_data(results, 'fertility', self.tokenizer_names)
        if not self._has_valid_data(data):
            return
        
        fig, ax = plt.subplots(1, 1, figsize=PlotConfig.SINGLE_PLOT_SIZE)
        
        # Get metadata for dynamic labeling
        title = MetricConfig.get_metric_title('fertility', data['metadata'])
        ylabel = MetricConfig.get_metric_ylabel('fertility', data['metadata'])
        
        fig.suptitle(f'Global Fertility Analysis', fontsize=PlotConfig.TITLE_FONTSIZE)
        
        fertilities = list(data['global_values'].values())
        labels = list(data['global_values'].keys())
        
        if fertilities:
            bars = ax.bar(labels, fertilities, color='skyblue', alpha=PlotConfig.ALPHA_BARS)
            self._setup_subplot(ax, title, 'Tokenizer', ylabel)
            ax.set_ylim(PlotConfig.get_dynamic_ylim(fertilities))
            
            # Add value labels on bars
            self._add_value_labels(ax, bars, fertilities)
        
        plt.tight_layout()
        self._save_plot('fertility_comparison.png')
    
    def _plot_per_language_analysis(self, results: Dict[str, Any]) -> None:
        """Plot per-language performance analysis."""
        # Identify available languages across all metrics
        all_languages = set()
        available_metrics = []
        
        metrics_to_check = ['compression_ratio', 'fertility', 'type_token_ratio', 'vocabulary_utilization']
        
        for metric_name in metrics_to_check:
            if metric_name in results:
                data = DataExtractor.extract_metric_data(results, metric_name, self.tokenizer_names)
                if data.get('available_languages'):
                    all_languages.update(data['available_languages'])
                    available_metrics.append(metric_name)
        
        if not all_languages or not available_metrics:
            return
        
        all_languages = sorted(list(all_languages))
        n_metrics = min(4, len(available_metrics))  # Maximum 4 subplots
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Per-Language Performance Analysis', fontsize=PlotConfig.TITLE_FONTSIZE)
        
        for idx, metric_name in enumerate(available_metrics[:n_metrics]):
            ax = axes[idx // 2, idx % 2]
            
            data = DataExtractor.extract_metric_data(results, metric_name, self.tokenizer_names)
            if not data.get('per_language_data'):
                ax.set_visible(False)
                continue
            
            # Prepare data for grouped bar chart
            languages = sorted([lang for lang in data['per_language_data'].keys() 
                              if lang in all_languages])
            
            if not languages:
                ax.set_visible(False)
                continue
            
            # Create grouped bar data
            plot_data = {}
            for tok_name in self.tokenizer_names:
                plot_data[tok_name] = [
                    data['per_language_data'][lang].get(tok_name, 0) 
                    for lang in languages
                ]
            
            # Create grouped bar plot
            ylabel = MetricConfig.get_metric_ylabel(metric_name, data['metadata'])
            title = MetricConfig.get_metric_title(metric_name, data['metadata']) + ' by Language'
            
            self._create_grouped_bar_plot(ax, plot_data, languages, title, ylabel)
        
        # Hide unused subplots
        for idx in range(n_metrics, 4):
            axes[idx // 2, idx % 2].set_visible(False)
        
        plt.tight_layout()
        self._save_plot('per_language_analysis.png')