"""
Renderer for grouped analysis plots.
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


class GroupedRenderer(PlotRenderer):
    """Renderer for grouped analysis plots."""
    
    def __init__(self, tokenizer_names: List[str], save_dir: str):
        """Initialize grouped renderer with grouped plots subdirectory."""
        super().__init__(tokenizer_names, save_dir)
        self.grouped_dir = os.path.join(save_dir, "grouped_plots")
        os.makedirs(self.grouped_dir, exist_ok=True)
    
    def render(self, grouped_results: Dict[str, Dict[str, Any]]) -> None:
        """Render grouped analysis plots."""
        if not grouped_results:
            return
        
        try:
            for group_type, group_data in grouped_results.items():
                if not group_data:
                    continue
                
                logger.info(f"Generating plots for {group_type}")
                
                # Generate different types of grouped plots
                self._plot_group_comparison_bars(group_data, group_type)
                self._plot_cross_group_heatmap(group_data, group_type)
                self._plot_group_summary_dashboard(group_data, group_type)
            
            logger.info("Grouped analysis plots rendered successfully")
        except Exception as e:
            logger.error(f"Error rendering grouped plots: {e}")
            raise
    
    def _plot_group_comparison_bars(self, group_data: Dict[str, Any], group_type: str) -> None:
        """Plot bar charts comparing key metrics across groups."""
        key_metrics = ['fertility', 'compression_ratio', 'vocabulary_utilization']
        
        for metric in key_metrics:
            extracted_data = DataExtractor.extract_grouped_data(group_data, metric, self.tokenizer_names)
            
            if not extracted_data.get('group_names') or not extracted_data.get('data_matrix'):
                continue
            
            # Create the bar plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            group_names = extracted_data['group_names']
            data_matrix = extracted_data['data_matrix']
            
            # Create grouped bars
            x_pos, bar_width = PlotConfig.calculate_bar_layout(len(group_names), len(self.tokenizer_names))
            colors = PlotConfig.get_colors(len(self.tokenizer_names))
            
            for i, tok_name in enumerate(self.tokenizer_names):
                values = [row[i] for row in data_matrix]
                offset = (i - len(self.tokenizer_names)/2) * bar_width + bar_width/2
                ax.bar(x_pos + offset, values, bar_width, label=tok_name, 
                      alpha=PlotConfig.ALPHA_BARS, color=colors[i])
            
            # Setup plot
            ylabel = self._get_grouped_ylabel(metric, extracted_data['metadata'])
            title = f'{metric.replace("_", " ").title()} Comparison by {group_type.replace("_", " ").title()}'
            
            self._setup_subplot(ax, title, f'{group_type.replace("_", " ").title()}', ylabel)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(group_names)
            ax.legend(fontsize=PlotConfig.LEGEND_FONTSIZE)
            
            plt.tight_layout()
            self._save_plot(f'{group_type}_{metric}_comparison.png')
    
    def _plot_cross_group_heatmap(self, group_data: Dict[str, Any], group_type: str) -> None:
        """Plot heatmap showing tokenizer performance across different groups."""
        key_metrics = ['fertility', 'compression_ratio', 'vocabulary_utilization']
        
        for metric in key_metrics:
            extracted_data = DataExtractor.extract_grouped_data(group_data, metric, self.tokenizer_names)
            
            if not extracted_data.get('group_names') or not extracted_data.get('data_matrix'):
                continue
            
            group_names = extracted_data['group_names']
            data_matrix = np.array(extracted_data['data_matrix'])
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Choose colormap based on metric
            if data_matrix.max() > data_matrix.min():
                cmap = 'RdYlGn'
            else:
                cmap = 'viridis'
            
            self._create_heatmap(ax, data_matrix, group_names, self.tokenizer_names,
                               f'{metric.replace("_", " ").title()} Across {group_type.replace("_", " ").title()}',
                               cmap, add_colorbar=True, add_annotations=True)
            
            # Add ylabel for axes
            ylabel = self._get_grouped_ylabel(metric, extracted_data['metadata'])
            ax.set_ylabel(f'{group_type.replace("_", " ").title()}')
            ax.set_xlabel('Tokenizers')
            
            # Add colorbar with normalization-aware label
            im = ax.get_images()[0]
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(ylabel, rotation=270, labelpad=20)
            
            plt.tight_layout()
            self._save_plot(f'{group_type}_{metric}_heatmap.png')
    
    def _plot_group_summary_dashboard(self, group_data: Dict[str, Any], group_type: str) -> None:
        """Plot a comprehensive dashboard summarizing group analysis."""
        if not group_data:
            return
        
        # Set up the dashboard layout
        fig = plt.figure(figsize=PlotConfig.DASHBOARD_FIGURE_SIZE)
        gs = fig.add_gridspec(3, 4, hspace=PlotConfig.SUBPLOT_HSPACE, wspace=PlotConfig.SUBPLOT_WSPACE)
        
        fig.suptitle(f'{group_type.replace("_", " ").title()} Analysis Dashboard', 
                    fontsize=PlotConfig.TITLE_FONTSIZE)
        
        # Key metrics to display
        key_metrics = ['fertility', 'compression_ratio', 'vocabulary_utilization']
        
        # Plot key metrics
        for idx, metric in enumerate(key_metrics):
            if idx >= 4:  # Only show first 4 metrics
                break
            
            ax = fig.add_subplot(gs[idx // 2, (idx % 2) * 2])
            
            extracted_data = DataExtractor.extract_grouped_data(group_data, metric, self.tokenizer_names)
            
            if not extracted_data.get('group_names') or not extracted_data.get('data_matrix'):
                ax.set_visible(False)
                continue
            
            group_names = extracted_data['group_names']
            data_matrix = extracted_data['data_matrix']
            
            # Create grouped bar chart
            x_pos, bar_width = PlotConfig.calculate_bar_layout(len(group_names), len(self.tokenizer_names))
            colors = PlotConfig.get_colors(len(self.tokenizer_names))
            
            for i, tok_name in enumerate(self.tokenizer_names):
                values = [row[i] for row in data_matrix]
                offset = (i - len(self.tokenizer_names)/2) * bar_width + bar_width/2
                ax.bar(x_pos + offset, values, bar_width, 
                      label=tok_name if idx == 0 else "", 
                      alpha=PlotConfig.ALPHA_BARS, color=colors[i])
            
            # Setup subplot
            ylabel = self._get_grouped_ylabel(metric, extracted_data['metadata'])
            title = metric.replace('_', ' ').title()
            
            self._setup_subplot(ax, title, '', ylabel, rotation=45, add_grid=True)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(group_names, rotation=45)
        
        # Add legend
        if len(key_metrics) > 0:
            ax_legend = fig.add_subplot(gs[0, 2:])
            ax_legend.axis('off')
            
            # Get legend handles from first subplot
            handles, labels = [], []
            for ax in fig.get_axes():
                if ax.get_legend_handles_labels()[0]:
                    handles, labels = ax.get_legend_handles_labels()
                    break
            
            if handles:
                ax_legend.legend(handles, labels, loc='center', title='Tokenizers',
                               fontsize=PlotConfig.LEGEND_FONTSIZE)
        
        # Add summary statistics
        ax_summary = fig.add_subplot(gs[1:, 2:])
        ax_summary.axis('off')
        
        # Create summary text
        summary_text = self._create_summary_text(group_data, group_type)
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                       fontsize=PlotConfig.AXIS_LABEL_FONTSIZE, verticalalignment='top', 
                       fontfamily='monospace')
        
        plt.tight_layout()
        self._save_plot(f'{group_type}_summary_dashboard.png')
    
    def _create_summary_text(self, group_data: Dict[str, Any], group_type: str) -> str:
        """Create summary text for dashboard."""
        summary_text = f"Summary for {group_type.replace('_', ' ').title()}:\n\n"
        summary_text += f"Number of groups: {len(group_data)}\n"
        summary_text += f"Groups analyzed: {', '.join(group_data.keys())}\n\n"
        
        # Add performance highlights if fertility data is available
        fertility_rankings = self._get_fertility_rankings(group_data)
        if fertility_rankings:
            summary_text += "Performance highlights (Fertility):\n"
            for group_name, (best_tok, best_value) in fertility_rankings.items():
                summary_text += f"• {group_name}: {best_tok} ({best_value:.2f})\n"
        
        return summary_text
    
    def _get_fertility_rankings(self, group_data: Dict[str, Any]) -> Dict[str, tuple]:
        """Get fertility rankings for each group."""
        rankings = {}
        
        for group_name, results in group_data.items():
            if 'fertility' not in results:
                continue
            
            best_tok = None
            best_value = float('inf')
            
            for tok_name in self.tokenizer_names:
                if (tok_name in results['fertility'].get('per_tokenizer', {}) and 
                    'global' in results['fertility']['per_tokenizer'][tok_name]):
                    
                    global_data = results['fertility']['per_tokenizer'][tok_name]['global']
                    value = DataExtractor.extract_global_value('fertility', global_data)
                    
                    if value < best_value:
                        best_value = value
                        best_tok = tok_name
            
            if best_tok:
                rankings[group_name] = (best_tok, best_value)
        
        return rankings
    
    def _get_grouped_ylabel(self, metric_name: str, metadata: Dict[str, Any]) -> str:
        """Get normalization-aware ylabel for grouped analysis plots."""
        if metadata:
            return MetricConfig.get_metric_ylabel(metric_name, metadata)
        else:
            return MetricConfig.METRIC_YLABELS.get(metric_name, metric_name.replace('_', ' ').title())
    
    def _save_plot(self, filename: str, **kwargs) -> None:
        """Save plot in grouped plots directory."""
        default_kwargs = {
            'dpi': PlotConfig.DEFAULT_DPI,
            'bbox_inches': 'tight'
        }
        default_kwargs.update(kwargs)
        
        filepath = os.path.join(self.grouped_dir, filename)
        plt.savefig(filepath, **default_kwargs)
        plt.close()
        logger.debug(f"Saved grouped plot: {filepath}")