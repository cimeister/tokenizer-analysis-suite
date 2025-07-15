"""
Base class for plot renderers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

from ..plot_config import PlotConfig, MetricConfig
from ..data_extraction import DataExtractor

logger = logging.getLogger(__name__)


class PlotRenderer(ABC):
    """Abstract base class for plot renderers."""
    
    def __init__(self, tokenizer_names: List[str], save_dir: str):
        """
        Initialize renderer.
        
        Args:
            tokenizer_names: List of tokenizer names
            save_dir: Directory to save plots
        """
        self.tokenizer_names = tokenizer_names
        self.save_dir = save_dir
        self.config = PlotConfig()
        self.metric_config = MetricConfig()
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
    
    @abstractmethod
    def render(self, results: Dict[str, Any]) -> None:
        """
        Render plots for this renderer type.
        
        Args:
            results: Analysis results dictionary
        """
        pass
    
    def _save_plot(self, filename: str, **kwargs) -> None:
        """
        Save plot with consistent formatting.
        
        Args:
            filename: Name of the file to save
            **kwargs: Additional arguments for savefig
        """
        default_kwargs = {
            'dpi': PlotConfig.DEFAULT_DPI,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        default_kwargs.update(kwargs)
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Get current figure
        fig = plt.gcf()
        
        # Save the figure
        fig.savefig(filepath, **default_kwargs)
        
        # Close the figure
        plt.close(fig)
        
        logger.debug(f"Saved plot: {filepath}")
    
    def _setup_subplot(self, ax: plt.Axes, title: str, xlabel: str, ylabel: str,
                      rotation: int = 45, add_grid: bool = True) -> None:
        """
        Set up subplot with consistent formatting.
        
        Args:
            ax: Matplotlib axes object
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            rotation: X-axis label rotation
            add_grid: Whether to add grid
        """
        ax.set_title(title, fontsize=PlotConfig.TITLE_FONTSIZE)
        ax.set_xlabel(xlabel, fontsize=PlotConfig.AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=PlotConfig.AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis='x', rotation=rotation, labelsize=PlotConfig.TICK_LABEL_FONTSIZE)
        ax.tick_params(axis='y', labelsize=PlotConfig.TICK_LABEL_FONTSIZE)
        
        if add_grid:
            ax.grid(True, alpha=PlotConfig.ALPHA_GRID)
    
    def _add_value_labels(self, ax: plt.Axes, bars, values: List[float], 
                         format_str: str = '{:.3f}') -> None:
        """
        Add value labels on top of bars.
        
        Args:
            ax: Matplotlib axes object
            bars: Bar objects from bar plot
            values: List of values to display
            format_str: Format string for values
        """
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   format_str.format(value), ha='center', va='bottom',
                   fontsize=PlotConfig.ANNOTATION_FONTSIZE)
    
    def _add_reference_line(self, ax: plt.Axes, value: float, 
                           label: str = None, **kwargs) -> None:
        """
        Add horizontal reference line.
        
        Args:
            ax: Matplotlib axes object
            value: Y-value for reference line
            label: Optional label for the line
            **kwargs: Additional arguments for axhline
        """
        default_kwargs = {
            'color': 'red',
            'linestyle': '--',
            'alpha': 0.7
        }
        default_kwargs.update(kwargs)
        
        ax.axhline(y=value, **default_kwargs)
        if label:
            ax.text(0.02, value, label, transform=ax.get_yaxis_transform(),
                   fontsize=PlotConfig.ANNOTATION_FONTSIZE, va='bottom')
    
    def _create_grouped_bar_plot(self, ax: plt.Axes, data: Dict[str, List[float]],
                                labels: List[str], title: str, ylabel: str,
                                add_legend: bool = True) -> None:
        """
        Create grouped bar plot.
        
        Args:
            ax: Matplotlib axes object
            data: Dictionary mapping group names to values
            labels: X-axis labels
            title: Plot title
            ylabel: Y-axis label
            add_legend: Whether to add legend
        """
        x_pos, bar_width = PlotConfig.calculate_bar_layout(len(labels), len(data))
        colors = PlotConfig.get_colors(len(data))
        
        all_values = []
        for i, (group_name, values) in enumerate(data.items()):
            offset = (i - len(data)/2) * bar_width + bar_width/2
            bars = ax.bar(x_pos + offset, values, bar_width, 
                         label=group_name, alpha=PlotConfig.ALPHA_BARS,
                         color=colors[i])
            all_values.extend(values)
        
        self._setup_subplot(ax, title, 'Tokenizers', ylabel)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45)
        
        if add_legend:
            ax.legend(fontsize=PlotConfig.LEGEND_FONTSIZE)
        
        # Set dynamic y-limits
        if all_values:
            ax.set_ylim(PlotConfig.get_dynamic_ylim(all_values))
    
    def _create_heatmap(self, ax: plt.Axes, data: np.ndarray, 
                       row_labels: List[str], col_labels: List[str],
                       title: str, cmap: str = 'viridis', 
                       add_colorbar: bool = True, add_annotations: bool = True) -> None:
        """
        Create heatmap plot.
        
        Args:
            ax: Matplotlib axes object
            data: 2D array of values
            row_labels: Row labels
            col_labels: Column labels
            title: Plot title
            cmap: Colormap name
            add_colorbar: Whether to add colorbar
            add_annotations: Whether to add value annotations
        """
        im = ax.imshow(data, aspect='auto', cmap=cmap)
        
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_title(title, fontsize=PlotConfig.TITLE_FONTSIZE)
        
        if add_colorbar:
            plt.colorbar(im, ax=ax)
        
        if add_annotations:
            # Add value annotations
            for i in range(len(row_labels)):
                for j in range(len(col_labels)):
                    value = data[i, j]
                    color = 'white' if value < np.median(data) else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                           color=color, fontsize=PlotConfig.ANNOTATION_FONTSIZE)
    
    def _validate_data(self, results: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that required data exists in results.
        
        Args:
            results: Analysis results dictionary
            required_keys: List of required keys
            
        Returns:
            True if all required keys exist, False otherwise
        """
        for key in required_keys:
            if key not in results:
                logger.warning(f"Missing required key for plotting: {key}")
                return False
        return True
    
    def _has_valid_data(self, data: Dict[str, Any]) -> bool:
        """
        Check if data dictionary has valid plotting data.
        
        Args:
            data: Data dictionary
            
        Returns:
            True if data is valid for plotting
        """
        if not data:
            return False
        
        # Check if we have any non-zero values
        if 'global_values' in data:
            return bool(data['global_values'])
        
        if 'per_language_data' in data:
            return bool(data['per_language_data'])
        
        return True