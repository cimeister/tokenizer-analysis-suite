"""
Configuration and constants for tokenizer analysis plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class PlotConfig:
    """Centralized configuration for all plot styling and layout."""
    
    # Figure sizes
    BASIC_FIGURE_SIZE = (12, 12)
    WIDE_FIGURE_SIZE = (15, 6)
    TALL_FIGURE_SIZE = (10, 12)
    DASHBOARD_FIGURE_SIZE = (20, 12)
    SINGLE_PLOT_SIZE = (10, 6)
    
    # Colors and styling
    DEFAULT_DPI = 300
    ALPHA_BARS = 0.8
    ALPHA_FILL = 0.3
    ALPHA_GRID = 0.3
    
    # Layout parameters
    TIGHT_LAYOUT_PAD = 1.08
    SUBPLOT_HSPACE = 0.3
    SUBPLOT_WSPACE = 0.3
    
    # Text and labels
    TITLE_FONTSIZE = 16
    AXIS_LABEL_FONTSIZE = 12
    TICK_LABEL_FONTSIZE = 10
    LEGEND_FONTSIZE = 10
    ANNOTATION_FONTSIZE = 8
    
    # Plot limits and margins
    YLIM_BUFFER_RATIO = 0.1
    BAR_WIDTH_RATIO = 0.8
    
    @staticmethod
    def setup_style() -> None:
        """Set up matplotlib and seaborn styling."""
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    @staticmethod
    def get_colors(n_items: int) -> List[str]:
        """Get a list of distinct colors for plotting."""
        if n_items <= 10:
            return sns.color_palette("husl", n_items)
        else:
            return plt.cm.Set3(np.linspace(0, 1, n_items))
    
    @staticmethod
    def get_dynamic_ylim(data: List[float], buffer_ratio: float = None, 
                        lower_bound: Optional[float] = 0.0) -> Tuple[float, float]:
        """
        Calculate dynamic y-axis limits to zoom in on data.
        
        Args:
            data: List of data points for the y-axis
            buffer_ratio: Percentage of data range to add as padding
            lower_bound: Minimum y-axis value (None for no constraint)
            
        Returns:
            Tuple of (min_limit, max_limit) for y-axis
        """
        if buffer_ratio is None:
            buffer_ratio = PlotConfig.YLIM_BUFFER_RATIO
            
        if not data or len(data) == 0:
            return (0, 1)
        
        min_val = min(data)
        max_val = max(data)
        
        if min_val == max_val:
            padding = abs(min_val * buffer_ratio) if min_val != 0 else 0.1
            return (min_val - padding, max_val + padding)
        
        data_range = max_val - min_val
        padding = data_range * buffer_ratio
        
        lower_limit = min_val - padding if lower_bound is None else max(lower_bound, min_val - padding)
        upper_limit = max_val + padding
        
        return (lower_limit, upper_limit)
    
    @staticmethod
    def calculate_bar_layout(n_groups: int, n_items: int) -> Tuple[np.ndarray, float]:
        """
        Calculate bar chart layout parameters.
        
        Args:
            n_groups: Number of groups (x-axis positions)
            n_items: Number of items per group (different bars)
            
        Returns:
            Tuple of (x_positions, bar_width)
        """
        bar_width = PlotConfig.BAR_WIDTH_RATIO / n_items
        x_positions = np.arange(n_groups)
        return x_positions, bar_width
    
    @staticmethod
    def get_subplot_layout(n_plots: int) -> Tuple[int, int]:
        """
        Get optimal subplot layout for given number of plots.
        
        Args:
            n_plots: Number of subplots needed
            
        Returns:
            Tuple of (nrows, ncols)
        """
        if n_plots == 1:
            return 1, 1
        elif n_plots == 2:
            return 1, 2
        elif n_plots <= 4:
            return 2, 2
        elif n_plots <= 6:
            return 2, 3
        elif n_plots <= 9:
            return 3, 3
        else:
            # For more than 9 plots, use a grid that accommodates all
            ncols = int(np.ceil(np.sqrt(n_plots)))
            nrows = int(np.ceil(n_plots / ncols))
            return nrows, ncols


class MetricConfig:
    """Configuration for metric-specific plotting."""
    
    # Metric display names
    METRIC_NAMES = {
        'fertility': 'Fertility',
        'compression_ratio': 'Compression Ratio',
        'type_token_ratio': 'Type-Token Ratio',
        'vocabulary_utilization': 'Vocabulary Utilization',
        'token_length': 'Token Length',
        'avg_tokens_per_line': 'Tokens per Line',
        'unigram_entropy': 'Unigram Entropy',
        'avg_token_rank': 'Average Token Rank',
        'renyi_efficiency': 'Rényi Efficiency',
        'morphological_alignment': 'Morphological Alignment',
        'tokenizer_fairness_gini': 'Tokenizer Fairness (Gini)',
        'language_costs': 'Language Costs'
    }
    
    # Y-axis labels with units
    METRIC_YLABELS = {
        'fertility': 'Tokens per Unit',
        'compression_ratio': 'Compression Ratio',
        'type_token_ratio': 'Type-Token Ratio',
        'vocabulary_utilization': 'Vocabulary Utilization',
        'token_length': 'Length per Token',
        'avg_tokens_per_line': 'Tokens per Line',
        'unigram_entropy': 'Entropy (bits)',
        'avg_token_rank': 'Average Rank',
        'gini_coefficient': 'Gini Coefficient',
        'cost_ratio': 'Cost Ratio',
        'language_costs': 'Token Cost'
    }
    
    # Metrics that should have reference lines
    REFERENCE_LINES = {
        'compression_ratio': 1.0,  # Equal compression
        'type_token_ratio': None,
        'vocabulary_utilization': None,
        'gini_coefficient': None,
    }
    
    # Color schemes for different metric types
    COLOR_SCHEMES = {
        'basic': 'Set2',
        'information': 'viridis',
        'fairness': 'RdYlBu_r',
        'grouped': 'Set3'
    }
    
    @staticmethod
    def get_metric_ylabel(metric_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Get appropriate y-axis label for a metric.
        
        Args:
            metric_name: Name of the metric
            metadata: Optional metadata with normalization info
            
        Returns:
            Formatted y-axis label
        """
        if metadata:
            if metric_name == 'fertility':
                norm_method = metadata.get('normalization_method', 'units')
                return f"Tokens per {norm_method.title()[:-1]}"
            elif metric_name == 'compression_ratio':
                norm_method = metadata.get('normalization_method', 'units')
                return f"# {norm_method.title()} / # Tokens"
            elif metric_name == 'language_costs':
                cost_unit = metadata.get('cost_unit', 'tokens per unit')
                return f"Cost ({cost_unit.title()})"
        
        return MetricConfig.METRIC_YLABELS.get(metric_name, metric_name.replace('_', ' ').title())
    
    @staticmethod
    def get_metric_title(metric_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Get appropriate title for a metric plot.
        
        Args:
            metric_name: Name of the metric
            metadata: Optional metadata with normalization info
            
        Returns:
            Formatted plot title
        """
        base_title = MetricConfig.METRIC_NAMES.get(metric_name, metric_name.replace('_', ' ').title())
        
        if metadata and metric_name in ['fertility', 'compression_ratio']:
            norm_method = metadata.get('normalization_method', 'units')
            return f"{base_title} ({norm_method.title()})"
        
        return base_title