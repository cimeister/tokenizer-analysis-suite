"""
Simple visualization coordinator for tokenizer analysis results.
"""

from typing import Dict, List, Any, Optional
import os
import logging

from .plots import generate_all_plots

logger = logging.getLogger(__name__)


class TokenizerVisualizer:
    """Simple coordinator for tokenizer analysis visualization."""
    
    def __init__(self, tokenizer_names: List[str], save_dir: str = "results", 
                 show_global_lines: bool = True, per_language_plots: bool = False, 
                 faceted_plots: bool = False):
        """
        Initialize visualizer.
        
        Args:
            tokenizer_names: List of tokenizer names
            save_dir: Directory to save plots
            show_global_lines: Whether to show global average reference lines
            per_language_plots: Whether to generate per-language plots
            faceted_plots: Whether to generate faceted plots
        """
        self.tokenizer_names = tokenizer_names
        self.save_dir = save_dir
        self.show_global_lines = show_global_lines
        self.per_language_plots = per_language_plots
        self.faceted_plots = faceted_plots
        
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Initialized TokenizerVisualizer for {len(tokenizer_names)} tokenizers")
    
    def generate_all_plots(self, results: Dict[str, Any], print_pairwise: bool = False) -> None:
        """Generate all available plots for the results."""
        logger.info(f"Generating plots in {self.save_dir}")
        
        try:
            generate_all_plots(results, self.save_dir, self.tokenizer_names,
                             show_global_lines=self.show_global_lines,
                             per_language_plots=self.per_language_plots,
                             faceted_plots=self.faceted_plots)
            logger.info(f"All plots saved to {self.save_dir}")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            raise
    
    def plot_grouped_analysis(self, grouped_results: Dict[str, Dict[str, Any]], 
                             overall_results: Dict[str, Any] = None, 
                             reference_line_method: str = 'macro') -> None:
        """Generate plots for grouped analysis results."""
        if not grouped_results:
            logger.warning("No grouped results provided for plotting")
            return
        
        try:
            logger.info("Generating grouped analysis plots")
            generate_all_plots({}, self.save_dir, self.tokenizer_names,
                             grouped_results=grouped_results,
                             show_global_lines=self.show_global_lines,
                             per_language_plots=False,  # Not applicable for grouped plots
                             faceted_plots=False)       # Not applicable for grouped plots
            
            logger.info(f"Grouped plots saved to {self.save_dir}/grouped_plots")
        except Exception as e:
            logger.error(f"Error generating grouped plots: {e}")
            raise