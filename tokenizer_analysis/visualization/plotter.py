"""
Refactored visualization utilities for tokenizer analysis results.
"""

from typing import Dict, List, Any, Optional
import os
import logging

from .plot_factory import PlotFactory
from .plot_config import PlotConfig

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
        
        # Initialize plot factory
        self.plot_factory = PlotFactory(tokenizer_names, save_dir)
        
        logger.info(f"Initialized TokenizerVisualizer with {len(tokenizer_names)} tokenizers")
    
    def generate_all_plots(self, results: Dict[str, Any], print_pairwise: bool = False) -> None:
        """
        Generate all available plots for the results.
        
        Args:
            results: Analysis results dictionary
            print_pairwise: Whether to generate pairwise comparison plots
        """
        logger.info(f"Generating plots in {self.save_dir}")
        
        try:
            self.plot_factory.render_all(results, enable_pairwise=print_pairwise)
            logger.info(f"All plots saved to {self.save_dir}")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            raise
    
    def plot_grouped_analysis(self, grouped_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Generate plots for grouped analysis results.
        
        Args:
            grouped_results: Results grouped by categories (e.g., script families, resource levels)
        """
        if not grouped_results:
            logger.warning("No grouped results provided for plotting")
            return
        
        try:
            grouped_renderer = self.plot_factory.get_renderer('grouped')
            grouped_renderer.render(grouped_results)
            logger.info("Grouped analysis plots generated successfully")
        except Exception as e:
            logger.error(f"Error generating grouped plots: {e}")
            raise
    
    def plot_basic_metrics_comparison(self, results: Dict[str, Any]) -> None:
        """
        Plot basic tokenization metrics comparison.
        
        Args:
            results: Analysis results dictionary
        """
        try:
            basic_renderer = self.plot_factory.get_renderer('basic')
            basic_renderer.render(results)
            logger.info("Basic metrics plots generated successfully")
        except Exception as e:
            logger.error(f"Error generating basic metrics plots: {e}")
            raise
    
    def plot_information_theoretic_metrics(self, results: Dict[str, Any]) -> None:
        """
        Plot information-theoretic metrics.
        
        Args:
            results: Analysis results dictionary
        """
        try:
            info_renderer = self.plot_factory.get_renderer('information')
            info_renderer.render(results)
            logger.info("Information-theoretic plots generated successfully")
        except Exception as e:
            logger.error(f"Error generating information-theoretic plots: {e}")
            raise
    
    def plot_unigram_distribution_metrics(self, results: Dict[str, Any]) -> None:
        """
        Plot unigram distribution metrics.
        
        Args:
            results: Analysis results dictionary
        """
        try:
            info_renderer = self.plot_factory.get_renderer('information')
            info_renderer.render(results)
            logger.info("Unigram distribution plots generated successfully")
        except Exception as e:
            logger.error(f"Error generating unigram distribution plots: {e}")
            raise
    
    def plot_renyi_entropy_curves(self, results: Dict[str, Any]) -> None:
        """
        Plot Rényi entropy curves for different alpha values.
        
        Args:
            results: Analysis results dictionary
        """
        try:
            info_renderer = self.plot_factory.get_renderer('information')
            info_renderer.render(results)
            logger.info("Rényi entropy plots generated successfully")
        except Exception as e:
            logger.error(f"Error generating Rényi entropy plots: {e}")
            raise
    
    def plot_morphological_metrics(self, results: Dict[str, Any]) -> None:
        """
        Plot morphological alignment metrics.
        
        Args:
            results: Analysis results dictionary
        """
        try:
            fairness_renderer = self.plot_factory.get_renderer('fairness')
            fairness_renderer.render(results)
            logger.info("Morphological metrics plots generated successfully")
        except Exception as e:
            logger.error(f"Error generating morphological metrics plots: {e}")
            raise
    
    def plot_pairwise_comparisons(self, results: Dict[str, Any]) -> None:
        """
        Plot pairwise comparison matrices for key metrics.
        
        Args:
            results: Analysis results dictionary
        """
        try:
            # TODO: Implement dedicated pairwise renderer
            logger.warning("Pairwise comparison plots not yet implemented in new architecture")
        except Exception as e:
            logger.error(f"Error generating pairwise comparison plots: {e}")
            raise
    
    def plot_per_language_analysis(self, results: Dict[str, Any]) -> None:
        """
        Plot per-language performance analysis.
        
        Args:
            results: Analysis results dictionary
        """
        try:
            basic_renderer = self.plot_factory.get_renderer('basic')
            basic_renderer.render(results)
            logger.info("Per-language analysis plots generated successfully")
        except Exception as e:
            logger.error(f"Error generating per-language analysis plots: {e}")
            raise
    
    def plot_fertility_comparison(self, results: Dict[str, Any]) -> None:
        """
        Create a dedicated plot for fertility metric.
        
        Args:
            results: Analysis results dictionary
        """
        try:
            basic_renderer = self.plot_factory.get_renderer('basic')
            basic_renderer.render(results)
            logger.info("Fertility comparison plots generated successfully")
        except Exception as e:
            logger.error(f"Error generating fertility comparison plots: {e}")
            raise
    
    def plot_tokenizer_fairness_gini(self, results: Dict[str, Any]) -> None:
        """
        Plot Tokenizer Fairness Gini coefficient and related metrics.
        
        Args:
            results: Analysis results dictionary
        """
        try:
            fairness_renderer = self.plot_factory.get_renderer('fairness')
            fairness_renderer.render(results)
            logger.info("Tokenizer fairness plots generated successfully")
        except Exception as e:
            logger.error(f"Error generating tokenizer fairness plots: {e}")
            raise
    
    def plot_lorenz_curves(self, results: Dict[str, Any]) -> None:
        """
        Plot Lorenz curves for tokenizer fairness visualization.
        
        Args:
            results: Analysis results dictionary
        """
        try:
            fairness_renderer = self.plot_factory.get_renderer('fairness')
            fairness_renderer.render(results)
            logger.info("Lorenz curves plots generated successfully")
        except Exception as e:
            logger.error(f"Error generating Lorenz curves plots: {e}")
            raise
    
    def validate_results(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate results for plotting.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Dictionary mapping renderer types to validation status
        """
        return self.plot_factory.validate_results(results)
    
    def get_available_renderers(self) -> List[str]:
        """
        Get list of available renderer types.
        
        Returns:
            List of renderer type names
        """
        return self.plot_factory.get_available_renderers()
    
    def clear_cache(self) -> None:
        """Clear cached renderer instances."""
        self.plot_factory.clear_cache()

