"""
Factory for creating and managing plot renderers.
"""

from typing import Dict, List, Any, Optional
import logging

from .renderers import (
    PlotRenderer, BasicMetricsRenderer, InformationRenderer, 
    FairnessRenderer, GroupedRenderer
)
from .plot_config import PlotConfig

logger = logging.getLogger(__name__)


class PlotFactory:
    """Factory for creating and managing plot renderers."""
    
    # Registry of available renderer types
    RENDERER_REGISTRY = {
        'basic': BasicMetricsRenderer,
        'information': InformationRenderer,
        'fairness': FairnessRenderer,
        'grouped': GroupedRenderer
    }
    
    def __init__(self, tokenizer_names: List[str], save_dir: str):
        """
        Initialize plot factory.
        
        Args:
            tokenizer_names: List of tokenizer names
            save_dir: Directory to save plots
        """
        self.tokenizer_names = tokenizer_names
        self.save_dir = save_dir
        self._renderers: Dict[str, PlotRenderer] = {}
        
        # Setup plotting style
        PlotConfig.setup_style()
    
    def get_renderer(self, renderer_type: str) -> PlotRenderer:
        """
        Get renderer instance for specified type.
        
        Args:
            renderer_type: Type of renderer ('basic', 'information', 'fairness', 'grouped')
            
        Returns:
            Renderer instance
            
        Raises:
            ValueError: If renderer type is not supported
        """
        if renderer_type not in self.RENDERER_REGISTRY:
            raise ValueError(f"Unsupported renderer type: {renderer_type}. "
                           f"Available types: {list(self.RENDERER_REGISTRY.keys())}")
        
        # Create renderer if not already cached
        if renderer_type not in self._renderers:
            renderer_class = self.RENDERER_REGISTRY[renderer_type]
            self._renderers[renderer_type] = renderer_class(self.tokenizer_names, self.save_dir)
            logger.debug(f"Created {renderer_type} renderer")
        
        return self._renderers[renderer_type]
    
    def render_all(self, results: Dict[str, Any], grouped_results: Optional[Dict[str, Dict[str, Any]]] = None,
                  enable_pairwise: bool = False) -> None:
        """
        Render all available plots using appropriate renderers.
        
        Args:
            results: Analysis results dictionary
            grouped_results: Optional grouped analysis results
            enable_pairwise: Whether to enable pairwise comparisons
        """
        logger.info(f"Generating plots in {self.save_dir}")
        
        try:
            # Render basic metrics
            basic_renderer = self.get_renderer('basic')
            basic_renderer.render(results)
            
            # Render information-theoretic metrics
            info_renderer = self.get_renderer('information')
            info_renderer.render(results)
            
            # Render fairness metrics
            fairness_renderer = self.get_renderer('fairness')
            fairness_renderer.render(results)
            
            # Render grouped analysis if available
            if grouped_results:
                grouped_renderer = self.get_renderer('grouped')
                grouped_renderer.render(grouped_results)
            
            # Render pairwise comparisons if enabled
            if enable_pairwise:
                self._render_pairwise_comparisons(results)
            
            logger.info(f"All plots saved to {self.save_dir}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            raise
    
    def render_specific(self, results: Dict[str, Any], renderer_types: List[str]) -> None:
        """
        Render plots using specific renderer types.
        
        Args:
            results: Analysis results dictionary
            renderer_types: List of renderer types to use
        """
        for renderer_type in renderer_types:
            try:
                renderer = self.get_renderer(renderer_type)
                if renderer_type == 'grouped':
                    # Grouped renderer needs special handling
                    logger.warning("Grouped renderer requires grouped_results parameter")
                    continue
                renderer.render(results)
                logger.info(f"Rendered {renderer_type} plots")
            except Exception as e:
                logger.error(f"Error rendering {renderer_type} plots: {e}")
                continue
    
    def _render_pairwise_comparisons(self, results: Dict[str, Any]) -> None:
        """
        Render pairwise comparison plots.
        
        Args:
            results: Analysis results dictionary
        """
        if len(self.tokenizer_names) < 2:
            return
        
        # TODO: Implement pairwise comparison renderer
        # This would be a specialized renderer for pairwise comparisons
        logger.info("Pairwise comparison rendering not yet implemented")
    
    def get_available_renderers(self) -> List[str]:
        """
        Get list of available renderer types.
        
        Returns:
            List of renderer type names
        """
        return list(self.RENDERER_REGISTRY.keys())
    
    def validate_results(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate results for different renderer types.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Dictionary mapping renderer types to validation status
        """
        validation_status = {}
        
        # Basic metrics validation
        basic_metrics = ['fertility', 'compression_ratio', 'token_length']
        validation_status['basic'] = any(metric in results for metric in basic_metrics)
        
        # Information metrics validation
        info_metrics = ['type_token_ratio', 'vocabulary_utilization', 'renyi_efficiency', 'avg_tokens_per_line']
        validation_status['information'] = any(metric in results for metric in info_metrics)
        
        # Fairness metrics validation
        fairness_metrics = ['tokenizer_fairness_gini', 'lorenz_curve_data', 'morphological_alignment']
        validation_status['fairness'] = any(metric in results for metric in fairness_metrics)
        
        # Grouped validation would require grouped_results
        validation_status['grouped'] = False
        
        return validation_status
    
    def clear_cache(self) -> None:
        """Clear cached renderer instances."""
        self._renderers.clear()
        logger.debug("Cleared renderer cache")


class RenderingError(Exception):
    """Custom exception for rendering errors."""
    pass


def create_plot_factory(tokenizer_names: List[str], save_dir: str) -> PlotFactory:
    """
    Convenience function to create a plot factory.
    
    Args:
        tokenizer_names: List of tokenizer names
        save_dir: Directory to save plots
        
    Returns:
        PlotFactory instance
    """
    return PlotFactory(tokenizer_names, save_dir)