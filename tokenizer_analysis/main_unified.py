"""
Unified main module supporting both raw and pre-tokenized input modes.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

from .core.input_types import TokenizedData, InputSpecification, TokenizerProtocol
from .core.input_providers import InputProvider, create_input_provider
from .core.input_utils import create_simple_specifications, InputValidator
from .metrics.base_unified import BaseMetrics
from .metrics.basic_unified import BasicTokenizationMetrics
from .metrics.information_theoretic import InformationTheoreticMetrics
from .metrics.gini import TokenizerGiniMetrics
from .metrics.morphological import MorphologicalMetrics
from .visualization import TokenizerVisualizer
from .config import NormalizationConfig, DEFAULT_NORMALIZATION_CONFIG
from .config.language_metadata import LanguageMetadata

logger = logging.getLogger(__name__)


class UnifiedTokenizerAnalyzer:
    """
    Unified tokenizer analyzer supporting both raw and pre-tokenized inputs.
    
    This class provides a clean interface for tokenizer analysis using the new
    TokenizedData format without any legacy compatibility.
    """
    
    def __init__(self, 
                 input_provider: InputProvider,
                 normalization_config: Optional[NormalizationConfig] = None,
                 language_metadata: Optional[LanguageMetadata] = None,
                 plot_save_dir: str = "tokenizer_analysis_plots",
                 morphological_config: Optional[Dict[str, str]] = None):
        """
        Initialize unified analyzer.
        
        Args:
            input_provider: InputProvider instance with tokenized data
            normalization_config: Configuration for normalization method
            language_metadata: Optional language metadata for grouping
            plot_save_dir: Directory to save plots
            morphological_config: Optional morphological dataset configuration
        """
        # Validate input provider
        validation_report = InputValidator.validate_input_provider(input_provider)
        if not validation_report['valid']:
            logger.error("Input provider validation failed:")
            for error in validation_report['errors']:
                logger.error(f"  - {error}")
            raise ValueError("Invalid input provider configuration")
        
        self.input_provider = input_provider
        self.tokenizer_names = input_provider.get_tokenizer_names()
        self.norm_config = normalization_config or DEFAULT_NORMALIZATION_CONFIG
        self.language_metadata = language_metadata
        self.plot_save_dir = plot_save_dir
        
        # Initialize metrics classes
        self.basic_metrics = BasicTokenizationMetrics(
            input_provider, normalization_config, language_metadata
        )
        
        # Initialize information-theoretic metrics
        self.info_metrics = InformationTheoreticMetrics(
            input_provider, normalization_config=normalization_config, language_metadata=language_metadata
        )
        
        # Initialize Gini metrics
        self.gini_metrics = TokenizerGiniMetrics(
            input_provider, normalization_config=normalization_config, language_metadata=language_metadata
        )
        
        # Initialize morphological metrics if config provided
        self.morphological_metrics = None
        if morphological_config:
            self.morphological_metrics = MorphologicalMetrics(
                input_provider, morphological_config=morphological_config
            )
        
        # Initialize visualizer
        self.visualizer = TokenizerVisualizer(self.tokenizer_names, plot_save_dir)
        
        logger.info(f"Initialized unified analyzer with {len(self.tokenizer_names)} tokenizers: {self.tokenizer_names}")
        for name in self.tokenizer_names:
            vocab_size = self.input_provider.get_vocab_size(name)
            logger.info(f"  {name}: {vocab_size} tokens")
    
    def run_analysis(self,
                    save_plots: bool = True,
                    include_morphological: bool = False,  # Disabled until updated
                    verbose: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive tokenizer analysis.
        
        Args:
            save_plots: Whether to generate and save plots
            include_morphological: Whether to include morphological analysis (not yet implemented)
            verbose: Whether to print detailed results
            
        Returns:
            Analysis results dictionary
        """
        logger.info("Starting unified tokenizer analysis...")
        
        tokenized_data = self.input_provider.get_tokenized_data()
        languages = self.input_provider.get_languages()
        
        logger.info(f"Analyzing {len(languages)} languages: {languages}")
        logger.info(f"Tokenizers: {self.tokenizer_names}")
        
        results = {}
        
        # Run basic tokenization metrics
        logger.info("Computing basic tokenization metrics...")
        basic_results = self.basic_metrics.compute(tokenized_data)
        results.update(basic_results)
        
        if verbose:
            self._print_basic_results(basic_results)
        
        # Run information-theoretic metrics
        logger.info("Computing information-theoretic metrics...")
        info_results = self.info_metrics.compute(tokenized_data)
        results.update(info_results)
        
        # Run Gini metrics
        logger.info("Computing Gini metrics...")
        gini_results = self.gini_metrics.compute(tokenized_data)
        results.update(gini_results)
        
        # Run morphological metrics if available
        if self.morphological_metrics and include_morphological:
            logger.info("Computing morphological metrics...")
            morphological_results = self.morphological_metrics.compute(tokenized_data)
            results.update(morphological_results)
        
        # Generate plots
        if save_plots:
            logger.info("Generating plots...")
            self.visualizer.generate_all_plots(results, print_pairwise=False)
        
        logger.info("Analysis completed successfully!")
        return results
    
    def run_grouped_analysis(self,
                           group_by: Union[str, List[str]] = ['script_families', 'resource_levels'],
                           save_plots: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Run analysis grouped by language categories.
        
        Args:
            group_by: Group type(s) to analyze by
            save_plots: Whether to generate grouped plots
            
        Returns:
            Dictionary mapping group types to group analysis results
        """
        if not self.language_metadata:
            raise ValueError("Language metadata required for grouped analysis")
        
        if isinstance(group_by, str):
            group_by = [group_by]
        
        grouped_results = {}
        
        for group_type in group_by:
            logger.info(f"Running grouped analysis by {group_type}")
            
            if group_type not in self.language_metadata.analysis_groups:
                logger.warning(f"Group type {group_type} not found in language metadata")
                continue
            
            group_results = {}
            
            for group_name, group_languages in self.language_metadata.analysis_groups[group_type].items():
                logger.info(f"Analyzing group: {group_name}")
                
                # Filter tokenized data to this group
                filtered_data = self._filter_data_by_languages(group_languages)
                
                if not filtered_data:
                    logger.warning(f"No data found for group {group_name}")
                    continue
                
                # Run analysis on filtered data
                group_results[group_name] = self.basic_metrics.compute(filtered_data)
            
            grouped_results[group_type] = group_results
        
        # Generate grouped plots
        if save_plots and grouped_results:
            logger.info("Generating grouped plots...")
            self.visualizer.plot_grouped_analysis(grouped_results)
        
        return grouped_results
    
    def _filter_data_by_languages(self, target_languages: List[str]) -> Dict[str, List[TokenizedData]]:
        """Filter tokenized data to include only specified languages."""
        all_data = self.input_provider.get_tokenized_data()
        filtered_data = {}
        
        for tok_name, data_list in all_data.items():
            filtered_list = [data for data in data_list if data.language in target_languages]
            if filtered_list:
                filtered_data[tok_name] = filtered_list
        
        return filtered_data
    
    def _print_basic_results(self, results: Dict[str, Any]):
        """Print basic metrics results."""
        print("\n" + "="*60)
        print("BASIC TOKENIZATION METRICS RESULTS")
        print("="*60)
        
        # Print fertility results
        if 'fertility' in results:
            fertility_data = results['fertility']
            metadata = fertility_data.get('metadata', {})
            norm_method = metadata.get('normalization_method', 'units')
            
            print(f"\n📊 FERTILITY ANALYSIS ({norm_method})")
            print("-" * 40)
            
            for tok_name in self.tokenizer_names:
                if tok_name in fertility_data['per_tokenizer']:
                    global_stats = fertility_data['per_tokenizer'][tok_name]['global']
                    mean_fertility = global_stats.get('mean', 0.0)
                    std_fertility = global_stats.get('std', 0.0)
                    print(f"{tok_name:20}: {mean_fertility:.3f} ± {std_fertility:.3f} tokens/{norm_method[:-1]}")
        
        # Print token length results
        if 'token_length' in results:
            print(f"\n📏 TOKEN LENGTH ANALYSIS")
            print("-" * 40)
            
            for tok_name in self.tokenizer_names:
                if tok_name in results['token_length']['per_tokenizer']:
                    char_stats = results['token_length']['per_tokenizer'][tok_name]['character_length']
                    mean_length = char_stats.get('mean', 0.0)
                    std_length = char_stats.get('std', 0.0)
                    print(f"{tok_name:20}: {mean_length:.2f} ± {std_length:.2f} chars/token")
        
        # Print vocabulary utilization
        if 'vocabulary_utilization' in results:
            print(f"\n📚 VOCABULARY UTILIZATION")
            print("-" * 40)
            
            for tok_name in self.tokenizer_names:
                if tok_name in results['vocabulary_utilization']['per_tokenizer']:
                    util_data = results['vocabulary_utilization']['per_tokenizer'][tok_name]
                    utilization = util_data.get('global_utilization', 0.0)
                    used_tokens = util_data.get('global_used_tokens', 0)
                    vocab_size = util_data.get('global_vocab_size', 0)
                    print(f"{tok_name:20}: {utilization:.1%} ({used_tokens:,}/{vocab_size:,} tokens)")
        
        # Print type-token ratio
        if 'type_token_ratio' in results:
            print(f"\n🔤 TYPE-TOKEN RATIO")
            print("-" * 40)
            
            for tok_name in self.tokenizer_names:
                if tok_name in results['type_token_ratio']['per_tokenizer']:
                    ttr_data = results['type_token_ratio']['per_tokenizer'][tok_name]
                    ttr = ttr_data.get('global_ttr', 0.0)
                    types = ttr_data.get('global_types', 0)
                    tokens = ttr_data.get('global_tokens', 0)
                    print(f"{tok_name:20}: {ttr:.4f} ({types:,} types / {tokens:,} tokens)")
        
        print("\n" + "="*60)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analysis configuration and capabilities."""
        return {
            'tokenizer_names': self.tokenizer_names,
            'num_tokenizers': len(self.tokenizer_names),
            'languages': self.input_provider.get_languages(),
            'num_languages': len(self.input_provider.get_languages()),
            'vocab_sizes': {name: self.input_provider.get_vocab_size(name) for name in self.tokenizer_names},
            'normalization_method': self.norm_config.method.value,
            'has_language_metadata': self.language_metadata is not None,
            'analysis_groups': list(self.language_metadata.analysis_groups.keys()) if self.language_metadata else [],
            'plot_save_dir': self.plot_save_dir
        }


# Convenience functions for creating analyzers from different input types

def create_analyzer_from_raw_inputs(tokenizer_configs: Dict[str, Dict],
                                   language_texts: Dict[str, Union[str, List[str]]],
                                   **kwargs) -> UnifiedTokenizerAnalyzer:
    """
    Create analyzer from raw tokenizer configs and texts.
    
    Args:
        tokenizer_configs: Dict mapping tokenizer names to configs
        language_texts: Dict mapping languages to texts (strings or lists of strings)
        **kwargs: Additional arguments for UnifiedTokenizerAnalyzer
        
    Returns:
        UnifiedTokenizerAnalyzer instance
    """
    from .utils import load_tokenizer_from_config
    
    # Load tokenizers
    tokenizers = {}
    for name, config in tokenizer_configs.items():
        logger.info(f"Loading tokenizer: {name}")
        tokenizers[name] = load_tokenizer_from_config(config)
    
    # Create specifications
    tokenizer_text_pairs = {}
    for name, tokenizer in tokenizers.items():
        tokenizer_text_pairs[name] = (tokenizer, language_texts)
    
    specifications = create_simple_specifications(tokenizer_text_pairs)
    input_provider = create_input_provider(specifications)
    
    return UnifiedTokenizerAnalyzer(input_provider, **kwargs)


def create_analyzer_from_tokenized_data(tokenized_data: Dict[str, List[TokenizedData]],
                                       vocabularies: Dict[str, Union[int, TokenizerProtocol]],
                                       **kwargs) -> UnifiedTokenizerAnalyzer:
    """
    Create analyzer from pre-tokenized data.
    
    Args:
        tokenized_data: Dict mapping tokenizer names to TokenizedData lists
        vocabularies: Dict mapping tokenizer names to vocab sizes or tokenizer objects
        **kwargs: Additional arguments for UnifiedTokenizerAnalyzer
        
    Returns:
        UnifiedTokenizerAnalyzer instance
    """
    from .core.input_utils import SimpleVocabulary
    
    specifications = {}
    for tok_name, data_list in tokenized_data.items():
        # Create vocabulary provider
        if tok_name in vocabularies:
            vocab = vocabularies[tok_name]
            if isinstance(vocab, int):
                vocab = SimpleVocabulary(vocab)
            elif not hasattr(vocab, 'vocab_size'):
                raise ValueError(f"Invalid vocabulary for {tok_name}: must be int or have vocab_size property")
        else:
            # Estimate vocab size from data
            max_token_id = max(max(data.tokens) for data in data_list if data.tokens)
            vocab = SimpleVocabulary(max_token_id + 1)
        
        spec = InputSpecification(
            tokenizer_name=tok_name,
            vocabulary=vocab,
            tokenized_data=data_list
        )
        specifications[tok_name] = spec
    
    input_provider = create_input_provider(specifications)
    return UnifiedTokenizerAnalyzer(input_provider, **kwargs)


def create_analyzer_from_input_provider(input_provider: InputProvider,
                                       **kwargs) -> UnifiedTokenizerAnalyzer:
    """
    Create analyzer from existing InputProvider.
    
    Args:
        input_provider: InputProvider instance
        **kwargs: Additional arguments for UnifiedTokenizerAnalyzer
        
    Returns:
        UnifiedTokenizerAnalyzer instance
    """
    return UnifiedTokenizerAnalyzer(input_provider, **kwargs)