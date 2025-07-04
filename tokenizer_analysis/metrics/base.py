"""
Base metrics class with common statistical operations and patterns.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class BaseMetrics(ABC):
    """Base class for tokenizer metrics with common utilities."""
    
    def __init__(self, tokenizers: Dict[str, Any], tokenizer_names: Optional[List[str]] = None):
        """
        Initialize base metrics.
        
        Args:
            tokenizers: Dictionary mapping tokenizer names to tokenizer objects
            tokenizer_names: List of tokenizer names to analyze (default: all)
        """
        self.tokenizers = tokenizers
        self.tokenizer_names = tokenizer_names or list(tokenizers.keys())
        
    @staticmethod
    def compute_basic_stats(values: List[float]) -> Dict[str, float]:
        """Compute basic statistics for a list of values."""
        if not values:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0,
                'sum': 0
            }
            
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values),
            'sum': sum(values)
        }
    
    @staticmethod
    def aggregate_per_language_results(per_language_data: Dict[str, List[float]], 
                                     aggregation_func: str = 'mean') -> Dict[str, Dict[str, float]]:
        """
        Aggregate per-language results into summary statistics.
        
        Args:
            per_language_data: Dict mapping language codes to lists of values
            aggregation_func: How to aggregate ('mean', 'sum', 'median')
            
        Returns:
            Dict with per-language and global aggregated results
        """
        results = {'per_language': {}, 'global': {}}
        all_values = []
        
        for lang, values in per_language_data.items():
            if values:
                if aggregation_func == 'mean':
                    results['per_language'][lang] = np.mean(values)
                elif aggregation_func == 'sum':
                    results['per_language'][lang] = np.sum(values)
                elif aggregation_func == 'median':
                    results['per_language'][lang] = np.median(values)
                else:
                    raise ValueError(f"Unknown aggregation function: {aggregation_func}")
                
                all_values.extend(values)
        
        # Global aggregation
        if all_values:
            if aggregation_func == 'mean':
                results['global'] = np.mean(all_values)
            elif aggregation_func == 'sum':
                results['global'] = np.sum(all_values)
            elif aggregation_func == 'median':
                results['global'] = np.median(all_values)
        
        return results
    
    def compute_pairwise_ratios(self, values_dict: Dict[str, float], 
                              metric_name: str) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise ratios between tokenizers for a given metric.
        
        Args:
            values_dict: Dict mapping tokenizer names to metric values
            metric_name: Name of the metric for logging
            
        Returns:
            Dict with pairwise ratio results
        """
        results = {}
        tokenizer_names = list(values_dict.keys())
        
        for i, tok1 in enumerate(tokenizer_names):
            for j, tok2 in enumerate(tokenizer_names):
                if i != j:
                    pair_key = f"{tok1}_vs_{tok2}"
                    if values_dict[tok2] != 0:
                        ratio = values_dict[tok1] / values_dict[tok2]
                        results[pair_key] = {
                            'ratio': ratio,
                            'difference': values_dict[tok1] - values_dict[tok2],
                            'relative_change': (values_dict[tok1] - values_dict[tok2]) / values_dict[tok2]
                        }
                    else:
                        logger.warning(f"Division by zero in {metric_name} ratio for {pair_key}")
                        results[pair_key] = {
                            'ratio': float('inf') if values_dict[tok1] > 0 else 1.0,
                            'difference': values_dict[tok1],
                            'relative_change': float('inf') if values_dict[tok1] > 0 else 0.0
                        }
        
        return results
    
    def encode_texts_batch(self, language_texts: Dict[str, List[str]]) -> Dict[str, Dict[str, List[List[int]]]]:
        """
        Encode all texts for all tokenizers efficiently.
        
        Args:
            language_texts: Dict mapping language codes to lists of texts
            
        Returns:
            Dict mapping tokenizer names to language-encoded texts
        """
        from ..utils import encode_text
        
        all_encodings = {}
        for name in self.tokenizer_names:
            all_encodings[name] = {}
            for lang, texts in language_texts.items():
                if texts:  # Only process non-empty text lists
                    logger.info(f"Encoding {len(texts)} {lang} texts with {name}...")
                    
                    # Encode the texts
                    encoding_result = encode_text(self.tokenizers[name], texts)
                    token_ids = encoding_result["input_ids"]
                    tokens = encoding_result["tokens"]
                    
                    # Log example for first text in this batch
                    if len(texts) > 0 and len(token_ids) > 0:
                        example_text = texts[0]
                        example_token_ids = token_ids[0] if isinstance(token_ids[0], list) else token_ids
                        example_tokens = tokens[0] if isinstance(tokens[0], list) else tokens
                        
                        logger.info(f"  📝 Example for {name} on {lang}:")
                        logger.info(f"    Input: '{example_text[:100]}{'...' if len(example_text) > 100 else ''}'")
                        logger.info(f"    Token IDs: {example_token_ids[:20]}{'...' if len(example_token_ids) > 20 else ''}")
                        logger.info(f"    Tokens: {example_tokens[:20]}{'...' if len(example_tokens) > 20 else ''}")
                        logger.info(f"    Total tokens: {len(example_token_ids)}")
                    
                    # Store the token IDs for further processing
                    all_encodings[name][lang] = token_ids
                else:
                    all_encodings[name][lang] = []
        
        return all_encodings
    
    def filter_valid_data(self, language_texts: Dict[str, List[str]], 
                         all_encodings: Dict[str, Dict[str, List[List[int]]]]) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[List[int]]]]]:
        """
        Filter out languages/texts with no valid data.
        
        Args:
            language_texts: Original language texts
            all_encodings: All tokenizer encodings
            
        Returns:
            Filtered language texts and encodings
        """
        valid_languages = set()
        for lang, texts in language_texts.items():
            if texts and any(text.strip() for text in texts):
                valid_languages.add(lang)
        
        filtered_texts = {lang: texts for lang, texts in language_texts.items() if lang in valid_languages}
        filtered_encodings = {}
        
        for tok_name, lang_encodings in all_encodings.items():
            filtered_encodings[tok_name] = {lang: encodings for lang, encodings in lang_encodings.items() if lang in valid_languages}
        
        return filtered_texts, filtered_encodings
    
    @abstractmethod
    def compute(self, language_texts: Dict[str, List[str]], 
                all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """
        Compute the metric(s) for all tokenizers.
        
        Args:
            language_texts: Dict mapping language codes to lists of texts
            all_encodings: Pre-computed encodings (optional, will compute if None)
            
        Returns:
            Dict containing metric results
        """
        pass
    
    def print_results(self, results: Dict[str, Any], metric_name: str, print_pairwise: bool = False) -> None:
        """Print formatted results for a metric."""
        print(f"\n=== {metric_name.upper()} ===")
        
        # Print per-tokenizer results if available
        if 'per_tokenizer' in results:
            for tok_name, tok_results in results['per_tokenizer'].items():
                if isinstance(tok_results, dict):
                    if 'global' in tok_results:
                        print(f"{tok_name}: {tok_results['global']:.4f}")
                    else:
                        # Print first few key metrics
                        for key, value in list(tok_results.items())[:3]:
                            if isinstance(value, (int, float)):
                                print(f"{tok_name} {key}: {value:.4f}")
                else:
                    print(f"{tok_name}: {tok_results:.4f}")
        
        # Print pairwise comparisons if available
        if 'pairwise_comparisons' in results and print_pairwise:
            print("Pairwise comparisons:")
            for pair, comparison in results['pairwise_comparisons'].items():
                if 'ratio' in comparison:
                    print(f"  {pair}: ratio={comparison['ratio']:.3f}, change={comparison['relative_change']:+.1%}")