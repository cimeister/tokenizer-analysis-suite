"""
Tokenizer Fairness Gini coefficient implementation.

This module implements the Tokenizer Fairness Gini (TFG) coefficient,
which measures how equitably a tokenizer treats different languages.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging

from .base import BaseMetrics, TokenizedDataProcessor
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider
from ..config import NormalizationConfig, TextNormalizer, DEFAULT_NORMALIZATION_CONFIG, LINES_CONFIG
from ..config.language_metadata import LanguageMetadata
from ..constants import MIN_LANGUAGES_FOR_GINI

logger = logging.getLogger(__name__)


class TokenizerGiniMetrics(BaseMetrics):
    """
    Implements Tokenizer Fairness Gini coefficient and related metrics.
    
    The TFG measures fairness by computing token costs per language and
    calculating the Gini coefficient of the distribution of these costs.
    """
    
    def __init__(self, input_provider: InputProvider, 
                 normalization_config: Optional[NormalizationConfig] = None,
                 language_metadata: Optional[LanguageMetadata] = None):
        super().__init__(input_provider)
        self.norm_config = LINES_CONFIG#normalization_config or DEFAULT_NORMALIZATION_CONFIG
        self.normalizer = TextNormalizer(self.norm_config)
        self.language_metadata = language_metadata
    
    def compute_tokenizer_fairness_gini(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute Tokenizer Fairness Gini (TFG) coefficient.
        
        The TFG is defined as:
        
        1. For each language ℓ, compute token cost on parallel corpus:
           c_ℓ = (number of tokens) / (number of raw bytes, characters or lines)
           
        2. Compute mean cost: μ = (1/n) * Σ c_ℓ
        
        3. Compute TFG:
           TFG = Σᵢ Σⱼ |c_i - c_j| / (2 * n² * μ)
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict containing TFG coefficients and related metrics for each tokenizer
        """
        
        results = {
            'per_tokenizer': {},
            'metadata': {
                'description': 'Tokenizer Fairness Gini coefficient measures equitable treatment across languages',
                'formula': 'TFG = Σᵢ Σⱼ |c_i - c_j| / (2 * n² * μ)',
                'interpretation': 'Lower values indicate more equitable treatment (0 = perfect equality)',
                'normalization_method': self.norm_config.method.value,
            }
        }
        
        # Extract all languages from the tokenized data
        all_languages = set()
        for tok_data in tokenized_data.values():
            for data in tok_data:
                all_languages.add(data.language)
        
        languages = list(all_languages)
        n_languages = len(languages)
        
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
                
            logger.info(f"Computing TFG for tokenizer: {tok_name}")
            
            tok_data = tokenized_data[tok_name]
            
            # Step 1: Compute token costs per language
            language_costs = {}
            total_costs = []
            
            # Group data by language
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
            
            for lang in languages:
                if lang not in lang_groups:
                    continue
                    
                lang_data = lang_groups[lang]
                
                # Aggregate tokens and normalization units for this language
                total_tokens = 0
                total_normalization_units = 0
                
                for data in lang_data:
                    if data.text and data.text.strip():  # Skip empty texts
                        total_tokens += len(data.tokens)
                        total_normalization_units += self.normalizer.get_normalization_count(data.text)
                
                if total_normalization_units > 0:
                    # Token cost: tokens per normalization unit
                    cost = total_tokens / total_normalization_units
                    language_costs[lang] = cost
                    total_costs.append(cost)
                    
                    logger.debug(f"  {lang}: {total_tokens} tokens / {total_normalization_units} {self.norm_config.method.value} = {cost:.4f}")
            
            if len(language_costs) < MIN_LANGUAGES_FOR_GINI:
                logger.warning(f"Insufficient language data for TFG calculation for {tok_name}")
                results['per_tokenizer'][tok_name] = {
                    'gini_coefficient': 0.0,
                    'mean_cost': 0.0,
                    'language_costs': language_costs,
                    'warning': 'Insufficient language data for meaningful TFG calculation'
                }
                continue
            
            # Step 2: Compute mean cost
            mu = np.mean(total_costs)
            
            # Step 3: Compute TFG using the exact formula
            # TFG = Σᵢ Σⱼ |c_i - c_j| / (2 * n² * μ)
            sum_absolute_differences = 0.0
            n = len(total_costs)
            
            for i in range(n):
                for j in range(n):
                    sum_absolute_differences += abs(total_costs[i] - total_costs[j])
            
            # Apply the TFG formula
            if mu > 0 and n > 0:
                tfg = sum_absolute_differences / (2 * n * n * mu)
            else:
                tfg = 0.0
            
            # Additional statistics for analysis
            min_cost = min(total_costs)
            max_cost = max(total_costs)
            std_cost = np.std(total_costs)
            
            # Compute cost ratios (max/min)
            cost_ratio = max_cost / min_cost if min_cost > 0 else float('inf')
            
            # Identify most and least efficient languages
            sorted_langs = sorted(language_costs.items(), key=lambda x: x[1])
            most_efficient = sorted_langs[0]  # Lowest cost (most efficient)
            least_efficient = sorted_langs[-1]  # Highest cost (least efficient)
            
            results['per_tokenizer'][tok_name] = {
                'gini_coefficient': tfg,
                'mean_cost': mu,
                'std_cost': std_cost,
                'min_cost': min_cost,
                'max_cost': max_cost,
                'cost_ratio': cost_ratio,
                'language_costs': language_costs,
                'most_efficient_language': most_efficient,
                'least_efficient_language': least_efficient,
                'num_languages': len(language_costs),
                'sorted_language_costs': sorted_langs
            }
            
            logger.info(f"  TFG: {tfg:.4f}, Mean cost: {mu:.4f}, Cost ratio: {cost_ratio:.2f}")
        
        return results
    
    def compute_lorenz_curve_data(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute Lorenz curve data for visualizing tokenizer fairness.
        
        The Lorenz curve shows the cumulative distribution of token costs,
        useful for visualizing inequality across languages.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict containing Lorenz curve data for each tokenizer
        """
        
        results = {
            'per_tokenizer': {},
            'metadata': {
                'description': 'Lorenz curve data for visualizing tokenizer fairness',
                'x_axis': 'Cumulative proportion of languages (sorted by efficiency)',
                'y_axis': 'Cumulative proportion of total token cost'
            }
        }
        
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
                
            tok_data = tokenized_data[tok_name]
            
            # Get token costs per language
            language_costs = {}
            
            # Group data by language
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
            
            for lang, lang_data in lang_groups.items():
                total_tokens = 0
                total_normalization_units = 0
                
                for data in lang_data:
                    if data.text and data.text.strip():
                        total_tokens += len(data.tokens)
                        total_normalization_units += self.normalizer.get_normalization_count(data.text)
                
                if total_normalization_units > 0:
                    cost = total_tokens / total_normalization_units
                    language_costs[lang] = cost
            
            if len(language_costs) < MIN_LANGUAGES_FOR_GINI:
                results['per_tokenizer'][tok_name] = {
                    'warning': 'Insufficient data for Lorenz curve'
                }
                continue
            
            # Sort languages by cost (most efficient first)
            sorted_items = sorted(language_costs.items(), key=lambda x: x[1])
            sorted_languages = [item[0] for item in sorted_items]
            sorted_costs = [item[1] for item in sorted_items]
            
            # Compute cumulative proportions
            n_languages = len(sorted_costs)
            total_cost = sum(sorted_costs)
            
            # X-axis: cumulative proportion of languages
            x_values = [0.0]  # Start at 0
            x_values.extend([(i + 1) / n_languages for i in range(n_languages)])
            
            # Y-axis: cumulative proportion of total cost
            y_values = [0.0]  # Start at 0
            cumulative_cost = 0.0
            for cost in sorted_costs:
                cumulative_cost += cost
                y_values.append(cumulative_cost / total_cost)
            
            # Perfect equality line (diagonal)
            equality_line = x_values.copy()
            
            results['per_tokenizer'][tok_name] = {
                'x_values': x_values,
                'y_values': y_values,
                'equality_line': equality_line,
                'sorted_languages': sorted_languages,
                'sorted_costs': sorted_costs,
                'total_cost': total_cost,
                'n_languages': n_languages
            }
        
        return results
    
    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """
        Compute all Gini-related metrics.
        
        Args:
            tokenized_data: Optional dict mapping tokenizer names to TokenizedData lists.
                          If None, will use input_provider's data.
            
        Returns:
            Dict containing all Gini metrics and Lorenz curve data
        """
        if tokenized_data is None:
            tokenized_data = self.get_tokenized_data()
            
        results = {}
        
        # Compute TFG
        tfg_results = self.compute_tokenizer_fairness_gini(tokenized_data)
        results['tokenizer_fairness_gini'] = tfg_results
        
        # Compute Lorenz curve data
        lorenz_results = self.compute_lorenz_curve_data(tokenized_data)
        results['lorenz_curve_data'] = lorenz_results
        
        return results
    
    # Convenience methods for common groupings
    def compute_by_script_family(self, language_texts: Dict[str, List[str]], 
                                all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """Compute TFG metrics grouped by script family."""
        return self.compute_by_groups(language_texts, all_encodings, 'script_families', self.compute)
    
    def compute_by_resource_level(self, language_texts: Dict[str, List[str]], 
                                 all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """Compute TFG metrics grouped by resource level."""
        return self.compute_by_groups(language_texts, all_encodings, 'resource_levels', self.compute)