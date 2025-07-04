"""
Tokenizer Fairness Gini coefficient implementation.

This module implements the Tokenizer Fairness Gini (TFG) coefficient,
which measures how equitably a tokenizer treats different languages.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging

from .base import BaseMetrics

logger = logging.getLogger(__name__)


class TokenizerGiniMetrics(BaseMetrics):
    """
    Implements Tokenizer Fairness Gini coefficient and related metrics.
    
    The TFG measures fairness by computing token costs per language and
    calculating the Gini coefficient of the distribution of these costs.
    """
    
    def compute_tokenizer_fairness_gini(self, language_texts: Dict[str, List[str]], 
                                       all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """
        Compute Tokenizer Fairness Gini (TFG) coefficient.
        
        The TFG is defined as:
        
        1. For each language ℓ, compute token cost on parallel corpus:
           c_ℓ = (number of tokens) / (number of raw bytes)
           
        2. Compute mean cost: μ = (1/n) * Σ c_ℓ
        
        3. Compute TFG:
           TFG = Σᵢ Σⱼ |c_i - c_j| / (2 * n² * μ)
        
        Args:
            language_texts: Dict mapping language codes to lists of texts
            all_encodings: Pre-computed encodings (optional)
            
        Returns:
            Dict containing TFG coefficients and related metrics for each tokenizer
        """
        if all_encodings is None:
            all_encodings = self.encode_texts_batch(language_texts)
        
        language_texts, all_encodings = self.filter_valid_data(language_texts, all_encodings)
        
        results = {
            'per_tokenizer': {},
            'metadata': {
                'description': 'Tokenizer Fairness Gini coefficient measures equitable treatment across languages',
                'formula': 'TFG = Σᵢ Σⱼ |c_i - c_j| / (2 * n² * μ)',
                'interpretation': 'Lower values indicate more equitable treatment (0 = perfect equality)'
            }
        }
        
        languages = list(language_texts.keys())
        n_languages = len(languages)
        
        for tok_name in self.tokenizer_names:
            logger.info(f"Computing TFG for tokenizer: {tok_name}")
            
            # Step 1: Compute token costs per language
            language_costs = {}
            total_costs = []
            
            for lang in languages:
                if lang not in all_encodings[tok_name] or not language_texts[lang]:
                    continue
                
                # Aggregate tokens and bytes for this language
                total_tokens = 0
                total_bytes = 0
                
                for text, tokens in zip(language_texts[lang], all_encodings[tok_name][lang]):
                    if text.strip():  # Skip empty texts
                        total_tokens += len(tokens)
                        total_bytes += len(text.encode('utf-8'))
                
                if total_bytes > 0:
                    # Token cost: tokens per byte
                    cost = total_tokens / total_bytes
                    language_costs[lang] = cost
                    total_costs.append(cost)
                    
                    logger.debug(f"  {lang}: {total_tokens} tokens / {total_bytes} bytes = {cost:.4f}")
            
            if len(language_costs) < 2:
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
    
    def compute_lorenz_curve_data(self, language_texts: Dict[str, List[str]], 
                                 all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """
        Compute Lorenz curve data for visualizing tokenizer fairness.
        
        The Lorenz curve shows the cumulative distribution of token costs,
        useful for visualizing inequality across languages.
        
        Args:
            language_texts: Dict mapping language codes to lists of texts
            all_encodings: Pre-computed encodings (optional)
            
        Returns:
            Dict containing Lorenz curve data for each tokenizer
        """
        if all_encodings is None:
            all_encodings = self.encode_texts_batch(language_texts)
        
        language_texts, all_encodings = self.filter_valid_data(language_texts, all_encodings)
        
        results = {
            'per_tokenizer': {},
            'metadata': {
                'description': 'Lorenz curve data for visualizing tokenizer fairness',
                'x_axis': 'Cumulative proportion of languages (sorted by efficiency)',
                'y_axis': 'Cumulative proportion of total token cost'
            }
        }
        
        for tok_name in self.tokenizer_names:
            # Get token costs per language
            language_costs = {}
            
            for lang, texts in language_texts.items():
                if lang not in all_encodings[tok_name] or not texts:
                    continue
                
                total_tokens = 0
                total_bytes = 0
                
                for text, tokens in zip(texts, all_encodings[tok_name][lang]):
                    if text.strip():
                        total_tokens += len(tokens)
                        total_bytes += len(text.encode('utf-8'))
                
                if total_bytes > 0:
                    cost = total_tokens / total_bytes
                    language_costs[lang] = cost
            
            if len(language_costs) < 2:
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
    
    def compute(self, language_texts: Dict[str, List[str]], 
                all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """
        Compute all Gini-related metrics.
        
        Args:
            language_texts: Dict mapping language codes to lists of texts
            all_encodings: Pre-computed encodings (optional)
            
        Returns:
            Dict containing all Gini metrics and Lorenz curve data
        """
        results = {}
        
        # Compute TFG
        tfg_results = self.compute_tokenizer_fairness_gini(language_texts, all_encodings)
        results['tokenizer_fairness_gini'] = tfg_results
        
        # Compute Lorenz curve data
        lorenz_results = self.compute_lorenz_curve_data(language_texts, all_encodings)
        results['lorenz_curve_data'] = lorenz_results
        
        return results