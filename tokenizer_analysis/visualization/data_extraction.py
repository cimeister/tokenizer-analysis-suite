"""
Data extraction utilities for tokenizer analysis visualization.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DataExtractor:
    """Centralized data extraction and processing for visualization."""
    
    @staticmethod
    def extract_per_language_value(metric_name: str, value: Any) -> float:
        """
        Extract numeric value from per-language data structures.
        
        Args:
            metric_name: Name of the metric
            value: Value from per_language data structure
            
        Returns:
            Numeric value for plotting
        """
        if isinstance(value, (int, float)):
            return value
        
        if isinstance(value, dict):
            # Handle known metric structures
            if metric_name == 'fertility':
                return value.get('mean', 0.0)
            elif metric_name == 'vocabulary_utilization':
                return value.get('utilization', 0.0)
            elif metric_name == 'type_token_ratio':
                return value.get('ttr', 0.0)
            elif metric_name == 'avg_tokens_per_line':
                return value.get('avg_tokens_per_line', 0.0)
            elif metric_name in ['unigram_entropy', 'avg_token_rank']:
                return value.get(metric_name, 0.0)
            elif 'mean' in value:
                return value['mean']
            else:
                logger.warning(f"Unknown per_language structure for {metric_name}: {value}")
                return 0.0
        
        logger.warning(f"Unexpected value type for {metric_name}: {type(value)}")
        return 0.0
    
    @staticmethod
    def extract_global_value(metric_name: str, data: Any) -> float:
        """
        Extract global value from metric data.
        
        Args:
            metric_name: Name of the metric
            data: Data structure from results
            
        Returns:
            Global numeric value
        """
        if isinstance(data, (int, float)):
            return data
        
        if isinstance(data, dict):
            # Handle different global data structures
            if metric_name == 'fertility' and 'mean' in data:
                return data['mean']
            elif metric_name == 'compression_ratio':
                return data  # Already a float
            elif metric_name == 'type_token_ratio':
                return data.get('global_ttr', 0.0)
            elif metric_name == 'vocabulary_utilization':
                return data.get('global_utilization', 0.0)
            elif metric_name == 'avg_tokens_per_line':
                return data.get('global_avg', 0.0)
            elif 'mean' in data:
                return data['mean']
            elif 'global' in data:
                return DataExtractor.extract_global_value(metric_name, data['global'])
            else:
                logger.warning(f"Unknown global structure for {metric_name}: {data}")
                return 0.0
        
        logger.warning(f"Unexpected global value type for {metric_name}: {type(data)}")
        return 0.0
    
    @staticmethod
    def extract_metric_data(results: Dict[str, Any], metric_name: str, 
                           tokenizer_names: List[str]) -> Dict[str, Any]:
        """
        Extract all data for a specific metric.
        
        Args:
            results: Full results dictionary
            metric_name: Name of the metric to extract
            tokenizer_names: List of tokenizer names
            
        Returns:
            Dictionary with extracted metric data
        """
        if metric_name not in results:
            return {}
        
        metric_data = results[metric_name]
        extracted = {
            'metadata': metric_data.get('metadata', {}),
            'global_values': {},
            'per_language_data': {},
            'available_languages': set()
        }
        
        # Extract global values
        if 'per_tokenizer' in metric_data:
            for tok_name in tokenizer_names:
                if tok_name in metric_data['per_tokenizer']:
                    tok_data = metric_data['per_tokenizer'][tok_name]
                    
                    # Extract global value
                    if 'global' in tok_data:
                        extracted['global_values'][tok_name] = DataExtractor.extract_global_value(
                            metric_name, tok_data['global']
                        )
                    
                    # Extract per-language values
                    if 'per_language' in tok_data:
                        for lang, lang_value in tok_data['per_language'].items():
                            if lang not in extracted['per_language_data']:
                                extracted['per_language_data'][lang] = {}
                            
                            extracted['per_language_data'][lang][tok_name] = (
                                DataExtractor.extract_per_language_value(metric_name, lang_value)
                            )
                            extracted['available_languages'].add(lang)
        
        # Handle direct per_language structure (like type_token_ratio)
        elif 'per_language' in metric_data:
            for lang, tok_values in metric_data['per_language'].items():
                if lang not in extracted['per_language_data']:
                    extracted['per_language_data'][lang] = {}
                
                for tok_name, value in tok_values.items():
                    if tok_name in tokenizer_names:
                        extracted['per_language_data'][lang][tok_name] = (
                            DataExtractor.extract_per_language_value(metric_name, value)
                        )
                        extracted['available_languages'].add(lang)
        
        extracted['available_languages'] = sorted(list(extracted['available_languages']))
        return extracted
    
    @staticmethod
    def extract_renyi_data(results: Dict[str, Any], tokenizer_names: List[str]) -> Dict[str, Any]:
        """
        Extract Rényi entropy data for curve plotting.
        
        Args:
            results: Full results dictionary
            tokenizer_names: List of tokenizer names
            
        Returns:
            Dictionary with Rényi entropy data
        """
        if 'renyi_efficiency' not in results:
            return {}
        
        renyi_data = results['renyi_efficiency']
        extracted = {
            'alphas': [],
            'entropy_curves': {tok: [] for tok in tokenizer_names},
            'per_language_data': {}
        }
        
        # Extract alpha values
        alphas = set()
        for tok_name in tokenizer_names:
            if tok_name in renyi_data.get('per_tokenizer', {}):
                tok_results = renyi_data['per_tokenizer'][tok_name]
                for key in tok_results:
                    if key.startswith('renyi_'):
                        try:
                            alpha = float(key.split('_')[1])
                            alphas.add(alpha)
                        except (ValueError, IndexError):
                            continue
        
        extracted['alphas'] = sorted(list(alphas))
        
        # Extract entropy curves
        for tok_name in tokenizer_names:
            if tok_name in renyi_data.get('per_tokenizer', {}):
                tok_results = renyi_data['per_tokenizer'][tok_name]
                
                for alpha in extracted['alphas']:
                    key = f'renyi_{alpha}'
                    if key in tok_results:
                        entropy_value = tok_results[key].get('overall', 0.0)
                        extracted['entropy_curves'][tok_name].append(entropy_value)
                    else:
                        extracted['entropy_curves'][tok_name].append(0.0)
        
        # Extract per-language data for heatmaps
        if extracted['alphas'] and 2.0 in extracted['alphas']:
            all_languages = set()
            for tok_name in tokenizer_names:
                if tok_name in renyi_data.get('per_tokenizer', {}):
                    tok_results = renyi_data['per_tokenizer'][tok_name]
                    if 'renyi_2.0' in tok_results:
                        all_languages.update(
                            k for k in tok_results['renyi_2.0'].keys() if k != 'overall'
                        )
            
            extracted['per_language_data'] = {
                'languages': sorted(list(all_languages)),
                'data_matrix': []
            }
            
            for tok_name in tokenizer_names:
                row = []
                if tok_name in renyi_data.get('per_tokenizer', {}):
                    tok_results = renyi_data['per_tokenizer'][tok_name]
                    if 'renyi_2.0' in tok_results:
                        for lang in extracted['per_language_data']['languages']:
                            row.append(tok_results['renyi_2.0'].get(lang, 0.0))
                    else:
                        row = [0.0] * len(extracted['per_language_data']['languages'])
                else:
                    row = [0.0] * len(extracted['per_language_data']['languages'])
                
                extracted['per_language_data']['data_matrix'].append(row)
        
        return extracted
    
    @staticmethod
    def extract_gini_data(results: Dict[str, Any], tokenizer_names: List[str]) -> Dict[str, Any]:
        """
        Extract Gini coefficient and fairness data.
        
        Args:
            results: Full results dictionary
            tokenizer_names: List of tokenizer names
            
        Returns:
            Dictionary with Gini fairness data
        """
        if 'tokenizer_fairness_gini' not in results:
            return {}
        
        gini_results = results['tokenizer_fairness_gini']
        extracted = {
            'metadata': gini_results.get('metadata', {}),
            'gini_coefficients': {},
            'cost_ratios': {},
            'language_costs': {},
            'summary_stats': {}
        }
        
        per_tokenizer = gini_results.get('per_tokenizer', {})
        
        for tok_name in tokenizer_names:
            if tok_name in per_tokenizer and 'warning' not in per_tokenizer[tok_name]:
                data = per_tokenizer[tok_name]
                
                extracted['gini_coefficients'][tok_name] = data.get('gini_coefficient', 0.0)
                extracted['cost_ratios'][tok_name] = data.get('cost_ratio', 0.0)
                extracted['language_costs'][tok_name] = data.get('language_costs', {})
                
                extracted['summary_stats'][tok_name] = {
                    'mean_cost': data.get('mean_cost', 0.0),
                    'min_cost': data.get('min_cost', 0.0),
                    'max_cost': data.get('max_cost', 0.0),
                    'most_efficient': data.get('most_efficient_language', ('', 0.0)),
                    'least_efficient': data.get('least_efficient_language', ('', 0.0))
                }
        
        return extracted
    
    @staticmethod
    def extract_lorenz_data(results: Dict[str, Any], tokenizer_names: List[str]) -> Dict[str, Any]:
        """
        Extract Lorenz curve data.
        
        Args:
            results: Full results dictionary
            tokenizer_names: List of tokenizer names
            
        Returns:
            Dictionary with Lorenz curve data
        """
        if 'lorenz_curve_data' not in results:
            return {}
        
        lorenz_results = results['lorenz_curve_data']
        extracted = {}
        
        per_tokenizer = lorenz_results.get('per_tokenizer', {})
        
        for tok_name in tokenizer_names:
            if tok_name in per_tokenizer and 'warning' not in per_tokenizer[tok_name]:
                data = per_tokenizer[tok_name]
                
                extracted[tok_name] = {
                    'x_values': data.get('x_values', []),
                    'y_values': data.get('y_values', []),
                    'equality_line': data.get('equality_line', []),
                    'n_languages': data.get('n_languages', 0),
                    'sorted_languages': data.get('sorted_languages', [])
                }
        
        return extracted
    
    @staticmethod
    def extract_grouped_data(grouped_results: Dict[str, Dict[str, Any]], 
                           metric_name: str, tokenizer_names: List[str]) -> Dict[str, Any]:
        """
        Extract data for grouped analysis plots.
        
        Args:
            grouped_results: Results grouped by categories
            metric_name: Name of the metric to extract
            tokenizer_names: List of tokenizer names
            
        Returns:
            Dictionary with grouped metric data
        """
        extracted = {
            'group_names': [],
            'data_matrix': [],
            'metadata': {}
        }
        
        for group_name, results in grouped_results.items():
            if metric_name not in results:
                continue
            
            extracted['group_names'].append(group_name)
            
            # Get metadata from first available group
            if not extracted['metadata'] and 'metadata' in results[metric_name]:
                extracted['metadata'] = results[metric_name]['metadata']
            
            # Extract values for each tokenizer
            row_data = []
            for tok_name in tokenizer_names:
                if (tok_name in results[metric_name].get('per_tokenizer', {}) and 
                    'global' in results[metric_name]['per_tokenizer'][tok_name]):
                    
                    global_data = results[metric_name]['per_tokenizer'][tok_name]['global']
                    value = DataExtractor.extract_global_value(metric_name, global_data)
                    row_data.append(value)
                else:
                    row_data.append(0.0)
            
            extracted['data_matrix'].append(row_data)
        
        return extracted