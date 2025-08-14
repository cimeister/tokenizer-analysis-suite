"""
Information-theoretic metrics including entropy, compression, and vocabulary utilization.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from collections import Counter
import logging

from .base import BaseMetrics, TokenizedDataProcessor
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider
from ..config import TextMeasurementConfig, TextMeasurer, DEFAULT_LINE_MEASUREMENT_CONFIG
from ..config.language_metadata import LanguageMetadata
from ..constants import DEFAULT_RENYI_ALPHAS, SHANNON_ENTROPY_ALPHA

logger = logging.getLogger(__name__)


class InformationTheoreticMetrics(BaseMetrics):
    """Information-theoretic analysis metrics."""
    
    def __init__(self, input_provider: InputProvider,
                 renyi_alphas: Optional[List[float]] = None, 
                 measurement_config: Optional[TextMeasurementConfig] = None,
                 language_metadata: Optional[LanguageMetadata] = None):
        """
        Initialize information-theoretic metrics.

        Args:
            input_provider: InputProvider instance
            renyi_alphas: List of alpha values for Rényi entropy (default: [1.0, 2.0, 3.0])
            measurement_config: Configuration for text measurement method
            language_metadata: Optional language metadata for grouping
        """
        super().__init__(input_provider)
        self.renyi_alphas = renyi_alphas or DEFAULT_RENYI_ALPHAS
        # Default to lines for information-theoretic analysis (as was hardcoded before)
        self.measurement_config = measurement_config or DEFAULT_LINE_MEASUREMENT_CONFIG
        self.text_measurer = TextMeasurer(self.measurement_config)
        self.language_metadata = language_metadata
    
    def compute_renyi_entropy(self, token_counts: Counter, alpha: float) -> float:
        """
        Compute Rényi entropy of order alpha for token distribution.
        
        Args:
            token_counts: Counter of token frequencies
            alpha: Order of Rényi entropy
            
        Returns:
            Rényi entropy value
        """
        if not token_counts:
            return 0.0
        
        total_count = sum(token_counts.values())
        probabilities = [count / total_count for count in token_counts.values()]
        
        if alpha == SHANNON_ENTROPY_ALPHA:
            # Shannon entropy (limit case)
            return -sum(p * np.log2(p) for p in probabilities if p > 0)
        else:
            # General Rényi entropy
            sum_p_alpha = sum(p ** alpha for p in probabilities if p > 0)
            if sum_p_alpha <= 0:
                return 0.0
            return (1 / (1 - alpha)) * np.log2(sum_p_alpha)
    
    def compute_renyi_efficiency_analysis(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute Rényi efficiency metrics for all tokenizers.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with Rényi efficiency results
        """
        
        results = {
            'per_tokenizer': {},
            'per_language': {},
            'pairwise_comparisons': {}
        }
        
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
                
            tok_results = {}
            tok_data = tokenized_data[tok_name]
            
            # Collect all tokens for global entropy
            global_token_counts = Counter()
            per_lang_token_counts = {}
            
            # Group data by language
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
            
            for lang, lang_data in lang_groups.items():
                lang_token_counts = Counter()
                
                for data in lang_data:
                    for token in data.tokens:
                        global_token_counts[token] += 1
                        lang_token_counts[token] += 1
                
                per_lang_token_counts[lang] = lang_token_counts
            
            # Compute Rényi entropy for each alpha
            for alpha in self.renyi_alphas:
                alpha_key = f'renyi_{alpha}'
                tok_results[alpha_key] = {}
                
                # Global entropy
                global_entropy = self.compute_renyi_entropy(global_token_counts, alpha)
                tok_results[alpha_key]['overall'] = global_entropy
                
                # Per-language entropy
                for lang, lang_counts in per_lang_token_counts.items():
                    lang_entropy = self.compute_renyi_entropy(lang_counts, alpha)
                    tok_results[alpha_key][lang] = lang_entropy
            
            results['per_tokenizer'][tok_name] = tok_results
        
        # Aggregate per-language results
        all_languages = set()
        for tok_results in results['per_tokenizer'].values():
            for alpha in self.renyi_alphas:
                alpha_key = f'renyi_{alpha}'
                if alpha_key in tok_results:
                    all_languages.update(k for k in tok_results[alpha_key].keys() if k != 'overall')
        
        for alpha in self.renyi_alphas:
            alpha_key = f'renyi_{alpha}'
            results['per_language'][alpha_key] = {}
            for lang in all_languages:
                results['per_language'][alpha_key][lang] = {}
                for tok_name in self.tokenizer_names:
                    if (alpha_key in results['per_tokenizer'][tok_name] and 
                        lang in results['per_tokenizer'][tok_name][alpha_key]):
                        results['per_language'][alpha_key][lang][tok_name] = results['per_tokenizer'][tok_name][alpha_key][lang]
        
        # Compute pairwise comparisons for Shannon entropy (alpha=1.0)
        if 1.0 in self.renyi_alphas:
            shannon_entropies = {name: results['per_tokenizer'][name]['renyi_1.0']['overall'] 
                               for name in self.tokenizer_names}
            results['pairwise_comparisons']['shannon'] = self.compute_pairwise_comparisons(
                shannon_entropies, 'shannon_entropy'
            )
        
        return results
    
    def compute_compression_ratio(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute compression ratios: average of individual (normalization_unit / tokens) ratios.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with compression ratio results
        """
        
        results = {
            'per_tokenizer': {},
            'per_language': {},
            'pairwise_comparisons': {}
        }
        
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
                
            tok_data = tokenized_data[tok_name]
            per_lang_ratios = {}
            all_individual_ratios = []  # Store individual text ratios
            
            # Group data by language
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
            
            for lang, lang_data in lang_groups.items():
                lang_ratios = []
                
                for data in lang_data:
                    if data.text and data.text.strip():  # Skip empty texts
                        # Use configurable normalization
                        normalization_count = self.text_measurer.get_unit_count(data.text)
                        if normalization_count > 0 and data.tokens:
                            ratio = normalization_count / len(data.tokens)
                            lang_ratios.append(ratio)
                            all_individual_ratios.append(ratio)
                
                if lang_ratios:
                    # Average of individual ratios for this language
                    per_lang_ratios[lang] = np.mean(lang_ratios)
            
            # Global compression: compute full statistics from individual ratios
            if all_individual_ratios:
                global_stats = self.compute_basic_stats(all_individual_ratios)
            else:
                global_stats = self.empty_stats()
                global_stats['mean'] = 1.0  # Default compression ratio
            
            results['per_tokenizer'][tok_name] = {
                'global': global_stats,
                'per_language': per_lang_ratios,
                'num_texts_analyzed': len(all_individual_ratios)
            }
        
        # Add metadata
        results['metadata'] = {
            'normalization_method': self.measurement_config.method.value
        }
        
        # Compute pairwise comparisons
        global_ratios = {name: results['per_tokenizer'][name]['global']['mean'] 
                        for name in self.tokenizer_names if name in results['per_tokenizer']}
        results['pairwise_comparisons'] = self.compute_pairwise_comparisons(
            global_ratios, 'compression_ratio'
        )
        
        return results
        
    

    def compute_unigram_distribution_metrics(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Computes metrics based on the unigram distribution of tokens for each language.
    
        This includes:
        1.  Unigram Distribution Entropy: The Shannon entropy of the token frequency
            distribution for each language.
        2.  Average Token Rank: The average rank of tokens (by frequency) observed
            in the corpus for each language.
    
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
    
        Returns:
            A dictionary containing the computed metrics, structured by tokenizer and language,
            including global metrics and pairwise comparisons.
        """
        
        results = {
            'per_tokenizer': {},
            'per_language': {
                'unigram_entropy': {},
                'avg_token_rank': {}
            },
            'pairwise_comparisons': {}
        }
    
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
                
            tok_data = tokenized_data[tok_name]
            per_lang_metrics = {}
            global_token_counts = Counter()
            all_token_sequences = []
    
            # Group data by language
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
    
            for lang, lang_data in lang_groups.items():
                # Flatten all tokens for the language
                lang_tokens = TokenizedDataProcessor.flatten_all_tokens(lang_data)
                if not lang_tokens:
                    continue
    
                # 1. Compute per-language unigram distribution and metrics
                lang_token_counts = Counter(lang_tokens)
                unigram_entropy = self.compute_renyi_entropy(lang_token_counts, alpha=1.0)
                
                ranked_tokens = [token for token, count in lang_token_counts.most_common()]
                token_to_rank = {token: rank + 1 for rank, token in enumerate(ranked_tokens)}
                
                lang_ranks = [token_to_rank[token] for token in lang_tokens]
                avg_token_rank = np.mean(lang_ranks) if lang_ranks else 0.0
                
                per_lang_metrics[lang] = {
                    'unigram_entropy': unigram_entropy,
                    'avg_token_rank': avg_token_rank,
                    'total_tokens': len(lang_tokens),
                    'unique_tokens': len(lang_token_counts)
                }
    
                # Aggregate for global metrics
                global_token_counts.update(lang_tokens)
                all_token_sequences.extend([data.tokens for data in lang_data])
    
            # 2. Compute global metrics for the tokenizer
            global_unigram_entropy = self.compute_renyi_entropy(global_token_counts, alpha=1.0)
            
            global_avg_token_rank = 0.0
            if global_token_counts:
                globally_ranked_tokens = [token for token, count in global_token_counts.most_common()]
                global_token_to_rank = {token: rank + 1 for rank, token in enumerate(globally_ranked_tokens)}
                
                all_global_ranks = [global_token_to_rank[token] for seq in all_token_sequences for token in seq]
                global_avg_token_rank = np.mean(all_global_ranks) if all_global_ranks else 0.0
    
            results['per_tokenizer'][tok_name] = {
                'global_unigram_entropy': global_unigram_entropy,
                'global_avg_token_rank': global_avg_token_rank,
                'per_language': per_lang_metrics
            }
    
        # 3. Aggregate per-language results for easier comparison across tokenizers
        all_languages = set()
        for tok_results in results['per_tokenizer'].values():
            all_languages.update(tok_results['per_language'].keys())
    
        for lang in all_languages:
            results['per_language']['unigram_entropy'][lang] = {}
            results['per_language']['avg_token_rank'][lang] = {}
            for tok_name in self.tokenizer_names:
                lang_stats = results['per_tokenizer'][tok_name]['per_language'].get(lang)
                if lang_stats:
                    results['per_language']['unigram_entropy'][lang][tok_name] = lang_stats['unigram_entropy']
                    results['per_language']['avg_token_rank'][lang][tok_name] = lang_stats['avg_token_rank']
    
        # 4. Compute pairwise comparisons for global metrics
        global_entropies = {name: res['global_unigram_entropy'] for name, res in results['per_tokenizer'].items()}
        global_ranks = {name: res['global_avg_token_rank'] for name, res in results['per_tokenizer'].items()}
        
        results['pairwise_comparisons']['global_unigram_entropy'] = self.compute_pairwise_comparisons(
            global_entropies, 'global_unigram_entropy'
        )
        results['pairwise_comparisons']['global_avg_token_rank'] = self.compute_pairwise_comparisons(
            global_ranks, 'global_avg_token_rank'
        )
    
        return results


    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """Compute all information-theoretic metrics."""
        if tokenized_data is None:
            tokenized_data = self.get_tokenized_data()
            
        results = {}
        
        results['compression_ratio'] = self.compute_compression_ratio(tokenized_data)
        results['renyi_efficiency'] = self.compute_renyi_efficiency_analysis(tokenized_data)
        results['unigram_distribution_metrics'] = self.compute_unigram_distribution_metrics(tokenized_data)

        return results
    
    # Convenience methods for common groupings
    def compute_by_script_family(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """Compute information-theoretic metrics grouped by script family."""
        # This functionality should be handled by the analyzer's grouped analysis
        # For now, just return regular compute results
        return self.compute(tokenized_data)
    
    def compute_by_resource_level(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """Compute information-theoretic metrics grouped by resource level."""
        # This functionality should be handled by the analyzer's grouped analysis
        # For now, just return regular compute results
        return self.compute(tokenized_data)