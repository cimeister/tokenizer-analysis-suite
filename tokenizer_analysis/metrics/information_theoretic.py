"""
Information-theoretic metrics including entropy, type-token ratio, and vocabulary utilization.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from collections import Counter
import logging

from .base import BaseMetrics
from ..config import NormalizationConfig, TextNormalizer, DEFAULT_NORMALIZATION_CONFIG, LINES_CONFIG

logger = logging.getLogger(__name__)


class InformationTheoreticMetrics(BaseMetrics):
    """Information-theoretic analysis metrics."""
    
    def __init__(self, tokenizers: Dict[str, Any], tokenizer_names: Optional[List[str]] = None,
                 renyi_alphas: Optional[List[float]] = None, normalization_config: Optional[NormalizationConfig] = None):
        """
        Initialize information-theoretic metrics.
        
        Args:
            tokenizers: Dictionary mapping tokenizer names to tokenizer objects
            tokenizer_names: List of tokenizer names to analyze
            renyi_alphas: List of alpha values for Rényi entropy (default: [1.0, 2.0, 3.0])
        """
        super().__init__(tokenizers, tokenizer_names)
        self.renyi_alphas = renyi_alphas or [1.0, 2.0, 3.0]
        self.norm_config = LINES_CONFIG #normalization_config or DEFAULT_NORMALIZATION_CONFIG
        self.normalizer = TextNormalizer(self.norm_config)
    
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
        
        if alpha == 1.0:
            # Shannon entropy (limit case)
            return -sum(p * np.log2(p) for p in probabilities if p > 0)
        else:
            # General Rényi entropy
            sum_p_alpha = sum(p ** alpha for p in probabilities if p > 0)
            if sum_p_alpha <= 0:
                return 0.0
            return (1 / (1 - alpha)) * np.log2(sum_p_alpha)
    
    def compute_renyi_efficiency_analysis(self, language_texts: Dict[str, List[str]], 
                                        all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """
        Compute Rényi efficiency metrics for all tokenizers.
        """
        if all_encodings is None:
            all_encodings = self.encode_texts_batch(language_texts)
        
        language_texts, all_encodings = self.filter_valid_data(language_texts, all_encodings)
        
        results = {
            'per_tokenizer': {},
            'per_language': {},
            'pairwise_comparisons': {}
        }
        
        for tok_name in self.tokenizer_names:
            tok_results = {}
            
            # Collect all tokens for global entropy
            global_token_counts = Counter()
            per_lang_token_counts = {}
            
            for lang, encodings in all_encodings[tok_name].items():
                if not encodings:
                    continue
                
                lang_token_counts = Counter()
                for token_sequence in encodings:
                    for token in token_sequence:
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
            results['pairwise_comparisons']['shannon'] = self.compute_pairwise_ratios(
                shannon_entropies, 'shannon_entropy'
            )
        
        return results
    
    def compute_compression_ratio(self, language_texts: Dict[str, List[str]], 
                                          all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """
        Compute compression ratios: average of individual (normalization_unit / tokens) ratios.
        
        Args:
            language_texts: Text data by language
            all_encodings: Pre-computed encodings
        """
        if all_encodings is None:
            all_encodings = self.encode_texts_batch(language_texts)
        
        language_texts, all_encodings = self.filter_valid_data(language_texts, all_encodings)
        
        results = {
            'per_tokenizer': {},
            'per_language': {},
            'pairwise_comparisons': {}
        }
        
        for tok_name in self.tokenizer_names:
            per_lang_ratios = {}
            all_individual_ratios = []  # Store individual text ratios
            
            for lang, texts in language_texts.items():
                if not texts or lang not in all_encodings[tok_name]:
                    continue
                
                lang_ratios = []
                for text, tokens in zip(texts, all_encodings[tok_name][lang]):
                    if text.strip():  # Skip empty texts
                        # Use configurable normalization
                        normalization_count = self.normalizer.get_normalization_count(text)
                        if normalization_count > 0:
                            ratio = normalization_count / len(tokens)
                            lang_ratios.append(ratio)
                            all_individual_ratios.append(ratio)
                
                if lang_ratios:
                    # Average of individual ratios for this language
                    per_lang_ratios[lang] = np.mean(lang_ratios)
            
            # Global compression: average of all individual ratios
            global_compression = np.mean(all_individual_ratios) if all_individual_ratios else 1.0
            
            results['per_tokenizer'][tok_name] = {
                'global': global_compression,
                'per_language': per_lang_ratios,
                'num_texts_analyzed': len(all_individual_ratios)
            }
        
        # Add metadata
        results['metadata'] = {
            'normalization_method': self.norm_config.method.value,
            'description': self.normalizer.get_description(),
            'unit': 'lines'
        }
        
        # Compute pairwise comparisons
        global_ratios = {name: results['per_tokenizer'][name]['global'] 
                        for name in self.tokenizer_names}
        results['pairwise_comparisons'] = self.compute_pairwise_ratios(
            global_ratios, 'compression_ratio'
        )
        
        return results
        
    

    def compute_unigram_distribution_metrics(self, language_texts: Dict[str, List[str]], 
                                             all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """
        Computes metrics based on the unigram distribution of tokens for each language.
    
        This includes:
        1.  Unigram Distribution Entropy: The Shannon entropy of the token frequency
            distribution for each language.
        2.  Average Token Rank: The average rank of tokens (by frequency) observed
            in the corpus for each language.
    
        Args:
            language_texts: A dictionary mapping language codes to lists of text samples.
            all_encodings: Pre-computed tokenizations. If None, they will be computed.
    
        Returns:
            A dictionary containing the computed metrics, structured by tokenizer and language,
            including global metrics and pairwise comparisons.
        """
        if all_encodings is None:
            all_encodings = self.encode_texts_batch(language_texts)
        
        language_texts, all_encodings = self.filter_valid_data(language_texts, all_encodings)
        
        results = {
            'per_tokenizer': {},
            'per_language': {
                'unigram_entropy': {},
                'avg_token_rank': {}
            },
            'pairwise_comparisons': {}
        }
    
        for tok_name in self.tokenizer_names:
            per_lang_metrics = {}
            global_token_counts = Counter()
            all_token_sequences = []
    
            for lang, encodings in all_encodings[tok_name].items():
                if not encodings:
                    continue
    
                # Flatten all encodings for the language
                lang_tokens = [token for seq in encodings for token in seq]
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
                all_token_sequences.extend(encodings)
    
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
        
        results['pairwise_comparisons']['global_unigram_entropy'] = self.compute_pairwise_ratios(
            global_entropies, 'global_unigram_entropy'
        )
        results['pairwise_comparisons']['global_avg_token_rank'] = self.compute_pairwise_ratios(
            global_ranks, 'global_avg_token_rank'
        )
    
        return results


    def compute(self, language_texts: Dict[str, List[str]], 
                all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """Compute all information-theoretic metrics."""
        results = {}
        
        if all_encodings is None:
            all_encodings = self.encode_texts_batch(language_texts)

        results['compression_ratio'] = self.compute_compression_ratio(language_texts, all_encodings)
        results['renyi_efficiency'] = self.compute_renyi_efficiency_analysis(language_texts, all_encodings)
        results['unigram_distribution_metrics'] = self.compute_unigram_distribution_metrics(language_texts, all_encodings)

        return results