"""
Information-theoretic metrics including entropy, type-token ratio, and vocabulary utilization.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from collections import Counter
import logging

from .base import BaseMetrics

logger = logging.getLogger(__name__)


class InformationTheoreticMetrics(BaseMetrics):
    """Information-theoretic analysis metrics."""
    
    def __init__(self, tokenizers: Dict[str, Any], tokenizer_names: Optional[List[str]] = None,
                 renyi_alphas: Optional[List[float]] = None):
        """
        Initialize information-theoretic metrics.
        
        Args:
            tokenizers: Dictionary mapping tokenizer names to tokenizer objects
            tokenizer_names: List of tokenizer names to analyze
            renyi_alphas: List of alpha values for Rényi entropy (default: [1.0, 2.0, 3.0])
        """
        super().__init__(tokenizers, tokenizer_names)
        self.renyi_alphas = renyi_alphas or [1.0, 2.0, 3.0]
    
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
    
    def compute_type_token_ratio(self, language_texts: Dict[str, List[str]], 
                               all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """
        Compute type-token ratio (TTR) for all tokenizers.
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
            per_lang_ttr = {}
            global_types = set()
            global_tokens = 0
            
            for lang, encodings in all_encodings[tok_name].items():
                if not encodings:
                    continue
                
                lang_types = set()
                lang_tokens = 0
                
                for token_sequence in encodings:
                    for token in token_sequence:
                        lang_types.add(token)
                        global_types.add(token)
                        lang_tokens += 1
                        global_tokens += 1
                
                if lang_tokens > 0:
                    lang_ttr = len(lang_types) / lang_tokens
                    per_lang_ttr[lang] = {
                        'ttr': lang_ttr,
                        'types': len(lang_types),
                        'tokens': lang_tokens
                    }
            
            # Global TTR
            global_ttr = len(global_types) / global_tokens if global_tokens > 0 else 0.0
            
            results['per_tokenizer'][tok_name] = {
                'global_ttr': global_ttr,
                'global_types': len(global_types),
                'global_tokens': global_tokens,
                'per_language': per_lang_ttr
            }
        
        # Aggregate per-language results
        all_languages = set()
        for tok_results in results['per_tokenizer'].values():
            all_languages.update(tok_results['per_language'].keys())
        
        for lang in all_languages:
            results['per_language'][lang] = {}
            for tok_name in self.tokenizer_names:
                if lang in results['per_tokenizer'][tok_name]['per_language']:
                    results['per_language'][lang][tok_name] = results['per_tokenizer'][tok_name]['per_language'][lang]['ttr']
        
        # Compute pairwise comparisons
        global_ttrs = {name: results['per_tokenizer'][name]['global_ttr'] 
                      for name in self.tokenizer_names}
        results['pairwise_comparisons'] = self.compute_pairwise_ratios(
            global_ttrs, 'type_token_ratio'
        )
        
        return results
    
    def compute_vocabulary_utilization(self, language_texts: Dict[str, List[str]], 
                                     all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """
        Compute vocabulary utilization (percentage of vocabulary actually used).
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
            # Get full vocabulary size
            try:
                full_vocab_size = len(self.tokenizers[tok_name].get_vocab())
            except AttributeError:
                logger.warning(f"Could not get vocabulary size for {tok_name}")
                full_vocab_size = 1  # Avoid division by zero
            
            per_lang_utilization = {}
            used_tokens_global = set()
            
            for lang, encodings in all_encodings[tok_name].items():
                if not encodings:
                    continue
                
                used_tokens_lang = set()
                for token_sequence in encodings:
                    for token in token_sequence:
                        used_tokens_lang.add(token)
                        used_tokens_global.add(token)
                
                lang_utilization = len(used_tokens_lang) / full_vocab_size
                per_lang_utilization[lang] = {
                    'utilization': lang_utilization,
                    'used_tokens': len(used_tokens_lang),
                    'vocab_size': full_vocab_size
                }
            
            # Global utilization
            global_utilization = len(used_tokens_global) / full_vocab_size
            
            results['per_tokenizer'][tok_name] = {
                'global_utilization': global_utilization,
                'used_tokens': len(used_tokens_global),
                'vocab_size': full_vocab_size,
                'per_language': per_lang_utilization
            }
        
        # Aggregate per-language results
        all_languages = set()
        for tok_results in results['per_tokenizer'].values():
            all_languages.update(tok_results['per_language'].keys())
        
        for lang in all_languages:
            results['per_language'][lang] = {}
            for tok_name in self.tokenizer_names:
                if lang in results['per_tokenizer'][tok_name]['per_language']:
                    results['per_language'][lang][tok_name] = results['per_tokenizer'][tok_name]['per_language'][lang]['utilization']
        
        # Compute pairwise comparisons
        global_utilizations = {name: results['per_tokenizer'][name]['global_utilization'] 
                             for name in self.tokenizer_names}
        results['pairwise_comparisons'] = self.compute_pairwise_ratios(
            global_utilizations, 'vocabulary_utilization'
        )
        
        return results
    
    def compute_average_tokens_per_line(self, language_texts: Dict[str, List[str]], 
                                      all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """
        Compute average tokens per line for all tokenizers.
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
            per_lang_avg = {}
            all_token_counts = []
            
            for lang, encodings in all_encodings[tok_name].items():
                if not encodings:
                    continue
                
                lang_token_counts = [len(tokens) for tokens in encodings]
                if lang_token_counts:
                    per_lang_avg[lang] = {
                        'avg_tokens_per_line': np.mean(lang_token_counts),
                        'std_tokens_per_line': np.std(lang_token_counts),
                        'total_lines': len(lang_token_counts)
                    }
                    all_token_counts.extend(lang_token_counts)
            
            # Global average
            global_avg = np.mean(all_token_counts) if all_token_counts else 0.0
            
            results['per_tokenizer'][tok_name] = {
                'global_avg': global_avg,
                'global_std': np.std(all_token_counts) if all_token_counts else 0.0,
                'total_lines': len(all_token_counts),
                'per_language': per_lang_avg
            }
        
        # Aggregate per-language results
        all_languages = set()
        for tok_results in results['per_tokenizer'].values():
            all_languages.update(tok_results['per_language'].keys())
        
        for lang in all_languages:
            results['per_language'][lang] = {}
            for tok_name in self.tokenizer_names:
                if lang in results['per_tokenizer'][tok_name]['per_language']:
                    results['per_language'][lang][tok_name] = results['per_tokenizer'][tok_name]['per_language'][lang]['avg_tokens_per_line']
        
        # Compute pairwise comparisons
        global_avgs = {name: results['per_tokenizer'][name]['global_avg'] 
                      for name in self.tokenizer_names}
        results['pairwise_comparisons'] = self.compute_pairwise_ratios(
            global_avgs, 'avg_tokens_per_line'
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
        
        results['renyi_efficiency'] = self.compute_renyi_efficiency_analysis(language_texts, all_encodings)
        results['type_token_ratio'] = self.compute_type_token_ratio(language_texts, all_encodings)
        results['vocabulary_utilization'] = self.compute_vocabulary_utilization(language_texts, all_encodings)
        results['avg_tokens_per_line'] = self.compute_average_tokens_per_line(language_texts, all_encodings)
        results['unigram_distribution_metrics'] = self.compute_unigram_distribution_metrics(language_texts, all_encodings)

        
        return results