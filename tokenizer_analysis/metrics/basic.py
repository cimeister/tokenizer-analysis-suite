from typing import Dict, List, Any, Optional, Set
import numpy as np
from collections import Counter
import logging
import unicodedata

from .base import BaseMetrics
from ..config import NormalizationConfig, TextNormalizer, DEFAULT_NORMALIZATION_CONFIG

logger = logging.getLogger(__name__)


class BasicTokenizationMetrics(BaseMetrics):
    
    def __init__(self, tokenizers: Dict[str, Any], tokenizer_names: List[str], 
                 normalization_config: Optional[NormalizationConfig] = None):
        super().__init__(tokenizers, tokenizer_names)
        self.norm_config = normalization_config or DEFAULT_NORMALIZATION_CONFIG
        self.normalizer = TextNormalizer(self.norm_config)
    
    def compute_fertility_analysis(self, language_texts: Dict[str, List[str]], 
                                    all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """
        Compute fertility analysis using the configured normalization method.
        
        Returns:
            Dict with 'fertility' results based on the configured normalization
        """
        if all_encodings is None:
            all_encodings = self.encode_texts_batch(language_texts)
        
        language_texts, all_encodings = self.filter_valid_data(language_texts, all_encodings)
        
        results = {
            'fertility': {
                'per_tokenizer': {},
                'per_language': {},
                'pairwise_comparisons': {},
                'metadata': {
                    'normalization_method': self.norm_config.method.value,
                    'description': self.normalizer.get_description(),
                    'short_description': self.normalizer.get_short_description()
                }
            }
        }
        
        for tok_name in self.tokenizer_names:
            per_lang_fertility = {}
            all_fertilities = []
            
            for lang, texts in language_texts.items():
                if not texts or lang not in all_encodings[tok_name]:
                    continue
                
                lang_fertilities = []
                
                for text, tokens in zip(texts, all_encodings[tok_name][lang]):
                    if text.strip():
                        # Get normalization count using configured method
                        norm_count = self.normalizer.get_normalization_count(text)
                        if norm_count > 0:
                            fertility = len(tokens) / norm_count
                            lang_fertilities.append(fertility)
                            all_fertilities.append(fertility)
                
                # Store per-language statistics
                if lang_fertilities:
                    per_lang_fertility[lang] = {
                        'mean': np.mean(lang_fertilities),
                        'std': np.std(lang_fertilities),
                        'median': np.median(lang_fertilities)
                    }
            
            # Global fertility statistics
            global_stats = self.compute_basic_stats(all_fertilities)
            
            results['fertility']['per_tokenizer'][tok_name] = {
                'global': global_stats,
                'per_language': per_lang_fertility
            }
        
        # Compute pairwise comparisons
        global_fertilities = {name: results['fertility']['per_tokenizer'][name]['global']['mean'] 
                             for name in self.tokenizer_names}
        
        results['fertility']['pairwise_comparisons'] = self.compute_pairwise_ratios(
            global_fertilities, 'fertility'
        )
        
        return results
    
    def compute_token_length_analysis(self) -> Dict[str, Any]:
        """
        Analyze token lengths with proper handling of special tokens.
        Uses the configured normalization method to determine the primary unit.
        """
        # Determine primary unit based on normalization method
        if self.norm_config.method == self.norm_config.method.BYTES:
            primary_unit = 'bytes'
            use_bytes = True
        else:
            primary_unit = 'characters'
            use_bytes = False
            
        results = {
            'per_tokenizer': {}, 
            'detailed_analysis': {},
            'metadata': {
                'primary_unit': primary_unit,
                'normalization_method': self.norm_config.method.value,
                'encoding': getattr(self.norm_config, 'encoding', 'utf-8')
            }
        }
        
        for name in self.tokenizer_names:
            try:
                vocab = list(self.tokenizers[name].get_vocab().keys())
                
                # Character lengths (more interpretable)
                char_lengths = [len(token) for token in vocab]
                
                # Careful byte length analysis
                byte_lengths = []
                special_token_count = 0
                long_token_count = 0
                
                for token in vocab:
                    # Handle special tokens
                    clean_token = token
                    if token.startswith('▁'):  # SentencePiece space marker
                        clean_token = token[1:]
                        special_token_count += 1
                    elif token.startswith('Ġ'):  # GPT-style space marker
                        clean_token = token[1:]
                        special_token_count += 1
                    elif token.startswith('##'):  # BERT-style continuation
                        clean_token = token[2:]
                        special_token_count += 1
                    
                    byte_len = len(clean_token.encode('utf-8'))
                    byte_lengths.append(byte_len)
                    
                    if byte_len > 20:  # Flag unusually long tokens
                        long_token_count += 1
                
                # Compute statistics for both character and byte lengths
                char_stats = self.compute_basic_stats(char_lengths)
                byte_stats = self.compute_basic_stats(byte_lengths)
                
                # Use primary unit based on use_bytes flag
                primary_stats = byte_stats if use_bytes else char_stats
                secondary_stats = char_stats if use_bytes else byte_stats
                
                results['per_tokenizer'][name] = {
                    'primary_length': primary_stats,
                    'secondary_length': secondary_stats,
                    'character_length': char_stats,  # Keep for backwards compatibility
                    'byte_length': byte_stats,      # Keep for backwards compatibility
                    'vocab_size': len(vocab)
                }
                
                # Detailed analysis
                results['detailed_analysis'][name] = {
                    'special_tokens': special_token_count,
                    'long_tokens_over_20_bytes': long_token_count,
                    'char_byte_ratio': np.mean(char_lengths) / np.mean(byte_lengths) if np.mean(byte_lengths) > 0 else 1.0
                }
                
                # Sanity check warnings
                if np.mean(byte_lengths) > 10:
                    logger.warning(f"{name}: High average byte length ({np.mean(byte_lengths):.2f}) - check for encoding issues")
                if long_token_count > len(vocab) * 0.01:  # More than 1% long tokens
                    logger.warning(f"{name}: Many long tokens ({long_token_count}) - potential encoding artifacts")
                
            except AttributeError:
                logger.warning(f"Could not analyze token lengths for {name}")
                results['per_tokenizer'][name] = {
                    'character_length': self.compute_basic_stats([]),
                    'byte_length': self.compute_basic_stats([])
                }
        
        # Compute pairwise comparisons using primary metric
        if len(self.tokenizer_names) > 1:
            primary_unit = 'byte' if use_bytes else 'character'
            primary_mean_lengths = {name: results['per_tokenizer'][name]['primary_length']['mean'] 
                                  for name in self.tokenizer_names}
            results['pairwise_comparisons'] = self.compute_pairwise_ratios(
                primary_mean_lengths, f'token_{primary_unit}_length'
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
    
    def validate_metric_sanity(self, results: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Perform sanity checks on computed metrics.
        
        Returns:
            Dict mapping metric names to lists of warning messages
        """
        warnings = {'compression_ratio': [], 'fertility': [], 'token_length': []}
        
        # Compression ratio checks
        if 'compression_ratio' in results:
            for tok_name, tok_results in results['compression_ratio']['per_tokenizer'].items():
                ratio = tok_results['global']
                if ratio < 0.05:
                    warnings['compression_ratio'].append(f"{tok_name}: Very low compression ratio ({ratio:.3f}) - may indicate issue")
                elif ratio > 1.0:
                    warnings['compression_ratio'].append(f"{tok_name}: Compression ratio > 1.0 ({ratio:.3f}) - tokens/chars should be < 1")
                elif ratio > 0.8:
                    warnings['compression_ratio'].append(f"{tok_name}: High compression ratio ({ratio:.3f}) - check if correct")
        
        # Fertility checks
        if 'fertility' in results:
            for tok_name, tok_results in results['fertility']['per_tokenizer'].items():
                fertility = tok_results['global']['mean']
                if fertility < 0.1:
                    warnings['fertility'].append(f"{tok_name}: Very low fertility ({fertility:.2f}) - check calculation")
                elif fertility > 10.0:
                    warnings['fertility'].append(f"{tok_name}: Very high fertility ({fertility:.2f}) - may indicate character-level tokenization")
        
        # Token length checks  
        if 'token_length' in results:
            for tok_name, tok_results in results['token_length']['per_tokenizer'].items():
                if 'byte_length' in tok_results:
                    byte_mean = tok_results['byte_length']['mean']
                    if byte_mean > 15:
                        warnings['token_length'].append(f"{tok_name}: High average byte length ({byte_mean:.2f}) - check encoding")
        
        return warnings

    def test_token_length_analysis_validity(self):
        """
        Validate the compute_token_length_analysis function,
        focusing on correct character vs. byte length calculations and stats.
        """
        # 1. Define a mock tokenizer class for controlled testing
        class MockTokenizer:
            def __init__(self, vocab):
                self._vocab = {token: i for i, token in enumerate(vocab)}
            def get_vocab(self):
                return self._vocab
    
        # 2. Define test cases
        test_cases = [
            {
                'name': 'ASCII', 'vocab': ['hello', 'world'],
                'expected_char': {'mean': 5.0, 'sum': 10},
                'expected_byte': {'mean': 5.0, 'sum': 10}
            },
            {
                'name': 'Tokenizer Prefixes', 
                'vocab': [' a', 'Ġb', '##c', 'd'], # Note: The space is U+2581, not a standard space
                'expected_char': {'mean': 2.0, 'sum': 8}, # len() includes prefixes: 2+2+3+1=8
                'expected_byte': {'mean': 1.0, 'sum': 4}  # .encode() is on cleaned tokens: 1+1+1+1=4
            },
            {
                'name': 'Multi-byte (Latin)', 'vocab': ['café', 'brûlée'],
                'expected_char': {'mean': 5.0, 'sum': 10},
                'expected_byte': {'mean': 6.5, 'sum': 13}
            },
            {
                'name': 'Multi-byte (CJK)', 'vocab': ['你好', '世界'],
                'expected_char': {'mean': 2.0, 'sum': 4},
                'expected_byte': {'mean': 6.0, 'sum': 12}
            },
            {
                'name': 'Mixed Vocab', 'vocab': ['a', 'Ġé', '你好'],
                'expected_char': {'mean': 1.67, 'sum': 5},
                'expected_byte': {'mean': 3.0, 'sum': 9}
            }
        ]
    
        print("\n--- Running Validity Tests for compute_token_length_analysis ---")
        passed_count = 0
        failed_count = 0
    
        # Temporarily replace instance attributes for testing
        original_tokenizers = self.tokenizers
        original_names = self.tokenizer_names
    
        for i, case in enumerate(test_cases):
            print(f"\n[Test Case {i+1}: {case['name']}]")
            try:
                mock_tok_name = f"mock_{case['name']}"
                self.tokenizers = {mock_tok_name: MockTokenizer(case['vocab'])}
                self.tokenizer_names = [mock_tok_name]
    
                if not hasattr(self, 'compute_basic_stats'):
                    self.compute_basic_stats = compute_basic_stats.__get__(self)
                
                results = self.compute_token_length_analysis()
                
                char_stats = results['per_tokenizer'][mock_tok_name]['character_length']
                byte_stats = results['per_tokenizer'][mock_tok_name]['byte_length']
    
                char_mean_ok = abs(char_stats['mean'] - case['expected_char']['mean']) < 0.01
                char_sum_ok = char_stats['sum'] == case['expected_char']['sum']
                byte_mean_ok = abs(byte_stats['mean'] - case['expected_byte']['mean']) < 0.01
                byte_sum_ok = byte_stats['sum'] == case['expected_byte']['sum']
    
                if char_mean_ok and char_sum_ok and byte_mean_ok and byte_sum_ok:
                    print("  -> PASSED")
                    passed_count += 1
                else:
                    print("  -> FAILED")
                    if not char_mean_ok: print(f"    - Char Mean Fail: Expected ~{case['expected_char']['mean']:.2f}, Got {char_stats['mean']:.2f}")
                    if not char_sum_ok: print(f"    - Char Sum Fail: Expected {case['expected_char']['sum']}, Got {char_stats['sum']}")
                    if not byte_mean_ok: print(f"    - Byte Mean Fail: Expected ~{case['expected_byte']['mean']:.2f}, Got {byte_stats['mean']:.2f}")
                    if not byte_sum_ok: print(f"    - Byte Sum Fail: Expected {case['expected_byte']['sum']}, Got {byte_stats['sum']}")
                    failed_count += 1
            except Exception as e:
                import traceback
                print(f"  -> ERROR: An exception occurred: {e}")
                traceback.print_exc()
                failed_count += 1
    
        self.tokenizers = original_tokenizers
        self.tokenizer_names = original_names
    
        print("\n--- Test Summary ---")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        print("--------------------")
        return {'passed': passed_count, 'failed': failed_count}

    
    def compute(self, language_texts: Dict[str, List[str]], 
                all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """Compute all basic tokenization metrics using configured normalization."""
        results = {}
                
        # Fertility analysis using configured normalization
        fertility_results = self.compute_fertility_analysis(language_texts, all_encodings)
        results.update(fertility_results)
        
        results['token_length'] = self.compute_token_length_analysis()
        results['vocabulary_overlap'] = self.compute_vocabulary_overlap()
        results['type_token_ratio'] = self.compute_type_token_ratio(language_texts, all_encodings)
        results['vocabulary_utilization'] = self.compute_vocabulary_utilization(language_texts, all_encodings)
        results['avg_tokens_per_line'] = self.compute_average_tokens_per_line(language_texts, all_encodings)
        
        # Perform sanity checks
        validation_warnings = self.validate_metric_sanity(results)
        if any(validation_warnings.values()):
            results['validation_warnings'] = validation_warnings
            logger.warning("Metric validation found potential issues - check 'validation_warnings' in results")
        
        return results
    
    # Include the original methods for backward compatibility
    def compute_vocabulary_overlap(self) -> Dict[str, Any]:
        """Compute vocabulary overlap analysis between all tokenizers."""
        results = {
            'per_tokenizer': {},
            'pairwise_overlaps': {},
            'vocabulary_sizes': {}
        }
        
        # Get vocabularies
        vocabularies = {}
        for name in self.tokenizer_names:
            try:
                vocab = set(self.tokenizers[name].get_vocab().keys())
                vocabularies[name] = vocab
                results['vocabulary_sizes'][name] = len(vocab)
            except AttributeError:
                logger.warning(f"Could not get vocabulary for {name}")
                vocabularies[name] = set()
                results['vocabulary_sizes'][name] = 0
        
        # Compute pairwise overlaps
        for i, tok1 in enumerate(self.tokenizer_names):
            for j, tok2 in enumerate(self.tokenizer_names):
                if i < j:  # Only compute upper triangle
                    vocab1, vocab2 = vocabularies[tok1], vocabularies[tok2]
                    intersection = vocab1.intersection(vocab2)
                    union = vocab1.union(vocab2)
                    
                    overlap_metrics = {
                        'intersection_size': len(intersection),
                        'union_size': len(union),
                        'jaccard_similarity': len(intersection) / len(union) if union else 0.0,
                        'overlap_percentage': len(intersection) / min(len(vocab1), len(vocab2)) if min(len(vocab1), len(vocab2)) > 0 else 0.0,
                        'unique_to_first': len(vocab1 - vocab2),
                        'unique_to_second': len(vocab2 - vocab1)
                    }
                    
                    results['pairwise_overlaps'][f"{tok1}_vs_{tok2}"] = overlap_metrics
        
        return results
    
