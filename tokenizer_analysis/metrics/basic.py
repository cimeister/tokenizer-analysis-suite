from typing import Dict, List, Any, Optional, Set
import numpy as np
from collections import Counter
import logging
import unicodedata

from .base import BaseMetrics

logger = logging.getLogger(__name__)


class BasicTokenizationMetrics(BaseMetrics):
    
    def compute_compression_ratio(self, language_texts: Dict[str, List[str]], 
                                          all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None,
                                          use_bytes: bool = True) -> Dict[str, Any]:
        """
        Compute compression ratios: average of individual (tokens/bytes) ratios.
        
        Args:
            language_texts: Text data by language
            all_encodings: Pre-computed encodings
            use_bytes: If True, use byte length; if False, use character length
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
                    if len(text) > 0:
                        # Use bytes or characters as specified
                        text_length = len(text.encode('utf-8')) if use_bytes else len(text)
                        ratio = text_length / len(tokens)
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
        results['metadata'] = {'unit': 'bytes' if use_bytes else 'characters'}
        
        # Compute pairwise comparisons
        global_ratios = {name: results['per_tokenizer'][name]['global'] 
                        for name in self.tokenizer_names}
        results['pairwise_comparisons'] = self.compute_pairwise_ratios(
            global_ratios, 'compression_ratio'
        )
        
        return results
    
    def compute_fertility_analysis(self, language_texts: Dict[str, List[str]], 
                                    all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None) -> Dict[str, Any]:
        """
        Compute comprehensive fertility analysis: both whitespace-delimited and character-based.
        
        Returns:
            Dict with 'whitespace_fertility' and 'character_fertility' results
        """
        if all_encodings is None:
            all_encodings = self.encode_texts_batch(language_texts)
        
        language_texts, all_encodings = self.filter_valid_data(language_texts, all_encodings)
        
        results = {
            'whitespace_fertility': {
                'per_tokenizer': {},
                'per_language': {},
                'pairwise_comparisons': {}
            },
            'character_fertility': {
                'per_tokenizer': {},
                'per_language': {},
                'pairwise_comparisons': {}
            },
            'metadata': {
                'whitespace_description': 'Tokens per whitespace-delimited word',
                'character_description': 'Tokens per character'
            }
        }
        
        for tok_name in self.tokenizer_names:
            # Whitespace-delimited fertility (tokens per word)
            ws_per_lang_fertility = {}
            ws_all_fertilities = []
            
            # Character-based fertility (tokens per character)
            char_per_lang_fertility = {}
            char_all_fertilities = []
            
            for lang, texts in language_texts.items():
                if not texts or lang not in all_encodings[tok_name]:
                    continue
                
                ws_lang_fertilities = []
                char_lang_fertilities = []
                
                for text, tokens in zip(texts, all_encodings[tok_name][lang]):
                    if text.strip():
                        # Whitespace-delimited fertility (tokens per word)
                        words = text.split()
                        if len(words) > 0:
                            ws_fertility = len(tokens) / len(words)
                            ws_lang_fertilities.append(ws_fertility)
                            ws_all_fertilities.append(ws_fertility)
                        
                        # Character-based fertility (tokens per character)
                        chars = len(text)
                        if chars > 0:
                            char_fertility = len(tokens) / chars
                            char_lang_fertilities.append(char_fertility)
                            char_all_fertilities.append(char_fertility)
                
                # Store per-language statistics
                if ws_lang_fertilities:
                    ws_per_lang_fertility[lang] = {
                        'mean': np.mean(ws_lang_fertilities),
                        'std': np.std(ws_lang_fertilities),
                        'median': np.median(ws_lang_fertilities)
                    }
                
                if char_lang_fertilities:
                    char_per_lang_fertility[lang] = {
                        'mean': np.mean(char_lang_fertilities),
                        'std': np.std(char_lang_fertilities),
                        'median': np.median(char_lang_fertilities)
                    }
            
            # Global fertility statistics
            ws_global_stats = self.compute_basic_stats(ws_all_fertilities)
            char_global_stats = self.compute_basic_stats(char_all_fertilities)
            
            results['whitespace_fertility']['per_tokenizer'][tok_name] = {
                'global': ws_global_stats,
                'per_language': ws_per_lang_fertility
            }
            
            results['character_fertility']['per_tokenizer'][tok_name] = {
                'global': char_global_stats,
                'per_language': char_per_lang_fertility
            }
        
        # Compute pairwise comparisons for both fertility types
        ws_global_fertilities = {name: results['whitespace_fertility']['per_tokenizer'][name]['global']['mean'] 
                                for name in self.tokenizer_names}
        char_global_fertilities = {name: results['character_fertility']['per_tokenizer'][name]['global']['mean'] 
                                  for name in self.tokenizer_names}
        
        results['whitespace_fertility']['pairwise_comparisons'] = self.compute_pairwise_ratios(
            ws_global_fertilities, 'whitespace_fertility'
        )
        results['character_fertility']['pairwise_comparisons'] = self.compute_pairwise_ratios(
            char_global_fertilities, 'character_fertility'
        )
        
        return results
    
    def compute_token_length_analysis(self, use_bytes: bool = True) -> Dict[str, Any]:
        """
        Analyze token lengths with proper handling of special tokens.
        
        Args:
            use_bytes: If True, use byte lengths as primary metric; if False, use character lengths
        """
        results = {
            'per_tokenizer': {}, 
            'detailed_analysis': {},
            'metadata': {'primary_unit': 'bytes' if use_bytes else 'characters'}
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
    
    def validate_metric_sanity(self, results: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Perform sanity checks on computed metrics.
        
        Returns:
            Dict mapping metric names to lists of warning messages
        """
        warnings = {'compression_ratio': [], 'fertility (whitespace-delimited)': [], 'token_length': []}
        
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
        if 'fertility (whitespace-delimited)' in results:
            for tok_name, tok_results in results['fertility (whitespace-delimited)']['per_tokenizer'].items():
                fertility = tok_results['global']['mean']
                if fertility < 0.5:
                    warnings['fertility (whitespace-delimited)'].append(f"{tok_name}: Very low fertility ({fertility:.2f}) - check calculation")
                elif fertility > 10.0:
                    warnings['fertility (whitespace-delimited)'].append(f"{tok_name}: Very high fertility ({fertility:.2f}) - may indicate character-level tokenization")
        
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
                all_encodings: Optional[Dict[str, Dict[str, List[List[int]]]]] = None,
                use_bytes: bool = True) -> Dict[str, Any]:
        """Compute all corrected basic tokenization metrics using bytes as default unit."""
        results = {}
        
        results['compression_ratio'] = self.compute_compression_ratio(language_texts, all_encodings, use_bytes=use_bytes)
        
        # Comprehensive fertility analysis (both whitespace-delimited and character-based)
        fertility_results = self.compute_fertility_analysis(language_texts, all_encodings)
        results.update(fertility_results)
        
        results['token_length'] = self.compute_token_length_analysis(use_bytes=use_bytes)
        results['vocabulary_overlap'] = self.compute_vocabulary_overlap()
        
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
    
