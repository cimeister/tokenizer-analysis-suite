"""
Basic tokenization metrics using unified TokenizedData interface.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from collections import Counter
import logging

from .base import BaseMetrics, TokenizedDataProcessor
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider
from ..config import TextMeasurementConfig, TextMeasurer, DEFAULT_TEXT_MEASUREMENT_CONFIG
from ..config.language_metadata import LanguageMetadata
from ..constants import (
    FALLBACK_WORDS_PER_TOKEN
)

logger = logging.getLogger(__name__)


class BasicTokenizationMetrics(BaseMetrics):
    """Basic tokenization metrics: fertility, token_length, type_token_ratio, vocabulary_utilization, avg_tokens_per_line."""
    
    def __init__(self, 
                 input_provider: InputProvider,
                 measurement_config: Optional[TextMeasurementConfig] = None,
                 language_metadata: Optional[LanguageMetadata] = None):
        """
        Initialize basic metrics.
        
        Args:
            input_provider: InputProvider instance
            measurement_config: Configuration for text measurement method
            language_metadata: Optional language metadata for grouping
        """
        super().__init__(input_provider)
        self.measurement_config = measurement_config or DEFAULT_TEXT_MEASUREMENT_CONFIG
        self.language_metadata = language_metadata
        self.text_measurer = TextMeasurer(self.measurement_config)
    
    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """
        Compute basic tokenization metrics.
        
        Args:
            tokenized_data: Optional tokenized data dict. If None, uses input_provider data.
            
        Returns:
            Dictionary with basic metrics results
        """
        if tokenized_data is None:
            tokenized_data = self.get_tokenized_data()
        
        results = {}
        
        # Compute fertility analysis
        results.update(self.compute_fertility_analysis(tokenized_data))
        
        # Compute token length analysis
        results.update(self.compute_token_length_analysis(tokenized_data))
        
        # Compute vocabulary utilization
        results.update(self.compute_vocabulary_utilization_analysis(tokenized_data))
        
        # Compute type-token ratio
        results.update(self.compute_type_token_ratio_analysis(tokenized_data))
        
        # Compute average tokens per line
        results.update(self.compute_avg_tokens_per_line_analysis(tokenized_data))
        
        return results
    
    def compute_fertility_analysis(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute fertility analysis using configured normalization method.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with fertility results
        """
        normalization_unit = self.measurement_config.method.value.lower()
        
        results = {
            'fertility': {
                'per_tokenizer': {},
                'per_language': {},
                'pairwise_comparisons': {},
                'metadata': {
                    'normalization_method': normalization_unit,
                    'description': f'Average number of tokens per {normalization_unit[:-1]}',
                    'short_description': f'tokens/{normalization_unit[:-1]}'
                }
            }
        }
        
        global_values = {}
        
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
            
            tok_data = tokenized_data[tok_name]
            
            # Compute global fertility
            global_fertility = self._compute_fertility_stats(tok_data, normalization_unit)
            
            # Compute per-language fertility
            per_lang_fertility = {}
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
            
            for language, lang_data in lang_groups.items():
                lang_fertility = self._compute_fertility_stats(lang_data, normalization_unit)
                per_lang_fertility[language] = lang_fertility
            
            results['fertility']['per_tokenizer'][tok_name] = {
                'global': global_fertility,
                'per_language': per_lang_fertility
            }
            
            global_values[tok_name] = global_fertility.get('mean', 0.0)
        
        # Compute pairwise comparisons
        if len(global_values) >= 2:
            results['fertility']['pairwise_comparisons'] = self.compute_pairwise_comparisons(
                global_values, 'fertility'
            )
        
        return results
    
    def _compute_fertility_stats(self, tokenized_data: List[TokenizedData], 
                                normalization_unit: str) -> Dict[str, float]:
        """Compute fertility statistics for a list of TokenizedData."""
        if not tokenized_data:
            return self.empty_stats()
        
        fertilities = []
        
        for data in tokenized_data:
            num_tokens = len(data.tokens)
            
            if data.text:
                num_units = self.text_measurer.get_unit_count(data.text)
            else:
                if normalization_unit == 'words':
                    num_units = max(1, int(num_tokens * FALLBACK_WORDS_PER_TOKEN))  # Rough estimate
                else:
                    continue  # Skip if no text
            
            if num_units > 0:
                fertility = num_tokens / num_units
                fertilities.append(fertility)
        
        if not fertilities:
            return self.empty_stats()
        
        return self.compute_basic_stats(fertilities)
    
    def compute_token_length_analysis(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute token length analysis.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with token length results
        """
        results = {
            'token_length': {
                'per_tokenizer': {},
                'metadata': {
                    'primary_unit': 'characters',
                    'description': 'Average character length per token'
                }
            }
        }
        
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
            
            tok_data = tokenized_data[tok_name]
            
            # Calculate character lengths where text is available
            char_lengths = []
            for data in tok_data:
                if data.text and data.tokens:
                    avg_char_length = len(data.text) / len(data.tokens)
                    char_lengths.append(avg_char_length)
            
            if char_lengths:
                char_stats = self.compute_basic_stats(char_lengths)
                results['token_length']['per_tokenizer'][tok_name] = {
                    'character_length': char_stats,
                    'primary_length': char_stats
                }
            else:
                empty_stats = self.empty_stats()
                results['token_length']['per_tokenizer'][tok_name] = {
                    'character_length': empty_stats,
                    'primary_length': empty_stats
                }
        
        return results
    
    def compute_vocabulary_utilization_analysis(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute vocabulary utilization analysis.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with vocabulary utilization results
        """
        results = {
            'vocabulary_utilization': {
                'per_tokenizer': {},
                'metadata': {
                    'description': 'Proportion of vocabulary actually used',
                    'metric_range': '[0.0, 1.0]'
                }
            }
        }
        
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
            
            tok_data = tokenized_data[tok_name]
            vocab_size = self.get_vocab_size(tok_name)
            
            # Compute global utilization
            global_util = self._compute_vocabulary_utilization(tok_data, vocab_size)
            
            # Compute per-language utilization
            per_lang_util = {}
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
            
            for language, lang_data in lang_groups.items():
                lang_util = self._compute_vocabulary_utilization(lang_data, vocab_size)
                per_lang_util[language] = lang_util
            
            results['vocabulary_utilization']['per_tokenizer'][tok_name] = {
                'global_utilization': global_util['utilization'],
                'global_used_tokens': global_util['used_tokens'],
                'global_vocab_size': global_util['vocab_size'],
                'per_language': per_lang_util
            }
        
        return results
    
    def _compute_vocabulary_utilization(self, tokenized_data: List[TokenizedData], vocab_size: int) -> Dict[str, Any]:
        """Compute vocabulary utilization for a list of TokenizedData."""
        unique_tokens = TokenizedDataProcessor.get_unique_tokens(tokenized_data)
        used_tokens = len(unique_tokens)
        
        return {
            'utilization': self.safe_divide(used_tokens, vocab_size, 0.0),
            'used_tokens': used_tokens,
            'vocab_size': vocab_size,
            'unused_tokens': vocab_size - used_tokens
        }
    
    def compute_type_token_ratio_analysis(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute type-token ratio analysis.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with type-token ratio results
        """
        results = {
            'type_token_ratio': {
                'per_tokenizer': {},
                'per_language': {},
                'metadata': {
                    'description': 'Ratio of unique tokens to total tokens (lexical diversity)',
                    'metric_range': '[0.0, 1.0]'
                }
            }
        }
        
        # Global per-language TTR (aggregated across tokenizers)
        all_languages = set()
        for tok_data in tokenized_data.values():
            for data in tok_data:
                all_languages.add(data.language)
        
        per_language_results = {}
        for language in all_languages:
            per_language_results[language] = {}
            
            for tok_name in self.tokenizer_names:
                if tok_name not in tokenized_data:
                    continue
                
                # Get data for this tokenizer and language
                lang_data = [data for data in tokenized_data[tok_name] 
                           if data.language == language]
                
                if lang_data:
                    ttr_stats = self._compute_type_token_ratio(lang_data)
                    per_language_results[language][tok_name] = ttr_stats['ttr']
        
        results['type_token_ratio']['per_language'] = per_language_results
        
        # Global per-tokenizer TTR
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
            
            tok_data = tokenized_data[tok_name]
            global_ttr = self._compute_type_token_ratio(tok_data)
            
            results['type_token_ratio']['per_tokenizer'][tok_name] = {
                'global_ttr': global_ttr['ttr'],
                'global_types': global_ttr['types'],
                'global_tokens': global_ttr['tokens']
            }
        
        return results
    
    def _compute_type_token_ratio(self, tokenized_data: List[TokenizedData]) -> Dict[str, Any]:
        """Compute type-token ratio for a list of TokenizedData."""
        all_tokens = TokenizedDataProcessor.flatten_all_tokens(tokenized_data)
        unique_tokens = set(all_tokens)
        
        total_tokens = len(all_tokens)
        unique_count = len(unique_tokens)
        
        return {
            'ttr': self.safe_divide(unique_count, total_tokens, 0.0),
            'types': unique_count,
            'tokens': total_tokens
        }
    
    def compute_avg_tokens_per_line_analysis(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute average tokens per line analysis.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with average tokens per line results
        """
        results = {
            'avg_tokens_per_line': {
                'per_tokenizer': {},
                'metadata': {
                    'description': 'Average number of tokens per line of text',
                    'unit': 'tokens/line'
                }
            }
        }
        
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
            
            tok_data = tokenized_data[tok_name]
            
            # Calculate tokens per line where text is available
            tokens_per_line = []
            total_lines = 0
            
            for data in tok_data:
                if data.text and data.tokens:
                    lines = data.text.split('\n')
                    num_lines = len(lines)
                    total_lines += num_lines
                    
                    if num_lines > 0:
                        tpl = len(data.tokens) / num_lines
                        tokens_per_line.append(tpl)
            
            if tokens_per_line:
                tpl_stats = self.compute_basic_stats(tokens_per_line)
                results['avg_tokens_per_line']['per_tokenizer'][tok_name] = {
                    'global_avg': tpl_stats['mean'],
                    'global_std': tpl_stats['std'],
                    'global_std_err': tpl_stats['std_err'],
                    'total_lines': total_lines,
                    'stats': tpl_stats
                }
            else:
                results['avg_tokens_per_line']['per_tokenizer'][tok_name] = {
                    'global_avg': 0.0,
                    'global_std': 0.0,
                    'global_std_err': 0.0,
                    'total_lines': 0,
                    'stats': self.empty_stats()
                }
        
        return results