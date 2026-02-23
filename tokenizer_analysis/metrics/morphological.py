"""
Morphological alignment metrics for tokenizer analysis.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict
import logging

from .base import BaseMetrics, TokenizedDataProcessor
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider
from ..loaders import MorphologicalDataLoader
from ..constants import Validation

logger = logging.getLogger(__name__)


class MorphologicalMetrics(BaseMetrics):
    """Morphological alignment analysis metrics."""
    
    def __init__(self, input_provider: InputProvider,
                 morphological_config: Optional[Dict[str, str]] = None):
        """
        Initialize morphological metrics.
        
        Args:
            input_provider: InputProvider instance
            morphological_config: Configuration for morphological datasets
        """
        super().__init__(input_provider)
        self.morphological_loader = MorphologicalDataLoader(morphological_config)
        if morphological_config:
            self.morphological_loader.load_all_datasets()

        self._token_cleaning_cache = {}
    
    def compute_morphological_alignment(self, word: str, tokenizer_tokens: List[str], 
                                          language: str) -> Optional[Dict[str, Any]]:
        """
        Compute morphological alignment metrics for a single word with improved
        token cleaning and boundary detection logic.
        
        Args:
            word: The original word.
            tokenizer_tokens: List of tokens from the tokenizer for the given word.
            language: Language code for fetching morphemes.
            
        Returns:
            Dictionary with alignment metrics or None if no morphological data.
        """
        # Early exit if no tokens
        if not tokenizer_tokens:
            return None
            
        morphemes = self.morphological_loader.get_morphemes(word, language)
        if not morphemes:
            return None
    
        # Clean tokenizer tokens (remove special markers if present)
        clean_tokens = []
        for token in tokenizer_tokens:
            if token in self._token_cleaning_cache:
                clean_token = self._token_cleaning_cache[token]
            else:
                clean_token = self._clean_token(token)
                self._token_cleaning_cache[token] = clean_token

            if clean_token:
                clean_tokens.append(clean_token)
    
        if not clean_tokens:
            return None
    
        # Reconstruct the word from the cleaned tokens to get accurate boundaries.
        reconstructed_from_clean = "".join(clean_tokens)
    
        # Calculate morpheme boundaries
        morpheme_boundaries = self._fix_morpheme_boundaries(morphemes, word)
        
        # Calculate token boundaries with validation 
        token_boundaries = []
        word_lower = word.lower()
        reconstructed_lower = reconstructed_from_clean.lower()
        
        if word_lower == reconstructed_lower:
            # Direct reconstruction - calculate boundaries 
            pos = 0
            for token in clean_tokens:
                token_len = len(token)
                token_boundaries.append((pos, pos + token_len))
                pos += token_len
        else:
            # Fuzzy alignment fallback (only if needed)
            token_boundaries = self._fuzzy_align_tokens_optimized(word, clean_tokens, word_lower)
        
        # Fast validation check
        if not token_boundaries or not self._validate_boundaries_fast(token_boundaries, len(word)):
            # Final fallback: proportional distribution (optimized)
            if clean_tokens:
                num_tokens = len(clean_tokens)
                word_len = len(word)
                token_boundaries = [(int(i * word_len / num_tokens), 
                                   int((i + 1) * word_len / num_tokens) if i < num_tokens - 1 else word_len) 
                                  for i in range(num_tokens)]
        
        if not token_boundaries:
            return None
        
        morpheme_boundary_set = set()
        for start, end in morpheme_boundaries:
            morpheme_boundary_set.add(end)
        morpheme_boundary_set.discard(len(word))
    
        token_boundary_set = set()
        for start, end in token_boundaries:
            token_boundary_set.add(end)
        token_boundary_set.discard(len(word))
        
        true_positives = len(morpheme_boundary_set.intersection(token_boundary_set))
        false_positives = len(token_boundary_set) - true_positives
        false_negatives = len(morpheme_boundary_set) - true_positives
        
        # Precision = TP / (TP + FP), Recall = TP / (TP + FN)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Morpheme preservation score
        preserved_morphemes = sum(1 for morpheme in morphemes if morpheme in clean_tokens)
        morpheme_preservation = preserved_morphemes / len(morphemes) if morphemes else 0.0
        
        # Over-segmentation score
        over_segmentation = max(0, len(clean_tokens) - len(morphemes)) / len(morphemes) if morphemes else 0.0
        
        return {
            'boundary_precision': precision,
            'boundary_recall': recall,
            'boundary_f1': f1,
            'morpheme_preservation': morpheme_preservation,
            'over_segmentation': over_segmentation,
            'morpheme_count': len(morphemes),
            'token_count': len(clean_tokens),
            'alignment_type': 'boundary'
        }
    
    def _fuzzy_align_tokens_optimized(self, word: str, tokens: List[str], word_lower: str = None) -> List[tuple]:
        """
        Optimized fuzzy alignment with pre-computed lowercase and early exits.
        
        Args:
            word: Original word
            tokens: List of tokenizer tokens
            word_lower: Pre-computed lowercase word (optional)
            
        Returns:
            List of (start, end) tuples for token boundaries
        """
        if word_lower is None:
            word_lower = word.lower()
        
        boundaries = []
        pos = 0
        word_len = len(word)
        
        for token in tokens:
            if pos >= word_len:
                break
                
            token_lower = token.lower()
            token_len = len(token_lower)
            
            # Find the token in the remaining part of the word
            remaining_word = word_lower[pos:]
            token_pos = remaining_word.find(token_lower)
            
            if token_pos >= 0:
                start = pos + token_pos
                end = start + token_len
                boundaries.append((start, end))
                pos = end
            else:
                # Token not found exactly - use proportional positioning
                remaining_len = len(remaining_word)
                if remaining_len > 0:
                    # Estimate position proportionally
                    estimated_len = min(token_len, remaining_len)
                    end = pos + estimated_len
                    boundaries.append((pos, end))
                    pos = end
                else:
                    break
        
        return boundaries
    
    def _validate_boundaries_fast(self, boundaries: List[tuple], word_length: int) -> bool:
        """
        Fast validation with minimal logging for performance.
        
        Args:
            boundaries: List of (start, end) tuples
            word_length: Length of the original word
            
        Returns:
            True if boundaries are valid, False otherwise
        """
        if not boundaries:
            return False
        
        prev_end = 0
        for start, end in boundaries:
            # Quick bounds and overlap check
            if start < 0 or end > word_length or start >= end or start < prev_end:
                return False
            prev_end = end
        
        return True
    
    def _validate_boundaries(self, boundaries: List[tuple], word_length: int) -> bool:
        """
        Validate that boundaries are well-formed and within word bounds.
        
        Args:
            boundaries: List of (start, end) tuples
            word_length: Length of the original word
            
        Returns:
            True if boundaries are valid, False otherwise
        """
        if not boundaries:
            return False
        
        prev_end = 0
        for start, end in boundaries:
            # Check bounds
            if start < 0 or end > word_length or start >= end:
                logger.debug(f"Invalid boundary ({start}, {end}) for word length {word_length}")
                return False
            
            # Check for gaps or overlaps
            if start < prev_end:
                logger.debug(f"Overlapping boundaries: previous ended at {prev_end}, current starts at {start}")
                return False
            
            prev_end = end
        
        return True
    
    def _fix_morpheme_boundaries(self, morphemes: List[str], word: str) -> List[tuple]:
        """
        Compute morpheme boundaries with validation and error correction.
        
        Args:
            morphemes: List of morpheme strings
            word: Original word
            
        Returns:
            List of validated (start, end) boundary tuples
        """
        if not morphemes or not word:
            return []
        
        # Try direct concatenation first
        morpheme_boundaries = []
        pos = 0
        
        for morpheme in morphemes:
            morpheme_start = pos
            pos += len(morpheme)
            morpheme_boundaries.append((morpheme_start, pos))
        
        # Validate boundaries
        if self._validate_boundaries(morpheme_boundaries, len(word)):
            reconstructed = ''.join(morphemes)
            if reconstructed.lower() == word.lower():
                return morpheme_boundaries
        
        # Fallback: Try to find morphemes within the word
        logger.debug(f"Direct concatenation failed for word '{word}' with morphemes {morphemes}")
        fallback_boundaries = []
        word_lower = word.lower()
        pos = 0
        
        for morpheme in morphemes:
            morpheme_lower = morpheme.lower()
            # Find morpheme in remaining part of word
            remaining_word = word_lower[pos:]
            morph_pos = remaining_word.find(morpheme_lower)
            
            if morph_pos >= 0:
                start = pos + morph_pos
                end = start + len(morpheme_lower)
                fallback_boundaries.append((start, end))
                pos = end
            else:
                # Can't find this morpheme - skip it
                logger.debug(f"Could not locate morpheme '{morpheme}' in word '{word}'")
                continue
        
        if self._validate_boundaries(fallback_boundaries, len(word)):
            return fallback_boundaries
        
        # Ultimate fallback: distribute morphemes proportionally
        logger.debug(f"Using proportional fallback for word '{word}' with morphemes {morphemes}")
        if len(morphemes) == 1:
            return [(0, len(word))]
        
        proportional_boundaries = []
        step = len(word) / len(morphemes)
        for i in range(len(morphemes)):
            start = int(i * step)
            end = int((i + 1) * step) if i < len(morphemes) - 1 else len(word)
            proportional_boundaries.append((start, end))
        
        return proportional_boundaries
    
    def _align_words_to_tokens(self, text: str, tokens: List[str]) -> List[tuple]:
        """
        Aligns words in a text to their corresponding tokens using a more robust
        reconstruction-based approach.
    
        Args:
            text: Original text string.
            tokens: List of token strings from the tokenizer.
    
        Returns:
            List of (word, corresponding_tokens) tuples.
        """
        if not tokens:
            return []
    
        alignments = []
        current_word_tokens = []
        reconstructed_word = ""
        
        # Clean words from original text for comparison
        original_words = [word.strip('.,!?;:"()[]{}') for word in text.lower().split()]
        original_word_iter = iter(original_words)
        current_original_word = next(original_word_iter, None)
    
        for i, token in enumerate(tokens):
            # Determine if a token starts a new word. This is a crucial heuristic.
            # True if it has a space prefix, or if it's the very first token.
            is_new_word_starter = (token.startswith('Ġ') or token.startswith(' ')) and current_word_tokens
    
            if is_new_word_starter:
                # Finalize the previous word
                if current_word_tokens:
                    alignments.append((reconstructed_word, current_word_tokens))
                
                # Start a new word
                current_word_tokens = [token]
                clean_token = token[1:] # Strip the space prefix
                reconstructed_word = clean_token
            else:
                # Continue the current word
                clean_token = token
                if token.startswith('##'):
                    clean_token = token[2:]
                
                current_word_tokens.append(token)
                reconstructed_word += clean_token
    
        # Add the last word
        if current_word_tokens:
            alignments.append((reconstructed_word, current_word_tokens))
    
        # This part is a heuristic to handle punctuation that becomes its own token.
        # It merges the alignment of a punctuation-only word with the previous word.
        final_alignments = []
        i = 0
        while i < len(alignments):
            current_word, current_tokens = alignments[i]
            # If the "word" is just punctuation, merge it with the previous word.
            if i > 0 and all(c in '.,!?;:"()[]{}' for c in current_word):
                prev_word, prev_tokens = final_alignments[-1]
                final_alignments[-1] = (prev_word, prev_tokens + current_tokens)
            else:
                final_alignments.append(alignments[i])
            i += 1
        
        # Final step: Match reconstructed words to original words to get the canonical form
        result = []
        for i, (recon_word, recon_tokens) in enumerate(final_alignments):
            if i < len(original_words):
                result.append((original_words[i], recon_tokens))
        
        # If there are more tokens than words (e.g., trailing punctuation), append to the last word
        if len(tokens) > sum(len(t) for w, t in result):
            remaining_token_idx = sum(len(t) for w, t in result)
            if result:
                w, t = result[-1]
                result[-1] = (w, t + tokens[remaining_token_idx:])
    
        return result

    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """Compute morphological alignment metrics."""
        if tokenized_data is None:
            tokenized_data = self.input_provider.get_tokenized_data()
        
        if not self.morphological_loader.morphological_data:
            logger.info("No morphological data loaded. Skipping morphological analysis.")
            return {
                'morphological_alignment': {
                    'message': 'No morphological data available'
                }
            }
        
        results = {
            'per_tokenizer': {},
            'per_language': {},
            'summary': {}
        }
        
        # Initialize metrics for each tokenizer
        for name in self.tokenizer_names:
            results['per_tokenizer'][name] = {
                'boundary_precision': {},
                'boundary_recall': {},
                'boundary_f1': {},
                'morpheme_preservation': {},
                'over_segmentation': {}
            }
        
        total_words_analyzed = 0
        
        for name in self.tokenizer_names:
            
            if name not in tokenized_data:
                continue
            
            # Group tokenized data by language
            lang_data = {}
            for data in tokenized_data[name]:
                lang = data.language
                if lang not in lang_data:
                    lang_data[lang] = []
                lang_data[lang].append(data)
            
            for lang, data_list in lang_data.items():
                logger.info(f"Analyzing morphological alignment for {lang} with {name}...")
                
                # Initialize metrics for this language
                for metric in results['per_tokenizer'][name]:
                    if lang not in results['per_tokenizer'][name][metric]:
                        results['per_tokenizer'][name][metric][lang] = {'values': [], 'count': 0, 'mean': 0.0}
                
                lang_words_analyzed = 0
                tokenizer_obj = self.input_provider.get_tokenizer(name)
                
                for tokenized_data_item in data_list:
                    # Convert token IDs to tokens
                    tokens = self._convert_ids_to_tokens(tokenizer_obj, tokenized_data_item.tokens)
                    
                    # Get word-token alignment for this text
                    word_token_alignments = self._align_words_to_tokens(tokenized_data_item.text, tokens)
                    
                    # Process words in batches to reduce function call overhead
                    for word, word_tokens in word_token_alignments:
                        if len(word) < Validation.MIN_WORD_LENGTH:  # Skip very short words
                            continue
                        
                        # Compute alignment metrics
                        alignment = self.compute_morphological_alignment(
                            word, word_tokens, lang
                        )
                        
                        if alignment:
                            lang_words_analyzed += 1
                            # Update metrics efficiently
                            for metric, value in alignment.items():
                                if metric in results['per_tokenizer'][name] and isinstance(value, (int, float)):
                                    results['per_tokenizer'][name][metric][lang]['values'].append(value)
                
                # Compute statistics for this language
                for metric in ['boundary_precision', 'boundary_recall', 'boundary_f1',
                             'morpheme_preservation', 'over_segmentation']:
                    if lang in results['per_tokenizer'][name][metric]:
                        values = results['per_tokenizer'][name][metric][lang]['values']
                        if values:
                            results['per_tokenizer'][name][metric][lang]['count'] = len(values)
                            results['per_tokenizer'][name][metric][lang]['mean'] = np.mean(values)
                            results['per_tokenizer'][name][metric][lang]['std'] = np.std(values)
                        else:
                            results['per_tokenizer'][name][metric][lang]['count'] = 0
                            results['per_tokenizer'][name][metric][lang]['mean'] = 0.0
                
                total_words_analyzed += lang_words_analyzed
        
        # Compute summary statistics
        for name in self.tokenizer_names:
            if any(results['per_tokenizer'][name]['boundary_f1'].values()):
                f1_values = []
                preservation_values = []
                
                for lang in results['per_tokenizer'][name]['boundary_f1']:
                    if results['per_tokenizer'][name]['boundary_f1'][lang]['count'] > 0:
                        f1_values.append(results['per_tokenizer'][name]['boundary_f1'][lang]['mean'])
                        preservation_values.append(results['per_tokenizer'][name]['morpheme_preservation'][lang]['mean'])
                
                if f1_values:
                    n_languages = len(f1_values)
                    results['summary'][name] = {
                        'avg_boundary_f1': np.mean(f1_values),
                        'avg_morpheme_preservation': np.mean(preservation_values),
                        'languages_analyzed': n_languages,
                        'total_words_analyzed': total_words_analyzed,
                        'avg_boundary_f1_std_err': np.std(f1_values) / np.sqrt(n_languages) if n_languages > 1 else 0.0,
                        'avg_morpheme_preservation_std_err': np.std(preservation_values) / np.sqrt(n_languages) if n_languages > 1 else 0.0
                    }
        
        return {'morphological_alignment': results}
    
    def print_results(self, results: Dict[str, Any]):
        """Print morphological metrics results."""
        if 'morphological_alignment' not in results:
            return
            
        morphological_data = results['morphological_alignment']
        
        # Handle case where no morphological data is available
        if 'message' in morphological_data:
            print(f"\n🔤 MORPHOLOGICAL ANALYSIS")
            print("-" * 40)
            print(f"Status: {morphological_data['message']}")
            return
        
        print("\n" + "="*60)
        print("MORPHOLOGICAL ALIGNMENT RESULTS")
        print("="*60)
        
        # Print summary statistics
        if 'summary' in morphological_data:
            print(f"\n🎯 SUMMARY STATISTICS")
            print("-" * 40)
            
            for tok_name in self.tokenizer_names:
                if tok_name in morphological_data['summary']:
                    summary = morphological_data['summary'][tok_name]
                    avg_f1 = summary.get('avg_boundary_f1', 0.0)
                    avg_preservation = summary.get('avg_morpheme_preservation', 0.0)
                    languages_count = summary.get('languages_analyzed', 0)
                    words_count = summary.get('total_words_analyzed', 0)
                    
                    print(f"{tok_name:20}:")
                    print(f"  {'Boundary F1':15}: {avg_f1:.3f}")
                    print(f"  {'Preservation':15}: {avg_preservation:.3f}")
                    print(f"  {'Languages':15}: {languages_count}")
                    print(f"  {'Words analyzed':15}: {words_count:,}")
        
        # Print detailed per-tokenizer results
        if 'per_tokenizer' in morphological_data:
            print(f"\n📊 DETAILED METRICS")
            print("-" * 60)
            
            for tok_name in self.tokenizer_names:
                if tok_name not in morphological_data['per_tokenizer']:
                    continue
                    
                tok_data = morphological_data['per_tokenizer'][tok_name]
                
                print(f"\n{tok_name}:")
                print("-" * 30)
                
                # Print boundary F1 scores by language
                if 'boundary_f1' in tok_data:
                    print("  Boundary F1 by language:")
                    for lang, lang_data in tok_data['boundary_f1'].items():
                        if lang_data.get('count', 0) > 0:
                            mean_f1 = lang_data.get('mean', 0.0)
                            std_f1 = lang_data.get('std', 0.0)
                            count = lang_data.get('count', 0)
                            print(f"    {lang:12}: {mean_f1:.3f} ± {std_f1:.3f} (n={count})")
                
                # Print morpheme preservation by language
                if 'morpheme_preservation' in tok_data:
                    print("  Morpheme preservation by language:")
                    for lang, lang_data in tok_data['morpheme_preservation'].items():
                        if lang_data.get('count', 0) > 0:
                            mean_pres = lang_data.get('mean', 0.0)
                            std_pres = lang_data.get('std', 0.0)
                            count = lang_data.get('count', 0)
                            print(f"    {lang:12}: {mean_pres:.3f} ± {std_pres:.3f} (n={count})")
        
        print("\n" + "="*60)