"""
Morphological data loader for various datasets.
"""

import os
import json
import re
import logging
from typing import Dict, List, Optional
import pandas as pd
from .constants import ISO639_1_to_ISO639_2, FINEWEB_to_ISO639_2

logger = logging.getLogger(__name__)


class MorphologicalDataLoader:
    """
    Loader for various morphological datasets including LADEC, MorphoLex, MorphyNet, and DagoBert.
    Provides unified interface for accessing morphological segmentations.
    """
    
    def __init__(self, morphological_config: Optional[Dict[str, str]] = None):
        """
        Initialize morphological data loader.
        
        Args:
            morphological_config: Dict mapping dataset names to file paths
                Example: {
                    'ladec': 'path/to/ladec.txt',
                    'morpholex': 'path/to/morpholex.csv',
                    'morphynet': 'path/to/morphynet.tsv',
                    'dagobert': 'path/to/dagobert.json'
                }
        """
        self.config = morphological_config or {}
        self.morphological_data = {}  # lang -> dataset -> word -> morphemes
        self.available_languages = set()
        
        # Performance optimization: Cache for repeated lookups
        self._lookup_cache = {}  # (word, language) -> morphemes
    
    def load_all_datasets(self) -> None:
        """Load all configured morphological datasets."""
        for dataset_name, file_path in self.config.items():
            if not os.path.exists(file_path):
                logger.warning(f"Morphological dataset file not found: {file_path}")
                continue
                
            try:
                if dataset_name.lower() == 'ladec':
                    self._load_ladec(file_path)
                elif dataset_name.lower() == 'morpholex':
                    self._load_morpholex(file_path)
                elif dataset_name.lower() == 'morphynet':
                    self._load_morphynet(file_path)
                elif dataset_name.lower() == 'dagobert':
                    self._load_dagobert(file_path)
                else:
                    logger.warning(f"Unknown morphological dataset: {dataset_name}")
            except Exception as e:
                logger.error(f"Error loading {dataset_name}: {e}")
    
    def _load_ladec(self, file_path: str) -> None:
        """Load LADEC dataset (Gagné et al., 2019)."""
        logger.info(f"Loading LADEC from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    word = parts[0].lower()
                    morphemes = parts[1].split('-')  # Assuming morphemes separated by '-'
                    lang = 'en'  # Assume English for LADEC
                    if lang not in self.morphological_data:
                        self.morphological_data[lang] = {}
                    if 'ladec' not in self.morphological_data[lang]:
                        self.morphological_data[lang]['ladec'] = {}
                    self.morphological_data[lang]['ladec'][word] = morphemes
                    self.available_languages.add(lang)
    
    def _load_morpholex(self, file_path: str) -> None:
        """Load MorphoLex dataset (Sánchez-Gutiérrez et al., 2018)."""
        logger.info(f"Loading MorphoLex from {file_path}")
        try:
            df = pd.read_csv(file_path)
            word_col = 'Word' if 'Word' in df.columns else 'word'
            morph_col = 'MorphoLexSegm' if 'MorphoLexSegm' in df.columns else 'morphemes'
            
            for _, row in df.iterrows():
                word = str(row[word_col]).lower()
                morphemes_str = str(row[morph_col])
                if pd.isna(morphemes_str):
                    continue
                morphemes = re.split(r'[.\-+]', morphemes_str)
                morphemes = [m.strip() for m in morphemes if m.strip()]
                
                lang = 'en'  # Assume English for MorphoLex
                if lang not in self.morphological_data:
                    self.morphological_data[lang] = {}
                if 'morpholex' not in self.morphological_data[lang]:
                    self.morphological_data[lang]['morpholex'] = {}
                self.morphological_data[lang]['morpholex'][word] = morphemes
                self.available_languages.add(lang)
        except Exception as e:
            logger.error(f"Error loading MorphoLex: {e}")
    
    def _load_morphynet(self, file_path: str) -> None:
        """Load MorphyNet dataset (Batsuren et al., 2021)."""
        logger.info(f"Loading MorphyNet from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    lang = parts[0]
                    word = parts[1].lower()
                    derivation = parts[2]
                    
                    # Parse derivation: base_word→derived_word:POS
                    if '→' in derivation and ':' in derivation:
                        base_derived, pos = derivation.rsplit(':', 1)
                        if '→' in base_derived:
                            base_word, derived_word_check = base_derived.split('→', 1)
                            
                            # Infer morphemes by analyzing the transformation
                            morphemes = self._infer_morphemes_from_derivation(base_word, word)
                            
                            if lang not in self.morphological_data:
                                self.morphological_data[lang] = {}
                            if 'morphynet' not in self.morphological_data[lang]:
                                self.morphological_data[lang]['morphynet'] = {}
                            self.morphological_data[lang]['morphynet'][word] = morphemes
                            self.available_languages.add(lang)

    def _infer_morphemes_from_derivation(self, base: str, derived: str) -> List[str]:
        """
        Infer morpheme segmentation from base→derived transformation.
        This is a heuristic approach that handles common morphological patterns.
        """
        base = base.lower()
        derived = derived.lower()
        
        # Find common prefix
        common_prefix_len = 0
        for i in range(min(len(base), len(derived))):
            if base[i] == derived[i]:
                common_prefix_len += 1
            else:
                break
        
        # Find common suffix (from the end)
        common_suffix_len = 0
        for i in range(1, min(len(base), len(derived)) + 1):
            if base[-i] == derived[-i]:
                common_suffix_len += 1
            else:
                break
        
        # Handle different morphological patterns
        if derived.startswith(base):
            # Suffixation: base + suffix
            suffix = derived[len(base):]
            if suffix:
                return [base, suffix]
            else:
                return [base]
        elif base.startswith(derived):
            # Truncation: derived is shorter than base
            return [derived]
        elif derived.endswith(base):
            # Prefixation: prefix + base
            prefix = derived[:-len(base)]
            if prefix:
                return [prefix, base]
            else:
                return [base]
        elif common_prefix_len > 0:
            # Complex derivation with stem change
            # Try to identify the stem and affixes
            stem = base[:common_prefix_len]
            
            # Look for common patterns
            if len(derived) > len(base):
                # Possibly prefix + stem + suffix
                remaining_derived = derived[common_prefix_len:]
                remaining_base = base[common_prefix_len:]
                
                # Check if it's a simple prefix addition
                if remaining_derived.endswith(remaining_base):
                    prefix = remaining_derived[:-len(remaining_base)]
                    return [prefix, base] if prefix else [base]
                else:
                    # More complex - just use stem + remainder
                    return [stem, remaining_derived] if remaining_derived else [stem]
            else:
                # Use the common stem plus any suffix
                suffix = derived[common_prefix_len:]
                return [stem, suffix] if suffix else [stem]
        else:
            # No clear relationship - return as single morpheme
            return [derived]

    def _load_dagobert(self, file_path: str) -> None:
        """Load DagoBert dataset."""
        logger.info(f"Loading DagoBert from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for entry in data:
            if 'word' in entry and 'morphemes' in entry:
                word = entry['word'].lower()
                morphemes = entry['morphemes']
                lang = entry.get('language', 'de')  # Default to German
                
                if lang not in self.morphological_data:
                    self.morphological_data[lang] = {}
                if 'dagobert' not in self.morphological_data[lang]:
                    self.morphological_data[lang]['dagobert'] = {}
                self.morphological_data[lang]['dagobert'][word] = morphemes
                self.available_languages.add(lang)
    
    def get_morphemes(self, word: str, language: str, dataset: Optional[str] = None) -> Optional[List[str]]:
        """
        Get morphological segmentation for a word with caching for performance.
        
        Args:
            word: The word to segment
            language: Language code
            dataset: Specific dataset to use, or None for any available
            
        Returns:
            List of morphemes or None if not found
        """
        word = word.lower()
        
        # Check cache first
        cache_key = (word, language, dataset)
        if cache_key in self._lookup_cache:
            return self._lookup_cache[cache_key]
        
        # Language normalization
        original_language = language
        if language not in self.morphological_data:
            if language in ISO639_1_to_ISO639_2 and ISO639_1_to_ISO639_2[language] in self.morphological_data:
                language = ISO639_1_to_ISO639_2[language]
            elif language in FINEWEB_to_ISO639_2 and FINEWEB_to_ISO639_2[language] in self.morphological_data:
                language = FINEWEB_to_ISO639_2[language]
            else:
                self._lookup_cache[cache_key] = None
                return None
        
        result = None
        if dataset:
            if dataset in self.morphological_data[language]:
                result = self.morphological_data[language][dataset].get(word)
        else:
            # Try all available datasets for this language
            for dataset_name, words in self.morphological_data[language].items():
                if word in words:
                    result = words[word]
                    break
        
        # Cache the result
        self._lookup_cache[cache_key] = result
        return result