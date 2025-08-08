"""
MorphScore metrics for tokenizer analysis.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging

from .base import BaseMetrics
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider, RawTokenizationProvider
#TODO: move this mapping
from ..loaders.constants import ISO639_1_to_FLORES

logger = logging.getLogger(__name__)

try:
    from morphscore import MorphScore
    MORPHSCORE_AVAILABLE = True
except ImportError:
    logger.warning("MorphScore library not available. MorphScore metrics will be disabled.")
    MORPHSCORE_AVAILABLE = False


class MorphScoreMetrics(BaseMetrics):
    """MorphScore metrics for tokenizer evaluation."""
    
    def __init__(self, 
                 input_provider: InputProvider,
                 data_dir: str = "morphscore_data",
                 language_subset: Optional[List[str]] = None,
                 by_split: bool = False,
                 freq_scale: bool = True,
                 exclude_single_tok: bool = False
                ):
        """
        Initialize MorphScore metrics.
        
        Args:
            input_provider: InputProvider instance
            data_dir: Path to morphological data directory
            language_subset: Optional list of language codes to evaluate
            by_split: Whether to evaluate train/val/test separately
            freq_scale: Whether to use frequency scaling
            exclude_single_tok: Whether to exclude single token words
        """
        super().__init__(input_provider)
        
        if not MORPHSCORE_AVAILABLE:
            raise ImportError("MorphScore library is required for MorphScore metrics")
        
        # Validate that input provider supports tokenizer access
        if not isinstance(input_provider, RawTokenizationProvider):
            raise ValueError("MorphScore metrics require RawTokenizationProvider (tokenizer access)")
        
        # Store MorphScore configuration
        self.data_dir = data_dir
        self.language_subset = language_subset
        self.by_split = by_split
        self.freq_scale = freq_scale
        self.exclude_single_tok = exclude_single_tok
        
        # Get available languages from input provider
        self.available_languages = input_provider.get_languages()
        
        # Use provided subset or all available languages
        if language_subset is None:
            self.target_languages = self.available_languages
        else:
            # Filter to only languages available in the input provider
            self.target_languages = [lang for lang in language_subset if lang in self.available_languages]

        self.target_languages = [ISO639_1_to_FLORES.get(lang, lang) for lang in self.target_languages]
        logger.info(f"MorphScore metrics initialized for {len(self.target_languages)} languages: {self.target_languages}")
    
    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """
        Compute MorphScore metrics.
        
        Args:
            tokenized_data: Optional tokenized data (not used for MorphScore)
            
        Returns:
            Dictionary with MorphScore results
        """
        if not MORPHSCORE_AVAILABLE:
            return {
                'morphscore': {
                    'error': 'MorphScore library not available'
                }
            }
        
        logger.info("Computing MorphScore metrics...")
        
        # Initialize MorphScore evaluator
        morph_score = MorphScore(
            data_dir=self.data_dir,
            language_subset=self.target_languages,
            by_split=self.by_split,
            freq_scale=self.freq_scale,
            exclude_single_tok=self.exclude_single_tok
        )
        
        results = {
            'per_tokenizer': {},
            'metadata': {
                'by_split': self.by_split,
                'freq_scale': self.freq_scale,
                'exclude_single_tok': self.exclude_single_tok,
                'target_languages': self.target_languages
            }
        }
        
        # Evaluate each tokenizer
        for tok_name in self.tokenizer_names:
            logger.info(f"Evaluating MorphScore for tokenizer: {tok_name}")
            try:
                # Get tokenizer object
                tokenizer = self.input_provider.get_tokenizer(tok_name)
                
                # Evaluate with MorphScore
                morph_results = morph_score.eval(tokenizer)
                
                # Process results
                tokenizer_results = {
                    'per_language': {},
                    'summary': {
                        'languages_evaluated': 0,
                        'total_samples': 0,
                        'avg_morphscore_recall': 0.0,
                        'avg_morphscore_precision': 0.0,
                        'avg_micro_f1': 0.0,
                        'avg_macro_f1': 0.0
                    }
                }
                
                # Collect per-language results
                recall_values = []
                precision_values = []
                micro_f1_values = []
                macro_f1_values = []
                total_samples = 0
                
                for lang_code, lang_results in morph_results.items():
                    if lang_results and lang_code in self.target_languages:
                        if 'error' in lang_results:
                            logger.warning(f"Error evaluating MorphScore for {tok_name} on {lang_code}: {lang_results['error']}")
                            continue
                            
                        tokenizer_results['per_language'][lang_code] = lang_results

                        # Collect values for averaging
                        recall_values.append(lang_results['morphscore_recall'])
                        precision_values.append(lang_results['morphscore_precision'])
                        micro_f1_values.append(lang_results['micro_f1'])
                        macro_f1_values.append(lang_results['macro_f1'])
                        total_samples += lang_results['num_samples']
                
                # Compute summary statistics
                if recall_values:
                    n_languages = len(recall_values)
                    tokenizer_results['summary'] = {
                        'languages_evaluated': n_languages,
                        'total_samples': total_samples,
                        'avg_morphscore_recall': np.mean(recall_values),
                        'avg_morphscore_precision': np.mean(precision_values),
                        'avg_micro_f1': np.mean(micro_f1_values),
                        'avg_macro_f1': np.mean(macro_f1_values),
                        'avg_morphscore_recall_std': np.std(recall_values),
                        'avg_morphscore_precision_std': np.std(precision_values),
                        'avg_micro_f1_std': np.std(micro_f1_values),
                        'avg_macro_f1_std': np.std(macro_f1_values),
                        'avg_morphscore_recall_std_err': np.std(recall_values) / np.sqrt(n_languages),
                        'avg_morphscore_precision_std_err': np.std(precision_values) / np.sqrt(n_languages),
                        'avg_micro_f1_std_err': np.std(micro_f1_values) / np.sqrt(n_languages),
                        'avg_macro_f1_std_err': np.std(macro_f1_values) / np.sqrt(n_languages)
                    }
                
                results['per_tokenizer'][tok_name] = tokenizer_results
                
                logger.info(f"MorphScore evaluation completed for {tok_name}: "
                           f"{len(recall_values)} languages evaluated, "
                           f"{total_samples} total samples")
                
            except Exception as e:
                logger.error(f"Error evaluating MorphScore for {tok_name}: {e}")
                results['per_tokenizer'][tok_name] = {
                    'error': str(e),
                    'per_language': {},
                    'summary': {
                        'languages_evaluated': 0,
                        'total_samples': 0,
                        'avg_morphscore_recall': 0.0,
                        'avg_morphscore_precision': 0.0,
                        'avg_micro_f1': 0.0,
                        'avg_macro_f1': 0.0
                    }
                }
        
        return {'morphscore': results}
    
    def print_results(self, results: Dict[str, Any], per_lang: bool = False):
        """Print MorphScore metrics results."""
        if 'morphscore' not in results:
            return
        
        morphscore_data = results['morphscore']
        
        # Handle error case
        if 'error' in morphscore_data:
            print(f"\n🏛️ MORPHSCORE ANALYSIS")
            print("-" * 40)
            print(f"Error: {morphscore_data['error']}")
            return
        
        print("\n" + "="*60)
        print("MORPHSCORE RESULTS")
        print("="*60)
        
        # Print configuration info
        metadata = morphscore_data.get('metadata', {})
        print(f"\n⚙️ CONFIGURATION")
        print("-" * 40)
        print(f"By split: {metadata.get('by_split', 'N/A')}")
        print(f"Frequency scale: {metadata.get('freq_scale', 'N/A')}")
        print(f"Exclude single token: {metadata.get('exclude_single_tok', 'N/A')}")
        print(f"Target languages: {len(metadata.get('target_languages', []))}")
        
        # Print summary statistics
        print(f"\n🎯 SUMMARY STATISTICS")
        print("-" * 40)
        
        for tok_name in self.tokenizer_names:
            if tok_name in morphscore_data['per_tokenizer']:
                tok_data = morphscore_data['per_tokenizer'][tok_name]
                
                if 'error' in tok_data:
                    print(f"{tok_name:20}: Error - {tok_data['error']}")
                    continue
                
                summary = tok_data.get('summary', {})
                recall = summary.get('avg_morphscore_recall', 0.0)
                precision = summary.get('avg_morphscore_precision', 0.0)
                micro_f1 = summary.get('avg_micro_f1', 0.0)
                macro_f1 = summary.get('avg_macro_f1', 0.0)
                langs = summary.get('languages_evaluated', 0)
                samples = summary.get('total_samples', 0)
                
                print(f"{tok_name:20}:")
                print(f"  {'Recall':15}: {recall:.3f}")
                print(f"  {'Precision':15}: {precision:.3f}")
                print(f"  {'Micro F1':15}: {micro_f1:.3f}")
                print(f"  {'Macro F1':15}: {macro_f1:.3f}")
                print(f"  {'Languages':15}: {langs}")
                print(f"  {'Samples':15}: {samples:,}")
        
        # Print detailed per-language results
        if per_lang:
            print(f"\n📊 PER-LANGUAGE RESULTS")
            print("-" * 60)
            
            for tok_name in self.tokenizer_names:
                if tok_name not in morphscore_data['per_tokenizer']:
                    continue
                    
                tok_data = morphscore_data['per_tokenizer'][tok_name]
                
                if 'error' in tok_data or not tok_data.get('per_language'):
                    continue
                
                print(f"\n{tok_name}:")
                print("-" * 30)
                
                for lang_code, lang_results in tok_data['per_language'].items():
                    recall = lang_results.get('morphscore_recall', 0.0)
                    precision = lang_results.get('morphscore_precision', 0.0)
                    micro_f1 = lang_results.get('micro_f1', 0.0)
                    macro_f1 = lang_results.get('macro_f1', 0.0)
                    samples = lang_results.get('num_samples', 0)
                    
                    print(f"  {lang_code}:")
                    print(f"    Recall: {recall:.3f}, Precision: {precision:.3f}")
                    print(f"    Micro F1: {micro_f1:.3f}, Macro F1: {macro_f1:.3f}")
                    print(f"    Samples: {samples:,}")
            
        print("\n" + "="*60)