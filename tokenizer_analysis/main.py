import logging
import random
from typing import Dict, List, Any, Optional
import numpy as np

from .metrics import BasicTokenizationMetrics, InformationTheoreticMetrics, MorphologicalMetrics, TokenizerGiniMetrics
from .loaders import MorphologicalDataLoader, load_multilingual_data
from .visualization import TokenizerVisualizer

logger = logging.getLogger(__name__)


class TokenizerAnalyzer:
    """Main class for comprehensive tokenizer analysis."""
    
    def __init__(self, tokenizer_configs: Dict[str, Dict], 
                 morphological_config: Optional[Dict[str, str]] = None,
                 renyi_alphas: Optional[List[float]] = None,
                 plot_save_dir: str = "tokenizer_analysis_plots"):
        """
        Initialize tokenizer analyzer.
        
        Args:
            tokenizer_configs: Dictionary mapping tokenizer names to their configs
            morphological_config: Configuration for morphological datasets
            renyi_alphas: List of alpha values for Rényi entropy computation
            plot_save_dir: Directory to save plots
        """
        from .utils import load_tokenizer_from_config
        
        self.tokenizer_configs = tokenizer_configs
        self.tokenizers = {}
        self.tokenizer_names = list(tokenizer_configs.keys())
        self.renyi_alphas = renyi_alphas or [0.5, 1.0, 2.0, 3.0]
        
        # Load all tokenizers
        for name, config in tokenizer_configs.items():
            logger.info(f"Loading tokenizer: {name}")
            self.tokenizers[name] = load_tokenizer_from_config(config)
        
        # Initialize metrics classes
        self.basic_metrics = BasicTokenizationMetrics(self.tokenizers, self.tokenizer_names)
        self.info_metrics = InformationTheoreticMetrics(
            self.tokenizers, self.tokenizer_names, self.renyi_alphas
        )
        self.morphological_metrics = MorphologicalMetrics(
            self.tokenizers, self.tokenizer_names, morphological_config
        )
        self.gini_metrics = TokenizerGiniMetrics(self.tokenizers, self.tokenizer_names)
        
        # Initialize visualizer
        self.visualizer = TokenizerVisualizer(self.tokenizer_names, plot_save_dir)
        
        logger.info(f"Initialized analyzer with {len(self.tokenizers)} tokenizers: {self.tokenizer_names}")
        for name in self.tokenizer_names:
            vocab_size = self.tokenizers[name].get_vocab_size()
            logger.info(f"  {name}: {vocab_size} tokens")
    
    def load_language_texts(self, language_files: Optional[Dict[str, str]] = None,
                           language_config_path: Optional[str] = None,
                           num_samples_per_lang: int = 1000,
                           random_seed: int = 42) -> Dict[str, List[str]]:
        """
        Load and sample texts from language-specific files or directories.
        
        Args:
            language_files: Dict mapping language codes to file paths (old format)
            language_config_path: Path to JSON config with language->directory mappings (new format)
            num_samples_per_lang: Number of samples per language
            random_seed: Random seed for reproducible sampling
            
        Returns:
            Dict mapping language codes to lists of texts
        """
        if language_config_path:
            logger.info(f"Loading multilingual data from config: {language_config_path}")
            return load_multilingual_data(language_config_path, num_samples_per_lang)
        
        elif language_files:
            language_texts = {}
            random.seed(random_seed)
            
            for lang, file_path in language_files.items():
                try:
                    logger.info(f"Loading texts for {lang} from {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        all_lines = [line.strip() for line in f if line.strip()]
                    
                    # Sample lines for this language
                    sampled_lines = random.sample(all_lines, min(num_samples_per_lang, len(all_lines)))
                    language_texts[lang] = sampled_lines
                    logger.info(f"Loaded {len(sampled_lines)} texts for {lang}")
                    
                except FileNotFoundError:
                    logger.error(f"Language file not found: {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {lang} from {file_path}: {e}")
            
            return language_texts
        
        else:
            raise ValueError("Must provide either language_files or language_config_path")
    
    def run_full_analysis(self, language_texts: Dict[str, List[str]], 
                         save_plots: bool = True,
                         include_morphological: bool = True,
                         verbose: bool = True,
                         pairwise: bool = False) -> Dict[str, Any]:
        """
        Run comprehensive tokenizer analysis.
        
        Args:
            language_texts: Dict mapping language codes to lists of texts
            save_plots: Whether to generate and save plots
            include_morphological: Whether to include morphological analysis
            verbose: Whether to print detailed results
            
        Returns:
            Dict containing all analysis results
        """
        logger.info("Starting comprehensive tokenizer analysis...")
        
        # Filter out empty language data
        language_texts = {lang: texts for lang, texts in language_texts.items() 
                         if texts and any(text.strip() for text in texts)}
        
        if not language_texts:
            raise ValueError("No valid language texts provided")
        
        logger.info(f"Analyzing {len(language_texts)} languages: {list(language_texts.keys())}")
        
        # Encode all texts once for efficiency
        logger.info("Encoding all texts for all tokenizers...")
        all_encodings = self.basic_metrics.encode_texts_batch(language_texts)
        
        results = {}
        
        # Run basic tokenization metrics (using bytes as default unit)
        logger.info("Computing basic tokenization metrics (using bytes as base unit)...")
        basic_results = self.basic_metrics.compute(language_texts, all_encodings, use_bytes=True)
        results.update(basic_results)
        
        if verbose:
            self._print_basic_results(basic_results)
        
        # Run information-theoretic metrics
        logger.info("Computing information-theoretic metrics...")
        info_results = self.info_metrics.compute(language_texts, all_encodings)
        results.update(info_results)
        
        if verbose:
            self._print_information_theoretic_results(info_results)
        
        # Run morphological analysis if requested
        if include_morphological:
            logger.info("Computing morphological alignment metrics...")
            morph_results = self.morphological_metrics.compute(language_texts, all_encodings)
            results.update(morph_results)
            
            if verbose:
                self._print_morphological_results(morph_results)
        
        # Run Gini fairness analysis
        logger.info("Computing Tokenizer Fairness Gini metrics...")
        gini_results = self.gini_metrics.compute(language_texts, all_encodings)
        results.update(gini_results)
        
        if verbose:
            self._print_gini_results(gini_results)
        
        # Generate plots
        if save_plots:
            logger.info("Generating visualization plots...")
            self.visualizer.generate_all_plots(results, print_pairwise=pairwise)
        
        # Print summary
        if verbose:
            self._print_comprehensive_summary(results, language_texts)
        
        logger.info("Analysis complete!")
        return results
    
    def _print_basic_results(self, results: Dict[str, Any]) -> None:
        """Print basic tokenization results."""
        print(f"\n{'='*60}")
        print("BASIC TOKENIZATION METRICS")
        print(f"{'='*60}")
        
        # Vocabulary sizes
        if 'vocabulary_overlap' in results:
            print("\nVocabulary Sizes:")
            for name, size in results['vocabulary_overlap']['vocabulary_sizes'].items():
                print(f"  {name}: {size:,} tokens")
        
        # Token lengths
        if 'token_length' in results:
            # Check what unit is being used as primary
            unit = 'characters'  # Default
            if 'metadata' in results['token_length']:
                unit = results['token_length']['metadata'].get('primary_unit', 'characters')
            
            print(f"\nAverage Token Length ({unit}):")
            for name, stats in results['token_length']['per_tokenizer'].items():
                # Use primary_length if available (new structure)
                if 'primary_length' in stats:
                    primary_mean = stats['primary_length']['mean']
                    primary_std = stats['primary_length']['std']
                    print(f"  {name}: {primary_mean:.2f} ± {primary_std:.2f} {unit}")
                    
                    # Show secondary unit in parentheses
                    if 'secondary_length' in stats:
                        sec_mean = stats['secondary_length']['mean']
                        sec_std = stats['secondary_length']['std']
                        sec_unit = 'characters' if unit == 'bytes' else 'bytes'
                        print(f"    ({sec_mean:.2f} ± {sec_std:.2f} {sec_unit})")
                # Fallback to old structure
                elif 'character_length' in stats:
                    char_mean = stats['character_length']['mean']
                    char_std = stats['character_length']['std']
                    print(f"  {name}: {char_mean:.2f} ± {char_std:.2f} chars")
                    if 'byte_length' in stats:
                        byte_mean = stats['byte_length']['mean']
                        byte_std = stats['byte_length']['std']
                        print(f"    ({byte_mean:.2f} ± {byte_std:.2f} bytes)")
                else:
                    # Very old structure fallback
                    print(f"  {name}: {stats.get('mean', 0):.2f} ± {stats.get('std', 0):.2f}")
        
        # Compression ratios
        if 'compression_ratio' in results:
            # Check what unit was used
            unit = 'characters'  # Default
            if 'metadata' in results['compression_ratio']:
                unit = results['compression_ratio']['metadata'].get('unit', 'characters')
            
            print(f"\nGlobal Compression Ratios ({unit}/token):")
            for name, stats in results['compression_ratio']['per_tokenizer'].items():
                print(f"  {name}: {stats['global']:.4f}")
        
        # Fertility metrics (both types)
        if 'whitespace_fertility' in results:
            print("\nWhitespace-Delimited Fertility (tokens/word):")
            for name, stats in results['whitespace_fertility']['per_tokenizer'].items():
                print(f"  {name}: {stats['global']['mean']:.4f} ± {stats['global']['std']:.4f}")
        
        if 'character_fertility' in results:
            print("\nCharacter-Based Fertility (tokens/character):")
            for name, stats in results['character_fertility']['per_tokenizer'].items():
                print(f"  {name}: {stats['global']['mean']:.4f} ± {stats['global']['std']:.4f}")
    
    def _print_information_theoretic_results(self, results: Dict[str, Any]) -> None:
        """Print information-theoretic results."""
        print(f"\n{'='*60}")
        print("INFORMATION-THEORETIC METRICS")
        print(f"{'='*60}")
        
        # Type-Token Ratio
        if 'type_token_ratio' in results:
            print("\nGlobal Type-Token Ratio:")
            for name, stats in results['type_token_ratio']['per_tokenizer'].items():
                print(f"  {name}: {stats['global_ttr']:.6f}")
        
        # Shannon Entropy
        if 'renyi_efficiency' in results:
            print("\nShannon Entropy (α=1.0):")
            for name, stats in results['renyi_efficiency']['per_tokenizer'].items():
                if 'renyi_1.0' in stats:
                    print(f"  {name}: {stats['renyi_1.0']['overall']:.3f} bits")
        
        # Vocabulary Utilization
        if 'vocabulary_utilization' in results:
            print("\nVocabulary Utilization:")
            for name, stats in results['vocabulary_utilization']['per_tokenizer'].items():
                print(f"  {name}: {stats['global_utilization']:.3%}")
        
        # Average tokens per line
        if 'avg_tokens_per_line' in results:
            print("\nAverage Tokens per Line:")
            for name, stats in results['avg_tokens_per_line']['per_tokenizer'].items():
                print(f"  {name}: {stats['global_avg']:.2f}")

        if 'unigram_distribution_metrics' in results:
            print("\nAverage Unigram Entropy:")
            
            for name, stats in results['unigram_distribution_metrics']['per_tokenizer'].items():
                print(f"  {name}: {stats['global_unigram_entropy']:.2f}")

        # Average tokens per line
        if 'unigram_distribution_metrics' in results:
            print("\nAverage Token Rank:")
            for name, stats in results['unigram_distribution_metrics']['per_tokenizer'].items():
                print(f"  {name}: {stats['global_avg_token_rank']:.2f}")
                
    
    def _print_morphological_results(self, results: Dict[str, Any]) -> None:
        """Print morphological analysis results."""
        if 'morphological_alignment' not in results:
            return
        
        morph_results = results['morphological_alignment']
        if 'message' in morph_results:
            print(f"\n{'='*60}")
            print("MORPHOLOGICAL ANALYSIS")
            print(f"{'='*60}")
            print(morph_results['message'])
            return
        
        print(f"\n{'='*60}")
        print("MORPHOLOGICAL ALIGNMENT METRICS")
        print(f"{'='*60}")
        
        if 'summary' in morph_results:
            for name, summary in morph_results['summary'].items():
                print(f"\n{name}:")
                print(f"  Boundary F1: {summary.get('avg_boundary_f1', 0):.3f}")
                print(f"  Morpheme Preservation: {summary.get('avg_morpheme_preservation', 0):.3f}")
                print(f"  Languages Analyzed: {summary.get('languages_analyzed', 0)}")
                print(f"  Words Analyzed: {summary.get('total_words_analyzed', 0)}")
    
    def _print_gini_results(self, results: Dict[str, Any]) -> None:
        """Print Tokenizer Fairness Gini results."""
        if 'tokenizer_fairness_gini' not in results:
            return
        
        gini_results = results['tokenizer_fairness_gini']
        
        print(f"\n{'='*60}")
        print("TOKENIZER FAIRNESS GINI ANALYSIS")
        print(f"{'='*60}")
        
        if 'metadata' in gini_results:
            print(f"Description: {gini_results['metadata']['description']}")
            print(f"Formula: {gini_results['metadata']['formula']}")
            print(f"Interpretation: {gini_results['metadata']['interpretation']}")
        
        print("\nTokenizer Fairness Gini (TFG) Coefficients:")
        
        for name, stats in gini_results['per_tokenizer'].items():
            if 'warning' in stats:
                print(f"  {name}: {stats['warning']}")
                continue
            
            tfg = stats['gini_coefficient']
            mean_cost = stats['mean_cost']
            cost_ratio = stats['cost_ratio']
            n_langs = stats['num_languages']
            
            print(f"  {name}:")
            print(f"    TFG: {tfg:.4f}")
            print(f"    Mean cost: {mean_cost:.4f} tokens/byte")
            print(f"    Cost ratio (max/min): {cost_ratio:.2f}")
            print(f"    Languages analyzed: {n_langs}")
            
            # Show most and least efficient languages
            most_eff = stats['most_efficient_language']
            least_eff = stats['least_efficient_language']
            print(f"    Most efficient: {most_eff[0]} ({most_eff[1]:.4f})")
            print(f"    Least efficient: {least_eff[0]} ({least_eff[1]:.4f})")
            print()
    
    def _print_comprehensive_summary(self, results: Dict[str, Any], 
                                   language_texts: Dict[str, List[str]]) -> None:
        """Print comprehensive analysis summary."""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        print(f"Tokenizers Analyzed: {len(self.tokenizer_names)}")
        print(f"Languages Analyzed: {len(language_texts)}")
        
        total_texts = sum(len(texts) for texts in language_texts.values())
        print(f"Total Texts Analyzed: {total_texts:,}")
        
        # Highlight key findings
        print(f"\n{'='*60}")
        print("KEY FINDINGS")
        print(f"{'='*60}")
        
        # Most efficient tokenizer (lowest average tokens per text)
        if 'avg_tokens_per_line' in results:
            efficiency_scores = {name: stats['global_avg'] 
                               for name, stats in results['avg_tokens_per_line']['per_tokenizer'].items()}
            if efficiency_scores:
                most_efficient = min(efficiency_scores.items(), key=lambda x: x[1])
                print(f"Most Efficient (fewest tokens/text): {most_efficient[0]} ({most_efficient[1]:.1f} tokens)")
        
        # Highest vocabulary utilization
        if 'vocabulary_utilization' in results:
            util_scores = {name: stats['global_utilization'] 
                          for name, stats in results['vocabulary_utilization']['per_tokenizer'].items()}
            if util_scores:
                highest_util = max(util_scores.items(), key=lambda x: x[1])
                print(f"Highest Vocabulary Utilization: {highest_util[0]} ({highest_util[1]:.1%})")
        
        # Best morphological alignment
        if ('morphological_alignment' in results and 
            'summary' in results['morphological_alignment']):
            f1_scores = {name: summary.get('avg_boundary_f1', 0) 
                        for name, summary in results['morphological_alignment']['summary'].items()}
            if f1_scores and any(f1_scores.values()):
                best_morph = max(f1_scores.items(), key=lambda x: x[1])
                print(f"Best Morphological Alignment: {best_morph[0]} (F1: {best_morph[1]:.3f})")
        
        # Gini fairness highlights
        if 'tokenizer_fairness_gini' in results:
            gini_scores = {name: stats['gini_coefficient'] 
                          for name, stats in results['tokenizer_fairness_gini']['per_tokenizer'].items()
                          if 'warning' not in stats}
            if gini_scores:
                most_fair = min(gini_scores.items(), key=lambda x: x[1])
                print(f"Most Fair Tokenizer (lowest TFG): {most_fair[0]} (TFG: {most_fair[1]:.4f})")
        
        # Pairwise comparison highlights
        compression_ratios = {}
        if 'compression_ratio' in results:
            compression_ratios = {name: stats['global'] 
                                for name, stats in results['compression_ratio']['per_tokenizer'].items()}
        
        if len(compression_ratios) >= 2:
            sorted_ratios = sorted(compression_ratios.items(), key=lambda x: x[1])
            print(f"Lowest Compression Ratio: {sorted_ratios[0][0]} ({sorted_ratios[0][1]:.4f})")
            print(f"Highest Compression Ratio: {sorted_ratios[-1][0]} ({sorted_ratios[-1][1]:.4f})")
    
    def compare_tokenizers_pairwise(self, language_texts: Dict[str, List[str]], 
                                   tokenizer1: str, tokenizer2: str,
                                   save_plots: bool = True) -> Dict[str, Any]:
        """
        Perform detailed pairwise comparison between two specific tokenizers.
        
        Args:
            language_texts: Dict mapping language codes to lists of texts
            tokenizer1: Name of first tokenizer
            tokenizer2: Name of second tokenizer
            save_plots: Whether to save comparison plots
            
        Returns:
            Dict containing pairwise comparison results
        """
        if tokenizer1 not in self.tokenizers or tokenizer2 not in self.tokenizers:
            raise ValueError(f"Tokenizers {tokenizer1} and {tokenizer2} must be loaded")
        
        logger.info(f"Performing pairwise comparison: {tokenizer1} vs {tokenizer2}")
        
        # Create temporary analyzer with just these two tokenizers
        temp_configs = {
            tokenizer1: self.tokenizer_configs[tokenizer1],
            tokenizer2: self.tokenizer_configs[tokenizer2]
        }
        
        temp_analyzer = TokenizerAnalyzer(
            temp_configs, 
            plot_save_dir=f"pairwise_{tokenizer1}_vs_{tokenizer2}"
        )
        
        # Run full analysis
        results = temp_analyzer.run_full_analysis(
            language_texts, 
            save_plots=save_plots, 
            verbose=True,
            pairwise=True
        )
        
        return results