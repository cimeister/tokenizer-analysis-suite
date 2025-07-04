"""
python scripts/compare_tokenizers.py --tokenizer-config configs/tokenizer_config.json --language-config configs/language_config.json --morphological-config configs/morphological_config.json --samples-per-lang 3000 --output-dir analysis_results --verbose
"""
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from tokenizer_analysis import TokenizerAnalyzer
from tokenizer_analysis.utils import setup_environment

# Setup environment
setup_environment()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tokenizer_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config_from_file(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_sample_configs() -> Dict[str, Dict]:
    """Create sample tokenizer configurations for testing."""
    dirr = "data_large_128k_byte/"
    return {
        "standard_soft": {
            "class": "standard",
            "path": dirr + "tokenizers_correct_loss/standard_unigramlm_soft.json"
        },
        "langspec_soft": {
            "class": "langspec",
            "base_path": dirr + "tokenizers_correct_loss/standard_unigramlm_soft.json",
            "language_paths": {
                "en": dirr + "tokenizers_correct_loss/langspec_soft_en_em_probs.json",
                "es": dirr + "tokenizers_correct_loss/langspec_soft_es_em_probs.json",
                "de": dirr + "tokenizers_correct_loss/langspec_soft_de_em_probs.json",
                "ar": dirr + "tokenizers_correct_loss/langspec_soft_ar_em_probs.json",
                "ru": dirr + "tokenizers_correct_loss/langspec_soft_ru_em_probs.json"
            }
        },
        "tokmix": {
            "class": "standard",
            "path": dirr + "tokenizers_correct_loss/tokmix.json"
        },
        "multigram_soft": {
            "class": "standard",
            "path": dirr + "tokenizers_correct_loss/multigramlm_soft.json"
        }
    }


def create_sample_language_files() -> Dict[str, str]:
    """Create sample language file mappings."""
    return {
        "en": "parallel/en/eval.txt",
        "es": "parallel/es/eval.txt",
        "de": "parallel/de/eval.txt",
        "ar": "parallel/ar/eval.txt",
        "ru": "parallel/ru/eval.txt"
    }


def create_sample_morphological_config() -> Dict[str, str]:
    """Create sample morphological dataset configuration."""
    return {
        "morphynet": "morph_data/MORPHYNET/morphynet_multilingual.tsv"
    }


def slim_results_for_json(results: Dict) -> Dict:
    """Create a slimmed-down version of results for JSON export."""
    slimmed = {}
    
    # Keep only essential summary statistics, not raw data
    for metric_name, metric_data in results.items():
        if isinstance(metric_data, dict):
            slimmed_metric = {}
            
            # For per-tokenizer results, keep only summary stats
            if 'per_tokenizer' in metric_data:
                slimmed_metric['per_tokenizer'] = {}
                for tok_name, tok_data in metric_data['per_tokenizer'].items():
                    if isinstance(tok_data, dict):
                        # Keep essential stats but remove raw arrays
                        tok_summary = {}
                        for key, value in tok_data.items():
                            if key in ['global', 'global_ttr', 'global_utilization', 'global_avg']:
                                if isinstance(value, dict):
                                    # Keep only mean/std, not raw values
                                    filtered_value = {k: v for k, v in value.items() 
                                                    if k in ['mean', 'std', 'median', 'count'] and not k.endswith('_lengths')}
                                    tok_summary[key] = filtered_value
                                else:
                                    tok_summary[key] = value
                            elif key == 'per_language':
                                # Include per-language results for analysis
                                tok_summary[key] = value
                            elif key.startswith('renyi_') and isinstance(value, dict):
                                # Keep overall entropy values but not per-language details
                                tok_summary[key] = {'overall': value.get('overall')}
                            elif key in ['gini_coefficient', 'mean_cost', 'std_cost', 'min_cost', 'max_cost', 
                                        'cost_ratio', 'most_efficient_language', 'least_efficient_language', 
                                        'num_languages', 'language_costs', 'warning']:
                                # Keep all Gini-related metrics
                                tok_summary[key] = value
                            elif key in ['sorted_languages', 'sorted_costs', 'total_cost', 'n_languages', 
                                        'x_values', 'y_values', 'equality_line']:
                                # Keep Lorenz curve data but limit array sizes if needed
                                if isinstance(value, list) and len(value) > 100:
                                    # For very large arrays, keep only key points
                                    step = len(value) // 50  # Keep ~50 points
                                    tok_summary[key] = value[::step] if step > 1 else value[:50]
                                else:
                                    tok_summary[key] = value
                        slimmed_metric['per_tokenizer'][tok_name] = tok_summary
            
            # Keep pairwise comparisons (they're already summary data)
            if 'pairwise_comparisons' in metric_data:
                slimmed_metric['pairwise_comparisons'] = metric_data['pairwise_comparisons']
            
            # Keep vocabulary sizes
            if 'vocabulary_sizes' in metric_data:
                slimmed_metric['vocabulary_sizes'] = metric_data['vocabulary_sizes']
            
            # Keep summary stats for morphological analysis
            if metric_name == 'morphological_alignment' and 'summary' in metric_data:
                slimmed_metric['summary'] = metric_data['summary']
            
            # Keep metadata for Gini metrics
            if metric_name in ['tokenizer_fairness_gini', 'lorenz_curve_data'] and 'metadata' in metric_data:
                slimmed_metric['metadata'] = metric_data['metadata']
            
            # Keep global results
            if 'global' in metric_data and metric_name != 'morphological_alignment':
                slimmed_metric['global'] = metric_data['global']
            
            # Include per-language results at the top level
            if 'per_language' in metric_data:
                slimmed_metric['per_language'] = metric_data['per_language']
            
            slimmed[metric_name] = slimmed_metric
        else:
            slimmed[metric_name] = metric_data
    
    return slimmed


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Enhanced modular tokenizer analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-tokenizer analysis (supports any number of tokenizers)
  python scripts/compare_tokenizers.py --use-sample-data
  
  # Load from configuration files (supports 2+ tokenizers)
  python scriptscompare_tokenizers.py --tokenizer-config tokenizers.json --language-config languages.json
  
  # Pairwise comparison only (restricts to 2 specific tokenizers)
  python scriptscompare_tokenizers.py --pairwise tok1 tok2 --use-sample-data
  
  # Skip morphological analysis and plots for faster processing
  python scriptscompare_tokenizers.py --use-sample-data --no-morphological --no-plots
  
  # Save both summary and full detailed results
  python scriptscompare_tokenizers.py --use-sample-data --save-full-results
        """
    )
    
    # Configuration options
    parser.add_argument(
        "--tokenizer-config", 
        type=str,
        help="JSON file with tokenizer configurations"
    )
    parser.add_argument(
        "--language-config",
        type=str,
        help="JSON file mapping language codes to directories with JSON/Parquet files or direct text file paths"
    )
    parser.add_argument(
        "--morphological-config",
        type=str,
        help="JSON file with morphological dataset configurations"
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Use sample/demo data for testing"
    )
    
    # Analysis options
    parser.add_argument(
        "--pairwise",
        nargs=2,
        metavar=("TOK1", "TOK2"),
        help="Perform pairwise comparison between two specific tokenizers"
    )
    parser.add_argument(
        "--no-morphological",
        action="store_true",
        help="Skip morphological analysis"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--samples-per-lang",
        type=int,
        default=2000,
        help="Number of text samples per language (default: 2000)"
    )
    parser.add_argument(
        "--renyi-alphas",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 3.0],
        help="Alpha values for Rényi entropy analysis"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tokenizer_analysis_results",
        help="Directory for output plots and logs"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test functions"
    )
    parser.add_argument(
        "--save-full-results",
        action="store_true",
        help="Save full detailed results (large file) in addition to summary"
    )
    
    args = parser.parse_args()
    
    # Load configurations
    if args.use_sample_data:
        logger.info("Using sample data for demonstration")
        tokenizer_configs = create_sample_configs()
        language_files = create_sample_language_files()
        morphological_config = create_sample_morphological_config() if not args.no_morphological else None
    else:
        # Load from files
        if not args.tokenizer_config:
            raise ValueError("Must specify --tokenizer-config or use --use-sample-data")
        
        tokenizer_configs = load_config_from_file(args.tokenizer_config)
        
        if args.language_config:
            # Load language configuration (supports both directory and file paths)
            language_config_path = args.language_config
            language_files = None  # Will be handled by new loader
        else:
            language_files = create_sample_language_files()
            language_config_path = None
            logger.warning("No language config specified, using sample file paths")
        
        morphological_config = None
        if not args.no_morphological and args.morphological_config:
            morphological_config = load_config_from_file(args.morphological_config)
    
    # Validate tokenizer configs
    if len(tokenizer_configs) < 1:
        raise ValueError("At least one tokenizer must be configured")
    
    if args.pairwise and len(args.pairwise) == 2:
        tok1, tok2 = args.pairwise
        if tok1 not in tokenizer_configs or tok2 not in tokenizer_configs:
            raise ValueError(f"Pairwise tokenizers {tok1}, {tok2} must be in configuration")
        # Filter to only these two tokenizers
        tokenizer_configs = {tok1: tokenizer_configs[tok1], tok2: tokenizer_configs[tok2]}
    
    # Initialize analyzer
    logger.info("Initializing tokenizer analyzer...")
    analyzer = TokenizerAnalyzer(
        tokenizer_configs=tokenizer_configs,
        morphological_config=morphological_config,
        renyi_alphas=args.renyi_alphas,
        plot_save_dir=args.output_dir
    )
    if args.test:
        analyzer.basic_metrics.test_token_length_analysis_validity()
        analyzer.morphological_metrics.test_word_token_alignment_robustness()
        analyzer.morphological_metrics.test_morphological_alignment_logic()
        exit(0)
    
    # Load language texts
    logger.info("Loading language texts...")
    language_texts = analyzer.load_language_texts(
        language_files=language_files,
        language_config_path=language_config_path if 'language_config_path' in locals() else None,
        num_samples_per_lang=args.samples_per_lang
    )
    
    if not language_texts:
        raise ValueError("No valid language texts loaded")
    
    # Run analysis
    logger.info("Starting tokenizer analysis...")
    
    if args.pairwise:
        # Pairwise comparison
        results = analyzer.compare_tokenizers_pairwise(
            language_texts=language_texts,
            tokenizer1=args.pairwise[0],
            tokenizer2=args.pairwise[1],
            save_plots=not args.no_plots
        )
    else:
        # Full multi-tokenizer analysis
        results = analyzer.run_full_analysis(
            language_texts=language_texts,
            save_plots=not args.no_plots,
            include_morphological=not args.no_morphological,
            verbose=args.verbose
        )
    
    # Save results to JSON (slimmed version)
    results_file = Path(args.output_dir) / "analysis_results.json"
    logger.info(f"Saving slimmed results to {results_file}")
    
    # Create slimmed version and convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    slimmed_results = slim_results_for_json(results)
    results_json = convert_for_json(slimmed_results)
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Optionally save full results
    if args.save_full_results:
        full_results_file = Path(args.output_dir) / "analysis_results_full.json"
        logger.info(f"Saving full results to {full_results_file}")
        full_results_json = convert_for_json(results)
        with open(full_results_file, 'w') as f:
            json.dump(full_results_json, f, indent=2)
    
    logger.info("Analysis complete!")
    print(f"\nResults saved to: {args.output_dir}")
    if not args.no_plots:
        print(f"Plots saved to: {args.output_dir}")
    print(f"Summary results: {results_file}")
    if args.save_full_results:
        print(f"Full detailed results: {Path(args.output_dir) / 'analysis_results_full.json'}")

if __name__ == "__main__":
    main()
