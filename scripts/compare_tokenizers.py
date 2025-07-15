"""
Unified tokenizer analysis script supporting both raw tokenizers and pre-tokenized data.

Raw tokenizer examples:
python scripts/compare_tokenizers.py --use-sample-data
python scripts/compare_tokenizers.py --tokenizer-config configs/tokenizer_config.json --language-config configs/language_config.json --morphological-config configs/morphological_config.json --normalization-config configs/normalization_config_bytes.json --samples-per-lang 3000 --output-dir analysis_results --verbose --run-grouped-analysis

Pre-tokenized data examples:
python scripts/compare_tokenizers.py --tokenized-data-file tokenized_data.json --language-config configs/language_config.json
python scripts/compare_tokenizers.py --tokenized-data-file tokenized_data.pkl --tokenized-data-config tokenized_config.json --language-config configs/language_config.json --run-grouped-analysis
"""
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from tokenizer_analysis.main_unified import UnifiedTokenizerAnalyzer, create_analyzer_from_raw_inputs, create_analyzer_from_tokenized_data
from tokenizer_analysis.utils import setup_environment, load_tokenizer_from_config
from tokenizer_analysis.config.language_metadata import LanguageMetadata
from tokenizer_analysis.loaders.multilingual_data import load_multilingual_data
from tokenizer_analysis.core.input_utils import InputLoader
from tokenizer_analysis.constants import (
    TextProcessing,
    DataProcessing
)

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


def create_sample_language_metadata() -> str:
    """Create sample LanguageMetadata configuration and return path to temp file."""
    import tempfile
    
    sample_metadata = {
        "languages": {
            "eng_Latn": {
                "name": "English",
                "iso_code": "en", 
                "data_path": "parallel/en/eval.txt"
            },
            "spa_Latn": {
                "name": "Spanish",
                "iso_code": "es",
                "data_path": "parallel/es/eval.txt"
            },
            "deu_Latn": {
                "name": "German", 
                "iso_code": "de",
                "data_path": "parallel/de/eval.txt"
            },
            "arb_Arab": {
                "name": "Arabic",
                "iso_code": "ar",
                "data_path": "parallel/ar/eval.txt"
            },
            "rus_Cyrl": {
                "name": "Russian",
                "iso_code": "ru",
                "data_path": "parallel/ru/eval.txt"
            }
        },
        "analysis_groups": {
            "script_families": {
                "Latin": ["eng_Latn", "spa_Latn", "deu_Latn"],
                "Arabic": ["arb_Arab"],
                "Cyrillic": ["rus_Cyrl"]
            },
            "resource_levels": {
                "high": ["eng_Latn", "spa_Latn", "deu_Latn", "arb_Arab", "rus_Cyrl"],
                "medium": [],
                "low": []
            }
        }
    }
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_metadata, f, indent=2)
        return f.name


def create_sample_morphological_config() -> Dict[str, str]:
    """Create sample morphological dataset configuration."""
    return {
        "morphynet": "morph_data/MORPHYNET/morphynet_multilingual.tsv"
    }


def load_tokenized_data(tokenized_data_file: str, tokenized_data_config: Optional[str] = None) -> Dict:
    """
    Load pre-tokenized data from file.
    
    Args:
        tokenized_data_file: Path to the tokenized data file (JSON or pickle)
        tokenized_data_config: Optional configuration file for tokenized data
        
    Returns:
        Dictionary containing tokenized data
    """
    logger.info(f"Loading pre-tokenized data from {tokenized_data_file}")
    
    # Load the tokenized data
    tokenized_data = InputLoader.load_from_file(tokenized_data_file)
    
    # Load config if provided
    if tokenized_data_config:
        config = load_config_from_file(tokenized_data_config)
        # Merge or apply config as needed
        logger.info(f"Applied tokenized data configuration from {tokenized_data_config}")
    
    return tokenized_data


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
                                if isinstance(value, list) and len(value) > TextProcessing.LARGE_ARRAY_THRESHOLD:
                                    # For very large arrays, keep only key points
                                    step = len(value) // TextProcessing.ARRAY_SAMPLING_POINTS  # Keep ~50 points
                                    tok_summary[key] = value[::step] if step > 1 else value[:TextProcessing.ARRAY_SAMPLING_POINTS]
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
  # Multi-tokenizer analysis with raw tokenizers (supports any number of tokenizers)
  python scripts/compare_tokenizers.py --use-sample-data
  
  # Load from configuration files (supports 2+ tokenizers)
  python scripts/compare_tokenizers.py --tokenizer-config tokenizers.json --language-config languages.json
  
  # Use pre-tokenized data from file
  python scripts/compare_tokenizers.py --tokenized-data-file tokenized_data.json --language-config languages.json
  
  # Use pre-tokenized data with configuration
  python scripts/compare_tokenizers.py --tokenized-data-file tokenized_data.pkl --tokenized-data-config tokenized_config.json --language-config languages.json
  
  # Filter by script family and run grouped analysis (includes grouped plots)
  python scripts/compare_tokenizers.py --use-sample-data --filter-script-family Latin --run-grouped-analysis
  
  # Filter by resource level  
  python scripts/compare_tokenizers.py --use-sample-data --filter-resource-level high
  
  # Run comprehensive grouped analysis across all script families and resource levels
  python scripts/compare_tokenizers.py --use-sample-data --run-grouped-analysis
  
  # Pairwise comparison only (restricts to 2 specific tokenizers)
  python scripts/compare_tokenizers.py --pairwise tok1 tok2 --use-sample-data
  
  # Skip morphological analysis and plots for faster processing
  python scripts/compare_tokenizers.py --use-sample-data --no-morphological --no-plots
  
  # Save both summary and full detailed results
  python scripts/compare_tokenizers.py --use-sample-data --save-full-results
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
        help="JSON file with LanguageMetadata configuration (languages + analysis groups)"
    )
    parser.add_argument(
        "--morphological-config",
        type=str,
        help="JSON file with morphological dataset configurations"
    )
    parser.add_argument(
        "--normalization-config",
        type=str,
        help="JSON file with normalization configuration (method, pretokenization, etc.)"
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Use sample/demo data for testing"
    )
    
    # NEW: Pre-tokenized data support
    parser.add_argument(
        "--tokenized-data-config",
        type=str,
        help="JSON file with pre-tokenized data configuration"
    )
    parser.add_argument(
        "--tokenized-data-file",
        type=str,
        help="Path to pre-tokenized data file (JSON or pickle)"
    )
    
    # Analysis options
    parser.add_argument(
        "--pairwise",
        nargs=2,
        metavar=("TOK1", "TOK2"),
        help="Perform pairwise comparison between two specific tokenizers"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--samples-per-lang",
        type=int,
        default=DataProcessing.DEFAULT_MAX_SAMPLES,
        help=f"Number of text samples per language (default: {DataProcessing.DEFAULT_MAX_SAMPLES})"
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
    
    # NEW: LanguageMetadata filtering options
    parser.add_argument(
        "--filter-script-family",
        type=str,
        help="Filter languages by script family (e.g., 'Latin', 'Arabic', 'CJK')"
    )
    
    parser.add_argument(
        "--filter-resource-level", 
        type=str,
        help="Filter languages by resource level (e.g., 'high', 'medium', 'low')"
    )
    
    # NEW: Grouped analysis option
    parser.add_argument(
        "--run-grouped-analysis",
        action="store_true",
        help="Run analysis grouped by script families and resource levels"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine input mode based on provided arguments
    use_tokenized_data = args.tokenized_data_file is not None
    
    # Load configurations
    if args.use_sample_data and not use_tokenized_data:
        logger.info("Using sample data for demonstration")
        tokenizer_configs = create_sample_configs()
        language_config_path = create_sample_language_metadata()
        morphological_config = create_sample_morphological_config()
        normalization_config = None  # Use default for sample data
    elif use_tokenized_data:
        # Pre-tokenized data mode
        if not args.tokenized_data_file:
            raise ValueError("Must specify --tokenized-data-file for pre-tokenized mode")
        
        tokenizer_configs = None  # Will be inferred from tokenized data
        tokenized_data = load_tokenized_data(args.tokenized_data_file, args.tokenized_data_config)
        
        # Still need language config for metadata
        if args.language_config:
            language_config_path = args.language_config
        else:
            language_config_path = create_sample_language_metadata()
            logger.warning("No language config specified, using sample metadata")
        
        morphological_config = None
        if args.morphological_config:
            morphological_config = load_config_from_file(args.morphological_config)
        
        # Load normalization configuration
        normalization_config = None
        if args.normalization_config:
            from tokenizer_analysis.config import NormalizationConfig
            norm_config_dict = load_config_from_file(args.normalization_config)
            normalization_config = NormalizationConfig.from_dict(norm_config_dict)
    else:
        # Raw tokenizer mode
        if not args.tokenizer_config:
            raise ValueError("Must specify --tokenizer-config or use --use-sample-data")
        
        tokenizer_configs = load_config_from_file(args.tokenizer_config)
        
        if args.language_config:
            # Load language configuration (supports both directory and file paths)
            language_config_path = args.language_config
        else:
            language_config_path = create_sample_language_metadata()
            logger.warning("No language config specified, using sample metadata")
        
        morphological_config = None
        if args.morphological_config:
            morphological_config = load_config_from_file(args.morphological_config)
        
        # Load normalization configuration
        normalization_config = None
        if args.normalization_config:
            from tokenizer_analysis.config import NormalizationConfig
            norm_config_dict = load_config_from_file(args.normalization_config)
            normalization_config = NormalizationConfig.from_dict(norm_config_dict)
    
    # Load language metadata
    logger.info("Loading language metadata...")
    language_metadata = LanguageMetadata(language_config_path)
    
    # Initialize analyzer based on input mode
    logger.info("Initializing unified tokenizer analyzer...")
    
    if use_tokenized_data:
        # Pre-tokenized data mode
        analyzer = create_analyzer_from_tokenized_data(
            tokenized_data=tokenized_data,
            normalization_config=normalization_config,
            language_metadata=language_metadata,
            plot_save_dir=args.output_dir,
            morphological_config=morphological_config
        )
    else:
        # Raw tokenizer mode
        # Validate tokenizer configs
        from tokenizer_analysis.constants import MIN_TOKENIZERS_FOR_COMPARISON
        if not tokenizer_configs or len(tokenizer_configs) < MIN_TOKENIZERS_FOR_COMPARISON:
            raise ValueError("At least one tokenizer must be configured")
        
        if args.pairwise and len(args.pairwise) == 2:
            tok1, tok2 = args.pairwise
            if tok1 not in tokenizer_configs or tok2 not in tokenizer_configs:
                raise ValueError(f"Pairwise tokenizers {tok1}, {tok2} must be in configuration")
            # Filter to only these two tokenizers
            tokenizer_configs = {tok1: tokenizer_configs[tok1], tok2: tokenizer_configs[tok2]}
        
        # Load language texts
        logger.info("Loading language texts...")
        filter_by_group = None
        if args.filter_script_family:
            filter_by_group = ('script_families', args.filter_script_family)
        elif args.filter_resource_level:
            filter_by_group = ('resource_levels', args.filter_resource_level)
        
        language_texts = load_multilingual_data(
            language_metadata=language_metadata,
            max_texts_per_language=args.samples_per_lang,
            filter_by_group=filter_by_group
        )
        
        if not language_texts:
            raise ValueError("No valid language texts loaded")
        
        # Initialize unified analyzer using convenience function
        analyzer = create_analyzer_from_raw_inputs(
            tokenizer_configs=tokenizer_configs,
            language_texts=language_texts,
            normalization_config=normalization_config,
            language_metadata=language_metadata,
            plot_save_dir=args.output_dir,
            morphological_config=morphological_config
        )
    if args.test:
        # TODO: Update test methods for unified system
        logger.warning("Test methods not yet updated for unified system")
        exit(0)
    
    # Run analysis
    logger.info("Starting tokenizer analysis...")
    
    if args.pairwise:
        # Pairwise comparison - for now, just run regular analysis and filter results
        logger.info(f"Running pairwise comparison: {args.pairwise[0]} vs {args.pairwise[1]}")
        results = analyzer.run_analysis(
            save_plots=not args.no_plots,
            include_morphological=morphological_config is not None,
            verbose=args.verbose
        )
    else:
        # Full multi-tokenizer analysis
        results = analyzer.run_analysis(
            save_plots=not args.no_plots,
            include_morphological=morphological_config is not None,
            verbose=args.verbose
        )
        
        # NEW: Run grouped analysis if requested
        if args.run_grouped_analysis and analyzer.language_metadata:
            logger.info("Running grouped analysis by script families and resource levels...")
            
            # Use the unified analyzer's built-in grouped analysis
            grouped_results = analyzer.run_grouped_analysis(
                group_by=['script_families', 'resource_levels'],
                save_plots=not args.no_plots
            )
            
            # Add grouped results to main results
            results['grouped_analysis'] = grouped_results
    
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
