# TokEval

A framework for evaluating tokenizers across languages with comprehensive metrics and multilingual fairness analysis.

## Features

- **Multi-tokenizer comparison**: Compare any number of tokenizers simultaneously
- **Comprehensive metrics**: Basic tokenization, information-theoretic, morphological, and fairness metrics
- **Multilingual fairness**: Gini coefficient analysis for cross-language compression equity
- **MorphScore integration**: Carefully crafted morphological analysis ([Arnett et. al. 2025](https://arxiv.org/abs/2507.06378))
- **Flexible data inputs**: Raw texts, pre-tokenized data, JSON, and parquet files
- **Academic outputs**: LaTeX table generation and publication-ready plots
- **Performance optimized**: Efficient analysis with pre-tokenized data caching

## Setup

### Requirements
- Python 3.8+
- Git (for submodules)

### Installation
```bash
git clone https://github.com/cimeister/tokenizer-analysis.git
cd tokenizer-analysis
pip install -e .

# Install morphological analysis module
git submodule update --init --recursive
cd morphscore && pip install -e . && cd ..
```

## Usage Examples

### Basic Analysis
```bash
# Quick start with sample data
python scripts/run_tokenizer_analysis.py --use-sample-data

# Custom tokenizers and languages (see example configs below)
python scripts/run_tokenizer_analysis.py \
    --tokenizer-config configs/tokenizer_config.json \
    --language-config configs/language_config.json \
    --output-dir results/
```

### Advanced Analysis Options
```bash
# Grouped analysis by script families and resource levels
python scripts/run_tokenizer_analysis.py --use-sample-data --run-grouped-analysis

# Filter by script family (generates separate analysis for Latin scripts only)
python scripts/run_tokenizer_analysis.py --use-sample-data --filter-script-family Latin

# Include morphological analysis with MorphScore
python scripts/run_tokenizer_analysis.py --use-sample-data --morphscore

# Generate per-language plots (grouped bar charts with languages on x-axis)
python scripts/run_tokenizer_analysis.py --use-sample-data --per-language-plots

# Generate faceted plots (subplots with tied y-axes)
python scripts/run_tokenizer_analysis.py --use-sample-data --faceted-plots
```

### Output Options
```bash
# Generate LaTeX tables for academic papers
python scripts/run_tokenizer_analysis.py --use-sample-data --generate-latex-tables

# Save both summary and detailed results
python scripts/run_tokenizer_analysis.py --use-sample-data --save-full-results

# Skip plot generation (analysis only)
python scripts/run_tokenizer_analysis.py --use-sample-data --no-plots
```

### Pre-tokenized Data
```bash
# Save tokenized data for reuse (automatically creates config file)
python scripts/run_tokenizer_analysis.py --use-sample-data \
    --save-tokenized-data --tokenized-data-output-path my_data.pkl

# Use pre-tokenized data (config file auto-generated as my_data_config.json)
python scripts/run_tokenizer_analysis.py \
    --tokenized-data-file my_data.pkl \
    --tokenized-data-config my_data_config.json \
    --language-config configs/language_config.json
```

### Output Structure
```
results/
├── fertility.png              # Tokens per word/character
├── compression_rate.png       # Text compression efficiency
├── vocabulary_utilization.png # Vocabulary usage
├── <metric name>.png          # Other supported metrics
├── grouped_plots/                  # Cross-tokenizer comparisons
│   ├── script_family_comparison.png
│   └── resource_level_analysis.png
├── per-language/                   # Language-specific analysis
│   ├── fertility_by_language.png
│   └── compression_by_language.png
├── latex_tables/                   # Academic publication tables
│   ├── basic_metrics.tex
│   └── comprehensive_analysis.tex
├── analysis_results_summary.json   # Key metrics summary
└── analysis_results_full.json     # Detailed results (with --save-full-results)
```

### Tokenizer Configuration
```json
{
  "tokenizer1": {
    "class": "custom_bpe",
    "path": "/path/to/tokenizer"
  },
  "tokenizer2": {
    "class": "huggingface",
    "path": "bert-base-uncased"
  }
}
```

### Language Configuration
```json
{
  "languages": {
    "en": "/path/to/english/data",
    "fr": "/path/to/french/file.txt",
    "de": "/path/to/german/corpus.json"
  }
}
```

### Text Measurement Configuration

The framework supports different text "length" measurements so that metrics can be normalized using different units:

```bash
# Use byte-level measurement (default)
python scripts/run_tokenizer_analysis.py --use-sample-data \
    --measurement-config configs/text_measurement_config_bytes.json

# Use line-based measurement for parallel corpora
python scripts/run_tokenizer_analysis.py --use-sample-data \
    --measurement-config configs/text_measurement_config_lines.json

# Use word-based measurement with HuggingFace whitespace tokenization
python scripts/run_tokenizer_analysis.py --use-sample-data \
    --measurement-config configs/text_measurement_config_words_hf.json
```

#### Available Measurement Methods

**Byte Counting:**
```json
{
  "method": "bytes",
  "byte_counting_method": "utf8"
}
```
- `"utf8"`: Standard UTF-8 encoding (default for compression)
- `"hf_tokenizer"`: Uses HuggingFace tokenizer's pre-tokenizer for byte counting

**Character Counting:**
```json
{
  "method": "characters"
}
```
- Counts Unicode characters (useful for non-Latin scripts)

**Line Counting:**
```json
{
  "method": "lines",
  "line_counting_method": "python_split",
  "include_empty_lines": false
}
```
- `"python_split"`: Uses Python's `str.splitlines()` (default for Gini)
- `"regex"`: Custom regex-based line splitting (requires `custom_regex`)
- `include_empty_lines`: Whether to count empty lines

**Word Counting:**
```json
{
  "method": "words", (default for fertility)
  "word_counting_method": "whitespace",
  "include_empty_words": false
}
```
- `"whitespace"`: Simple whitespace splitting
- `"hf_whitespace"`: HuggingFace whitespace pre-tokenizer
- `"regex"`: Custom regex-based word splitting (requires `custom_regex`)

**Custom Regex:**
```json
{
  "method": "words",
  "word_counting_method": "regex", 
  "custom_regex": "\\S+",
  "include_empty_words": false
}
```
- Allows custom regex patterns to override default counting methods
- Works with both word and line counting methods

## Metrics

### Basic Tokenization Metrics
- **Compression Rate**: Text size (bytes/chars/lines) per token - measures encoding efficiency
- **Fertility**: Tokens per word and per character - measures tokenization granularity  
- **Token Length**: Average token size in bytes/characters - measures vocabulary efficiency
- **Type-Token Ratio**: Unique tokens / total tokens - measures vocabulary usage diversity

### Information-Theoretic Metrics  
- **Rényi Entropy**: Information content at different α values - generalizes Shannon entropy
- **Vocabulary Utilization**: Fraction of vocabulary actually used - measures vocabulary efficiency
- **Entropy Analysis**: Token frequency distributions and information content

### Morphological Metrics
- **Boundary Precision/Recall**: How well tokens align with morpheme boundaries
- **Morpheme Preservation**: Whether morphemes remain intact after tokenization
- **MorphScore V2**: Advanced morphological evaluation ([Arnett et. al. 2025](https://arxiv.org/abs/2507.06378))

### Multilingual Fairness
- **Tokenizer Gini Coefficient**: Measures equitable treatment across languages, defined as:  

* $`L = \{1, \dots, n\}`$ be the set of languages, each weighted equally.  
* For every language $`\ell \in L`$, define the **token cost**  
```math
  c_\ell \;=\;
  \frac{\text{number of tokens produced by the tokenizer on language }\ell}
       {\text{number of raw bytes (or lines for parallel ds) in the same text}}
```
  (lower $`c_\ell`$ ⇒ cheaper encoding, higher ⇒ more byte-hungry).

* Let the mean cost be  
```math
  \mu \;=\; \frac{1}{n}\;\sum_{\ell=1}^{n} c_\ell.
```

Then the **Tokenizer Fairness Gini** with equal weights is  

```math
\mathrm{TFG}
=\frac{\displaystyle\sum_{i=1}^{n}\sum_{j=1}^{n} \lvert c_i - c_j \rvert}
        {2\,n^2\,\mu}
```
* **Range:** $`0 \le \mathrm{TFG} \le 1`$  
  * $`0`$: perfect parity (every language has identical byte-normalised token cost).  
  * $`1`$: maximal unfairness.

## Module Structure

```
tokenizer_analysis/
├── __init__.py                    # Main package exports
├── main.py                        # UnifiedTokenizerAnalyzer orchestration class
├── constants.py                   # Package-level constants
├── config/                        # Configuration modules
│   ├── __init__.py
│   ├── language_metadata.py      # LanguageMetadata for grouping analysis
│   └── text_measurement.py       # Text measurement configuration
├── core/                          # Core data structures and providers
│   ├── input_providers.py        # InputProvider implementations
│   ├── input_types.py            # TokenizedData and core types
│   ├── input_utils.py            # Input loading and validation utilities
│   └── validation.py             # Data validation functions
├── metrics/                       # Metrics computation modules
│   ├── __init__.py
│   ├── base.py                   # BaseMetrics with common utilities
│   ├── basic.py                  # Basic tokenization metrics
│   ├── information_theoretic.py  # Information-theoretic metrics
│   ├── morphological.py          # Morphological boundary alignment
│   ├── morphscore.py             # MorphScore neural evaluation
│   └── gini.py                   # Multilingual fairness metrics
├── loaders/                       # Data loading modules
│   ├── __init__.py
│   ├── constants.py              # Language code mappings (ISO639-1 to FLORES)
│   ├── morphological.py          # Morphological dataset loader
│   └── multilingual_data.py      # Multilingual text dataset loader
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── text_utils.py             # Text processing utilities
│   └── tokenizer_utils.py        # Tokenizer loading utilities
└── visualization/                 # Plotting and visualization
    ├── __init__.py
    ├── plotter.py                # TokenizerVisualizer main class
    ├── plots.py                  # Core plotting functions
    ├── data_extraction.py        # Data extraction for plotting
    ├── latex_tables.py           # LaTeX table generation
    └── visualization_config.py   # Visualization configuration
```
If you use this tokenizer analysis framework in your research, please cite it as follows:

```bibtex
@software{meister_tokenizer_analysis_2025,
  title = {TokEval: A Tokenizer Analysis Suite},
  author = {Meister, Clara},
  year = {2025},
  url = {https://github.com/cimeister/tokenizer-analysis}
}
```
