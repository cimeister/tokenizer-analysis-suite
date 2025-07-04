# Tokenizer Analysis Framework

A comprehensive framework for analyzing and comparing tokenizers across multiple languages and metrics.

## Features

- **Multi-tokenizer comparison**: Compare any number of tokenizers simultaneously
- **Comprehensive metrics**: Basic tokenization, information-theoretic, and morphological metrics
- **Multi-format support**: JSON, Parquet, and text file inputs
- **Visualization**: Automated plot generation for all metrics
- **Morphological analysis**: Boundary detection and preservation metrics
- **Scalable**: Optimized for large-scale multilingual analysis

## Installation

```bash
git clone https://github.com/cimeister/tokenizer-analysis.git
cd tokenizer-analysis
pip install -e .
```

## Quick Start

```bash
python scripts/compare_tokenizers.py \
    --tokenizer-config configs/tokenizer_config.json \
    --language-config configs/language_config.json \
    --output-dir results/
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

## Metrics

### Basic Tokenization Metrics
- **Compression Ratio**: Text size (bytes/chars) per token - measures encoding efficiency
- **Fertility**: Tokens per word (whitespace-delimited) and per character - measures tokenization granularity  
- **Token Length**: Average token size in bytes/characters - measures vocabulary design
- **Vocabulary Overlap**: Shared tokens between tokenizers - measures tokenizer similarity

### Information-Theoretic Metrics  
- **Type-Token Ratio**: Unique tokens / total tokens - measures vocabulary diversity
- **Rényi Entropy**: Information content at different α values - generalizes Shannon entropy
- **Vocabulary Utilization**: Fraction of vocabulary used - measures vocabulary efficiency
- **Unigram Distribution**: Token frequency analysis and rank statistics

### Morphological Metrics
- **Boundary Precision/Recall**: How well tokens align with morpheme boundaries
- **Morpheme Preservation**: Whether morphemes remain intact after tokenization
- **Word-Token Alignment**: Quality of tokenization relative to linguistic structure

### Multilingual Fairness
- **Tokenizer Gini**: Measures equitable treatment across languages, defined as:  

* \(L = \{1, \dots, n\}\) be the set of languages, each weighted equally.  
* For every language \(\ell \in L\), define the **token cost**  
```math
  c_\ell \;=\;
  \frac{\text{number of tokens produced by the tokenizer on language }\ell}
       {\text{number of raw **bytes** in the same text}}
```
  (lower $`\(c_\ell\)`$ ⇒ cheaper encoding, higher ⇒ more byte-hungry).

* Let the mean cost be  
```math
  \mu \;=\; \frac{1}{n}\;\sum_{\ell=1}^{n} c_\ell.
```

Then the **Tokenizer Fairness Gini** with equal weights is  

```math
\operatorname{TFG}
=\frac{\displaystyle\sum_{i=1}^{n}\sum_{j=1}^{n} \lvert c_i - c_j \rvert}
        {2\,n^2\,\mu}
```
* **Range:** $`\(0 \le \operatorname{TFG} \le 1\)`$  
  * $`\(0\)`$: perfect parity (every language has identical byte-normalised token cost).  
  * $`\(1\)`$: maximal unfairness.

## Module Structure

```
tokenizer_analysis/
├── __init__.py              # Main package exports
├── main.py                  # TokenizerAnalyzer orchestration class
├── metrics/                 # Metrics computation modules
│   ├── __init__.py
│   ├── base.py             # BaseMetrics with common utilities
│   ├── basic.py            # Basic tokenization metrics
│   ├── information_theoretic.py  # Information-theoretic metrics
│   └── morphological.py    # Morphological alignment metrics
│   └── gini.py    # Gini fairness metrics
├── loaders/                 # Data loading modules
│   ├── __init__.py
│   └── morphological.py    # Morphological dataset loader
│   └── constants.py    # Language code mapping
│   └── multilingual_data.py    # Text dataset loader
└── visualization/           # Plotting and visualization
    ├── __init__.py
    └── plotter.py          # TokenizerVisualizer class
```