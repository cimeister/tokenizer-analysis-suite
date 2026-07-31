"""
Unified tokenizer analysis script supporting both raw tokenizers and pre-tokenized data.

Raw tokenizer examples:
tokenizer-analysis --use-sample-data
tokenizer-analysis --tokenizer-config configs/tokenizer_config.json --language-config configs/language_config.json --measurement-config configs/text_measurement_config_bytes.json --samples-per-lang 3000 --output-dir analysis_results --verbose --run-grouped-analysis

Pre-tokenized data examples:
tokenizer-analysis --tokenized-data-file tokenized_data.json --language-config configs/language_config.json
tokenizer-analysis --tokenized-data-file tokenized_data.pkl --tokenized-data-config tokenized_config.json --language-config configs/language_config.json --run-grouped-analysis
"""
import logging
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tokenizer_analysis import create_analyzer_from_raw_inputs, create_analyzer_from_tokenized_data
from tokenizer_analysis.utils import setup_environment
from tokenizer_analysis.config.language_metadata import (
    LanguageMetadata,
    PACKAGE_ROOT,
    resolve_corpus_path,
)
from tokenizer_analysis.loaders.multilingual_data import (
    load_multilingual_data,
    ParquetEngineMissing,
)
from tokenizer_analysis.core.input_utils import InputLoader
from tokenizer_analysis.metrics.redundancy import MERGES as _MERGES
from tokenizer_analysis.constants import (
    DEFAULT_CER_TIME_BUDGET_S,
    LARGE_ARRAY_THRESHOLD,
    ARRAY_SAMPLING_POINTS,
    DEFAULT_MAX_SAMPLES,
    MIN_TOKENIZERS_FOR_PLOTS,
)
from tokenizer_analysis.visualization.visualization_config import LaTeXFormatting


class ConfigurationError(ValueError):
    """A config file the user named has the wrong shape or contents.

    Kept separate from the package's own exceptions so main() can print it
    without a traceback: the message names the flag, the file and the expected
    shape, which is the whole of what a caller needs.
    """


class OutputGenerationError(RuntimeError):
    """A requested output artifact could not be written.

    Raised at the end of a run so the results file is still saved and reported.
    main() turns it into exit 1 with the message and no traceback, since the
    cause is already logged and names the flag and the path.
    """


logger = logging.getLogger(__name__)


def _configure_cli_logging(log_file: str = "tokenizer_analysis.log") -> None:
    """Configure root logging for the console script.

    Called from ``main()``, never at import. Doing this at module scope meant
    that merely importing ``tokenizer_analysis.cli.run_analysis`` from another
    program reconfigured that program's root logger and created a log file in
    whatever directory it happened to be running in. A library must leave
    logging configuration to the application.
    """
    setup_environment()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def load_config_from_file(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_sample_configs() -> Dict[str, Dict]:
    """Create sample tokenizer configurations for testing."""
    return {
        "bpe": {
            "class": "huggingface",
            "path": "tokenizers/bpe.json"
        },
        "unigramlm": {
            "class": "huggingface",
            "path": "tokenizers/unigramlm.json"
        }
    }


# One message, used wherever a shipped config names a corpus file that is not
# on disk. FLORES+ is CC-BY-SA 4.0 and is fetched rather than redistributed, so
# a fresh checkout has the configs but not the corpus, and the error has to say
# which command produces it.
FLORES_MISSING_HINT = (
    "The FLORES+ corpus is not in {directory}. This repository does not "
    "redistribute it: fetch it with `uv run python scripts/fetch_flores.py` "
    "(the Hugging Face dataset is gated, so run `hf auth login` first). Use "
    "`--input` or `--language-config` to analyze your own corpus instead."
)


def _require_flores_corpus(paths) -> None:
    """Abort with the fetch instruction when a named FLORES+ file is absent."""
    missing = [p for p in paths if not Path(p).exists()]
    if not missing:
        return
    directory = Path(missing[0]).parent
    raise FileNotFoundError(
        f"{len(missing)} of {len(paths)} corpus file(s) named by this run do "
        f"not exist, starting with {missing[0]}. "
        + FLORES_MISSING_HINT.format(directory=directory)
    )


def create_sample_language_metadata() -> str:
    """Create sample LanguageMetadata configuration and return path to temp file."""
    import tempfile
    
    sample_metadata = {
        "languages": {
            "eng_Latn": {
                "name": "English",
                "iso_code": "en", 
                "data_path": "parallel/eng_Latn.txt"
            },
            "spa_Latn": {
                "name": "Spanish",
                "iso_code": "es",
                "data_path": "parallel/spa_Latn.txt"
            },
            "deu_Latn": {
                "name": "German", 
                "iso_code": "de",
                "data_path": "parallel/deu_Latn.txt"
            },
            "arb_Arab": {
                "name": "Arabic",
                "iso_code": "ar",
                "data_path": "parallel/arb_Arab.txt"
            },
            "rus_Cyrl": {
                "name": "Russian",
                "iso_code": "ru",
                "data_path": "parallel/rus_Cyrl.txt"
            }
        },
        "analysis_groups": {
            "script_family": {
                "Latin": ["eng_Latn", "spa_Latn", "deu_Latn"],
                "Arabic": ["arb_Arab"],
                "Cyrillic": ["rus_Cyrl"]
            },
            "resource_level": {
                "high": ["eng_Latn", "spa_Latn", "deu_Latn"],
                "medium": ["arb_Arab", "rus_Cyrl"],
                "low": []
            },
            "geographic_region": {
                "Western_Europe": ["eng_Latn", "spa_Latn", "deu_Latn"],
                "Middle_East": ["arb_Arab"],
                "Eastern_Europe": ["rus_Cyrl"]
            },
            "language_family": {
                "Indo_European": ["eng_Latn", "spa_Latn", "deu_Latn", "rus_Cyrl"],
                "Afro_Asiatic": ["arb_Arab"]
            }
        }
    }
    
    # Check before writing the config, so --use-sample-data on a fresh checkout
    # names the fetch command instead of failing several steps later with a
    # per-language load report.
    _require_flores_corpus([
        resolve_corpus_path(entry["data_path"])
        for entry in sample_metadata["languages"].values()
    ])

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_metadata, f, indent=2)
        return f.name


def create_sample_morphscore_config(data_dir: str = "morphscore_data") -> Dict[str, any]:
    """Create sample MorphScore configuration with default settings."""
    return {
        "data_dir": data_dir,
        "language_subset": None,
        "by_split": False,
        "freq_scale": True,
        "exclude_single_tok": False
    }

def _resolve_code_ast_config(args) -> Optional[Dict]:
    """Return the code-AST config dict from CLI args, or None if disabled.

    SILENT-FALLBACK HAZARD (fixed 2026-07-13). Omitting --code-ast-config does NOT disable the
    code metrics: it runs them on SYNTHETIC generated code. That produced a whole intrinsic run
    whose code metrics (AST boundary alignment, identifier fragmentation, indentation
    consistency) looked plausible but were computed on toy samples rather than real StarCoder
    data. AST full-alignment differed by 14% (0.562 synthetic vs 0.493 StarCoder) and silently
    corrupted every downstream correlation that used it. Warn loudly.
    """
    if args.code_ast_config:
        config = load_config_from_file(args.code_ast_config)
        # Check the shape here, where the flag and the file path are both in
        # hand. Left to the loader, a JSON array surfaced two files later as
        # AttributeError: 'list' object has no attribute 'items', swallowed at
        # one call site and uncaught at the other.
        from tokenizer_analysis.loaders.code_data import CodeDataLoader
        try:
            CodeDataLoader._validate_config(config)
        except (TypeError, ValueError) as e:
            raise ConfigurationError(f"--code-ast-config {args.code_ast_config}: {e}")
        return config
    if not args.no_code_ast:
        logger.warning(
            "=" * 78 + "\n"
            "NO --code-ast-config GIVEN: the code metrics (AST alignment, identifier "
            "fragmentation, indentation consistency) will be computed on SYNTHETIC sample code, "
            "NOT on real code. These numbers are NOT comparable to a run that passed "
            "--code-ast-config starcoder_ast_config.json. Pass --no-code-ast to disable them "
            "outright, or --code-ast-config to use real data.\n" + "=" * 78
        )
        return {}  # Empty dict triggers synthetic sample generation
    return None


_DROPPABLE_STAT_KEYS = {'sum', 'std_err', 'min', 'max'}


def _strip_stats(d):
    """Keep only {mean, std, median, count} from a compute_basic_stats() dict."""
    if isinstance(d, dict) and {'mean', 'std', 'count'} <= d.keys():
        return {k: v for k, v in d.items() if k not in _DROPPABLE_STAT_KEYS}
    return d


def _strip_per_language(per_lang):
    """Apply _strip_stats to each entry in a per-language dict."""
    if not isinstance(per_lang, dict):
        return per_lang
    return {lang: _strip_stats(v) if isinstance(v, dict) else v
            for lang, v in per_lang.items()}


def _sample_array(value):
    """Downsample large arrays for JSON export."""
    if isinstance(value, list) and len(value) > LARGE_ARRAY_THRESHOLD:
        step = len(value) // ARRAY_SAMPLING_POINTS
        return value[::step] if step > 1 else value[:ARRAY_SAMPLING_POINTS]
    return value


def _slim_tokenizer_entry(
    metric_name: str,
    tok_data: dict,
    tok_name: Optional[str] = None,
    metric_data: Optional[dict] = None,
) -> dict:
    """Normalize a single tokenizer's data to consistent {global, per_language, ...} keys.

    ``tok_name`` and ``metric_data`` are only needed by branches that must
    reach outside this tokenizer's ``per_tokenizer`` entry (e.g. operator
    isolation's pooled ``summary``, which lives as a sibling of
    ``per_tokenizer`` rather than inside it). Every other branch ignores them.
    """
    if not isinstance(tok_data, dict):
        return tok_data

    out = {}

    # Fields folded in from a redundant metric (see metrics/redundancy.py) are
    # carried through unchanged. Every branch below rebuilds `out` from a
    # per-metric whitelist, so without this they would be dropped and the merge
    # would silently delete the secondary metric rather than relocate it.
    for _sec, _pri, _field, _ in _MERGES:
        if metric_name == _pri and _field in tok_data:
            out[_field] = tok_data[_field]

    # --- Metrics with a standard 'global' stats dict ---
    if metric_name in ('fertility', 'avg_tokens_per_line'):
        if 'global' in tok_data:
            out['global'] = _strip_stats(tok_data['global'])
        elif 'stats' in tok_data:
            out['global'] = _strip_stats(tok_data['stats'])
        if 'per_language' in tok_data:
            out['per_language'] = _strip_per_language(tok_data['per_language'])
        # avg_tokens_per_line: keep total_lines if present
        if metric_name == 'avg_tokens_per_line' and 'total_lines' in tok_data:
            out['global'] = out.get('global', {})
            out['global']['total_lines'] = tok_data['total_lines']

    # --- Compression rate ---
    elif metric_name == 'compression_rate':
        if 'global' in tok_data:
            out['global'] = tok_data['global']  # {compression_rate, total_units, total_tokens}
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']

    # --- Vocabulary utilization ---
    elif metric_name == 'vocabulary_utilization':
        out['global'] = {
            'utilization': tok_data.get('global_utilization'),
            'used_tokens': tok_data.get('global_used_tokens'),
            'vocab_size': tok_data.get('global_vocab_size'),
        }
        # Cross-language dispersion of the utilization ratio
        out['per_language_mean'] = tok_data.get('per_language_mean')
        out['per_language_std'] = tok_data.get('per_language_std')
        out['per_language_cov'] = tok_data.get('per_language_cov')
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']

    # --- Type-token ratio ---
    elif metric_name == 'type_token_ratio':
        out['global'] = {
            'ttr': tok_data.get('global_ttr'),
            'types': tok_data.get('global_types'),
            'tokens': tok_data.get('global_tokens'),
        }

    # --- Unigram distribution metrics ---
    elif metric_name == 'unigram_distribution_metrics':
        out['global'] = {
            'unigram_entropy': tok_data.get('global_unigram_entropy'),
            'avg_token_rank': tok_data.get('global_avg_token_rank'),
        }
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']

    # --- Bigram entropy ---
    elif metric_name == 'bigram_entropy':
        out['global'] = {
            'bigram_entropy': tok_data.get('global_bigram_entropy'),
            'total_bigrams': tok_data.get('global_total_bigrams'),
            'types_evaluated': tok_data.get('global_types_evaluated'),
            'types_excluded': tok_data.get('global_types_excluded'),
        }
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']

    # --- Rényi efficiency: pivot from {renyi_X: {overall, lang...}} to {global: {renyi_X: v}, per_language: {lang: {renyi_X: v}}} ---
    elif metric_name == 'renyi_efficiency':
        global_vals = {}
        lang_vals = {}  # lang -> {renyi_X: v}
        for key, value in tok_data.items():
            if key.startswith('renyi_') and isinstance(value, dict):
                if 'overall' in value:
                    global_vals[key] = value['overall']
                for lang, score in value.items():
                    if lang != 'overall':
                        lang_vals.setdefault(lang, {})[key] = score
        out['global'] = global_vals
        if lang_vals:
            out['per_language'] = lang_vals

    # --- Tokenizer fairness (Gini) ---
    elif metric_name == 'tokenizer_fairness_gini':
        out['global'] = {
            'gini_coefficient': tok_data.get('gini_coefficient'),
            'mean_cost': tok_data.get('mean_cost'),
            'std_cost': tok_data.get('std_cost'),
            'cost_ratio': tok_data.get('cost_ratio'),
            'num_languages': tok_data.get('num_languages'),
        }
        if 'warning' in tok_data:
            out['global']['warning'] = tok_data['warning']
        if 'language_costs' in tok_data:
            out['per_language'] = tok_data['language_costs']

    # --- Lorenz curve data: keep as-is but sample large arrays ---
    elif metric_name == 'lorenz_curve_data':
        for key, value in tok_data.items():
            out[key] = _sample_array(value)

    # --- Token length: keep character_length and byte_length sub-dicts stripped ---
    elif metric_name == 'token_length':
        for key in ('character_length', 'byte_length', 'primary_length'):
            if key in tok_data:
                out[key] = _strip_stats(tok_data[key])

    # --- UTF-8 metrics ---
    elif metric_name in ('utf8_token_integrity', 'utf8_char_split'):
        if 'global' in tok_data:
            out['global'] = tok_data['global']
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']

    # --- MorphScore ---
    elif metric_name == 'morphscore':
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']
        if 'summary' in tok_data:
            out['global'] = tok_data['summary']
        if 'error' in tok_data:
            out['error'] = tok_data['error']

    # --- AST boundary alignment: drop by_category, rename overall->global, by_language->per_language ---
    elif metric_name == 'ast_boundary_alignment':
        if 'overall' in tok_data:
            out['global'] = tok_data['overall']
        if 'by_language' in tok_data:
            out['per_language'] = tok_data['by_language']
        # by_category dropped (full results only)

    # --- Digit boundary metrics: overall is per-language, keep by_digit_length and by_bucket ---
    elif metric_name in ('three_digit_boundary_alignment', 'digit_split_variability'):
        if 'overall' in tok_data:
            out['per_language'] = tok_data['overall']
        if 'by_digit_length' in tok_data:
            out['by_digit_length'] = tok_data['by_digit_length']
        if 'by_bucket' in tok_data:
            out['by_bucket'] = tok_data['by_bucket']

    # --- Numeric magnitude consistency: same as digit boundary + scaling ---
    elif metric_name == 'numeric_magnitude_consistency':
        if 'overall' in tok_data:
            out['per_language'] = tok_data['overall']
        if 'by_digit_length' in tok_data:
            out['by_digit_length'] = tok_data['by_digit_length']
        if 'scaling' in tok_data:
            out['scaling'] = tok_data['scaling']

    # --- Operator isolation rate: rename by_language->per_language, drop all by_category (full results only) ---
    elif metric_name == 'operator_isolation_rate':
        if 'by_language' in tok_data:
            per_lang = {}
            for lang, lang_data in tok_data['by_language'].items():
                if isinstance(lang_data, dict):
                    # Drop nested by_category from each language entry
                    entry = {k: v for k, v in lang_data.items() if k != 'by_category'}
                    per_lang[lang] = entry
                else:
                    per_lang[lang] = lang_data
            out['per_language'] = per_lang

        # `tok_data` here is metric_data['per_tokenizer'][tok_name], which
        # holds only by_category/by_language. The pooled rate for this
        # tokenizer lives in the sibling `summary` dict built by
        # _build_operator_results (math.py), so it has to be read from
        # `metric_data` rather than from `tok_data`. The pool is a
        # micro-average weighted by operator instances across prose, code and
        # math, so with a real code corpus it sits close to the code-domain
        # rate; by_domain (below) is what makes that pooled number readable.
        if metric_data is not None and tok_name is not None:
            global_stats = metric_data.get('summary', {}).get(tok_name)
            if global_stats:
                out['global'] = global_stats

            by_domain_full = metric_data.get('by_domain')
            if by_domain_full:
                by_domain_out = {}
                for domain, dom_data in by_domain_full.items():
                    dom_stats = dom_data.get('summary', {}).get(tok_name)
                    if dom_stats:
                        by_domain_out[domain] = dom_stats
                if by_domain_out:
                    out['by_domain'] = by_domain_out

    # --- Reconstruction fidelity: domain-keyed data ---
    elif metric_name == 'reconstruction_fidelity':
        for key, value in tok_data.items():
            if key == 'overall':
                out['global'] = value
            elif key in ('by_domain', 'by_language'):
                out['per_' + key.split('_', 1)[1]] = value
            else:
                out[key] = value

    # --- Identifier fragmentation ---
    elif metric_name == 'identifier_fragmentation':
        if 'overall' in tok_data:
            out['global'] = tok_data['overall']
        per_lang = tok_data.get('per_language') or tok_data.get('by_language')
        if per_lang:
            out['per_language'] = per_lang

    # --- Indentation consistency ---
    elif metric_name in ('indentation_preservation', 'indentation_consistency'):
        if 'by_language' in tok_data:
            out['per_language'] = tok_data['by_language']
        if 'overall' in tok_data:
            out['global'] = tok_data['overall']

    # --- Fallback: copy keys with basic renaming ---
    else:
        for key, value in tok_data.items():
            if key == 'overall':
                out['global'] = value
            elif key == 'by_language':
                out['per_language'] = value
            elif key == 'by_category':
                pass  # per_category only in full results
            elif key == 'global' and isinstance(value, dict):
                out['global'] = _strip_stats(value)
            elif key == 'per_language':
                out['per_language'] = _strip_per_language(value)
            else:
                out[key] = value

    return out


def _rename_by_category(obj):
    """Recursively rename by_category → per_category in a results dict."""
    if isinstance(obj, dict):
        return {
            ('per_category' if k == 'by_category' else k): _rename_by_category(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_rename_by_category(item) for item in obj]
    return obj


def slim_results_for_json(results: Dict) -> Dict:
    """Create a slimmed-down version of results for JSON export.

    Normalizes every metric to a consistent schema:
      {metric: {per_tokenizer: {tok: {global, per_language, ...}}, per_language, metadata}}
    Drops pairwise_comparisons, summary (derivable/duplicate data), and
    redundant stat fields (sum, std_err, min, max).
    """
    slimmed = {}

    for metric_name, metric_data in results.items():
        if not isinstance(metric_data, dict):
            slimmed[metric_name] = metric_data
            continue

        # Provenance is not a metric; it has no per_tokenizer block and must
        # survive verbatim, so the slimming branches below never see it.
        if metric_name == 'run_metadata':
            slimmed[metric_name] = metric_data
            continue

        # Grouped results are {group_type: {group_name: {metric: block}}}, one
        # level deeper than everything else here, so the branches below found no
        # per_tokenizer key and wrote an empty dict. A run with
        # --run-grouped-analysis and no --save-full-results published
        # "grouped_analysis": {} and lost the whole grouped result. Slim each
        # group's metric block with the same rules instead.
        if metric_name == 'grouped_analysis':
            slimmed[metric_name] = {
                group_type: {
                    group_name: slim_results_for_json(group_result)
                    for group_name, group_result in groups.items()
                }
                for group_type, groups in metric_data.items()
            }
            continue

        out = {}

        # Normalize per-tokenizer entries
        if 'per_tokenizer' in metric_data:
            out['per_tokenizer'] = {
                tok: _slim_tokenizer_entry(metric_name, tok_data, tok, metric_data)
                for tok, tok_data in metric_data['per_tokenizer'].items()
            }

        # Cross-tokenizer per-language leaderboard (top-level)
        per_lang = metric_data.get('per_language') or metric_data.get('by_language')
        if per_lang:
            out['per_language'] = per_lang

        # Keep vocabulary sizes if present
        if 'vocabulary_sizes' in metric_data:
            out['vocabulary_sizes'] = metric_data['vocabulary_sizes']

        # Keep metadata
        if 'metadata' in metric_data:
            out['metadata'] = metric_data['metadata']

        # Renyi publishes the pre-1.0 normalization alongside the published one
        # so an older value stays reproducible. Keep it; it is not derivable
        # from the primary block, since the divisors differ per tokenizer and
        # per language.
        if 'observed_normalization' in metric_data:
            out['observed_normalization'] = metric_data['observed_normalization']

        # Successor entropy under the reference normalizer and aggregation,
        # published next to this library's own definition for comparison.
        if 'reference_definition' in metric_data:
            out['reference_definition'] = metric_data['reference_definition']

        # Operator isolation: keep the per-domain split (prose / code / math).
        #
        # It is NOT derivable from the slimmed per_tokenizer block, because that block POOLS the
        # three corpora. The corpus decides the answer (a matched pretokenizer pair differs by 0.60
        # on prose and 0.01 on math), so dropping by_domain here would leave the summary file
        # carrying only the pooled number, and any consumer reading it would silently get a
        # different quantity than the one it asked for. Keep each domain's per-tokenizer summary and
        # the source corpus it was computed on; the bulky per-tokenizer/per-language detail stays in
        # analysis_results_full.json.
        if 'by_domain' in metric_data:
            out['by_domain'] = {
                domain: {
                    'summary': dom_data.get('summary', {}),
                    'source': dom_data.get('source'),
                }
                for domain, dom_data in metric_data['by_domain'].items()
            }

        # Deliberately drop: pairwise_comparisons, summary
        slimmed[metric_name] = out

    return slimmed


# --- Deprecation shims for flags/choices removed in 1.0.0 ---
# The standalone morphological boundary metric was removed; MorphScore replaces it.
# These stubs make old command lines fail with a clear pointer instead of an opaque
# argparse error. See MIGRATION.md.
_MIGRATION_HINT = "See MIGRATION.md for the full list of changes."

_REMOVED_FLAG_MESSAGES = {
    "--morphological-config": (
        "--morphological-config was removed in tokenizer-intrinsic-evals 1.0.0. The standalone "
        "morphological boundary metric no longer exists; use MorphScore instead via "
        "--morphscore (defaults) or --morphscore-config <file>. " + _MIGRATION_HINT
    ),
}

_LATEX_TABLE_TYPES = ("basic", "information", "comprehensive")


class _RemovedFlagAction(argparse.Action):
    """Argparse action for flags removed in 1.0.0: fail loudly with a pointer."""

    def __init__(self, option_strings, dest, **kwargs):
        # Accept an optional trailing value so both "--flag" and "--flag x" reach us
        # with our message rather than an "expected one argument" error.
        kwargs["nargs"] = "?"
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(_REMOVED_FLAG_MESSAGES.get(
            option_string,
            f"{option_string} was removed in tokenizer-intrinsic-evals 1.0.0. " + _MIGRATION_HINT,
        ))


class _LatexTableTypesAction(argparse.Action):
    """Validate --latex-table-types, pointing users off the removed 'morphological' type."""

    def __call__(self, parser, namespace, values, option_string=None):
        for value in values:
            if value == "morphological":
                parser.error(
                    "--latex-table-types no longer supports 'morphological' (removed in "
                    "tokenizer-analysis 1.0.0 with the standalone morphological metric). "
                    "Use MorphScore via --morphscore. Valid types: "
                    + ", ".join(_LATEX_TABLE_TYPES) + ". " + _MIGRATION_HINT
                )
            if value not in _LATEX_TABLE_TYPES:
                parser.error(
                    "argument --latex-table-types: invalid choice: %r (choose from %s)"
                    % (value, ", ".join(repr(v) for v in _LATEX_TABLE_TYPES))
                )
        setattr(namespace, self.dest, values)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for tokenizer analysis."""
    parser = argparse.ArgumentParser(
        description="Enhanced modular tokenizer analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-tokenizer analysis with raw tokenizers (supports any number of tokenizers)
  uv run tokenizer-analysis --use-sample-data
  
  # Load from configuration files (supports 1+ tokenizers)
  uv run tokenizer-analysis --tokenizer-config tokenizers.json --language-config languages.json
  
  # Use pre-tokenized data from file
  uv run tokenizer-analysis --tokenized-data-file tokenized_data.json --language-config languages.json
  
  # Use pre-tokenized data with configuration
  uv run tokenizer-analysis --tokenized-data-file tokenized_data.pkl --tokenized-data-config tokenized_config.json --language-config languages.json
  
  # Filter by script family and run grouped analysis (includes grouped plots)
  uv run tokenizer-analysis --use-sample-data --filter-script-family Latin --run-grouped-analysis
  
  # Filter by resource level  
  uv run tokenizer-analysis --use-sample-data --filter-resource-level high
  
  # Run grouped analysis across all script families and resource levels
  uv run tokenizer-analysis --use-sample-data --run-grouped-analysis
  
  # Pairwise comparison only (restricts to 2 specific tokenizers)
  uv run tokenizer-analysis --pairwise tok1 tok2 --use-sample-data
  
  # Skip plots for faster processing
  uv run tokenizer-analysis --use-sample-data --no-plots
  
  # Enable MorphScore analysis with default settings
  uv run tokenizer-analysis --use-sample-data --morphscore
  
  # Use custom MorphScore configuration
  uv run tokenizer-analysis --tokenizer-config tokenizers.json --language-config languages.json --morphscore-config morphscore.json
  
  # Explicitly disable MorphScore (useful when config file has it enabled)
  uv run tokenizer-analysis --use-sample-data
  
  # Generate LaTeX tables for results
  uv run tokenizer-analysis --use-sample-data --generate-latex-tables
  
  # Generate specific LaTeX table types
  uv run tokenizer-analysis --use-sample-data --generate-latex-tables --latex-table-types basic comprehensive
  
  # Generate per-language plots in addition to standard plots
  uv run tokenizer-analysis --use-sample-data --per-language-plots
  
  # Generate per-language plots with additional faceted plots (subplots per tokenizer)
  uv run tokenizer-analysis --use-sample-data --per-language-plots --faceted-plots
  
  # Generate grouped analysis with additional faceted plots for grouped metrics
  uv run tokenizer-analysis --use-sample-data --run-grouped-analysis --faceted-plots
  
  # Generate custom LaTeX tables from configuration file
  uv run tokenizer-analysis --use-sample-data --custom-latex-config custom_tables.json
  
  # Save both summary and full detailed results
  uv run tokenizer-analysis --use-sample-data --save-full-results
  
  # Save tokenized data for later reuse
  uv run tokenizer-analysis --use-sample-data --save-tokenized-data
  
  # Save tokenized data to specific path
  uv run tokenizer-analysis --tokenizer-config tokenizers.json --language-config languages.json --save-tokenized-data --tokenized-data-output-path my_tokenized_data.pkl
  
  # Complete workflow: generate tokenized data, then reuse it
  # Step 1: Generate and save tokenized data
  uv run tokenizer-analysis --use-sample-data --save-tokenized-data --tokenized-data-output-path results/tokenized_data.pkl
  # Step 2: Use the saved tokenized data (much faster)
  uv run tokenizer-analysis --tokenized-data-file results/tokenized_data.pkl --tokenized-data-config configs/sample_tokenized_config.json --language-config languages.json
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
        "--input",
        type=str,
        metavar="PATH",
        help="Analyze a single corpus: a file (.txt/.json/.jsonl/.parquet) or a "
             "directory of them. Takes exactly one path; for several corpora use "
             "--language-config. Cannot be combined with --language-config or "
             "--use-sample-data."
    )
    parser.add_argument(
        "--input-label",
        type=str,
        metavar="NAME",
        help="Name for the --input corpus in the results (default: the file or "
             "directory stem). Appears as the single key under per_language."
    )
    parser.add_argument(
        "--morphscore-config",
        type=str,
        help="JSON file with MorphScore configuration (requires raw tokenization)"
    )
    # Removed in 1.0.0: fail with a pointer instead of an opaque argparse error.
    parser.add_argument(
        "--morphological-config",
        action=_RemovedFlagAction,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--morphscore",
        action="store_true",
        help="Enable MorphScore analysis with default settings (requires raw tokenization)"
    )
    parser.add_argument(
        "--morphscore-data-dir",
        type=str,
        default="morphscore_data",
        help="Directory containing data for MorphScore analysis"
    )
    parser.add_argument(
        "--measurement-config",
        type=str,
        help="JSON file with text measurement configuration (method, counting functions, etc.)"
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Run the built-in demo: two bundled tokenizers over five FLORES+ "
             "languages. Supplies its own tokenizers and corpus, so it cannot be "
             "combined with --tokenizer-config, --language-config, --input or "
             "--measurement-config. Requires a source checkout (the demo data "
             "lives in parallel/ and tokenizers/, not in the installed package)."
    )
    
    # NEW: Pre-tokenized data support
    parser.add_argument(
        "--tokenized-data-config",
        type=str,
        help="JSON file with pre-tokenized data configuration including vocabulary file paths"
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
        "--no-global-lines",
        action="store_true",
        help="Hide global average reference lines in grouped/per-language plots"
    )
    parser.add_argument(
        "--samples-per-lang",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Number of text samples per language (default: {DEFAULT_MAX_SAMPLES})"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output plots and logs"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
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
    
    parser.add_argument(
        "--run-grouped-analysis",
        action="store_true",
        help="Run analysis grouped by script families and resource levels"
    )
    
    # LaTeX table generation options
    parser.add_argument(
        "--generate-latex-tables",
        action="store_true",
        help="Generate LaTeX tables for analysis results"
    )
    parser.add_argument(
        "--latex-table-types",
        nargs="+",
        default=["basic", "comprehensive"],
        action=_LatexTableTypesAction,
        metavar="{basic,information,comprehensive}",
        help="Types of (default) LaTeX tables to generate"
    )
    parser.add_argument(
        "--latex-output-dir",
        type=str,
        help="Directory for LaTeX table output (default: same as --output-dir)"
    )
    parser.add_argument(
        "--custom-latex-config",
        type=str,
        help="JSON configuration file for custom LaTeX tables"
    )
    
    # Plot generation options
    parser.add_argument(
        "--per-language-plots",
        action="store_true",
        help="Generate per-language plots in addition to individual plots (does not apply to grouped analysis)"
    )
    parser.add_argument(
        "--faceted-plots",
        action="store_true",
        help="Generate additional faceted plots (one subplot per tokenizer with shared y-axis) for grouped analysis (--run-grouped-analysis) and per-language plots (--per-language-plots). Normal plots are still generated."
    )
    
    parser.add_argument(
        "--no-digit-boundary",
        action="store_true",
        help="Skip digit boundary alignment, digit split variability, numeric magnitude consistency, and operator isolation analysis"
    )
    parser.add_argument(
        "--math-data",
        type=str,
        help="Path to math-rich text file (.txt or .json) for digit boundary metrics. "
             "When provided, this data is tokenized per-tokenizer and used instead of "
             "the general dataset for digit boundary, digit split variability, numeric "
             "magnitude consistency, and operator isolation metrics."
    )
    parser.add_argument(
        "--use-builtin-math-data",
        action="store_true",
        help="Use the bundled math sample dataset (285 expressions) for "
             "digit boundary metrics instead of the general input text. "
             "Ignored if --math-data is also provided."
    )
    parser.add_argument(
        "--code-ast-config",
        type=str,
        help="JSON file mapping programming languages to code file/directory paths for AST boundary analysis"
    )
    parser.add_argument(
        "--no-code-ast",
        action="store_true",
        help="Skip AST boundary alignment analysis"
    )
    parser.add_argument(
        "--no-utf8-integrity",
        action="store_true",
        help="Skip UTF-8 character boundary integrity analysis"
    )
    parser.add_argument(
        "--no-reconstruction",
        action="store_true",
        help="Skip reconstruction fidelity analysis (decode round-trip)"
    )
    parser.add_argument(
        "--cer-time-budget",
        type=float,
        default=DEFAULT_CER_TIME_BUDGET_S,
        metavar="SECONDS",
        help="Max seconds to spend on CER computation per tokenizer. "
             "After a warmup phase the total time is extrapolated; if it "
             "exceeds this budget, CER and whitespace fidelity are skipped "
             "for that tokenizer. 0 disables the budget (default: 10.0)"
    )
    # Tokenized data saving options
    parser.add_argument(
        "--save-tokenized-data",
        action="store_true",
        help="Save tokenized data to file (only when processing raw data)"
    )
    parser.add_argument(
        "--tokenized-data-output-path",
        type=str,
        help="Path to save tokenized data (default: output_dir/tokenized_data.pkl)"
    )
    return parser


def _file_digest(path: str) -> Optional[str]:
    """Short SHA-256 of a file, or None if it cannot be read.

    Truncated to 16 hex characters: enough to tell two artifacts apart in a
    provenance record, short enough to read in a diff.
    """
    import hashlib

    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]
    except OSError:
        return None


def _git_state() -> Dict[str, Any]:
    """Commit of the repo this package runs from, and whether it was modified.

    The commit alone does not describe the code that ran: on a modified working
    tree it names the last commit, not what executed. Every development run is
    in that state, so a results file recording only the hash overstates what it
    can reproduce.
    """
    import subprocess

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    state: Dict[str, Any] = {"commit": None, "dirty": None}
    try:
        head = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, timeout=5,
        )
        if head.returncode != 0:
            return state
        state["commit"] = head.stdout.decode().strip()
        status = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, timeout=10,
        )
        if status.returncode == 0:
            state["dirty"] = bool(status.stdout.decode().strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return state


def _build_run_metadata(args: argparse.Namespace, tokenizer_configs: Dict) -> Dict:
    """Describe the run, so a results file can be traced back to what made it.

    Without this a results file is untraceable: nothing recorded the package
    version, the code revision, which configs were read, or which tokenizer
    files were measured. Two files with different numbers gave no way to tell
    whether the tokenizer changed, the corpus changed, or the library did.

    Hashes cover the inputs that decide the numbers, so a rerun that produces
    different values can be attributed.
    """
    from tokenizer_analysis import __version__

    # On the --tokenized-data-file path no tokenizer file is read, so the only
    # artifacts to hash are the vocabulary dumps the cache names. Those are not
    # the tokenizers that produced the ids, and the class is not known, so the
    # entries say what they are rather than imitating the full form.
    replaying_a_cache = args.tokenized_data_file is not None
    tokenizers = {}
    for name, cfg in (tokenizer_configs or {}).items():
        entry: Dict[str, Any] = {}
        if replaying_a_cache:
            entry["source"] = "vocabulary file from --tokenized-data-config"
        else:
            entry["class"] = cfg.get("class", "huggingface")
        path = cfg.get("path")
        if path:
            entry["path"] = path
            digest = _file_digest(path)
            if digest:
                entry["sha256_16"] = digest
        tokenizers[name] = entry

    configs = {}
    for flag, value in (
        ("tokenizer_config", args.tokenizer_config),
        ("language_config", args.language_config),
        ("measurement_config", args.measurement_config),
        ("code_ast_config", args.code_ast_config),
        ("morphscore_config", args.morphscore_config),
    ):
        if value:
            configs[flag] = {"path": value, "sha256_16": _file_digest(value)}

    git = _git_state()
    return {
        "package_version": __version__,
        "git_commit": git["commit"],
        "git_tree_modified": git["dirty"],
        "configs": configs,
        "tokenizers": tokenizers,
        "corpus": {
            "input": args.input,
            "input_label": args.input_label,
            "use_sample_data": bool(args.use_sample_data),
            "tokenized_data_file": args.tokenized_data_file,
            "samples_per_lang": args.samples_per_lang,
        },
        "metrics_disabled": sorted(
            flag for flag, off in (
                ("code_ast", args.no_code_ast),
                ("digit_boundary", args.no_digit_boundary),
                ("utf8_integrity", args.no_utf8_integrity),
                ("reconstruction", args.no_reconstruction),
            ) if off
        ),
    }


def _corpus_source_from_input(input_path: str, label: Optional[str]) -> str:
    """Write a one-language config for --input and return its path.

    Reuses the LanguageMetadata path rather than adding a second corpus-loading
    route, so a single corpus goes through exactly the same loader, measurement
    and metric code as a multilingual run. The only difference is that there is
    one group instead of several.
    """
    import tempfile

    path = Path(input_path)
    if not path.exists():
        raise ValueError(
            f"--input path does not exist: {input_path}. Pass a text file "
            "(.txt/.json/.jsonl/.parquet) or a directory containing them."
        )

    corpus_label = label or path.stem or path.name
    if not corpus_label:
        raise ValueError(
            f"Could not derive a corpus label from --input {input_path!r}; "
            "pass --input-label explicitly."
        )

    # Absolute, because a relative data_path in a language config is anchored
    # at the package root (see config.language_metadata.resolve_corpus_path)
    # while --input is a command-line path and means what it means from the
    # working directory the user typed it in.
    config = {"languages": {corpus_label: {"data_path": str(path.resolve())}}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f, indent=2)
        logger.info(
            "Single-corpus mode: analyzing %s under the label %r.",
            path, corpus_label,
        )
        return f.name


def _validate_corpus_source(args: argparse.Namespace) -> None:
    """Reject ambiguous or absent corpus sources before any work happens.

    Three flags can supply a corpus and they are mutually exclusive.
    --use-sample-data used to win silently: it overwrote --tokenizer-config,
    --language-config and --measurement-config with the demo values, so a run
    that named the user's tokenizers and corpus produced a complete, plausible
    results file describing the bundled demo instead. Nothing in the output said
    so. Refuse the combination rather than pick one.
    """
    if args.tokenized_data_file is not None:
        return  # Pre-tokenized mode supplies its own data; validated separately.

    overridden = [
        flag for flag, value in (
            ("--tokenizer-config", args.tokenizer_config),
            ("--language-config", args.language_config),
            ("--input", args.input),
            ("--measurement-config", args.measurement_config),
        ) if value
    ]
    if args.use_sample_data and overridden:
        raise ValueError(
            "--use-sample-data supplies its own tokenizers, corpus and "
            "measurement settings, so it conflicts with "
            + ", ".join(overridden)
            + ". Earlier versions accepted this and silently ignored those flags, "
            "reporting demo numbers under your configuration. Drop "
            "--use-sample-data to analyze your own data, or drop the other flags "
            "to run the demo."
        )

    if args.input and args.language_config:
        raise ValueError(
            "--input analyzes a single corpus and --language-config analyzes a "
            "set of them; pass one or the other. To analyze several corpora, list "
            "them under 'languages' in the --language-config file."
        )

    if not (args.use_sample_data or args.input or args.language_config):
        raise ValueError(
            "No corpus given. Pass --input PATH for a single corpus, "
            "--language-config FILE for several, or --use-sample-data for the "
            "bundled demo. Earlier versions fell back to the bundled five-language "
            "FLORES+ sample here, which meant results could describe demo data "
            "rather than yours."
        )


def run_from_args(args: argparse.Namespace):
    """Run tokenizer analysis from a parsed CLI namespace."""

    _validate_corpus_source(args)

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
        measurement_config = None  # Use default for sample data
        
        # Configure MorphScore for sample data
        morphscore_config = None
        if args.morphscore or args.morphscore_config:
            if args.morphscore_config:
                morphscore_config = load_config_from_file(args.morphscore_config)
            else:
                morphscore_config = create_sample_morphscore_config(args.morphscore_data_dir)

        code_ast_config = _resolve_code_ast_config(args)
    elif use_tokenized_data:
        # Pre-tokenized data mode
        if not args.tokenized_data_file:
            raise ValueError("Must specify --tokenized-data-file for pre-tokenized mode")
        
        # Load tokenized data
        tokenized_data = InputLoader.load_from_file(args.tokenized_data_file)

        # No tokenizer files are read on this path, so run_metadata records the
        # vocabulary files the cache names instead. Leaving the name unbound
        # crashed the whole replay at the last line of the run, after every
        # metric had been computed, with UnboundLocalError: cannot access local
        # variable 'tokenizer_configs'.
        tokenizer_configs: Dict = {}

        # Load vocabulary files if config provided
        vocabularies = {}
        if args.tokenized_data_config:
            config = load_config_from_file(args.tokenized_data_config)
            if 'vocabulary_files' in config:
                vocabularies = InputLoader.load_vocabularies_from_config(config['vocabulary_files'])
                tokenizer_configs = {
                    name: {'path': path}
                    for name, path in config['vocabulary_files'].items()
                }
        
        # If no vocabularies loaded, estimate from data
        if not vocabularies:
            logger.warning("No vocabulary files loaded, will estimate vocabulary sizes from tokenized data")
        
        # Language metadata is still needed for grouping and per-language labels.
        # It is not inferable from a pickle of token ids, so require it rather
        # than substituting the bundled five-language sample, which would label
        # the user's data with FLORES+ language codes it has nothing to do with.
        if not args.language_config:
            raise ValueError(
                "--tokenized-data-file needs --language-config: the pickle holds "
                "token ids, not the language labels or groupings the metrics "
                "report against. Pass the same language config used when the "
                "data was tokenized."
            )
        language_config_path = args.language_config
        
        # Load text measurement configuration
        measurement_config = None
        if args.measurement_config:
            from tokenizer_analysis.config import TextMeasurementConfig
            measurement_config_dict = load_config_from_file(args.measurement_config)
            measurement_config = TextMeasurementConfig.from_dict(measurement_config_dict)

        # MorphScore not supported with pre-tokenized data
        morphscore_config = None
        if args.morphscore or args.morphscore_config:
            logger.warning("MorphScore analysis not supported with pre-tokenized data. Requires raw tokenization.")
            morphscore_config = None

        # AST boundary metrics not supported with pre-tokenized data (needs encode())
        code_ast_config = None
        if args.code_ast_config:
            logger.warning("AST boundary analysis not supported with pre-tokenized data. Requires raw tokenization.")
    else:
        # Raw tokenizer mode
        if not args.tokenizer_config:
            raise ValueError(
                "Must specify --tokenizer-config (a JSON file mapping tokenizer "
                "names to {\"class\": ..., \"path\": ...}), or --use-sample-data "
                "to run the bundled demo."
            )

        tokenizer_configs = load_config_from_file(args.tokenizer_config)

        if args.input:
            language_config_path = _corpus_source_from_input(
                args.input, args.input_label
            )
        else:
            # Load language configuration. This is a JSON file; a directory is an error.
            language_config_path = args.language_config

        # Load text measurement configuration
        measurement_config = None
        if args.measurement_config:
            from tokenizer_analysis.config import TextMeasurementConfig
            measurement_config_dict = load_config_from_file(args.measurement_config)
            measurement_config = TextMeasurementConfig.from_dict(measurement_config_dict)
        
        # Configure MorphScore for raw tokenizer mode
        morphscore_config = None
        if args.morphscore or args.morphscore_config:
            if args.morphscore_config:
                morphscore_config = load_config_from_file(args.morphscore_config)
            else:
                morphscore_config = create_sample_morphscore_config(args.morphscore_data_dir)

        code_ast_config = _resolve_code_ast_config(args)

    # Load language metadata
    logger.info("Loading language metadata...")
    language_metadata = LanguageMetadata(language_config_path)
    
    # Initialize analyzer based on input mode
    logger.info("Initializing unified tokenizer analyzer...")
    
    if use_tokenized_data:
        # Pre-tokenized data mode
        analyzer = create_analyzer_from_tokenized_data(
            tokenized_data=tokenized_data,
            vocabularies=vocabularies,
            measurement_config=measurement_config,
            language_metadata=language_metadata,
            plot_save_dir=args.output_dir,
            morphscore_config=morphscore_config,
            code_ast_config=code_ast_config,
            show_global_lines=not args.no_global_lines,
            per_language_plots=args.per_language_plots,
            faceted_plots=args.faceted_plots,
            math_data_path=args.math_data,
            use_builtin_math_data=args.use_builtin_math_data
        )
    else:
        # Raw tokenizer mode
        # Validate tokenizer configs
        if not tokenizer_configs or len(tokenizer_configs) < 1:
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
            filter_by_group = ('script_family', args.filter_script_family)
        elif args.filter_resource_level:
            filter_by_group = ('resource_level', args.filter_resource_level)
        
        try:
            language_texts = load_multilingual_data(
                language_metadata=language_metadata,
                max_texts_per_language=args.samples_per_lang,
                filter_by_group=filter_by_group
            )
        except ValueError as e:
            # Every ValueError this loader raises describes the config or the
            # corpus it names: an unknown group, a language with no data path, a
            # path that did not load. The message is the whole of what a caller
            # needs, so print it without a stack.
            message = str(e)
            # A config shipped in configs/ names files under parallel/, which
            # this repository does not redistribute, so on a fresh checkout the
            # per-language report is a wall of missing paths with no hint about
            # what produces them.
            flores_dir = str(PACKAGE_ROOT / "parallel")
            if flores_dir in message and not Path(flores_dir).exists():
                message += " " + FLORES_MISSING_HINT.format(directory=flores_dir)
            raise ConfigurationError(message)
        
        if not language_texts:
            raise ValueError("No valid language texts loaded")
        
        # Initialize unified analyzer using convenience function
        analyzer = create_analyzer_from_raw_inputs(
            tokenizer_configs=tokenizer_configs,
            language_texts=language_texts,
            measurement_config=measurement_config,
            language_metadata=language_metadata,
            plot_save_dir=args.output_dir,
            morphscore_config=morphscore_config,
            code_ast_config=code_ast_config,
            show_global_lines=not args.no_global_lines,
            per_language_plots=args.per_language_plots,
            faceted_plots=args.faceted_plots,
            math_data_path=args.math_data,
            use_builtin_math_data=args.use_builtin_math_data
        )
    # Run analysis
    logger.info("Starting tokenizer analysis...")
    
    if args.pairwise:
        # Pairwise comparison - for now, just run regular analysis and filter results
        logger.info(f"Running pairwise comparison: {args.pairwise[0]} vs {args.pairwise[1]}")
        results = analyzer.run_analysis(
            save_plots=not args.no_plots,

            include_morphscore=morphscore_config is not None,
            include_digit_boundary=not args.no_digit_boundary,
            include_code_ast=not args.no_code_ast,
            include_utf8_integrity=not args.no_utf8_integrity,
            include_reconstruction=not args.no_reconstruction,
            verbose=args.verbose,
            save_tokenized_data=args.save_tokenized_data,
            tokenized_data_path=args.tokenized_data_output_path,
            cer_time_budget_s=args.cer_time_budget,
        )
    else:
        # Full multi-tokenizer analysis
        results = analyzer.run_analysis(
            save_plots=not args.no_plots,

            include_morphscore=morphscore_config is not None,
            include_digit_boundary=not args.no_digit_boundary,
            include_code_ast=not args.no_code_ast,
            include_utf8_integrity=not args.no_utf8_integrity,
            include_reconstruction=not args.no_reconstruction,
            verbose=args.verbose,
            save_tokenized_data=args.save_tokenized_data,
            tokenized_data_path=args.tokenized_data_output_path,
            cer_time_budget_s=args.cer_time_budget,
        )
        
        if args.run_grouped_analysis and analyzer.language_metadata:
            logger.info("Running grouped analysis by script families and resource levels...")
            
            # Use the unified analyzer's built-in grouped analysis
            # Pass base results to avoid recomputing metrics
            grouped_results = analyzer.run_grouped_analysis(
                group_by=analyzer.language_metadata.analysis_groups.keys(),
                save_plots=not args.no_plots,
                base_results=results,
                include_reconstruction=not args.no_reconstruction,
                cer_time_budget_s=args.cer_time_budget,
            )
            
            # Add grouped results to main results
            results['grouped_analysis'] = grouped_results
    
    # Record what produced this file, before writing it.
    results['run_metadata'] = _build_run_metadata(args, tokenizer_configs)

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
        full_results_json = convert_for_json(_rename_by_category(results))
        with open(full_results_file, 'w') as f:
            json.dump(full_results_json, f, indent=2)
    
    # Requested outputs that could not be produced. The analysis itself and its
    # results file are unaffected, so the run finishes and reports where they
    # are, then exits non-zero naming what is missing. Exiting 0 here published
    # a "Results saved to" line for a run that skipped an output the user asked
    # for by name.
    unproduced_outputs: List[str] = []

    # Generate LaTeX tables if requested
    if args.generate_latex_tables:
        logger.info("Generating LaTeX tables...")
        latex_output_dir = args.latex_output_dir or os.path.join(args.output_dir, "latex_tables")
        
        formatting_options = {
            'bold_best': LaTeXFormatting.BOLD_BEST,
            'include_std_err': LaTeXFormatting.INCLUDE_STD_ERR,
            'std_err_size': LaTeXFormatting.STD_ERROR_SIZE
        }
        
        try:
            latex_tables = analyzer.generate_latex_tables(
                results=results,
                output_dir=latex_output_dir,
                table_types=args.latex_table_types,
                **formatting_options
            )
            
            logger.info(f"Generated {len(latex_tables)} LaTeX tables in {latex_output_dir}")
            for table_type, content in latex_tables.items():
                print(f"LaTeX {table_type} table: {latex_output_dir}/{table_type}_metrics_table.tex")
                
        except Exception as e:
            logger.error(
                f"--generate-latex-tables: no table was written to {latex_output_dir}: {e}"
            )
            unproduced_outputs.append(f"--generate-latex-tables ({latex_output_dir})")
    
    # Generate custom LaTeX tables if config provided
    if args.custom_latex_config:
        logger.info(f"Loading custom LaTeX configuration from {args.custom_latex_config}")
        try:
            custom_config = load_config_from_file(args.custom_latex_config)
            latex_output_dir = args.latex_output_dir or args.output_dir
            
            # Prepare formatting options
            formatting_options = {
                'bold_best': LaTeXFormatting.BOLD_BEST,
                'include_std_err': LaTeXFormatting.INCLUDE_STD_ERR,
                'std_err_size': LaTeXFormatting.STD_ERROR_SIZE
            }
            
            # Generate each custom table defined in config
            for table_name, table_config in custom_config.items():
                if not isinstance(table_config, dict):
                    logger.warning(f"Skipping invalid table config: {table_name}")
                    continue
                
                metrics = table_config.get('metrics', [])
                caption = table_config.get('caption', f"Custom Table: {table_name}")
                label = table_config.get('label', f"tab:custom_{table_name}")
                
                if not metrics:
                    logger.warning(f"No metrics specified for table {table_name}")
                    continue
                
                logger.info(f"Generating custom LaTeX table '{table_name}' with metrics: {metrics}")
                
                custom_output_path = f"{latex_output_dir}/custom_{table_name}_table.tex"
                custom_table = analyzer.generate_custom_latex_table(
                    results=results,
                    custom_metrics=metrics,
                    output_path=custom_output_path,
                    caption=caption,
                    label=label,
                    **formatting_options
                )
                
                if custom_table:
                    logger.info(f"Custom LaTeX table '{table_name}' saved to {custom_output_path}")
                    print(f"Custom LaTeX table '{table_name}': {custom_output_path}")
                else:
                    logger.warning(f"Custom LaTeX table '{table_name}' generation failed")
                    
        except Exception as e:
            logger.error(
                f"--custom-latex-config {args.custom_latex_config}: no table was "
                f"written: {e}"
            )
            unproduced_outputs.append(
                f"--custom-latex-config {args.custom_latex_config}"
            )
    
    logger.info("Analysis complete!")
    print(f"\nResults saved to: {args.output_dir}")
    if not args.no_plots:
        print(f"Plots saved to: {args.output_dir}")
    print(f"Summary results: {results_file}")

    if unproduced_outputs:
        raise OutputGenerationError(
            "The analysis finished and its results were written, but these "
            "requested outputs were not produced: "
            + "; ".join(unproduced_outputs)
            + ". See the errors above."
        )
    if args.save_full_results:
        print(f"Full detailed results: {Path(args.output_dir) / 'analysis_results_full.json'}")


def main(argv: Optional[List[str]] = None):
    """Parse CLI arguments and run tokenizer analysis."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_cli_logging()
    try:
        run_from_args(args)
    except (ParquetEngineMissing, ConfigurationError, OutputGenerationError,
            FileNotFoundError, IsADirectoryError) as e:
        # Each of these names the artifact and what to do about it, so a
        # traceback adds only noise. Every other exception keeps its traceback:
        # it is a defect in this package and the stack is the useful part.
        logger.error(str(e))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
