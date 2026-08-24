"""
Basic tokenization metrics using unified TokenizedData interface.
"""

from bisect import bisect_left
from typing import Dict, List, Any, Optional, Tuple, Mapping, Sequence
from collections import defaultdict
import logging
import time
import unicodedata

import numpy as np

from .base import BaseMetrics, TokenizedDataProcessor, format_optional
from ..core.input_types import (
    CODE_CORPUS, MATH_CORPUS, NO_TOKENIZER_ERRORS, TokenizedData,
)
from ..core.input_providers import InputProvider
from ..loaders.corpora import resolve_math_corpus
from ..config import TextMeasurementConfig, TextMeasurer, DEFAULT_TEXT_MEASUREMENT_CONFIG, DEFAULT_WORD_MEASUREMENT_CONFIG
from ..config.language_metadata import LanguageMetadata
from ..constants import (
    AGGREGATION_MEAN_OF_RATIOS,
    AGGREGATION_MICRO_POOLED,
    AGGREGATION_SET_UNION,
    DEFAULT_CER_TIME_BUDGET_S,
)
from ..utils.text_utils import load_math_data, BUILTIN_MATH_SAMPLES_PATH

logger = logging.getLogger(__name__)

# Whitespace, for whitespace_fidelity.  Definition: ASCII space/tab/newline/
# carriage-return PLUS every Unicode "space separator" (general category Zs:
# NBSP U+00A0, thin space U+2009, ideographic space U+3000, ...).  Zero-width
# *format* characters (category Cf, e.g. ZWSP U+200B) are deliberately NOT
# whitespace. Their loss is captured by exact_match_rate / CER instead, so
# each loss type lands in the semantically correct measure.  Category-based
# (not a hardcoded set) so all 17 Zs separators are covered and it is
# future-proof.  This widened the metric from the historical ASCII-only set;
# see WHITESPACE_DEFINITION echoed into reconstruction_fidelity metadata.
_ASCII_WS = frozenset(' \t\n\r')
WHITESPACE_DEFINITION = "ascii(space,tab,nl,cr)+unicode_Zs"


#: Memo for _is_ws. unicodedata.category is a C call but not a free one, and
#: reconstruction fidelity asks this question once per character of every text
#: it scores. Bounded in practice by the alphabet of the corpus.
_WS_MEMO: Dict[str, bool] = {}


def _is_ws(ch: str) -> bool:
    """True for ASCII whitespace or any Unicode space separator (Zs)."""
    known = _WS_MEMO.get(ch)
    if known is None:
        known = ch in _ASCII_WS or unicodedata.category(ch) == 'Zs'
        _WS_MEMO[ch] = known
    return known


def _ws_counts(text: str) -> Tuple[int, int, int]:
    """``(whitespace, newlines, tabs)`` in *text*.

    Counted through the distinct characters rather than position by position.
    ``_is_ws`` is a Python call and the texts here run to tens of thousands of
    characters, so asking it once per distinct character and letting
    ``str.count`` do the tallying moved this from the dominant cost of
    whitespace fidelity to a rounding error on it.
    """
    total = 0
    for ch in set(text):
        if _is_ws(ch):
            total += text.count(ch)
    return total, text.count("\n"), text.count("\t")


_CER_WARMUP = 50


class BasicTokenizationMetrics(BaseMetrics):
    """Basic tokenization metrics: fertility, token_length,
    vocabulary_utilization, reconstruction_fidelity.

    ``compute()`` also returns ``type_token_ratio`` and
    ``avg_tokens_per_line``, but neither is a top-level key in a published
    results file. ``merge_redundant_metrics`` files ``type_token_ratio`` under
    ``vocabulary_utilization`` and ``avg_tokens_per_line`` under
    ``compression_rate``, which ``InformationTheoreticMetrics`` computes. See
    metrics/redundancy.py for the identity behind each merge.
    """

    def __init__(self,
                 input_provider: InputProvider,
                 measurement_config: Optional[TextMeasurementConfig] = None,
                 language_metadata: Optional[LanguageMetadata] = None,
                 code_texts: Optional[Dict[str, List[str]]] = None,
                 math_data_path: Optional[str] = None,
                 use_builtin_math_data: bool = False,
                 fertility_use_global_config: bool = False):
        """
        Initialize basic metrics.

        Args:
            input_provider: InputProvider instance
            measurement_config: Configuration for text measurement method
            language_metadata: Optional language metadata for grouping
            code_texts: Optional pre-loaded code texts mapping languages to
                snippets. For a caller who builds this class on its own rather
                than through ``UnifiedTokenizerAnalyzer``, which registers the
                corpus on the input provider instead. Passing this when a code
                corpus is already registered raises, rather than measuring one
                and reporting it under a request for the other.
            math_data_path: Optional path to math-rich text file. Subject to
                the same conflict check as *code_texts*.
            use_builtin_math_data: Whether to use built-in math samples.
                Subject to the same conflict check as *code_texts*.
            fertility_use_global_config: If True, fertility uses *measurement_config*
                instead of the default words-based normalization.
        """
        super().__init__(input_provider)
        self.measurement_config = measurement_config or DEFAULT_TEXT_MEASUREMENT_CONFIG
        self.language_metadata = language_metadata
        if fertility_use_global_config:
            self.fertility_measurement_config = self.measurement_config
        else:
            self.fertility_measurement_config = DEFAULT_WORD_MEASUREMENT_CONFIG
        self.fertility_text_measurer = TextMeasurer(self.fertility_measurement_config)

        # Code and math data for reconstruction fidelity. The run resolves both
        # corpora once and registers them on the provider; the constructor
        # arguments are the path for a caller that builds this class directly.
        #
        # A corpus marked synthetic is the bundled samples, which nobody asked
        # to have measured. Reconstruction fidelity has never scored them: with
        # no --code-ast-config it reports no code domain at all while the AST
        # metrics run on the bundled code, and with no --math-data and no
        # --use-builtin-math-data it reports no math domain while the
        # operator-isolation math domain runs on the bundled math. Reading
        # `synthetic` here is what keeps that, and it must stay a separate test
        # from "is there a corpus at all": one rule covering both would give
        # this metric a code domain it has never had. Pinned by
        # TestTheDefaultCodeConfigurationIsAsymmetric in
        # tests/test_output_contract.py.
        #
        # The math texts are held as (label, text) pairs rather than as bare
        # texts. The label is the key a registered corpus is encoded under, and
        # reconstruction fidelity reads those encodings back by (label, text);
        # see _shared_corpus_ids. A flat list of texts would leave position as
        # the only way to pair a text with its encoding, which is the defect
        # that keying by text prevents.
        # Mapping/Sequence, not Dict[str, List[str]]: a registered Corpus hands
        # back tuples behind a read-only view, and dict() of it keeps the
        # tuples. Nothing here mutates them, and the annotation should not
        # invite it.
        self._code_texts: Mapping[str, Sequence[str]] = {}
        self._math_items: List[Tuple[str, str]] = []
        # Which corpora these texts came out of the registry, decided here and
        # not re-decided at compute time. Constructing DigitBoundaryMetrics
        # against the same provider registers corpora, so a metric built before
        # it, holding the caller's own texts, would otherwise find a corpus at
        # compute time and look its texts up in an encoding of a different one.
        # That failed loudly, but several frames from the constructor that
        # caused it.
        self._corpus_backed: Dict[str, bool] = {}

        code_corpus = self._corpus_or_refuse_arguments(
            CODE_CORPUS, {"code_texts": bool(code_texts)}
        )
        self._corpus_backed[CODE_CORPUS] = code_corpus is not None
        if code_corpus is None:
            self._code_texts = code_texts or {}
        elif not code_corpus.synthetic:
            self._code_texts = dict(code_corpus.texts)

        math_corpus = self._corpus_or_refuse_arguments(
            MATH_CORPUS,
            {"math_data_path": bool(math_data_path),
             "use_builtin_math_data": bool(use_builtin_math_data)},
        )
        self._corpus_backed[MATH_CORPUS] = math_corpus is not None
        if math_corpus is None:
            if math_data_path or use_builtin_math_data:
                # resolve_math_corpus, not load_math_data: it is the one
                # definition of which math corpus a run measures, and it
                # refuses a path that loads zero texts rather than reporting no
                # math domain and saying nothing. Reading the file here instead
                # accepted that silently, while DigitBoundaryMetrics, given the
                # same argument, aborted.
                resolved = resolve_math_corpus(
                    math_data_path, use_builtin_math_data,
                )
                self._math_items = [
                    (label, text)
                    for label, texts in resolved.texts.items()
                    for text in texts
                ]
        elif not math_corpus.synthetic:
            self._math_items = [
                (label, text)
                for label, texts in math_corpus.texts.items()
                for text in texts
            ]

    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None,
                include_reconstruction: bool = True,
                cer_time_budget_s: float = DEFAULT_CER_TIME_BUDGET_S,
                include_code_math: bool = True) -> Dict[str, Any]:
        """
        Compute basic tokenization metrics.

        Args:
            tokenized_data: Optional tokenized data dict. If None, uses input_provider data.
            include_reconstruction: Whether to include reconstruction fidelity analysis.
            cer_time_budget_s: Max seconds to spend on CER per tokenizer before
                skipping.  0 disables the budget (always compute).
            include_code_math: Whether reconstruction fidelity reports the code
                and math corpora. False for a run over one language group,
                which is a group of prose languages and contains no code or
                math. See compute_reconstruction_fidelity_analysis.

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

        # Compute reconstruction fidelity
        if include_reconstruction:
            results.update(self.compute_reconstruction_fidelity_analysis(
                tokenized_data, cer_time_budget_s=cer_time_budget_s,
                include_code_math=include_code_math))

        return results
    
    def compute_fertility_analysis(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute fertility analysis using configured normalization method.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with fertility results
        """
        normalization_unit = self.fertility_measurement_config.method.value.lower()
        
        results = {
            'fertility': {
                'per_tokenizer': {},
                'per_language': {},
                'metadata': {
                    'normalization_method': normalization_unit,
                    'description': f'Average number of tokens per {normalization_unit[:-1]}',
                    'short_description': f'tokens/{normalization_unit[:-1]}',
                    # Not micro_pooled: this is the unweighted mean of one
                    # ratio per document, so a long document counts the same
                    # as a short one. docs/METRICS.md has always described it
                    # that way; the label did not.
                    'aggregation': AGGREGATION_MEAN_OF_RATIOS,
                    'count_unit': 'documents',
                }
            }
        }
        
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
                if lang_fertility['count'] == 0 and lang_data:
                    logger.warning(
                        "Fertility: language '%s' produced 0 valid samples. "
                        "This typically occurs for languages without whitespace "
                        "word boundaries (e.g., Chinese, Japanese, Thai) when "
                        "using whitespace-based word counting.",
                        language,
                    )
                per_lang_fertility[language] = lang_fertility
            
            results['fertility']['per_tokenizer'][tok_name] = {
                'global': global_fertility,
                'per_language': per_lang_fertility
            }

        return results
    
    def _compute_fertility_stats(self, tokenized_data: List[TokenizedData], 
                                normalization_unit: str) -> Dict[str, float]:
        """Compute fertility statistics for a list of TokenizedData."""
        if not tokenized_data:
            return self.empty_stats()
        
        fertilities = []
        zero_token_documents = 0

        for data in tokenized_data:
            if not data.text or not data.text.strip():
                continue  # Skip if no text available

            num_tokens = len(data.tokens)
            if not num_tokens:
                # The tokenizer erased this document. Its fertility is a
                # defined 0, and averaging that in drags the mean toward zero
                # with nothing in the output saying a document vanished. The
                # count below is what says it.
                zero_token_documents += 1
                continue
            num_units = self.fertility_text_measurer.get_unit_count(data.text)

            if num_units > 0:
                fertility = num_tokens / num_units
                fertilities.append(fertility)

        if not fertilities:
            stats = self.empty_stats()
        else:
            stats = self.compute_basic_stats(fertilities)
        stats['zero_token_documents'] = zero_token_documents
        return stats
    
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
                    'units': ['characters', 'bytes'],
                    'description': 'Average character and byte length per token',
                    'aggregation': AGGREGATION_MEAN_OF_RATIOS,
                    # Documents, not tokens. count is len(char_lengths), one
                    # entry per document: 3250 on the committed benchmark
                    # against 109014 to 271337 actual tokens.
                    'count_unit': 'documents',
                    'global_duplicates': (
                        'global repeats primary_length. Every metric in this '
                        'file publishes a global block, so this one does too '
                        'even though the same numbers sit under primary_length '
                        'beside it.'
                    ),
                }
            }
        }

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            tok_data = tokenized_data[tok_name]

            char_lengths = []
            byte_lengths = []
            for data in tok_data:
                if data.text and data.text.strip() and data.tokens:
                    n_tokens = len(data.tokens)
                    char_lengths.append(len(data.text) / n_tokens)
                    byte_lengths.append(len(data.text.encode('utf-8')) / n_tokens)

            # 'global' is a deliberate duplicate of 'primary_length': the
            # results-file schema gives every metric a global block with no
            # exceptions, and this metric's global quantity is the primary
            # (character) length it already reports.
            if char_lengths:
                char_stats = self.compute_basic_stats(char_lengths)
                byte_stats = self.compute_basic_stats(byte_lengths)
                results['token_length']['per_tokenizer'][tok_name] = {
                    'character_length': char_stats,
                    'byte_length': byte_stats,
                    'primary_length': char_stats,
                    'global': char_stats,
                }
            else:
                empty_stats = self.empty_stats()
                results['token_length']['per_tokenizer'][tok_name] = {
                    'character_length': empty_stats,
                    'byte_length': empty_stats,
                    'primary_length': empty_stats,
                    'global': empty_stats,
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
                    'description': (
                        'Proportion of vocabulary actually used. '
                        'per_language_std and per_language_cov measure dispersion '
                        'of the per-language utilization ratio across languages.'
                    ),
                    'metric_range': '[0.0, 1.0]',
                    'std_ddof': 1,
                    'aggregation': AGGREGATION_SET_UNION,
                    'count_unit': 'vocabulary entries',
                    'per_language_dispersion': (
                        'aggregation labels the global block only. '
                        'per_language_mean, per_language_std and '
                        'per_language_cov sit beside global and are the mean, '
                        'sample standard deviation and coefficient of '
                        'variation of the per-language utilization ratio '
                        'across languages, which is a spread rather than an '
                        'average of any kind.'
                    ),
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

            per_lang_util = {}
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)

            for language, lang_data in lang_groups.items():
                used = len(TokenizedDataProcessor.get_unique_tokens(lang_data))
                per_lang_util[language] = {
                    # None for a zero-size vocabulary, matching the global
                    # sibling below, which answers the same question. The two
                    # gave different answers in the same run.
                    'utilization': self.safe_divide(used, vocab_size),
                    'used_tokens': used,
                    'vocab_size': vocab_size,
                    'unused_tokens': vocab_size - used,
                    # count is the schema-wide name for how many items of the
                    # metric's count_unit this language's entry covers. Here
                    # that unit is vocabulary entries, so it repeats
                    # used_tokens rather than measuring something else.
                    'count': used,
                }

            # Cross-language dispersion of the utilization ratio. Computed on
            # the ratio (not used_tokens) so it stays comparable across
            # tokenizers with different vocab sizes.
            # None entries are dropped rather than counted as zero: a
            # language whose utilization is undefined must not pull the mean
            # or the dispersion toward it.
            util_ratios = [
                stats['utilization'] for stats in per_lang_util.values()
                if stats['utilization'] is not None
            ]
            n = len(util_ratios)
            if n >= 2:
                util_mean = float(np.mean(util_ratios))
                util_std = float(np.std(util_ratios, ddof=1))
                util_cov = util_std / util_mean if util_mean > 0 else None
            else:
                # A dispersion over fewer than two languages is undefined, not
                # zero.  Publishing 0.0 here read as "the same utilization in
                # every language" on a single-corpus run, which the --input
                # route makes an ordinary case rather than an edge one.  This
                # matches tokenizer_fairness_gini, which already returns None
                # below MIN_LANGUAGES_FOR_GINI.
                util_mean = float(util_ratios[0]) if n == 1 else None
                util_std = None
                util_cov = None

            results['vocabulary_utilization']['per_tokenizer'][tok_name] = {
                'global_utilization': global_util['utilization'],
                'global_used_tokens': global_util['used_tokens'],
                'global_vocab_size': global_util['vocab_size'],
                'per_language': per_lang_util,
                'per_language_mean': util_mean,
                'per_language_std': util_std,
                'per_language_cov': util_cov,
            }

        return results

    def _compute_vocabulary_utilization(self, tokenized_data: List[TokenizedData], vocab_size: int) -> Dict[str, Any]:
        """Compute vocabulary utilization for a list of TokenizedData."""
        unique_tokens = TokenizedDataProcessor.get_unique_tokens(tokenized_data)
        used_tokens = len(unique_tokens)
        
        return {
            # A zero vocabulary size leaves utilization undefined rather than
            # zero. With a real vocabulary this is an ordinary division, and an
            # empty corpus gives a true 0 of N.
            'utilization': self.safe_divide(used_tokens, vocab_size),
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
            # None, not 0.0, when no token was emitted. A type-token ratio of
            # 0.0 cannot be measured (it would need types to be zero while
            # tokens is not), so it can only mean "nothing was tokenized", and
            # the 'tokens' field beside it says that plainly.
            'ttr': self.safe_divide(unique_count, total_tokens),
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
                if data.text and data.text.strip() and data.tokens:
                    # str.splitlines(), so blank lines count and a single
                    # trailing newline does not add a phantom empty line. The
                    # numerator is the token count of the whole text, and a
                    # blank line still contributes its newline token, so
                    # dropping blank lines from the denominator alone made the
                    # ratio larger than the text supports: 'a\n\nb\n\n' with 4
                    # tokens reported 2.0 against the 1.0 its four lines give.
                    num_lines = len(data.text.splitlines())
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
                # Nothing measured, so null rather than a zero that reads as a
                # measured rate of zero tokens per line.
                results['avg_tokens_per_line']['per_tokenizer'][tok_name] = {
                    'global_avg': None,
                    'global_std': None,
                    'global_std_err': None,
                    'total_lines': 0,
                    'stats': self.empty_stats()
                }

        return results

    # ------------------------------------------------------------------
    # Reconstruction Fidelity
    # ------------------------------------------------------------------

    def _shared_corpus_ids(
        self,
    ) -> Dict[str, Dict[str, Dict[Tuple[str, str], List[int]]]]:
        """The registered code and math corpora already encoded, ids indexed by (label, text).

        Shape: corpus name -> tokenizer name -> (label, text) -> token ids. A
        corpus is absent from the result when the run registered none, and the
        code/math loop in :meth:`compute_reconstruction_fidelity_analysis` then
        encodes each text itself as it always has. That fallback is what keeps
        this class constructible on its own: per_example.py, scripts/ and most
        of the test suite build it against a provider carrying no corpora.

        A corpus this metric measures no text from is absent as well. That is
        the synthetic corpus, which __init__ leaves out of _code_texts and
        _math_items, and reading it here would encode a corpus nothing goes on
        to score.

        The index is keyed by ``(label, text)`` and never by position. The
        provider's encode keeps only texts satisfying ``text and text.strip()``,
        so its list for a label is shorter than this metric's whenever one of
        those texts is empty or whitespace-only. Pairing the two lists by index
        after one such drop would score every later text against the following
        text's ids, and nothing in the results would show it. The one cause of
        such a text this package used to produce, ``max_snippet_chars``
        truncating an indented file down to its leading whitespace, is fixed
        where it arose, in ``CodeDataLoader._load_language``. A corpus supplied
        by a caller can still contain one, so the key stays the text. Two
        identical texts collapse onto one entry, which is harmless: identical
        text encodes to identical ids.

        The provider encodes with offsets, which this metric does not read. In a
        run where another metric also reads the corpus (the AST metrics and
        operator isolation both do, and both need the offsets) that costs
        nothing, and it removes the second encoding of the same texts this loop
        used to make. In a run where reconstruction fidelity is the only reader,
        the offsets are paid for and thrown away. That is accepted rather than
        avoided with an ids-only provider method: such a method would either
        encode the corpus a second time or need a second cache, and the
        configuration it would serve is narrow (a real code or math corpus with
        both the AST metrics and the digit metrics turned off). Its cost there
        is bounded. For HuggingFaceTokenizer and CustomBPETokenizer the offsets
        come free and the provider encodes one batch per label: 4.00 s against
        10.29 s for per-text calls over the 1500 files
        benchmarks/open_source/code_ast_config.json names. Both of those figures
        are encode_with_offsets. The per-text encode() this loop replaced
        computes no offsets and was not timed, so they do not establish that the
        shared path is faster for a run where this metric is the only reader. UniMixLMTokenizer
        subclasses HuggingFaceTokenizer but overrides the batch method back to a
        per-text loop, because langspec encoding scores each text against every
        per-language tokenizer, so it is not one of them. The script_bpe
        wrappers compute offsets by replaying the pretokenizer, at 2.16x the
        cost of the ids alone, and have no batch path, so they are the ones that
        pay.
        """
        lookup: Dict[str, Dict[str, Dict[Tuple[str, str], List[int]]]] = {}
        measured = (
            (CODE_CORPUS, bool(self._code_texts)),
            (MATH_CORPUS, bool(self._math_items)),
        )
        for corpus_name, has_texts in measured:
            # self._corpus_backed, not the registry: what this metric measures
            # was decided in __init__, and a corpus registered after that by
            # another metric's constructor does not change it.
            if not has_texts or not self._corpus_backed.get(corpus_name):
                continue
            encoded = self.input_provider.get_corpus_data(corpus_name)
            per_corpus: Dict[str, Dict[Tuple[str, str], List[int]]] = {}
            for tok_name, records in encoded.items():
                per_tokenizer: Dict[Tuple[str, str], List[int]] = {}
                for record in records:
                    # A record carrying no text cannot be matched to a text of
                    # this metric's. InputProvider._encode_corpus always sets
                    # it; a provider supplying its own records for this corpus
                    # need not. Such a record is left out of the index, and the
                    # text it was meant for then raises from the loop below.
                    if record.text is None:
                        continue
                    per_tokenizer[(record.language, record.text)] = record.tokens
                per_corpus[tok_name] = per_tokenizer
            lookup[corpus_name] = per_corpus
        return lookup

    def compute_reconstruction_fidelity_analysis(
        self, tokenized_data: Dict[str, List[TokenizedData]],
        cer_time_budget_s: float = DEFAULT_CER_TIME_BUDGET_S,
        include_code_math: bool = True,
    ) -> Dict[str, Any]:
        """Compute encode-decode round-trip fidelity metrics.

        For each tokenizer that supports decoding, collects text from
        language data, code data, and math data, then measures:
        - Exact match rate
        - Character error rate (CER)
        - UNK token rate
        - Whitespace fidelity

        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            cer_time_budget_s: Max seconds to spend on CER per tokenizer.
                The total CER time is extrapolated on two triggers: after
                ``_CER_WARMUP`` non-exact texts, and on every non-exact text
                once the CER time already spent exceeds this budget, whichever
                comes first. The second trigger exists for a tokenizer whose
                individual CER calls are slow enough that the warmup itself
                overruns the budget. If the projection exceeds this
                budget the CER and whitespace-fidelity computations are skipped
                for the rest of the tokenizer and reported as ``None``.
                Set to ``0`` to disable the budget (always compute).
            include_code_math: Whether to report the code and math corpora
                alongside the prose languages. False for a run over one
                language group: the group selects prose languages, the code and
                math corpora belong to no language, and reporting the whole of
                both inside every group made each group's ``global`` a figure
                measured mostly on texts the group does not contain. It also
                put the same code and math texts into every group's CER budget.
        """
        results: Dict[str, Any] = {
            'reconstruction_fidelity': {
                'per_tokenizer': {},
                'summary': {},
                'metadata': {
                    'description': 'Encode->decode reconstruction quality: '
                                   'exact_match_rate, CER, unk_token_rate, '
                                   'whitespace_fidelity.',
                    # Self-describing so any result file is traceable: this
                    # metric was widened from ASCII-only to also count Unicode
                    # Zs separators (NBSP/thin/ideographic/...).  Pre-change
                    # whitespace_fidelity numbers are NOT comparable.
                    'whitespace_definition': WHITESPACE_DEFINITION,
                    'aggregation': AGGREGATION_MICRO_POOLED,
                    # One label cannot describe this block. Six of the seven
                    # rates divide summed counts, which is micro_pooled;
                    # mean_cer is the mean of one ratio per document, because
                    # _character_error_rate normalises by each reference's own
                    # length before the sum. Named rather than averaged over.
                    'aggregation_exceptions': {
                        'mean_cer': AGGREGATION_MEAN_OF_RATIOS,
                    },
                    'count_unit': 'documents',
                    # Conditional, because a grouped run reports prose only.
                    # A fixed string here described code and math domains that
                    # every grouped block is now guaranteed not to contain, so a
                    # consumer reading the metadata to decide which keys to
                    # expect got a contract the output did not meet.
                    'per_domain': (
                        'This metric breaks down by domain rather than by '
                        'language: per_domain holds one entry per language of '
                        'the prose corpus, one per programming language and '
                        'one for math. Each entry carries count in documents, '
                        'so it is the per_language block under another name.'
                        if include_code_math else
                        'This metric breaks down by domain rather than by '
                        'language: per_domain holds one entry per language of '
                        'the prose corpus. This run reports no code or math '
                        'domain, because it covers one group of prose '
                        'languages and the code and math corpora belong to no '
                        'language. Each entry carries count in documents, so '
                        'it is the per_language block under another name.'
                    ),
                },
            }
        }

        # The code and math texts this call reports. Both are empty for a
        # grouped run: see include_code_math above. Read from here on rather
        # than from self, so the two corpora leave the calculation together,
        # including the CER budget below.
        code_texts = self._code_texts if include_code_math else {}
        math_items = self._math_items if include_code_math else []

        # The ids the provider already made for the code and math corpora,
        # empty when the run registered neither.
        shared_ids = self._shared_corpus_ids() if include_code_math else {}

        # Every code and math text, counted before the `text.strip()` filter the
        # loop below applies. This over-counts a whitespace-only text, and that
        # is deliberate: the count feeds the estimate of how much CER work is
        # left, which decides whether cer_skipped flips, so filtering it here
        # would move published values. It does not depend on the tokenizer, so
        # it is counted once for the whole loop.
        total_code_math_texts = (
            sum(len(snippets) for snippets in code_texts.values())
            + len(math_items)
        )

        for tok_name in self.tokenizer_names:
            try:
                tokenizer = self.input_provider.get_tokenizer(tok_name)
            except NO_TOKENIZER_ERRORS + (AttributeError,) as e:
                # The tuple InputProvider._encode_corpus skips on, so the two
                # agree on what "this provider cannot supply a tokenizer"
                # means, plus AttributeError, which _encode_corpus deliberately
                # excludes. It belongs here and not there: this reads
                # self.input_provider, which need not be an InputProvider
                # subclass and so need not have the method at all, while
                # _encode_corpus calls its own inherited get_tokenizer, where
                # the attribute always resolves. This was a bare
                # ``except Exception``, which also skipped a tokenizer whose
                # loader had a genuine defect and reported the run as a success
                # with that tokenizer missing.
                #
                # Reachable only when no corpus is registered. With one,
                # _shared_corpus_ids above has already called get_corpus_data,
                # and _encode_corpus does not catch AttributeError, so it
                # propagates out of the analysis before this loop starts.
                logger.warning(
                    "Reconstruction fidelity: skipping %s (could not load tokenizer: %s)",
                    tok_name, e,
                )
                continue

            if not tokenizer.can_decode():
                logger.info("Reconstruction fidelity: skipping %s (no decode support)", tok_name)
                continue

            # Whether this tokenizer gets the code and math domains. Its prose
            # domains are computed either way, from the ids already in the
            # TokenizedData it was handed, which need no encoder.
            tokenizer_code_math = bool(total_code_math_texts)
            if total_code_math_texts:
                # This metric selects on can_decode() alone, and the code/math
                # loop below then needs raw text encoded. A tokenizer that
                # decodes but cannot encode reaches that call and raises
                # NotImplementedError out of the whole analysis. No wrapper in
                # this package is both: PreTokenizedDataTokenizer reports
                # can_decode() false and the check above already skipped it, so
                # this is reachable through a caller's own tokenizer object
                # rather than through anything shipped here.
                # Decided 2026-08-19: skip rather than crash the run, which is
                # what InputProvider._encode_corpus already does with the same
                # tokenizer. Only the code and math domains are skipped: this
                # used to `continue`, which dropped the tokenizer from the
                # results entirely and took its prose numbers with it, though
                # those come from ids that are already in hand.
                if not InputProvider.can_encode_raw_text(tokenizer):
                    logger.warning(
                        "Reconstruction fidelity: %s decodes but cannot encode "
                        "raw text, so it gets no code or math domain from the "
                        "%d code/math texts in this run. Its prose domains are "
                        "reported as usual. Pre-tokenized input supplies ids, "
                        "not an encoder.",
                        tok_name, total_code_math_texts,
                    )
                    tokenizer_code_math = False
                missing = sorted(
                    corpus_name for corpus_name, per_corpus in shared_ids.items()
                    if tok_name not in per_corpus
                ) if tokenizer_code_math else []
                if missing:
                    raise ValueError(
                        f"Tokenizer {tok_name!r} passes "
                        "InputProvider.can_encode_raw_text, which is the same "
                        "predicate the provider selects on, but the shared "
                        f"{', '.join(map(repr, missing))} corpus holds no "
                        "encoding for it. Encoding it here instead would undo "
                        "the single encoding this path exists for, and skipping "
                        "it would drop it from the reconstruction results with "
                        "nothing in the output saying it is absent."
                    )

            unk_id = tokenizer.get_unk_token_id()

            # Per-domain accumulators
            domain_stats: Dict[str, Dict[str, Any]] = defaultdict(
                lambda: {
                    'exact_matches': 0, 'total': 0,
                    'cer_sum': 0.0,
                    'unk_tokens': 0, 'total_tokens': 0,
                    'ws_preserved': 0, 'ws_total': 0,
                    'ind_preserved': 0, 'ind_total': 0,
                    'nl_preserved': 0, 'nl_total': 0,
                    'tab_preserved': 0, 'tab_total': 0,
                    'decode_failures': 0,
                }
            )

            has_data = False

            # CER time-budget state
            cer_elapsed = 0.0
            cer_skipped = False
            n_cer_calls = 0
            # Count total texts for extrapolation
            total_lang_texts = 0
            if tok_name in tokenized_data:
                total_lang_texts = sum(
                    1 for td in tokenized_data[tok_name]
                    if td.text and td.text.strip()
                )
            # tokenizer_code_math, not the raw total: a tokenizer that gets no
            # code or math domain never processes those texts, and counting
            # them here left the projection of remaining CER work inflated by
            # the whole corpus, which flips cer_skipped and publishes null for
            # a tokenizer that had the budget to compute them.
            total_all_texts = total_lang_texts + (
                total_code_math_texts if tokenizer_code_math else 0
            )
            texts_processed = 0

            # Language data: reuse tokens/offsets already stored in TokenizedData
            if tok_name in tokenized_data:
                for td in tokenized_data[tok_name]:
                    if not td.text or not td.text.strip():
                        continue
                    has_data = True
                    text = td.text
                    texts_processed += 1

                    token_ids = td.tokens
                    stats = domain_stats[td.language]
                    stats['total_tokens'] += len(token_ids)

                    if unk_id is not None:
                        stats['unk_tokens'] += token_ids.count(unk_id)

                    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                    if decoded is None:
                        self._record_decode_failure(stats, tok_name, td.language, text)
                        continue

                    stats['total'] += 1

                    if decoded == text:
                        stats['exact_matches'] += 1
                        if not cer_skipped:
                            self._accumulate_whitespace(stats, text, decoded, True)
                    else:
                        if not cer_skipped:
                            # Both inside the timed region. --cer-time-budget
                            # exists to bound the cost of scoring a lossy
                            # decode, and the whitespace alignment is the same
                            # order as the edit distance beside it. cer_elapsed
                            # used to close before the whitespace call, so the
                            # budget could neither see that cost nor stop it.
                            t0 = time.monotonic()
                            stats['cer_sum'] += self._character_error_rate(text, decoded)
                            self._accumulate_whitespace(stats, text, decoded, False)
                            cer_elapsed += time.monotonic() - t0
                            n_cer_calls += 1

                            # Check the budget after the warmup, or as soon as
                            # the warmup has itself spent it. The call count
                            # alone is not enough: one CER call is an edit
                            # distance over two long strings, so a lossy
                            # tokenizer can spend minutes inside a single call
                            # and never reach the 50th. Measured on
                            # bert-base-uncased over 5035 texts, the warmup ran
                            # past 10 minutes with the budget set to 120s.
                            if (cer_time_budget_s > 0
                                    and (n_cer_calls == _CER_WARMUP
                                         or cer_elapsed > cer_time_budget_s)):
                                # Estimate remaining non-exact texts
                                exact_so_far = sum(
                                    ds['exact_matches']
                                    for ds in domain_stats.values()
                                )
                                total_so_far = sum(
                                    ds['total']
                                    for ds in domain_stats.values()
                                )
                                exact_rate = (exact_so_far / total_so_far
                                              if total_so_far else 0.0)
                                remaining_texts = total_all_texts - texts_processed
                                est_remaining_nonexact = (
                                    (1 - exact_rate) * remaining_texts
                                )
                                time_per_cer = cer_elapsed / n_cer_calls
                                projected = (
                                    cer_elapsed
                                    + time_per_cer * est_remaining_nonexact
                                )
                                if projected > cer_time_budget_s:
                                    cer_skipped = True
                                    logger.warning(
                                        "CER time budget exceeded for %s: "
                                        "%.1fs projected (budget %.1fs). "
                                        "Skipping CER and whitespace fidelity "
                                        "for remaining texts.",
                                        tok_name, projected,
                                        cer_time_budget_s,
                                    )

            # Code/math data: not in the prose TokenizedData. Each text
            # carries the corpus and the label it is filed under, which is how
            # its ids are found in shared_ids, and the domain it is reported
            # under, which is a different string: "code_python" for the python
            # label of the code corpus.
            #
            # The `text.strip()` filter is the same one
            # InputProvider._encode_corpus applies, so every text kept here has
            # an entry in shared_ids and a text it drops has none.
            code_math_pairs: List[Tuple[str, str, str, str]] = []
            # Empty when this tokenizer cannot encode raw text; it still gets
            # its prose domains from the ids it was handed.
            for lang, snippets in (code_texts if tokenizer_code_math else {}).items():
                domain = f"code_{lang}"
                for snippet in snippets:
                    if snippet and snippet.strip():
                        code_math_pairs.append((snippet, domain, CODE_CORPUS, lang))
            for label, text in (math_items if tokenizer_code_math else []):
                if text and text.strip():
                    code_math_pairs.append((text, "math", MATH_CORPUS, label))

            for text, domain, corpus_name, label in code_math_pairs:
                has_data = True
                texts_processed += 1
                stats = domain_stats[domain]

                per_tokenizer = shared_ids.get(corpus_name)
                if per_tokenizer is None:
                    # Nothing registered this corpus, so there is no shared
                    # encoding to read and this loop encodes as it always has.
                    #
                    # encode(), not encode_with_offsets(): this loop reads the
                    # ids and throws the offsets away. Both return the same ids,
                    # which TestEncodeConsistency asserts for every wrapper. For
                    # a HuggingFace tokenizer the offsets come free, but the
                    # script_bpe wrappers compute them by replaying the
                    # pretokenizer, measured at 2.16x the cost of encoding on
                    # this corpus shape, so this loop was paying that for
                    # nothing.
                    token_ids = tokenizer.encode(text)
                else:
                    token_ids = per_tokenizer[tok_name].get((label, text))
                    if token_ids is None:
                        raise ValueError(
                            f"The shared {corpus_name!r} corpus holds no "
                            f"encoding of a {label!r} text for tokenizer "
                            f"{tok_name!r}. The text this metric measures and "
                            "the texts the provider encoded are supposed to be "
                            "the same strings; they are not. Scoring this text "
                            "against another text's ids is what keying the "
                            "lookup by text instead of by position prevents, so "
                            "it fails here rather than reporting a number "
                            "measured against the wrong source. Text starts: "
                            f"{text[:60]!r}"
                        )
                stats['total_tokens'] += len(token_ids)

                if unk_id is not None:
                    stats['unk_tokens'] += token_ids.count(unk_id)

                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                if decoded is None:
                    self._record_decode_failure(stats, tok_name, domain, text)
                    continue

                stats['total'] += 1

                if decoded == text:
                    stats['exact_matches'] += 1
                    if not cer_skipped:
                        self._accumulate_whitespace(stats, text, decoded, True)
                else:
                    if not cer_skipped:
                        t0 = time.monotonic()
                        stats['cer_sum'] += self._character_error_rate(text, decoded)
                        self._accumulate_whitespace(stats, text, decoded, False)
                        cer_elapsed += time.monotonic() - t0
                        n_cer_calls += 1

                        # Same two-sided check as the language loop above.
                        if (cer_time_budget_s > 0
                                and not cer_skipped
                                and (n_cer_calls == _CER_WARMUP
                                     or cer_elapsed > cer_time_budget_s)):
                            exact_so_far = sum(
                                ds['exact_matches']
                                for ds in domain_stats.values()
                            )
                            total_so_far = sum(
                                ds['total']
                                for ds in domain_stats.values()
                            )
                            exact_rate = (exact_so_far / total_so_far
                                          if total_so_far else 0.0)
                            remaining_texts = total_all_texts - texts_processed
                            est_remaining_nonexact = (
                                (1 - exact_rate) * remaining_texts
                            )
                            time_per_cer = cer_elapsed / n_cer_calls
                            projected = (
                                cer_elapsed
                                + time_per_cer * est_remaining_nonexact
                            )
                            if projected > cer_time_budget_s:
                                cer_skipped = True
                                logger.warning(
                                    "CER time budget exceeded for %s: "
                                    "%.1fs projected (budget %.1fs). "
                                    "Skipping CER and whitespace fidelity "
                                    "for remaining texts.",
                                    tok_name, projected,
                                    cer_time_budget_s,
                                )

            if not has_data:
                continue

            # Log after main loop using accumulated counts
            n_lang = sum(
                ds['total'] for dom, ds in domain_stats.items()
                if not dom.startswith('code_') and dom != 'math'
            )
            n_code_math = sum(
                ds['total'] for dom, ds in domain_stats.items()
                if dom.startswith('code_') or dom == 'math'
            )
            logger.info(
                "Reconstruction fidelity: decoding %d texts for %s (%d language, %d code/math)",
                n_lang + n_code_math, tok_name, n_lang, n_code_math,
            )

            # Aggregate per-domain and overall
            per_domain: Dict[str, Dict[str, Any]] = {}
            overall = {
                'exact_matches': 0, 'total': 0,
                'cer_sum': 0.0,
                'unk_tokens': 0, 'total_tokens': 0,
                'ws_preserved': 0, 'ws_total': 0,
                'ind_preserved': 0, 'ind_total': 0,
                'nl_preserved': 0, 'nl_total': 0,
                'tab_preserved': 0, 'tab_total': 0,
                'decode_failures': 0,
            }

            for domain, ds in domain_stats.items():
                count = ds['total']
                # No stand-in defaults. docs/OUTPUT.md's contract is that a
                # value which could not be computed is null, and each of these
                # four had a different zero-denominator case that read as a
                # measurement: exact_match_rate and mean_cer 0.0 on a domain
                # where every decode failed, unk_token_rate 0.0 for a tokenizer
                # that declares no UNK id, whitespace_fidelity 1.0 for a text
                # holding no whitespace at all. A mean_cer of 0.0 beside a
                # count of 0 read as a perfect round trip.
                per_domain[domain] = {
                    'exact_match_rate': self.safe_divide(ds['exact_matches'], count),
                    'mean_cer': None if cer_skipped else self.safe_divide(ds['cer_sum'], count),
                    'unk_token_rate': self.safe_divide(ds['unk_tokens'], ds['total_tokens']),
                    'whitespace_fidelity': None if cer_skipped else self.safe_divide(ds['ws_preserved'], ds['ws_total']),
                    'indentation_fidelity': None if cer_skipped else self.safe_divide(ds['ind_preserved'], ds['ind_total']),
                    'newline_fidelity': None if cer_skipped else self.safe_divide(ds['nl_preserved'], ds['nl_total']),
                    'tab_fidelity': None if cer_skipped else self.safe_divide(ds['tab_preserved'], ds['tab_total']),
                    'count': count,
                    'decode_failures': ds['decode_failures'],
                    'total_tokens': ds['total_tokens'],
                }
                for k in overall:
                    overall[k] += ds[k]

            total = overall['total']
            tok_result = {
                'by_domain': per_domain,
                'overall': {
                    'exact_match_rate': self.safe_divide(overall['exact_matches'], total),
                    'mean_cer': None if cer_skipped else self.safe_divide(overall['cer_sum'], total),
                    'unk_token_rate': self.safe_divide(overall['unk_tokens'], overall['total_tokens']),
                    'whitespace_fidelity': None if cer_skipped else self.safe_divide(overall['ws_preserved'], overall['ws_total']),
                    'indentation_fidelity': None if cer_skipped else self.safe_divide(overall['ind_preserved'], overall['ind_total']),
                    'newline_fidelity': None if cer_skipped else self.safe_divide(overall['nl_preserved'], overall['nl_total']),
                    'tab_fidelity': None if cer_skipped else self.safe_divide(overall['tab_preserved'], overall['tab_total']),
                    'count': total,
                    'decode_failures': overall['decode_failures'],
                    'total_tokens': overall['total_tokens'],
                },
            }
            if cer_skipped:
                tok_result['cer_skipped'] = True
            results['reconstruction_fidelity']['per_tokenizer'][tok_name] = tok_result
            results['reconstruction_fidelity']['summary'][tok_name] = {
                'exact_match_rate': tok_result['overall']['exact_match_rate'],
                'mean_cer': tok_result['overall']['mean_cer'],
                'unk_token_rate': tok_result['overall']['unk_token_rate'],
                'whitespace_fidelity': tok_result['overall']['whitespace_fidelity'],
                'indentation_fidelity': tok_result['overall']['indentation_fidelity'],
                'newline_fidelity': tok_result['overall']['newline_fidelity'],
                'tab_fidelity': tok_result['overall']['tab_fidelity'],
                'texts_analyzed': total,
                # Here as well as in overall: latex_tables.py and main.py read
                # summary alone, and would otherwise render a rate over
                # survivors with nothing saying texts were dropped.
                'decode_failures': overall['decode_failures'],
                'total_tokens_analyzed': overall['total_tokens'],
            }
            if cer_skipped:
                results['reconstruction_fidelity']['summary'][tok_name]['cer_skipped'] = True
            cer_msg = ("SKIPPED" if cer_skipped
                       else format_optional(tok_result['overall']['mean_cer'], '.4f'))
            logger.info(
                "Reconstruction fidelity: %s done, %d texts decoded, "
                "exact_match=%.3f, mean_cer=%s",
                tok_name, total,
                tok_result['overall']['exact_match_rate'],
                cer_msg,
            )

        return results

    @staticmethod
    def _character_error_rate(reference: str, hypothesis: str) -> float:
        """Levenshtein edit distance between *reference* and *hypothesis*,
        normalized by the length of *reference*.

        CER = levenshtein(reference, hypothesis) / len(reference)

        Returns 0.0 when the strings are identical or the reference is empty.
        CER can exceed 1.0 when the hypothesis is longer than the reference.
        Uses a two-row DP approach: O(n*m) time, O(min(n,m)) space.

        Optimized with common prefix/suffix stripping to reduce the DP
        matrix to the differing region only.
        """
        if reference == hypothesis:
            return 0.0
        ref_len = len(reference)
        if ref_len == 0:
            return 0.0
        if not hypothesis:
            return 1.0

        # Strip common prefix
        prefix = 0
        min_len = min(ref_len, len(hypothesis))
        while prefix < min_len and reference[prefix] == hypothesis[prefix]:
            prefix += 1

        # Strip common suffix (don't overlap with prefix)
        r_end = ref_len
        h_end = len(hypothesis)
        while r_end > prefix and h_end > prefix and reference[r_end - 1] == hypothesis[h_end - 1]:
            r_end -= 1
            h_end -= 1

        # Work on the differing region only
        a = reference[prefix:r_end]
        b = hypothesis[prefix:h_end]
        n = len(a)
        m = len(b)

        # If one side is empty after stripping, distance = length of the other
        if n == 0:
            return m / ref_len
        if m == 0:
            return n / ref_len

        # Swap so the DP row is the shorter dimension (space optimization).
        # Levenshtein distance is symmetric so the result is unchanged.
        rows, cols = n, m
        if cols > rows:
            a, b = b, a
            rows, cols = cols, rows

        prev = list(range(cols + 1))
        curr = [0] * (cols + 1)

        for i in range(1, rows + 1):
            curr[0] = i
            ai = a[i - 1]
            prev_im1 = prev[0]
            for j in range(1, cols + 1):
                pj = prev[j]
                sub = prev_im1 if ai == b[j - 1] else prev_im1 + 1
                ins = curr[j - 1] + 1
                dele = pj + 1
                curr[j] = sub if sub <= ins and sub <= dele else (ins if ins <= dele else dele)
                prev_im1 = pj
            prev, curr = curr, prev

        return prev[cols] / ref_len

    @staticmethod
    def _record_decode_failure(stats, tok_name: str, domain: str, text: str) -> None:
        """Count and name a text whose decode returned None.

        One definition for both reconstruction loops. The text leaves every
        reconstruction denominator, so without the count a domain with one
        failure in three published exact_match_rate 1.0 with nothing saying it
        was the rate over the two that decoded.

        Its tokens stay in total_tokens and unk_tokens deliberately.
        total_tokens is the domain's token count and count is the texts that
        decoded; making the first conditional on decoding would turn
        unk_token_rate into a rate over survivors and break the
        docs/OUTPUT.md correspondence with summary.total_tokens_analyzed.

        The wrapper logs its own exception, but names neither the domain nor
        the text, so this adds the line a reader of the run log needs.
        """
        stats['decode_failures'] += 1
        logger.warning(
            "%s failed to decode a %s text; it is excluded from the "
            "reconstruction rates and counted in decode_failures. "
            "Text starts: %r",
            tok_name, domain, text[:60],
        )

    def _accumulate_whitespace(self, stats, text: str, decoded: str, exact: bool) -> None:
        """Add one text's whitespace outcome to a domain's accumulators.

        One definition for both reconstruction loops. They have diverged
        before, and there are four families to keep in step: overall
        whitespace, indentation, newlines and tabs.

        The three sub-rates exist because the roll-up cannot answer the
        question this metric is for, which is whether preprocessing damaged
        code and tabular structure. Indentation is 42% of the whitespace in the
        bundled code corpus and 0% of FLORES prose, where 95% is inter-word
        spaces that carry no structure, so destroying every indent in the code
        corpus moves whitespace_fidelity only to 0.578 while collapsing inner
        spaces harmlessly leaves it at 0.980. A Makefile tab replaced by four
        spaces scores 0.80 and trailing whitespace stripped scores 0.50: the
        benign change looks worse than the fatal one.

        On an exact round trip nothing needs aligning: every whitespace
        character and every indent survived by definition.
        """
        if exact:
            ws, newlines, tabs = _ws_counts(text)
            _, indents = self._indentation_fidelity(text, text)
            stats['ws_preserved'] += ws
            stats['ws_total'] += ws
            stats['nl_preserved'] += newlines
            stats['nl_total'] += newlines
            stats['tab_preserved'] += tabs
            stats['tab_total'] += tabs
            stats['ind_preserved'] += indents
            stats['ind_total'] += indents
            return

        matched, nl_matched, tab_matched, ws_total = self._whitespace_matches(text, decoded)
        ind_matched, ind_total = self._indentation_fidelity(text, decoded)
        stats['ws_preserved'] += matched
        stats['ws_total'] += ws_total
        stats['nl_preserved'] += nl_matched
        stats['nl_total'] += text.count("\n")
        stats['tab_preserved'] += tab_matched
        stats['tab_total'] += text.count("\t")
        stats['ind_preserved'] += ind_matched
        stats['ind_total'] += ind_total

    @staticmethod
    def _whitespace_matches(original: str, decoded: str) -> Tuple[int, int, int, int]:
        """Whitespace of *original* that survives the round trip, by kind.

        Returns ``(matched, matched_newlines, matched_tabs, total)``, all
        counted over *original*. ``matched`` includes the newline and tab
        counts, so a caller wanting spaces alone subtracts them.

        A whitespace character counts as preserved when it is matched under a
        maximum-length common subsequence of the two full texts, choosing among
        those of maximum length the one matching the most whitespace. Two
        cheaper rules were measured and rejected. A greedy forward scan, which
        this used to be, left its pointer behind on any character it could not
        match, so every later whitespace was compared at the wrong index and
        intact whitespace counted as lost: "El nino esta aqui" decoded from
        "El nino esta aqui" with the accents dropped reported 2 of 3. Aligning
        the two whitespace streams alone is context-blind and credits
        whitespace that was destroyed: "hello world" decoded as "helloworld "
        scores 1 of 1 for a deleted word boundary against a space appended
        somewhere else.

        The common prefix and suffix are matched by construction and their
        whitespace credited without entering the table, which is what keeps
        this affordable. An untrimmed table over a 39.6k-character text with
        one substituted character ran 7.9 seconds against 3.4 milliseconds for
        the character error rate it shares a time budget with. Trimming was
        verified to change no result, over 974,974 pairs against an oracle that
        enumerates subsequences rather than restating this table.

        The table carries the newline and tab tallies of its own optimum
        instead of being walked back afterward, so the whole computation stays
        in two rows. A traceback would need the full table, which on a lossy
        decode of a large file is the size of the text squared.
        """
        total, total_nl, total_tab = _ws_counts(original)
        if total == 0:
            return (0, 0, 0, 0)
        if original == decoded:
            return (total, total_nl, total_tab, total)

        lim = min(len(original), len(decoded))
        pre = 0
        while pre < lim and original[pre] == decoded[pre]:
            pre += 1
        suf = 0
        while (suf < lim - pre
               and original[len(original) - 1 - suf] == decoded[len(decoded) - 1 - suf]):
            suf += 1
        edges = original[:pre] + original[len(original) - suf:]
        base_ws, base_nl, base_tab = _ws_counts(edges)

        a = original[pre:len(original) - suf]
        b = decoded[pre:len(decoded) - suf]
        n, m = len(a), len(b)
        # Cell: (length, whitespace, newlines, tabs). Ordered on the first two
        # only; the last two ride along so the split that is reported belongs
        # to the alignment that was actually chosen.
        prev = [(0, 0, 0, 0)] * (m + 1)
        for i in range(n - 1, -1, -1):
            cur = [(0, 0, 0, 0)] * (m + 1)
            ai = a[i]
            ai_ws = 1 if _is_ws(ai) else 0
            ai_nl = 1 if ai == "\n" else 0
            ai_tab = 1 if ai == "\t" else 0
            for k in range(m - 1, -1, -1):
                if ai == b[k]:
                    L, W, N, T = prev[k + 1]
                    cur[k] = (L + 1, W + ai_ws, N + ai_nl, T + ai_tab)
                else:
                    down = prev[k]
                    right = cur[k + 1]
                    cur[k] = down if (down[0], down[1]) >= (right[0], right[1]) else right
            prev = cur

        _, ws, nl, tab = prev[0]
        return (ws + base_ws, nl + base_nl, tab + base_tab, total)

    @classmethod
    def _whitespace_fidelity(
        cls,
        original: str,
        decoded: str,
    ) -> Tuple[int, int]:
        """``(num_preserved, num_total_ws)`` in *original*.

        The pair that the reconstruction loops and
        ``diagnostics/sanity_check.py`` accumulate. See
        ``_whitespace_matches`` for the rule.
        """
        matched, _nl, _tab, total = cls._whitespace_matches(original, decoded)
        return (matched, total)

    @staticmethod
    def _indentation_fidelity(original: str, decoded: str) -> Tuple[int, int]:
        """Lines of *original* whose leading whitespace survives, and how many have any.

        Run-exact, deliberately. Indentation depth is a property of the whole
        run, so a four-space indent arriving as three is broken code rather
        than three quarters preserved, and the character count that
        ``whitespace_fidelity`` reports gives it 0.75. Each line's leading
        whitespace is compared as one string.

        The indents are matched as a sequence rather than line against line, so
        a dropped or inserted line shifts nothing after it. Blank lines carry
        no indentation to preserve and are skipped on both sides.

        Returns ``(num_preserved, num_lines_with_indentation)``.
        """
        def indents(text: str) -> List[str]:
            out: List[str] = []
            for line in text.split("\n"):
                if not line.strip():
                    continue
                lead = line[:len(line) - len(line.lstrip())]
                if lead:
                    out.append(lead)
            return out

        a, b = indents(original), indents(decoded)
        total = len(a)
        if total == 0:
            return (0, 0)
        # Same trim as the character alignment, for the same reason: an
        # unchanged or lightly changed file has every indent in the common
        # prefix, which turns a table over the whole file into no table at all.
        lim = min(len(a), len(b))
        pre = 0
        while pre < lim and a[pre] == b[pre]:
            pre += 1
        suf = 0
        while suf < lim - pre and a[len(a) - 1 - suf] == b[len(b) - 1 - suf]:
            suf += 1
        credited = pre + suf
        a, b = a[pre:len(a) - suf], b[pre:len(b) - suf]
        n, m = len(a), len(b)
        prev = [0] * (m + 1)
        for i in range(n - 1, -1, -1):
            cur = [0] * (m + 1)
            for k in range(m - 1, -1, -1):
                cur[k] = prev[k + 1] + 1 if a[i] == b[k] else max(prev[k], cur[k + 1])
            prev = cur
        return (prev[0] + credited, total)
