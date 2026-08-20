"""
Basic tokenization metrics using unified TokenizedData interface.
"""

from bisect import bisect_left
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
import time
import unicodedata

import numpy as np

from .base import BaseMetrics, TokenizedDataProcessor, format_optional
from ..core.input_types import CODE_CORPUS, MATH_CORPUS, TokenizedData
from ..core.input_providers import InputProvider
from ..config import TextMeasurementConfig, TextMeasurer, DEFAULT_TEXT_MEASUREMENT_CONFIG, DEFAULT_WORD_MEASUREMENT_CONFIG
from ..config.language_metadata import LanguageMetadata
from ..constants import (
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


def _is_ws(ch: str) -> bool:
    """True for ASCII whitespace or any Unicode space separator (Zs)."""
    return ch in _ASCII_WS or unicodedata.category(ch) == 'Zs'


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
                snippets. Read only when the input provider carries no
                registered code corpus, which is the case when this class is
                constructed on its own rather than by
                ``UnifiedTokenizerAnalyzer``.
            math_data_path: Optional path to math-rich text file. Read under
                the same condition as *code_texts*.
            use_builtin_math_data: Whether to use built-in math samples. Read
                under the same condition as *code_texts*.
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
        self._code_texts: Dict[str, List[str]] = {}
        self._math_items: List[Tuple[str, str]] = []

        code_corpus = self._registered_corpus(CODE_CORPUS)
        if code_corpus is None:
            self._code_texts = code_texts or {}
        elif not code_corpus.synthetic:
            self._code_texts = dict(code_corpus.texts)

        math_corpus = self._registered_corpus(MATH_CORPUS)
        if math_corpus is None:
            # No corpus is registered, so nothing indexes these texts and the
            # label is never used to look one up. MATH_CORPUS is the label
            # loaders/corpora.py gives the math texts it resolves, so the two
            # paths agree on it.
            if math_data_path:
                self._math_items = [
                    (MATH_CORPUS, text) for text in load_math_data(math_data_path)
                ]
            elif use_builtin_math_data:
                self._math_items = [
                    (MATH_CORPUS, text)
                    for text in load_math_data(BUILTIN_MATH_SAMPLES_PATH)
                ]
        elif not math_corpus.synthetic:
            self._math_items = [
                (label, text)
                for label, texts in math_corpus.texts.items()
                for text in texts
            ]

    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None,
                include_reconstruction: bool = True,
                cer_time_budget_s: float = DEFAULT_CER_TIME_BUDGET_S) -> Dict[str, Any]:
        """
        Compute basic tokenization metrics.

        Args:
            tokenized_data: Optional tokenized data dict. If None, uses input_provider data.
            include_reconstruction: Whether to include reconstruction fidelity analysis.
            cer_time_budget_s: Max seconds to spend on CER per tokenizer before
                skipping.  0 disables the budget (always compute).

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
                tokenized_data, cer_time_budget_s=cer_time_budget_s))

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
                    'aggregation': AGGREGATION_MICRO_POOLED,
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
        
        for data in tokenized_data:
            if not data.text or not data.text.strip():
                continue  # Skip if no text available

            num_tokens = len(data.tokens)
            num_units = self.fertility_text_measurer.get_unit_count(data.text)
            
            if num_units > 0:
                fertility = num_tokens / num_units
                fertilities.append(fertility)
        
        if not fertilities:
            return self.empty_stats()
        
        return self.compute_basic_stats(fertilities)
    
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
                    'aggregation': AGGREGATION_MICRO_POOLED,
                    'count_unit': 'tokens',
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
        those texts is whitespace-only, which max_snippet_chars produces by
        truncating an indented file down to its leading whitespace. Pairing the
        two lists by index after one such drop would score every later text
        against the following text's ids, and nothing in the results would show
        it. Two identical texts collapse onto one entry, which is harmless:
        identical text encodes to identical ids.

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
        is bounded. For HuggingFaceTokenizer, UniMixLMTokenizer and
        CustomBPETokenizer the offsets come free and the provider encodes one
        batch per label: 4.00 s against 10.29 s for per-text calls over the
        1500 files benchmarks/open_source/code_ast_config.json names, so those
        wrappers are faster here than the per-text encode() this replaced. The
        script_bpe wrappers compute offsets by replaying the pretokenizer, at
        2.16x the cost of the ids alone, and have no batch path, so they are
        the ones that pay.
        """
        lookup: Dict[str, Dict[str, Dict[Tuple[str, str], List[int]]]] = {}
        measured = (
            (CODE_CORPUS, bool(self._code_texts)),
            (MATH_CORPUS, bool(self._math_items)),
        )
        for corpus_name, has_texts in measured:
            if not has_texts or self._registered_corpus(corpus_name) is None:
                continue
            encoded = self.input_provider.get_tokenized_data(corpus_name)
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
                    'count_unit': 'documents',
                    'per_domain': (
                        'This metric breaks down by domain rather than by '
                        'language: per_domain holds one entry per language of '
                        'the prose corpus, one per programming language and '
                        'one for math. Each entry carries count in documents, '
                        'so it is the per_language block under another name.'
                    ),
                },
            }
        }

        # The ids the provider already made for the code and math corpora,
        # empty when the run registered neither.
        shared_ids = self._shared_corpus_ids()

        # Every code and math text, counted before the `text.strip()` filter the
        # loop below applies. This over-counts a whitespace-only text, and that
        # is deliberate: the count feeds the estimate of how much CER work is
        # left, which decides whether cer_skipped flips, so filtering it here
        # would move published values. It does not depend on the tokenizer, so
        # it is counted once for the whole loop.
        total_code_math_texts = (
            sum(len(snippets) for snippets in self._code_texts.values())
            + len(self._math_items)
        )

        for tok_name in self.tokenizer_names:
            try:
                tokenizer = self.input_provider.get_tokenizer(tok_name)
            except Exception as e:
                logger.warning(
                    "Reconstruction fidelity: skipping %s (could not load tokenizer: %s)",
                    tok_name, e,
                )
                continue

            if not tokenizer.can_decode():
                logger.info("Reconstruction fidelity: skipping %s (no decode support)", tok_name)
                continue

            if total_code_math_texts:
                # This metric selects on can_decode() alone, and the code/math
                # loop below then needs raw text encoded. A tokenizer that
                # decodes but cannot encode (a pre-tokenized provider's) used to
                # reach that call and raise NotImplementedError out of the whole
                # analysis. Decided 2026-08-19: skip it with a warning instead,
                # which is what InputProvider._encode_corpus already does with
                # the same tokenizer, so it is left out of the encoded corpus
                # rather than crashing the run. The check is conditional on
                # there being code or math text because with none the loop
                # encodes nothing, and such a tokenizer's prose numbers are
                # measurable exactly as they have always been.
                if not InputProvider._can_encode_raw_text(tokenizer):
                    logger.warning(
                        "Reconstruction fidelity: skipping %s (it decodes but "
                        "cannot encode raw text, and this run has %d code/math "
                        "texts to encode; its prose domains are skipped with "
                        "it). Pre-tokenized input supplies ids, not an encoder.",
                        tok_name, total_code_math_texts,
                    )
                    continue
                missing = sorted(
                    corpus_name for corpus_name, per_corpus in shared_ids.items()
                    if tok_name not in per_corpus
                )
                if missing:
                    raise ValueError(
                        f"Tokenizer {tok_name!r} passes "
                        "InputProvider._can_encode_raw_text, which is the same "
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
            total_all_texts = total_lang_texts + total_code_math_texts
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
                        continue

                    stats['total'] += 1

                    if decoded == text:
                        stats['exact_matches'] += 1
                        if not cer_skipped:
                            ws_total = sum(1 for c in text if _is_ws(c))
                            stats['ws_preserved'] += ws_total
                            stats['ws_total'] += ws_total
                    else:
                        if not cer_skipped:
                            t0 = time.monotonic()
                            stats['cer_sum'] += self._character_error_rate(text, decoded)
                            cer_elapsed += time.monotonic() - t0
                            n_cer_calls += 1

                            ws_preserved, ws_total = self._whitespace_fidelity(text, decoded)
                            stats['ws_preserved'] += ws_preserved
                            stats['ws_total'] += ws_total

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
            for lang, snippets in self._code_texts.items():
                domain = f"code_{lang}"
                for snippet in snippets:
                    if snippet and snippet.strip():
                        code_math_pairs.append((snippet, domain, CODE_CORPUS, lang))
            for label, text in self._math_items:
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
                    continue

                stats['total'] += 1

                if decoded == text:
                    stats['exact_matches'] += 1
                    if not cer_skipped:
                        ws_total = sum(1 for c in text if _is_ws(c))
                        stats['ws_preserved'] += ws_total
                        stats['ws_total'] += ws_total
                else:
                    if not cer_skipped:
                        t0 = time.monotonic()
                        stats['cer_sum'] += self._character_error_rate(text, decoded)
                        cer_elapsed += time.monotonic() - t0
                        n_cer_calls += 1

                        ws_preserved, ws_total = self._whitespace_fidelity(text, decoded)
                        stats['ws_preserved'] += ws_preserved
                        stats['ws_total'] += ws_total

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
            }

            for domain, ds in domain_stats.items():
                count = ds['total']
                per_domain[domain] = {
                    'exact_match_rate': self.safe_divide(ds['exact_matches'], count, 0.0),
                    'mean_cer': None if cer_skipped else self.safe_divide(ds['cer_sum'], count, 0.0),
                    'unk_token_rate': self.safe_divide(ds['unk_tokens'], ds['total_tokens'], 0.0),
                    'whitespace_fidelity': None if cer_skipped else self.safe_divide(ds['ws_preserved'], ds['ws_total'], 1.0),
                    'count': count,
                    'total_tokens': ds['total_tokens'],
                }
                for k in overall:
                    overall[k] += ds[k]

            total = overall['total']
            tok_result = {
                'by_domain': per_domain,
                'overall': {
                    'exact_match_rate': self.safe_divide(overall['exact_matches'], total, 0.0),
                    'mean_cer': None if cer_skipped else self.safe_divide(overall['cer_sum'], total, 0.0),
                    'unk_token_rate': self.safe_divide(overall['unk_tokens'], overall['total_tokens'], 0.0),
                    'whitespace_fidelity': None if cer_skipped else self.safe_divide(overall['ws_preserved'], overall['ws_total'], 1.0),
                    'count': total,
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
                'texts_analyzed': total,
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
    def _whitespace_fidelity(
        original: str,
        decoded: str,
    ) -> Tuple[int, int]:
        """Count whitespace chars preserved through encode-decode round-trip.

        Uses a greedy forward scan with indexed lookup to align original
        characters to decoded characters, then checks whether each
        whitespace position in the original is preserved.

        Returns ``(num_preserved, num_total_ws)`` in the original.
        """
        total_ws = sum(1 for c in original if _is_ws(c))
        if total_ws == 0:
            return (0, 0)

        if original == decoded:
            return (total_ws, total_ws)

        # Greedy forward scan with indexed lookup
        char_positions = defaultdict(list)
        for i, c in enumerate(decoded):
            char_positions[c].append(i)

        preserved = 0
        j = 0
        for c in original:
            if not _is_ws(c):
                positions = char_positions.get(c)
                if positions is not None:
                    idx = bisect_left(positions, j)
                    if idx < len(positions):
                        j = positions[idx] + 1
            else:
                if j < len(decoded) and decoded[j] == c:
                    preserved += 1
                    j += 1

        return (preserved, total_ws)