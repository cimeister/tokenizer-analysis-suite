"""
Tokenizer Fairness Gini coefficient implementation.

This module implements the Tokenizer Fairness Gini (TFG) coefficient,
which measures how equitably a tokenizer treats different languages.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging

from .base import BaseMetrics, TokenizedDataProcessor, format_optional
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider
from ..config import (
    TextMeasurementConfig, TextMeasurer, DEFAULT_TEXT_MEASUREMENT_CONFIG,
    NormalizationMethod,
)
from ..config.language_metadata import LanguageMetadata
from ..constants import AGGREGATION_MACRO_LANGUAGES, MIN_LANGUAGES_FOR_GINI

logger = logging.getLogger(__name__)


class TokenizerGiniMetrics(BaseMetrics):
    """
    Implements Tokenizer Fairness Gini coefficient and related metrics.
    
    The TFG measures fairness by computing token costs per language and
    calculating the Gini coefficient of the distribution of these costs.
    """
    
    def __init__(self, input_provider: InputProvider, 
                 measurement_config: Optional[TextMeasurementConfig] = None,
                 language_metadata: Optional[LanguageMetadata] = None):
        super().__init__(input_provider)
        # Whatever unit the run measures text in, bytes by default. The comment
        # here used to say lines. It never matched the code: this assignment has
        # always taken DEFAULT_TEXT_MEASUREMENT_CONFIG, which is the byte
        # config, and this module does not import the line config at all. Cost
        # per byte is the intended quantity: it is the one unit that means the
        # same thing in every script.
        self.measurement_config = measurement_config or DEFAULT_TEXT_MEASUREMENT_CONFIG
        self.text_measurer = TextMeasurer(self.measurement_config)
        self.language_metadata = language_metadata
    
    @classmethod
    def _fairness_block(cls, tok_name, language_costs):
        """The published fairness block for one tokenizer's language costs.

        One function for both cases, because they were two literal dictionaries
        that had to be kept in step by hand and were not: the fewer-than-two
        case omitted seven of the keys the other one carries, directly below a
        comment saying every block carries the same keys.

        Only two values are genuinely undefined with a single language: the
        coefficient, because inequality across one language means nothing, and
        the standard deviation, because one value has no spread. The smallest
        and largest cost are that language's cost, their ratio is 1.0, and the
        most and least efficient language are both that one. Publishing null
        for those would be a stand-in where a real value exists.
        """
        # Sorted, so the sum is order-independent: floating point addition is
        # not associative, and an order that varied gave a different last digit
        # between runs of the same commit.
        total_costs = [language_costs[lang] for lang in sorted(language_costs)]
        enough = len(language_costs) >= MIN_LANGUAGES_FOR_GINI

        if not language_costs:
            mean_cost = min_cost = max_cost = cost_ratio = None
            sorted_langs = []
            most_efficient = least_efficient = None
        else:
            mean_cost = float(np.mean(total_costs))
            min_cost = min(total_costs)
            max_cost = max(total_costs)
            # None, not inf: json.dump writes bare Infinity, which is not valid
            # JSON and fails strict parsers such as JavaScript's JSON.parse.
            cost_ratio = max_cost / min_cost if min_cost > 0 else None
            sorted_langs = sorted(language_costs.items(), key=lambda x: x[1])
            most_efficient = sorted_langs[0]
            least_efficient = sorted_langs[-1]

        block = {
            'gini_coefficient': cls._gini_of(total_costs) if enough else None,
            'mean_cost': mean_cost,
            # ddof=1 to match BaseMetrics.compute_basic_stats and
            # vocabulary_utilization's per_language_std, which both use the
            # sample convention. At 13 languages the population form
            # understated this by 4.1%.
            'std_cost': float(np.std(total_costs, ddof=1)) if enough else None,
            'min_cost': min_cost,
            'max_cost': max_cost,
            'cost_ratio': cost_ratio,

            'language_costs': language_costs,
            'most_efficient_language': most_efficient,
            'least_efficient_language': least_efficient,
            'num_languages': len(language_costs),
            'sorted_language_costs': sorted_langs,
        }
        if not enough:
            block['warning'] = (
                f'Undefined for fewer than {MIN_LANGUAGES_FOR_GINI} languages '
                f'(got {len(language_costs)})'
            )
        return block

    @staticmethod
    def _gini_of(costs: List[float]) -> Optional[float]:
        """TFG over an already-ordered cost vector.

        ``TFG = sum_i sum_j |c_i - c_j| / (2 n^2 mu)``. Returns None when the
        mean is zero, which means no tokens were produced for any language, so
        there is no inequality to report rather than equality to report.

        *costs* must arrive in a fixed order. The double sum is floating point
        addition, which is not associative, so an order that varied gave a
        different last digit on each run.
        """
        n = len(costs)
        if n == 0:
            return None
        mu = float(np.mean(costs))
        if mu <= 0:
            return None
        total = 0.0
        for i in range(n):
            for j in range(n):
                total += abs(costs[i] - costs[j])
        return total / (2 * n * n * mu)

    def _compute_lines_per_language(
        self, tok_data: List[TokenizedData], languages: List[str]
    ) -> Dict[str, int]:
        """Line count per language, counted by the shared TextMeasurer.

        Uses the library's own line counter rather than a local one, so a
        change to what counts as a line reaches this and the ``lines``
        measurement config together.
        """
        measurer = TextMeasurer(
            TextMeasurementConfig(method=NormalizationMethod.LINES)
        )
        lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
        lines: Dict[str, int] = {}
        for lang in languages:
            if lang not in lang_groups:
                continue
            lines[lang] = sum(
                measurer.get_unit_count(d.text)
                for d in lang_groups[lang]
                if d.text and d.text.strip()
            )
        return lines

    def _per_line_block(
        self, tok_data: List[TokenizedData], languages: List[str]
    ) -> Optional[Dict[str, Any]]:
        """The coefficient with each language's cost taken per line.

        Returns None when the languages do not all have the same line count, or
        when fewer than ``MIN_LANGUAGES_FOR_GINI`` have any lines at all. The
        caller writes null in that case; the reason a reader needs is in the
        metric's ``metadata.per_line_normalization``.
        """
        lines = self._compute_lines_per_language(tok_data, languages)
        lines = {lang: n for lang, n in lines.items() if n > 0}
        if len(lines) < MIN_LANGUAGES_FOR_GINI:
            return None
        counts = set(lines.values())
        if len(counts) != 1:
            logger.info(
                "Per-line Gini not published: line counts differ across "
                "languages (%s). Lines are only comparable across languages "
                "when each language has the same number of them.",
                ", ".join(f"{lang}={n}" for lang, n in sorted(lines.items())),
            )
            return None

        lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
        costs: Dict[str, float] = {}
        for lang in sorted(lines):
            tokens = sum(
                len(d.tokens) for d in lang_groups[lang]
                if d.text and d.text.strip()
            )
            costs[lang] = tokens / lines[lang]

        ordered = [costs[lang] for lang in sorted(costs)]
        min_cost, max_cost = min(ordered), max(ordered)
        return {
            'gini_coefficient': self._gini_of(ordered),
            'lines_per_language': counts.pop(),
            'mean_cost': float(np.mean(ordered)),
            'cost_ratio': max_cost / min_cost if min_cost > 0 else None,
            'language_costs': costs,
            'num_languages': len(costs),
        }

    def _compute_language_costs(self, tok_data: List[TokenizedData],
                                languages: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute per-language token costs (tokens / normalization units).

        Args:
            tok_data: Tokenized data for a single tokenizer.
            languages: Optional list of languages to consider.  If ``None``,
                all languages present in *tok_data* are used.

        Returns:
            Dict mapping language code to token cost.
        """
        lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
        if languages is None:
            languages = sorted(lang_groups.keys())

        language_costs: Dict[str, float] = {}
        for lang in languages:
            if lang not in lang_groups:
                continue
            lang_data = lang_groups[lang]
            total_tokens = 0
            total_normalization_units = 0
            for data in lang_data:
                if data.text and data.text.strip():
                    total_tokens += len(data.tokens)
                    total_normalization_units += self.text_measurer.get_unit_count(data.text)
            if total_normalization_units > 0:
                language_costs[lang] = total_tokens / total_normalization_units
                logger.debug(
                    "  %s: %d tokens / %d %s = %.4f",
                    lang, total_tokens, total_normalization_units,
                    self.measurement_config.method.value,
                    language_costs[lang],
                )
        return language_costs

    def compute_tokenizer_fairness_gini(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute Tokenizer Fairness Gini (TFG) coefficient.

        The TFG is defined as:

        1. For each language ℓ, compute token cost on parallel corpus:
           c_ℓ = (number of tokens) / (number of raw bytes, characters or lines)

        2. Compute mean cost: μ = (1/n) * Σ c_ℓ

        3. Compute TFG:
           TFG = Σᵢ Σⱼ |c_i - c_j| / (2 * n² * μ)

        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists

        Returns:
            Dict containing TFG coefficients and related metrics for each tokenizer
        """

        results = {
            'per_tokenizer': {},
            'metadata': {
                'description': 'Tokenizer Fairness Gini coefficient measures equitable treatment across languages',
                'formula': 'TFG = Σᵢ Σⱼ |c_i - c_j| / (2 * n² * μ)',
                'interpretation': 'Lower values indicate more equitable treatment (0 = perfect equality)',
                'max_attainable': '(n-1)/n for n languages, not 1',
                'aggregation': AGGREGATION_MACRO_LANGUAGES,
                'count_unit': 'languages',
                'per_language_count': (
                    'Not published. The count_unit is languages and a '
                    'per_language entry is one language, so the count would be '
                    '1 for every entry. The number of languages behind the '
                    'coefficient is global.num_languages.'
                ),
                'std_ddof': 1,
                'normalization_method': self.measurement_config.method.value,
                'per_line_normalization': (
                    'per_tokenizer.<tok>.per_line_normalization holds the same '
                    'coefficient with each language cost taken as tokens per '
                    'line instead of per '
                    f'{self.measurement_config.method.value}. On a parallel '
                    'corpus, where line i of every language is the same '
                    'sentence, that is the one to read: it compares tokenizers '
                    'on identical content, which no other unit does. It is null '
                    'unless every language has the same line count, which is '
                    'necessary for a parallel corpus and not sufficient. Equal '
                    'counts do not establish that line i is a translation of '
                    'line i; that part cannot be checked here and is the '
                    "caller's to know."
                ),
                'comparability': (
                    'Comparable only across runs using the same language set and the '
                    'same normalization method. Latin-only and full-13-language '
                    'subsets of the same corpus rank tokenizers at Spearman 0.28, '
                    'and line-normalized against byte-normalized at -0.11.'
                ),
            }
        }

        # Extract all languages from the tokenized data
        all_languages = set()
        for tok_data in tokenized_data.values():
            for data in tok_data:
                all_languages.add(data.language)

        languages = list(all_languages)

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            logger.info(f"Computing TFG for tokenizer: {tok_name}")

            tok_data = tokenized_data[tok_name]
            language_costs = self._compute_language_costs(tok_data, languages)
            # Sorted by language code, so the cost vector is summed in the same
            # order on every run. The mean and the Gini double sum are floating
            # point additions, which are not associative, so an order that
            # varied gave a different last digit each time: the same five
            # per-language costs produced mean_cost 0.19080542854735213 and
            # 0.1908054285473521 under two values of PYTHONHASHSEED. A library
            # whose results file records the commit that produced it should not
            # produce two numbers from one commit.
            total_costs = [language_costs[lang] for lang in sorted(language_costs)]

            if len(language_costs) < MIN_LANGUAGES_FOR_GINI:
                logger.warning(
                    "Tokenizer fairness Gini needs at least %d languages; %s has %d, "
                    "so the coefficient is reported as null rather than 0.0.",
                    MIN_LANGUAGES_FOR_GINI, tok_name, len(language_costs),
                )

            # One block builder for both cases. They were two literal
            # dictionaries kept in step by hand, and were not: the
            # fewer-than-two case omitted seven keys, directly below a comment
            # saying every block carries the same keys.
            results['per_tokenizer'][tok_name] = self._fairness_block(
                tok_name, language_costs,
            )
            if len(language_costs) < MIN_LANGUAGES_FOR_GINI:
                # The per-line coefficient is undefined here for the same
                # reason the main one is.
                results['per_tokenizer'][tok_name]['per_line_normalization'] = None
                continue

            # Names the reporting below still uses, read back from the block
            # so there is one place they are computed.
            block = results['per_tokenizer'][tok_name]
            mu = block['mean_cost']
            n = len(total_costs)
            tfg = block['gini_coefficient']
            min_cost = block['min_cost']
            max_cost = block['max_cost']
            cost_ratio = block['cost_ratio']
            std_cost = block['std_cost']
            sorted_langs = block['sorted_language_costs']
            most_efficient = block['most_efficient_language']
            least_efficient = block['least_efficient_language']

            # The same coefficient with each language's cost normalized by its
            # line count instead of by the configured unit.
            #
            # On a parallel corpus, line i of every language is the same
            # sentence, so tokens per line compares tokenizers on identical
            # content. The configured unit does not: under bytes, a language
            # whose script needs three bytes per character is charged three
            # times the denominator for the same sentence, which flatters a
            # tokenizer on exactly the languages that fragment most. Over the
            # nine tokenizers of benchmarks/open_source the two rank at Spearman
            # 0.650 and disagree on which tokenizer is the fairest:
            # XLM-RoBERTa is fourth at 0.0976 under bytes and first at 0.0494
            # under lines.
            #
            # Published only when every language has the same line count. That
            # is necessary for a parallel corpus and not sufficient, but it is
            # the part that can be checked; without it the per-line costs are
            # not comparable and the coefficient would be a number with no
            # meaning rather than an absent one.
            per_line = self._per_line_block(tok_data, sorted(language_costs))
            if per_line is not None:
                results['per_tokenizer'][tok_name]['per_line_normalization'] = per_line
            else:
                results['per_tokenizer'][tok_name]['per_line_normalization'] = None


            # tfg is None when the mean cost is 0 and cost_ratio is None when the
            # cheapest language cost 0, both published as null rather than as a
            # number. Formatting a null here raised TypeError and took the whole
            # metric down on a corpus that produced no tokens for some language.
            logger.info(
                "  TFG: %s, Mean cost: %.4f, Cost ratio: %s",
                format_optional(tfg, ".4f"), mu, format_optional(cost_ratio, ".2f"),
            )
        
        return results
    
    def compute_lorenz_curve_data(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute Lorenz curve data for visualizing tokenizer fairness.
        
        The Lorenz curve shows the cumulative distribution of token costs,
        useful for visualizing inequality across languages.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict containing Lorenz curve data for each tokenizer
        """
        
        results = {
            'per_tokenizer': {},
            'metadata': {
                'description': 'Lorenz curve data for visualizing tokenizer fairness',
                'x_axis': 'Cumulative proportion of languages (sorted by efficiency)',
                'y_axis': 'Cumulative proportion of total token cost'
            }
        }
        
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            tok_data = tokenized_data[tok_name]
            language_costs = self._compute_language_costs(tok_data)

            if len(language_costs) < MIN_LANGUAGES_FOR_GINI:
                results['per_tokenizer'][tok_name] = {
                    'warning': 'Insufficient data for Lorenz curve'
                }
                continue
            
            # Sort languages by cost (most efficient first)
            sorted_items = sorted(language_costs.items(), key=lambda x: x[1])
            sorted_languages = [item[0] for item in sorted_items]
            sorted_costs = [item[1] for item in sorted_items]
            
            # Compute cumulative proportions
            n_languages = len(sorted_costs)
            total_cost = sum(sorted_costs)
            
            # X-axis: cumulative proportion of languages
            x_values = [0.0]  # Start at 0
            x_values.extend([(i + 1) / n_languages for i in range(n_languages)])
            
            # Y-axis: cumulative proportion of total cost
            y_values = [0.0]  # Start at 0
            cumulative_cost = 0.0
            for cost in sorted_costs:
                cumulative_cost += cost
                y_values.append(cumulative_cost / total_cost)
            
            # Perfect equality line (diagonal)
            equality_line = x_values.copy()
            
            results['per_tokenizer'][tok_name] = {
                'x_values': x_values,
                'y_values': y_values,
                'equality_line': equality_line,
                'sorted_languages': sorted_languages,
                'sorted_costs': sorted_costs,
                'total_cost': total_cost,
                'n_languages': n_languages
            }
        
        return results
    
    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """
        Compute all Gini-related metrics.
        
        Args:
            tokenized_data: Optional dict mapping tokenizer names to TokenizedData lists.
                          If None, will use input_provider's data.
            
        Returns:
            Dict containing all Gini metrics and Lorenz curve data
        """
        if tokenized_data is None:
            tokenized_data = self.get_tokenized_data()
            
        results = {}
        
        # Compute TFG
        tfg_results = self.compute_tokenizer_fairness_gini(tokenized_data)
        results['tokenizer_fairness_gini'] = tfg_results
        
        # Compute Lorenz curve data
        lorenz_results = self.compute_lorenz_curve_data(tokenized_data)
        results['lorenz_curve_data'] = lorenz_results
        
        return results