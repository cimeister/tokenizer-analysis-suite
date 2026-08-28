"""
Digit boundary alignment, digit split variability, numeric magnitude
consistency, and operator isolation metrics.

Based on Singh & Strouse (2024, arXiv:2402.14903), which showed that
right-to-left tokenization of numbers (enforced by comma-grouped formats)
improved arithmetic accuracy by >22 pp.  The mechanism: when digit
positions within operands and result occupy the same position within their
respective tokens, the model can learn a consistent positional mapping.

Four failure modes are measured here:

1. **Three-Digit Boundary Alignment Score**: the tokenizer splits at the
   wrong positions inside a number.
2. **Digit Split Variability**: the tokenizer splits numbers of
   the same digit length at *different* positions depending on the
   specific digit values (value-dependent BPE merges).
3. **Numeric Magnitude Consistency**: does fertility (tokens per digit)
   scale predictably across digit magnitudes?
4. **Operator Isolation Rate**: are mathematical operators tokenized as
   isolated units?
"""

import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import spearmanr
import logging

from .base import BaseMetrics, TokenizedDataProcessor, format_optional
from ..constants import AGGREGATION_MEAN_OF_RATIOS, AGGREGATION_MICRO_POOLED
from ..core.input_types import (
    corpus_size,
    CODE_CORPUS, MATH_CORPUS, PROSE_CORPUS, PROSE_SOURCE, Corpus, TokenizedData,
    published_language_key,
)
from ..core.input_providers import InputProvider
from ..loaders.corpora import code_corpus_from_texts, resolve_math_corpus

logger = logging.getLogger(__name__)


def operator_metadata(include_code_math: bool = True,
                      filtered: bool = False) -> Dict[str, Any]:
    """The published ``operator_isolation_rate`` metadata block.

    One definition, because the description names the domains the block holds
    and a second copy drifted before: main.py's grouped filter used to copy
    the whole-corpus metadata, so every group block said "Code and math
    always run" beside an empty ``per_tokenizer`` and no ``by_domain``
    (RELEASE_AUDIT Q35.2 R2).

    ``include_code_math=True`` is the whole-corpus run. ``False`` is the
    per-group recompute, whose block carries ``by_domain`` holding the
    domains that ran, which is empty at the default settings (prose off).
    ``filtered=True`` is main.py's filter over base results, whose block is
    re-aggregated from the base run and carries no ``by_domain`` at all.
    """
    if filtered:
        description = (
            "How often a mathematical operator is a token of its own, "
            "re-aggregated over this group's prose languages from the base "
            "run. The code and math domains belong to no language group and "
            "are not included; prose rows appear only with "
            "--operator-prose-domain. The corpus-level by_domain split is "
            "reported once in the top-level results, not per group."
        )
    elif include_code_math:
        description = (
            "How often a mathematical operator is a token of its own, "
            "pooled over the domains that ran, with the split kept in "
            "by_domain. Code and math always run; prose runs only "
            "with --operator-prose-domain. Read by_domain to see which."
        )
    else:
        description = (
            "How often a mathematical operator is a token of its own, "
            "pooled over the domains that ran, with the split kept in "
            "by_domain. This run covers one group of prose languages, "
            "so the code and math domains do not run; prose runs only "
            "with --operator-prose-domain. by_domain holds the domains "
            "that ran, and is empty when prose is off."
        )
    return {
        "description": description,
        "aggregation": AGGREGATION_MICRO_POOLED,
        "count_unit": "operator occurrences",
    }


def magnitude_metadata(grouped: bool = False) -> Dict[str, Any]:
    """The published ``numeric_magnitude_consistency`` metadata block.

    The grouped variant exists because the ``scaling`` fit pools every
    language of the run, so main.py's grouped filter drops it from a group
    block rather than copying the whole-corpus fit under a group label, and
    the description has to stop promising it (RELEASE_AUDIT Q35.2 R1).
    """
    if grouped:
        description = (
            "Tokens per digit, re-aggregated over this group's languages "
            "from the base run. The scaling fit pools every language of the "
            "run, so a group block does not carry one; it is in the "
            "top-level results."
        )
    else:
        description = (
            "Tokens per digit, and how it scales with the number of digits."
        )
    return {
        "description": description,
        # mean_fertility is the mean of tokens-per-digit over numbers, one
        # ratio each, not a ratio of summed counts. Same shape as
        # three_digit_boundary_alignment beside it, and built from the same
        # accumulator, so the two carry the same label.
        "aggregation": AGGREGATION_MEAN_OF_RATIOS,
        "count_unit": "numbers",
    }


def _relative_to_package(path: Any) -> Any:
    """Render a path inside the installed package relative to it.

    A path the caller supplied is returned unchanged: it is theirs and naming it
    in full is the point. Only the bundled corpora, whose location depends on
    where the package happens to sit, are shortened.
    """
    if not isinstance(path, str):
        return path
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        inside = os.path.commonpath([os.path.abspath(path), package_root]) == package_root
    except ValueError:
        return path
    return os.path.join(
        "tokenizer_analysis", os.path.relpath(os.path.abspath(path), package_root)
    ) if inside else path


class DigitBoundaryMetrics(BaseMetrics):
    """Digit boundary alignment and digit split variability.

    Worked examples: Three-Digit Boundary Alignment
    =============================================

    ``"1234567"`` (L=7), ideal boundaries: {1, 4}  (i.e. ``"1|234|567"``)

    * Tokenized as ``"1" "234" "567"``   -> actual {1,4} -> P=1.0 R=1.0 F1=1.0
    * Tokenized as ``"1234" "567"``      -> actual {4}   -> P=1.0 R=0.5 F1=0.67
    * Tokenized as ``"12" "345" "67"``   -> actual {2,5} -> P=0.0 R=0.0 F1=0.0
    * Tokenized as ``"1234567"``         -> actual {}    -> P=1.0 R=0.0 F1=0.0

    ``"42"`` (L=2), ideal boundaries: {}

    * Tokenized as ``"42"``   -> actual {} -> P=1.0 R=1.0 F1=1.0
    * Tokenized as ``"4" "2"`` -> actual {1} -> P=0.0 R=1.0 F1=0.0

    Worked examples: Digit Split Variability
    =================================================

    The published ``entropy`` is Shannon entropy in bits, not normalized. These
    examples used to quote a normalized value beside it, which no field carries.

    L=4, numbers ``["1234","5678","9012","3456"]`` all tokenized as
    ``["X","XXX"]`` -> patterns: {(1,): 4}.  H = 0.0 bits

    L=4, ``"1234"`` -> ``(2,)``, ``"5678"`` -> ``(1,)``,
    ``"9012"`` -> ``(2,)``, ``"3456"`` -> ``(1,)``.
    Distribution: {(2,): 0.5, (1,): 0.5}.
    H = 1.0 bit
    """

    _DIGIT_SPAN = re.compile(r'\d+')
    # NOTE: The hyphen-minus ``-`` is always treated as an operator, even when
    # it appears as a unary negative sign (e.g. ``-42``).  Disambiguating
    # unary minus from subtraction requires expression parsing, and for
    # tokenizer evaluation the simpler rule is sufficient.
    _OPERATOR_SPAN = re.compile(r'(?:\*\*|//|<<|>>|<=|>=|=>|==|!=|&&|\|\||\?:|[+\-*/=<>!&|^~%])')

    _OPERATOR_CATEGORIES: Dict[str, List[str]] = {
        "arithmetic": ["+", "-", "*", "/", "//", "%", "**"],
        "comparison": ["<", ">", "<=", ">=", "==", "!="],
        "assignment": ["=", "=>"],
        "logical_bitwise": ["&", "|", "^", "~", "&&", "||"],
        "shift": ["<<", ">>"],
        "ternary": ["?:"],
    }
    _OPERATOR_TO_CATEGORY: Dict[str, str] = {}
    for _cat, _ops in _OPERATOR_CATEGORIES.items():
        for _op in _ops:
            _OPERATOR_TO_CATEGORY[_op] = _cat

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        input_provider: InputProvider,
        math_data_path: Optional[str] = None,
        use_builtin_math_data: bool = False,
        code_texts: Optional[Dict[str, List[str]]] = None,
        include_prose_operators: bool = False,
    ):
        super().__init__(input_provider)
        # Whether the main corpus is scored as a domain of operator isolation.
        #
        # Off by default. An operator is a code construct, and the regex counts
        # a hyphen, a slash and an exclamation mark, all of which are ordinary
        # punctuation in prose. On the nine-tokenizer benchmark prose supplied
        # 568 of 455558 operator occurrences, 0.12%, so it moved the pooled
        # number by almost nothing while being the domain that costs the most to
        # score: a web corpus is far larger than a code corpus and 78.5% of its
        # documents contain at least one character the regex matches.
        self._include_prose_operators = include_prose_operators

        # The corpora for the per-domain operator-isolation split. Prose is the
        # corpus the metric already ran on; code and math are registered on the
        # provider by the run, and built here from the constructor arguments
        # when this class is constructed on its own. Supplying both is refused
        # rather than resolved, so the corpus reported is always the corpus the
        # caller asked for.
        self._code_corpus = self._corpus_or_refuse_arguments(
            # bool(): an empty code_texts asks for nothing, unlike the
            # code_config of the AST metrics, where {} means the bundled
            # samples.
            CODE_CORPUS, {"code_texts": bool(code_texts)}
        )
        self._math_corpus = self._corpus_or_refuse_arguments(
            MATH_CORPUS,
            {"math_data_path": bool(math_data_path),
             "use_builtin_math_data": bool(use_builtin_math_data)},
        )

        # Both corpora are resolved before either is registered, so a failure
        # resolving the second leaves the provider untouched. Registering as
        # they resolved meant a math path that does not exist aborted with the
        # code corpus already on the provider, and the retry then failed with
        # "a corpus named 'code' is already registered", naming neither the
        # original problem nor the fix. UnifiedTokenizerAnalyzer does the same
        # thing for the same reason.
        built_code = (
            code_corpus_from_texts(code_texts)
            if self._code_corpus is None else None
        )
        built_math = (
            resolve_math_corpus(math_data_path, use_builtin_math_data)
            if self._math_corpus is None else None
        )
        if built_code is not None:
            self._code_corpus = self._register_corpus(built_code)
        if built_math is not None:
            self._math_corpus = self._register_corpus(built_math)

        # The digit metrics run on the math corpus only when the caller asked
        # for one. The bundled samples are registered as `synthetic` so that the
        # math operator-isolation domain always has something to score, and the
        # digit metrics then fall back to the prose corpus, which is what they
        # have always done. compute() logs what that costs.
        self._math_texts: List[str] = []
        if not self._math_corpus.synthetic:
            self._math_texts = [
                text for texts in self._math_corpus.texts.values() for text in texts
            ]

        if include_prose_operators:
            logger.info(
                "Operator isolation domains: prose=multilingual, math=%s, code=%s",
                self._math_corpus.source, self._code_corpus.source,
            )
        else:
            logger.info(
                "Operator isolation domains: math=%s, code=%s. The prose domain "
                "is off; pass --operator-prose-domain to score the main corpus "
                "as well.",
                self._math_corpus.source, self._code_corpus.source,
            )

    # ------------------------------------------------------------------
    # Digit-span boundary extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _find_number_spans(text: str) -> List[Tuple[int, int, str]]:
        """Return ``[(start, end, digit_string), ...]`` for every ``\\d+`` in *text*."""
        return [(m.start(), m.end(), m.group()) for m in DigitBoundaryMetrics._DIGIT_SPAN.finditer(text)]

    @staticmethod
    def _get_digit_span_boundaries(
        char_to_token: List[Optional[int]],
        span_start: int,
        span_end: int,
    ) -> Optional[List[int]]:
        """Return internal boundary positions *within* a digit span.

        *span_start* and *span_end* are character offsets in whatever text
        *char_to_token* is indexed by. ``compute()`` and ``compute_per_text()``
        pass source-text positions and a map built from the encoder's offsets.
        A boundary at position *k* means there is a token change between the
        (k-1)-th and k-th digit of the span.

        Every position in ``[span_start, span_end)`` has to be covered by a
        token. The callers check that first and skip the number otherwise,
        because an uncovered position holds ``None``, which compares unequal to
        any token index and would be counted as a boundary.

        Returns ``None`` when the span exceeds the mapped region.
        """
        num_digits = span_end - span_start

        if span_end > len(char_to_token):
            return None

        boundaries: List[int] = []
        for k in range(1, num_digits):
            if char_to_token[span_start + k] != char_to_token[span_start + k - 1]:
                boundaries.append(k)

        return boundaries

    # ------------------------------------------------------------------
    # Ideal (right-aligned) boundaries
    # ------------------------------------------------------------------

    @staticmethod
    def _ideal_boundaries(num_digits: int) -> Set[int]:
        """Right-aligned grouping boundaries at positions 3, 6, 9, ... from the right.

        >>> DigitBoundaryMetrics._ideal_boundaries(7)
        {1, 4}
        >>> DigitBoundaryMetrics._ideal_boundaries(3)
        set()
        >>> DigitBoundaryMetrics._ideal_boundaries(6)
        {3}
        """
        result: Set[int] = set()
        pos = num_digits - 3
        while pos > 0:
            result.add(pos)
            pos -= 3
        return result

    # ------------------------------------------------------------------
    # Boundary scoring with vacuous-case semantics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_boundaries(
        actual: Set[int], ideal: Set[int]
    ) -> Dict[str, float]:
        """Compute precision / recall / F1 with vacuous-case rules.

        Vacuous-case table
        (P = precision, R = recall):

        ========  =====  ===  ===  ===  =========================================
        actual    ideal   P    R   F1   Rationale
        --------  -----  ---  ---  ---  -----------------------------------------
        empty     empty  1.0  1.0  1.0  Short number, single token: nothing to
                                         get wrong, nothing to miss.
        non-empty empty  0.0  1.0  0.0  Short number needlessly split: all
                                         boundaries are spurious (P=0).  R=1 by
                                         vacuous truth (zero ideal boundaries
                                         were all trivially recovered).  F1=0
                                         because precision dominates.
        empty     non-∅  1.0  0.0  0.0  Long number kept as single token: no
                                         spurious boundaries (P=1) but none of
                                         the ideal ones were produced (R=0).
        non-empty non-∅  TP/(TP+FP)  TP/(TP+FN)  harmonic mean
        ========  =====  ===  ===  ===  =========================================
        """
        if not actual and not ideal:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if actual and not ideal:
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
        if not actual and ideal:
            return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

        tp = len(actual & ideal)
        fp = len(actual - ideal)
        fn = len(ideal - actual)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    # ------------------------------------------------------------------
    # Pattern entropy
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pattern_entropy(
        patterns: List[tuple],
    ) -> Dict[str, Any]:
        """Shannon entropy (bits) of the boundary-pattern distribution.

        Returns a dict with *entropy* (in bits), *num_patterns*,
        *dominant_pattern*, *dominant_pattern_freq*, and *count*.
        """
        if not patterns:
            return {
                "entropy": 0.0,
                "num_patterns": 0,
                "dominant_pattern": (),
                "dominant_pattern_freq": 0.0,
                "count": 0,
            }

        counts: Dict[tuple, int] = defaultdict(int)
        for p in patterns:
            counts[p] += 1

        total = len(patterns)
        k = len(counts)

        dominant_pattern = max(counts, key=counts.get)  # type: ignore[arg-type]
        dominant_freq = counts[dominant_pattern] / total

        if k <= 1:
            return {
                "entropy": 0.0,
                "num_patterns": k,
                "dominant_pattern": dominant_pattern,
                "dominant_pattern_freq": dominant_freq,
                "count": total,
            }

        entropy = 0.0
        for cnt in counts.values():
            p = cnt / total
            if p > 0:
                entropy -= p * math.log2(p)

        return {
            "entropy": entropy,
            "num_patterns": k,
            "dominant_pattern": dominant_pattern,
            "dominant_pattern_freq": dominant_freq,
            "count": total,
        }

    # ------------------------------------------------------------------
    # Bucket helpers
    # ------------------------------------------------------------------

    _SHORT_THRESHOLD = 3  # digit lengths <= this are "short"

    @staticmethod
    def _digit_length_bucket(length: int) -> str:
        """Return the per-length key: ``'1'`` .. ``'9'`` or ``'10+'``."""
        if length <= 9:
            return str(length)
        return "10+"

    @staticmethod
    def _is_short_bucket(bucket: str) -> bool:
        """Whether *bucket* (a digit-length key) falls in the short category.

        Short = digit lengths 1..3.  The ``'10+'`` bucket is always long.
        """
        if bucket.endswith("+"):
            return False
        return int(bucket) <= DigitBoundaryMetrics._SHORT_THRESHOLD

    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------

    def compute(
        self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None,
        include_code_math: bool = True,
    ) -> Dict[str, Any]:
        """Compute digit boundary alignment, digit split variability,
        numeric magnitude consistency, and operator isolation rate.

        When *math_data_path* was provided at construction time, the digit
        metrics tokenize the loaded math texts and use that data **instead of**
        the ``tokenized_data`` parameter.

        *include_code_math* says whether this call covers the whole corpus or
        one language group. With it off, the digit metrics measure nothing:
        neither the maths corpus, which is not part of the group, nor the
        group's prose, which is a different quantity from the one these fields
        hold in a whole-corpus run. It used to read the maths corpus whatever
        this said, so every group reported identical whole-corpus figures.

        Operator isolation is reported per domain under
        ``operator_isolation_rate["by_domain"]``, and the top-level ``summary``
        pools them. Each domain records the corpus it used. **code** and
        **math** run unless *include_code_math* is False. **prose**, the main corpus, runs only
        when the instance was built with ``include_prose_operators=True``. With
        code and math off and prose off, which is a grouped run at the default
        settings, ``by_domain`` is empty.

        Returns a dict with four top-level keys:
        ``three_digit_boundary_alignment``, ``digit_split_variability``,
        ``numeric_magnitude_consistency``, and ``operator_isolation_rate``.
        """
        # Prose is whatever corpus the caller (or the provider) supplies. It is
        # kept separate so operator isolation can still report on prose even
        # when dedicated math texts replace the corpus for the digit metrics.
        prose_data = (
            tokenized_data
            if tokenized_data is not None
            else self.input_provider.get_tokenized_data()
        )

        if not include_code_math:
            # A language group is a subset of the prose corpus, and neither the
            # maths corpus nor the code corpus is part of it. Measuring nothing
            # is the honest answer.
            #
            # The two alternatives are both worse. Reading the maths corpus, as
            # this used to, gives every group the same whole-corpus figures.
            # Falling through to the prose branch below measures a different
            # quantity: that branch's own comment records that its numbers are
            # not comparable with a run that has a maths corpus, and that most
            # of the sample is vacuous. Publishing either under the same field
            # name reports one thing under the name of another.
            #
            # The filtered grouped path already produced empty digit blocks,
            # because it selects the group's prose languages and the rows are
            # keyed by the maths corpus. This makes the unfiltered path agree
            # with it instead of publishing whole-corpus numbers.
            tokenized_data = {}
            logger.info(
                "include_code_math is off, so the digit metrics measure "
                "nothing: neither the maths corpus nor the group's prose is "
                "the corpus these numbers are defined over."
            )
        elif self._math_texts:
            # The registered math corpus IS _math_texts whenever the caller
            # asked for math texts, so the digit corpus and the math operator
            # domain below share one encoding.
            tokenized_data = self.input_provider.get_corpus_data(MATH_CORPUS)
            logger.info(
                "Using %d dedicated math texts for digit boundary metrics",
                len(self._math_texts),
            )
        else:
            # No dedicated math corpus, so the digit metrics run on whatever
            # prose corpus was loaded. This branch used to be silent. On
            # FLORES the observed digit lengths are 1 to 4 only, so a metric
            # named for three-digit grouping never sees a single ideal
            # boundary, and 74.2% of the sample falls in the vacuous
            # length-3-or-under case, where avg_recall and avg_uniform_chunk
            # equal the corpus short-number share identically for every
            # tokenizer that does not split digits. Numbers from this branch
            # are not comparable with numbers from a run that passed
            # --math-data or --use-builtin-math-data.
            tokenized_data = prose_data
            logger.warning(
                "%s\nNO MATH CORPUS GIVEN: the digit metrics "
                "(three_digit_boundary_alignment, digit_split_variability, "
                "numeric_magnitude_consistency) will be computed on the prose "
                "corpus, which carries few long numbers. These numbers are NOT "
                "comparable to a run that passed --math-data or "
                "--use-builtin-math-data. Pass --no-digit-boundary to disable "
                "them outright.\n%s", "=" * 78, "=" * 78,
            )

        # ---- accumulators ----
        # alignment: tok -> lang -> digit_length_str -> list of per-number dicts
        alignment_acc: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        # entropy: tok -> lang -> digit_length_str -> list of boundary pattern tuples
        entropy_acc: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            tokenizer_obj = self.input_provider.get_tokenizer(tok_name)
            lang_groups = TokenizedDataProcessor.group_by_language(
                tokenized_data[tok_name]
            )
            docs_without_offsets = 0
            spans_not_covered = 0

            for lang, data_list in lang_groups.items():
                for item in data_list:
                    if item.text is None or not item.text.strip():
                        continue

                    # Quick check: any digits at all? (Operator isolation is
                    # accumulated separately, per domain, further below.)
                    has_digits = bool(self._DIGIT_SPAN.search(item.text))
                    if not has_digits:
                        continue

                    # Digits are located in the SOURCE and resolved to tokens
                    # through the encoder's own offsets, which is what operator
                    # isolation and the code metrics already do.
                    #
                    # This used to map source positions into a reconstruction
                    # built by concatenating cleaned token strings.
                    # ``_process_token`` removes one leading space per token
                    # rather than all of them, so a token whose surface is
                    # several spaces (``ĠĠĠ`` in Llama 3, OLMo 2, Qwen 2.5 and
                    # Mistral NeMo) left residual spaces in the reconstruction
                    # that no source character corresponds to.
                    # ``_build_source_to_recon_map`` then resynchronized on one
                    # of those residual spaces and every position after it was
                    # off by however many spaces had accumulated, so digits were
                    # read against the wrong tokens. The same defect measured on
                    # ``ast_boundary_alignment`` published a
                    # ``full_alignment_rate`` of 0.127 for Llama 3 where the
                    # offsets give 0.519.
                    #
                    # The offsets are used untrimmed here, unlike
                    # ``ASTBoundaryMetrics._trim_word_start_space_offsets``. That
                    # trim exists because the ByteLevel ``trim_offsets`` flag
                    # decides whether a word-initial token such as ``Ġfoo``
                    # reports the space in its range, which changes whether an
                    # AST node beginning at ``foo`` looks token-initial. Nothing
                    # here reads a token's claimed start: the boundaries are
                    # token changes between consecutive digits, and a digit is
                    # never the word-start space, so the flag does not affect the
                    # digit metrics. Measured over the 633 numbers of the
                    # bundled math corpus plus two indented code snippets, the
                    # boundaries are the same with and without the trim for all
                    # nine benchmark tokenizers, including gpt2
                    # (``trim_offsets`` false) and gpt-neox-20b (true).
                    #
                    # The SentencePiece word-start marker does affect them: its
                    # reported range covers the first character of the following
                    # word. See _char_to_token_from_offsets for why
                    # keep_later_token_on_overlap is the correct reading of it.
                    char_to_token = self._char_to_token_from_offsets(
                        len(item.text), item.offsets,
                        keep_later_token_on_overlap=True,
                    )
                    if char_to_token is None:
                        if InputProvider.can_encode_raw_text(tokenizer_obj):
                            raise ValueError(
                                "The digit metrics need character offsets, and "
                                f"tokenizer {tok_name!r} encodes text but "
                                "reported none. Offsets are the only exact way "
                                "to say which token covers which digit. The "
                                "mapping this replaced rebuilt the text from "
                                "token surfaces and mismapped every position "
                                "after a run of whitespace, without reporting "
                                "anything. Use a tokenizer whose "
                                "encode_with_offsets() returns offsets, or "
                                "disable these metrics with "
                                "--no-digit-boundary."
                            )
                        # Pre-tokenized input carries ids and text but no
                        # offsets, so the correspondence cannot be established
                        # at all. The tokenizer is left out of the digit results
                        # entirely rather than scored, which is what operator
                        # isolation already does for the same input.
                        docs_without_offsets += 1
                        continue

                    # Find digit spans on the *original* text to avoid
                    # merging adjacent numbers separated by whitespace.
                    spans = self._find_number_spans(item.text)

                    for src_start, src_end, digit_str in spans:
                        num_digits = len(digit_str)
                        if num_digits == 0:
                            continue

                        if any(
                            char_to_token[i] is None
                            for i in range(src_start, src_end)
                        ):
                            # Some digit of this number is covered by no token,
                            # so the number was not measured. It is skipped
                            # rather than scored.
                            spans_not_covered += 1
                            continue

                        bucket = self._digit_length_bucket(num_digits)

                        boundaries = self._get_digit_span_boundaries(
                            char_to_token, src_start, src_end,
                        )
                        if boundaries is None:
                            continue

                        actual = set(boundaries)
                        ideal = self._ideal_boundaries(num_digits)
                        scores = self._score_boundaries(actual, ideal)

                        # Uniform-chunk: all token pieces have same digit count
                        # Reconstruct chunk lengths from boundaries
                        bnd_list = sorted(boundaries)
                        chunk_lengths = []
                        prev = 0
                        for b in bnd_list:
                            chunk_lengths.append(b - prev)
                            prev = b
                        chunk_lengths.append(num_digits - prev)
                        uniform_chunk = 1.0 if len(set(chunk_lengths)) <= 1 else 0.0

                        single_token = 1 if len(bnd_list) == 0 else 0

                        num_tokens = len(bnd_list) + 1
                        fertility_per_digit = num_tokens / num_digits

                        alignment_acc[tok_name][lang][bucket].append({
                            "precision": scores["precision"],
                            "recall": scores["recall"],
                            "f1": scores["f1"],
                            "uniform_chunk": uniform_chunk,
                            "single_token": single_token,
                            "fertility_per_digit": fertility_per_digit,
                            # Real digit count of this number, not the bucket
                            # label. The '10+' bucket holds numbers of many
                            # different lengths, so this is what lets the
                            # fertility-scaling fit use each bucket's true mean
                            # length instead of a fixed stand-in.
                            "num_digits": num_digits,
                        })

                        pattern = tuple(sorted(boundaries))
                        entropy_acc[tok_name][lang][bucket].append(pattern)

            if docs_without_offsets:
                logger.warning(
                    "%s: %d document(s) with digits reported no character "
                    "offsets, so %s has no three_digit_boundary_alignment, "
                    "digit_split_variability or numeric_magnitude_consistency "
                    "entry. Absent, not zero: a zero would read as a tokenizer "
                    "that splits every number wrongly.",
                    tok_name, docs_without_offsets, tok_name,
                )
            if spans_not_covered:
                logger.warning(
                    "%s: %d number(s) had a digit that no token covered and "
                    "were left out of the digit metrics. The encoding offsets "
                    "did not cover part of the source.",
                    tok_name, spans_not_covered,
                )

        # ---- operator isolation: code / math, and prose when asked for ----
        # Prose is opt-in. See the constructor for why: it supplied 0.12% of the
        # operator occurrences on the benchmark and most of the runtime on a web
        # corpus. When it is off, nothing about the main corpus reaches this
        # metric, so `by_domain` has two entries and the pooled figure is a
        # code-and-math number.
        domain_accs = {}
        if self._include_prose_operators:
            domain_accs[PROSE_CORPUS] = self._accumulate_operators(prose_data)
        # include_code_math is False for a run over one language group. The
        # group selects prose languages and the code and math corpora belong to
        # no language, so reporting the whole of both inside every group made
        # each group's pooled figure a number measured mostly on texts the group
        # does not contain. With prose operators off, which is the default, a
        # group then has no operator domain at all, and by_domain is empty
        # rather than describing corpora the group never saw.
        if include_code_math:
            domain_accs[CODE_CORPUS] = self._accumulate_operators(
                self.input_provider.get_corpus_data(CODE_CORPUS)
            )
            domain_accs[MATH_CORPUS] = self._accumulate_operators(
                self.input_provider.get_corpus_data(MATH_CORPUS)
            )
        # Bundled paths are recorded relative to the package. They are derived
        # from __file__, so the absolute form bakes the author's checkout
        # directory into every results file, and a reader comparing two files
        # sees a difference that is only where the package was installed.
        # Keyed on domain_accs, so a domain that did not run is described
        # nowhere rather than being described with an empty result beside it.
        all_sources = {
            PROSE_CORPUS: PROSE_SOURCE,
            CODE_CORPUS: _relative_to_package(self._code_corpus.source),
            MATH_CORPUS: _relative_to_package(self._math_corpus.source),
        }
        domain_sources = {d: all_sources[d] for d in domain_accs}
        # Size of each domain's corpus, so the pooled micro-average can be traced
        # back to which corpus supplied the operators. Prose is described as a
        # Corpus too, recovered from the data the metric was handed, because the
        # prose texts are never registered: they are whatever corpus the caller
        # (or the provider) supplied for this call.
        all_corpora = {
            CODE_CORPUS: lambda: self._code_corpus.stats(),
            MATH_CORPUS: lambda: self._math_corpus.stats(),
            # corpus_size, not a throwaway Corpus: prose is never registered,
            # and building one copied every prose text into a tuple on each
            # call to reach the same four numbers.
            PROSE_CORPUS: lambda: corpus_size(
                self._texts_by_language(prose_data)
            ),
        }
        domain_corpora = {d: all_corpora[d]() for d in domain_accs}

        # The top-level per_tokenizer/summary pool whichever domains ran. The
        # pool is a micro-average over operator instances, so it is weighted by
        # how many operators each corpus contributes: with a real code dataset,
        # code dominates. The per-domain denominators are published in
        # ``domain_operator_counts`` so that weighting is visible rather than
        # implicit.
        operator_results = self._build_operator_results(
            self._merge_operator_accs(domain_accs), include_code_math,
        )
        operator_results["by_domain"] = {
            domain: dict(
                self._build_operator_results(acc, include_code_math),
                source=domain_sources[domain],
                corpus=domain_corpora[domain],
            )
            for domain, acc in domain_accs.items()
        }
        operator_results["domain_operator_counts"] = {
            tok_name: {
                domain: (
                    operator_results["by_domain"][domain]["summary"]
                    .get(tok_name, {})
                    .get("total_operators", 0)
                )
                for domain in domain_accs
            }
            for tok_name in operator_results["summary"]
        }

        # ---- build result structures ----
        alignment_results = self._build_alignment_results(alignment_acc)
        entropy_results = self._build_entropy_results(entropy_acc)
        magnitude_results = self._build_magnitude_results(alignment_acc)

        return {
            "three_digit_boundary_alignment": alignment_results,
            "digit_split_variability": entropy_results,
            "numeric_magnitude_consistency": magnitude_results,
            "operator_isolation_rate": operator_results,
        }

    # ------------------------------------------------------------------
    # Operator-isolation helpers (per-domain)
    # ------------------------------------------------------------------

    @staticmethod
    def _texts_by_language(
        tokenized_data: Dict[str, List[TokenizedData]]
    ) -> Dict[str, List[str]]:
        """Recover a ``{language: [text, ...]}`` view of an already-tokenized corpus.

        Every tokenizer sees the same texts, so one tokenizer's items describe the
        corpus. Used only to report the prose corpus size.
        """
        for items in tokenized_data.values():
            by_lang: Dict[str, List[str]] = defaultdict(list)
            for item in items:
                # text and text.strip(), matching every scored path and
                # corpus_size. A truthy check alone counted a whitespace-only
                # text into the published prose corpus size that no metric read.
                if item.text and item.text.strip():
                    by_lang[item.language].append(item.text)
            return dict(by_lang)
        return {}

    @staticmethod
    def _char_to_token_from_offsets(
        n_chars: int,
        offsets: Optional[List[Tuple[int, int]]],
        keep_later_token_on_overlap: bool = False,
    ) -> Optional[List[Optional[int]]]:
        """Map each source character index to the token covering it.

        Returns None when the tokenizer reported no offsets, so the caller can
        skip rather than guess. Offsets are exact; inferring the same
        correspondence from token surfaces is not, because a byte-level
        vocabulary does not store the source characters.

        *keep_later_token_on_overlap* decides which token keeps a character
        that two tokens claim. Ranges do overlap: a SentencePiece vocabulary
        emits the word-start marker as its own token, and HuggingFace reports a
        range for it that covers the first character of the following word.
        xlm-roberta-base encodes ``1234567`` as ``▁``, ``1234``, ``567`` with
        offsets (0,1), (0,4) and (4,7), so character 0 is claimed by both the
        marker and ``1234``. The marker produces no source character, so it
        belongs to ``1234``, which ``keep_later_token_on_overlap=True``
        reports. It is also the rule ``ASTBoundaryMetrics._map_from_offsets``
        applies.

        Every call site passes True: the digit metrics in ``compute()`` and in
        ``compute_per_text()``, and operator isolation in
        ``_accumulate_operators``. The parameter is kept and its default left at
        False, so a new caller that omits it gets the earlier token rather than
        the rule the three existing call sites use. Operator isolation kept the
        earlier token until that call site was switched. Switching it moved
        xlm-roberta's code domain, ``overall_isolation_rate`` 0.6947 to 0.6769 and
        ``overall_compound_preservation_rate`` 0.7223 to 0.8445 over 454693
        operators, and its math domain, 0.8788 to 0.8754 and 0.5167 to 0.6167
        over 297. The other eight benchmark tokenizers are unchanged to every
        digit.

        Those eight do report overlapping ranges, on a different character: when
        a byte-level vocabulary splits one multi-byte character across two
        tokens, both tokens report that character's range. Over 30 snippets of
        the benchmark code corpus, every character two of their tokens claimed
        was non-ASCII, against 241 of xlm-roberta's 244, 58 of which matched
        ``_OPERATOR_SPAN``. An operator is ASCII, so the rule only changes an
        operator's attribution in the marker case.
        """
        if not offsets:
            return None
        mapping: List[Optional[int]] = [None] * n_chars
        for tok_idx, span in enumerate(offsets):
            if not span:
                continue
            start, end = span
            if start < 0:
                start = 0
            if end > n_chars:
                end = n_chars
            if end <= start:
                continue
            if keep_later_token_on_overlap:
                # Every character of the range takes this token, so the range
                # is written in one slice assignment rather than one Python
                # loop iteration per character. Same list either way; the loop
                # cost 1.09 s against 0.50 s for the assignment over 2.93 M
                # characters of web text encoded with gpt2.
                mapping[start:end] = [tok_idx] * (end - start)
            else:
                for ci in range(start, end):
                    if mapping[ci] is None:
                        mapping[ci] = tok_idx
        return mapping

    @staticmethod
    def _iter_operator_scores(
        text: str,
        offsets: List[Tuple[int, int]],
        char_to_token: List[Optional[int]],
    ):
        """Score every operator in *text*, one 4-tuple per operator.

        Yields ``(category, isolated, is_compound, compound_preserved)``.
        *char_to_token* is ``_char_to_token_from_offsets(len(text), offsets,
        keep_later_token_on_overlap=True)``.

        ``_accumulate_operators`` and ``compute_per_text`` both call this, so
        the corpus path and the per-document path score a given text
        identically. They used to hold two copies of the body and a comment
        asking the next reader to keep them in step.

        Cost is proportional to the number of operators and to how many
        characters the tokens covering them span, not to the length of the
        text. The version this replaced built a dict from token index to a
        Python set of character indices over the whole document, which is one
        set allocation per token and one insertion per character, to answer a
        question asked only about the tokens covering an operator. Over
        8.4 M characters of web text the two together cost 10.67 s for gpt2
        against 1.57 s, for the same counts.
        """
        n_chars = len(text)
        for m in DigitBoundaryMetrics._OPERATOR_SPAN.finditer(text):
            op_str = m.group()
            category = DigitBoundaryMetrics._OPERATOR_TO_CATEGORY.get(op_str)
            if category is None:
                continue
            op_start = m.start()
            op_end = m.end()

            op_token_indices = set()
            for ci in range(op_start, op_end):
                ti = char_to_token[ci]
                if ti is not None:
                    op_token_indices.add(ti)
            if not op_token_indices:
                continue

            # Isolated = the covering tokens carry no non-whitespace character
            # from outside the operator span, i.e. the operator is not glued to
            # an operand.
            #
            # Whitespace is excluded from the test because these are source
            # coordinates. A token such as 'G=' covers the space before the
            # operator as well as the operator, and that token is still an
            # isolated operator: the space is not an operand. The
            # reconstruction-coordinate version this replaced got that for free
            # by stripping whitespace before matching.
            #
            # A character a token owns lies inside that token's own offset
            # span, so those spans are the only place to look. The equality
            # test against char_to_token is what makes it ownership rather than
            # coverage: where two tokens claim a character the later one takes
            # it, and the earlier one is not answerable for it.
            #
            # That test is defensive. It fires only when a token covering an
            # operator is itself overlapped by a later token, and over 27283
            # operators of FLORES, web text and the benchmark code corpus it
            # fired for none of gpt2, xlm-roberta-base, bert-base-uncased and
            # a SuperBPE vocabulary. It is kept because it is one comparison
            # and it makes this equivalent to the whole-document reverse map it
            # replaced for any offset shape, rather than only for the shapes
            # those four report.
            outside_non_ws = False
            for ti in op_token_indices:
                start, end = offsets[ti]
                if start < 0:
                    start = 0
                if end > n_chars:
                    end = n_chars
                for ci in range(start, end):
                    if op_start <= ci < op_end:
                        continue
                    if char_to_token[ci] != ti:
                        continue
                    if not text[ci].isspace():
                        outside_non_ws = True
                        break
                if outside_non_ws:
                    break

            is_compound = len(op_str) > 1
            yield (
                category,
                not outside_non_ws,
                is_compound,
                is_compound and len(op_token_indices) == 1,
            )

    def _accumulate_operators(
        self, tokenized_data: Dict[str, List[TokenizedData]]
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:
        """Accumulate operator isolation counts over one corpus.

        Returns tok -> lang -> category -> {isolated, total, compound_ok,
        compound_total}.
        """
        acc: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(
                lambda: {"isolated": 0, "total": 0, "compound_ok": 0, "compound_total": 0}
            ))
        )
        skipped_no_offsets = 0

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            tokenizer_obj = self.input_provider.get_tokenizer(tok_name)
            lang_groups = TokenizedDataProcessor.group_by_language(
                tokenized_data[tok_name]
            )

            for lang, data_list in lang_groups.items():
                for item in data_list:
                    if item.text is None or not item.text.strip():
                        continue
                    if not self._OPERATOR_SPAN.search(item.text):
                        continue

                    # Operators are located in the SOURCE and resolved to
                    # tokens through the encoder's own offsets.
                    #
                    # This used to run the operator regex over a reconstruction
                    # built by concatenating token strings, and every position
                    # was a reconstruction coordinate. Two ways that gave wrong
                    # answers. Special tokens the surface pattern does not
                    # recognise were spliced in as literal text, so a Mistral-form
                    # BOS token '<s>' contributed a '<' and a '>' that were
                    # counted as two operators: apertus reported an isolation
                    # rate of 6/8 = 0.750 on a sentence whose true rate is
                    # 6/6 = 1.000, for every text. And a superword token like
                    # 'G=G' reconstructs with a trailing space, which defeated
                    # the isolation subset test even though the token covers only
                    # '=' in the source.
                    #
                    # The SentencePiece word-start marker reports a range that
                    # covers the first character of the following word, so two
                    # tokens claim that character. See
                    # _char_to_token_from_offsets for why the later token is the
                    # right reading of it.
                    char_to_token = self._char_to_token_from_offsets(
                        len(item.text), item.offsets,
                        keep_later_token_on_overlap=True,
                    )
                    if char_to_token is None:
                        skipped_no_offsets += 1
                        continue

                    for category, isolated, is_compound, preserved in (
                        self._iter_operator_scores(
                            item.text, item.offsets, char_to_token
                        )
                    ):
                        # Indexed here rather than hoisted out of the loop, so
                        # a language whose only operator characters have no
                        # category does not gain an empty entry that the
                        # per-character version never created. '!' matches
                        # _OPERATOR_SPAN and is in no category, so a corpus can
                        # reach this loop and yield nothing.
                        cat_acc = acc[tok_name][lang][category]
                        cat_acc["total"] += 1
                        if isolated:
                            cat_acc["isolated"] += 1
                        # Compound preservation: multi-char operator -> 1 token
                        if is_compound:
                            cat_acc["compound_total"] += 1
                            if preserved:
                                cat_acc["compound_ok"] += 1

        if skipped_no_offsets:
            logger.warning(
                "Operator isolation skipped %d document(s) whose tokenizer "
                "reported no character offsets. Offsets are the only exact way "
                "to say which token covers an operator; the metric is measured "
                "on the remaining documents rather than estimated on these.",
                skipped_no_offsets,
            )

        return acc

    @classmethod
    def _merge_operator_accs(
        cls,
        domain_accs: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, int]]]]]
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:
        """Sum the per-domain accumulators into the pooled accumulator.

        The three domains do not share a language namespace: prose is keyed by
        FLORES code, code by programming language, math by the literal "math".
        Prose keys are kept bare so existing per-language consumers and the
        language-group filter keep matching FLORES codes. Code and math keys
        take the same form reconstruction fidelity publishes in ``per_domain``,
        ``code_<lang>`` and a bare ``math``, so one results file does not name
        the same thing two ways.

        The published key used to be ``code:python`` and ``math:math``, and the
        colon was load-bearing rather than cosmetic: a namespace no FLORES code
        can contain kept a code or math row from being summed into a prose one,
        and from being selected by ``_filter_operator_results``, which picks a
        language group with ``l in target_languages``. The underscore form has
        no such guarantee, since ``--input`` names languages after files and a
        corpus holding ``math.txt`` produces a prose language called ``math``.

        So the pairing is kept whole here and the collision is refused rather
        than relocated: two domains that would publish under one key abort,
        naming it. Reporting one number for both is the substitution this
        package refuses everywhere else, and it would be invisible in the
        output.
        """
        merged: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(
                lambda: {"isolated": 0, "total": 0, "compound_ok": 0, "compound_total": 0}
            ))
        )
        # published key -> the domain that claimed it, per tokenizer
        claimed: Dict[str, Dict[str, str]] = defaultdict(dict)
        for domain, acc in domain_accs.items():
            for tok_name, langs in acc.items():
                for lang, categories in langs.items():
                    key = published_language_key(domain, lang)
                    owner = claimed[tok_name].setdefault(key, domain)
                    if owner != domain:
                        raise ValueError(
                            f"The {owner!r} and {domain!r} domains would both "
                            f"publish operator counts under by_language[{key!r}] "
                            f"for {tok_name!r}. Summing them would report one "
                            "number for two corpora, and a language group "
                            "selecting that key would pull in the other domain. "
                            "Rename the offending language in the input corpus."
                        )
                    for category, counts in categories.items():
                        target = merged[tok_name][key][category]
                        for count_key in ("isolated", "total", "compound_ok", "compound_total"):
                            target[count_key] += counts[count_key]
        return merged


    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_alignment_results(
        self, acc: Dict[str, Dict[str, Dict[str, list]]]
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "per_tokenizer": {},
            "summary": {},
            "metadata": {
                "description": (
                    "How often the tokenizer's boundaries inside a number fall "
                    "at the three-digit positions comma grouping would use."
                ),
                # mean_f1 averages one F1 per number. count_unit is right:
                # the count really is digit spans.
                "aggregation": AGGREGATION_MEAN_OF_RATIOS,
                "count_unit": "digit spans",
            },
        }

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {
                "by_digit_length": {},
                "by_bucket": {"short": {}, "long": {}},
                "overall": {},
            }

            all_f1: List[float] = []
            all_prec: List[float] = []
            all_rec: List[float] = []
            all_uniform: List[float] = []
            all_single: List[int] = []
            total_numbers = 0
            languages_seen: set = set()

            for lang in sorted(acc.get(tok_name, {})):
                lang_all_f1: List[float] = []
                lang_all_prec: List[float] = []
                lang_all_rec: List[float] = []
                lang_short_f1: List[float] = []
                lang_long_f1: List[float] = []

                for dl_str in sorted(acc[tok_name][lang]):
                    items = acc[tok_name][lang][dl_str]
                    if not items:
                        continue

                    is_short = self._is_short_bucket(dl_str)

                    f1s = [it["f1"] for it in items]
                    precs = [it["precision"] for it in items]
                    recs = [it["recall"] for it in items]
                    uniforms = [it["uniform_chunk"] for it in items]
                    singles = [it["single_token"] for it in items]

                    if dl_str not in tok_data["by_digit_length"]:
                        tok_data["by_digit_length"][dl_str] = {}

                    tok_data["by_digit_length"][dl_str][lang] = {
                        "mean_f1": float(np.mean(f1s)),
                        "mean_precision": float(np.mean(precs)),
                        "mean_recall": float(np.mean(recs)),
                        "mean_uniform_chunk": float(np.mean(uniforms)),
                        "single_token_frac": float(np.mean(singles)),
                        "count": len(items),
                    }

                    lang_all_f1.extend(f1s)
                    lang_all_prec.extend(precs)
                    lang_all_rec.extend(recs)

                    if is_short:
                        lang_short_f1.extend(f1s)
                    else:
                        lang_long_f1.extend(f1s)

                # Per-language bucket aggregation
                if lang_short_f1:
                    tok_data["by_bucket"]["short"][lang] = {
                        "mean_f1": float(np.mean(lang_short_f1)),
                        "count": len(lang_short_f1),
                    }
                if lang_long_f1:
                    tok_data["by_bucket"]["long"][lang] = {
                        "mean_f1": float(np.mean(lang_long_f1)),
                        "count": len(lang_long_f1),
                    }

                # Overall per-language
                if lang_all_f1:
                    tok_data["overall"][lang] = {
                        "mean_f1": float(np.mean(lang_all_f1)),
                        "mean_precision": float(np.mean(lang_all_prec)),
                        "mean_recall": float(np.mean(lang_all_rec)),
                        "count": len(lang_all_f1),
                    }
                    all_f1.extend(lang_all_f1)
                    all_prec.extend(lang_all_prec)
                    all_rec.extend(lang_all_rec)
                    all_uniform.extend(
                        it["uniform_chunk"]
                        for dl_items in acc[tok_name][lang].values()
                        for it in dl_items
                    )
                    all_single.extend(
                        it["single_token"]
                        for dl_items in acc[tok_name][lang].values()
                        for it in dl_items
                    )
                    total_numbers += len(lang_all_f1)
                    languages_seen.add(lang)

            results["per_tokenizer"][tok_name] = tok_data

            # Summary
            if all_f1:
                results["summary"][tok_name] = {
                    "avg_f1": float(np.mean(all_f1)),
                    "avg_precision": float(np.mean(all_prec)),
                    "avg_recall": float(np.mean(all_rec)),
                    "avg_uniform_chunk": float(np.mean(all_uniform)),
                    "single_token_frac": float(np.mean(all_single)),
                    "numbers_analyzed": total_numbers,
                    "languages_analyzed": len(languages_seen),
                }

        return results

    def _build_entropy_results(
        self, acc: Dict[str, Dict[str, Dict[str, list]]]
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {"per_tokenizer": {}, "summary": {}}

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {
                "by_digit_length": {},
                "by_bucket": {"short": {}, "long": {}},
                "overall": {},
            }

            # Collect raw patterns across all languages for pooled summary
            pooled_long_patterns: List[tuple] = []
            pooled_short_patterns: List[tuple] = []
            total_numbers = 0
            languages_seen: set = set()

            for lang in sorted(acc.get(tok_name, {})):
                lang_short_patterns: List[tuple] = []
                lang_long_patterns: List[tuple] = []

                for dl_str in sorted(acc[tok_name][lang]):
                    patterns = acc[tok_name][lang][dl_str]
                    if not patterns:
                        continue

                    is_short = self._is_short_bucket(dl_str)

                    stats = self._compute_pattern_entropy(patterns)

                    if dl_str not in tok_data["by_digit_length"]:
                        tok_data["by_digit_length"][dl_str] = {}

                    tok_data["by_digit_length"][dl_str][lang] = stats

                    if is_short:
                        lang_short_patterns.extend(patterns)
                    else:
                        lang_long_patterns.extend(patterns)

                # Per-language bucket aggregation
                if lang_short_patterns:
                    tok_data["by_bucket"]["short"][lang] = self._compute_pattern_entropy(
                        lang_short_patterns
                    )
                if lang_long_patterns:
                    tok_data["by_bucket"]["long"][lang] = self._compute_pattern_entropy(
                        lang_long_patterns
                    )

                # Overall per-language
                all_lang_patterns = lang_short_patterns + lang_long_patterns
                if all_lang_patterns:
                    tok_data["overall"][lang] = self._compute_pattern_entropy(
                        all_lang_patterns
                    )
                    total_numbers += len(all_lang_patterns)
                    languages_seen.add(lang)

                # Accumulate for pooled summary
                pooled_short_patterns.extend(lang_short_patterns)
                pooled_long_patterns.extend(lang_long_patterns)

            results["per_tokenizer"][tok_name] = tok_data

            if languages_seen:
                long_stats = self._compute_pattern_entropy(pooled_long_patterns)
                short_stats = self._compute_pattern_entropy(pooled_short_patterns)
                # _compute_pattern_entropy returns 0.0 for an empty list, with
                # count 0 beside it to say so. This summary drops the counts,
                # so a bucket that held no number would publish 0.0, which
                # reads as perfectly consistent chunking. Publish None instead,
                # and carry the counts so each entropy is interpretable where
                # it is read.
                results["summary"][tok_name] = {
                    "entropy_long": long_stats["entropy"] if long_stats["count"] else None,
                    "entropy_short": short_stats["entropy"] if short_stats["count"] else None,
                    "long_numbers_analyzed": long_stats["count"],
                    "short_numbers_analyzed": short_stats["count"],
                    "numbers_analyzed": total_numbers,
                    "languages_analyzed": len(languages_seen),
                }

        return results

    # ------------------------------------------------------------------
    # Fertility scaling (for numeric magnitude consistency)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fertility_scaling(
        bucket_records: Dict[str, List[Dict[str, float]]],
    ) -> Dict[str, Any]:
        """Compute scaling statistics across digit-length buckets.

        Given ``{bucket_str: [{"fertility_per_digit": ..., "num_digits": ...}, ...]}``
        (one record per number), returns per-bucket stats plus overall scaling
        indicators (Spearman rho, coefficient of variation of mean fertility,
        and a linear fit).

        Buckets '1' through '9' hold numbers of exactly that many digits, so
        their mean digit length equals the bucket label. The '10+' bucket
        holds numbers of every length from 10 digits up, so its mean digit
        length is whatever the corpus actually contains. The linear fit below
        uses each bucket's own mean digit length and its own mean of the
        actual per-number token counts, so it stays exact for '10+' instead of
        assuming every number in that bucket is 10 digits long.

        spearman_rho keeps the coarser bucket-order representation ('10+'
        pinned at digit length 10): it is a rank correlation over bucket
        order, not a fit to a specific x value, so the same simplification
        that breaks the linear fit does not apply to it.

        The fit rests on at most 10 bucket points ('1' through '9' plus
        '10+'), so its slope and R-squared are coarse.
        """
        per_bucket: Dict[str, Dict[str, float]] = {}
        digit_lengths: List[float] = []       # bucket-order x, for spearman_rho ('10+' pinned at 10.0)
        mean_fertilities: List[float] = []
        mean_digit_lengths: List[float] = []  # true mean num_digits per bucket, for the linear fit
        mean_token_counts: List[float] = []   # true mean per-number token count per bucket, for the linear fit

        for bucket_str in sorted(bucket_records, key=lambda x: (len(x), x)):
            records = bucket_records[bucket_str]
            if not records:
                continue
            fertilities = [r["fertility_per_digit"] for r in records]
            num_digits_list = [r["num_digits"] for r in records]
            # Each number's own token count, recovered exactly from that
            # number's own fertility and digit count: fertility_per_digit is
            # num_tokens / num_digits for that specific number, so multiplying
            # back by that same number's num_digits returns its num_tokens.
            token_counts = [
                r["fertility_per_digit"] * r["num_digits"] for r in records
            ]

            m = float(np.mean(fertilities))
            s = float(np.std(fertilities))
            mean_digit_length = float(np.mean(num_digits_list))
            per_bucket[bucket_str] = {
                "mean_fertility": m,
                "std_fertility": s,
                "count": len(records),
                "mean_digit_length": mean_digit_length,
            }

            # For spearman_rho, use the bucket-order digit length (10+ pinned
            # at 10), unchanged from before.
            if bucket_str.endswith("+"):
                dl = 10.0
            else:
                dl = float(bucket_str)
            digit_lengths.append(dl)
            mean_fertilities.append(m)

            mean_digit_lengths.append(mean_digit_length)
            mean_token_counts.append(float(np.mean(token_counts)))

        result: Dict[str, Any] = {"per_bucket": per_bucket}

        if len(digit_lengths) < 2:
            result["spearman_rho"] = None
            result["spearman_p"] = None
            # None for the same reason as its three siblings in this branch:
            # a dispersion over fewer than two buckets is undefined, and 0.0
            # is a real measurement meaning every bucket has the same mean
            # fertility.
            result["cv_of_mean_fertility"] = None
            result["linear_fit"] = None
            return result

        dl_arr = np.array(digit_lengths)
        mf_arr = np.array(mean_fertilities)

        # Spearman rank correlation. Constant input leaves it undefined, which
        # is what scipy reports as NaN; publishing rho 0.0 with p 1.0 asserted
        # a definite absence of monotonic relationship from a statistic that
        # was never computed. cv_of_mean_fertility still records 0.0 for a
        # genuinely constant non-zero fertility, so the finding is not lost.
        if np.std(mf_arr) == 0.0:
            result["spearman_rho"] = None
            result["spearman_p"] = None
        else:
            rho, p_val = spearmanr(dl_arr, mf_arr)
            # None, not NaN, for anything scipy could not compute. Two buckets
            # give a defined rho and an undefined p, because a rank correlation
            # over two points has no significance level; a constant x gives
            # neither. NaN reaching the results file makes it invalid JSON, and
            # null is what the rest of the pipeline means by not computed.
            result["spearman_rho"] = float(rho) if np.isfinite(rho) else None
            result["spearman_p"] = float(p_val) if np.isfinite(p_val) else None

        # Coefficient of variation of mean fertility across buckets
        overall_mean = float(np.mean(mf_arr))
        overall_std = float(np.std(mf_arr))
        # None, not 0.0, when the mean is zero: a coefficient of variation of
        # 0.0 is a real measurement meaning every bucket has the same mean
        # fertility, so it must not double as "undefined". Same convention as
        # spearman_rho above.
        result["cv_of_mean_fertility"] = (
            overall_std / overall_mean if overall_mean > 0 else None
        )

        # Linear fit: mean_tokens = slope * num_digits + intercept.
        # x is each bucket's own mean digit length; y is each bucket's own
        # mean of the actual per-number token counts. The previous
        # implementation used mean_fertility * a representative digit length
        # in place of y, which is only exact when every number in the bucket
        # has the same length, and is false for '10+'.
        mdl_arr = np.array(mean_digit_lengths)
        mtc_arr = np.array(mean_token_counts)
        coeffs = np.polyfit(mdl_arr, mtc_arr, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])

        # R^2
        predicted = slope * mdl_arr + intercept
        ss_res = float(np.sum((mtc_arr - predicted) ** 2))
        ss_tot = float(np.sum((mtc_arr - np.mean(mtc_arr)) ** 2))
        # None, not 1.0, when the total sum of squares is zero. That happens
        # when every bucket has the same mean token count, which leaves R^2
        # undefined; publishing 1.0 reported a perfect fit for a fit that was
        # never determined.
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

        result["linear_fit"] = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
        }

        return result

    # ------------------------------------------------------------------
    # Magnitude result builder
    # ------------------------------------------------------------------

    def _build_magnitude_results(
        self, alignment_acc: Dict[str, Dict[str, Dict[str, list]]]
    ) -> Dict[str, Any]:
        """Build the ``numeric_magnitude_consistency`` result dict.

        Derives fertility values from the ``fertility_per_digit`` field
        stored in each *alignment_acc* entry, eliminating the need for a
        separate magnitude accumulator.
        """
        results: Dict[str, Any] = {
            "per_tokenizer": {},
            "summary": {},
            # Single-sourced in magnitude_metadata so main.py's grouped
            # filter, which drops the corpus-level scaling fit, can publish
            # the description that matches its block.
            "metadata": magnitude_metadata(),
        }

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {
                "by_digit_length": {},
                "overall": {},
                "scaling": {},
            }

            all_fertilities: List[float] = []
            # Collect per-number records (fertility_per_digit + num_digits)
            # across all languages, for the global scaling fit. Records, not
            # bare fertility floats, so that fit can use each bucket's own
            # true mean digit length rather than a fixed stand-in for the
            # open-ended '10+' bucket.
            global_bucket_records: Dict[str, List[Dict[str, float]]] = defaultdict(list)
            languages_seen: set = set()

            for lang in sorted(alignment_acc.get(tok_name, {})):
                lang_fertilities: List[float] = []

                for dl_str in sorted(alignment_acc[tok_name][lang]):
                    items = alignment_acc[tok_name][lang][dl_str]
                    if not items:
                        continue

                    values = [it["fertility_per_digit"] for it in items]

                    if dl_str not in tok_data["by_digit_length"]:
                        tok_data["by_digit_length"][dl_str] = {}

                    tok_data["by_digit_length"][dl_str][lang] = {
                        "mean_fertility": float(np.mean(values)),
                        "std_fertility": float(np.std(values)),
                        "count": len(values),
                    }

                    lang_fertilities.extend(values)
                    global_bucket_records[dl_str].extend(
                        {
                            "fertility_per_digit": it["fertility_per_digit"],
                            "num_digits": it["num_digits"],
                        }
                        for it in items
                    )

                if lang_fertilities:
                    tok_data["overall"][lang] = {
                        "mean_fertility": float(np.mean(lang_fertilities)),
                        "count": len(lang_fertilities),
                    }
                    all_fertilities.extend(lang_fertilities)
                    languages_seen.add(lang)

            # Scaling stats (across all languages pooled)
            tok_data["scaling"] = self._compute_fertility_scaling(
                global_bucket_records
            )

            results["per_tokenizer"][tok_name] = tok_data

            # Summary
            if all_fertilities:
                scaling = tok_data["scaling"]
                summary: Dict[str, Any] = {
                    "avg_fertility": float(np.mean(all_fertilities)),
                    # No 0.0 default. The sibling on the next line has none,
                    # _compute_fertility_scaling always sets this key on both
                    # of its paths, and the key now holds None when the
                    # dispersion is undefined, so a default could only ever
                    # fabricate a zero.
                    "cv_of_mean_fertility": scaling.get("cv_of_mean_fertility"),
                    "spearman_rho": scaling.get("spearman_rho"),
                    "numbers_analyzed": len(all_fertilities),
                    "languages_analyzed": len(languages_seen),
                }
                if scaling.get("linear_fit"):
                    summary["linear_r_squared"] = scaling["linear_fit"]["r_squared"]
                    summary["linear_slope"] = scaling["linear_fit"]["slope"]
                results["summary"][tok_name] = summary

        return results

    # ------------------------------------------------------------------
    # Operator result builder
    # ------------------------------------------------------------------

    def _build_operator_results(
        self, acc: Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
        include_code_math: bool = True,
    ) -> Dict[str, Any]:
        """Build the ``operator_isolation_rate`` result dict.

        *include_code_math* only reaches the published metadata, which has to
        describe the domains this run actually has.
        """
        results: Dict[str, Any] = {
            "per_tokenizer": {},
            "summary": {},
            # Conditional for the same reason reconstruction fidelity's
            # per_domain string is: a grouped run has no code or math domain,
            # and a fixed string described domains the block cannot hold.
            # Single-sourced in operator_metadata so main.py's grouped filter
            # publishes the same contract rather than a copy that can drift.
            "metadata": operator_metadata(include_code_math),
        }

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {
                "by_category": {},
                "by_language": {},
            }

            # Global totals across all languages and categories
            total_isolated = 0
            total_ops = 0
            total_compound_ok = 0
            total_compound = 0

            # Aggregate by category (across all languages)
            category_totals: Dict[str, Dict[str, int]] = defaultdict(
                lambda: {"isolated": 0, "total": 0, "compound_ok": 0, "compound_total": 0}
            )

            for lang in sorted(acc.get(tok_name, {})):
                lang_data: Dict[str, Any] = {"by_category": {}}
                lang_isolated = 0
                lang_total = 0
                lang_compound_ok = 0
                lang_compound_total = 0

                for category in sorted(acc[tok_name][lang]):
                    cat_data = acc[tok_name][lang][category]
                    t = cat_data["total"]
                    iso = cat_data["isolated"]
                    cok = cat_data["compound_ok"]
                    ctot = cat_data["compound_total"]

                    lang_data["by_category"][category] = {
                        # None, not 0.0: a category with no operator of that
                        # kind was not measured, and 0.0 reads as a tokenizer
                        # that isolated none of them.
                        "isolation_rate": iso / t if t > 0 else None,
                        "compound_preservation_rate": cok / ctot if ctot > 0 else None,
                        "total": t,
                        "compound_total": ctot,
                        # raw counts so a filtered subset can be re-aggregated exactly
                        "isolated": iso,
                        "compound_ok": cok,
                    }

                    lang_isolated += iso
                    lang_total += t
                    lang_compound_ok += cok
                    lang_compound_total += ctot

                    category_totals[category]["isolated"] += iso
                    category_totals[category]["total"] += t
                    category_totals[category]["compound_ok"] += cok
                    category_totals[category]["compound_total"] += ctot

                if lang_total > 0:
                    lang_data["isolation_rate"] = lang_isolated / lang_total
                    lang_data["compound_preservation_rate"] = (
                        lang_compound_ok / lang_compound_total
                        if lang_compound_total > 0 else None
                    )
                    lang_data["total"] = lang_total
                    # count is the schema-wide name for how many items of the
                    # metric's count_unit this entry covers. The unit here is
                    # operator occurrences, so it repeats total.
                    lang_data["count"] = lang_total
                    # raw counts so a filtered subset can be re-aggregated exactly
                    lang_data["isolated"] = lang_isolated
                    lang_data["compound_total"] = lang_compound_total
                    lang_data["compound_ok"] = lang_compound_ok
                    tok_data["by_language"][lang] = lang_data

                total_isolated += lang_isolated
                total_ops += lang_total
                total_compound_ok += lang_compound_ok
                total_compound += lang_compound_total

            # Per-category aggregation
            for category in sorted(category_totals):
                ct = category_totals[category]
                t = ct["total"]
                tok_data["by_category"][category] = {
                    "isolation_rate": ct["isolated"] / t if t > 0 else None,
                    "compound_preservation_rate": (
                        ct["compound_ok"] / ct["compound_total"]
                        if ct["compound_total"] > 0 else None
                    ),
                    "total": t,
                    "compound_total": ct["compound_total"],
                }

            results["per_tokenizer"][tok_name] = tok_data

            # Summary
            if total_ops > 0:
                results["summary"][tok_name] = {
                    "overall_isolation_rate": total_isolated / total_ops,
                    "overall_compound_preservation_rate": (
                        total_compound_ok / total_compound
                        if total_compound > 0 else None
                    ),
                    "total_operators": total_ops,
                    "total_compound_operators": total_compound,
                }

        return results

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print digit boundary alignment and digit split variability results."""
        # ---- Alignment ----
        align = results.get("three_digit_boundary_alignment")
        if align:
            print("\n" + "=" * 60)
            print("THREE-DIGIT BOUNDARY ALIGNMENT RESULTS")
            print("=" * 60)

            if "summary" in align:
                print("\nSUMMARY STATISTICS")
                print("-" * 40)
                for tok_name in self.tokenizer_names:
                    if tok_name in align["summary"]:
                        s = align["summary"][tok_name]
                        print(f"{tok_name}:")
                        print(f"  {'Avg F1':20}: {s['avg_f1']:.3f}")
                        print(f"  {'Avg Precision':20}: {s['avg_precision']:.3f}")
                        print(f"  {'Avg Recall':20}: {s['avg_recall']:.3f}")
                        print(f"  {'Uniform Chunk':20}: {s['avg_uniform_chunk']:.3f}")
                        print(f"  {'Single-Token Frac':20}: {s['single_token_frac']:.3f}")
                        print(f"  {'Numbers Analyzed':20}: {s['numbers_analyzed']:,}")
                        print(f"  {'Languages':20}: {s['languages_analyzed']}")

            if "per_tokenizer" in align:
                print("\nDETAILED (by digit length)")
                print("-" * 60)
                for tok_name in self.tokenizer_names:
                    tok = align["per_tokenizer"].get(tok_name, {})
                    by_dl = tok.get("by_digit_length", {})
                    if not by_dl:
                        continue
                    print(f"\n{tok_name}:")
                    for dl in sorted(by_dl, key=lambda x: (len(x), x)):
                        lang_data = by_dl[dl]
                        for lang, d in sorted(lang_data.items()):
                            print(
                                f"  L={dl:>3} {lang:12}: "
                                f"F1={d['mean_f1']:.3f}  "
                                f"P={d['mean_precision']:.3f}  "
                                f"R={d['mean_recall']:.3f}  "
                                f"uniform={d['mean_uniform_chunk']:.2f}  "
                                f"n={d['count']}"
                            )

        # ---- Entropy ----
        ent = results.get("digit_split_variability")
        if ent:
            print("\n" + "=" * 60)
            print("DIGIT SPLIT VARIABILITY RESULTS")
            print("=" * 60)

            if "summary" in ent:
                print("\nSUMMARY STATISTICS (pooled across languages)")
                print("-" * 40)
                for tok_name in self.tokenizer_names:
                    if tok_name in ent["summary"]:
                        s = ent["summary"][tok_name]
                        print(f"{tok_name}:")
                        print(f"  {'Entropy (long)':25}: {format_optional(s['entropy_long'], '.3f')} bits")
                        print(f"  {'Entropy (short)':25}: {format_optional(s['entropy_short'], '.3f')} bits")
                        print(f"  {'Numbers Analyzed':25}: {s['numbers_analyzed']:,}")
                        print(f"  {'Languages':25}: {s['languages_analyzed']}")

            if "per_tokenizer" in ent:
                print("\nDETAILED (by digit length)")
                print("-" * 60)
                for tok_name in self.tokenizer_names:
                    tok = ent["per_tokenizer"].get(tok_name, {})
                    by_dl = tok.get("by_digit_length", {})
                    if not by_dl:
                        continue
                    print(f"\n{tok_name}:")
                    for dl in sorted(by_dl, key=lambda x: (len(x), x)):
                        lang_data = by_dl[dl]
                        for lang, d in sorted(lang_data.items()):
                            dom = d.get("dominant_pattern", ())
                            print(
                                f"  L={dl:>3} {lang:12}: "
                                f"H={d['entropy']:.3f} bits  "
                                f"K={d['num_patterns']}  "
                                f"dom={dom} ({d['dominant_pattern_freq']:.0%})  "
                                f"n={d['count']}"
                            )

        # ---- Magnitude Consistency ----
        mag = results.get("numeric_magnitude_consistency")
        if mag:
            print("\n" + "=" * 60)
            print("NUMERIC MAGNITUDE CONSISTENCY RESULTS")
            print("=" * 60)

            if "summary" in mag:
                print("\nSUMMARY STATISTICS")
                print("-" * 40)
                for tok_name in self.tokenizer_names:
                    if tok_name in mag["summary"]:
                        s = mag["summary"][tok_name]
                        print(f"{tok_name}:")
                        print(f"  {'Avg Fertility':25}: {s['avg_fertility']:.3f}")
                        print(f"  {'CV of Mean Fertility':25}: "
                              f"{format_optional(s['cv_of_mean_fertility'], '.3f')}")
                        rho = s.get('spearman_rho')
                        print(f"  {'Spearman rho':25}: {format_optional(rho)}")
                        if 'linear_r_squared' in s:
                            print(f"  {'Linear R^2':25}: {format_optional(s['linear_r_squared'], '.3f')}")
                            print(f"  {'Linear Slope':25}: {s['linear_slope']:.3f}")
                        print(f"  {'Numbers Analyzed':25}: {s['numbers_analyzed']:,}")
                        print(f"  {'Languages':25}: {s['languages_analyzed']}")

            if "per_tokenizer" in mag:
                print("\nDETAILED (by digit length)")
                print("-" * 60)
                for tok_name in self.tokenizer_names:
                    tok = mag["per_tokenizer"].get(tok_name, {})
                    by_dl = tok.get("by_digit_length", {})
                    if not by_dl:
                        continue
                    print(f"\n{tok_name}:")
                    for dl in sorted(by_dl, key=lambda x: (len(x), x)):
                        lang_data = by_dl[dl]
                        for lang, d in sorted(lang_data.items()):
                            print(
                                f"  L={dl:>3} {lang:12}: "
                                f"fert={d['mean_fertility']:.3f} "
                                f"(std={d['std_fertility']:.3f})  "
                                f"n={d['count']}"
                            )

        # ---- Operator Isolation ----
        ops = results.get("operator_isolation_rate")
        if ops:
            print("\n" + "=" * 60)
            print("OPERATOR ISOLATION RATE RESULTS")
            print("=" * 60)

            if "summary" in ops:
                print("\nSUMMARY STATISTICS")
                print("-" * 40)
                for tok_name in self.tokenizer_names:
                    if tok_name in ops["summary"]:
                        s = ops["summary"][tok_name]
                        print(f"{tok_name}:")
                        # overall_compound_preservation_rate is None when the
                        # corpus held no compound operator, so there was no rate
                        # to compute (see _build_operator_results).
                        print(f"  {'Isolation Rate':30}: {format_optional(s['overall_isolation_rate'])}")
                        print(f"  {'Compound Preservation':30}: {format_optional(s['overall_compound_preservation_rate'])}")
                        print(f"  {'Total Operators':30}: {s['total_operators']:,}")
                        print(f"  {'Total Compound Ops':30}: {s['total_compound_operators']:,}")

            if "per_tokenizer" in ops:
                print("\nBY CATEGORY")
                print("-" * 60)
                for tok_name in self.tokenizer_names:
                    tok = ops["per_tokenizer"].get(tok_name, {})
                    by_cat = tok.get("by_category", {})
                    if not by_cat:
                        continue
                    print(f"\n{tok_name}:")
                    for cat, d in sorted(by_cat.items()):
                        # Both rates are None for a category the corpus had no
                        # operator of, in the compound case even when it had
                        # simple ones.
                        cpr = d.get('compound_preservation_rate')
                        ct = d.get('compound_total', 0)
                        print(
                            f"  {cat:20}: "
                            f"isolation={format_optional(d['isolation_rate'])}  "
                            f"compound={format_optional(cpr)} ({ct} compound)  "
                            f"n={d['total']}"
                        )

            print("\n" + "=" * 60)

    # ------------------------------------------------------------------
    # Per-text entry point (independent of compute()'s aggregator pipeline)
    # ------------------------------------------------------------------

    def compute_per_text(
        self,
        tokenizer_obj: Any,
        text: str,
    ) -> Dict[str, Any]:
        """Compute digit boundary alignment metrics for ONE text under ONE
        tokenizer, returning a per-document summary.

        Mirrors the per-text body inside ``compute()`` for the digit metrics but
        does not pool across a corpus. Used by per-example correlation analysis.
        The standard ``compute()`` workflow is unaffected; this method snapshots
        and restores ``self._char_decode_table``, ``self._special_tokens`` and
        ``self._subword_markers`` so calling it does not mutate aggregator
        state.

        *tokenizer_obj* has to report character offsets, either through a
        wrapper's ``encode_with_offsets`` or through a ``tokenizers.Encoding``
        with an ``offsets`` attribute. One that does not raises ``ValueError``
        naming it, rather than returning a row: the shortfall applies to every
        text the caller would pass, so it is not a per-row condition.

        ``operator_isolation_rate`` and ``compound_operator_preserved_rate``
        are measured on the same character offsets as the digit metrics, with
        the body ``_accumulate_operators`` uses, so this method and
        ``compute()`` score a given text identically.

        Returns a dict with keys: ``n_digit_spans``, ``mean_digit_f1``,
        ``mean_fertility_per_digit``, ``single_token_number_rate``,
        ``uniform_chunk_rate``, ``n_operators``, ``operator_isolation_rate``,
        ``n_compound_operators``, ``compound_operator_preserved_rate``,
        ``n_tokens`` (the token count produced for the whole text, useful
        as a regression covariate), and ``parse_status``.
        NaN is returned for ratios with empty denominators (e.g. no digit
        spans in the text).

        ``parse_status`` says which of four paths produced the row, so a
        caller joining many rows can tell an unmeasured text from a measured
        one: ``"ok"``, ``"empty"`` for an empty text, ``"no_digits_or_operators"``
        when the text contains neither, and ``"encode_error:<ExceptionName>"``
        when the tokenizer raised. Every status other than ``"ok"`` comes with
        zero span and operator counts and NaN rates;
        ``"no_digits_or_operators"`` still reports the text's real
        ``n_tokens``, the other two report 0.
        """
        empty = {
            "n_digit_spans": 0,
            "mean_digit_f1": float("nan"),
            "mean_fertility_per_digit": float("nan"),
            "single_token_number_rate": float("nan"),
            "uniform_chunk_rate": float("nan"),
            "n_operators": 0,
            "operator_isolation_rate": float("nan"),
            "n_compound_operators": 0,
            "compound_operator_preserved_rate": float("nan"),
            "n_tokens": 0,
            "parse_status": "empty",
        }

        if not text or not text.strip():
            return empty

        # ---- Encode and build char→token map (mirrors compute() body) ----
        # The digit metrics are measured against the encoder's own
        # character offsets, so they are required rather than optional.
        try:
            enc_offsets: Optional[List[Tuple[int, int]]]
            if hasattr(tokenizer_obj, "encode_with_offsets"):
                ids_raw, enc_offsets = tokenizer_obj.encode_with_offsets(text)
                token_ids = list(ids_raw)
            else:
                try:
                    token_ids_raw = tokenizer_obj.encode(text, add_special_tokens=False)
                except TypeError:
                    token_ids_raw = tokenizer_obj.encode(text)
                token_ids = (
                    list(token_ids_raw.ids)
                    if hasattr(token_ids_raw, "ids")
                    else list(token_ids_raw)
                )
                enc_offsets = (
                    [tuple(pair) for pair in token_ids_raw.offsets]
                    if hasattr(token_ids_raw, "offsets")
                    else None
                )
        except Exception as e:
            out = dict(empty)
            out["parse_status"] = f"encode_error:{type(e).__name__}"
            return out

        # Source character → token index. See the comment in compute() for
        # why the reconstruction this replaced was wrong and why the
        # word-start-space trim the code metrics apply is not needed here.
        source_char_to_token = self._char_to_token_from_offsets(
            len(text), enc_offsets, keep_later_token_on_overlap=True
        )
        if source_char_to_token is None:
            named = (
                tokenizer_obj.get_name()
                if hasattr(tokenizer_obj, "get_name")
                else getattr(tokenizer_obj, "name_or_path", None)
                or type(tokenizer_obj).__name__
            )
            raise ValueError(
                "The digit metrics need character offsets, and tokenizer "
                f"{named!r} reported none. Offsets are the only exact way "
                "to say which token covers which digit. The shortfall "
                "applies to every text the caller would pass, so it is "
                "raised rather than returned as a row: pass a tokenizer "
                "whose encode_with_offsets() returns offsets, or one whose "
                "encode() returns a tokenizers.Encoding."
            )

        # Match compute() early-exit semantics: both checks on the original text.
        has_digits = bool(self._DIGIT_SPAN.search(text))
        has_operators = bool(self._OPERATOR_SPAN.search(text))

        if not has_digits and not has_operators:
            out = dict(empty)
            out["n_tokens"] = len(token_ids)
            out["parse_status"] = "no_digits_or_operators"
            return out

        # ---- Per-digit-span scoring ----
        digit_f1s: List[float] = []
        digit_fertility: List[float] = []
        single_token_flags: List[float] = []
        uniform_chunk_flags: List[float] = []

        if has_digits:
            for src_start, src_end, digit_str in self._find_number_spans(text):
                num_digits = len(digit_str)
                if num_digits == 0:
                    continue
                if any(
                    source_char_to_token[i] is None
                    for i in range(src_start, src_end)
                ):
                    # A digit no token covers: the number was not measured.
                    continue

                boundaries = self._get_digit_span_boundaries(
                    source_char_to_token, src_start, src_end,
                )
                if boundaries is None:
                    continue

                actual = set(boundaries)
                ideal = self._ideal_boundaries(num_digits)
                scores = self._score_boundaries(actual, ideal)

                bnd_list = sorted(boundaries)
                chunk_lengths: List[int] = []
                prev = 0
                for b in bnd_list:
                    chunk_lengths.append(b - prev)
                    prev = b
                chunk_lengths.append(num_digits - prev)
                uniform_chunk = 1.0 if len(set(chunk_lengths)) <= 1 else 0.0
                single_token = 1.0 if len(bnd_list) == 0 else 0.0
                num_tokens = len(bnd_list) + 1
                fertility_per_digit = num_tokens / num_digits

                digit_f1s.append(scores["f1"])
                digit_fertility.append(fertility_per_digit)
                single_token_flags.append(single_token)
                uniform_chunk_flags.append(uniform_chunk)

        # ---- Per-operator scoring ----
        isolated_flags: List[float] = []
        compound_total = 0
        compound_ok = 0

        if has_operators:
            # Operators are located in the SOURCE and resolved to tokens
            # through the same offsets the digit scoring above used, by the
            # same helper ``_accumulate_operators`` calls, so this method and
            # ``compute()`` score a given text identically.
            #
            # This used to run the operator regex over a reconstruction
            # built by concatenating cleaned token strings, which
            # ``compute()`` stopped doing when ``_accumulate_operators``
            # moved onto the offsets, so the two disagreed. Two ways the
            # reconstruction gave wrong answers. A special token the surface
            # pattern does not recognise was spliced in as literal text, so
            # a Mistral-form BOS token '<s>' contributed a '<' and a '>'
            # that were counted as two operators. And a superword token such
            # as 'G=G' reconstructs with a trailing space, which defeated the
            # isolation test even though the token covers only '=' in the
            # source.
            for _category, isolated, is_compound, preserved in (
                self._iter_operator_scores(text, enc_offsets, source_char_to_token)
            ):
                isolated_flags.append(1.0 if isolated else 0.0)
                if is_compound:
                    compound_total += 1
                    if preserved:
                        compound_ok += 1

        n_digit_spans = len(digit_f1s)
        n_operators = len(isolated_flags)
        return {
            "n_digit_spans": n_digit_spans,
            "mean_digit_f1": float(np.mean(digit_f1s)) if n_digit_spans else float("nan"),
            "mean_fertility_per_digit": (
                float(np.mean(digit_fertility)) if n_digit_spans else float("nan")
            ),
            "single_token_number_rate": (
                float(np.mean(single_token_flags)) if n_digit_spans else float("nan")
            ),
            "uniform_chunk_rate": (
                float(np.mean(uniform_chunk_flags)) if n_digit_spans else float("nan")
            ),
            "n_operators": n_operators,
            "operator_isolation_rate": (
                float(np.mean(isolated_flags)) if n_operators else float("nan")
            ),
            "n_compound_operators": compound_total,
            "compound_operator_preserved_rate": (
                (compound_ok / compound_total) if compound_total else float("nan")
            ),
            "n_tokens": len(token_ids),
            "parse_status": "ok",
        }
