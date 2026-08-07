"""Fold metrics that measure the same thing into the metric that owns it.

The 1.0 audit computed Spearman rank correlations across 37 tokenizers on a
13-language FLORES+ corpus and found six pairs where the second metric adds no
ordering information over the first. Four of them are algebraic identities, not
empirical correlations: they hold exactly, for every tokenizer, by construction.

Publishing both members of such a pair is not neutral. A reader scanning the
results file sees two independent-looking columns agree and reads that as
corroboration, when it is one measurement printed twice. In a table of 23
metrics that is a real distortion of how much evidence is on the page.

Merging happens here, after every metric has computed, rather than inside the
metric classes. The secondary metric's own computation is unchanged and still
tested on its own terms; only the shape of the published results changes. That
keeps this reversible and keeps the evidence for each merge in one place.

The pairs, with the measurement that justifies each:

``avg_tokens_per_line`` into ``compression_rate``
    ``compression_rate(lines) * avg_tokens_per_line == 1.000000`` for every
    tokenizer; Spearman -1.000. On a fixed corpus the unit count does not depend
    on the tokenizer, so the two are reciprocals under every measurement method,
    not only under ``lines``.

``type_token_ratio`` into ``vocabulary_utilization``
    ``(ttr / utilization) * (total_tokens / vocab_size) == 1.000000`` for every
    tokenizer. TTR's numerator, the distinct-token count, is byte-for-byte
    ``vocabulary_utilization.used_tokens``; its denominator is the token count
    that ``compression_rate`` already reports. Raw TTR also falls with stream
    length, and the streams differ by a factor of 1.91 across these tokenizers,
    so its independent variation is mostly that artifact.

``unigram_distribution_metrics`` into ``renyi_efficiency``
    ``renyi_1.0 * log2(declared vocabulary size) == unigram_entropy`` to zero
    relative error over all 37 tokenizers: the unnormalized numerator of a
    value the Renyi block already publishes. Measured on gpt2 in the
    open-source benchmark: 0.5826525136289539 times log2(50257) gives
    9.099305825198964, which is the published ``global_unigram_entropy`` to the
    last digit.

    This said "observed types" until 1.0.2, which was left over from the
    pre-1.0 normalization. Renyi divides by the declared vocabulary size
    following Zouhar et al. 2023, as ``renyi_efficiency.metadata.normalizer``
    states; the observed-types variant is published separately under
    ``observed_normalization``. With observed types the same product gives
    7.503, so the sentence justified the merge with an identity that does not
    hold for the value being merged.

``utf8_char_split`` into ``utf8_token_integrity``
    ``total_splits`` reproduces ``trailing_incomplete`` to within 4% for 35 of
    37 tokenizers and exactly for 6; Spearman -0.954 against
    ``completeness_rate``. A character split across tokens is the same event as
    a token ending mid-character, counted from the other side. The per-byte-width
    breakdown and ``aligned_fraction`` are carried over, because the token-side
    metric does not have them.

``lorenz_curve_data`` into ``tokenizer_fairness_gini``
    ``1 - 2 * area_under(lorenz) == gini_coefficient`` to six decimal places for
    every tokenizer. It is the same statistic in plot form.

``digit_split_variability`` into ``three_digit_boundary_alignment``
    Spearman -0.992 between the pooled ``entropy_short`` and ``avg_f1``.
    The pooled entropy is also partly an artifact: it mixes digit lengths, so
    24 of 37 tokenizers report exactly the corpus length-mixture constant. The
    per-digit-length entropies, which is what the metric's docstring actually
    describes, are carried over as a field.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# (secondary metric, primary metric, key the secondary is filed under,
#  which of the secondary's fields to keep; None keeps the whole block)
MERGES: List[Tuple[str, str, str, Optional[Tuple[str, ...]]]] = [
    ("avg_tokens_per_line", "compression_rate", "tokens_per_line", None),
    ("type_token_ratio", "vocabulary_utilization", "type_token_ratio", None),
    ("unigram_distribution_metrics", "renyi_efficiency", "unigram_distribution", None),
    (
        "utf8_char_split",
        "utf8_token_integrity",
        "char_split",
        ("split_rate", "splits_per_1k_tokens", "splits_per_1k_multibyte",
         "total_splits", "total_multibyte_chars", "unaligned_multibyte_chars",
         "aligned_fraction", "alignment_mismatches", "per_byte_width"),
    ),
    ("lorenz_curve_data", "tokenizer_fairness_gini", "lorenz_curve", None),
    (
        "digit_split_variability",
        "three_digit_boundary_alignment",
        "split_variability",
        ("by_digit_length", "by_bucket"),
    ),
]

# Human-readable justification per merge, echoed into the primary's metadata so
# a results file explains itself without the reader consulting this module.
_EVIDENCE: Dict[str, str] = {
    "avg_tokens_per_line": (
        "exact reciprocal of compression_rate on a fixed corpus "
        "(product 1.000000, Spearman -1.000 over 37 tokenizers)"
    ),
    "type_token_ratio": (
        "algebraic restatement of vocabulary_utilization; its numerator is "
        "used_tokens and its denominator is the token count compression_rate "
        "reports (identity holds to 1.000000 over 37 tokenizers)"
    ),
    "unigram_distribution_metrics": (
        "unigram_entropy is the unnormalized numerator of renyi_1.0 "
        "(identity to zero relative error over 37 tokenizers)"
    ),
    "utf8_char_split": (
        "counts the same events as utf8_token_integrity from the character side "
        "(Spearman -0.954 with completeness_rate; total_splits within 4% of "
        "trailing_incomplete for 35 of 37 tokenizers)"
    ),
    "lorenz_curve_data": (
        "1 - 2*area(lorenz) equals gini_coefficient to six decimal places for "
        "every tokenizer; the same statistic in plot form"
    ),
    "digit_split_variability": (
        "Spearman -0.992 between pooled entropy_short and "
        "three_digit_boundary_alignment avg_f1; the pooled entropy also mixes "
        "digit lengths, so 24 of 37 tokenizers report the corpus "
        "length-mixture constant rather than a tokenizer property"
    ),
}


def _keep(block: Any, fields: Optional[Tuple[str, ...]]) -> Any:
    """Restrict a per-tokenizer entry to *fields*, when a subset was requested."""
    if fields is None or not isinstance(block, dict):
        return block
    kept = {k: v for k, v in block.items() if k in fields}
    # Preserve the global/per_language split when the entry has one.
    for nested in ("global", "per_language"):
        if isinstance(block.get(nested), dict):
            inner = block[nested]
            kept[nested] = (
                {k: v for k, v in inner.items() if k in fields}
                if nested == "global"
                else inner
            )
    return kept


def merge_redundant_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Fold each redundant metric into its primary, in place, and return *results*.

    A merge is skipped when either metric is absent, so a run that disabled a
    metric family behaves normally.
    """
    for secondary, primary, field, fields in MERGES:
        if secondary not in results or primary not in results:
            continue

        sec_block = results[secondary]
        pri_block = results[primary]
        sec_per_tok = sec_block.get("per_tokenizer", {})
        pri_per_tok = pri_block.setdefault("per_tokenizer", {})

        for tok_name, sec_data in sec_per_tok.items():
            pri_entry = pri_per_tok.setdefault(tok_name, {})
            if isinstance(pri_entry, dict):
                pri_entry[field] = _keep(sec_data, fields)

        metadata = pri_block.setdefault("metadata", {})
        merged = metadata.setdefault("merged_metrics", {})
        merged[secondary] = {
            "reported_under": field,
            "reason": _EVIDENCE.get(secondary, "redundant with this metric"),
        }
        if isinstance(sec_block.get("metadata"), dict):
            merged[secondary]["source_metadata"] = sec_block["metadata"]

        del results[secondary]
        logger.info(
            "Merged %s into %s (reported under %r): %s",
            secondary, primary, field, _EVIDENCE.get(secondary, ""),
        )

    return results
