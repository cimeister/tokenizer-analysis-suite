"""Tests for the contract `analysis_results.json` makes to its readers.

The README documents this file's shape and nothing enforced it. These tests
cover the properties a downstream consumer relies on and that the 1.0 audit
found broken: the metric key set, that absent values are null rather than a
plausible zero, that the file is strict JSON, and that it says what produced it.

They run the CLI in-process against the bundled demo, so they exercise the real
assembly path rather than a mock of it.
"""
import json
import subprocess
import sys

import pytest

from .conftest import requires_flores

from tokenizer_analysis.metrics.base import BaseMetrics
from tokenizer_analysis.metrics.redundancy import MERGES, merge_redundant_metrics


REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def _reject_non_standard(constant: str):
    raise ValueError(f"non-standard JSON token: {constant}")


@pytest.fixture(scope="module")
def demo_results(tmp_path_factory):
    """Run the bundled demo once and return the parsed results file.

    Kept module-scoped because encoding dominates the runtime; the tests below
    only read it.
    """
    out = tmp_path_factory.mktemp("demo")
    proc = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--use-sample-data", "--samples-per-lang", "10",
         "--no-plots", "--no-code-ast", "--output-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            "demo run failed with exit "
            f"{proc.returncode}:\n{proc.stderr.decode(errors='replace')[-3000:]}"
        )
    return json.loads((out / "analysis_results.json").read_text())


@requires_flores
def test_results_file_is_strict_json(demo_results, tmp_path_factory):
    """No NaN or Infinity tokens, which Python writes but strict parsers reject.

    `gini.cost_ratio` used to be float('inf') when the minimum cost was zero,
    which json.dump serializes as the bare token Infinity. JavaScript's
    JSON.parse and most non-Python parsers refuse to read that file at all.
    """
    text = json.dumps(demo_results)
    json.loads(text, parse_constant=_reject_non_standard)


@requires_flores
def test_run_metadata_identifies_what_produced_the_file(demo_results):
    """A results file must be traceable to the code and inputs behind it."""
    meta = demo_results.get("run_metadata")
    assert meta is not None, "results file carries no provenance block"
    assert meta["package_version"]
    assert "tokenizers" in meta and meta["tokenizers"], "no tokenizers recorded"
    for name, entry in meta["tokenizers"].items():
        assert "class" in entry, f"{name} has no class recorded"
    assert meta["corpus"]["samples_per_lang"] == 10


@requires_flores
def test_merged_metrics_are_not_top_level_and_are_reachable(demo_results):
    """Each merged metric moved under its primary, and the move is recorded.

    A merge must relocate the data, not delete it. The slim writer rebuilds each
    per-tokenizer entry from a per-metric whitelist, so a folded field is only
    one missing whitelist entry away from vanishing silently.
    """
    for secondary, primary, field, _ in MERGES:
        if primary not in demo_results:
            continue
        assert secondary not in demo_results, (
            f"{secondary} should be reported under {primary}, not at top level"
        )
        per_tok = demo_results[primary]["per_tokenizer"]
        for tok, entry in per_tok.items():
            assert field in entry, (
                f"{primary}.{tok} is missing the merged field {field!r} "
                f"(from {secondary})"
            )
        merged = demo_results[primary].get("metadata", {}).get("merged_metrics", {})
        assert secondary in merged, (
            f"{primary} does not record that it absorbed {secondary}"
        )


@requires_flores
def test_every_metric_has_a_per_tokenizer_block(demo_results):
    """The documented shape is {metric: {per_tokenizer: {tok: ...}}}."""
    for name, block in demo_results.items():
        if name == "run_metadata":
            continue
        assert isinstance(block, dict), f"{name} is not a dict"
        assert "per_tokenizer" in block, f"{name} has no per_tokenizer block"


@pytest.fixture(scope="module")
def full_and_slim_results(tmp_path_factory):
    """Run the bundled demo with --save-full-results and return both files, parsed.

    A separate run from demo_results above: that fixture never passes
    --save-full-results, so it has no full file to compare against.
    """
    out = tmp_path_factory.mktemp("full_and_slim")
    proc = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--use-sample-data", "--samples-per-lang", "10",
         "--no-plots", "--no-code-ast", "--save-full-results",
         "--output-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            "demo run failed with exit "
            f"{proc.returncode}:\n{proc.stderr.decode(errors='replace')[-3000:]}"
        )
    slim = json.loads((out / "analysis_results.json").read_text())
    full = json.loads((out / "analysis_results_full.json").read_text())
    return slim, full


def _leaf_paths(obj, prefix=()):
    """Yield (key_path, value) for every leaf under obj, including empty dicts.

    A key path is the tuple of keys walked to reach a value that is not
    itself a dict, or is an empty dict (which has no further keys to walk
    into but is still a value a consumer can read).
    """
    if isinstance(obj, dict):
        if not obj:
            yield prefix, {}
        for key, value in obj.items():
            yield from _leaf_paths(value, prefix + (key,))
    elif isinstance(obj, list):
        yield prefix, obj
    else:
        yield prefix, obj


@requires_flores
def test_slim_file_is_a_strict_projection_of_the_full_file(full_and_slim_results):
    """Every value analysis_results.json publishes must exist, unchanged, at the
    same key path in analysis_results_full.json.

    This is the property normalize_results and select_results exist to
    guarantee: normalize_results renames and pivots raw results to their
    published key names without deleting anything, analysis_results_full.json
    is written from that output directly, and select_results (used by
    slim_results_for_json) only deletes keys from it to build
    analysis_results.json. Because the second pass never renames, every path
    the slim file publishes must already be present, under the same name and
    with the same value, in the output of the first pass.

    Before this split, a single function renamed keys while selecting them
    (overall to global, by_language to per_language, and so on), so a value
    read out of the slim file could not be looked up at the same path in the
    full file: the full file still used the raw, pre-rename names. A demo run
    measured 1022 slim paths with no counterpart in the full file. This test
    asserts that count is zero, so a future change to either pass cannot
    reintroduce a rename that only one of the two files sees.

    run_metadata is provenance, added to the slim file only, after both files
    are written from the same normalized results, and is excluded from the
    comparison for that reason.
    """
    slim, full = full_and_slim_results

    full_paths = dict(_leaf_paths(full))
    slim_paths = {
        path: value for path, value in _leaf_paths(slim) if path[0] != "run_metadata"
    }

    missing = [path for path in slim_paths if path not in full_paths]
    differing = [
        path for path in slim_paths
        if path in full_paths and full_paths[path] != slim_paths[path]
    ]

    assert not missing, (
        f"{len(missing)} slim path(s) have no counterpart in the full file, "
        f"starting with {missing[:5]}"
    )
    assert not differing, (
        f"{len(differing)} slim path(s) differ in value from the full file, "
        f"starting with {differing[:5]}"
    )


def test_merge_is_a_no_op_when_the_secondary_is_absent():
    """A run that disabled a metric family must not trip the merge step."""
    results = {"compression_rate": {"per_tokenizer": {"t": {"global": {}}}}}
    merged = merge_redundant_metrics(results)
    assert "compression_rate" in merged
    assert merged["compression_rate"]["per_tokenizer"]["t"] == {"global": {}}


class TestNoDataIsNull:
    """Absent values must be null, never a number that reads as a measurement.

    0.0 is a legal value for nearly every rate here, so publishing it for the
    empty case made 'no UNK token exists' and 'no UNK was emitted' identical in
    the output, among others.
    """

    def test_safe_divide_returns_none_on_zero_denominator(self):
        assert BaseMetrics.safe_divide(1.0, 0.0) is None

    def test_safe_divide_still_divides(self):
        assert BaseMetrics.safe_divide(1.0, 4.0) == 0.25

    def test_safe_divide_honours_an_explicit_default(self):
        """A caller wanting a number for the empty case must say so."""
        assert BaseMetrics.safe_divide(1.0, 0.0, default=1.0) == 1.0

    def test_empty_stats_reports_none_not_zero(self):
        stats = BaseMetrics.empty_stats()
        for key in ("mean", "median", "std", "std_err", "min", "max"):
            assert stats[key] is None, f"{key} should be None for an empty sample"
        # Sample size is genuinely zero, so it stays numeric.
        assert stats["count"] == 0
        assert stats["sum"] == 0


@requires_flores
def test_single_corpus_does_not_report_perfect_fairness(tmp_path):
    """Gini is undefined for one corpus and must say so, not report 0.0.

    0.0 is the value for perfect equality, so the degenerate case was
    indistinguishable from the best possible result. It was also contradicted by
    the language_costs entry in the same object.
    """
    out = tmp_path / "mono"
    proc = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--tokenizer-config", "configs/sample_tokenizers.json",
         "--input", "parallel/eng_Latn.txt",
         "--samples-per-lang", "10", "--no-plots", "--no-code-ast",
         "--output-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-2000:]

    results = json.loads((out / "analysis_results.json").read_text())
    gini = results["tokenizer_fairness_gini"]["per_tokenizer"]["bpe"]["global"]
    assert gini["gini_coefficient"] is None
    assert gini["num_languages"] == 1
    assert "warning" in gini
    # mean_cost is a real measurement even with one corpus, so it stays a number.
    assert isinstance(gini["mean_cost"], float) and gini["mean_cost"] > 0


@requires_flores
def test_grouped_analysis_runs_after_the_metric_merge(tmp_path):
    """--run-grouped-analysis must read the merged results, not the old keys.

    run_analysis folds six metrics into the metric that owns the measurement
    before returning, and run_grouped_analysis then filters those base results
    by language group. It went on reading the pre-merge top-level key
    digit_split_variability, so the whole flag exited 1 with
    KeyError: 'digit_split_variability'. Nothing covered the flag, so the merge
    shipped with it broken.
    """
    out = tmp_path / "grouped"
    proc = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--use-sample-data", "--samples-per-lang", "5",
         "--run-grouped-analysis", "--no-plots", "--no-code-ast",
         "--output-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-3000:]


def test_a_nested_merged_block_is_filtered_to_the_group_languages():
    """The block the merge nested must be language-filtered like its parent.

    _filter_digit_boundary_results passes unrecognized keys through untouched.
    After the merge, split_variability is one of those keys and holds
    per-language numbers, so a language group inherited values computed over
    every language in the run.
    """
    from tokenizer_analysis.main import UnifiedTokenizerAnalyzer

    merged_shape = {'per_tokenizer': {'bpe': {
        'by_digit_length': {'2': {'eng_Latn': 0.5, 'arb_Arab': 0.6}},
        'overall': {'eng_Latn': 0.7, 'arb_Arab': 0.8},
        'split_variability': {
            'by_digit_length': {'2': {'eng_Latn': 0.1, 'arb_Arab': 0.3}},
            'by_bucket': {'short': {'eng_Latn': 1.0, 'arb_Arab': 2.0}, 'long': {}},
        },
    }}}
    filtered = UnifiedTokenizerAnalyzer._filter_digit_boundary_results(
        object.__new__(UnifiedTokenizerAnalyzer), merged_shape, ['eng_Latn']
    )
    nested = filtered['per_tokenizer']['bpe']['split_variability']
    assert list(nested['by_digit_length']['2']) == ['eng_Latn']
    assert list(nested['by_bucket']['short']) == ['eng_Latn']


@requires_flores
def test_tokenized_data_cache_can_be_replayed(tmp_path):
    """--save-tokenized-data then --tokenized-data-file must complete.

    The replay path never bound `tokenizer_configs`, and the last line of
    run_from_args reads it to build run_metadata, so every replay crashed with
    UnboundLocalError after computing every metric and wrote nothing. The
    provenance block broke the cache workflow, and neither had a test.
    """
    saved = tmp_path / "saved"
    save = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--use-sample-data", "--samples-per-lang", "5", "--no-plots",
         "--no-code-ast", "--save-tokenized-data", "--output-dir", str(saved)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    assert save.returncode == 0, save.stderr.decode(errors="replace")[-2000:]

    replayed = tmp_path / "replayed"
    replay = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--tokenized-data-file", str(saved / "tokenized_data.pkl"),
         "--tokenized-data-config", str(saved / "tokenized_data_config.json"),
         "--language-config", str(saved / "tokenized_data_language_config.json"),
         "--no-plots", "--no-code-ast", "--output-dir", str(replayed)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    assert replay.returncode == 0, replay.stderr.decode(errors="replace")[-2000:]

    first = json.loads((saved / "analysis_results.json").read_text())
    again = json.loads((replayed / "analysis_results.json").read_text())
    assert (first["fertility"]["per_tokenizer"]["bpe"]["global"]["mean"]
            == again["fertility"]["per_tokenizer"]["bpe"]["global"]["mean"])
    assert "run_metadata" in again


def test_run_metadata_records_the_flags_that_change_the_numbers():
    """A flag that narrows what is measured must be on the record.

    --filter-script-family, --filter-resource-level and --use-builtin-math-data
    all change the published values and all left run_metadata byte-identical,
    because the block named a hand-maintained list of interesting flags and
    those three were not on it. The block now diffs the parsed namespace against
    the parser's own defaults, so a flag added later is covered without anyone
    remembering to add it.
    """
    from tokenizer_analysis.cli.run_analysis import build_parser, _non_default_arguments

    base = build_parser().parse_args(["--use-sample-data"])
    assert _non_default_arguments(base) == {"use_sample_data": True}

    for flag, value in (
        ("--filter-script-family", "Latin"),
        ("--filter-resource-level", "high"),
    ):
        args = build_parser().parse_args(["--use-sample-data", flag, value])
        recorded = _non_default_arguments(args)
        assert value in recorded.values(), f"{flag} is absent from {recorded}"

    args = build_parser().parse_args(["--use-sample-data", "--use-builtin-math-data"])
    assert _non_default_arguments(args).get("use_builtin_math_data") is True

    # Where the output goes is not what was measured, so two runs differing only
    # in that must not look like different measurements.
    args = build_parser().parse_args(["--use-sample-data", "--output-dir", "/tmp/x",
                                      "--no-plots", "--save-full-results"])
    assert _non_default_arguments(args) == {"use_sample_data": True}


# ----------------------------------------------------------------------
# The schema contract, asserted over every metric rather than a sample.
# ----------------------------------------------------------------------

# Imported rather than restated, so the test cannot drift from the constant the
# metrics assign from.
from tokenizer_analysis.constants import AGGREGATION_LABELS as _ALLOWED_AGGREGATIONS


def _metrics(results):
    """Every metric block in a results file, by name."""
    return {
        name: block for name, block in results.items()
        if name != "run_metadata" and isinstance(block, dict)
    }


@requires_flores
def test_every_metric_publishes_a_global(demo_results):
    """No exemptions. A consumer reads one key for the headline value.

    Five metrics had none before 1.0.0, and one of them, trigram_entropy,
    published the same four numbers as flat global_* siblings instead, so a
    parser written against bigram_entropy silently found nothing. token_length
    and encoding_speed carry a global that duplicates an existing block, which
    is deliberate: an exception in the schema costs a reader more than a
    duplicated number does.
    """
    missing = []
    for name, block in _metrics(demo_results).items():
        for tok, entry in (block.get("per_tokenizer") or {}).items():
            if isinstance(entry, dict) and "global" not in entry:
                missing.append(f"{name}.per_tokenizer.{tok}")
    assert not missing, "no global block on: " + ", ".join(sorted(missing))


@requires_flores
def test_every_metric_declares_its_aggregation(demo_results):
    """Which average `global` reports, from a fixed set.

    global meant a ratio of sums in one metric, a mean of per-document ratios
    in another, an unweighted mean across languages in a third and a set union
    in a fourth, with nothing in the output saying which. On the bundled
    parallel corpus every language holds the same number of lines, so micro and
    macro agree and the difference is invisible until someone runs an unequal
    corpus.
    """
    bad = {}
    for name, block in _metrics(demo_results).items():
        label = (block.get("metadata") or {}).get("aggregation")
        if label not in _ALLOWED_AGGREGATIONS:
            bad[name] = label
    assert not bad, (
        "aggregation missing or not one of "
        f"{sorted(_ALLOWED_AGGREGATIONS)}: {bad}"
    )


@requires_flores
def test_every_per_language_entry_carries_a_count_or_says_why_not(demo_results):
    """So a consumer can re-derive the other weighting.

    A metric whose per-language entry is one language, so that the count would
    be 1 for every entry, says so in metadata.per_language_count instead. That
    is the only accepted reason to omit it: an entry that is simply silent is
    the failure this asserts against.
    """
    offenders = []
    for name, block in _metrics(demo_results).items():
        metadata = block.get("metadata") or {}
        if metadata.get("per_language_count"):
            continue
        for tok, entry in (block.get("per_tokenizer") or {}).items():
            per_lang = (entry or {}).get("per_language")
            if not isinstance(per_lang, dict):
                continue
            for lang, value in per_lang.items():
                if isinstance(value, dict) and "count" not in value:
                    offenders.append(f"{name}.{tok}.{lang}")
                    break
    assert not offenders, (
        "per_language entries with no count and no stated reason: "
        + ", ".join(sorted(offenders))
    )


@requires_flores
def test_every_metric_names_the_unit_its_count_is_in(demo_results):
    """A count is not interpretable without its unit.

    Documents, tokens, digit spans and AST nodes are not interchangeable, and
    the same field name carries all four across the file.
    """
    missing = [
        name for name, block in _metrics(demo_results).items()
        if not (block.get("metadata") or {}).get("count_unit")
    ]
    assert not missing, "no count_unit in metadata: " + ", ".join(sorted(missing))


def test_a_two_bucket_corpus_still_writes_strict_json(tmp_path):
    """The corpus shape that broke CI: `NaN` is not valid JSON.

    `scipy.stats.spearmanr` over two points returns a defined rho and an
    undefined p, because a rank correlation over two points has no significance
    level. `numeric_magnitude_consistency` wrote that NaN straight out, and
    `json.dump` renders it as the bare token `NaN`, which no strict parser
    accepts. The demo corpus happens to produce three or more digit-length
    buckets, so the existing strict-JSON test never saw it; a corpus whose
    numbers span two buckets does.

    Two layers are asserted here: the metric publishes null, and the serializer
    would have converted any remaining non-finite float anyway.
    """
    corpus = tmp_path / "eng_Latn.txt"
    # Numbers 0 to 199, so digit lengths 1, 2 and 3, which collapse to two
    # populated buckets after the short/long split.
    corpus.write_text(
        "\n".join(f"Item number {i} costs {i * 7} in total." for i in range(200)) + "\n"
    )
    languages = tmp_path / "languages.json"
    languages.write_text(json.dumps({
        "languages": {"eng_Latn": {"name": "English", "data_path": str(corpus)}},
    }))

    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--tokenizer-config", "configs/sample_tokenizers.json",
         "--language-config", str(languages), "--samples-per-lang", "50",
         "--no-plots", "--no-code-ast", "--output-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-2000:]

    # parse_constant fires on NaN, Infinity and -Infinity, the three tokens
    # json.dump will happily write and no other parser will read.
    json.loads((out / "analysis_results.json").read_text(),
               parse_constant=_reject_non_standard)


def test_the_serializer_converts_any_non_finite_float_to_null():
    """The backstop, independent of which metric produced the value."""
    from tokenizer_analysis.cli.run_analysis import _convert_for_json_public as convert

    converted = convert({
        "a": float("nan"), "b": float("inf"), "c": float("-inf"),
        "d": [1.0, float("nan")], "e": {"f": float("inf")}, "g": 0.5,
    })
    assert converted == {
        "a": None, "b": None, "c": None,
        "d": [1.0, None], "e": {"f": None}, "g": 0.5,
    }
    json.dumps(converted, allow_nan=False)


def test_cer_skipped_survives_slimming():
    """A null mean_cer is ambiguous without the flag that explains it.

    mean_cer and whitespace_fidelity are null both when the character error
    rate exceeded --cer-time-budget and when there was nothing to measure.
    cer_skipped is the only field that separates the two, and METRICS.md and
    the benchmark README both tell readers to consult it.

    It is written only when a tokenizer actually exceeds the budget, so a demo
    corpus small enough to finish never produces it. That is why this is a
    unit test over a hand-built results dict rather than an assertion on a demo
    run: a comparison against demo output cannot see this field at all, which
    is how the 1.0.2 schema refactor dropped it without any gate noticing.
    """
    from tokenizer_analysis.cli.run_analysis import slim_results_for_json

    results = {
        "reconstruction_fidelity": {
            "per_tokenizer": {
                "tok": {
                    "overall": {
                        "exact_match_rate": 0.03,
                        "mean_cer": None,
                        "whitespace_fidelity": None,
                        "count": 10,
                        "total_tokens": 100,
                    },
                    "by_domain": {},
                    "cer_skipped": True,
                }
            },
            "metadata": {"aggregation": "micro_pooled"},
        }
    }

    slimmed = slim_results_for_json(results)
    entry = slimmed["reconstruction_fidelity"]["per_tokenizer"]["tok"]
    assert entry.get("cer_skipped") is True, (
        "cer_skipped was dropped, so a reader cannot tell a skipped "
        "mean_cer from one that had nothing to measure"
    )
    assert entry["global"]["mean_cer"] is None


@pytest.fixture(scope="module")
def degenerate_run(tmp_path_factory):
    """A corpus too small for most metrics to have anything to measure.

    One short document and two tokenizers.  Bigram and trigram contexts do not
    clear their occurrence thresholds, there is one language so cross-language
    dispersion is undefined, and several counts are zero.  This is the shape
    that produced the 1.0.2 defects: a value that could not be computed was
    published as 0.0 or 1.0, and one path raised TypeError instead.
    """
    work = tmp_path_factory.mktemp("degenerate")
    corpus = work / "tiny.txt"
    corpus.write_text("The cat sat.\n", encoding="utf-8")

    toks = work / "toks.json"
    toks.write_text(json.dumps({
        "bpe": {"class": "huggingface", "path": "tokenizers/bpe.json"},
        "unigramlm": {"class": "huggingface", "path": "tokenizers/unigramlm.json"},
    }), encoding="utf-8")

    out = work / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--tokenizer-config", str(toks), "--input", str(corpus),
         "--no-plots", "--no-code-ast", "--output-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            "a corpus with nothing to measure must still complete; exit "
            f"{proc.returncode}:\n{proc.stderr.decode(errors='replace')[-3000:]}"
        )
    return json.loads((out / "analysis_results.json").read_text())


def test_a_corpus_with_nothing_to_measure_publishes_null_not_a_number(degenerate_run):
    """A count of zero may not sit beside a number that reads as a measurement.

    This is the invariant behind every 1.0.2 metric fix.  Where a global block
    reports it measured nothing, every rate and mean in that block has to be
    null: a bigram entropy of 0.0 means every context has exactly one
    successor, a compression rate of 1.0 means one unit per token, and a
    completeness rate of 1.0 means every token was well formed.  None of those
    may stand in for "there was nothing to measure".

    Fields that count things (count, total_tokens, used_tokens and the like)
    stay numeric: zero of something is a true statement about the sample.
    """
    counters = ("count", "total", "types_evaluated", "types_excluded",
                "used_tokens", "vocab_size", "num_samples", "unmappable",
                "parse_error_spans", "num_languages", "num_depth_levels")
    offenders = []

    for name, block in _metrics(degenerate_run).items():
        for tok, entry in (block.get("per_tokenizer") or {}).items():
            glob = (entry or {}).get("global")
            if not isinstance(glob, dict):
                continue
            count = glob.get("count")
            if count != 0:
                continue
            for field, value in glob.items():
                if field in counters or field.startswith("total_"):
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    offenders.append(f"{name}.{tok}.global.{field} = {value} beside count 0")

    assert not offenders, (
        "a measured-looking number published where count is 0:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_run_metadata_records_the_inputs_that_decide_the_numbers(degenerate_run):
    """Provenance the 1.0.1 file did not carry.

    A results file has to say which corpus and which tokenizers produced it,
    not only which config named them.  Two runs over different snapshots of a
    corpus under one config were previously indistinguishable, and a Hub-side
    retokenization would move every number with nothing recording it.
    """
    meta = degenerate_run["run_metadata"]

    assert meta.get("timestamp_utc"), "no timestamp, so two runs of one commit are indistinguishable"
    assert meta["timestamp_utc"].endswith("+00:00"), "timestamp is not UTC"

    for name, entry in meta["tokenizers"].items():
        assert entry.get("sha256_16") or entry.get("hub_revision"), (
            f"tokenizer {name} is recorded by path alone, so a change to it "
            "would move every number with nothing saying so"
        )

    digest = meta["corpus"]["digest"]
    assert digest.get("n_languages"), "no corpus digest"
    for lang, entry in digest["files"].items():
        assert entry.get("sha256_16"), f"corpus {lang} has no hash"
        assert entry.get("bytes"), f"corpus {lang} has no byte count"


@requires_flores
def test_corpus_paths_under_the_working_directory_are_recorded_relative(demo_results):
    """So a committed results file does not name whoever produced it.

    The digest records a path per language. A corpus inside the working
    directory is recorded relative to it; one outside keeps its absolute path,
    because there is nothing to make it relative to. The benchmark reads
    parallel/ from the repository root, so its committed results file must
    carry short paths rather than a home directory.
    """
    files = demo_results["run_metadata"]["corpus"]["digest"]["files"]
    absolute = {lang: e["path"] for lang, e in files.items() if e["path"].startswith("/")}
    assert not absolute, (
        "corpus paths under the working directory recorded as absolute: "
        + ", ".join(f"{k} -> {v}" for k, v in sorted(absolute.items()))
    )
