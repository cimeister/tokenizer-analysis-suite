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


def test_results_file_is_strict_json(demo_results, tmp_path_factory):
    """No NaN or Infinity tokens, which Python writes but strict parsers reject.

    `gini.cost_ratio` used to be float('inf') when the minimum cost was zero,
    which json.dump serializes as the bare token Infinity. JavaScript's
    JSON.parse and most non-Python parsers refuse to read that file at all.
    """
    text = json.dumps(demo_results)
    json.loads(text, parse_constant=_reject_non_standard)


def test_run_metadata_identifies_what_produced_the_file(demo_results):
    """A results file must be traceable to the code and inputs behind it."""
    meta = demo_results.get("run_metadata")
    assert meta is not None, "results file carries no provenance block"
    assert meta["package_version"]
    assert "tokenizers" in meta and meta["tokenizers"], "no tokenizers recorded"
    for name, entry in meta["tokenizers"].items():
        assert "class" in entry, f"{name} has no class recorded"
    assert meta["corpus"]["samples_per_lang"] == 10


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


def test_every_metric_has_a_per_tokenizer_block(demo_results):
    """The documented shape is {metric: {per_tokenizer: {tok: ...}}}."""
    for name, block in demo_results.items():
        if name == "run_metadata":
            continue
        assert isinstance(block, dict), f"{name} is not a dict"
        assert "per_tokenizer" in block, f"{name} has no per_tokenizer block"


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
