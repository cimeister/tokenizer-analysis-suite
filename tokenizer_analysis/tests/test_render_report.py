"""The benchmark report renderer had no test at all.

`benchmarks/open_source/render_report.py` writes REPORT.md, which is the most
widely read artifact in the repository, and nothing covered it. The parts that
matter are the ones that can go wrong silently: a column reading the wrong key
renders `n/a` rather than raising, and a legend that has fallen behind the
severities the checker emits renders an unknown state as a bare string in a
cell.

The renderer is a script outside the package, so it is loaded by path.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "benchmarks" / "open_source" / "render_report.py"


@pytest.fixture(scope="module")
def render():
    spec = importlib.util.spec_from_file_location("render_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _results():
    """Two tokenizers, one with a per-line Gini and one without."""
    def block(vocab, comp, tpl, fert, gini, per_line):
        return {
            "vocabulary_utilization": {"global": {"vocab_size": vocab}},
            "compression_rate": {
                "global": {"compression_rate": comp},
                "tokens_per_line": {"global_avg": tpl},
                "per_language": {"eng_Latn": {"compression_rate": comp}},
            },
            "fertility": {"global": {"mean": fert}},
            "tokenizer_fairness_gini": {
                "global": {"gini_coefficient": gini},
                "per_line_normalization": per_line,
            },
        }

    a = block(50257, 3.21, 83.5, 1.42, 0.2000,
              {"gini_coefficient": 0.2933, "num_languages": 13})
    b = block(250002, 4.05, 45.9, 1.10, 0.0976, None)
    out = {}
    for metric in a:
        out[metric] = {"per_tokenizer": {"gpt2": a[metric], "xlm-roberta": b[metric]}}
    out["run_metadata"] = {"package_version": "test"}
    return out


def _sanity(severities):
    """One tokenizer entry per key of *severities*, keyed by full check name."""
    names = [
        "C1 byte-level 256-coverage",
        "C17 strict byte-alphabet vocab presence",
        "C6 digit handling",
        "C16 vocab reachability",
    ]
    per_tok = {}
    for tok, sevs in severities.items():
        per_tok[tok] = {
            "overall_severity": "fail",
            "checks": {n: {"severity": s} for n, s in zip(names, sevs)},
        }
    return {"tokenizer_sanity_check": {"per_tokenizer": per_tok}}


def _section(report, heading):
    """The text of one `## ` section, up to the next one.

    Three tables in the report have a row per tokenizer, so a search over the
    whole file finds the wrong one.
    """
    start = report.index(heading)
    rest = report.index("\n## ", start + len(heading))
    return report[start:rest]


def _headline_row(report, name):
    rows = [l for l in _section(report, "## Headline").splitlines()
            if l.startswith(f"| {name} |")]
    assert len(rows) == 1, f"expected one headline row for {name}, got {rows}"
    return rows[0]


def _render(render, tmp_path, sanity=None):
    results_path = tmp_path / "analysis_results.json"
    results_path.write_text(json.dumps(_results()), encoding="utf-8")
    argv = ["--results", str(results_path), "--output", str(tmp_path / "REPORT.md")]
    if sanity is not None:
        sanity_path = tmp_path / "sanity_results.json"
        sanity_path.write_text(json.dumps(sanity), encoding="utf-8")
        argv += ["--sanity", str(sanity_path)]
    assert render.main(argv) == 0
    return (tmp_path / "REPORT.md").read_text(encoding="utf-8")


class TestHeadlineColumns:
    def test_both_gini_columns_are_present_and_differ(self, render, tmp_path):
        report = _render(render, tmp_path)
        assert "Gini per byte (down)" in report
        assert "Gini per line (down)" in report
        # Distinct values, so the test cannot pass by rendering one twice.
        assert "0.2000" in report and "0.2933" in report

    def test_an_absent_per_line_block_renders_as_not_available(self, render, tmp_path):
        """A null block is a corpus that is not line-parallel, not a zero."""
        report = _render(render, tmp_path)
        row = _headline_row(report, "XLM-RoBERTa base")
        assert "0.0976" in row, row
        assert "| n/a |" in row, row
        assert "0.000" not in row, row

    def test_tokens_per_line_comes_from_the_results_file(self, render, tmp_path):
        report = _render(render, tmp_path)
        assert "Tokens/line (down)" in report
        assert "83.5" in report and "45.9" in report

    def test_no_number_in_the_headline_is_typed_by_hand(self, render, tmp_path):
        """Every headline cell must trace to a key of the fixture.

        A hardcoded figure would survive every other assertion here.
        """
        report = _render(render, tmp_path)
        cells = [c.strip() for c in _headline_row(report, "GPT-2").strip("|").split("|")]
        assert cells == ["GPT-2", "50,257", "3.210", "83.5", "1.420",
                         "0.2000", "0.2933", "n/a", "n/a", "n/a"]


class TestHealthMatrix:
    def test_one_column_per_check_in_the_sanity_file(self, render, tmp_path):
        sanity = _sanity({"gpt2": ["pass", "warn", "warn", "fail"],
                          "xlm-roberta": ["pass", "pass", "pass", "pass"]})
        report = _render(render, tmp_path, sanity)
        header = next(l for l in report.splitlines() if l.startswith("| Tokenizer | [C1]"))
        for cid in ("C1", "C17", "C6", "C16"):
            assert f"[{cid}](" in header, header
        # Four checks plus the name column.
        assert header.count("|") == 6, header

    def test_the_legend_covers_every_severity_emitted(self, render, tmp_path):
        """The assertion that catches a sixth severity being added later.

        A severity with no glyph renders as its own bare string in a cell and
        as nothing in the legend, which reads as a state the reader is meant
        to recognise.
        """
        emitted = ["pass", "warn", "fail", "not_applicable"]
        sanity = _sanity({"gpt2": emitted,
                          "xlm-roberta": ["unverifiable"] * 4})
        report = _render(render, tmp_path, sanity)
        legend = report[report.index("Legend, in worsening order:"):]
        for sev in set(emitted) | {"unverifiable"}:
            assert f"| `{sev}` |" in legend, f"{sev} is in the matrix and not the legend"

    def test_a_severity_absent_from_the_run_is_absent_from_the_legend(self, render, tmp_path):
        sanity = _sanity({"gpt2": ["pass"] * 4, "xlm-roberta": ["pass"] * 4})
        report = _render(render, tmp_path, sanity)
        legend = report[report.index("Legend, in worsening order:"):]
        assert "| `pass` |" in legend
        assert "| `fail` |" not in legend

    def test_the_column_links_resolve_in_the_docs_page(self, render, tmp_path):
        """A header link into SANITY_CHECKS.md must reach a real heading."""
        import re

        sanity = _sanity({"gpt2": ["pass"] * 4, "xlm-roberta": ["pass"] * 4})
        report = _render(render, tmp_path, sanity)
        page = (REPO_ROOT / "docs" / "SANITY_CHECKS.md").read_text(encoding="utf-8")
        headings = set()
        for line in page.splitlines():
            if line.startswith("#"):
                text = line.lstrip("#").strip().lower()
                headings.add(re.sub(r"[^\w\s-]", "", text).replace(" ", "-"))
        anchors = re.findall(r"\]\(\.\./\.\./docs/SANITY_CHECKS\.md#([\w-]+)\)", report)
        assert anchors, "the health matrix emitted no column links"
        missing = [a for a in anchors if a not in headings]
        assert not missing, f"docs/SANITY_CHECKS.md has no heading for {missing}"

    def test_no_sanity_file_omits_the_section_rather_than_half_rendering_it(
            self, render, tmp_path):
        report = _render(render, tmp_path)
        assert "## Tokenizer health" not in report
        # The rest of the report is still written.
        assert "## Headline" in report

    def test_a_sanity_file_with_no_tokenizers_emits_no_section(self, render, tmp_path):
        """Present but empty is the case the missing-file check does not cover.

        A header over an empty table reads as nine tokenizers with nothing
        wrong, which is the opposite of what an empty file means.
        """
        report = _render(render, tmp_path, {"tokenizer_sanity_check": {"per_tokenizer": {}}})
        assert "## Tokenizer health" not in report
        assert "## Headline" in report

    def test_a_named_sanity_file_that_is_absent_is_an_error(self, render, tmp_path):
        """Silently skipping a file the caller named would hide a typo."""
        results_path = tmp_path / "analysis_results.json"
        results_path.write_text(json.dumps(_results()), encoding="utf-8")
        with pytest.raises(SystemExit, match="does not exist"):
            render.main(["--results", str(results_path),
                         "--output", str(tmp_path / "REPORT.md"),
                         "--sanity", str(tmp_path / "nope.json")])
