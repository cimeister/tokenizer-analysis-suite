"""Tests for CLI argument handling and language-config parsing.

Neither had any coverage before 1.0, and both carried defects that produced
wrong results rather than errors: the CLI silently substituted demo data for the
user's, and LanguageMetadata read group keys no shipped config writes.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import requires_flores

from tokenizer_analysis.cli.run_analysis import (
    build_parser,
    _validate_corpus_source,
    slim_results_for_json,
)
from tokenizer_analysis.config.language_metadata import LanguageMetadata


REPO_ROOT = Path(__file__).resolve().parents[2]


def _args(**overrides):
    """Parse an empty command line, then override specific fields."""
    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestCorpusSourceValidation:
    """The three ways to supply a corpus are mutually exclusive, and one is required.

    `--use-sample-data` used to win silently: it overwrote --tokenizer-config,
    --language-config and --measurement-config with the demo values, so a run
    naming the user's tokenizers and corpus produced a complete, plausible
    results file describing the bundled demo instead.
    """

    def test_no_corpus_source_is_an_error(self):
        with pytest.raises(ValueError) as excinfo:
            _validate_corpus_source(_args(tokenizer_config="t.json"))
        message = str(excinfo.value)
        for flag in ("--input", "--language-config", "--use-sample-data"):
            assert flag in message, f"error should name {flag}"

    @pytest.mark.parametrize(
        "conflicting",
        ["tokenizer_config", "language_config", "input", "measurement_config"],
    )
    def test_sample_data_conflicts_with_user_config(self, conflicting):
        with pytest.raises(ValueError) as excinfo:
            _validate_corpus_source(
                _args(use_sample_data=True, **{conflicting: "x.json"})
            )
        assert "--use-sample-data" in str(excinfo.value)
        assert conflicting.replace("_", "-") in str(excinfo.value)

    def test_input_conflicts_with_language_config(self):
        with pytest.raises(ValueError) as excinfo:
            _validate_corpus_source(
                _args(tokenizer_config="t.json", input="c.txt",
                      language_config="l.json")
            )
        assert "--input" in str(excinfo.value)
        assert "--language-config" in str(excinfo.value)

    @pytest.mark.parametrize("source", [
        {"use_sample_data": True},
        {"tokenizer_config": "t.json", "input": "c.txt"},
        {"tokenizer_config": "t.json", "language_config": "l.json"},
    ])
    def test_each_single_source_is_accepted(self, source):
        _validate_corpus_source(_args(**source))

    def test_pretokenized_mode_is_validated_separately(self):
        """A cache supplies its own data, so the corpus flags do not apply."""
        _validate_corpus_source(_args(tokenized_data_file="cache.pkl"))


class TestRemovedFlagsFailLoudly:
    """A removed flag must name its replacement, not produce an argparse error."""

    def _run(self, *flags):
        return subprocess.run(
            [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
             "--use-sample-data", *flags],
            cwd=REPO_ROOT, capture_output=True, timeout=300,
        )

    def test_morphological_config_points_at_morphscore(self):
        proc = self._run("--morphological-config", "x.json")
        assert proc.returncode != 0
        message = proc.stderr.decode(errors="replace")
        assert "--morphscore" in message
        assert "MIGRATION.md" in message

    def test_removed_latex_table_type_lists_the_valid_ones(self):
        proc = self._run("--latex-table-types", "morphological")
        assert proc.returncode != 0
        message = proc.stderr.decode(errors="replace")
        assert "comprehensive" in message, "should list the valid table types"


class TestLanguageMetadataGroupKeys:
    """Group accessors read the key shipped configs actually write.

    They read `script_families`/`resource_levels` (plural) while every config in
    configs/ and the CLI's own sample metadata write the singular form, so
    get_script_families() returned [] and get_script_family() returned 'Unknown'
    for every language. The CLI happened to work because it passes the config's
    own keys through a generic path; the accessors, which are public API, did not.
    """

    @pytest.fixture()
    def core_config(self):
        path = REPO_ROOT / "configs" / "core_lang_config.json"
        if not path.exists():
            pytest.skip("configs/core_lang_config.json not present")
        return LanguageMetadata(str(path))

    def test_singular_keys_resolve(self, core_config):
        families = core_config.get_script_families()
        assert families, "no script families resolved from a config that defines them"
        assert core_config.get_script_family("eng_Latn") != "Unknown"

    def test_plural_keys_also_resolve(self, tmp_path):
        cfg = tmp_path / "plural.json"
        cfg.write_text(json.dumps({
            "languages": {"eng_Latn": {"data_path": "x.txt"}},
            "analysis_groups": {"script_families": {"Latin": ["eng_Latn"]}},
        }))
        meta = LanguageMetadata(str(cfg))
        assert meta.get_script_families() == ["Latin"]
        assert meta.get_script_family("eng_Latn") == "Latin"

    def test_unknown_language_is_unknown_not_an_error(self, core_config):
        assert core_config.get_script_family("zzz_Zzzz") == "Unknown"


class TestLanguageMetadataShortForm:
    """The `{"code": "/path"}` form the README documents must work.

    It raised `AttributeError: 'str' object has no attribute 'get'`, so the
    documented shorthand was unusable.
    """

    @pytest.fixture()
    def short_form(self, tmp_path):
        cfg = tmp_path / "short.json"
        cfg.write_text(json.dumps({"languages": {"en": "/data/en.txt"}}))
        return LanguageMetadata(str(cfg))

    def test_data_path_resolves(self, short_form):
        assert short_form.get_data_path("en") == "/data/en.txt"

    def test_language_paths_resolves(self, short_form):
        assert short_form.get_language_paths() == {"en": "/data/en.txt"}

    def test_name_falls_back_to_the_code(self, short_form):
        assert short_form.get_language_name("en") == "en"

    def test_long_form_still_works(self, tmp_path):
        cfg = tmp_path / "long.json"
        cfg.write_text(json.dumps({
            "languages": {"en": {"name": "English", "data_path": "/data/en.txt"}}
        }))
        meta = LanguageMetadata(str(cfg))
        assert meta.get_data_path("en") == "/data/en.txt"
        assert meta.get_language_name("en") == "English"


class TestOperatorIsolationSlimProjection:
    """slim_results_for_json's operator_isolation_rate branch must expose the
    pooled rate and the prose/code/math split inside each tokenizer's own
    entry, not just per_language.

    math.py's DigitBoundaryMetrics.compute() already produces a pooled
    per_tokenizer/summary (weighted by operator instances across prose, code
    and math) and a by_domain split, but before this change the slim
    projection read only by_language, so analysis_results.json silently
    dropped both.
    """

    @staticmethod
    def _domain_block(isolated, total):
        """A minimal per-domain block shaped like _build_operator_results'
        output: per_tokenizer (unused here) plus the pooled summary."""
        return {
            "per_tokenizer": {"tok": {"by_category": {}, "by_language": {}}},
            "summary": {
                "tok": {
                    "overall_isolation_rate": isolated / total,
                    "overall_compound_preservation_rate": 0.0,
                    "total_operators": total,
                    "total_compound_operators": 0,
                }
            },
            "source": "fake corpus",
        }

    def test_global_and_by_domain_appear_alongside_per_language(self):
        metric_data = {
            "per_tokenizer": {
                "tok": {
                    "by_category": {"arithmetic": {"isolation_rate": 1.0, "total": 143}},
                    "by_language": {
                        "en": {
                            "by_category": {"arithmetic": {"isolation_rate": 1.0}},
                            "isolation_rate": 1.0,
                            "total": 143,
                        }
                    },
                }
            },
            "summary": {
                "tok": {
                    "overall_isolation_rate": 143 / 160,
                    "overall_compound_preservation_rate": 0.0,
                    "total_operators": 160,
                    "total_compound_operators": 0,
                }
            },
            "by_domain": {
                "prose": self._domain_block(8, 10),
                "code": self._domain_block(90, 100),
                "math": self._domain_block(45, 50),
            },
            "domain_operator_counts": {"tok": {"prose": 10, "code": 100, "math": 50}},
        }

        slimmed = slim_results_for_json({"operator_isolation_rate": metric_data})
        tok_out = slimmed["operator_isolation_rate"]["per_tokenizer"]["tok"]

        # global: the pooled rate plus its raw count, unchanged from summary.
        assert tok_out["global"] == metric_data["summary"]["tok"]

        # per_language: unchanged, by_category still stripped.
        assert tok_out["per_language"]["en"]["isolation_rate"] == pytest.approx(1.0)
        assert "by_category" not in tok_out["per_language"]["en"]

        # by_domain: all three domains, each with the same rate-plus-count
        # shape as global, and no leaked by_category.
        assert set(tok_out["by_domain"]) == {"prose", "code", "math"}
        assert tok_out["by_domain"]["prose"]["overall_isolation_rate"] == pytest.approx(0.8)
        assert tok_out["by_domain"]["code"]["overall_isolation_rate"] == pytest.approx(0.9)
        assert tok_out["by_domain"]["math"]["overall_isolation_rate"] == pytest.approx(0.9)
        assert "by_category" not in tok_out["by_domain"]["code"]


class TestCorpusSegmentationIsNotCumulative:
    """One segmentation per file, not several concatenated.

    extract_texts_with_fallback_strategies ran its strategies additively
    despite the name, so the same content came back twice under two
    segmentations and every per-document metric was computed over a corpus
    twice the size of the file. A line-per-document file of 10 or more lines
    was correct, which is why FLORES+ never showed it.
    """

    def test_paragraph_file_returns_paragraphs_only(self):
        from tokenizer_analysis.utils.text_utils import (
            extract_texts_with_fallback_strategies,
        )
        paragraphs = [
            "\n".join(f"Paragraph {p} line {l} with enough text here." for l in range(3))
            for p in range(12)
        ]
        texts = extract_texts_with_fallback_strategies("\n\n".join(paragraphs) + "\n", 2000)
        assert len(texts) == 12, (
            "12 paragraphs of 3 lines returned the 12 paragraphs and then all 36 "
            f"lines inside them; got {len(texts)}"
        )

    def test_short_line_file_returns_each_line_once(self):
        from tokenizer_analysis.utils.text_utils import (
            extract_texts_with_fallback_strategies,
        )
        # Under 10 lines took the sentence split as well. The sentence and the
        # line differ by the trailing period, so the duplicate check missed it.
        source = "\n".join(f"Sentence number {i} is here." for i in range(4)) + "\n"
        texts = extract_texts_with_fallback_strategies(source, 2000)
        assert len(texts) == 4, f"4 lines became {len(texts)} texts"

    def test_line_file_above_the_threshold_is_unchanged(self):
        """The case FLORES+ exercises, which was always correct."""
        from tokenizer_analysis.utils.text_utils import (
            extract_texts_with_fallback_strategies,
        )
        source = "\n".join(f"Sentence number {i} is here." for i in range(20)) + "\n"
        assert len(extract_texts_with_fallback_strategies(source, 2000)) == 20
