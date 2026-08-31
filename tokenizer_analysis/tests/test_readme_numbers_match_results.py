"""The README's benchmark numbers must equal the committed results file.

README's "Results for nine open-source tokenizers" quotes ten values from
``benchmarks/open_source/analysis_results.json``.  Nothing regenerates the
README: ``benchmarks/open_source/run.sh`` rewrites ``REPORT.md`` and the results
file and leaves the front page alone, so a rerun that moves a value leaves the
README stating the old one with no test failing.

That failure mode is not hypothetical here.  The Gini figures in ``REPORT.md``
were typed by hand until 1.0.3 and drifted, which is why
``render_report.py`` derives them now.  Putting hand-typed numbers back on the
README reintroduces it one file over, so each one is derived here and asserted
to appear.

This is a different check from ``test_docs_match_results``, which resolves
documented key *paths* and compares no value.  The README prose quotes values
and names no paths, so that test does not reach it.

What it does not catch: prose edited to say the opposite of what the data shows
while every digit stays put.  Rewriting "the order reverses" to "the order
holds" leaves all three tests green.  These guard the numbers against a
regeneration that moves them, and the claims against data that stops supporting
them.  Neither reads the sentences.
"""

import json
import re
from pathlib import Path

import pytest
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "benchmarks" / "open_source" / "analysis_results.json"
README = REPO_ROOT / "README.md"

# The heading of the section under test.  Restricting the search to it stops a
# number matching some unrelated figure elsewhere on the page.
SECTION_HEADING = "## Results for nine open-source tokenizers"


def _section(text: str) -> str:
    start = text.index(SECTION_HEADING)
    end = text.index("\n## ", start + len(SECTION_HEADING))
    return text[start:end]


def _expected(results: dict) -> list:
    """Every number the section quotes, derived from the results file.

    Returns (what it measures, rendered string) so a failure names the
    quantity rather than only the digits.
    """
    comp = {k: v["global"]["compression_rate"]
            for k, v in results["compression_rate"]["per_tokenizer"].items()}
    eng = {k: v["per_language"]["eng_Latn"]["compression_rate"]
           for k, v in results["compression_rate"]["per_tokenizer"].items()}
    vocab = {k: v["global"]["vocab_size"]
             for k, v in results["vocabulary_utilization"]["per_tokenizer"].items()}
    gini = results["tokenizer_fairness_gini"]["per_tokenizer"]
    by_byte = {k: v["global"]["gini_coefficient"] for k, v in gini.items()}
    by_line = {k: v["per_line_normalization"]["gini_coefficient"] for k, v in gini.items()}
    ops = results["operator_isolation_rate"]["per_tokenizer"]

    order = sorted(comp)
    vocab_rho = spearmanr([vocab[k] for k in order], [comp[k] for k in order]).statistic
    gini_rho = spearmanr([by_byte[k] for k in order], [by_line[k] for k in order]).statistic

    bert = ops["bert-base"]["global"]
    others = [v["global"]["overall_compound_preservation_rate"]
              for k, v in ops.items() if k != "bert-base"]

    return [
        ("GPT-2 bytes per token, pooled", f"{comp['gpt2']:.3f}"),
        ("XLM-RoBERTa bytes per token, pooled", f"{comp['xlm-roberta']:.3f}"),
        ("GPT-2 bytes per token on eng_Latn", f"{eng['gpt2']:.3f}"),
        ("XLM-RoBERTa bytes per token on eng_Latn", f"{eng['xlm-roberta']:.3f}"),
        ("Spearman, vocabulary size against bytes per token", f"{vocab_rho:.3f}"),
        ("Spearman, per-byte Gini against per-line Gini", f"{gini_rho:.3f}"),
        ("Llama 3 per-byte Gini", f"{by_byte['llama-3']:.4f}"),
        ("Llama 3 per-line Gini", f"{by_line['llama-3']:.4f}"),
        ("XLM-RoBERTa per-line Gini", f"{by_line['xlm-roberta']:.4f}"),
        ("XLM-RoBERTa per-byte Gini", f"{by_byte['xlm-roberta']:.4f}"),
        ("BERT operator isolation rate", f"{bert['overall_isolation_rate']:.3f}"),
        ("BERT compound preservation rate", f"{bert['overall_compound_preservation_rate']:.3f}"),
        ("compound operators scored", f"{bert['total_compound_operators']:,}"),
        ("lowest compound preservation of the other eight", f"{min(others):.3f}"),
        ("highest compound preservation of the other eight", f"{max(others):.3f}"),
    ]


@pytest.mark.skipif(not RESULTS.is_file(), reason=f"{RESULTS} is not present")
def test_readme_quotes_the_committed_numbers():
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    section = _section(README.read_text(encoding="utf-8"))

    missing = [f"{what} is {value} in {RESULTS.name}, and the README section does not say it"
               for what, value in _expected(results) if value not in section]
    assert not missing, "\n".join(missing)


@pytest.mark.skipif(not RESULTS.is_file(), reason=f"{RESULTS} is not present")
def test_readme_claims_still_hold():
    """The section makes four claims that no single number would catch.

    Each is the point of the sentence it sits in, so a rerun that reversed one
    would leave every digit above still present and the prose wrong.
    """
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    comp = {k: v["global"]["compression_rate"]
            for k, v in results["compression_rate"]["per_tokenizer"].items()}
    eng = {k: v["per_language"]["eng_Latn"]["compression_rate"]
           for k, v in results["compression_rate"]["per_tokenizer"].items()}
    gini = results["tokenizer_fairness_gini"]["per_tokenizer"]
    by_byte = {k: v["global"]["gini_coefficient"] for k, v in gini.items()}
    by_line = {k: v["per_line_normalization"]["gini_coefficient"] for k, v in gini.items()}

    assert min(comp, key=comp.get) == "gpt2" and max(comp, key=comp.get) == "xlm-roberta", (
        "the README gives GPT-2 and XLM-RoBERTa base as the ends of the pooled "
        f"compression range; they are now {min(comp, key=comp.get)} and {max(comp, key=comp.get)}"
    )
    assert max(eng, key=eng.get) == "gpt2" and min(eng, key=eng.get) == "xlm-roberta", (
        "the README says the order reverses on English; it no longer does"
    )
    assert min(by_byte, key=by_byte.get) == "llama-3", (
        "the README gives Llama 3 as the lowest per-byte Gini; it is now "
        f"{min(by_byte, key=by_byte.get)}"
    )
    assert min(by_line, key=by_line.get) == "xlm-roberta", (
        "the README gives XLM-RoBERTa base as the lowest per-line Gini; it is now "
        f"{min(by_line, key=by_line.get)}"
    )


@pytest.mark.skipif(not RESULTS.is_file(), reason=f"{RESULTS} is not present")
def test_readme_corpus_description_matches_the_run():
    """The three corpus sizes in the opening sentence.

    render_report.py hardcodes these in REPORT.md, so REPORT.md cannot be the
    thing that checks them.  per_domain counts can.
    """
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    section = _section(README.read_text(encoding="utf-8"))
    domains = results["reconstruction_fidelity"]["per_tokenizer"]["gpt2"]["per_domain"]

    code = {k: v["count"] for k, v in domains.items() if k.startswith("code_")}
    assert f"{sum(code.values())} source files across {len(code)} programming languages" in section
    assert f"{domains['math']['count']}\nbundled math expressions" in section or \
           f"{domains['math']['count']} bundled math expressions" in section
    n_langs = results["run_metadata"]["corpus"]["digest"]["n_languages"]
    assert f"{n_langs} FLORES+ languages" in section
