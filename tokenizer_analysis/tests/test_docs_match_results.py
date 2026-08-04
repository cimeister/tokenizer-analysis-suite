"""Every results key the documentation names must exist in a real results file.

README.md and METRICS.md tell a reader where to find each metric's value.  When
the output schema moves and the documentation does not, those paths send the
reader to a KeyError, or worse to a neighbouring value that looks plausible.

That happened between 1.0.0 and 1.0.2.  README claimed five metrics had no
``global`` block forty lines after stating that every metric has one, and
METRICS.md gave the trigram headline as
``trigram_entropy.per_tokenizer.<tok>.global_trigram_entropy``, a flat key the
slimming layer had stopped emitting.  Both were found by reading the documents
against the output by hand.  This test does it mechanically instead.

The reference file is the committed open-source benchmark, which is a real run
over nine tokenizers and thirteen languages.  Metrics absent from it, such as
``morphscore`` when the run did not enable MorphScore, are skipped rather than
failed: this test checks that a documented path resolves, not that the
benchmark exercises every metric.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "benchmarks" / "open_source" / "analysis_results.json"
DOCS = ("README.md", "METRICS.md")

# A backtick-quoted dotted path, e.g. `fertility.per_tokenizer.<tok>.global`.
PATH_PATTERN = re.compile(r"`([a-z0-9_]+(?:\.[a-zA-Z0-9_<>]+)+)`")

# Paths that appear in prose describing a rename rather than naming a key of
# the published file.  Each needs the reason it is here, so the list cannot
# grow into a way of silencing a real break.
# Empty since 1.0.2. It held `morphscore.per_tokenizer.<tok>.summary`, on the
# reasoning that README named it as the input of a rename rather than as a key
# of the published file. That reasoning stopped holding when slimming became a
# strict projection: the rename now happens before either file is written, so
# the key exists in neither, and the whitelist was suppressing the failure that
# would have caught the stale README paragraph. Add an entry here only with the
# reason it is not simply a stale document.
KNOWN_PRE_SLIM_PATHS: set = set()


def _documented_paths():
    """Every dotted path the docs name, with the file and line it came from."""
    for name in DOCS:
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in PATH_PATTERN.finditer(line):
                yield name, line_no, match.group(1)


def _resolve(node, parts, tokenizer):
    """Follow a dotted path, substituting a real key for <tok> and <lang>."""
    for part in parts:
        if not isinstance(node, dict):
            return False, None
        if part == "<tok>":
            part = tokenizer
        elif part == "<lang>":
            if not node:
                return False, None
            part = sorted(node)[0]
        if part not in node:
            return False, None
        node = node[part]
    return True, node


@pytest.mark.skipif(not RESULTS.is_file(), reason=f"{RESULTS} is not present")
def test_documented_result_paths_exist():
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    tokenizer = sorted(results["fertility"]["per_tokenizer"])[0]

    checked = 0
    unresolved = []
    for doc, line_no, dotted in _documented_paths():
        parts = dotted.split(".")
        # Only paths rooted at a metric this results file holds are checkable.
        if parts[0] not in results or dotted in KNOWN_PRE_SLIM_PATHS:
            continue
        checked += 1
        ok, _ = _resolve(results, parts, tokenizer)
        if not ok:
            unresolved.append(f"{doc}:{line_no} names `{dotted}`, which the results file does not hold")

    assert checked > 0, "no documented result paths were found to check"
    assert not unresolved, "\n".join(unresolved)
