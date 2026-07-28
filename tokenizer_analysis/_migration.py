"""Errors for names removed between tokenizer-analysis-suite 0.x and 1.0.0.

The CLI already fails removed flags with a message that names the replacement
and points at MIGRATION.md (see ``cli/run_analysis.py``). Without this module the
Python API did not: ``from tokenizer_analysis import MorphologicalMetrics`` raised
a bare ``ImportError`` with no indication that MorphScore replaced it, and
``from tokenizer_analysis.constants import DataProcessing`` gave no hint that the
namespace classes became module-level names.

Only names that existed in a *published* predecessor release are listed here.
Anything removed before 1.0.0 ever shipped has no installed base to break, so it
belongs in the changelog rather than in a runtime shim.

Usage: a module defines ``__getattr__`` (PEP 562) delegating to
:func:`make_module_getattr`, which raises only when someone actually touches the
removed name, so it costs nothing at import time.

One deliberate deviation from PEP 562, which says ``__getattr__`` should raise
``AttributeError``. The common failure is ``from tokenizer_analysis import
MorphologicalMetrics``, and the ``IMPORT_FROM`` opcode catches ``AttributeError``
and replaces it with its own "cannot import name" message, so the migration text
never reaches the user. Raising ``ImportError`` instead propagates unchanged
through both ``from X import Y`` and ``X.Y``. Verified against CPython 3.11.

The cost is that ``hasattr(module, "MorphologicalMetrics")`` raises instead of
returning ``False``. That is confined to the removed names: anything else gets a
normal ``AttributeError``, so ordinary introspection is unaffected.
"""

from typing import Dict, Iterable

MIGRATION_DOC = "MIGRATION.md"
REMOVED_IN = "1.0.0"

# name -> what to use instead. Keep the replacement concrete enough to act on.
_REMOVED: Dict[str, str] = {
    "MorphologicalMetrics": (
        "the standalone morphological boundary metric was removed; use MorphScore "
        "via MorphScoreMetrics (tokenizer_analysis.metrics.morphscore), or the "
        "--morphscore / --morphscore-config flags"
    ),
    "MorphologicalDataLoader": (
        "the standalone morphological loader was removed along with its metric; "
        "MorphScore loads its own data, configured by data_dir"
    ),
    # Constants moved from namespace classes to module-level names.
    "TextProcessing": (
        "constants are module-level names now; import the constant directly, e.g. "
        "`from tokenizer_analysis.constants import MIN_PARAGRAPH_LENGTH` instead of "
        "`TextProcessing.MIN_PARAGRAPH_LENGTH`"
    ),
    "DataProcessing": (
        "constants are module-level names now; import the constant directly, e.g. "
        "`from tokenizer_analysis.constants import DEFAULT_CHUNK_SIZE` instead of "
        "`DataProcessing.DEFAULT_CHUNK_SIZE`"
    ),
    "Statistics": (
        "constants are module-level names now; import the constant directly, e.g. "
        "`from tokenizer_analysis.constants import DEFAULT_RENYI_ALPHAS` instead of "
        "`Statistics.DEFAULT_RENYI_ALPHAS`"
    ),
    "Validation": (
        "constants are module-level names now; import the constant directly, e.g. "
        "`from tokenizer_analysis.constants import MIN_LANGUAGES_FOR_GINI` instead of "
        "`Validation.MIN_LANGUAGES_FOR_GINI`"
    ),
}


def removed_name_error(module_name: str, attr: str) -> Exception:
    """Build the error for *attr*, whether or not it is a known removed name.

    Known removed names get an ``ImportError`` carrying the migration text, for
    the reason in the module docstring. Everything else gets the ordinary
    ``AttributeError`` so normal attribute lookup and ``hasattr`` are unchanged.

    Returns rather than raises so the caller's ``__getattr__`` owns the raise and
    the traceback points at the access that failed.
    """
    replacement = _REMOVED.get(attr)
    if replacement is None:
        return AttributeError(f"module {module_name!r} has no attribute {attr!r}")
    return ImportError(
        f"{attr} was removed from {module_name} in tokenizer-intrinsic-evals "
        f"{REMOVED_IN}: {replacement}. See {MIGRATION_DOC}."
    )


def make_module_getattr(module_name: str, public: Iterable[str] = ()):
    """Return a PEP 562 ``__getattr__`` for *module_name*.

    ``__getattr__`` runs only after normal lookup fails, so this never shadows a
    name the module actually defines. *public* is accepted for readability at the
    call site and is not otherwise used.
    """
    del public

    def __getattr__(attr: str):
        raise removed_name_error(module_name, attr)

    return __getattr__
