"""Tests for the removed-name errors in ``tokenizer_analysis/_migration.py``.

These exist because the failure they guard is silent-by-default: without the
shims, ``from tokenizer_analysis import MorphologicalMetrics`` raised a bare
``ImportError`` that said nothing about MorphScore, so a user upgrading from
tokenizer-analysis-suite got no pointer to the replacement.
"""
import warnings

import pytest

from tokenizer_analysis._migration import MIGRATION_DOC, removed_name_error


REMOVED_IMPORTS = [
    ("tokenizer_analysis", "MorphologicalMetrics"),
    ("tokenizer_analysis.metrics", "MorphologicalMetrics"),
    ("tokenizer_analysis.loaders", "MorphologicalDataLoader"),
    ("tokenizer_analysis.constants", "DataProcessing"),
    ("tokenizer_analysis.constants", "TextProcessing"),
    ("tokenizer_analysis.constants", "Statistics"),
    ("tokenizer_analysis.constants", "Validation"),
]


@pytest.mark.parametrize("module_name,attr", REMOVED_IMPORTS)
def test_removed_name_points_at_migration_guide(module_name, attr):
    """A removed name must fail with the replacement and a CHANGELOG.md pointer.

    The error has to survive ``from X import Y``, which catches AttributeError
    and substitutes its own message. That is why the shim raises ImportError.
    """
    module = __import__(module_name, fromlist=["_"])
    with pytest.raises(ImportError) as excinfo:
        getattr(module, attr)
    message = str(excinfo.value)
    assert attr in message
    assert MIGRATION_DOC in message
    # The message must say what to use instead, not merely that it is gone.
    assert "removed" in message and ":" in message


def test_unknown_name_still_raises_plain_attribute_error():
    """Only removed names get the ImportError treatment.

    Everything else keeps normal semantics, so ``hasattr`` and introspection
    behave as they would on any module.
    """
    import tokenizer_analysis

    with pytest.raises(AttributeError):
        tokenizer_analysis.definitely_not_a_real_name
    assert hasattr(tokenizer_analysis, "definitely_not_a_real_name") is False


def test_live_exports_are_unaffected():
    """The shim must not shadow names the package actually defines."""
    from tokenizer_analysis import UnifiedTokenizerAnalyzer  # noqa: F401
    from tokenizer_analysis.constants import DEFAULT_CHUNK_SIZE

    assert DEFAULT_CHUNK_SIZE == 500


def test_removed_name_error_falls_back_for_unknown_attr():
    err = removed_name_error("some.module", "NeverExisted")
    assert isinstance(err, AttributeError)
    assert not isinstance(err, ImportError)


def test_standard_tokenizer_class_emits_deprecation_warning():
    """The 'standard' class alias must warn through the warnings machinery.

    It previously used logger.warning, which is invisible under
    ``-W error::DeprecationWarning`` and to pytest's warning capture, so a
    consumer suppressing logs would never learn the alias was going away.
    """
    from tokenizer_analysis.core.tokenizer_wrapper import create_tokenizer_wrapper

    with pytest.warns(DeprecationWarning, match="'standard' tokenizer class"):
        create_tokenizer_wrapper(
            "demo", {"class": "standard", "path": "tokenizers/bpe.json"}
        )


def test_load_tokenizer_from_config_emits_deprecation_warning():
    from tokenizer_analysis.utils import load_tokenizer_from_config

    with pytest.warns(DeprecationWarning, match="create_tokenizer_wrapper"):
        load_tokenizer_from_config(
            {"class": "huggingface", "path": "tokenizers/bpe.json"}, name="demo"
        )
