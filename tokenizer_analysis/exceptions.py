"""Exceptions the console scripts turn into a message rather than a traceback.

``ConfigurationError`` lived in ``cli/run_analysis.py`` until 1.0.3. It moved
here because ``core/`` and ``metrics/`` never import from ``cli/``, and the
tokenizer registry in ``core/tokenizer_wrapper.py`` needs to raise it: an
unknown tokenizer class is the user's typo, not a defect in this package, so it
should read as one line rather than as a stack. ``cli/run_analysis.py``
re-exports the name, so an existing ``from tokenizer_analysis.cli.run_analysis
import ConfigurationError`` keeps working.
"""


class ConfigurationError(ValueError):
    """A config file or flag value the user named is missing or wrong.

    Kept separate from the package's own exceptions so a console script can
    print it without a traceback: the message names the flag, the file and the
    expected shape, which is the whole of what a caller needs.

    Subclasses ``ValueError`` because it did so when it lived in the CLI module,
    and callers catch it that way.
    """
