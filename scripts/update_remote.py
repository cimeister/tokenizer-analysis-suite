#!/usr/bin/env python
"""Legacy shim — use ``uv run update-remote`` instead."""
import sys
import warnings

warnings.warn(
    "scripts/update_remote.py is deprecated. Use 'uv run update-remote' instead.",
    DeprecationWarning,
    stacklevel=1,
)

from tokenizer_analysis.cli.update_remote import main

sys.exit(main() or 0)
