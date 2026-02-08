#!/usr/bin/env python
"""Push a local RESULTS.md to a dedicated git branch on the remote.

Fetches the remote RESULTS.md, finds rows missing from it, merges them in
(local rows take priority for the same composite key), and pushes the
updated file — all without touching the working tree or current branch.

Examples:
    # Push results/RESULTS.md to origin/results (defaults)
    python scripts/update_remote.py

    # Validate local file format without pushing
    python scripts/update_remote.py --validate-local-results

    # Explicit file path
    python scripts/update_remote.py --results-file my_results.md

    # Custom remote and branch
    python scripts/update_remote.py --remote upstream --branch leaderboard
"""
import argparse
import logging
import os
import re
import sys

from tokenizer_analysis.visualization.markdown_tables import (
    MarkdownTableGenerator,
    push_results_to_branch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {'Tokenizer', 'Dataset', 'User', 'Date'}
COMPOSITE_KEY_PATTERN = re.compile(r'^.+\s*\(.+,\s*.+\)$')


def validate_results_file(filepath: str) -> bool:
    """Check that *filepath* is a well-formed RESULTS.md.

    Validates:
    - File exists and is non-empty
    - Contains a markdown table with header + separator + data rows
    - Required columns (Tokenizer, Dataset, User, Date) are present
    - Every row uses the composite key format: ``tokenizer (user, dataset)``
    - Every row has the correct number of columns

    Returns True if valid, False otherwise (errors are logged).
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return False

    headers, rows = MarkdownTableGenerator.parse_existing_markdown(filepath)

    if not headers:
        logger.error("No markdown table found in the file.")
        return False

    # Check required columns
    missing = REQUIRED_COLUMNS - set(headers)
    if missing:
        logger.error(f"Missing required columns: {', '.join(sorted(missing))}")
        return False

    if not rows:
        logger.error("Table has headers but no data rows.")
        return False

    num_cols = len(headers)
    errors = []
    for tok_key, row_map in rows.items():
        # Check composite key format
        if not COMPOSITE_KEY_PATTERN.match(tok_key):
            errors.append(
                f"  Row '{tok_key}': Tokenizer key should be in format "
                "'name (user, dataset)'"
            )

        # Check column count (row_map includes Tokenizer column)
        row_cols = len(row_map)
        if row_cols != num_cols:
            errors.append(
                f"  Row '{tok_key}': expected {num_cols} columns, got {row_cols}"
            )

    if errors:
        logger.error("Validation errors:\n" + "\n".join(errors))
        return False

    logger.info(
        f"Validation passed: {len(rows)} rows, {num_cols} columns, "
        f"all composite keys well-formed."
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Push a local RESULTS.md to a dedicated git branch on the remote.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results/RESULTS.md",
        help="Path to the local RESULTS.md file (default: results/RESULTS.md)",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default="origin",
        help="Git remote name (default: origin)",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="results",
        help="Git branch name on the remote (default: results)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--validate-local-results",
        action="store_true",
        help="Validate the local RESULTS.md format and exit (no push)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        logger.error(f"File not found: {args.results_file}")
        logger.error("Run the analysis with --update-results-md first to generate it.")
        sys.exit(1)

    # Validate-only mode
    if args.validate_local_results:
        valid = validate_results_file(args.results_file)
        sys.exit(0 if valid else 1)

    # Always validate before pushing
    if not validate_results_file(args.results_file):
        logger.error("Fix the issues above before pushing.")
        sys.exit(1)

    logger.info(f"Pushing {args.results_file} to {args.remote}/{args.branch}")
    success = push_results_to_branch(
        filepath=args.results_file,
        remote=args.remote,
        branch=args.branch,
        commit_message=args.commit_message,
    )

    if success:
        print(f"Results pushed to {args.remote}/{args.branch}")
    else:
        logger.error("Push failed. Your local RESULTS.md is fine — try again later.")
        sys.exit(1)


if __name__ == "__main__":
    main()
