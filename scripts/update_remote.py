#!/usr/bin/env python
"""Push a local results markdown file to a dedicated git branch on the remote.

Fetches the remote file, finds rows missing from it, merges them in
(local rows take priority for the same composite key), and pushes the
updated file — all without touching the working tree or current branch.

Each combination of dataset and normalization method produces its own
file on the branch (e.g. ``RESULTS_flores_bytes.md``).

Examples:
    # Push results/RESULTS_flores_bytes.md to origin/results
    python scripts/update_remote.py --results-file results/RESULTS_flores_bytes.md

    # Validate local file format without pushing
    python scripts/update_remote.py --results-file results/RESULTS_flores_bytes.md --validate-local-results

    # Remove your rows from a specific remote results file
    python scripts/update_remote.py --remove-my-results --remote-filename RESULTS_flores_bytes.md

    # Custom remote and branch
    python scripts/update_remote.py --results-file results/RESULTS.md --remote upstream --branch leaderboard
"""
import argparse
import getpass
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizer_analysis.visualization.markdown_tables import (
    MarkdownTableGenerator,
    push_results_to_branch,
    results_filename,
    _run_git,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {'Tokenizer', 'Dataset', 'User', 'Date'}
COMPOSITE_KEY_PATTERN = re.compile(r'^.+\s*\(.+,\s*.+\)$')


def _derive_remote_filename(filepath: str) -> str:
    """Derive the remote filename from a local filepath.

    If the filename matches the ``RESULTS_*`` convention, use it as-is.
    Otherwise fall back to ``RESULTS.md``.
    """
    basename = os.path.basename(filepath)
    if basename.startswith("RESULTS") and basename.endswith(".md"):
        return basename
    return "RESULTS.md"


def validate_results_file(filepath: str) -> bool:
    """Check that *filepath* is a well-formed results markdown file.

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


def remove_my_results(
    remote: str,
    branch: str,
    remote_filename: str = "RESULTS.md",
) -> bool:
    """Fetch the remote results file, remove the current user's rows, and push back.

    Parameters
    ----------
    remote : str
        Git remote name.
    branch : str
        Branch on the remote.
    remote_filename : str
        Name of the file on the results branch to clean.

    Returns True on success, False on failure.
    """
    username = getpass.getuser()
    remote_ref = f"{remote}/{branch}"

    # Fetch remote branch
    _run_git('fetch', remote, branch, check=False)

    # Read remote file
    show_result = _run_git('show', f'{remote_ref}:{remote_filename}', check=False)
    if show_result.returncode != 0 or not show_result.stdout.strip():
        logger.error(f"No {remote_filename} found on {remote_ref}. Nothing to remove.")
        return False

    # Write to temp file and parse
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
        tmp.write(show_result.stdout)
        tmp_path = tmp.name

    try:
        headers, rows = MarkdownTableGenerator.parse_existing_markdown(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not rows:
        logger.info(f"Remote {remote_filename} has no rows.")
        return True

    # Filter out current user's rows
    user_pattern = re.compile(
        r'\(' + re.escape(username) + r'(?:,\s*[^)]+)?\)$'
    )
    kept = {k: v for k, v in rows.items() if not user_pattern.search(k)}
    removed = len(rows) - len(kept)

    if removed == 0:
        logger.info(f"No rows found for user '{username}' on remote. Nothing to remove.")
        return True

    logger.info(f"Removing {removed} row(s) for user '{username}' from remote {remote_filename}.")

    # Rebuild the file
    data_headers = [h for h in headers if h != 'Tokenizer']
    full_headers = ['Tokenizer'] + data_headers
    separator = ['---'] * len(full_headers)

    table_rows = []
    for key, row_map in kept.items():
        row = [key] + [row_map.get(h, '---') for h in data_headers]
        table_rows.append(row)

    # Write cleaned file to temp location
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.md', delete=False
    ) as tmp:
        if table_rows:
            md = MarkdownTableGenerator._render_markdown(
                full_headers, separator, table_rows
            )
        else:
            md = MarkdownTableGenerator._render_markdown(
                full_headers, separator, []
            )
        tmp.write(md)
        tmp_path = tmp.name

    try:
        success = push_results_to_branch(
            filepath=tmp_path,
            remote=remote,
            branch=branch,
            commit_message=f"Remove results for user '{username}' from {remote_filename}",
            skip_merge=True,
            remote_filename=remote_filename,
        )
    finally:
        os.unlink(tmp_path)

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Push a local results file to a dedicated git branch on the remote.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to the local results markdown file "
             "(default: results/RESULTS.md, or auto-derived from --dataset / --normalization-method)",
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
        help="Validate the local results file format and exit (no push)",
    )
    parser.add_argument(
        "--remove-my-results",
        action="store_true",
        help="Remove all your rows from the specified remote results file",
    )
    parser.add_argument(
        "--remote-filename",
        type=str,
        default=None,
        help="Explicit filename on the results branch (e.g. RESULTS_flores_bytes.md). "
             "When not provided it is derived from --results-file or defaults to RESULTS.md.",
    )
    args = parser.parse_args()

    # Remove mode — works directly on remote, no local file needed.
    # Requires --remote-filename to know which file to clean.
    if args.remove_my_results:
        remote_fname = args.remote_filename or _derive_remote_filename(args.results_file or "RESULTS.md")
        success = remove_my_results(
            args.remote, args.branch, remote_filename=remote_fname
        )
        if success:
            print(f"Your results removed from {args.remote}/{args.branch}:{remote_fname}")
        else:
            logger.error("Failed to remove results. Try again later.")
            sys.exit(1)
        return

    # Default local path: if --results-file is not given, try to find
    # RESULTS*.md files in the results/ directory.
    if args.results_file:
        results_file = args.results_file
    else:
        results_dir = "results"
        candidates = sorted(Path(results_dir).glob("RESULTS*.md")) if Path(results_dir).is_dir() else []
        if len(candidates) == 1:
            results_file = str(candidates[0])
            logger.info(f"Auto-discovered results file: {results_file}")
        elif len(candidates) > 1:
            logger.error(
                f"Multiple results files found in {results_dir}/:\n"
                + "\n".join(f"  {c}" for c in candidates)
                + "\nPlease specify one with --results-file."
            )
            sys.exit(1)
        else:
            results_file = os.path.join(results_dir, "RESULTS.md")

    if not os.path.exists(results_file):
        logger.error(f"File not found: {results_file}")
        logger.error("Run the analysis with --update-results-md first to generate it.")
        sys.exit(1)

    # Determine the remote filename from the local file (unless overridden)
    remote_fname = args.remote_filename or _derive_remote_filename(results_file)

    # Validate-only mode
    if args.validate_local_results:
        valid = validate_results_file(results_file)
        sys.exit(0 if valid else 1)

    # Always validate before pushing
    if not validate_results_file(results_file):
        logger.error("Fix the issues above before pushing.")
        sys.exit(1)

    logger.info(f"Pushing {results_file} as {remote_fname} to {args.remote}/{args.branch}")
    success = push_results_to_branch(
        filepath=results_file,
        remote=args.remote,
        branch=args.branch,
        commit_message=args.commit_message,
        remote_filename=remote_fname,
    )

    if success:
        print(f"Results pushed to {args.remote}/{args.branch}:{remote_fname}")
    else:
        logger.error("Push failed. Your local file is fine — try again later.")
        sys.exit(1)


if __name__ == "__main__":
    main()
