#!/usr/bin/env python
"""Push a local RESULTS.md to a dedicated git branch on the remote.

Fetches the remote RESULTS.md, finds rows missing from it, merges them in
(local rows take priority for the same composite key), and pushes the
updated file — all without touching the working tree or current branch.

Examples:
    # Push results/RESULTS.md to origin/results (prompts for dataset name)
    python scripts/update_remote.py

    # Skip the prompt — use "default" as dataset name
    python scripts/update_remote.py --default-dataset

    # Specify dataset name directly
    python scripts/update_remote.py --dataset flores

    # Explicit file path
    python scripts/update_remote.py --results-file my_results.md

    # Custom remote and branch
    python scripts/update_remote.py --remote upstream --branch leaderboard
"""
import argparse
import getpass
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


def _set_dataset_in_file(filepath: str, dataset: str) -> None:
    """Update the Dataset column and composite keys for the current user's
    rows in *filepath*.

    Rows belonging to other users are left untouched.
    """
    headers, rows = MarkdownTableGenerator.parse_existing_markdown(filepath)
    if not rows:
        return

    username = getpass.getuser()

    # Pattern to match composite keys belonging to the current user
    # e.g. "bpe (alice)" or "bpe (alice, flores)"
    user_pattern = re.compile(
        r'^(?P<tok>.+?)\s*\('
        + re.escape(username)
        + r'(?:,\s*[^)]+)?\)$'
    )

    updated_rows: dict = {}
    for key, row_map in rows.items():
        m = user_pattern.match(key)
        if m:
            tok_name = m.group('tok').strip()
            new_key = f'{tok_name} ({username}, {dataset})'
            new_map = dict(row_map)
            new_map['Tokenizer'] = new_key
            new_map['Dataset'] = dataset
            updated_rows[new_key] = new_map
        else:
            updated_rows[key] = dict(row_map)

    # Ensure Dataset column exists in headers
    if 'Dataset' not in headers:
        # Insert before User if present, otherwise append
        if 'User' in headers:
            idx = headers.index('User')
            headers.insert(idx, 'Dataset')
        else:
            headers.append('Dataset')

    data_headers = [h for h in headers if h != 'Tokenizer']
    full_headers = ['Tokenizer'] + data_headers
    separator = ['---'] * len(full_headers)

    table_rows = []
    for key in updated_rows:
        row_map = updated_rows[key]
        row = [key] + [row_map.get(h, '---') for h in data_headers]
        table_rows.append(row)

    md = MarkdownTableGenerator._render_markdown(full_headers, separator, table_rows)

    from pathlib import Path
    Path(filepath).write_text(md, encoding='utf-8')
    logger.info(f"Updated dataset to '{dataset}' for user '{username}' in {filepath}")


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

    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to label rows with (e.g. 'flores', 'cc100')",
    )
    dataset_group.add_argument(
        "--default-dataset",
        action="store_true",
        help="Use 'default' as dataset name without prompting",
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        logger.error(f"File not found: {args.results_file}")
        logger.error("Run the analysis with --update-results-md first to generate it.")
        sys.exit(1)

    # Determine dataset name
    if args.dataset:
        dataset = args.dataset
    elif args.default_dataset:
        dataset = "default"
    else:
        dataset = input("Enter dataset name (or press Enter for 'default'): ").strip()
        if not dataset:
            dataset = "default"

    # Update the local file with the dataset name before pushing
    _set_dataset_in_file(args.results_file, dataset)

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
        logger.error("Push failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
