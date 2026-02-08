#!/usr/bin/env python
"""Push a local RESULTS.md to a dedicated git branch on the remote.

Fetches the remote RESULTS.md, finds rows missing from it, merges them in
(local rows take priority for the same composite key), and pushes the
updated file — all without touching the working tree or current branch.

Examples:
    # Push results/RESULTS.md to origin/results (defaults)
    python scripts/update_remote.py

    # Explicit file path
    python scripts/update_remote.py --results-file my_results.md

    # Custom remote and branch
    python scripts/update_remote.py --remote upstream --branch leaderboard
"""
import argparse
import logging
import os
import sys

from tokenizer_analysis.visualization.markdown_tables import push_results_to_branch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        logger.error(f"File not found: {args.results_file}")
        logger.error("Run the analysis with --update-results-md first to generate it.")
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
