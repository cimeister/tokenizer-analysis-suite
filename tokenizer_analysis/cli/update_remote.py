"""Push a local results markdown file to a dedicated git branch on the remote.

Fetches the remote file, finds rows missing from it, merges them in
(local rows take priority for the same composite key), and pushes the
updated file — all without touching the working tree or current branch.

Each combination of dataset and normalization method produces its own
file on the branch (e.g. ``RESULTS_flores_bytes.md``).

Examples:
    # Push results/RESULTS_flores_bytes.md to origin/results
    uv run update-remote --results-file results/RESULTS_flores_bytes.md

    # Validate local file format without pushing
    uv run update-remote --results-file results/RESULTS_flores_bytes.md --validate-local-results

    # Remove your rows from a specific remote results file
    uv run update-remote --remove-my-results RESULTS_flores_bytes.md

    # Remove your rows from all remote results files at once
    uv run update-remote --remove-my-results --all

    # Custom remote and branch
    uv run update-remote --results-file results/RESULTS.md --remote upstream --branch leaderboard
"""
import argparse
import getpass
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from tokenizer_analysis.visualization.markdown_tables import (
    MarkdownTableGenerator,
    _plots_dir_for_results_file,
    generate_bar_plots_from_markdown,
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


def _list_remote_results_files(remote: str, branch: str) -> List[str]:
    """Return a list of ``RESULTS*.md`` filenames on the remote branch."""
    remote_ref = f"{remote}/{branch}"
    _run_git('fetch', remote, branch, check=False)
    ls_result = _run_git('ls-tree', '--name-only', remote_ref, check=False)
    if ls_result.returncode != 0 or not ls_result.stdout.strip():
        return []
    return [
        name.strip()
        for name in ls_result.stdout.strip().splitlines()
        if name.strip().startswith("RESULTS") and name.strip().endswith(".md")
    ]


def _resolve_remove_targets(
    remove_arg: str,
    *,
    all_files: bool,
    remote_filename_override: Optional[str],
    results_file: Optional[str],
    remote: str,
    branch: str,
) -> List[str]:
    """Determine which remote file(s) to remove results from.

    Parameters
    ----------
    remove_arg : str
        The value passed to ``--remove-my-results``.  ``'__prompt__'`` when
        the flag was used without a value.
    all_files : bool
        ``--all`` flag: remove from every ``RESULTS*.md`` on the branch.
    remote_filename_override : str | None
        Explicit ``--remote-filename`` value (legacy; takes precedence).
    results_file : str | None
        ``--results-file`` value used to derive a default filename.
    remote, branch : str
        Git remote and branch names.

    Returns
    -------
    list[str]
        One or more remote filenames to process.
    """
    # --remote-filename takes precedence (legacy path)
    if remote_filename_override:
        return [remote_filename_override]

    # --all: every RESULTS file on the branch
    if all_files:
        files = _list_remote_results_files(remote, branch)
        if not files:
            logger.error(f"No RESULTS*.md files found on {remote}/{branch}.")
        return files

    # Explicit filename given as positional value of --remove-my-results
    if remove_arg != '__prompt__':
        return [remove_arg]

    # No filename given — list available files and let the user choose
    files = _list_remote_results_files(remote, branch)
    if not files:
        logger.error(f"No RESULTS*.md files found on {remote}/{branch}.")
        return []

    if len(files) == 1:
        logger.info(f"Only one results file on branch: {files[0]}")
        return files

    print(f"\nResults files on {remote}/{branch}:")
    for i, name in enumerate(files, 1):
        print(f"  {i}) {name}")
    print(f"  a) All of the above")

    choice = input("\nWhich file to remove your results from? [1]: ").strip()
    if not choice or choice == "1":
        return [files[0]]
    if choice.lower() == 'a':
        return files
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            return [files[idx]]
    except ValueError:
        # Maybe they typed the filename directly
        if choice in files:
            return [choice]
    logger.error(f"Invalid selection: {choice}")
    return []


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
    - Every row has a valid composite key (reconstructed by
      ``parse_existing_markdown`` from either old-format ``name (user, dataset)``
      or new-format ``name [Nk]`` with User + Dataset columns)
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
        # parse_existing_markdown always reconstructs "name (user, dataset)"
        # composite keys, so validate that format
        if not COMPOSITE_KEY_PATTERN.match(tok_key):
            # Check if User and Dataset columns are non-empty (new format)
            user_val = row_map.get('User', '').strip()
            dataset_val = row_map.get('Dataset', '').strip()
            if not user_val or not dataset_val:
                errors.append(
                    f"  Row '{tok_key}': Could not determine composite key. "
                    "Ensure User and Dataset columns are non-empty."
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
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.md', delete=False, encoding='utf-8',
    ) as tmp:
        tmp.write(show_result.stdout)
        tmp_path = tmp.name

    try:
        headers, rows = MarkdownTableGenerator.parse_existing_markdown(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not rows:
        logger.info(f"Remote {remote_filename} has no rows.")
        return True

    # Filter out current user's rows by matching the User column value
    # (case-insensitive to handle OS username casing differences).
    kept = {
        k: v for k, v in rows.items()
        if v.get('User', '').strip().lower() != username.lower()
    }
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
        display = row_map.get('Tokenizer', key)
        row = [display] + [row_map.get(h, '---') for h in data_headers]
        table_rows.append(row)

    # Write cleaned file to a temp directory using the real remote_filename
    # so that _plots_dir_for_results_file() derives the correct plot subdir.
    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_path = os.path.join(tmp_dir, remote_filename)
        if table_rows:
            md = MarkdownTableGenerator._render_markdown(
                full_headers, separator, table_rows
            )
        else:
            md = MarkdownTableGenerator._render_markdown(
                full_headers, separator, []
            )
        Path(tmp_path).write_text(md, encoding='utf-8')

        # Regenerate bar plots from the cleaned markdown (if rows remain)
        plot_dir = None
        if table_rows:
            try:
                plot_dir = generate_bar_plots_from_markdown(tmp_path)
            except Exception as e:
                logger.warning(f"Bar plot generation failed during removal: {e}")

        success = push_results_to_branch(
            filepath=tmp_path,
            remote=remote,
            branch=branch,
            commit_message=f"Remove results for user '{username}' from {remote_filename}",
            skip_merge=True,
            remote_filename=remote_filename,
            plot_dir=plot_dir,
        )
    finally:
        shutil.rmtree(tmp_dir)

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
        nargs='?',
        const='__prompt__',
        default=None,
        metavar='REMOTE_FILE',
        help="Remove all your rows from a remote results file. "
             "Provide the filename on the results branch "
             "(e.g. RESULTS_flores_bytes.md). "
             "If omitted, lists available files and prompts for selection. "
             "Use --all to remove from every RESULTS file on the branch.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="When combined with --remove-my-results, remove your rows "
             "from every RESULTS*.md file on the results branch.",
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
    if args.remove_my_results is not None:
        remote_fnames = _resolve_remove_targets(
            args.remove_my_results,
            all_files=args.all,
            remote_filename_override=args.remote_filename,
            results_file=args.results_file,
            remote=args.remote,
            branch=args.branch,
        )
        if not remote_fnames:
            logger.error("No results files to process.")
            sys.exit(1)

        any_failed = False
        for remote_fname in remote_fnames:
            success = remove_my_results(
                args.remote, args.branch, remote_filename=remote_fname
            )
            if success:
                print(f"Your results removed from {args.remote}/{args.branch}:{remote_fname}")
            else:
                logger.error(f"Failed to remove results from {remote_fname}.")
                any_failed = True

        if any_failed:
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

    # Determine plot directory for bar plots (if it exists)
    plot_dir = _plots_dir_for_results_file(results_file)
    if not os.path.isdir(plot_dir):
        plot_dir = None

    logger.info(f"Pushing {results_file} as {remote_fname} to {args.remote}/{args.branch}")
    success = push_results_to_branch(
        filepath=results_file,
        remote=args.remote,
        branch=args.branch,
        commit_message=args.commit_message,
        remote_filename=remote_fname,
        plot_dir=plot_dir,
    )

    if success:
        print(f"Results pushed to {args.remote}/{args.branch}:{remote_fname}")
    else:
        logger.error("Push failed. Your local file is fine — try again later.")
        sys.exit(1)


if __name__ == "__main__":
    main()
