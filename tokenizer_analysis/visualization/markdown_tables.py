"""
Markdown table generation for tokenizer analysis results.

Supports cumulative updates: new tokenizer rows are merged into an existing
RESULTS.md file so that a single table grows over successive runs.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class MarkdownTableGenerator:
    """Generate and cumulatively update Markdown tables from tokenizer analysis results."""

    def __init__(self, results: Dict[str, Any], tokenizer_names: List[str]):
        self.results = results
        self.tokenizer_names = tokenizer_names

        # Ordered list of metric configurations (determines column order)
        self.metric_configs: List[Dict[str, Any]] = [
            {
                'key': 'vocab_size',
                'title': 'Vocab Size',
                'key_path': ['vocabulary_utilization', 'per_tokenizer'],
                'value_key': 'global_vocab_size',
                'stat_key': None,
                'format': '{:,}',
            },
            {
                'key': 'fertility',
                'title': 'Fertility',
                'key_path': ['fertility', 'per_tokenizer'],
                'value_key': 'global',
                'stat_key': 'mean',
                'format': '{:.3f}',
            },
            {
                'key': 'compression_rate',
                'title': 'Compression Rate',
                'key_path': ['compression_ratio', 'per_tokenizer'],
                'value_key': 'global',
                'stat_key': 'mean',
                'format': '{:.3f}',
            },
            {
                'key': 'vocabulary_utilization',
                'title': 'Vocab Util.',
                'key_path': ['vocabulary_utilization', 'per_tokenizer'],
                'value_key': 'global_utilization',
                'stat_key': None,
                'format': '{:.3f}',
            },
            {
                'key': 'type_token_ratio',
                'title': 'TTR',
                'key_path': ['type_token_ratio', 'per_tokenizer'],
                'value_key': 'global_ttr',
                'stat_key': None,
                'format': '{:.4f}',
            },
            {
                'key': 'renyi_1.0',
                'title': 'Shannon Entropy',
                'key_path': ['renyi_efficiency', 'per_tokenizer'],
                'value_key': 'renyi_1.0',
                'stat_key': 'overall',
                'format': '{:.2f}',
            },
            {
                'key': 'avg_token_rank',
                'title': 'Avg Token Rank',
                'key_path': ['unigram_distribution_metrics', 'per_tokenizer'],
                'value_key': 'global_avg_token_rank',
                'stat_key': None,
                'format': '{:.1f}',
            },
            {
                'key': 'tokenizer_fairness_gini',
                'title': 'Gini',
                'key_path': ['tokenizer_fairness_gini', 'per_tokenizer'],
                'value_key': 'gini_coefficient',
                'stat_key': None,
                'format': '{:.3f}',
            },
            {
                'key': 'num_languages',
                'title': 'Languages',
                'key_path': ['tokenizer_fairness_gini', 'per_tokenizer'],
                'value_key': 'num_languages',
                'stat_key': None,
                'format': '{:d}',
            },
        ]

    # ------------------------------------------------------------------
    # Value extraction / formatting
    # ------------------------------------------------------------------

    def _extract_metric_value(
        self, metric_config: Dict[str, Any], tokenizer_name: str
    ) -> Optional[Any]:
        """Navigate the results dict and return a single scalar value (or None)."""
        try:
            data = self.results
            for key in metric_config['key_path']:
                if key not in data:
                    return None
                data = data[key]

            if tokenizer_name not in data:
                return None

            tokenizer_data = data[tokenizer_name]

            if metric_config['stat_key']:
                value_data = tokenizer_data.get(metric_config['value_key'], {})
                if isinstance(value_data, dict):
                    return value_data.get(metric_config['stat_key'])
                return value_data
            else:
                return tokenizer_data.get(metric_config['value_key'])
        except Exception as e:
            logger.warning(
                f"Error extracting metric {metric_config['title']} "
                f"for {tokenizer_name}: {e}"
            )
            return None

    @staticmethod
    def _format_value(value: Any, format_str: str) -> str:
        """Format *value* with *format_str*, or return ``'---'`` when None."""
        if value is None:
            return '---'
        try:
            return format_str.format(value)
        except (ValueError, TypeError):
            return str(value)

    # ------------------------------------------------------------------
    # Table generation
    # ------------------------------------------------------------------

    def generate_markdown_table(
        self, metrics: Optional[List[str]] = None
    ) -> str:
        """Return a full Markdown document with one row per tokenizer.

        Parameters
        ----------
        metrics : list[str], optional
            Metric keys to include.  ``None`` means *all* configured metrics.
        """
        configs = self._resolve_metrics(metrics)

        headers = ['Tokenizer'] + [c['title'] for c in configs]
        separator = ['---'] * len(headers)

        rows: List[List[str]] = []
        for tok_name in self.tokenizer_names:
            row = [tok_name]
            for cfg in configs:
                value = self._extract_metric_value(cfg, tok_name)
                row.append(self._format_value(value, cfg['format']))
            rows.append(row)

        return self._render_markdown(headers, separator, rows)

    # ------------------------------------------------------------------
    # Parsing an existing RESULTS.md
    # ------------------------------------------------------------------

    @staticmethod
    def parse_existing_markdown(
        filepath: str,
    ) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
        """Parse *filepath* and return ``(headers, rows_dict)``.

        ``rows_dict`` maps ``tokenizer_name -> {column_title: cell_value}``.
        Returns empty structures when the file doesn't exist or has no table.
        """
        path = Path(filepath)
        if not path.exists():
            return [], {}

        text = path.read_text(encoding='utf-8')
        lines = text.splitlines()

        # Find the header row (first line starting with '|')
        header_idx: Optional[int] = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('|') and '---' not in stripped:
                header_idx = i
                break

        if header_idx is None:
            return [], {}

        headers = [
            h.strip() for h in lines[header_idx].strip().strip('|').split('|')
        ]

        # Skip separator line
        data_start = header_idx + 2

        rows_dict: Dict[str, Dict[str, str]] = {}
        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped.startswith('|'):
                break
            cells = [c.strip() for c in stripped.strip('|').split('|')]
            if not cells:
                continue
            tok_name = cells[0]
            row_map: Dict[str, str] = {}
            for j, hdr in enumerate(headers):
                if j < len(cells):
                    row_map[hdr] = cells[j]
            rows_dict[tok_name] = row_map

        return headers, rows_dict

    # ------------------------------------------------------------------
    # Cumulative update
    # ------------------------------------------------------------------

    def update_markdown_file(
        self,
        filepath: str,
        metrics: Optional[List[str]] = None,
    ) -> str:
        """Merge current results into an existing RESULTS.md (or create it).

        * Existing tokenizer rows not in the current run are preserved.
        * Existing tokenizer rows that *are* in the current run are updated.
        * New tokenizers are appended.
        * Column order follows the current metric config; extra columns from
          the old file that aren't in the current config are appended at the
          end.

        Returns the rendered Markdown string.
        """
        configs = self._resolve_metrics(metrics)
        current_titles = [c['title'] for c in configs]

        old_headers, old_rows = self.parse_existing_markdown(filepath)

        # ----- Determine final column list -----
        # "Tokenizer" is always first; then current metric titles; then any
        # extra columns from the old file that we don't know about.
        extra_headers: List[str] = []
        if old_headers:
            for h in old_headers:
                if h != 'Tokenizer' and h not in current_titles:
                    extra_headers.append(h)

        all_titles = current_titles + extra_headers
        headers = ['Tokenizer'] + all_titles
        separator = ['---'] * len(headers)

        # ----- Build rows dict (old preserved, current overwritten) -----
        merged: Dict[str, Dict[str, str]] = {}

        # Start with old rows
        for tok_name, row_map in old_rows.items():
            merged[tok_name] = dict(row_map)

        # Overwrite / add current-run rows
        for tok_name in self.tokenizer_names:
            if tok_name not in merged:
                merged[tok_name] = {'Tokenizer': tok_name}
            for cfg in configs:
                value = self._extract_metric_value(cfg, tok_name)
                merged[tok_name][cfg['title']] = self._format_value(
                    value, cfg['format']
                )

        # ----- Determine row ordering -----
        # Current-run tokenizers first (preserving order), then old-only ones.
        ordered_names: List[str] = list(self.tokenizer_names)
        for name in merged:
            if name not in ordered_names:
                ordered_names.append(name)

        rows: List[List[str]] = []
        for tok_name in ordered_names:
            row_map = merged.get(tok_name, {})
            row = [tok_name]
            for title in all_titles:
                row.append(row_map.get(title, '---'))
            rows.append(row)

        md = self._render_markdown(headers, separator, rows)

        # Write the file
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding='utf-8')
        logger.info(f"Markdown results table saved to {filepath}")

        return md

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_metrics(
        self, metrics: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Return the list of metric configs to use."""
        if metrics is None:
            return list(self.metric_configs)
        key_map = {c['key']: c for c in self.metric_configs}
        resolved = [key_map[m] for m in metrics if m in key_map]
        if not resolved:
            logger.warning("No valid metrics specified; using all defaults")
            return list(self.metric_configs)
        return resolved

    @staticmethod
    def _render_markdown(
        headers: List[str],
        separator: List[str],
        rows: List[List[str]],
    ) -> str:
        """Render a complete Markdown document with header, timestamp, and table."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines = [
            '# Tokenizer Evaluation Results',
            '',
            f'_Last updated: {timestamp}_',
            '',
            '| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(separator) + ' |',
        ]
        for row in rows:
            lines.append('| ' + ' | '.join(row) + ' |')
        # Trailing newline
        lines.append('')
        return '\n'.join(lines)


def _run_git(
    *args: str, check: bool = True, capture: bool = True, stdin: str = None
) -> subprocess.CompletedProcess:
    """Run a git command and return the CompletedProcess."""
    cmd = ['git'] + list(args)
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
        stdin=subprocess.PIPE if stdin is not None else None,
        input=stdin,
    )


def push_results_to_branch(
    filepath: str,
    remote: str = "origin",
    branch: str = "results",
    commit_message: str = None,
) -> bool:
    """Commit *filepath* as ``RESULTS.md`` on *branch* and push, without
    touching the working tree or the current branch.

    The function uses low-level git plumbing (``hash-object``, ``mktree``,
    ``commit-tree``, ``update-ref``) so it never checks out another branch
    and never modifies the index or working directory.

    Before committing, the remote version of RESULTS.md (if any) is fetched
    and merged with the local file via
    :meth:`MarkdownTableGenerator.update_markdown_file`-style parsing so
    that rows added by other team members are preserved.

    Parameters
    ----------
    filepath : str
        Path to the local RESULTS.md that was already written by the
        current analysis run.
    remote : str
        Git remote name (default ``"origin"``).
    branch : str
        Target branch name on the remote (default ``"results"``).
    commit_message : str | None
        Custom commit message.  ``None`` generates a timestamped default.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on failure.
    """
    if commit_message is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        commit_message = f"Update RESULTS.md ({timestamp})"

    ref = f"refs/heads/{branch}"
    remote_ref = f"{remote}/{branch}"

    try:
        # 1. Fetch the remote branch (ok to fail if it doesn't exist yet)
        _run_git('fetch', remote, branch, check=False)

        # 2. Try to read RESULTS.md from the remote branch
        parent_sha: Optional[str] = None
        show_result = _run_git(
            'show', f'{remote_ref}:RESULTS.md', check=False
        )

        if show_result.returncode == 0 and show_result.stdout.strip():
            # Remote file exists — write it to a temp file so we can parse it
            remote_content = show_result.stdout

            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.md', delete=False
            ) as tmp:
                tmp.write(remote_content)
                tmp_path = tmp.name

            try:
                remote_headers, remote_rows = (
                    MarkdownTableGenerator.parse_existing_markdown(tmp_path)
                )
            finally:
                os.unlink(tmp_path)

            # If the remote has rows that the local file doesn't, merge them
            if remote_rows:
                local_headers, local_rows = (
                    MarkdownTableGenerator.parse_existing_markdown(filepath)
                )

                merged = False
                for tok_name, row_map in remote_rows.items():
                    if tok_name not in local_rows:
                        local_rows[tok_name] = row_map
                        merged = True

                if merged:
                    # Rewrite local file with merged data
                    # Determine column order: local headers first, then extra remote-only
                    all_headers = list(local_headers) if local_headers else []
                    for h in remote_headers:
                        if h not in all_headers:
                            all_headers.append(h)

                    data_headers = [h for h in all_headers if h != 'Tokenizer']
                    headers = ['Tokenizer'] + data_headers
                    separator = ['---'] * len(headers)

                    # Row order: local first, then remote-only
                    ordered_names = [
                        n for n in local_rows if n in local_rows
                    ]
                    for n in remote_rows:
                        if n not in ordered_names:
                            ordered_names.append(n)

                    rows = []
                    all_rows = {**remote_rows, **local_rows}  # local overwrites remote
                    for tok_name in ordered_names:
                        row_map = all_rows.get(tok_name, {})
                        row = [tok_name] + [
                            row_map.get(h, '---') for h in data_headers
                        ]
                        rows.append(row)

                    md = MarkdownTableGenerator._render_markdown(
                        headers, separator, rows
                    )
                    Path(filepath).write_text(md, encoding='utf-8')
                    logger.info(
                        "Merged remote rows into local RESULTS.md before pushing"
                    )

            # Get parent commit SHA for the branch
            rev_result = _run_git(
                'rev-parse', remote_ref, check=False
            )
            if rev_result.returncode == 0:
                parent_sha = rev_result.stdout.strip()

        # 3. Create a blob from the (possibly merged) local file
        blob_result = _run_git('hash-object', '-w', filepath)
        blob_sha = blob_result.stdout.strip()

        # 4. Build a tree containing only RESULTS.md
        tree_entry = f"100644 blob {blob_sha}\tRESULTS.md"
        tree_result = _run_git('mktree', stdin=tree_entry)
        tree_sha = tree_result.stdout.strip()

        # 5. Create a commit (with parent if branch already exists)
        commit_args = ['commit-tree', tree_sha, '-m', commit_message]
        if parent_sha:
            commit_args += ['-p', parent_sha]
        commit_result = _run_git(*commit_args)
        commit_sha = commit_result.stdout.strip()

        # 6. Point the local branch ref at the new commit
        _run_git('update-ref', ref, commit_sha)

        # 7. Push to remote
        _run_git('push', remote, f'{branch}:{branch}')

        logger.info(
            f"Pushed RESULTS.md to {remote}/{branch} (commit {commit_sha[:8]})"
        )
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr.strip()}")
        return False
    except Exception as e:
        logger.error(f"Failed to push results to {remote}/{branch}: {e}")
        return False
