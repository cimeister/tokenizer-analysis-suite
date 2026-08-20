"""
Code data loader for AST boundary alignment analysis.

Loads code snippets from disk or generates synthetic samples for testing.
"""

import json
import os
import glob
import logging
import re
from typing import Dict, List, Optional

from .multilingual_data import ParquetEngineMissing, _reraise_if_no_parquet_engine

logger = logging.getLogger(__name__)


class CodeDataLoader:
    """
    Loader for source code snippets used by AST boundary alignment metrics.

    Provides a unified interface for loading code files from disk or
    generating synthetic samples for quick testing.
    """

    # Language name -> file extensions
    _LANG_EXTENSIONS: Dict[str, List[str]] = {
        "python": [".py"],
        "javascript": [".js", ".mjs"],
        "java": [".java"],
        "c": [".c", ".h"],
        "cpp": [".cpp", ".cc", ".cxx", ".hpp"],
        "go": [".go"],
        "rust": [".rs"],
        "typescript": [".ts"],
        "php": [".php"],
        "ruby": [".rb"],
        "csharp": [".cs"],
        "scala": [".scala"],
        "swift": [".swift"],
        "kotlin": [".kt", ".kts"],
        "lua": [".lua"],
        "r": [".r", ".R"],
        "perl": [".pl", ".pm"],
        "haskell": [".hs"],
        "bash": [".sh", ".bash"],
    }

    # Language name -> tree-sitter grammar identifier
    _LANG_TO_TREESITTER: Dict[str, str] = {
        "python": "python",
        "javascript": "javascript",
        "java": "java",
        "c": "c",
        "cpp": "cpp",
        "go": "go",
        "rust": "rust",
        "typescript": "typescript",
        "php": "php",
        "ruby": "ruby",
        "csharp": "csharp",
        "scala": "scala",
        "swift": "swift",
        "kotlin": "kotlin",
        "lua": "lua",
        "r": "r",
        "perl": "perl",
        "haskell": "haskell",
        "bash": "bash",
    }

    # Default cap on snippets (files) loaded per language from files/parquet.
    # 0 means no cap: every file a code_config path matches is loaded. Off by
    # default since 1.0.0; a caller that wants a bounded corpus re-enables
    # this with --max-code-files-per-lang on the CLI (cli/run_analysis.py) or
    # by passing max_snippets_per_lang explicitly. Before 1.0.0 this defaulted
    # to 100, which silently dropped files beyond the first 100 per language
    # (by sort order) with nothing in the CLI or the results file recording
    # that a cap had been applied.
    DEFAULT_MAX_SNIPPETS_PER_LANG: int = 0

    # Default maximum character length kept per snippet; longer files are
    # truncated to this length. 0 means no cap: files are kept in full. Off
    # by default since 1.0.0; re-enable with --max-code-file-chars on the CLI
    # or by passing max_snippet_chars explicitly. Before 1.0.0 this defaulted
    # to 15_000, which silently discarded the tail of any longer file with
    # nothing in the CLI or the results file recording that truncation had
    # happened.
    MAX_SNIPPET_SIZE_CHARS: int = 0

    def __init__(
        self,
        code_config: Optional[Dict[str, str]] = None,
        max_snippets_per_lang: Optional[int] = None,
        max_snippet_chars: Optional[int] = None,
    ):
        """
        Args:
            code_config: Dict mapping language names to file/directory paths.
                Example: {"python": "code_data/python/", "javascript": "code_data/js/"}
            max_snippets_per_lang: Maximum number of files to keep per
                language. 0 disables the cap (keep every file found).
                ``None`` uses :attr:`DEFAULT_MAX_SNIPPETS_PER_LANG`, which is
                0 (no cap) unless the caller changed it.
            max_snippet_chars: Maximum character length kept for a single
                file; longer files are truncated to this length. 0 disables
                truncation (keep each file in full). ``None`` uses
                :attr:`MAX_SNIPPET_SIZE_CHARS`, which is 0 (no cap) unless
                the caller changed it.
        """
        self.config = self._validate_config(code_config)
        self.code_snippets: Dict[str, List[str]] = {}
        if max_snippets_per_lang is None:
            self.max_snippets_per_lang = self.DEFAULT_MAX_SNIPPETS_PER_LANG
        else:
            self.max_snippets_per_lang = max_snippets_per_lang
        if max_snippet_chars is None:
            self.max_snippet_chars = self.MAX_SNIPPET_SIZE_CHARS
        else:
            self.max_snippet_chars = max_snippet_chars

        # Per-language counts of files dropped by max_snippets_per_lang,
        # characters discarded by max_snippet_chars, and snippets dropped
        # because truncating them to max_snippet_chars left only whitespace,
        # filled in by _load_language. Lets a caller (e.g. run metadata)
        # report what a cap actually did without re-scanning the corpus; all
        # three are 0 for every language when no cap is active, which is the
        # default.
        self.dropped_file_counts: Dict[str, int] = {}
        self.truncated_char_counts: Dict[str, int] = {}
        self.dropped_whitespace_only_counts: Dict[str, int] = {}

    @staticmethod
    def _validate_config(code_config: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Check the shape of *code_config* here, where the name is known.

        A JSON array reaching this constructor used to surface two files later
        as ``AttributeError: 'list' object has no attribute 'items'``, swallowed
        in one caller and uncaught in the other, with neither mentioning the
        config or the flag that supplied it.
        """
        if code_config is None:
            return {}
        if not isinstance(code_config, dict):
            raise TypeError(
                "code_config must be an object mapping a language name to a "
                f"file or directory path, got {type(code_config).__name__}. "
                "From the CLI this is the --code-ast-config file, which must "
                'look like {"python": "code_data/python/"}.'
            )
        bad = {k: v for k, v in code_config.items() if not isinstance(v, str)}
        if bad:
            detail = ", ".join(
                f"{k}: {type(v).__name__}" for k, v in sorted(bad.items())
            )
            raise TypeError(
                f"Every code_config value must be a path string. Got {detail}."
            )
        unknown = sorted(set(code_config) - set(CodeDataLoader._LANG_EXTENSIONS))
        if unknown:
            raise ValueError(
                f"Unknown code language(s) in code_config: {', '.join(unknown)}. "
                "Supported: "
                f"{', '.join(sorted(CodeDataLoader._LANG_EXTENSIONS))}."
            )
        return dict(code_config)

    def load_all(self) -> None:
        """Load all configured code datasets.

        A configured path that does not exist aborts. Skipping it produced a
        results file whose code metrics were computed over fewer languages than
        the config named, with nothing in the output recording the difference.
        """
        missing = {
            lang: path for lang, path in self.config.items()
            if not os.path.exists(path)
        }
        if missing:
            detail = "; ".join(f"{lang}: {path}" for lang, path in sorted(missing.items()))
            raise FileNotFoundError(
                f"Code data path not found for {len(missing)} of "
                f"{len(self.config)} configured language(s): {detail}. Fix the "
                "path, or remove the language from the code config."
            )
        for lang, path in self.config.items():
            try:
                self._load_language(lang, path)
            except ParquetEngineMissing:
                raise
            except Exception as e:
                logger.error("Error loading code for %s from %s: %s", lang, path, e)

    def _load_language(self, lang: str, path: str) -> None:
        """Load code snippets for a single language from *path*.

        Respects :attr:`max_snippets_per_lang` (0 disables the cap: every
        matching file is kept) and :attr:`max_snippet_chars` (0 disables
        truncation: each file is kept in full). When a cap discards anything, a
        warning names the language and the amount, and the totals are added to
        :attr:`dropped_file_counts` / :attr:`truncated_char_counts` /
        :attr:`dropped_whitespace_only_counts`. A run with no active cap logs
        nothing extra.

        The caps are not accounted for the same way. Every file that is
        read is read in full, and :attr:`max_snippet_chars` is applied to the
        text afterward, so ``truncated_char_counts`` is the exact number of
        characters the ``s[:char_cap]`` slice itself discarded, whether or not
        the resulting snippet is later dropped for being whitespace-only (see
        below). :attr:`max_snippets_per_lang` is not applied only afterward:
        the directory branch below ``continue``s mid-walk once the cap is
        reached, counting the remaining files as found without reading them.
        ``dropped_file_counts`` therefore counts candidate paths that were
        never opened, not snippets that were built and thrown away. A path
        that would have read as empty, or whose every line fails strict UTF-8
        decoding, contributes nothing to ``code_snippets`` when it is read,
        but is still counted as dropped when the cap skips it. The two counts
        agree only when no cap is set, and the parquet and single-file
        branches, which have no early stop and are trimmed after the fact,
        are exact either way.

        Truncating a snippet to ``char_cap`` characters can leave only
        whitespace, when the kept prefix falls entirely inside deep
        indentation or a run of blank lines. ``_read_file`` and
        ``_read_parquet`` both already refuse to hand back a whitespace-only
        result; this rechecks after truncation for the same reason and drops
        the snippet rather than letting a blank string reach
        ``code_snippets``. Only the emptiness check is shared with those two
        paths. The truncated text itself is left exactly as ``s[:char_cap]``
        produced it: rstripping it here would change 4 of the 23 snippets the
        benchmark corpus truncates at 400 characters, which is a change to
        what is measured rather than to what is dropped. Each drop is counted in
        ``dropped_whitespace_only_counts`` (snippets, not characters: the same
        unit as ``dropped_file_counts``) and logged at warning level, so the
        loss is on the record instead of silent. It does not add anything to
        ``truncated_char_counts``: that counter stays exactly what the
        ``s[:char_cap]`` slice removed, an I/O-sized-cap number, separate from
        this data-loss count of snippets that ended up contributing nothing.
        """
        existing = len(self.code_snippets.get(lang, []))
        cap = self.max_snippets_per_lang
        cap_active = cap > 0
        char_cap = self.max_snippet_chars
        char_cap_active = char_cap > 0

        snippets: List[str] = []
        candidates_found = 0
        candidates_attempted = 0

        if os.path.isfile(path):
            if path.endswith(".parquet"):
                snippets.extend(self._read_parquet(path))
                candidates_found = candidates_attempted = len(snippets)
            else:
                candidates_found = candidates_attempted = 1
                text = self._read_file(path)
                if text:
                    snippets.append(text)
        elif os.path.isdir(path):
            extensions = self._LANG_EXTENSIONS.get(lang, [])
            for ext in extensions:
                for fpath in sorted(glob.glob(os.path.join(path, "**", f"*{ext}"), recursive=True)):
                    candidates_found += 1
                    if cap_active and existing + len(snippets) >= cap:
                        # Already have enough to reach the cap: count the
                        # file as found but do not read it.
                        continue
                    candidates_attempted += 1
                    text = self._read_file(fpath)
                    if text:
                        snippets.append(text)

        # Apply the file-count cap: only keep enough to reach the limit.
        # candidates_found > candidates_attempted already accounts for files
        # skipped mid-walk in the directory case above; this trim covers the
        # parquet/single-file case, where every row is read in one pass with
        # no early stop, so any excess only shows up here.
        dropped_files = candidates_found - candidates_attempted
        if cap_active and existing + len(snippets) > cap:
            keep = max(cap - existing, 0)
            dropped_files += len(snippets) - keep
            snippets = snippets[:keep]

        # Truncate oversized snippets and total the characters that costs.
        # A truncated snippet that comes out whitespace-only (e.g. char_cap
        # characters of leading indentation or blank lines) is dropped
        # rather than kept as a blank string, matching the whitespace-only
        # check _read_file and _read_parquet already apply on their paths.
        # The kept text is not otherwise altered: see the docstring on why
        # this does not rstrip the truncated prefix.
        truncated_chars = 0
        truncated_snippets = 0
        dropped_whitespace_only = 0
        if char_cap_active:
            capped_snippets: List[str] = []
            for s in snippets:
                if len(s) > char_cap:
                    truncated_chars += len(s) - char_cap
                    truncated_snippets += 1
                    capped = s[:char_cap]
                    # Only a truncated snippet can come out whitespace-only:
                    # _read_file and _read_parquet both refuse blank content, so
                    # a snippet that was not cut is non-blank already. Testing
                    # every snippet would have counted such a drop under a
                    # message naming a truncation that did not happen.
                    if not capped.strip():
                        dropped_whitespace_only += 1
                        continue
                else:
                    capped = s
                capped_snippets.append(capped)
            snippets = capped_snippets

        if dropped_files > 0:
            self.dropped_file_counts[lang] = (
                self.dropped_file_counts.get(lang, 0) + dropped_files
            )
            logger.warning(
                "%s: max_snippets_per_lang=%d dropped %d file(s) found under "
                "%s (kept %d of %d found).",
                lang, cap, dropped_files, path,
                existing + len(snippets), candidates_found,
            )
        if truncated_chars > 0:
            self.truncated_char_counts[lang] = (
                self.truncated_char_counts.get(lang, 0) + truncated_chars
            )
            logger.warning(
                "%s: max_snippet_chars=%d truncated content under %s, "
                "discarding %d character(s) total across %d snippet(s).",
                lang, char_cap, path, truncated_chars, truncated_snippets,
            )
        if dropped_whitespace_only > 0:
            self.dropped_whitespace_only_counts[lang] = (
                self.dropped_whitespace_only_counts.get(lang, 0)
                + dropped_whitespace_only
            )
            logger.warning(
                "%s: max_snippet_chars=%d truncation under %s left %d "
                "snippet(s) whitespace-only; dropped from the corpus "
                "instead of kept blank.",
                lang, char_cap, path, dropped_whitespace_only,
            )

        if snippets:
            self.code_snippets.setdefault(lang, []).extend(snippets)
            total = len(self.code_snippets[lang])
            logger.info(
                "Loaded %d code snippet(s) for %s (total: %d%s)",
                len(snippets), lang, total,
                f", capped at {cap}" if cap_active else "",
            )

    # U+FEFF as a leading character is a byte-order mark, not content.
    _BOM = "﻿"

    @classmethod
    def _normalize_source(cls, text: str) -> str:
        """Strip a leading BOM and normalize line endings to LF.

        Files are read in binary and decoded per line, so Python's
        universal-newline translation does not apply and plain ``utf-8`` leaves
        a BOM in place. Both then reach the metrics as if they were part of the
        code's layout.

        The BOM used to be the more damaging of the two, for a reason that no
        longer applies. A byte-level tokenizer re-encodes U+FEFF as three
        visible characters, so the reconstructed text started with characters
        the source does not contain; the greedy scan in
        ``BaseMetrics._build_source_to_recon_map`` stalled at index 0 and mapped
        the whole snippet to None, and every AST span in it was then charged to
        the tokenizer as misaligned. Two of the three bundled C# samples carry a
        BOM, which is why C# scored 0.13 to 0.19 against a 0.51 to 0.70 median
        for every tokenizer alike, and why identifier_fragmentation reported a
        negative tokens-per-identifier for C#. Both of those are gone
        independently of this strip: no metric builds a reconstruction any more,
        they all read the encoder's character offsets, and
        ``_identifier_stats`` keeps an unmapped identifier out of the rates
        rather than scoring it as -1 tokens.

        The offsets reason that remains is conditional. The offsets map U+FEFF
        to whichever token covers it, and when that is a token of its own, every
        AST span scores exactly as it would without the BOM. Measured on a C#
        snippet, full_alignment_rate is 0.9130 with and without the BOM under
        tokenizers/bpe.json and 1.0000 under tokenizers/unigramlm.json, with 0
        of 38 source characters unmapped in each of the four runs. When a
        vocabulary has a merge joining U+FEFF to the character after it, one
        token covers both the BOM and the first code character, and the first
        AST span's start boundary then scores as misaligned. Neither bundled
        tokenizer has such a merge.

        The strip also matters to the indentation metrics, which is not an
        offsets question: ``str.lstrip`` does not treat U+FEFF as whitespace,
        so ``_extract_line_indentation`` reads a BOM followed by four spaces
        and ``x = 1`` as indent width 0, against 4 for the same line without
        the BOM.

        CRLF matters for the indentation metrics, which measure leading
        whitespace per line; a trailing ``\\r`` is not indentation.
        """
        if text.startswith(cls._BOM):
            text = text[len(cls._BOM):]
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @classmethod
    def _read_file(cls, path: str, max_chars: Optional[int] = None) -> Optional[str]:
        """Read a text file, returning ``None`` on failure or empty content.

        Whitespace is preserved (including leading indentation) so that
        downstream metrics such as indentation consistency see the original
        layout.  Only trailing whitespace is removed.

        Lines that fail strict UTF-8 decoding are skipped individually
        rather than silently replacing invalid bytes.

        *max_chars* stops reading once that many characters have been read,
        bounding I/O on a large file instead of reading it fully and
        discarding most of it. ``None`` resolves to
        :attr:`MAX_SNIPPET_SIZE_CHARS`; either that or an explicit ``0``
        means no cap (the file is read to the end). ``_load_language`` does
        not use this early stop: it reads every file in full and applies
        ``max_snippet_chars`` itself afterward, so it can report exactly how
        many characters a cap discarded.
        """
        if max_chars is None:
            max_chars = cls.MAX_SNIPPET_SIZE_CHARS
        cap_active = max_chars > 0
        try:
            lines: List[str] = []
            chars_read = 0
            with open(path, "rb") as f:
                for raw_line in f:
                    if cap_active and chars_read >= max_chars:
                        break
                    try:
                        line = raw_line.decode("utf-8")
                    except UnicodeDecodeError:
                        logger.debug("Skipping non-UTF-8 line in %s", path)
                        continue
                    lines.append(line)
                    chars_read += len(line)
            text = "".join(lines)
            text = cls._normalize_source(text)
            if not text or not text.strip():
                return None
            return text.rstrip()
        except Exception as e:
            logger.debug("Could not read %s: %s", path, e)
            return None

    # Regex matching StarCoder-style metadata tags at the start of content.
    # Handles <reponame>..., <filename>..., <gh_stars>... in any order,
    # each terminated by the next tag or a newline.
    _STARCODER_META_RE = re.compile(
        r"^(?:<(?:reponame|filename|gh_stars)>[^\n]*\n?)+"
    )

    @classmethod
    def _strip_starcoder_metadata(cls, text: str) -> str:
        """Remove StarCoder metadata prefix tags from snippet content."""
        return cls._STARCODER_META_RE.sub("", text)

    @classmethod
    def _read_parquet(
        cls, path: str, content_column: str = "content", max_chars: Optional[int] = None
    ) -> List[str]:
        """Read code snippets from a parquet file.

        Expects a ``content`` column containing source code strings.
        StarCoder-style metadata prefixes (``<reponame>``, ``<filename>``,
        ``<gh_stars>``) are stripped automatically.

        *max_chars* truncates each row's content to that many characters.
        ``None`` resolves to :attr:`MAX_SNIPPET_SIZE_CHARS`; either that or
        an explicit ``0`` means no truncation. ``_load_language`` does not
        use this: it reads every row in full and applies
        ``max_snippet_chars`` itself afterward, so it can report exactly how
        many characters a cap discarded.
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas is required to read parquet files; skipping %s", path)
            return []

        if max_chars is None:
            max_chars = cls.MAX_SNIPPET_SIZE_CHARS
        cap_active = max_chars > 0

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            _reraise_if_no_parquet_engine(e, path)
            logger.error("Failed to read parquet file %s: %s", path, e)
            return []

        if content_column not in df.columns:
            logger.warning(
                "Parquet file %s has no '%s' column (columns: %s)",
                path, content_column, list(df.columns),
            )
            return []

        snippets: List[str] = []
        for raw in df[content_column]:
            if not isinstance(raw, str) or not raw.strip():
                continue
            text = cls._normalize_source(cls._strip_starcoder_metadata(raw))
            if cap_active:
                text = text[:max_chars]
            text = text.rstrip()
            if not text.strip():
                continue
            snippets.append(text)
        return snippets

    def get_code_snippets(self, lang: str) -> List[str]:
        """Return loaded code snippets for *lang*.

        Enforces :attr:`max_snippets_per_lang` as a final safety net so
        callers never receive more snippets than the configured cap.
        """
        snippets = self.code_snippets.get(lang, [])
        cap = self.max_snippets_per_lang
        if cap > 0 and len(snippets) > cap:
            return snippets[:cap]
        return snippets

    def get_languages(self) -> List[str]:
        """Return list of languages with loaded code data."""
        return sorted(self.code_snippets.keys())

    # ------------------------------------------------------------------
    # Synthetic sample generation
    # ------------------------------------------------------------------

    _BUILTIN_CODE_SAMPLES_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "sample_data",
        "code_samples.json",
    )

    @staticmethod
    def generate_synthetic_samples() -> Dict[str, List[str]]:
        """Return representative code snippets for each target language.

        Each snippet exercises all five AST node categories: identifiers,
        keywords, operators, literals, and delimiters.

        Samples are loaded from ``sample_data/code_samples.json``. They are real
        scraped files, so they carry real artifacts: two of the three C# samples
        begin with a byte-order mark and several use CRLF. Both are normalized
        here for the same reason they are normalized on the file and parquet
        paths, see :meth:`_normalize_source`.
        """
        path = CodeDataLoader._BUILTIN_CODE_SAMPLES_PATH
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        return {
            lang: [CodeDataLoader._normalize_source(s) for s in snippets]
            for lang, snippets in samples.items()
        }
