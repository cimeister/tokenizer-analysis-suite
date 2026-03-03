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

    # Default cap on snippets loaded per language from files/parquet.
    # Keeps memory usage bounded when large code corpora are provided.
    DEFAULT_MAX_SNIPPETS_PER_LANG: int = 100

    # Maximum character length for a single snippet.  Snippets longer than
    # this are truncated before storage.  Keeps pickle payloads and
    # tree-sitter parse times bounded.
    MAX_SNIPPET_SIZE_CHARS: int = 15_000

    def __init__(
        self,
        code_config: Optional[Dict[str, str]] = None,
        max_snippets_per_lang: Optional[int] = None,
    ):
        """
        Args:
            code_config: Dict mapping language names to file/directory paths.
                Example: {"python": "code_data/python/", "javascript": "code_data/js/"}
            max_snippets_per_lang: Maximum number of snippets to keep per
                language.  ``None`` uses :attr:`DEFAULT_MAX_SNIPPETS_PER_LANG`.
                Set to ``0`` to disable the cap entirely.
        """
        self.config = code_config or {}
        self.code_snippets: Dict[str, List[str]] = {}
        if max_snippets_per_lang is None:
            self.max_snippets_per_lang = self.DEFAULT_MAX_SNIPPETS_PER_LANG
        else:
            self.max_snippets_per_lang = max_snippets_per_lang

    def load_all(self) -> None:
        """Load all configured code datasets."""
        for lang, path in self.config.items():
            if not os.path.exists(path):
                logger.warning("Code data path not found for %s: %s", lang, path)
                continue
            try:
                self._load_language(lang, path)
            except Exception as e:
                logger.error("Error loading code for %s from %s: %s", lang, path, e)

    def _load_language(self, lang: str, path: str) -> None:
        """Load code snippets for a single language from *path*.

        Respects :attr:`max_snippets_per_lang`: when a cap is active
        (value > 0), only the first *N* snippets are kept.
        """
        existing = len(self.code_snippets.get(lang, []))
        cap = self.max_snippets_per_lang
        cap_active = cap > 0

        snippets: List[str] = []

        if os.path.isfile(path):
            if path.endswith(".parquet"):
                snippets.extend(self._read_parquet(path))
            else:
                text = self._read_file(path)
                if text:
                    snippets.append(text)
        elif os.path.isdir(path):
            extensions = self._LANG_EXTENSIONS.get(lang, [])
            for ext in extensions:
                for fpath in sorted(glob.glob(os.path.join(path, "**", f"*{ext}"), recursive=True)):
                    if cap_active and existing + len(snippets) >= cap:
                        break
                    text = self._read_file(fpath)
                    if text:
                        snippets.append(text)
                if cap_active and existing + len(snippets) >= cap:
                    break

        # Apply cap: only keep enough snippets to reach the limit
        if cap_active and existing + len(snippets) > cap:
            snippets = snippets[: cap - existing]

        # Truncate oversized snippets
        size_cap = self.MAX_SNIPPET_SIZE_CHARS
        snippets = [s[:size_cap] for s in snippets]

        if snippets:
            self.code_snippets.setdefault(lang, []).extend(snippets)
            total = len(self.code_snippets[lang])
            logger.info(
                "Loaded %d code snippet(s) for %s (total: %d%s)",
                len(snippets), lang, total,
                f", capped at {cap}" if cap_active else "",
            )

    @classmethod
    def _read_file(cls, path: str, max_chars: Optional[int] = None) -> Optional[str]:
        """Read a text file, returning ``None`` on failure or empty content.

        Whitespace is preserved (including leading indentation) so that
        downstream metrics such as indentation consistency see the original
        layout.  Only trailing whitespace is removed.
        """
        if max_chars is None:
            max_chars = cls.MAX_SNIPPET_SIZE_CHARS
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(max_chars)
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
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas is required to read parquet files; skipping %s", path)
            return []

        if max_chars is None:
            max_chars = cls.MAX_SNIPPET_SIZE_CHARS

        try:
            df = pd.read_parquet(path)
        except Exception as e:
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
            text = cls._strip_starcoder_metadata(raw)
            text = text[:max_chars].rstrip()
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

        Samples are loaded from ``sample_data/code_samples.json``.
        """
        path = CodeDataLoader._BUILTIN_CODE_SAMPLES_PATH
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
