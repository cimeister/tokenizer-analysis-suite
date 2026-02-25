"""
Code data loader for AST boundary alignment analysis.

Loads code snippets from disk or generates synthetic samples for testing.
"""

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
        """Read a text file, returning ``None`` on failure or empty content."""
        if max_chars is None:
            max_chars = cls.MAX_SNIPPET_SIZE_CHARS
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(max_chars)
            return text.strip() if text.strip() else None
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
            text = text[:max_chars].strip()
            if text:
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

    @staticmethod
    def generate_synthetic_samples() -> Dict[str, List[str]]:
        """Return representative code snippets for each target language.

        Each snippet exercises all five AST node categories: identifiers,
        keywords, operators, literals, and delimiters.
        """
        return {
            "python": [
                'def fibonacci(n):\n'
                '    if n <= 1:\n'
                '        return n\n'
                '    result = fibonacci(n - 1) + fibonacci(n - 2)\n'
                '    return result\n'
                '\n'
                'class Calculator:\n'
                '    PI = 3.14159\n'
                '    name = "calc"\n'
                '\n'
                '    def add(self, a, b):\n'
                '        return a + b\n'
                '\n'
                '    def is_positive(self, x):\n'
                '        return x > 0 and x != None\n'
                '\n'
                'values = [1, 2, 3]\n'
                'total = 0\n'
                'for v in values:\n'
                '    total += v\n'
                'flag = True\n',
            ],
            "javascript": [
                'function fibonacci(n) {\n'
                '    if (n <= 1) {\n'
                '        return n;\n'
                '    }\n'
                '    let result = fibonacci(n - 1) + fibonacci(n - 2);\n'
                '    const name = "fib";\n'
                '    return result;\n'
                '}\n'
                '\n'
                'class Calculator {\n'
                '    constructor(precision) {\n'
                '        this.precision = precision;\n'
                '    }\n'
                '    add(a, b) {\n'
                '        return a + b;\n'
                '    }\n'
                '}\n'
                '\n'
                'const values = [1, 2, 3];\n'
                'let total = 0;\n'
                'for (let i = 0; i < values.length; i++) {\n'
                '    total += values[i];\n'
                '}\n',
            ],
            "java": [
                'public class Calculator {\n'
                '    static final double PI = 3.14159;\n'
                '    private String name = "calc";\n'
                '\n'
                '    public int fibonacci(int n) {\n'
                '        if (n <= 1) {\n'
                '            return n;\n'
                '        }\n'
                '        int result = fibonacci(n - 1) + fibonacci(n - 2);\n'
                '        return result;\n'
                '    }\n'
                '\n'
                '    public boolean isPositive(int x) {\n'
                '        return x > 0 && x != 0;\n'
                '    }\n'
                '\n'
                '    public static void main(String[] args) {\n'
                '        Calculator calc = new Calculator();\n'
                '        int[] values = {1, 2, 3};\n'
                '        int total = 0;\n'
                '        for (int v : values) {\n'
                '            total += v;\n'
                '        }\n'
                '    }\n'
                '}\n',
            ],
            "c": [
                '#include <stdio.h>\n'
                '#include <stdbool.h>\n'
                '\n'
                'int fibonacci(int n) {\n'
                '    if (n <= 1) {\n'
                '        return n;\n'
                '    }\n'
                '    int result = fibonacci(n - 1) + fibonacci(n - 2);\n'
                '    return result;\n'
                '}\n'
                '\n'
                'bool is_positive(int x) {\n'
                '    return x > 0 && x != 0;\n'
                '}\n'
                '\n'
                'int main() {\n'
                '    int values[] = {1, 2, 3};\n'
                '    int total = 0;\n'
                '    const char* name = "calc";\n'
                '    for (int i = 0; i < 3; i++) {\n'
                '        total += values[i];\n'
                '    }\n'
                '    printf("total: %d\\n", total);\n'
                '    return 0;\n'
                '}\n',
            ],
            "cpp": [
                '#include <iostream>\n'
                '#include <string>\n'
                '#include <vector>\n'
                '\n'
                'class Calculator {\n'
                'public:\n'
                '    static constexpr double PI = 3.14159;\n'
                '    std::string name = "calc";\n'
                '\n'
                '    int fibonacci(int n) {\n'
                '        if (n <= 1) {\n'
                '            return n;\n'
                '        }\n'
                '        int result = fibonacci(n - 1) + fibonacci(n - 2);\n'
                '        return result;\n'
                '    }\n'
                '\n'
                '    bool isPositive(int x) {\n'
                '        return x > 0 && x != 0;\n'
                '    }\n'
                '};\n'
                '\n'
                'int main() {\n'
                '    Calculator calc;\n'
                '    std::vector<int> values = {1, 2, 3};\n'
                '    int total = 0;\n'
                '    for (auto v : values) {\n'
                '        total += v;\n'
                '    }\n'
                '    std::cout << total << std::endl;\n'
                '    return 0;\n'
                '}\n',
            ],
            "go": [
                'package main\n'
                '\n'
                'import "fmt"\n'
                '\n'
                'func fibonacci(n int) int {\n'
                '    if n <= 1 {\n'
                '        return n\n'
                '    }\n'
                '    result := fibonacci(n-1) + fibonacci(n-2)\n'
                '    return result\n'
                '}\n'
                '\n'
                'func isPositive(x int) bool {\n'
                '    return x > 0 && x != 0\n'
                '}\n'
                '\n'
                'func main() {\n'
                '    values := []int{1, 2, 3}\n'
                '    total := 0\n'
                '    name := "calc"\n'
                '    for _, v := range values {\n'
                '        total += v\n'
                '    }\n'
                '    fmt.Println(name, total)\n'
                '}\n',
            ],
            "rust": [
                'fn fibonacci(n: u64) -> u64 {\n'
                '    if n <= 1 {\n'
                '        return n;\n'
                '    }\n'
                '    let result = fibonacci(n - 1) + fibonacci(n - 2);\n'
                '    return result;\n'
                '}\n'
                '\n'
                'fn is_positive(x: i32) -> bool {\n'
                '    x > 0 && x != 0\n'
                '}\n'
                '\n'
                'fn main() {\n'
                '    let values = vec![1, 2, 3];\n'
                '    let mut total = 0;\n'
                '    let name = "calc";\n'
                '    for v in &values {\n'
                '        total += v;\n'
                '    }\n'
                '    println!("{} {}", name, total);\n'
                '}\n',
            ],
            "typescript": [
                'function fibonacci(n: number): number {\n'
                '    if (n <= 1) {\n'
                '        return n;\n'
                '    }\n'
                '    let result: number = fibonacci(n - 1) + fibonacci(n - 2);\n'
                '    const name: string = "fib";\n'
                '    return result;\n'
                '}\n'
                '\n'
                'class Calculator {\n'
                '    readonly PI: number = 3.14159;\n'
                '    add(a: number, b: number): number {\n'
                '        return a + b;\n'
                '    }\n'
                '}\n'
                '\n'
                'const values: number[] = [1, 2, 3];\n'
                'let total: number = 0;\n'
                'for (let i = 0; i < values.length; i++) {\n'
                '    total += values[i];\n'
                '}\n',
            ],
            "php": [
                '<?php\n'
                'function fibonacci(int $n): int {\n'
                '    if ($n <= 1) {\n'
                '        return $n;\n'
                '    }\n'
                '    $result = fibonacci($n - 1) + fibonacci($n - 2);\n'
                '    return $result;\n'
                '}\n'
                '\n'
                'class Calculator {\n'
                '    const PI = 3.14159;\n'
                '    public string $name = "calc";\n'
                '\n'
                '    public function add(int $a, int $b): int {\n'
                '        return $a + $b;\n'
                '    }\n'
                '}\n'
                '\n'
                '$values = [1, 2, 3];\n'
                '$total = 0;\n'
                'foreach ($values as $v) {\n'
                '    $total += $v;\n'
                '}\n'
                'echo $total;\n'
                '?>\n',
            ],
            "ruby": [
                'def fibonacci(n)\n'
                '  if n <= 1\n'
                '    return n\n'
                '  end\n'
                '  result = fibonacci(n - 1) + fibonacci(n - 2)\n'
                '  return result\n'
                'end\n'
                '\n'
                'class Calculator\n'
                '  PI = 3.14159\n'
                '  attr_accessor :name\n'
                '\n'
                '  def initialize\n'
                '    @name = "calc"\n'
                '  end\n'
                '\n'
                '  def add(a, b)\n'
                '    a + b\n'
                '  end\n'
                'end\n'
                '\n'
                'values = [1, 2, 3]\n'
                'total = 0\n'
                'values.each do |v|\n'
                '  total += v\n'
                'end\n'
                'flag = true\n',
            ],
            "csharp": [
                'using System;\n'
                '\n'
                'class Calculator {\n'
                '    const double PI = 3.14159;\n'
                '    string name = "calc";\n'
                '\n'
                '    int Fibonacci(int n) {\n'
                '        if (n <= 1) {\n'
                '            return n;\n'
                '        }\n'
                '        int result = Fibonacci(n - 1) + Fibonacci(n - 2);\n'
                '        return result;\n'
                '    }\n'
                '\n'
                '    bool IsPositive(int x) {\n'
                '        return x > 0 && x != 0;\n'
                '    }\n'
                '\n'
                '    static void Main() {\n'
                '        int[] values = {1, 2, 3};\n'
                '        int total = 0;\n'
                '        foreach (int v in values) {\n'
                '            total += v;\n'
                '        }\n'
                '        Console.WriteLine(total);\n'
                '    }\n'
                '}\n',
            ],
            "scala": [
                'object Calculator {\n'
                '  val PI: Double = 3.14159\n'
                '  var name: String = "calc"\n'
                '\n'
                '  def fibonacci(n: Int): Int = {\n'
                '    if (n <= 1) {\n'
                '      return n\n'
                '    }\n'
                '    val result = fibonacci(n - 1) + fibonacci(n - 2)\n'
                '    result\n'
                '  }\n'
                '\n'
                '  def isPositive(x: Int): Boolean = {\n'
                '    x > 0 && x != 0\n'
                '  }\n'
                '\n'
                '  def main(args: Array[String]): Unit = {\n'
                '    val values = List(1, 2, 3)\n'
                '    var total = 0\n'
                '    for (v <- values) {\n'
                '      total += v\n'
                '    }\n'
                '    println(total)\n'
                '  }\n'
                '}\n',
            ],
            "swift": [
                'func fibonacci(_ n: Int) -> Int {\n'
                '    if n <= 1 {\n'
                '        return n\n'
                '    }\n'
                '    let result = fibonacci(n - 1) + fibonacci(n - 2)\n'
                '    return result\n'
                '}\n'
                '\n'
                'class Calculator {\n'
                '    let pi = 3.14159\n'
                '    var name = "calc"\n'
                '\n'
                '    func add(_ a: Int, _ b: Int) -> Int {\n'
                '        return a + b\n'
                '    }\n'
                '\n'
                '    func isPositive(_ x: Int) -> Bool {\n'
                '        return x > 0 && x != 0\n'
                '    }\n'
                '}\n'
                '\n'
                'let values = [1, 2, 3]\n'
                'var total = 0\n'
                'for v in values {\n'
                '    total += v\n'
                '}\n',
            ],
            "kotlin": [
                'fun fibonacci(n: Int): Int {\n'
                '    if (n <= 1) {\n'
                '        return n\n'
                '    }\n'
                '    val result = fibonacci(n - 1) + fibonacci(n - 2)\n'
                '    return result\n'
                '}\n'
                '\n'
                'class Calculator {\n'
                '    val pi = 3.14159\n'
                '    var name = "calc"\n'
                '\n'
                '    fun add(a: Int, b: Int): Int {\n'
                '        return a + b\n'
                '    }\n'
                '\n'
                '    fun isPositive(x: Int): Boolean {\n'
                '        return x > 0 && x != 0\n'
                '    }\n'
                '}\n'
                '\n'
                'fun main() {\n'
                '    val values = listOf(1, 2, 3)\n'
                '    var total = 0\n'
                '    for (v in values) {\n'
                '        total += v\n'
                '    }\n'
                '    println(total)\n'
                '}\n',
            ],
            "lua": [
                'function fibonacci(n)\n'
                '    if n <= 1 then\n'
                '        return n\n'
                '    end\n'
                '    local result = fibonacci(n - 1) + fibonacci(n - 2)\n'
                '    return result\n'
                'end\n'
                '\n'
                'Calculator = {}\n'
                'Calculator.PI = 3.14159\n'
                'Calculator.name = "calc"\n'
                '\n'
                'function Calculator.add(a, b)\n'
                '    return a + b\n'
                'end\n'
                '\n'
                'local values = {1, 2, 3}\n'
                'local total = 0\n'
                'for _, v in ipairs(values) do\n'
                '    total = total + v\n'
                'end\n'
                'local flag = true\n',
            ],
            "r": [
                'fibonacci <- function(n) {\n'
                '  if (n <= 1) {\n'
                '    return(n)\n'
                '  }\n'
                '  result <- fibonacci(n - 1) + fibonacci(n - 2)\n'
                '  return(result)\n'
                '}\n'
                '\n'
                'is_positive <- function(x) {\n'
                '  return(x > 0 && x != 0)\n'
                '}\n'
                '\n'
                'values <- c(1, 2, 3)\n'
                'total <- 0\n'
                'name <- "calc"\n'
                'for (v in values) {\n'
                '  total <- total + v\n'
                '}\n'
                'flag <- TRUE\n'
                'cat(name, total, "\\n")\n',
            ],
            "perl": [
                'use strict;\n'
                'use warnings;\n'
                '\n'
                'sub fibonacci {\n'
                '    my ($n) = @_;\n'
                '    if ($n <= 1) {\n'
                '        return $n;\n'
                '    }\n'
                '    my $result = fibonacci($n - 1) + fibonacci($n - 2);\n'
                '    return $result;\n'
                '}\n'
                '\n'
                'my @values = (1, 2, 3);\n'
                'my $total = 0;\n'
                'my $name = "calc";\n'
                'foreach my $v (@values) {\n'
                '    $total += $v;\n'
                '}\n'
                'print "$name $total\\n";\n',
            ],
            "haskell": [
                'module Main where\n'
                '\n'
                'fibonacci :: Int -> Int\n'
                'fibonacci n\n'
                '  | n <= 1    = n\n'
                '  | otherwise = fibonacci (n - 1) + fibonacci (n - 2)\n'
                '\n'
                'isPositive :: Int -> Bool\n'
                'isPositive x = x > 0 && x /= 0\n'
                '\n'
                'name :: String\n'
                'name = "calc"\n'
                '\n'
                'main :: IO ()\n'
                'main = do\n'
                '  let values = [1, 2, 3]\n'
                '  let total = sum values\n'
                '  putStrLn (name ++ " " ++ show total)\n',
            ],
            "bash": [
                '#!/bin/bash\n'
                '\n'
                'fibonacci() {\n'
                '    local n=$1\n'
                '    if [ "$n" -le 1 ]; then\n'
                '        echo "$n"\n'
                '        return\n'
                '    fi\n'
                '    local a=$(fibonacci $((n - 1)))\n'
                '    local b=$(fibonacci $((n - 2)))\n'
                '    echo $((a + b))\n'
                '}\n'
                '\n'
                'values=(1 2 3)\n'
                'total=0\n'
                'name="calc"\n'
                'for v in "${values[@]}"; do\n'
                '    total=$((total + v))\n'
                'done\n'
                'echo "$name $total"\n',
            ],
        }
