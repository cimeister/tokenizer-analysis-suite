#!/usr/bin/env python3
"""Visualize how tokenizers handle Python code, with emphasis on whitespace.

Usage:
    python scripts/visualize_tokenization.py \\
        --tokenizer-config configs/baseline_tokenizers.json

    # Only show a subset of tokenizers
    python scripts/visualize_tokenization.py \\
        --tokenizer-config configs/baseline_tokenizers.json \\
        --tokenizers "GPT-4o" "Qwen 3" "Classical"

    # Use your own code file
    python scripts/visualize_tokenization.py \\
        --tokenizer-config configs/baseline_tokenizers.json \\
        --code-file my_script.py

    # Plain text (no colour escapes) for file output
    python scripts/visualize_tokenization.py \\
        --tokenizer-config configs/baseline_tokenizers.json --no-color > out.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer_analysis.core.tokenizer_wrapper import create_tokenizer_wrapper

# ── Sample Python snippet used when no --code-file is given ──────────────
DEFAULT_CODE = textwrap.dedent("""\
    import os
    from pathlib import Path

    def count_lines(path: str, include_empty: bool = True) -> int:
        \"\"\"Count lines in a file.\"\"\"
        total = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if include_empty or line.strip():
                    total += 1
        return total

    class FileProcessor:
        def __init__(self, root_dir: str):
            self.root = Path(root_dir)
            self._cache: dict[str, int] = {}

        def process(self):
            for f in self.root.rglob("*.py"):
                n = count_lines(str(f))
                self._cache[str(f)] = n
                if n >= 100:
                    print(f"  {f.name}: {n} lines")

    if __name__ == "__main__":
        proc = FileProcessor(os.getcwd())
        proc.process()
""")

# ── ANSI colour helpers ──────────────────────────────────────────────────
_BG_COLORS = [
    "\033[48;5;153m",  # light blue
    "\033[48;5;180m",  # tan
    "\033[48;5;151m",  # pale green
    "\033[48;5;182m",  # light purple
    "\033[48;5;216m",  # peach
    "\033[48;5;152m",  # pale cyan
]
_FG_BLACK = "\033[38;5;232m"
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"

_UNASSIGNED = -1  # sentinel for char_color positions not owned by any token


# ── Offset extraction ────────────────────────────────────────────────────

def _get_offsets(wrapper, text: str) -> Optional[List[Tuple[int, int]]]:
    """Try to get (start, end) character offsets for each token.

    Works with both ``tokenizers.Tokenizer`` and
    ``transformers.PreTrainedTokenizerFast``.  Returns *None* if offsets
    cannot be obtained.
    """
    raw = wrapper.get_underlying_tokenizer()
    if raw is None:
        return None

    # tokenizers.Tokenizer  →  Encoding.offsets
    if hasattr(raw, 'encode'):
        try:
            enc = raw.encode(text)
            if hasattr(enc, 'offsets'):
                return enc.offsets
        except Exception:
            pass

    # transformers.PreTrainedTokenizerFast / Slow
    if callable(getattr(raw, '__call__', None)):
        try:
            enc = raw(text, return_offsets_mapping=True)
            if 'offset_mapping' in enc:
                return [tuple(pair) for pair in enc['offset_mapping']]
        except Exception:
            pass

    return None


# ── Visualisation ─────────────────────────────────────────────────────────

def _ws_visible(ch: str) -> str:
    """Replace a single whitespace char with a visible glyph."""
    if ch == " ":
        return "\u00b7"   # middle dot
    if ch == "\t":
        return "\u2192"   # right arrow
    if ch == "\n":
        return "\u21b5"   # down-left arrow (pilcrow-ish)
    if ch == "\r":
        return "\u240d"
    return ch


def _fill_offsets(
    offsets: List[Tuple[int, int]],
    text_len: int,
) -> List[Tuple[int, int]]:
    """Fill gaps in offsets and clamp overlaps.

    Pre-tokenizers (e.g. GPT-2 ByteLevel) may consume whitespace that
    doesn't appear in any token's offset.  Gaps are assigned to the next
    real token.  Zero-length offsets (``s == e``) are kept as-is — they
    mark special tokens (BOS/EOS, byte-fallback, etc.).
    """
    filled: list[tuple[int, int]] = []
    prev_end = 0
    for s, e in offsets:
        if s == e:
            # Zero-length span: special/synthetic token — keep unchanged
            filled.append((s, e))
            continue
        # Clamp start to avoid overlap; extend back to fill any gap
        real_start = max(prev_end, 0) if prev_end > s else (prev_end if prev_end < s else s)
        filled.append((real_start, e))
        prev_end = e
    return filled


def _build_char_owner(
    offsets: List[Tuple[int, int]],
    text_len: int,
) -> list[int]:
    """Map each character position to the token index that owns it.

    Positions not covered by any token get ``_UNASSIGNED``.
    """
    owner: list[int] = [_UNASSIGNED] * text_len
    for idx, (s, e) in enumerate(offsets):
        for c in range(s, min(e, text_len)):
            owner[c] = idx
    return owner


def visualize_tokens(
    name: str,
    code: str,
    wrapper,
    use_color: bool,
) -> str:
    """Return a formatted string showing the tokenized code for one tokenizer."""
    ids = wrapper.encode(code)
    tokens = wrapper.convert_ids_to_tokens(ids)
    offsets = _get_offsets(wrapper, code)

    # If offsets are available, use them for a character-span view.
    if offsets and len(offsets) == len(ids):
        offsets = _fill_offsets(offsets, len(code))
        spans = [code[s:e] for s, e in offsets]
    else:
        offsets = None
        spans = None

    n_tokens = len(ids)
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────
    sep = "=" * 72
    if use_color:
        lines.append(f"\n{_BOLD}{name}{_RESET}  ({n_tokens} tokens)")
    else:
        lines.append(f"\n{name}  ({n_tokens} tokens)")
    lines.append(sep)

    # ── Token-coloured source view (line by line) ─────────────────────
    if spans is not None:
        char_color = _build_char_owner(offsets, len(code))

        source_lines = code.split("\n")
        pos = 0  # character position in `code`
        for line_no, src_line in enumerate(source_lines):
            line_end = pos + len(src_line)
            buf: list[str] = []
            if use_color:
                buf.append(f"{_DIM}{line_no + 1:3d}{_RESET} ")
            else:
                buf.append(f"{line_no + 1:3d} ")

            prev_tok_idx = _UNASSIGNED - 1  # force first-char colour switch
            for ci in range(pos, line_end):
                tok_idx = char_color[ci]
                ch = code[ci]
                vis = _ws_visible(ch)

                if tok_idx != prev_tok_idx:
                    if use_color:
                        bg_idx = tok_idx % len(_BG_COLORS)
                        buf.append(f"{_BG_COLORS[bg_idx]}{_FG_BLACK}")
                    elif prev_tok_idx != _UNASSIGNED - 1:
                        buf.append("|")
                    prev_tok_idx = tok_idx

                buf.append(vis)

            if use_color:
                buf.append(_RESET)

            lines.append("".join(buf))
            pos = line_end + 1  # skip the \n

    else:
        # Fallback: show raw token strings (Ġ/Ċ encoding) delimited by |
        if use_color:
            buf: list[str] = []
            for i, tok_str in enumerate(tokens):
                bg = _BG_COLORS[i % len(_BG_COLORS)]
                buf.append(f"{bg}{_FG_BLACK}{tok_str}{_RESET}")
            lines.append("".join(buf))
        else:
            lines.append("|".join(tokens))

    # ── Whitespace statistics ─────────────────────────────────────────
    if spans is not None:
        char_color = _build_char_owner(offsets, len(code))

        ws_only = 0          # tokens whose span is pure non-newline whitespace
        newline_toks = 0      # tokens whose span contains at least one \n
        special_toks = 0      # zero-length spans (BOS/EOS etc.)
        newline_indent_toks = 0  # tokens containing \n followed by spaces

        for sp in spans:
            if not sp:
                special_toks += 1
            elif "\n" in sp:
                newline_toks += 1
                # Check for merged newline+indent pattern
                after_last_nl = sp[sp.rfind("\n") + 1:]
                if after_last_nl and after_last_nl.isspace():
                    newline_indent_toks += 1
            elif sp.isspace():
                ws_only += 1

        # ── Indentation analysis (character-based) ────────────────────
        # For each indented line, find how many *distinct* tokens own
        # the leading-whitespace character positions.  This correctly
        # handles tokens that straddle \n boundaries (e.g. "\n    ")
        # because char_color assigns ownership per character.
        indent_patterns: dict[str, int] = {}
        indent_level_tokens: dict[int, list[int]] = {}
        total_indent_toks = 0

        source_lines = code.split("\n")
        pos = 0
        for src_line in source_lines:
            line_end = pos + len(src_line)
            leading_ws = len(src_line) - len(src_line.lstrip())
            if leading_ws > 0:
                # Collect the distinct tokens owning each indent char
                tok_ids_in_indent: list[int] = []
                for ci in range(pos, pos + leading_ws):
                    owner = char_color[ci]
                    if owner == _UNASSIGNED:
                        continue
                    if not tok_ids_in_indent or tok_ids_in_indent[-1] != owner:
                        tok_ids_in_indent.append(owner)

                n_indent_toks = len(tok_ids_in_indent)
                total_indent_toks += n_indent_toks
                indent_level_tokens.setdefault(leading_ws, []).append(n_indent_toks)

                # Build pattern: tuple of per-token space counts
                pattern_parts: list[int] = []
                for tid in tok_ids_in_indent:
                    count = sum(
                        1 for ci in range(pos, pos + leading_ws)
                        if char_color[ci] == tid
                    )
                    pattern_parts.append(count)
                pat_key = repr(tuple(pattern_parts)) if pattern_parts else "()"
                indent_patterns[pat_key] = indent_patterns.get(pat_key, 0) + 1

            pos = line_end + 1

        lines.append("")
        ws_summary = (
            f"  Whitespace tokens: {ws_only}/{n_tokens}"
            f"  |  Newline tokens: {newline_toks}"
            f"  |  Indentation tokens: {total_indent_toks}"
        )
        if newline_indent_toks:
            ws_summary += f" ({newline_indent_toks} merged with newline)"
        if special_toks:
            ws_summary += f"  |  Special: {len([sp for sp in spans if not sp])}"

        if use_color:
            lines.append(f"{_DIM}{ws_summary}{_RESET}")
        else:
            lines.append(ws_summary)

        if indent_patterns:
            sorted_pats = sorted(indent_patterns.items(), key=lambda x: -x[1])
            pat_strs = [f"{pat} x{cnt}" for pat, cnt in sorted_pats]
            indent_detail = f"  Indent patterns (spaces per token): {', '.join(pat_strs)}"
            if use_color:
                lines.append(f"{_DIM}{indent_detail}{_RESET}")
            else:
                lines.append(indent_detail)

        if indent_level_tokens:
            depth_summary_parts = []
            for depth in sorted(indent_level_tokens):
                counts = indent_level_tokens[depth]
                avg = sum(counts) / len(counts)
                depth_summary_parts.append(f"{depth}sp={avg:.1f}tok")

            depth_detail = f"  Tokens per indent depth: {', '.join(depth_summary_parts)}"
            if use_color:
                lines.append(f"{_DIM}{depth_detail}{_RESET}")
            else:
                lines.append(depth_detail)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize tokenization of Python code with whitespace emphasis"
    )
    parser.add_argument(
        "--tokenizer-config", required=True,
        help="JSON file with tokenizer configurations "
             "(same format as run_tokenizer_analysis.py)",
    )
    parser.add_argument(
        "--tokenizers", nargs="+", default=None,
        help="Subset of tokenizer names to show (default: all)",
    )
    parser.add_argument(
        "--code-file", default=None,
        help="Python source file to tokenize (default: built-in sample)",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colours (for piping to files)",
    )
    args = parser.parse_args()

    # ── Load code ────────────────────────────────────────────────────
    if args.code_file:
        code = Path(args.code_file).read_text(encoding="utf-8")
    else:
        code = DEFAULT_CODE

    # ── Load tokenizer configs ───────────────────────────────────────
    with open(args.tokenizer_config, encoding="utf-8") as f:
        all_configs = json.load(f)

    names = args.tokenizers or list(all_configs.keys())
    missing = [n for n in names if n not in all_configs]
    if missing:
        print(f"Error: tokenizer(s) not found in config: {missing}", file=sys.stderr)
        print(f"Available: {list(all_configs.keys())}", file=sys.stderr)
        sys.exit(1)

    use_color = not args.no_color and sys.stdout.isatty()

    # ── Show the source code for reference ───────────────────────────
    if use_color:
        print(f"\n{_BOLD}Source code{_RESET} ({len(code)} chars, "
              f"{len(code.splitlines())} lines):")
        print(f"{_DIM}{'─' * 72}{_RESET}")
        for i, line in enumerate(code.splitlines(), 1):
            print(f"{_DIM}{i:3d}{_RESET} {line}")
        print(f"{_DIM}{'─' * 72}{_RESET}")
    else:
        print(f"\nSource code ({len(code)} chars, "
              f"{len(code.splitlines())} lines):")
        print("─" * 72)
        for i, line in enumerate(code.splitlines(), 1):
            print(f"{i:3d} {line}")
        print("─" * 72)

    # ── Tokenize & display ───────────────────────────────────────────
    for name in names:
        config = all_configs[name]
        try:
            wrapper = create_tokenizer_wrapper(name, config)
        except Exception as e:
            print(f"\nSkipping {name}: {e}", file=sys.stderr)
            continue

        output = visualize_tokens(name, code, wrapper, use_color)
        print(output)

    print()


if __name__ == "__main__":
    main()
