#!/usr/bin/env python3
"""
Convert a tiktoken encoding into a HuggingFace tokenizers JSON file,
with built-in round-trip validation.

Pulls the vocabulary, pretokenizer regex, and special tokens directly from
the tiktoken Encoding object, then reconstructs BPE merges from token ranks.

Usage (convert):
    python tiktoken_to_hf.py cl100k_base cl100k_base_hf.json
    python tiktoken_to_hf.py o200k_base o200k_base_hf.json
    python tiktoken_to_hf.py gpt2 gpt2_hf.json

Usage (test):
    python tiktoken_to_hf.py --test              # offline synthetic test
    python tiktoken_to_hf.py --test gpt2         # also test a real encoding
    python tiktoken_to_hf.py --test cl100k_base  # also test cl100k_base

Requirements:
    pip install tiktoken tokenizers
"""

import argparse
import json
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
import re as _re

import tiktoken


# ═══════════════════════════════════════════════════════════════════════════
# Conversion logic
# ═══════════════════════════════════════════════════════════════════════════

def bytes_to_unicode() -> dict[int, str]:
    """
    Build the byte-to-unicode mapping used by GPT-2 / tiktoken BPE.
    Maps each byte value (0-255) to a visible unicode character.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def token_bytes_to_str(token: bytes, byte_encoder: dict[int, str]) -> str:
    """Convert raw token bytes to the GPT-2 style unicode string representation."""
    return "".join(byte_encoder[b] for b in token)


def _apply_bpe(token_bytes: bytes, merge_map: dict[tuple[bytes, bytes], int]) -> list[bytes]:
    """Simulate HuggingFace-style left-to-right BPE on raw bytes.

    *merge_map* maps ``(left, right)`` to the merge's position in the merge
    list (lower = applied earlier).  Returns the final list of byte-string
    pieces after all applicable merges have been applied.

    Uses a priority-queue approach: only considers pairs actually present
    in the current piece list, rather than iterating all known merges.
    """
    pieces: list[bytes] = [bytes([b]) for b in token_bytes]

    while len(pieces) > 1:
        # Find the adjacent pair with the lowest merge index.
        best_idx = None
        best_merge_pos = None
        for i in range(len(pieces) - 1):
            pair = (pieces[i], pieces[i + 1])
            merge_pos = merge_map.get(pair)
            if merge_pos is not None and (best_merge_pos is None or merge_pos < best_merge_pos):
                best_idx = i
                best_merge_pos = merge_pos
        if best_idx is None:
            break
        # Apply this merge left-to-right (all non-overlapping occurrences).
        left, right = pieces[best_idx], pieces[best_idx + 1]
        merged = left + right
        i = 0
        while i < len(pieces) - 1:
            if pieces[i] == left and pieces[i + 1] == right:
                pieces[i:i + 2] = [merged]
            else:
                i += 1

    return pieces


def reconstruct_merges(vocab: dict[bytes, int], byte_encoder: dict[int, str]) -> list[str]:
    """
    Reconstruct BPE merges from the tiktoken vocabulary.

    For each multi-byte token (by ascending rank), we simulate the
    HuggingFace left-to-right BPE algorithm using all previously
    reconstructed merges.  The last pair remaining before the final merge
    tells us exactly which merge rule to emit -- this guarantees the
    reconstructed merge list reproduces tiktoken's tokenization under
    HuggingFace's greedy left-to-right strategy.
    """
    token_set = set(vocab.keys())
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])

    merges: list[str] = []
    merge_map: dict[tuple[bytes, bytes], int] = {}

    fallback_count = 0

    for token, rank in sorted_tokens:
        if len(token) <= 1:
            continue

        pieces = _apply_bpe(token, merge_map)

        if len(pieces) == 2:
            left, right = pieces[0], pieces[1]
        else:
            fallback_count += 1
            best_split = None
            best_max_rank = None
            for i in range(1, len(token)):
                left_c = token[:i]
                right_c = token[i:]
                if left_c in token_set and right_c in token_set:
                    max_rank = max(vocab[left_c], vocab[right_c])
                    if best_max_rank is None or max_rank < best_max_rank:
                        best_max_rank = max_rank
                        best_split = (left_c, right_c)
            if best_split is None:
                continue
            left, right = best_split

        merge_map[(left, right)] = len(merges)
        left_str = token_bytes_to_str(left, byte_encoder)
        right_str = token_bytes_to_str(right, byte_encoder)
        merges.append(f"{left_str} {right_str}")

    if fallback_count > 0:
        print(f"  WARNING: Fallback heuristic used {fallback_count} times", file=sys.stderr)

    return merges


def build_hf_tokenizer_json(
    vocab: dict[bytes, int],
    merges: list[str],
    byte_encoder: dict[int, str],
    pat_str: str,
    special_tokens: dict[str, int],
) -> dict:
    """Build the HuggingFace tokenizers JSON structure."""

    str_vocab = OrderedDict()
    for token_bytes, rank in sorted(vocab.items(), key=lambda x: x[1]):
        token_str = token_bytes_to_str(token_bytes, byte_encoder)
        str_vocab[token_str] = rank

    added_tokens = []
    for token_str, token_id in sorted(special_tokens.items(), key=lambda x: x[1]):
        str_vocab[token_str] = token_id
        added_tokens.append(
            {
                "id": token_id,
                "content": token_str,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": pat_str},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": dict(str_vocab),
            "merges": merges,
        },
    }

    return tokenizer_json


def convert_encoding(enc: tiktoken.Encoding) -> dict:
    """Convert a tiktoken Encoding object to HF tokenizer JSON dict."""
    byte_encoder = bytes_to_unicode()
    merges = reconstruct_merges(enc._mergeable_ranks, byte_encoder)
    return build_hf_tokenizer_json(
        enc._mergeable_ranks, merges, byte_encoder,
        convert_possessive_to_atomic(enc._pat_str), enc._special_tokens,
    )


def validate(enc: tiktoken.Encoding, hf_json_path: str, name: str = "") -> bool:
    """Validate a converted HF tokenizer against the source tiktoken encoding.

    Loads the HF tokenizer from *hf_json_path*, encodes every string in the
    built-in test corpus with both tokenizers, and compares token IDs.

    Returns True if every test string produces identical IDs.
    """
    from tokenizers import Tokenizer as HFTokenizer

    hf_tok = HFTokenizer.from_file(hf_json_path)
    label = name or hf_json_path

    print(f"\nValidating {label}...")
    t0 = time.time()
    summary = _compare(enc, hf_tok, _TEST_CORPUS, label)
    summary.test_time_s = time.time() - t0

    _print_failures(summary)
    _print_summary(summary)

    return summary.ok
def convert_possessive_to_atomic(pattern: str) -> str:
    """Convert PCRE possessive quantifiers to atomic groups.

    Tokenizes the regex, identifies each possessive quantifier
    (a quantifier followed by +), and rewrites as (?>atom quantifier).

    Handles: character classes [...], escape sequences, unicode
    properties \\p{...}, groups (...), and nested structures.

    Returns the pattern unchanged if no possessive quantifiers are found.
    """
    tokens = _tokenize_regex(pattern)
    return _rewrite_possessives(tokens)


# ═══════════════════════════════════════════════════════════════════════════
# Regex tokenizer
# ═══════════════════════════════════════════════════════════════════════════

def _tokenize_regex(pattern: str) -> list[dict]:
    """Tokenize a regex into structural elements.

    Each token is a dict with 'type' and 'text':
      'atom'       — a single matchable unit (char, escape, class, group, property)
      'quant'      — a quantifier (?, *, +, {n,m})
      'possessive' — the + after a quantifier that makes it possessive
      'other'      — alternation |, anchors ^$, lookaheads, etc.
    """
    tokens = []
    i = 0

    while i < len(pattern):
        c = pattern[i]

        # ── Escape sequence (\x, \p{Lu}, etc.) ───────────────────────
        if c == '\\':
            esc, end = _read_escape(pattern, i)
            tokens.append({'type': 'atom', 'text': esc})
            i = end
            continue

        # ── Character class [...] ─────────────────────────────────────
        if c == '[':
            cls, end = _read_char_class(pattern, i)
            tokens.append({'type': 'atom', 'text': cls})
            i = end
            continue

        # ── Group (...) ───────────────────────────────────────────────
        if c == '(':
            grp, end = _read_group(pattern, i)
            # Lookaheads/lookbehinds are not quantifiable atoms
            if (len(grp) > 2 and grp[1] == '?'
                    and grp[2] in '=!<'):
                tokens.append({'type': 'other', 'text': grp})
            else:
                tokens.append({'type': 'atom', 'text': grp})
            i = end
            continue

        # ── Quantifiers ?, *, + ───────────────────────────────────────
        if c in '?*+':
            if c == '+' and tokens and tokens[-1]['type'] == 'quant':
                tokens.append({'type': 'possessive', 'text': '+'})
            else:
                tokens.append({'type': 'quant', 'text': c})
            i += 1
            continue

        # ── Brace quantifier {n}, {n,}, {n,m} ────────────────────────
        if c == '{':
            brace, end = _read_brace_quantifier(pattern, i)
            if brace is not None:
                tokens.append({'type': 'quant', 'text': brace})
                i = end
                if i < len(pattern) and pattern[i] == '+':
                    tokens.append({'type': 'possessive', 'text': '+'})
                    i += 1
                continue
            tokens.append({'type': 'atom', 'text': c})
            i += 1
            continue

        # ── Alternation, anchors ──────────────────────────────────────
        if c in '|^$':
            tokens.append({'type': 'other', 'text': c})
            i += 1
            continue

        # ── Dot (any char) ────────────────────────────────────────────
        if c == '.':
            tokens.append({'type': 'atom', 'text': c})
            i += 1
            continue

        # ── Literal character ─────────────────────────────────────────
        tokens.append({'type': 'atom', 'text': c})
        i += 1

    return tokens


def _read_escape(pattern: str, start: int) -> tuple[str, int]:
    """Read an escape sequence starting at \\."""
    i = start + 1
    if i >= len(pattern):
        return ('\\', i)

    c = pattern[i]
    # Unicode property: \p{...} or \P{...}
    if c in 'pP' and i + 1 < len(pattern) and pattern[i + 1] == '{':
        end = pattern.index('}', i + 2) + 1
        return (pattern[start:end], end)

    return (pattern[start:i + 1], i + 1)


def _read_char_class(pattern: str, start: int) -> tuple[str, int]:
    """Read a character class [...] including nested escapes."""
    i = start + 1
    if i < len(pattern) and pattern[i] == '^':
        i += 1
    if i < len(pattern) and pattern[i] == ']':
        i += 1  # literal ] at start

    while i < len(pattern):
        if pattern[i] == '\\' and i + 1 < len(pattern):
            if pattern[i + 1] in 'pP' and i + 2 < len(pattern) and pattern[i + 2] == '{':
                i = pattern.index('}', i + 3) + 1
            else:
                i += 2
        elif pattern[i] == ']':
            return (pattern[start:i + 1], i + 1)
        else:
            i += 1

    return (pattern[start:], len(pattern))


def _read_group(pattern: str, start: int) -> tuple[str, int]:
    """Read a parenthesized group, respecting nesting."""
    depth = 1
    i = start + 1

    while i < len(pattern) and depth > 0:
        if pattern[i] == '\\' and i + 1 < len(pattern):
            i += 2
        elif pattern[i] == '[':
            _, i = _read_char_class(pattern, i)
        elif pattern[i] == '(':
            depth += 1
            i += 1
        elif pattern[i] == ')':
            depth -= 1
            i += 1
        else:
            i += 1

    return (pattern[start:i], i)


def _read_brace_quantifier(pattern: str, start: int) -> tuple[str | None, int]:
    """Try to read {n}, {n,}, or {n,m}. Returns (text, end) or (None, start).

    Distinguishes quantifier braces (digits + comma) from unicode
    property braces like \\p{Lu} by checking contents.
    """
    i = start + 1
    while i < len(pattern) and pattern[i] in '0123456789,':
        i += 1

    if i < len(pattern) and pattern[i] == '}' and i > start + 1:
        inner = pattern[start + 1:i]
        if _re.match(r'^\d+(?:,\d*)?$', inner):
            return (pattern[start:i + 1], i + 1)

    return (None, start)


def _rewrite_possessives(tokens: list[dict]) -> str:
    """Rewrite atom + quantifier + possessive → (?>atom quantifier)."""
    result = []
    i = 0

    while i < len(tokens):
        if (i + 2 < len(tokens)
                and tokens[i]['type'] == 'atom'
                and tokens[i + 1]['type'] == 'quant'
                and tokens[i + 2]['type'] == 'possessive'):
            atom = tokens[i]['text']
            quant = tokens[i + 1]['text']
            result.append(f'(?>{atom}{quant})')
            i += 3
            continue

        if tokens[i]['type'] == 'possessive':
            i += 1  # shouldn't happen in well-formed regex
            continue

        result.append(tokens[i]['text'])
        i += 1

    return ''.join(result)


def convert(encoding_name: str, output_path: str) -> dict:
    """Load a tiktoken encoding by name, convert, save, and validate.

    After writing the HF tokenizer JSON to *output_path*, the corpus
    validation suite runs automatically.  Raises ``RuntimeError`` if any
    test string produces different token IDs between tiktoken and the
    converted HF tokenizer.
    """
    print(f"Loading tiktoken encoding: {encoding_name}")
    enc = tiktoken.get_encoding(encoding_name)

    vocab = enc._mergeable_ranks
    pat_str = convert_possessive_to_atomic(enc._pat_str)
    special_tokens = enc._special_tokens

    print(f"  Vocab size: {len(vocab)}")
    print(f"  Special tokens: {len(special_tokens)}")
    print(f"  Pretokenizer regex: {pat_str[:80]}{'...' if len(pat_str) > 80 else ''}")

    print("Reconstructing BPE merges (this may take a moment for large vocabs)...")
    tokenizer_json = convert_encoding(enc)
    print(f"  Reconstructed {len(tokenizer_json['model']['merges'])} merges")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {output_path}")
    print(f"  Total vocab (with special): {len(tokenizer_json['model']['vocab'])}")
    print(f"  Merges: {len(tokenizer_json['model']['merges'])}")

    # ── Validate the saved file against tiktoken ──────────────────────────
    ok = validate(enc, output_path, name=encoding_name)
    if not ok:
        raise RuntimeError(
            f"Validation FAILED for {encoding_name}: the converted tokenizer "
            f"at {output_path} does not produce identical token IDs. "
            f"See failure details above."
        )

    return tokenizer_json


# ═══════════════════════════════════════════════════════════════════════════
# Built-in test suite
# ═══════════════════════════════════════════════════════════════════════════

# GPT-2 style pretokenizer regex (used by the synthetic encoding)
_GPT2_REGEX = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?\p{L}+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _build_synthetic_encoding() -> tiktoken.Encoding:
    """Build a small but realistic tiktoken Encoding for offline testing.

    The vocabulary is hand-crafted with carefully ordered merges covering:
    single bytes (256), common English bigrams, trigrams built from bigrams,
    space-prefixed word pieces, digit sequences, punctuation combos,
    UTF-8 multi-byte characters, and longer word pieces up to 7 bytes.

    Every multi-byte token's BPE simulation trace is documented inline to
    verify that the merge reconstruction will produce exactly 2 pieces.
    """
    mergeable_ranks: dict[bytes, int] = {}
    for i in range(256):
        mergeable_ranks[bytes([i])] = i

    # Multi-byte tokens in strict merge order.  Each entry's comment traces
    # the BPE simulation with all earlier merges already in the merge map.
    # The simulation must yield exactly 2 pieces (the merge's left and right
    # components), except where the fallback heuristic is noted.
    extensions = [
        # ── Bigrams from single bytes (256-291) ──────────────────────────
        (b"th", 256),   (b"he", 257),   (b"in", 258),   (b"er", 259),
        (b"an", 260),   (b"re", 261),   (b"on", 262),   (b"en", 263),
        (b"at", 264),   (b"or", 265),   (b"es", 266),   (b"te", 267),
        (b"st", 268),   (b"ed", 269),   (b"ti", 270),   (b"is", 271),
        (b"al", 272),   (b"ar", 273),   (b"ou", 274),   (b"se", 275),
        (b"le", 276),   (b"it", 277),   (b"ng", 278),
        (b" t", 279),   (b" a", 280),   (b" s", 281),   (b" i", 282),
        (b" o", 283),   (b" w", 284),   (b" c", 285),   (b" h", 286),
        (b" m", 287),   (b" d", 288),   (b" f", 289),   (b" p", 290),
        (b" b", 291),

        # ── Trigrams (292-298) ────────────────────────────────────────────
        # "the": [t,h,e] -> [th,e]. Merge: th + e
        (b"the", 292),
        # "ing": [i,n,g] -> [in,g]. Merge: in + g
        (b"ing", 293),
        # "and": [a,n,d] -> [an,d]. Merge: an + d
        (b"and", 294),
        # "ent": [e,n,t] -> [en,t]. Merge: en + t
        (b"ent", 295),
        # "ter": [t,e,r] -> [te,r]. Merge: te + r
        (b"ter", 296),
        # "est": [e,s,t] -> [es,t]. Merge: es + t
        (b"est", 297),
        # "tion": [t,i,o,n] -> [ti,o,n] -> [ti,on]. Merge: ti + on
        (b"tion", 298),

        # ── Space-prefixed (299-308) ──────────────────────────────────────
        # " th": [SP,t,h] -> [SPt,h]. Merge: " t" + h
        (b" th", 299),
        # " the": [SP,t,h,e] -> [SPt,h,e] -> [SPth,e]. Merge: " th" + e
        (b" the", 300),
        # " in": [SP,i,n] -> [SP,in]. Merge: SP + in
        (b" in", 301),
        # " an": [SP,a,n] -> [SP,an]. Merge: SP + an
        (b" an", 302),
        # " is": [SP,i,s] -> [SP,is]. Merge: SP + is
        (b" is", 303),
        # " st": [SP,s,t] -> [SP,st]. Merge: SP + st
        (b" st", 304),
        # " and": [SP,a,n,d] -> [SP,an,d] -> [SPan,d]. Merge: " an" + d
        (b" and", 305),
        # " or": [SP,o,r] -> [SP,or]. Merge: SP + or
        (b" or", 306),
        # "ther": [t,h,e,r] -> [th,e,r] -> [the,r]. Merge: the + r
        (b"ther", 307),
        # " ther": [SP,t,h,e,r] -> [SPt,h,e,r] -> [SPth,e,r]
        #          -> [SPthe,r]. Merge: " the" + r
        (b" ther", 308),

        # ── Digit combos (309-312) ────────────────────────────────────────
        (b"10", 309),    # 1 + 0
        (b"00", 310),    # 0 + 0
        # "100": [1,0,0] -> [10,0]. Merge: 10 + 0
        (b"100", 311),
        # "000": [0,0,0] -> [00,0]. Merge: 00 + 0
        (b"000", 312),

        # ── Punctuation (313-314) ─────────────────────────────────────────
        (b"..", 313),    # . + .
        # "...": [.,.,.] -> [..,.]  Merge: .. + .
        (b"...", 314),

        # ── UTF-8 multi-byte (315-316) ────────────────────────────────────
        (b"\xc3\xa9", 315),   # e-acute: 2 UTF-8 bytes merged
        (b"\xc3\xb1", 316),   # n-tilde: 2 UTF-8 bytes merged

        # ── Word pieces (317-324) ─────────────────────────────────────────
        # "hat": [h,a,t] -> [h,at]. Merge: h + at
        (b"hat", 317),
        # "that": [t,h,a,t] -> [th,a,t] -> [th,at]. Merge: th + at
        (b"that", 318),
        # " that": [SP,t,h,a,t] -> [SPt,h,a,t] -> [SPth,a,t]
        #          -> [SPth,at]. Merge: " th" + at
        (b" that", 319),
        # "test": [t,e,s,t] -> [te,s,t] -> [te,st]. Merge: te + st
        (b"test", 320),
        # " test": [SP,t,e,s,t] -> [SP,te,s,t] -> [SP,te,st]
        #          -> [SP,test]. Merge: SP + test
        (b" test", 321),
        # "ting": [t,i,n,g] -> [ti,n,g] -> [ti,ng]. Merge: ti + ng
        (b"ting", 322),
        # "sting": [s,t,i,n,g] -> [st,i,n,g] -> [st,i,ng]
        #          3 pieces -> fallback picks st+ing (max_rank=293)
        #          over s+ting (max_rank=322)
        (b"sting", 323),
        # "testing": [t,e,s,t,i,n,g] -> [t,e,s,t,in,g] -> [te,s,t,in,g]
        #            -> [te,st,in,g] -> [te,st,ing] -> [test,ing]
        #            Merge: test + ing
        (b"testing", 324),
    ]

    for token, rank in extensions:
        mergeable_ranks[token] = rank

    special_tokens = {"<|endoftext|>": 325, "<|pad|>": 326}

    return tiktoken.Encoding(
        name="synthetic_test",
        pat_str=_GPT2_REGEX,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
        explicit_n_vocab=len(mergeable_ranks) + len(special_tokens),
    )


# ── Test corpus ───────────────────────────────────────────────────────────
# Each category targets specific tokenization behavior: boundary handling,
# byte-level fallback, regex splits, merge interactions, etc.

_TEST_CORPUS: dict[str, list[str]] = {
    "basic_english": [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "that is the thing",
        "the the the",
        "testing the thing",
        "testing is interesting",
    ],
    "whitespace": [
        "   leading spaces",
        "trailing spaces   ",
        "  both  sides  ",
        "multiple   spaces   between   words",
        "tabs\there\tand\tthere",
        "newline\nand\nanother",
        "\r\n windows style \r\n line breaks",
        "    indented\n    code\n    block",
    ],
    "numbers": [
        "123456789",
        "3.14159",
        "$1,234.56",
        "2024-01-15",
        "100 1000 10000",
        "0xFF00FF",
    ],
    "punctuation": [
        "!@#$%^&*()",
        "...ellipsis...",
        "a/b/c/d/e",
        "under_score_case",
        "---",
        "===",
    ],
    "code": [
        "def hello():\n    print('Hello!')\n",
        "for i in range(10):\n    x += i * 2\n",
        "if (x > 0) { return x; }",
        "SELECT * FROM users WHERE id = 42;",
        "const f = (x) => x * x;",
    ],
    "multilingual": [
        "caf\u00e9",                               # cafe (uses e token)
        "espa\u00f1ol",                             # espanol (uses n token)
        "na\u00efve",                               # naive
        "Hello \u4f60\u597d",                       # Hello + Chinese
        "\u3053\u3093\u306b\u3061\u306f",           # Japanese hiragana
        "\uc548\ub155\ud558\uc138\uc694",           # Korean
        "\u041f\u0440\u0438\u0432\u0435\u0442",     # Russian
    ],
    "contractions": [
        "I'm", "don't", "they're", "it's",
        "we've", "he'll", "won't", "let's",
    ],
    "repetition": [
        "aaa", "aaaa", "aaaaa", "aaaaaa",
        "the the the the the",
        "......",
        "------",
        "      ",
    ],
    "edge_cases": [
        "",          # empty string
        " ",         # single space
        "a",         # single character
        "\n",        # single newline
        "\t",        # single tab
        "a" * 200,   # long single-character repeat
        " " * 50,    # long whitespace
        "word " * 20,
    ],
    "passages": [
        (
            "The thing is that the thing in the world "
            "is the thing that is the interesting thing."
        ),
        (
            "Testing 100 things at 1000 iterations. "
            "The estimated time is 10000 seconds and the "
            "testing is done in the test folder."
        ),
        (
            "In the thing there is nothing and then "
            "there is something and that something is "
            "the thing that the thing is and that the "
            "thing is not."
        ),
    ],
}


# ── Test runner ───────────────────────────────────────────────────────────

@dataclass
class _TestResult:
    category: str
    text: str
    tiktoken_ids: list[int]
    hf_ids: list[int]
    match: bool
    error: str = ""


@dataclass
class _TestSummary:
    encoding_name: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    failures: list[_TestResult] = field(default_factory=list)
    conversion_time_s: float = 0.0
    test_time_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.errors == 0


def _compare(
    enc: tiktoken.Encoding,
    hf_tok,  # tokenizers.Tokenizer
    corpus: dict[str, list[str]],
    name: str,
) -> _TestSummary:
    """Encode every string with both tokenizers and compare token IDs."""

    summary = _TestSummary(encoding_name=name)

    for category, texts in corpus.items():
        cat_pass = cat_fail = 0

        for text in texts:
            summary.total += 1
            try:
                tt_ids = list(enc.encode(text, allowed_special="all"))
                hf_ids = list(hf_tok.encode(text).ids)
                match = tt_ids == hf_ids

                if match:
                    summary.passed += 1
                    cat_pass += 1
                else:
                    summary.failed += 1
                    cat_fail += 1
                    summary.failures.append(_TestResult(
                        category=category, text=text,
                        tiktoken_ids=tt_ids, hf_ids=hf_ids, match=False,
                    ))
            except Exception as e:
                summary.errors += 1
                cat_fail += 1
                summary.failures.append(_TestResult(
                    category=category, text=text,
                    tiktoken_ids=[], hf_ids=[], match=False, error=str(e),
                ))

        mark = "\u2713" if cat_fail == 0 else "\u2717"
        print(f"  {mark} {category:<20s}  {cat_pass}/{cat_pass + cat_fail}")

    return summary


def _print_failures(summary: _TestSummary, limit: int = 25):
    if not summary.failures:
        return
    print(f"\n{'=' * 72}")
    print("  FAILURE DETAILS")
    print("=" * 72)
    for i, r in enumerate(summary.failures[:limit]):
        preview = repr(r.text[:80]) + ("..." if len(r.text) > 80 else "")
        print(f"\n  [{i+1}] {r.category}")
        print(f"      Text:     {preview}")
        if r.error:
            print(f"      Error:    {r.error}")
        else:
            print(f"      tiktoken: {r.tiktoken_ids[:30]}"
                  f"{'...' if len(r.tiktoken_ids) > 30 else ''}")
            print(f"      HF:       {r.hf_ids[:30]}"
                  f"{'...' if len(r.hf_ids) > 30 else ''}")
            for j, (a, b) in enumerate(zip(r.tiktoken_ids, r.hf_ids)):
                if a != b:
                    print(f"      Diverges at position {j}: tiktoken={a} vs HF={b}")
                    break
            if len(r.tiktoken_ids) != len(r.hf_ids):
                print(f"      Lengths: tiktoken={len(r.tiktoken_ids)}"
                      f" vs HF={len(r.hf_ids)}")
    if len(summary.failures) > limit:
        print(f"\n  ... and {len(summary.failures) - limit} more (truncated)")


def _print_summary(summary: _TestSummary):
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY \u2014 {summary.encoding_name}")
    print("=" * 72)
    print(f"  Tests:      {summary.total}")
    print(f"  Passed:     {summary.passed}")
    print(f"  Failed:     {summary.failed}")
    print(f"  Errors:     {summary.errors}")
    if summary.conversion_time_s:
        print(f"  Convert:    {summary.conversion_time_s:.2f}s")
    print(f"  Test time:  {summary.test_time_s:.2f}s")
    print(f"\n  {'PASS' if summary.ok else 'FAIL'}\n")


def _run_test(enc: tiktoken.Encoding, name: str) -> _TestSummary:
    """Convert a tiktoken encoding, save to temp file, validate against corpus."""
    from tokenizers import Tokenizer as HFTokenizer

    print(f"Converting {name}...")
    t0 = time.time()
    tokenizer_json = convert_encoding(enc)
    conversion_time = time.time() - t0
    print(f"  {len(tokenizer_json['model']['merges'])} merges in {conversion_time:.2f}s")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)
        tmp_path = f.name

    hf_tok = HFTokenizer.from_file(tmp_path)

    print(f"\nComparing encodings:")
    t0 = time.time()
    summary = _compare(enc, hf_tok, _TEST_CORPUS, name)
    summary.test_time_s = time.time() - t0
    summary.conversion_time_s = conversion_time

    _print_failures(summary)
    _print_summary(summary)

    import os
    os.unlink(tmp_path)

    return summary


def run_tests(real_encoding: str | None = None) -> bool:
    """Run the built-in test suite.  Returns True if all tests pass.

    Always tests with a hand-crafted synthetic encoding (no network needed).
    If *real_encoding* is provided, also converts and tests that tiktoken
    encoding (requires network access to download data on first run).
    """
    print("=" * 72)
    print("  tiktoken \u2192 HuggingFace: Round-Trip Validation")
    print("=" * 72)
    print()

    all_ok = True

    # ── Synthetic encoding (always runs, no network) ──────────────────────
    print("\u2500" * 72)
    print("  Synthetic encoding (offline)")
    print("\u2500" * 72)
    enc = _build_synthetic_encoding()
    print(f"  Vocab: {len(enc._mergeable_ranks)} tokens, "
          f"{len(enc._special_tokens)} special\n")
    s = _run_test(enc, "synthetic")
    if not s.ok:
        all_ok = False

    # ── Real encoding (optional) ──────────────────────────────────────────
    if real_encoding:
        print("\u2500" * 72)
        print(f"  Real encoding: {real_encoding}")
        print("\u2500" * 72)
        try:
            enc = tiktoken.get_encoding(real_encoding)
            print(f"  Vocab: {len(enc._mergeable_ranks)} tokens, "
                  f"{len(enc._special_tokens)} special\n")
            s = _run_test(enc, real_encoding)
            if not s.ok:
                all_ok = False
        except Exception as e:
            print(f"  \u2717 Could not load {real_encoding}: {e}")
            print("    (Likely needs network access to download tiktoken data)\n")
            all_ok = False

    # ── Final verdict ─────────────────────────────────────────────────────
    print("=" * 72)
    if all_ok:
        print("  \u2713 ALL TESTS PASSED")
    else:
        print("  \u2717 SOME TESTS FAILED")
    print("=" * 72)

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert a tiktoken encoding to HuggingFace tokenizers JSON.",
        epilog=(
            "examples:\n"
            "  %(prog)s cl100k_base out.json      Convert cl100k_base\n"
            "  %(prog)s --test                     Run built-in validation\n"
            "  %(prog)s --test gpt2                Validate with gpt2 encoding\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test", nargs="?", const=True, default=False, metavar="ENCODING",
        help="Run round-trip validation tests. Optionally name a real tiktoken "
             "encoding to also test (e.g. gpt2, cl100k_base, o200k_base).",
    )
    parser.add_argument(
        "encoding", nargs="?",
        help="Tiktoken encoding name (e.g. cl100k_base, o200k_base, gpt2)",
    )
    parser.add_argument(
        "output", nargs="?",
        help="Output path for the HuggingFace tokenizer .json file",
    )
    args = parser.parse_args()

    # ── Test mode ─────────────────────────────────────────────────────────
    if args.test is not False:
        real_enc = args.test if isinstance(args.test, str) else args.encoding
        ok = run_tests(real_encoding=real_enc)
        sys.exit(0 if ok else 1)

    # ── Convert mode ──────────────────────────────────────────────────────
    if not args.encoding or not args.output:
        parser.error("Provide ENCODING and OUTPUT, or use --test for validation.")

    try:
        convert(args.encoding, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
