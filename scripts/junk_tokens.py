#!/usr/bin/env python
"""Count 'junk' vocabulary tokens per tokenizer, in three categories.

Junk = low-value vocab tokens that waste slots, computed directly from each
tokenizer's vocab (cleaned surface), independent of the analysis pipeline. Each
token is placed in at most one category (priority: punctuation -> web -> gibberish):

  1. punctuation  — a run of >= JUNK_MIN_LEN punctuation/symbol/whitespace chars
     with NO letters or digits (decorative separators: `--------`, `========`,
     long space runs).
  2. web/markup   — URL / HTML scrape residue. Strong markers only, so bare
     valid tokens like `http`, `https`, `www`, `.com` are NOT flagged: a token is
     web junk only if it contains `://`, `www.<char>`, a `.<tld>/` path, an HTML
     entity (`&nbsp;`, `&#8217;`), an HTML tag fragment (`</p`, `/>`, `<div `), or
     an HTML attribute (`href=`, `class="`).
  3. gibberish    — hash / random-alphanumeric IDs (base16/commit hashes, tracking
     codes): ASCII, length >= JUNK_GIBBERISH_MIN_LEN, mixing letters and digits with
     either a long pure-hex run or many letter<->digit transitions. The bar is high
     on purpose so normal identifiers (`utf8`, `int64`, `base64encoded`, `covid19`)
     are NOT flagged.

Usage:
    python scripts/junk_tokens.py --tokenizer-config configs/sample_tokenizers.json \
        --output results/report_junk_tokens.json [--min-len 8] [--only NAME ...]
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

from tokenizer_analysis.core.tokenizer_wrapper import create_tokenizer_wrapper
from tokenizer_analysis.diagnostics.sanity_check import clean_token, TokenizerSanityChecker

# --- constants (junk thresholds) ---
JUNK_MIN_LEN = 8                 # runs of >= this many punct/symbol/whitespace chars are "junk"
JUNK_GIBBERISH_MIN_LEN = 12      # min length for a hash/random-alnum gibberish token
JUNK_HEX_MIN_LEN = 16            # min length for a pure-hex string to count as a hash
JUNK_ALNUM_TRANSITIONS_MIN = 4   # min letter<->digit transitions for interleaved-id gibberish

# Category 2 web/markup: strong markers only (bare http/https/www/.com must not match).
# A "sentinel" token -- a single angle-bracket-wrapped identifier like <bos>, </s>,
# <unused2896>, <String>, <|endoftext|> -- is a special/control token or a bare single
# tag, NOT web junk, and is excluded (covers special tokens clean_token misses, e.g. Gemma).
_WEB_SENTINEL = re.compile(r"<[|/]?[\w|]+>")
_WEB_PATTERNS = [
    re.compile(r"://"),                                              # scheme separator (http://, ftp://)
    re.compile(r"www\.\S"),                                          # www. + a domain char
    re.compile(r"\.(?:com|org|net|edu|gov|io|co|ru|de|uk|fr|jp|cn|info|biz|html?|php|aspx?|xml|json)/"),  # tld/ext + path
    re.compile(r"&(?:[a-zA-Z]{2,8}|#\d{2,5}|#x[0-9a-fA-F]{2,5});"),  # HTML entity &nbsp; &amp; &#8217;
    re.compile(r"/>"),                                               # self-closing tag fragment
    re.compile(r"<[a-zA-Z][a-zA-Z0-9]*\s"),                          # open tag with attributes: <a , <p , <div
    re.compile(r"(?:href|src|class|style|rel|alt|title)\s*=\s*[\"']"),  # HTML attributes
]


def is_junk(s: str, min_len: int) -> bool:
    core = s.lstrip()
    if len(core) < min_len:
        return False
    return all(unicodedata.category(c)[0] in ("P", "S", "Z", "C") or c.isspace()
               for c in core)


def is_web_markup(s: str) -> bool:
    core = s.strip()
    if not core or _WEB_SENTINEL.fullmatch(core):
        return False  # special/sentinel token or bare single tag -> not web junk
    return any(p.search(core) for p in _WEB_PATTERNS)


def is_gibberish(s: str) -> bool:
    core = s.strip()
    if len(core) < JUNK_GIBBERISH_MIN_LEN or not core.isascii():
        return False  # gibberish IDs are ASCII; this also avoids e.g. CJK dates
    has_alpha = any(c.isalpha() for c in core)
    has_digit = any(c.isdigit() for c in core)
    if not (has_alpha and has_digit):
        return False
    # pure long hex (hash-like): hex chars only, both a letter and digit present
    if len(core) >= JUNK_HEX_MIN_LEN and re.fullmatch(r"[0-9a-fA-F]+", core):
        return True
    # otherwise require it to be (almost) all alphanumeric and heavily interleaved,
    # i.e. many letter<->digit transitions (random IDs), not a word with a stray number
    if sum(c.isalnum() for c in core) < 0.9 * len(core):
        return False
    trans = sum(1 for a, b in zip(core, core[1:])
                if a.isalnum() and b.isalnum() and (a.isdigit() != b.isdigit()))
    return trans >= JUNK_ALNUM_TRANSITIONS_MIN


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-len", type=int, default=JUNK_MIN_LEN)
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args(argv)
    cfg = json.load(open(args.tokenizer_config))
    if args.only:
        cfg = {k: v for k, v in cfg.items() if k in args.only}

    per_tok = {}
    for name, tcfg in cfg.items():
        try:
            w = create_tokenizer_wrapper(name, tcfg)
            chk = TokenizerSanityChecker(w, [], name=name)  # builds vocab + char_decode
            special_ids = w.get_special_token_ids()  # exclude declared special/reserved tokens
            counts = {"punctuation": 0, "web": 0, "gibberish": 0}
            ex = {"punctuation": [], "web": [], "gibberish": []}
            for raw, tid in chk.vocab.items():
                if tid in special_ids:
                    continue
                cleaned = clean_token(str(raw), chk.char_decode)
                if not cleaned:
                    continue
                # at most one category per token, priority punctuation -> web -> gibberish
                if is_junk(cleaned, args.min_len):
                    cat = "punctuation"
                elif is_web_markup(cleaned):
                    cat = "web"
                elif is_gibberish(cleaned):
                    cat = "gibberish"
                else:
                    continue
                counts[cat] += 1
                if len(ex[cat]) < 12:
                    ex[cat].append(cleaned[:40])
            per_tok[name] = {
                # backward-compatible fields = the punctuation category
                "junk_count": counts["punctuation"], "examples": ex["punctuation"],
                "web_count": counts["web"], "web_examples": ex["web"],
                "gibberish_count": counts["gibberish"], "gibberish_examples": ex["gibberish"],
                "total_junk": sum(counts.values()),
            }
            print(f"  {name:34s} punct={counts['punctuation']:4d} web={counts['web']:4d} "
                  f"gibberish={counts['gibberish']:4d}")
        except Exception as e:
            per_tok[name] = {"error": str(e)}
            print(f"  {name:34s} ERROR: {e}", file=sys.stderr)

    out = {"junk_tokens": {"definition":
                           "low-value vocab tokens in three categories (at most one per token): "
                           f"'punctuation' = runs of >={args.min_len} punctuation/symbol/whitespace "
                           "chars with no letters/digits; 'web' = URL/HTML scrape residue (://, www., "
                           ".tld/path, HTML entities/tags/attributes; bare http/www/.com not flagged); "
                           "'gibberish' = hash/random-alphanumeric IDs (ASCII, "
                           f">={JUNK_GIBBERISH_MIN_LEN} chars, long hex run or many letter<->digit "
                           "transitions)",
                           "min_len": args.min_len,
                           "gibberish_min_len": JUNK_GIBBERISH_MIN_LEN,
                           "per_tokenizer": per_tok}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
