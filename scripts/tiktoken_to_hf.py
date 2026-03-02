#!/usr/bin/env python3
"""
Convert a tiktoken encoding into a HuggingFace tokenizers JSON file.

Pulls the vocabulary, pretokenizer regex, and special tokens directly from
the tiktoken Encoding object, then reconstructs BPE merges from token ranks.

Usage:
    python tiktoken_to_hf.py ENCODING_NAME OUTPUT_JSON

Examples:
    python tiktoken_to_hf.py cl100k_base cl100k_base_hf.json
    python tiktoken_to_hf.py o200k_base o200k_base_hf.json
    python tiktoken_to_hf.py gpt2 gpt2_hf.json

Requirements:
    pip install tiktoken
"""

import argparse
import json
import sys
from collections import OrderedDict

import tiktoken


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


def reconstruct_merges(vocab: dict[bytes, int], byte_encoder: dict[int, str]) -> list[str]:
    """
    Reconstruct BPE merges from the tiktoken vocabulary.

    Every multi-byte token was formed by merging two shorter tokens that also
    exist in the vocabulary. For each token (by ascending rank), we find the
    split into two existing vocab tokens whose most-recently-formed component
    has the lowest rank.
    """
    token_set = set(vocab.keys())
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])

    merges = []

    for token, rank in sorted_tokens:
        if len(token) <= 1:
            continue

        best_split = None
        best_max_rank = None

        for i in range(1, len(token)):
            left = token[:i]
            right = token[i:]

            if left in token_set and right in token_set:
                max_rank = max(vocab[left], vocab[right])
                if best_max_rank is None or max_rank < best_max_rank:
                    best_max_rank = max_rank
                    best_split = (left, right)

        if best_split is not None:
            left_str = token_bytes_to_str(best_split[0], byte_encoder)
            right_str = token_bytes_to_str(best_split[1], byte_encoder)
            merges.append(f"{left_str} {right_str}")

    return merges


def build_hf_tokenizer_json(
    vocab: dict[bytes, int],
    merges: list[str],
    byte_encoder: dict[int, str],
    pat_str: str,
    special_tokens: dict[str, int],
) -> dict:
    """Build the HuggingFace tokenizers JSON structure."""

    # Build the string vocabulary: {unicode_str: rank}
    str_vocab = OrderedDict()
    for token_bytes, rank in sorted(vocab.items(), key=lambda x: x[1]):
        token_str = token_bytes_to_str(token_bytes, byte_encoder)
        str_vocab[token_str] = rank

    # Add special tokens
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert a tiktoken encoding to a HuggingFace tokenizers JSON."
    )
    parser.add_argument(
        "encoding",
        help="Tiktoken encoding name (e.g. cl100k_base, o200k_base, gpt2)",
    )
    parser.add_argument("output", help="Path for the output .json file")
    args = parser.parse_args()

    print(f"Loading tiktoken encoding: {args.encoding}")
    try:
        enc = tiktoken.get_encoding(args.encoding)
    except Exception as e:
        print(f"Error: Could not load encoding '{args.encoding}': {e}", file=sys.stderr)
        sys.exit(1)

    # Extract everything from the encoding object
    vocab: dict[bytes, int] = enc._mergeable_ranks
    pat_str: str = enc._pat_str
    special_tokens: dict[str, int] = enc._special_tokens

    print(f"  Vocab size: {len(vocab)}")
    print(f"  Special tokens: {len(special_tokens)}")
    print(f"  Pretokenizer regex: {pat_str[:80]}{'...' if len(pat_str) > 80 else ''}")

    byte_encoder = bytes_to_unicode()

    print("Reconstructing BPE merges (this may take a moment for large vocabs)...")
    merges = reconstruct_merges(vocab, byte_encoder)
    print(f"  Reconstructed {len(merges)} merges")

    print("Building HuggingFace tokenizer JSON...")
    tokenizer_json = build_hf_tokenizer_json(
        vocab, merges, byte_encoder, pat_str, special_tokens
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {args.output}")
    print(f"  Total vocab (with special): {len(tokenizer_json['model']['vocab'])}")
    print(f"  Merges: {len(tokenizer_json['model']['merges'])}")


if __name__ == "__main__":
    main()
