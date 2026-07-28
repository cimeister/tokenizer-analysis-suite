#!/usr/bin/env python
"""Vocabulary usage breakdown: Active / Rare / Uncommon / Unseen + Scaffold (corpus-based).

Run a tokenizer over a fixed corpus and, for every merge-created vocab token t,
measure how it is used. Each non-base BPE token is produced by exactly one merge,
so its binary merge tree is fixed; whenever an emitted token F occurs, its internal
tree nodes were necessarily formed and then consumed to build it. So decomposing
the actually-emitted tokens is an exact account of formations.

Per merge token t:
  final_freq[t]    = times t is emitted as a standalone FINAL token
  stepping_freq[t] = times t is an internal node of an emitted token (formed then consumed)
  formed[t]        = final_freq + stepping_freq
  standalone_rate[t] = final_freq / total_final_tokens   (corpus-size-invariant)
  survival[t]      = final_freq / formed                 (corpus-size-invariant)

Every merge token is treated identically -- byte-fragments (incomplete-UTF-8 pieces)
get NO special handling; they land in whatever bucket their behaviour puts them.
Thresholds are RATES/RATIOS (corpus-invariant), not absolute counts:
  UNCOMMON_RATE_PER_M  -- standalone_rate below this (per million) = "very rarely on its own"
  RARE_RATE_PER_M      -- standalone_rate below this (per million) = "rarely on its own"
  SCAFFOLD_SURVIVAL_MAX -- survival below this = "surfaces <10% of the times built" = mostly a merge step

Mutually-exclusive partition of the merge vocabulary by standalone rate (sum to 100%):
  Active   : formed>0 and standalone_rate >= RARE_RATE_PER_M/1e6              (on its own a normal amount)
  Rare     : formed>0 and UNCOMMON_RATE_PER_M/1e6 <= standalone_rate < RARE_RATE_PER_M/1e6
  Uncommon : formed>0 and standalone_rate < UNCOMMON_RATE_PER_M/1e6
  Unseen   : formed == 0   (never produced in ANY role on this corpus -- neither final nor a merge step)

Scaffold (OVERLAID, not part of the partition) = (Uncommon or Rare) with survival < SCAFFOLD_SURVIVAL_MAX:
  rarely standalone AND mostly a merge step -- the stepping-stone tokens.

Scaffold measures rarely-exercised embedding capacity, NOT removable waste (these tokens
are structurally required to build the tokens that do surface). Distinct from the absolute
"dead vocab" (normalizer-unreachable) and "junk" metrics. Unigram / merge-less tokenizers:
no merge tree -> not analyzed.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

# --- constants (corpus-invariant thresholds; reported in output) ---
UNCOMMON_RATE_PER_M = 1.0    # standalone_rate below this (per million) = Uncommon
RARE_RATE_PER_M = 5.0        # standalone_rate below this (per million) = Rare or Uncommon; at/above = Active
SCAFFOLD_SURVIVAL_MAX = 0.1  # survival below this = "surfaces <10% of the times built" = mostly a merge step
DEFAULT_MATH_BYTES = 12_000_000
DEFAULT_CODE_BYTES = 12_000_000
CAP = 2000               # cap chunk length to bound per-batch memory (boundary effects negligible)

FLORES = "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/flores_parallel_data"
FINEMATH = "/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceTB/finemath/finemath-4plus"
SC_PY = "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/starcoderdata/thresholds/python/threshold_0"
SC_JS = "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/starcoderdata/thresholds/javascript/threshold_0"

# Natural-language corpus: FineWeb2, one directory per language (<FW2>/<lang>/*.parquet), sampled
# over the FLORES-200 language set so the language coverage matches the FLORES panels.
# FineWeb2 carries no English (its eng_Latn directory is empty), so English comes from FineWeb-1.
# Restored 2026-07-17: this had been switched to reading FLORES-200 text itself, which changed every
# vocabulary-usage number in the reports (Unseen collapsed from ~20% to ~6% because FLORES is 211
# short parallel sentences per language rather than web text). The FineWeb sample is the intended
# corpus. The original sample's languages/seed/size were never recorded, so this is a NEW
# reproducible definition: FLORES-200 language set, equal byte budget per language, seed 0.
FW2 = "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-2_0_1-quality_33-filterrobots/data/output"
FW1_ENG = "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-1_3_0-quality_33-filterrobots/data/output"
DEFAULT_TEXT_BYTES = 45_000_000


def _strip_starcoder(t):
    for tag in ("reponame", "filename", "gh_stars"):
        t = re.sub(rf"<{tag}>[^<\n]*", "", t)
    return t


def build_corpus(math_bytes, code_bytes, text_bytes=DEFAULT_TEXT_BYTES):
    """FineWeb sample over the FLORES-200 language set + math (FineMath) and code (StarCoder
    py+js). Deterministic (seed 0). Returns a list of <=CAP-char raw-text chunks.

    Equal byte budget per language. A language that yields nothing is reported by name rather
    than silently dropped: a missing language changes the vocabulary-usage buckets, so it must
    be visible in the run log and in the artifact's `text_langs_empty`.
    """
    import pyarrow.parquet as pq
    random.seed(0)
    docs = []

    def sample(pattern, col, budget, strip=False):
        got = 0
        for shard in sorted(glob.glob(pattern)):
            for t in pq.read_table(shard, columns=[col]).column(col).to_pylist():
                if not t:
                    continue
                if strip:
                    t = _strip_starcoder(t)
                docs.extend(t[i:i + CAP] for i in range(0, len(t), CAP))
                got += len(t.encode("utf-8", "ignore"))
                if got >= budget:
                    return got
        return got

    langs = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{FLORES}/*.txt"))
    langs = [l for l in langs if not l.startswith("_")]   # drop _mapping_report, not a language
    per_lang = text_bytes // max(1, len(langs))
    tot, empty, short = 0, [], []
    for lang in langs:
        # FineWeb2 has no English; FineWeb-1 supplies it, nested under CC-MAIN-*/ dumps.
        pattern = (f"{FW1_ENG}/*/*.parquet" if lang == "eng_Latn"
                   else f"{FW2}/{lang}/*.parquet")
        got = sample(pattern, "text", per_lang)
        tot += got
        if got == 0:
            empty.append(lang)
        elif got < per_lang:
            short.append(lang)
    if empty:
        print(f"WARNING: {len(empty)} FLORES language(s) yielded NO FineWeb text and are absent "
              f"from the corpus: {empty}", flush=True)

    mb = sample(f"{FINEMATH}/*.parquet", "text", math_bytes)
    cpy = sample(f"{SC_PY}/*.parquet", "content", code_bytes // 2, strip=True)
    cjs = sample(f"{SC_JS}/*.parquet", "content", code_bytes // 2, strip=True)
    random.shuffle(docs)
    print(f"corpus: {len(docs)} chunks, FineWeb {tot/1e6:.1f} MB over {len(langs)-len(empty)}/"
          f"{len(langs)} langs ({len(short)} under budget) + math {mb/1e6:.1f} MB + "
          f"code {(cpy+cjs)/1e6:.1f} MB", flush=True)
    build_corpus.stats = {"text_bytes": tot, "text_langs": len(langs) - len(empty),
                          "text_langs_requested": len(langs), "text_langs_empty": empty,
                          "text_langs_under_budget": len(short), "per_lang_budget": per_lang}
    return docs


def split_merge(x):
    return tuple(x) if isinstance(x, list) else tuple(x.split(" "))


def load_local(path):
    tok = Tokenizer.from_file(path)
    model = json.load(open(path))["model"]
    vocab = tok.get_vocab()
    if model.get("type") != "BPE" or not model.get("merges"):
        return tok, vocab, None, model.get("type", "?") + " (no merges)"
    token2pair, viol = {}, 0
    for mg in model["merges"]:
        a, b = split_merge(mg)
        merged = a + b
        if merged in vocab and a in vocab and b in vocab:
            tid = vocab[merged]
            if tid in token2pair:
                viol += 1
            token2pair[tid] = (vocab[a], vocab[b])
    return tok, vocab, token2pair, f"BPE merge-tree (single-parent violations={viol})"


def analyze(tok, vocab, token2pair, corpus):
    final_freq = Counter()
    for i in range(0, len(corpus), 2000):
        for e in tok.encode_batch(corpus[i:i + 2000]):
            final_freq.update(e.ids)
    # exclude declared special/reserved added tokens (e.g. <bos>, <unused123>) from the
    # utilization denominator -- they are intentional additions, not "unused" learned vocab
    special_ids = set()
    try:
        special_ids = {int(i) for i in tok.get_added_tokens_decoder().keys()}
    except Exception:
        pass
    learned = [v for v in vocab.values() if v not in special_ids]
    util_full = (sum(1 for v in learned if final_freq.get(v, 0) > 0) / len(learned)) if learned else 0.0
    res = {"vocab_size": len(vocab), "n_special_tokens": len(special_ids),
           "util_on_corpus_full_vocab": round(util_full, 4)}

    if token2pair is None:
        res.update({"method": "final-usage only (no merges)", "stepping_applicable": False})
        return res

    @lru_cache(maxsize=None)
    def internal(tid):
        if tid not in token2pair:
            return ()
        a, b = token2pair[tid]
        out = []
        for c in (a, b):
            if c in token2pair:
                out.append(c)
                out += internal(c)
        return tuple(out)

    stepping_freq = Counter()
    for fid, n in final_freq.items():
        for node in internal(fid):
            stepping_freq[node] += n

    merge_ids = list(token2pair.keys())
    n_merge = len(merge_ids)
    total_final = sum(final_freq.values())
    rate_uncommon = UNCOMMON_RATE_PER_M / 1e6   # < this = Uncommon
    rate_active = RARE_RATE_PER_M / 1e6         # >= this = Active; in between = Rare

    def disp(tid):
        return tok.decode([tid])

    _bf = {}
    def bytefrag(tid):  # decode contains U+FFFD => incomplete-UTF-8 sub-character piece
        if tid not in _bf:
            _bf[tid] = "�" in disp(tid)
        return _bf[tid]

    # Partition by standalone rate (Active + Rare + Uncommon + Unseen = 100%); every merge token
    # treated identically -- byte-fragments get NO special handling.
    #   Active   : formed>0 and standalone_rate >= RARE_RATE_PER_M/1e6  (5/M)
    #   Rare     : formed>0 and UNCOMMON_RATE_PER_M/1e6 <= standalone_rate < RARE_RATE_PER_M/1e6
    #   Uncommon : formed>0 and standalone_rate < UNCOMMON_RATE_PER_M/1e6  (1/M)
    #   Unseen   : formed == 0
    # Scaffold (overlaid) = Uncommon or Rare, with survival < SCAFFOLD_SURVIVAL_MAX
    #   (rarely standalone AND mostly a merge step).
    active, rare, uncommon, unseen, scaffold = [], [], [], [], []
    for t in merge_ids:
        fin = final_freq.get(t, 0)
        formed = fin + stepping_freq.get(t, 0)
        if formed == 0:
            unseen.append(t)
            continue
        r = fin / total_final if total_final else 0.0
        if r >= rate_active:
            active.append(t)
        elif r >= rate_uncommon:
            rare.append(t)
        else:
            uncommon.append(t)
        if r < rate_active and fin / formed < SCAFFOLD_SURVIVAL_MAX:   # within Uncommon+Rare, mostly a merge step
            scaffold.append(t)

    built_rates = sorted(final_freq.get(t, 0) / total_final
                         for t in merge_ids if (final_freq.get(t, 0) + stepping_freq.get(t, 0)) > 0)
    med_rate_pm = built_rates[len(built_rates) // 2] * 1e6 if built_rates and total_final else None
    scaffold_bf = [t for t in scaffold if bytefrag(t)]   # context only: byte-fragment share of Scaffold

    child2parents = {}
    for tid, (a, b) in token2pair.items():
        child2parents.setdefault(a, []).append(tid)
        child2parents.setdefault(b, []).append(tid)

    def example(tid):
        par = max(child2parents.get(tid, []), key=lambda p: final_freq.get(p, 0), default=None)
        formed = final_freq.get(tid, 0) + stepping_freq.get(tid, 0)
        return {"token": disp(tid), "final": final_freq.get(tid, 0), "built": formed,
                "survival": round(final_freq.get(tid, 0) / formed, 4) if formed else None,
                "standalone_per_million": round(final_freq.get(tid, 0) / total_final * 1e6, 3) if total_final else None,
                "builds_into": disp(par) if par is not None else None}

    builtcnt = lambda t: final_freq.get(t, 0) + stepping_freq.get(t, 0)
    bf_set = set(scaffold_bf)
    ex_scaffold = sorted([t for t in scaffold if t not in bf_set], key=lambda t: -builtcnt(t))[:10]
    ex_scaffold_bf = sorted(scaffold_bf, key=lambda t: -builtcnt(t))[:5]

    def pct(lst):
        return round(100 * len(lst) / n_merge, 2)

    res.update({
        "method": "BPE merge-tree (rate+survival)",
        "stepping_applicable": True,
        "total_final_tokens": total_final,
        "n_merge_tokens": n_merge,
        "pct_active": pct(active),                       # formed>0, standalone_rate >= 5/M
        "pct_rare": pct(rare),                           # formed>0, 1/M <= standalone_rate < 5/M
        "pct_uncommon": pct(uncommon),                   # formed>0, standalone_rate < 1/M
        "pct_unseen": pct(unseen),                       # formed == 0 (never produced in any role)
        "pct_scaffold": pct(scaffold),                   # (Uncommon or Rare) and survival < 0.1
        "pct_scaffold_bytefrag": pct(scaffold_bf),       # context: % of vocab that is Scaffold AND a byte-fragment
        "median_standalone_per_million": round(med_rate_pm, 3) if med_rate_pm is not None else None,
        "examples_scaffold": [example(t) for t in ex_scaffold],
        "examples_scaffold_bytefrag": [example(t) for t in ex_scaffold_bf],
    })
    return res


def make_out(per_tok, math_bytes, code_bytes):
    return {"nonemitting_tokens": {
        "definition": ("Per merge token (base alphabet excluded): standalone_rate = final emissions / "
                       "all final tokens; survival = final / formed, formed = final + times built as an "
                       "internal merge step. Every merge token (byte-fragments included) is classified "
                       "by the same rule -- no special-casing. Partition by standalone rate (sum to "
                       "100%% of merge tokens): Active = formed>0 & standalone_rate >= %g/M; Rare = "
                       "formed>0 & %g/M <= standalone_rate < %g/M; Uncommon = formed>0 & "
                       "standalone_rate < %g/M; Unseen = formed == 0 (never produced in any role -- "
                       "neither a final token nor a merge step). Scaffold (overlaid, not part of the "
                       "partition) = Uncommon or Rare with survival < %g: rarely standalone AND mostly "
                       "a merge step. Corpus-relative; Scaffold is rarely-exercised embedding capacity, "
                       "not removable waste (structurally required); distinct from 'dead vocab' "
                       "(normalizer-unreachable) and 'junk'."
                       % (RARE_RATE_PER_M, UNCOMMON_RATE_PER_M, RARE_RATE_PER_M, UNCOMMON_RATE_PER_M,
                          SCAFFOLD_SURVIVAL_MAX)),
        "constants": {"UNCOMMON_RATE_PER_M": UNCOMMON_RATE_PER_M, "RARE_RATE_PER_M": RARE_RATE_PER_M,
                      "SCAFFOLD_SURVIVAL_MAX": SCAFFOLD_SURVIVAL_MAX,
                      "math_bytes": math_bytes, "code_bytes": code_bytes},
        "corpus": ("FineWeb sample over the FLORES-200 language set (FineWeb2 per language; "
                   "English from FineWeb-1, which is where English lives) + FineMath-4+ + "
                   "StarCoder (python+javascript). Equal byte budget per language, seed 0."),
        "corpus_stats": getattr(build_corpus, "stats", {}),
        "per_tokenizer": per_tok}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-config", default="configs/sample_tokenizers.json")
    ap.add_argument("--output", default="results/report_nonemitting_tokens.json")
    ap.add_argument("--math-bytes", type=int, default=DEFAULT_MATH_BYTES)
    ap.add_argument("--code-bytes", type=int, default=DEFAULT_CODE_BYTES)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--include-refs", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep tokenizers already in --output and only process the missing ones "
                         "(run repeatedly to finish under memory pressure / OOM)")
    args = ap.parse_args(argv)

    import gc
    cfg = json.load(open(args.tokenizer_config))
    if args.only:
        cfg = {k: v for k, v in cfg.items() if k in args.only}

    per_tok = {}
    if args.resume and os.path.exists(args.output):
        try:
            per_tok = json.load(open(args.output))["nonemitting_tokens"]["per_tokenizer"]
            print(f"resume: {len(per_tok)} tokenizer(s) already done; skipping them", flush=True)
        except Exception:
            per_tok = {}

    corpus = build_corpus(args.math_bytes, args.code_bytes)

    def write(per_tok):
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(make_out(per_tok, args.math_bytes, args.code_bytes),
                  open(args.output, "w"), ensure_ascii=False, indent=2)

    for name, tcfg in cfg.items():
        if name in per_tok:
            continue
        p = tcfg.get("path", "")
        is_local = p.endswith(".json") and os.path.exists(p)
        if not is_local and not args.include_refs:
            continue
        try:
            if is_local:
                tok, vocab, token2pair, method = load_local(p)
            else:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True).backend_tokenizer
                vocab, token2pair, method = tok.get_vocab(), None, "final-usage only (reference)"
            r = analyze(tok, vocab, token2pair, corpus)
            r["method"] = r.get("method", method)
            per_tok[name] = r
            if r.get("stepping_applicable"):
                print(f"  {name:32s} active%={r['pct_active']:.1f} rare%={r['pct_rare']:.1f} "
                      f"uncommon%={r['pct_uncommon']:.1f} unseen%={r['pct_unseen']:.1f} "
                      f"scaffold%={r['pct_scaffold']:.2f}", flush=True)
        except Exception as e:
            per_tok[name] = {"error": str(e)[:200]}
            print(f"  {name:32s} ERROR: {str(e)[:120]}", file=sys.stderr, flush=True)
        write(per_tok)
        gc.collect()

    write(per_tok)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
