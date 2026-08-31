# Tokenizer health checks

`tokenizer-sanity-check` runs 16 checks against one tokenizer or a config of
several and writes `sanity_results.json`. It answers a different question from
`tokenizer-analysis`: not how well a tokenizer compresses a corpus, but whether
it is intact. A tokenizer that loses text, holds vocabulary entries no input can
produce, or encodes the same string two ways will still produce a full set of
metrics, and none of those metrics will say so.

See [../README.md](../README.md) for `tokenizer-analysis` and
`tokenizer-visualize`,
[METRICS.md](METRICS.md) for what `tokenizer-analysis` measures, and
[VISUALIZATION.md](VISUALIZATION.md) for the plotting and LaTeX tools.

## Contents

- [Running it](#running-it)
- [Severities and the overall verdict](#severities-and-the-overall-verdict)
- [Exit codes](#exit-codes)
- [Where the probes come from](#where-the-probes-come-from)
- [The checks](#the-checks)
- [The results file](#the-results-file)
- [Why the bundled demo tokenizers do not pass](#why-the-bundled-demo-tokenizers-do-not-pass)

## Running it

```bash
# one tokenizer, positional CLASS:PATH
uv run tokenizer-sanity-check huggingface:gpt2

# every tokenizer in a config, one combined results file
uv run tokenizer-sanity-check \
    --tokenizer-config benchmarks/open_source/tokenizers.json \
    --output-dir results/
```

| Flag | Default | What it does |
|---|---|---|
| `tokenizer` (positional) | none | `CLASS:PATH` shorthand for a one-off check. Mutually exclusive with `--tokenizer-config`; passing both or neither raises |
| `--tokenizer-config FILE` | none | JSON of `{name: {class, path, ...}}`, the same format `tokenizer-analysis` takes. Every entry is checked and all of them go into one `sanity_results.json` |
| `--only NAME` | none | Restrict a `--tokenizer-config` run to one entry. Raises if the name is not in the config |
| `--output-dir DIR` | none | Where `sanity_results.json` is written. Without it the text report is printed and no file is written |
| `--use-sample-data` | off | Add FLORES probes on top of the built-in ones. Requires `--language-config` |
| `--language-config FILE` | none | The language metadata JSON `--use-sample-data` reads |
| `--probe-samples-per-lang N` | 50 (`SANITY_PROBE_SAMPLES_PER_LANG`) | Cap on FLORES texts per language |
| `--use-builtin-math-data` | off | Add the bundled math corpus as probes. C6 reads them |
| `--quiet` | off | Collapse passing checks in the text report. The results file is unaffected |
| `--exit-zero` | off | Exit 0 whatever the verdict |

## Severities and the overall verdict

Five severities, defined at `tokenizer_analysis/diagnostics/sanity_check.py:87-92`:

| Severity | Meaning |
|---|---|
| `pass` | The check ran and the tokenizer met its condition |
| `warn` | The check ran and found something that is a defect in some tokenizers and a design choice in others |
| `fail` | The check ran and found something no correct tokenizer does |
| `not_applicable` | The check does not apply. A non-byte-level tokenizer cannot fail a byte-coverage check |
| `unverifiable` | The check could not run, because the tokenizer does not expose what it needs |

The overall verdict is the worst of the 16, under the rank map at lines 94-96:
`pass` and `not_applicable` rank 0, `warn` and `unverifiable` rank 1, `fail`
ranks 2. Two consequences follow from that map. An `unverifiable` check forces
the overall verdict to at least `warn`, so a check that could not run is never
reported as one that passed. An `unverifiable` check can never on its own
produce `fail`, because nothing was measured to fail.

`summary.<tok>.n_warn` counts `warn` and `unverifiable` together, matching the
rank map (lines 1489-1490).

## Exit codes

| Code | Condition |
|---|---|
| 0 | Every tokenizer's overall verdict is `pass` |
| 1 | The worst overall verdict is `warn` |
| 2 | At least one tokenizer's overall verdict is `fail` |
| 3 | The tool raised before producing a report |

`--exit-zero` returns 0 in all four cases. It is needed whenever the command
runs inside a script under `set -e`, which is why
`benchmarks/open_source/run.sh` passes it: real tokenizers produce warnings, and
a warning is information rather than a reason to abandon the run.

## Where the probes come from

The built-in probes are always used, with no flag and no network, from
`tokenizer_analysis/diagnostics/probe_corpus.py`. There are 78 of them:

| Category | Count | What they hold |
|---|---:|---|
| `ascii_basic` | 4 | Plain ASCII. C3 treats a bug on any of these as a hard failure |
| `whitespace` | 12 | Double space, tab, newline, CRLF, leading and trailing spaces, blank lines, mixed indentation, NBSP, thin space, ideographic space, zero-width space |
| `digits` | 14 | Single digits through `1234567890`, `3.14159`, `1,000,000`, `0xDEADBEEF`, superscripts, Arabic-Indic and Devanagari digits |
| `combining_marks` | 8 | Base characters with combining marks |
| `nfc_nfd_pairs` | 14 | The 7 pairs of `NFC_NFD_PAIRS`, each in both forms |
| `casing` | 6 | Mixed and upper case |
| `control_chars` | 6 | Embedded NUL, BEL, ESC, DEL, NEL, leading BOM |
| `emoji_zwj` | 5 | A single emoji, a skin-tone modifier, a ZWJ sequence, a regional-indicator flag, a keycap sequence |
| `multiscript` | 9 | Latin, Cyrillic, Arabic, Han, Devanagari, Hangul, Hebrew, Greek, Thai |

Two additions are opt-in. `--use-sample-data` with `--language-config` adds up
to `--probe-samples-per-lang` FLORES texts per language, tagged `flores` and
noted with the language code, which is what makes C13 report per-language rather
than per-script. `--use-builtin-math-data` adds the bundled math corpus, tagged
`math`, which C6 reads alongside the digit probes. Both raise if they load
nothing, rather than continuing with an empty set.

Three checks do not read the probe list. C4 and C11 iterate `NFC_NFD_PAIRS`
directly. C16 uses `SANITY_CROSS_BOUNDARY_PROBE` from `constants.py` and five
one-character breaker strings of its own.

## The checks

There is no C9. Each entry below is read from the check's implementation. Every
threshold is a named constant in `tokenizer_analysis/constants.py:145-211`, and
all of them are echoed into `metadata.thresholds` of the results file, so a
report can be read against the values that produced it.

Nine checks can report `fail`: C1, C2, C3, C4, C7, C8, C10, C14 and C16. The
other seven cannot, and each entry says so.

### C1 byte-level 256-coverage

`check_byte_coverage`, lines 423-489. Category `behavioral` when a decoder is
available, `static` otherwise.

Whether a byte-level tokenizer can carry all 256 byte values without loss. With
a decoder, it encodes and decodes each of the 256 single-byte latin-1 strings
and compares against the input. That is the authoritative form: a byte with no
standalone vocabulary entry can still round-trip through a multi-token fallback.
Without a decoder it falls back to asking whether each byte is present as a
standalone vocabulary key, which is the weaker question that C17 asks in full.

- `not_applicable`: the tokenizer is not byte-level.
- `fail`: any byte fails the round trip, or, on the static fallback, any byte is
  absent from the vocabulary.
- `pass`: all 256 survive.

### C17 strict byte-alphabet vocab presence

`check_byte_alphabet_strict`, lines 502-546. Category `static`. Vocabulary only.

The strict form of C1: whether all 256 bytes are present as their own
single-token vocabulary entries, rather than merely reachable through a
fallback. The detail splits the missing bytes into valid UTF-8 lead bytes
(`0xC2` to `0xF4`) and the rest, because a missing lead byte affects text in the
supplementary Unicode planes.

- `not_applicable`: the tokenizer is not byte-level.
- `warn`: more than `SANITY_STRICT_BYTE_ALPHABET_WARN_COUNT` (0) bytes are
  missing, so any missing byte warns.
- No `fail`. Round-tripping still works through the fallback, which is what C1
  measures.

### C2 combining-mark mishandling

`check_combining_marks`, lines 552-598. Category `static`. Vocabulary only.

Counts vocabulary tokens that are made of nothing but combining marks, and the
fraction of tokens whose first character is a combining mark. A token that is
only a combining mark cannot be rendered on its own and indicates a vocabulary
built without regard for grapheme boundaries.

- `not_applicable`: no vocabulary entry could be examined.
- `fail`: any token is entirely combining marks, whatever the fraction. Failing
  that, the leading fraction is at or above
  `SANITY_MARK_LEADING_TOKEN_FAIL_FRAC` (0.02).
- `warn`: the leading fraction is at or above
  `SANITY_MARK_LEADING_TOKEN_WARN_FRAC` (0.005).
- `pass`: below both.

### C3 lossy-text root-cause

`check_roundtrip`, lines 691-715. Category `behavioral`. Reads every probe.

Encodes and decodes each probe and sorts the outcome into a bucket, which is
what separates loss the tokenizer's own normalizer causes from loss that is a
defect. The buckets are `clean`, `lossy_expected` (the output matches the
tokenizer's own normalized form of the input), `normalization_loss`,
`casing_loss_expected`, `unk_loss`, `byte_bug`, `casing_loss_bug`,
`merge_or_decode_bug`, and `lossy_unverifiable` for cases where the normalizer
cannot be introspected. The last four before `lossy_unverifiable` are the red
flags. `clean_frac` sums the first four buckets, `bug_frac` sums the red flags.

- `fail`: any `ascii_basic` probe lands in a red-flag bucket, whatever the
  fractions, or `bug_frac` is at or above `SANITY_ROUNDTRIP_BUG_FAIL_FRAC`
  (0.01).
- `warn`: `bug_frac` is above `SANITY_ROUNDTRIP_BUG_WARN_FRAC` (0.0), so any bug
  at all, or `clean_frac` is below `SANITY_ROUNDTRIP_CLEAN_PASS_FRAC` (1.0).
- `unverifiable`: no bug, but some probe could not be classified.
- `pass`: everything clean and everything classifiable.

### C4 faithful-pipeline conformance

`check_faithful_pipeline`, lines 721-752. Category `static` when unverifiable,
`behavioral` otherwise. Reads `NFC_NFD_PAIRS`.

Whether `encode()` applies the normalizer the tokenizer declares. For each pair
whose two forms the declared normalizer maps to the same string, `encode()` must
return the same ids for both. A difference means the encode path bypasses the
declared normalizer, so every metric computed through `encode()` describes a
different pipeline from the one the tokenizer advertises.

- `unverifiable`: the normalizer cannot be introspected. SentencePiece and the
  script_bpe wrappers land here.
- `fail`: any pair the normalizer unifies encodes to different ids.
- `pass`: no such pair.

### C5 whitespace handling

`check_whitespace`, lines 758-811. Category `behavioral`. Reads the 12
`whitespace` probes.

Whitespace round-trip fidelity, as the fraction of whitespace characters
preserved through encode and decode. The share of the vocabulary that is
whitespace-only is reported beside it and is not scored.

- `pass`: fidelity at `SANITY_WHITESPACE_FIDELITY_PASS_FRAC` (1.0), or no
  whitespace probe was measurable.
- `warn`: fidelity below 1.0.
- No `fail`, by design. WordPiece, SentencePiece and Metaspace tokenizers
  discard whitespace deliberately, so a hard failure here would report a design
  choice as a defect.

### C6 digit handling

`check_digits`, lines 817-904. Category `behavioral`. Reads the `digits` probes
and, with `--use-builtin-math-data`, the `math` probes.

How consistently the tokenizer splits numbers. It resolves each digit span to
tokens through character offsets, records the boundary pattern inside the span,
and computes `1 - H/log2(distinct patterns)` over the observed patterns. The
reported `chunking_direction` says whether the chunking is right-aligned,
left-aligned, neither, or mixed. Pure-digit vocabulary share and longest digit
run are reported and not scored.

- `not_applicable`: no digit span could be measured, which happens when the
  wrapper returns no character offsets.
- `pass`: consistency at or above `SANITY_DIGIT_CONSISTENCY_PASS` (0.99).
- `warn`: below it.
- No `fail`.

### C7 special-token sanity

`check_special_tokens`, lines 922-973. Category `static`. Vocabulary and the
tokenizer's declared special ids. No probes.

Reads the declared BOS, EOS, PAD and UNK ids and checks three things: that they
are distinct, that each is inside `[0, vocab_size)`, and that re-encoding a
special token's surface string returns exactly that one id.

- `fail`: an id is out of range, or a special token's surface does not re-encode
  to itself as a single token.
- `warn`: two special ids are the same, which is often deliberate (GPT-NeoX uses
  one id for BOS, EOS and UNK; Qwen shares EOS and PAD), or the tokenizer is not
  byte-level and has no UNK token, which leaves it with no way to represent
  characters it cannot encode.
- `pass`: none of the above.

### C8 determinism/idempotency

`check_determinism`, lines 979-1003. Category `behavioral`. Reads the first 50
probes.

Encodes each probe twice and compares, then compares a batch encode against a
per-text loop over the same texts. Non-determinism makes every other number in
every other tool irreproducible.

- `fail`: encoding the same text twice gives different ids.
- `warn`: the batch path and the per-text path disagree.
- `pass`: neither.

### C10 pretokenizer char conservation

`check_pretok_conservation`, lines 1009-1079. Category `behavioral`. Reads every
probe except the `control_chars` ones, which a normalizer may legitimately drop.

The fraction of non-whitespace source characters covered by the spans the
tokenizer's own pre-tokenizer reports. A pre-tokenizer that drops characters
drops them before the model ever sees them.

- `not_applicable`: the tokenizer exposes no pre-tokenizer.
- `unverifiable`: it exposes one, but no probe produced a measurable span.
- `fail`: coverage below `SANITY_PRETOK_CONSERVATION_FAIL_FRAC` (0.999).
- `pass`: at or above it.
- No `warn`.

A `pass` here is not proof that nothing was dropped. The check reads spans, and
a pre-tokenizer that drops a character between two it keeps can report one span
across all three, which leaves no gap to find. Measured on the SCRIPT BPE test
fixture, whose script configuration drops unassigned code points: `a￰b`
produces a single chunk spanning positions 0 to 3, so the dropped character at
position 1 reads as covered, and two probes that each lose one character give
`pass` at conservation 1.000000. Whether the loss is visible depends on whether
a merged chunk reaches across it. The exposure is private-use, unassigned,
noncharacter and surrogate code points, not ordinary text: none of the 78
built-in probes contains a character this configuration drops. An uncovered
character is evidence of loss; a covered one is not evidence against it.

### C11 NFC/NFD roundtrip

`check_nfc_nfd`, lines 1085-1107. Category `behavioral`. Reads `NFC_NFD_PAIRS`.

Runs both forms of each pair through the C3 classifier and collects any that
land in a red-flag bucket. Whether the two forms encode identically is reported
and not scored, because a tokenizer may legitimately keep them distinct.

- `warn`: any form lands in a red-flag bucket.
- `pass`: none does.
- No `fail`.

### C12 emoji/ZWJ/control

`check_emoji_control`, lines 1113-1130. Category `behavioral`. Reads the
`emoji_zwj` and `control_chars` probes.

Runs them through the C3 classifier and groups any red-flag results by bucket.

- `warn`: any probe lands in a red-flag bucket.
- `pass`: none does.
- No `fail`.

### C13 UNK-per-script

`check_unk_per_script`, lines 1136-1163. Category `behavioral`. Reads the
`multiscript` probes and, with `--use-sample-data`, the `flores` probes.

The share of tokens that are UNK, grouped by script or language. A script above
the threshold is undertrained rather than unsupported: the tokenizer loads and
runs, and the text it cannot represent is silently replaced.

- `not_applicable`: the tokenizer has no UNK token.
- `warn`: any group's UNK rate exceeds `SANITY_UNK_SCRIPT_WARN_RATE` (0.01).
- `pass`: no group does.
- No `fail`.

### C14 vocab integrity

`check_vocab_integrity`, lines 1169-1192. Category `static`. Vocabulary only.

Three structural properties: `len(get_vocab())` equals `get_vocab_size()`, no id
appears twice, and the ids are contiguous from 0.

- `fail`: the two sizes disagree, or an id is duplicated.
- `warn`: the ids are not contiguous from 0, and nothing already failed. A gap
  leaves an embedding row no token maps to.
- `pass`: none of the above.

### C15 token-length outliers

`check_token_outliers`, lines 1198-1232. Category `static`. Vocabulary only.

Vocabulary tokens longer than `SANITY_MAX_REASONABLE_TOKEN_CHARS` (64)
characters after marker stripping.

- `warn`: any such token.
- `pass`: none.
- No `fail`, by design. A long token can be a legitimate code or URL fragment,
  and it can equally be a scraped artifact; the check reports the count and
  leaves the reading to whoever knows the training data.

### C16 vocab reachability

`check_vocab_reachability`, lines 1282-1422. Category `behavioral`. Vocabulary,
plus its own probe strings.

For every non-special vocabulary token, whether any input can produce it. Each
token goes into one bucket:

| Bucket | Meaning |
|---|---|
| `self_reproducing` | Decoding the token and re-encoding gives its own id back |
| `context_only` | Not reachable standalone, but produced when embedded in a longer string, or its bytes are not valid UTF-8 on their own |
| `non_self_reproducing` | Not self-reproducing, with no introspectable normalizer and no pre-tokenizer split to explain it |
| `normalization_unreachable` | The declared normalizer folds the token's surface to something else, so no input can produce it |
| `pretokenizer_unreachable` | The pre-tokenizer splits the surface and no embedded context recovers it |
| `unverifiable` | The token could not be decoded, or the normalizer raised on it |

Tokenizers that merge across pre-tokenizer boundaries, such as SuperBPE, are
detected with `SANITY_CROSS_BOUNDARY_PROBE` and exempted from the
pretokenizer-unreachable count, because their whitespace-internal tokens are
reachable by design.

- `fail`: more than `SANITY_VOCAB_NORMALIZATION_DEAD_FAIL_COUNT` (0) tokens are
  normalization-unreachable. The normalizer guarantees no input produces them,
  which means the vocabulary was built without applying it.
- `warn`: more than `SANITY_VOCAB_UNREACHABLE_WARN_COUNT` (0) tokens are
  pretokenizer-unreachable. The slot is unusable capacity, but no input reaches
  it, so it cannot corrupt text or produce an UNK.
- `unverifiable`: neither, and some token could not be classified.
- `pass`: none of the above.

Measured on bert-base-uncased, 6823 of its vocabulary tokens are
pretokenizer-unreachable.

## The results file

`--output-dir` writes one `sanity_results.json` covering every tokenizer in the
run:

```
tokenizer_sanity_check
├── per_tokenizer.<tok>
│   ├── overall_severity                       # pass | warn | fail
│   ├── checks.<name>                          # 16 entries, keyed by full name
│   │   ├── name, category, severity
│   │   ├── observed, threshold
│   │   └── detail, rationale, examples        # examples capped at 20
│   ├── lossy_breakdown                        # C3 bucket counts
│   ├── vocab_reachability                     # C16 bucket counts
│   ├── vocab_composition                      # vocab_size, byte_style, n_special_tokens
│   └── components                             # normalizer, pretokenizer, decoder
├── summary.<tok>                              # overall_severity, n_fail, n_warn
└── metadata
    ├── description
    ├── thresholds                             # every SANITY_* constant
    └── components.<tok>
```

The `checks` keys are the full name strings, `C7 special-token sanity` rather
than `C7`, so a consumer selecting one check has to match on the whole string or
on its id prefix. The severity of one check is at
`tokenizer_sanity_check.per_tokenizer.<tok>.checks.<name>.severity`.

Running the same command twice on the same tokenizers writes the same file. The
checker sorts the vocabulary by token string before reading it, because
`tokenizers.Tokenizer.get_vocab()` builds its dict from a Rust HashMap that is
seeded per process, and C2, C15 and C16 draw their `examples` from that order.
Where a check finds more than 20 examples, the 20 published are the first in
that sorted order rather than an arbitrary 20.

`observed` and `threshold` are whatever shape the check reports: a count, a
fraction, a bucket dict, or a string such as `introspectable` where the
condition is not numeric.

## Why the bundled demo tokenizers do not pass

`tokenizers/bpe.json` and `tokenizers/unigramlm.json` are small tokenizers
trained on a short corpus for the Quick Start, and both come out `fail`. Nothing
in the checker special-cases them; the results are real measurements of small
tokenizers. `bpe` fails C1 and warns on C5, C6, C15 and C16.

C6 is the one the code records a number for: the comment at lines 870-876 gives
`bpe.json` a digit consistency of 0.3769 against the 0.99 threshold, and notes
that the same tokenizer reports 1.0000 once its character offsets are removed,
because with no offsets there is no digit span to measure and the check goes
`not_applicable`. A tokenizer that exposes less about itself scores no worse
here, which is the reason `not_applicable` and `unverifiable` are distinct
severities rather than both counting as a pass.
