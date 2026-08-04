# Metrics

Definitions for every metric TokEval computes. For the JSON key each metric is
written under, and the path to its headline value, see
[Metric names and results keys](README.md#metric-names-and-results-keys) in the
README. For which flag supplies the data a metric needs, see
[Metric families and the data they need](README.md#metric-families-and-the-data-they-need).

Section headings below give the top-level results key in backticks. Where a
value sits deeper in the file, the path is written out.

## Basic tokenization metrics

- **Compression rate** (`compression_rate`): total text units (bytes, characters
  or lines) divided by total tokens across the corpus.
- **Fertility** (`fertility`): tokens per word or per character.
- **Token length** (`token_length`): mean token size in bytes and in characters.
  This is a mean of per-document ratios, where `compression_rate` is a ratio of
  totals, so the two differ on short documents.
- **Type-token ratio**: unique tokens divided by total tokens. Written under
  `vocabulary_utilization.per_tokenizer.<tok>.type_token_ratio`, because it is
  that metric rescaled by vocabulary size over token count (see
  [Metrics reported under another metric](README.md#metrics-reported-under-another-metric)).
- **Vocabulary utilization** (`vocabulary_utilization`): the fraction of the
  declared vocabulary that the corpus used.

**Compression rate and measurement method.** Under the `lines` measurement
method, `compression_rate` and `avg_tokens_per_line` (reported under
`compression_rate.per_tokenizer.<tok>.tokens_per_line`) are exact reciprocals:
their product is 1.000000 for every tokenizer. Under every measurement method the
two correlate at Spearman -1.000, because a higher token count always lowers
`compression_rate` and raises `avg_tokens_per_line`, whichever unit
`compression_rate` divides by.

Within a single run, `compression_rate` computed under the `characters`,
`bytes` and `lines` methods correlates at exactly 1.0000 between every pair.
The numerator (the character, byte or line count of the corpus) is fixed once
the corpus is fixed and does not depend on the tokenizer; only the token count
in the denominator varies by tokenizer. Changing `--measurement-config`
therefore rescales `compression_rate` but does not change the ranking of
tokenizers within a run.

**Fertility and whitespace word counting.** The default word counting splits
text on whitespace. On FLORES+, 96.2% of Japanese lines and 69.3% of Chinese
lines hold exactly one whitespace-delimited token, so for those languages
`fertility` measures tokens per sentence rather than tokens per word. Median
per-language fertility runs from 1.261 (eng) to 38.195 (jpn), and the pooled
value across the FLORES+ languages correlates with fertility over the other 11
languages at Spearman 0.401. A pooled `fertility` computed over a corpus
containing CJK text reflects the language mix as much as the tokenizer.

**Vocabulary utilization and unreachable tokens.** The denominator is the
declared vocabulary size, which includes special and added tokens that
encoding with `add_special_tokens=False` can never emit. The unreachable count
is tokenizer-dependent: 1000 of 131072 vocabulary entries for one tokenizer in
a 37-tokenizer run, 4 for most of the others. For a tokenizer with a large
unreachable count, the maximum `vocabulary_utilization` can reach is capped
below 1.0 by that margin.

### Encoding speed (`encoding_speed`)

Wall-clock time spent encoding, per tokenizer:
`encoding_speed.per_tokenizer.<tok>` holds `mean_ms` (mean milliseconds per
sample), `total_s` (total seconds) and `num_samples`, plus a `global` block that
repeats the same three fields. It measures the run, not the tokenizer's quality,
so it has no per-language breakdown. Use it to size a larger run from a small
one.

## Information-theoretic metrics

### Rényi efficiency (`renyi_efficiency`)

`H_alpha / log2(|V|)` following
[Zouhar et al. 2023](https://aclanthology.org/2023.acl-long.284/), with `|V|`
the declared vocabulary size. Computed at alphas 1.0, 2.0, 2.5 and 3.0. The
pre-1.0 normalization, which divided by the number of token types observed in
the corpus, is still written under the top-level
`renyi_efficiency.observed_normalization`; the two rank tokenizers at Spearman
0.678 over 37 tokenizers, so they are not interchangeable.

**Average token rank** is written under
`renyi_efficiency.per_tokenizer.<tok>.unigram_distribution.global_avg_token_rank`:
the mean position of the corpus tokens within the frequency-ordered vocabulary.
It correlates with total token count at Spearman -0.977 across a 37-tokenizer
run and has no scale-free definition, so comparing it across runs of different
size compares the corpus sizes rather than the tokenizers.

The same block holds `global_unigram_entropy`. `renyi_1.0` multiplied by
`log2` of the declared vocabulary size reproduces `global_unigram_entropy` to
zero relative error over the same 37 tokenizers: `global_unigram_entropy` is
the unnormalized numerator of `renyi_1.0`, not merely correlated with it.

### Bigram entropy (`bigram_entropy`)

For each token type, the metric takes the distribution of tokens that follow it
in the corpus and measures how even that distribution is. A score of 1.0 means
every token type's successors are uniformly distributed; a score near 0 means
most token types are almost always followed by the same token. Token types
occurring fewer than 3 times (`min_bigram_occurrences`, configurable) are
excluded, because the estimate is noisy below that. Bigrams do not cross
document boundaries.

Based on the Shannon efficiency metric (η) of
[Poelman et al. 2025](https://aclanthology.org/2025.emnlp-main.369/), EMNLP.
This implementation deviates from the reference in four ways, listed in
`bigram_entropy.metadata.deviations_from_reference`. The most consequential is
the normalizer: this metric divides by the context's own successor count, the
reference divides by the corpus-wide accessor-domain size. The reference
normalizer and the reference unweighted aggregation are computed alongside and
written under `bigram_entropy.reference_definition`.

### Trigram entropy (`trigram_entropy`)

The same construction with a two-token context: for each bigram, the evenness of
the distribution of tokens that follow it. Poelman et al. define only the bigram
form, so there is no published value to compare this against. Token contexts
occurring fewer than 3 times (`min_trigram_occurrences`) are excluded.

The value sits in `trigram_entropy.per_tokenizer.<tok>.global`, which holds
`trigram_entropy`, `total_trigrams`, `types_evaluated` and `types_excluded`.

The default threshold discards 89 to 90 percent of context types and a median 70
percent of occurrences, and the ranking moves with it: Spearman 0.728 between
`min_trigram_occurrences` 3 and 25 over 37 tokenizers, against 0.985 for bigram
entropy at the same settings. Treat a trigram ranking as threshold-dependent.

## Morphological metrics

**MorphScore V2** (`morphscore_recall`, `morphscore_precision`): morphological
evaluation following [Arnett et al. 2025](https://arxiv.org/abs/2507.06378).
Enable with `--morphscore` or `--morphscore-config`. Requires raw tokenization
and the MorphScore submodule.

## Mathematical content metrics

Tokenizer handling of mathematical expressions, following Singh and Strouse
(2024, [arXiv:2402.14903](https://arxiv.org/abs/2402.14903)), who measured a
gain of more than 22 percentage points in arithmetic accuracy from right-to-left
tokenization of numbers. Disable with `--no-digit-boundary`.

> **Data scope:** the four metrics in this group do not all take their data from
> the same place, and the fallback applies to three of them only.
>
> The three digit metrics (`three_digit_boundary_alignment`, digit split
> variability, `numeric_magnitude_consistency`) use the dedicated math texts when
> `--math-data FILE` or `--use-builtin-math-data` is given, and write every
> per-language result in this group under the synthetic language key `math`.
> Without either flag they fall back to whatever numbers appear in the main
> corpus, and the run prints a warning naming all three. On the bundled
> five-language FLORES+ corpus at the default `--samples-per-lang`, the fallback
> gives 1797 digit spans: 385 of one digit, 600 of two, 355 of three, 451 of
> four, 4 of five and 2 of six. The place-value boundaries at positions L-6 and
> L-9 are therefore almost never exercised.
>
> `operator_isolation_rate` does not fall back. Its `math` domain always reads
> dedicated math texts, the bundled
> `tokenizer_analysis/sample_data/math_samples.json` when no flag names another
> file, and its `code` domain always reads code snippets, the bundled
> `sample_data/code_samples.json` when no `--code-ast-config` names a corpus.
> Only its `prose` domain comes from the main corpus. Each run logs the three
> sources on one line:
> `Operator isolation domains: prose=multilingual, math=..., code=...`.
>
> `--no-code-ast` drops the three AST metrics but leaves this metric's `code`
> domain running on the bundled samples. `--no-digit-boundary` drops
> `operator_isolation_rate` along with the three digit metrics.
>
> See [the full evaluation command](README.md#full-evaluation) for the invocation
> that supplies math data.

### Three-digit place-value boundary alignment (`three_digit_boundary_alignment`)

Whether numbers are tokenized with right-aligned 3-digit groupings matching
place-value structure (units, thousands, millions).

For each number of L digits, the ideal internal boundaries are at positions
L-3, L-6, L-9 counting from the left. The metric compares the actual internal
boundaries against those and reports precision, recall and F1. A number of 3
digits or fewer has no ideal boundary: kept as one token it scores F1 = 1.0,
split it scores F1 = 0.

**Example:** `1234567` has 7 digits, so the ideal boundaries are at positions 1
and 4, giving `1|234|567`. A tokenizer producing `1|234|567` places boundaries
at 1 and 4 and scores F1 = 1.0. A tokenizer producing `12|345|67` places two
internal boundaries, at positions 2 and 5, neither of which is ideal, so
precision, recall and F1 are all 0.0. `42` kept as one token scores F1 = 1.0:
no boundary was needed and none was placed. `42` split into `4|2` scores
precision 0.0 and F1 = 0.0, because a boundary was placed where none was needed.

Singh and Strouse showed that right-to-left digit grouping improves arithmetic
accuracy because corresponding digit positions across two operands then occupy
the same token positions.

### Digit split variability

Written under
`three_digit_boundary_alignment.per_tokenizer.<tok>.split_variability`. For
numbers of the same digit length, the Shannon entropy of the distribution of
boundary patterns. Low entropy means one splitting scheme is used consistently;
high entropy means the scheme varies from number to number. Entropy is computed
on patterns pooled across languages, not averaged per language. Reports entropy
in bits, the dominant pattern and its frequency, per digit-length bucket
(`by_digit_length`) and per short/long bucket (`by_bucket`).

**Example:** a corpus holds three 5-digit numbers. If all three are split
`XX|XXX`, that bucket has one pattern and entropy 0.0 bits. If they are split
`XX|XXX`, `X|XXXX` and `XXX|XX`, that bucket has three patterns at equal
frequency and entropy log2(3) = 1.585 bits. The first case is a single
scheme, correct or not. In the second, numbers of the same length receive
different segmentations.

A tokenizer with moderate F1 and low entropy applies one scheme that does not
match place value, which retraining can change. Moderate F1 with high entropy
means no single scheme is being applied.

### Numeric magnitude consistency (`numeric_magnitude_consistency`)

How tokens-per-digit varies across digit lengths. Everything sits under
`numeric_magnitude_consistency.per_tokenizer.<tok>.scaling`:

- `per_bucket.<digits>` holds `mean_fertility` (mean tokens per digit),
  `std_fertility`, `count` and `mean_digit_length` for that bucket.
- `spearman_rho` and `spearman_p`: rank correlation between bucket digit length
  and that bucket's **mean fertility per digit**.
- `cv_of_mean_fertility`: coefficient of variation of the same per-bucket
  **mean fertility per digit** values.
- `linear_fit`: `slope`, `intercept` and `r_squared` of a least-squares fit of
  each bucket's **mean token count** on that bucket's mean digit length. The fit
  is the only field computed on token counts rather than on fertility per digit.

The fit rests on at most 10 points (buckets `1` through `9` plus `10+`), so
`slope` and `r_squared` are coarse.

**Example:** a tokenizer has `0` to `999` as single vocabulary entries, so
1-digit numbers cost 1 token (1.0 tokens per digit), 2-digit numbers 1 token
(0.5), 3-digit numbers 1 token (0.333). At 4 digits it splits: `1234` becomes
`12|34`, 2 tokens (0.5). At 7 digits: `1234567` becomes `123|45|67`, 3 tokens
(0.429). Fertility per digit therefore rises from 0.333 to 0.5 between 3 and 4
digits after falling over 1 to 3 digits, so `spearman_rho` is -0.616 and
`cv_of_mean_fertility` is 0.420. The linear fit is computed on token counts
(1, 1, 1, 2, 3), which do rise with digit length, so it returns slope 0.368 and
R-squared 0.897. Read the two together: a high `r_squared` does not rule out the
discontinuity in tokens per digit that `cv_of_mean_fertility` reports.

Tokenizers trained on natural language often have dense vocabulary coverage for
small numbers and split larger numbers in ways that vary with the number, which
is what the per-bucket fertility values show.

### Operator isolation rate (`operator_isolation_rate`)

The fraction of mathematical operators tokenized as standalone tokens rather
than merged with adjacent content. Operators are located in the source text by
`_OPERATOR_SPAN` (`tokenizer_analysis/metrics/math.py`), which matches the
two-character forms `**`, `//`, `<<`, `>>`, `<=`, `>=`, `=>`, `==`, `!=`, `&&`,
`||`, `?:` before the single characters `+ - * / = < > ! & | ^ ~ %`. The
hyphen-minus `-` is always counted as an operator, including as a unary negative
sign such as `-42`, because telling the two apart requires parsing the
expression. A compound preservation sub-metric records whether a multi-character
operator (`**`, `<=`, `!=`) stayed in one token.

An operator counts as isolated when the tokens covering it hold no
non-whitespace character from outside the operator span. Leading or trailing
whitespace inside the same token does not break isolation, because a space is
not an operand.

**Example:** in `3+5>=8` the regex matches two operators, `+` and `>=`, so the
denominator is 2. A tokenizer producing `3` | `+` | `5` | `>=` | `8` isolates
both, isolation rate 2/2 = 1.0, and keeps `>=` in one token, compound
preservation 1/1 = 1.0. A tokenizer producing `3+` | `5` | `>` | `=` | `8`
merges `+` with the operand `3`, so `+` is not isolated, while `>=` is covered
only by tokens `>` and `=`, which hold nothing else, so `>=` is isolated:
isolation rate 1/2 = 0.5. Compound preservation is 0/1 = 0.0, because `>=` took
two tokens.

Merging an operator with its operand puts the operation and the value in one
embedding.

**How the global is computed:** operators are counted in three domains, prose,
code and math, and `global` pools all three, weighted by operator instances.
On the bundled demo,

```bash
uv run tokenizer-analysis --use-sample-data
```

`bpe` scores 0.7938 pooled over 3016 instances, against 0.6832 for code over
1932 instances, 0.9886 for prose over 787 and 0.9966 for math over 297. The
three domain rates are far apart, and the pooled figure is not close to any of
them: code supplies 64 percent of the instances but the pooled rate sits 0.11
above the code rate, because prose and math are both near 1.0. Quoting `global`
alone therefore reports a number that describes no domain.

The weights also move with the flags. `--samples-per-lang` changes the prose
corpus while the code and math corpora stay fixed at 1932 and 297 instances, so
the same tokenizer on the same demo scores 0.7285 pooled at
`--samples-per-lang 20` (prose 1.0 over 29 instances) and 0.7938 at the default
2000 (prose 0.9886 over 787). A `--code-ast-config` corpus changes the code
domain, and `--math-data` changes the math domain.

`by_domain` holds the three rates and the three instance counts separately, and
is written beside `global` for that reason. Read it before quoting the pooled
number.

## Reconstruction fidelity metrics (`reconstruction_fidelity`)

How lossy the encode-then-decode round trip is. Information is lost through
normalization, UNK substitution, whitespace handling and decode asymmetry. These
metrics run on language text, code and math data. They require that the
tokenizer supports decoding; for one that does not, the run logs
`Reconstruction fidelity: skipping <name> (no decode support)` and the tokenizer
is absent from this metric's results.

All four values below are fields of
`reconstruction_fidelity.per_tokenizer.<tok>.global`, with the same fields
repeated per corpus under `per_domain` (this metric uses `per_domain`, not
`per_language`).

### Round-trip exact match rate (`exact_match_rate`)

The fraction of texts for which `decode(encode(text)) == text`. 1.0 means the
tokenizer is lossless on the evaluated data.

**Example:** `"Hello, world!"` encodes to `[15496, 11, 995, 0]` and decodes back
to `"Hello, world!"`, an exact match. `"café"` that decodes to `"cafe"`, the
accent removed by normalization, is not an exact match.

### Character error rate (`mean_cer`)

Levenshtein edit distance between the original text and the decoded text,
divided by the length of the original: the fraction of single-character
insertions, deletions and substitutions needed to turn the decoded text back
into the original.

CER = 0 means a perfect round trip. CER can exceed 1.0 when the decoded text is
much longer than the original, for example when a tokenizer expands
byte-fallback tokens into multi-character escape sequences.

**Example:** original `"hello"` decoded as `"helo"` is edit distance 1 over 5
characters, CER 0.2. Original `"a"` decoded as `"abcd"` is edit distance 3 over
1 character, CER 3.0.

CER is the most expensive field in this group, and `--cer-time-budget` caps it
per tokenizer. When the budget is exceeded, `mean_cer` and `whitespace_fidelity`
are `null` for that tokenizer and the run logs the projection that triggered the
skip. `--cer-time-budget 0` disables the cap.

Which tokenizers report `mean_cer` therefore depends on how fast the machine
running the analysis is: a tokenizer skipped at a given `--cer-time-budget` on a
slow machine can complete at the same budget on a faster one.
`per_tokenizer.<tok>.cer_skipped` is `true` when the budget was exceeded for
that tokenizer, distinguishing a skipped value from a measured one.
`exact_match_rate` does not run the Levenshtein computation and is unaffected.

### UNK token rate (`unk_token_rate`)

The fraction of encoded tokens equal to the tokenizer's UNK token id: how much
of the input the tokenizer has no representation for. 0.0 means either no
unknown tokens were produced or the tokenizer reports no UNK token id at all,
in which case the count of UNK tokens is zero by construction rather than by
measurement. The two cases are not distinguishable from `unk_token_rate` alone.

**Example:** encoding `"𝕳𝖊𝖑𝖑𝖔"` to `[UNK, UNK, UNK, UNK, UNK]` gives UNK rate
1.0. Encoding `"Hello"` to `[15496]` gives 0.0.

### Whitespace fidelity (`whitespace_fidelity`)

The fraction of whitespace characters (spaces, tabs, newlines, plus the Unicode
Zs category) in the original text preserved through the round trip. Characters
are paired by a greedy forward scan. 1.0 means either every whitespace
character round-tripped or the evaluated text held no whitespace at all, in
which case fidelity is 1.0 by convention rather than by measurement. The two
cases are not distinguishable from `whitespace_fidelity` alone.

**Example:** original `"a b\tc"` decoded as `"a b c"`, the tab replaced by a
space, preserves 1 of 2 whitespace characters, fidelity 0.5.

## UTF-8 character boundary metrics (`utf8_token_integrity`)

How byte-level tokenizers handle multi-byte UTF-8 characters at token
boundaries. Runs on any text data with no extra configuration. Disable with
`--no-utf8-integrity`.

### Token UTF-8 completeness rate

`utf8_token_integrity.per_tokenizer.<tok>.global.completeness_rate`. The
fraction of content tokens whose bytes form complete UTF-8 characters. A token
such as `<0xC3>`, one byte of the two-byte sequence for `é`, is incomplete: it
holds the start of a character but not all of it. This is the designed behaviour
of byte-fallback tokenization, not an error. The rate measures how often the
vocabulary can represent a whole character rather than a sub-character byte
sequence.

**Example:** `é` (U+00E9) is the bytes `C3 A9`. A tokenizer that produces `caf` |
`é` for `café` emits two tokens, both holding complete UTF-8, completeness rate
1.0. A byte-fallback tokenizer that produces `caf` | `<0xC3>` | `<0xA9>` emits 3
content tokens, 2 of which hold incomplete UTF-8, completeness rate 1/3.

### Character boundary crossing rate

`utf8_token_integrity.per_tokenizer.<tok>.global.boundary_crossing_rate`. The
fraction of content tokens that hold bytes from more than one UTF-8 character
with at least one of those characters incomplete within the token. Such tokens
come from BPE merges that fused bytes across a character boundary.

This is distinct from a plain byte-fallback token. `<0xC3>` is incomplete but
does not cross a boundary: it holds bytes from exactly one character. A token
holding `A9 E4`, the tail byte of `é` merged with the leading byte of a CJK
character, spans two characters and completes neither.

**Example:** the bytes `C3 A9 E4 BD A0` are the characters `é你`. A tokenizer
that merges the last byte of `é` with the first byte of `你` produces `C3` |
`A9 E4` | `BD A0`. The middle token holds the continuation byte of `é` and the
leading byte of `你` and completes neither, so the crossing rate is 1/3.

A byte-fallback token can be recombined with its neighbours to reconstruct the
character. A boundary-crossing token cannot: its bytes belong to two characters,
and one embedding stands for both partial characters at once.

### Character boundary split count

`utf8_token_integrity.per_tokenizer.<tok>.char_split`. How many multi-byte
characters in the source text have their bytes spread across more than one
token. Each token's bytes are reconstructed and aligned to the source text to
decide this. `split_rate` is splits divided by aligned multi-byte characters;
`splits_per_1k_multibyte` and `splits_per_1k_tokens` are also reported, along
with a `per_byte_width` breakdown for 2-, 3- and 4-byte characters.

**Example:** `你好` holds two 3-byte characters (`你` = `E4 BD A0`, `好` =
`E5 A5 BD`). A tokenizer that keeps each character as one token has 0 splits. A
byte-fallback tokenizer that produces `<0xE4>` | `<0xBD>` | `<0xA0>` for `你` has
1 split, that character's bytes spanning 3 tokens. With `好` intact the split
rate is 1/2 = 0.5.

**Alignment reliability.** When a tokenizer does not reproduce the source bytes,
for example an English-trained tokenizer on Cyrillic or CJK where characters are
dropped or replaced by placeholders, alignment cannot map every source byte to a
token. A multi-byte character with any unaligned byte cannot be classified as
split or not split, so it is excluded from both the numerator and the
denominator and counted separately. Three fields report this per language and
globally: `unaligned_multibyte_chars` (the excluded count), `aligned_fraction`
(the share of multi-byte characters that aligned) and `alignment_mismatches`
(the raw count of unaligned source bytes). When no multi-byte character aligns,
`split_rate`, `splits_per_1k_multibyte` and the per-byte-width split rates are
`null` rather than `0.0`, so "no data" is not read as "no splits". A low
`aligned_fraction` means the split rate rests on few characters. When
`alignment_mismatches` is 0 every multi-byte character aligned and this handling
does not affect the split rate.

**Corpus resolution.** The denominator is the number of aligned multi-byte
characters in the evaluated text. On ASCII-dominant languages measured on a
small parallel corpus this is small (the English FLORES+ sample holds 57
multi-byte characters), so `split_rate` takes few distinct values and separates
Latin-script languages poorly. A corpus with more natural
multi-byte content, for example FineWeb2, gives better resolution.

The completeness rate counts tokens and the split count counts source
characters. A tokenizer can have few incomplete tokens overall and still split
most multi-byte characters, because one split character produces several
incomplete tokens.

## Code tokenization metrics

Source code is parsed with tree-sitter and the alignment between AST node
boundaries and token boundaries is measured. Tree-sitter is a core dependency
and is installed by default. Configure the corpus with `--code-ast-config`;
disable with `--no-code-ast`.

Configured for 19 languages, of which 16 are measured. Swift, Kotlin and Perl
are excluded because the node types their grammars use for identifiers are not
classified: the identifier share of classified leaves is 0.073, 0.058 and 0.000
against 0.19 to 0.37 for the supported languages. They are skipped with a named
warning rather than scored on a fraction of their code. A grammar that crashes
its parser process is likewise reported as unmeasured and named in the log.

Each language is parsed in its own subprocess, because a corrupt parse can abort
the whole process. A grammar that hangs is reported as unmeasured and named in
the log rather than scored. The per-language timeout defaults to 120 seconds;
raise it with `TOKEVAL_PARSE_TIMEOUT_S` on a loaded machine, where a grammar can
exceed the default and be dropped from a run that would otherwise measure it.

> **Data scope:** these metrics are **always** computed on dedicated source-code
> snippets, loaded via `--code-ast-config` or, without it, from small built-in
> synthetic samples. The general multilingual corpus passed to the analyzer is
> **never** used for this metric group, whatever the flags.

### AST leaf-node boundary alignment (`ast_boundary_alignment`)

Source code is parsed with tree-sitter, leaf-node spans are extracted, and the
fraction whose boundaries coincide with token boundaries is measured. Five
categories are tracked separately: identifiers, keywords, operators, literals
and delimiters. Reports start-alignment rate, end-alignment rate,
full-alignment rate and cross-boundary rate, per language and per category. The
per-category breakdown is dropped from `analysis_results.json` and kept in
`analysis_results_full.json`.

A node is start-aligned when it begins at position 0 or the token changes
between the character before it and its first character. It is end-aligned when
it ends at the end of the text or the token changes between its last character
and the character after. Full alignment requires both.

The correspondence between source characters and tokens comes from the
tokenizer's own offsets, not from decoding the tokens and matching the result
back against the source. One word-start space is dropped from each token's
range first, because `trim_offsets` is a ByteLevel post-processor flag that
changes the reported offsets without changing the tokenization: GPT-2 ships it
false and GPT-NeoX true, and reading the raw offsets scored GPT-2 at 0.433
against GPT-NeoX at 0.770 on token ids that were byte-identical.

A span that no token covers is counted as `unmappable` and excluded, rather than
scored as a missed boundary. Measured on 1500 real source files, 198 spans per
tokenizer are unmappable, 0.03%, and every one is a literal whose source text is
a single space, which no token covers once the word-start space is dropped.

**Example:** for the Python snippet `return total`, tree-sitter identifies
`return` (keyword, characters 0 to 6) and `total` (identifier, characters 7 to
12).

- Tokenized `return` | ` total`, the keyword `return` is start-aligned (it
  begins at 0) and end-aligned (the token changes at the space), so it is fully
  aligned. The identifier `total` is not start-aligned: its first character
  shares a token with the preceding space. Full alignment is 1/2 = 0.5. A token
  that is a leading space plus a word is the ordinary output of a byte-level
  pre-tokenizer, so this is the common case, not a pathological one.
- Tokenized `return` | ` ` | `total`, the space is its own token, both nodes are
  fully aligned, and full alignment is 2/2 = 1.0.
- Tokenized `ret` | `urn total`, `return` is start-aligned but not end-aligned
  (its last character shares a token with the space that follows), and `total`
  is not start-aligned. Full alignment is 0/2 = 0.0.

Code has a deterministic grammar, so AST node boundaries are derivable with no
manual annotation. A tokenizer that splits `return` into `ret` and `urn` splits
a syntactically atomic unit.

### Identifier fragmentation rate (`identifier_fragmentation`)

The fraction of programmer-defined identifiers split into more than one token,
plus the average number of tokens per identifier. Computed occurrence-weighted
from the same AST extraction pass, so a frequent identifier counts once per
occurrence.

**Example:** a Python file holds `self` (10 occurrences), `i` (5),
`process_data` (3) and `MyAuthenticationFactory` (1). The tokenizer keeps `self`
and `i` as single tokens, splits `process_data` into `process` | `_` | `data`
(3 tokens) and `MyAuthenticationFactory` into `My` | `Auth` | `entication` |
`Factory` (4 tokens). Fragmentation rate is 4 fragmented occurrences out of 19
total = 0.211. Average tokens per identifier is (10x1 + 5x1 + 3x3 + 1x4) / 19 =
1.474. The 10 occurrences of `self` supply 10 of the 19 terms in that average,
so the metric mostly describes the most frequent identifier rather than the
rarer, longer ones.

Identifiers are the names a programmer chose. Splitting `getUserName` at
arbitrary positions produces pieces that do not correspond to those names. The
current implementation does not distinguish splits at camelCase or snake_case
boundaries from arbitrary ones.

### Indentation depth proportionality correlation (`indentation_consistency`)

Whether the number of whitespace tokens produced for leading indentation grows
monotonically with nesting depth. The Spearman rank correlation (ρ) is computed
between indentation depth and the count of whitespace tokens in each line's
leading indentation.

Spearman is rank-based, so ρ = 1 means the count increases monotonically with
depth, not in proportion to it. The key is named for proportionality and the
statistic measures monotonicity.

Only tokens whose surface is entirely whitespace are counted, because a
pre-tokenizer that groups a leading space with the following word emits a token
spanning both, and that token is code. Depth is the line's leading-whitespace
width divided by the indent unit, which is inferred per snippet as the GCD of
its non-zero indent widths; it does not come from the parse tree. Evaluated only
on whitespace-significant languages (Python and Haskell), and only where a
language has at least 3 distinct depth levels.

**Example:** a Python file has lines at depths 1, 2, 3 and 4. A tokenizer that
encodes those indentations as 1, 2, 3 and 4 whitespace tokens gives ρ = 1.0. A
tokenizer that merges all indentation into a single token regardless of depth
(1, 1, 1, 1) gives ρ near 0.0. A tokenizer that uses more tokens for shallow
depths than for deep ones gives ρ < 0.

Where indentation depth maps monotonically to whitespace token count, nesting
depth is recoverable from the token sequence alone.

**How the global is computed:** `global.depth_proportionality_correlation` is
one Spearman correlation over the (depth, whitespace-token count) pairs of every
whitespace-significant language pooled together, not the mean of the
per-language correlations. The two differ. On the demo run named above, `bpe`
scores 0.7598 pooled against a per-language mean of 0.8121 (Python 0.9159,
Haskell 0.7083). Indent conventions differ by language, so the pooled value
depends on how much of each language the code corpus holds. Read `per_language`
for each language separately.

## Multilingual fairness

### Tokenizer Gini coefficient (`tokenizer_fairness_gini`)

Equitability of encoding cost across languages.

* Let $`L = \{1, \dots, n\}`$ be the set of languages, each weighted equally.
* For every language $`\ell \in L`$, the **token cost** is

```math
  c_\ell \;=\;
  \frac{\text{number of tokens produced by the tokenizer on language }\ell}
       {\text{number of measurement units in the same text}}
```

  A lower $`c_\ell`$ means the language is encoded in fewer tokens per unit of
  text. The unit is whatever `--measurement-config` sets, UTF-8 bytes by
  default.

* Let the mean cost be

```math
  \mu \;=\; \frac{1}{n}\;\sum_{\ell=1}^{n} c_\ell.
```

Then the **Tokenizer Fairness Gini** with equal weights is

```math
\mathrm{TFG}
=\frac{\displaystyle\sum_{i=1}^{n}\sum_{j=1}^{n} \lvert c_i - c_j \rvert}
        {2\,n^2\,\mu}
```

* **Range:** $`0 \le \mathrm{TFG} \le 1`$
  * $`0`$: every language has the same token cost per unit of text.
  * $`1`$: maximal inequality. The attainable maximum is $`1 - 1/n`$, so on 13
    languages no tokenizer can exceed 0.923.

**This is a fair comparison only on a parallel corpus.** The cost is tokens per
unit of text, so if the texts in two languages say different things, the ratio
between their costs is partly the ratio of what they say. FLORES+ is parallel,
which is why the shipped configs use it.

**Even on a parallel corpus, the byte unit is not neutral across scripts.** UTF-8
spends one byte per Latin character, two for Cyrillic and Greek, three for most
CJK and Devanagari. A tokenizer can therefore look cheaper on Chinese than on
English purely because the denominator is larger, with no difference in how well
it segments either. Two ways to read around it: compare the same tokenizer
across languages only when you have accounted for that, or set
`--measurement-config` to a lines config, where a parallel corpus gives every
language the same denominator, one line per sentence, and the cost becomes
tokens per sentence. The number under a lines config is not comparable with the
number under the byte default.

The underlying Lorenz curve is written under
`tokenizer_fairness_gini.per_tokenizer.<tok>.lorenz_curve`, from which
`1 - 2*area` is the Gini coefficient. With fewer than 2 languages the
coefficient is `null` and a sibling `warning` field records the reason.

### Cross-lingual vocabulary-utilization CoV

`vocabulary_utilization.per_tokenizer.<tok>.per_language_cov`. The coefficient
of variation (sample standard deviation divided by mean, `ddof=1`) of the
per-language vocabulary-utilization *ratio* across languages. Computed on the
ratio rather than on the raw used-token count, so it is comparable across
tokenizers with different vocabulary sizes.

Lower is better: a low value means each language uses a similarly sized share of
the vocabulary; a high value means the used vocabulary is concentrated in some
languages. This complements the Gini coefficient above, which measures the
equitability of per-language *encoding cost*, where this measures the balance of
per-language *vocabulary coverage*.

Requires at least 2 languages with mean utilization above 0. For a
single-language corpus it is `null` rather than a fabricated `0`, and it is
omitted from the plot. `per_language_mean` and `per_language_std` sit beside it.
The comparison plot is written to `vocab_util_cross_lingual_cov_individual.svg`;
that string is a plot filename, not a results key.
