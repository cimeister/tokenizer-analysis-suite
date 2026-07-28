# Security policy

## Supported versions

Fixes go to the latest released version. There are no long-term support
branches.

## Reporting a vulnerability

Report privately through GitHub's [security advisory
form](https://github.com/cimeister/tokenizer-intrinsic-evals/security/advisories/new)
rather than a public issue. Expect an acknowledgement within a week.

## Scope

This is an evaluation library: it reads tokenizer files, config files, and text
corpora that you point it at, and writes results. Things worth reporting:

- Code execution triggered by loading a tokenizer or config file. Note that
  `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` is used when
  loading a HuggingFace tokenizer, so a Hub repository you name can execute
  code by design. Only load tokenizers you trust.
- Path traversal or writes outside the directory given by `--output-dir`.
- Deserialization issues in the pre-tokenized data path, which uses `pickle`
  and will execute code from a malicious `.pkl`. Only load caches you produced.

Out of scope: a metric returning a number you disagree with. Open a normal
issue for that, ideally with a reproducer.
