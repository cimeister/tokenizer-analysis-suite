"""Resolution of the code and math corpora a run measures.

One definition of "which corpus does this run measure", used by the run
(``UnifiedTokenizerAnalyzer``, which resolves both corpora once and registers
them on the input provider) and by the metric classes when they are constructed
on their own with no provider registry to read. Two copies of these rules would
let a corpus built here and a corpus built there drift apart, and the results
file would then carry numbers measured on one while naming the other.

Every function returns a ``Corpus``, whose ``source`` is published as
``by_domain.<domain>.source`` and whose ``synthetic`` flag records that the
texts are the samples bundled with the package rather than data the caller
asked to have measured.
"""

import logging
from typing import Dict, List, Optional

from ..core.input_types import (
    CODE_CORPUS, CODE_DATASET_SOURCE, MATH_CORPUS, Corpus,
)
from ..utils.text_utils import load_math_data, BUILTIN_MATH_SAMPLES_PATH
from .code_data import CodeDataLoader

logger = logging.getLogger(__name__)


def synthetic_code_corpus() -> Corpus:
    """The code samples bundled with the package.

    ``synthetic`` is load-bearing rather than descriptive. With no code config
    the AST metrics and the operator-isolation code domain run on these samples
    while reconstruction fidelity gets no code domain at all. That asymmetry is
    deliberate; see ``TestTheDefaultCodeConfigurationIsAsymmetric`` in
    tests/test_output_contract.py.
    """
    return Corpus(
        name=CODE_CORPUS,
        texts=CodeDataLoader.generate_synthetic_samples(),
        source=CodeDataLoader._BUILTIN_CODE_SAMPLES_PATH,
        synthetic=True,
    )


def code_corpus_from_texts(
    code_texts: Optional[Dict[str, List[str]]],
) -> Corpus:
    """A code corpus from texts a caller loaded itself, or the bundled samples.

    A language whose list is empty is dropped, so that it is absent from the
    published corpus size rather than reported as a language that supplied no
    operators. ``CodeDataLoader`` never stores an empty list, so this only
    affects a caller passing *code_texts* by hand.
    """
    if not code_texts:
        return synthetic_code_corpus()
    return Corpus(
        name=CODE_CORPUS,
        # list(texts), not texts: Corpus is frozen, and aliasing the
        # caller's lists let an append after registration change a
        # corpus the provider had already encoded and cached, so
        # stats() and the published numbers came from different
        # contents.
        texts={lang: list(texts) for lang, texts in code_texts.items() if texts},
        source=CODE_DATASET_SOURCE,
        synthetic=False,
    )


def resolve_code_corpus(
    code_config: Optional[Dict[str, str]],
    max_snippets_per_lang: Optional[int] = None,
    max_snippet_chars: Optional[int] = None,
) -> Corpus:
    """The code corpus for a run: the caller's files, or the bundled samples.

    No try/except around the load: a code config the caller named explicitly
    either loads or aborts. Swallowing the failure left the operator-isolation
    code domain with zero samples and no signal beyond one warning line, while
    the same malformed config crashed uncaught further down with a raw
    AttributeError that named neither the flag nor the file.
    """
    if not code_config:
        return synthetic_code_corpus()
    loader = CodeDataLoader(
        code_config,
        max_snippets_per_lang=max_snippets_per_lang,
        max_snippet_chars=max_snippet_chars,
    )
    loader.load_all()
    # Substituting the bundled samples for a config that named real paths would
    # report code metrics computed on toy snippets under the name of the corpus
    # the caller asked for: measured 0.562 full AST alignment on the samples
    # against 0.493 on StarCoder for the same tokenizer.
    if not loader.code_snippets:
        raise ValueError(
            f"The code config named {', '.join(sorted(code_config))} but no "
            "snippet was read from any of those paths. Check that the "
            "directories hold files with the expected extensions."
        )
    return Corpus(
        name=CODE_CORPUS,
        texts=loader.code_snippets,
        source=CODE_DATASET_SOURCE,
        synthetic=False,
    )


def resolve_math_corpus(
    math_data_path: Optional[str] = None,
    use_builtin_math_data: bool = False,
) -> Corpus:
    """The math corpus for a run: the caller's file, or the bundled samples.

    The bundled samples are ``synthetic`` only when the caller asked for
    neither. They are what the operator-isolation math domain falls back to so
    that it always has something to score. The digit metrics and reconstruction
    fidelity do not read a synthetic corpus, so neither has ever run on the
    bundled samples unless --math-data or --use-builtin-math-data was passed.
    """
    if math_data_path:
        texts = load_math_data(math_data_path)
        if not texts:
            raise ValueError(
                f"math_data_path {math_data_path!r} loaded 0 texts. Refusing to "
                "fall back to the bundled samples: the run would silently measure "
                "a different corpus than the one asked for."
            )
        logger.info("Loaded %d math texts from %s", len(texts), math_data_path)
        return Corpus(
            name=MATH_CORPUS, texts={MATH_CORPUS: texts},
            source=math_data_path, synthetic=False,
        )
    texts = load_math_data(BUILTIN_MATH_SAMPLES_PATH)
    if use_builtin_math_data:
        logger.info("Loaded %d built-in math samples", len(texts))
    return Corpus(
        name=MATH_CORPUS, texts={MATH_CORPUS: texts},
        source=BUILTIN_MATH_SAMPLES_PATH,
        synthetic=not use_builtin_math_data,
    )
