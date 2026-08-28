"""
Input data types and abstractions for tokenizer analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Mapping, Optional, Sequence, Tuple, Union, Protocol, TYPE_CHECKING
from types import MappingProxyType
from abc import ABC, abstractmethod
import logging

if TYPE_CHECKING:
    from .tokenizer_wrapper import TokenizerWrapper

logger = logging.getLogger(__name__)


@dataclass
class TokenizedData:
    """Standardized format for tokenized text data."""
    
    tokenizer_name: str
    language: str
    tokens: List[int]
    text: Optional[str] = None  # Original text if available
    offsets: Optional[List[Tuple[int, int]]] = None  # Token-to-char offsets from encoding
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate tokenized data after initialization."""
        if not self.tokenizer_name:
            raise ValueError("tokenizer_name cannot be empty")
        if not self.language:
            raise ValueError("language cannot be empty")
        # No emptiness check on tokens. It rejected a legitimate result: a
        # tokenizer that encodes a text to zero tokens has measured something,
        # and refusing to construct the item turns that into a crash.
        #
        # The reasoning recorded here used to be that the construction sites in
        # this package filter blank text upstream, so the check could only ever
        # fire on a PreTokenizedProvider row. That is false, and the two
        # properties are different: a text of C0 control characters is
        # non-blank to str.strip() and still encodes to zero tokens under a
        # normalizer with clean_text=True, which is reachable with a user
        # corpus through --input. Such a record is kept and logged, and each
        # metric that reads one states its own position: fertility excludes it
        # and publishes the count, token_length and avg_tokens_per_line and
        # compression_rate exclude it because tokens are their denominator, and
        # the Gini blocks and reconstruction fidelity keep it because a
        # zero-cost language and a total round-trip failure are the findings
        # those metrics exist to report.
        if not isinstance(self.tokens, list) or not all(isinstance(t, int) for t in self.tokens):
            raise ValueError("tokens must be a list of integers")
        if not self.tokens and self.text and self.text.strip():
            logger.warning(
                "%s encoded a non-blank %s text to zero tokens; it is kept, "
                "counted in fertility.zero_token_documents, and excluded from "
                "the ratios whose denominator it would be. Text starts: %r",
                self.tokenizer_name, self.language, self.text[:40],
            )
        
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def token_count(self) -> int:
        """Get number of tokens."""
        return len(self.tokens)
    
    @property
    def unique_tokens(self) -> set:
        """Get set of unique token IDs."""
        return set(self.tokens)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'tokenizer_name': self.tokenizer_name,
            'language': self.language,
            'tokens': self.tokens,
            'text': self.text,
            'offsets': self.offsets,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenizedData':
        """Create TokenizedData from dictionary."""
        return cls(
            tokenizer_name=data['tokenizer_name'],
            language=data['language'],
            tokens=data['tokens'],
            text=data.get('text'),
            offsets=[tuple(o) for o in data['offsets']] if data.get('offsets') else None,
            metadata=data.get('metadata')
        )


#: The corpus every provider serves from its own data rather than from the
#: registry on ``InputProvider``. ``get_tokenized_data()`` returns it, and both
#: ``add_corpus`` and ``get_corpus_data`` refuse this name, so there is one way
#: to reach the prose texts rather than two that could disagree.
PROSE_CORPUS = "prose"

#: The two corpora the run registers with ``add_corpus``. Both names are also
#: published: as a domain of ``operator_isolation_rate.by_domain``, and, for
#: code, as the ``code_<language>`` prefix of a reconstruction-fidelity domain.
CODE_CORPUS = "code"
#: Used twice for math: as the corpus name, and as the single label inside it,
#: because the math corpus has no per-language split. The published
#: ``by_language`` key for it is therefore the bare corpus name, "math".
MATH_CORPUS = "math"

#: What a provider raises when it cannot supply a tokenizer object for a name
#: it lists. ``InputProvider.get_tokenizer`` raises NotImplementedError, and the
#: two shipped providers raise ValueError or KeyError for a name they do not
#: carry. Named once because ``_encode_corpus`` here and reconstruction fidelity
#: in metrics/basic.py both have to decide "can this tokenizer be scored at
#: all", and they used to answer it with two different catch clauses: a tuple,
#: and a bare ``except Exception`` that also swallowed genuine defects.
#:
#: AttributeError is deliberately not a member. This ABC defines
#: ``get_tokenizer``, so the attribute always resolves and an AttributeError can
#: only escape from inside a subclass's own implementation, where it is that
#: subclass's defect rather than a provider declining to supply a tokenizer.
#: Callers that reach a provider through a duck-typed reference, which may not
#: have the method at all, add it themselves; metrics/basic.py is the one that
#: does.
NO_TOKENIZER_ERRORS = (ValueError, KeyError, NotImplementedError)

#: ``Corpus.source`` for code the caller named with --code-ast-config, as
#: opposed to the bundled samples. Published as ``by_domain.code.source``,
#: which is what lets a reported code number be traced to the corpus it
#: measured, so the two must stay distinguishable.
CODE_DATASET_SOURCE = "code-ast dataset"
#: ``Corpus.source`` for prose, published as ``by_domain.prose.source``. The
#: prose corpus is not registered, so this names the corpus the run was given
#: rather than one of the corpora it loaded.
PROSE_SOURCE = "multilingual corpus"


def published_language_key(domain: str, lang: str) -> str:
    """The published key for one (domain, language) pair.

    Prose keeps its language name. Code becomes ``code_<lang>``. Maths is the
    bare corpus name, because there is one maths domain and the label inside it
    is not part of the key.

    Here rather than on a metric class because two files publish these keys and
    both already import this module. They used to build them separately, five
    places by hand against one function, and they agreed only by coincidence.
    """
    if domain == PROSE_CORPUS:
        return lang
    if domain == CODE_CORPUS:
        return f"{CODE_CORPUS}_{lang}"
    return domain


def is_corpus_domain_key(key: str) -> bool:
    """True when *key* names a code or maths domain rather than a language.

    Recognising a key is the same knowledge as building one, so it lives beside
    ``published_language_key``. Two places tested this by hand.
    """
    return key.startswith(f"{CODE_CORPUS}_") or key == MATH_CORPUS


def corpus_size(texts: 'Mapping[str, Sequence[str]]') -> Dict[str, Any]:
    """The published size block for a labelled set of texts.

    Module-level rather than only a Corpus method, because the prose domain of
    operator isolation reports the same block for texts that are never a
    registered corpus. Building a throwaway Corpus for it copied every prose
    text into a tuple on each compute() call to reach the same four numbers.
    """
    # The same filter every scored path applies. Without it this counted texts
    # no metric read: code_texts={"python": ["   ", "x = 1"]} published
    # n_texts 2 with one text encoded, under a block whose whole purpose is to
    # let a reported number be traced to what it measured.
    kept = {label: [t for t in group if t and t.strip()]
            for label, group in texts.items()}
    per_label = {label: len(group) for label, group in kept.items() if group}
    return {
        "n_texts": sum(per_label.values()),
        "n_chars": sum(len(t) for group in kept.values() for t in group),
        "n_languages": len(per_label),
        "texts_per_language": dict(sorted(per_label.items())),
    }


@dataclass(frozen=True)
class CorpusCaps:
    """What the file and character caps did while a corpus was built.

    Carried on the Corpus rather than on the loader that produced it, because
    the loader is not the common shape. ``resolve_code_corpus`` builds one on
    the real-files branch and discards it; the synthetic branch builds no
    loader at all. Both branches produce a Corpus, so that is where the
    provenance goes, and ``ASTBoundaryMetrics`` can report the truncation its
    corpus actually received instead of the loader default it was constructed
    with.

    Attributes:
        max_snippets_per_lang: the file cap in force, 0 for no cap.
        max_snippet_chars: the character cap in force, 0 for no cap. This is
            the value the texts were truncated with, not a value a caller
            asked for and nothing honoured.
        dropped_file_counts: label -> candidate files the file cap skipped.
        truncated_char_counts: label -> characters the character cap removed.
        dropped_whitespace_only_counts: label -> snippets dropped because
            truncation left them whitespace-only.
    """

    max_snippets_per_lang: int
    max_snippet_chars: int
    dropped_file_counts: Mapping[str, int]
    truncated_char_counts: Mapping[str, int]
    dropped_whitespace_only_counts: Mapping[str, int]

    def __post_init__(self):
        """Freeze the three count mappings, for the reason Corpus freezes texts."""
        for field in ("dropped_file_counts", "truncated_char_counts",
                      "dropped_whitespace_only_counts"):
            object.__setattr__(
                self, field, MappingProxyType(dict(getattr(self, field)))
            )


@dataclass(frozen=True)
class Corpus:
    """A named set of labelled texts, encoded once per tokenizer.

    The code and math corpora are resolved in ``loaders/corpora.py``, registered
    with ``InputProvider.add_corpus``, encoded by ``_encode_corpus`` and read
    back by every metric that measures them, as
    ``Dict[tokenizer_name, List[TokenizedData]]``. Each used to be loaded and
    encoded separately by each of the three metric classes that consume it.

    Prose is not registered. It is whatever texts the provider was constructed
    with, served by ``get_tokenized_data()``. A ``Corpus`` is still built for it
    in one place, ``DigitBoundaryMetrics``, to describe it in the per-domain
    report; that one is never registered and never looked up by name.

    Attributes:
        name: what ``get_corpus_data(name)`` asks for, so "code" or "math".
            A Corpus is also built for prose when a metric reports on the data
            it was handed, but that one is never registered and never looked
            up by name.
        texts: label -> texts. The label is a language for prose, a programming
            language for code, and "math" for math.
        source: where the texts came from. Already published as
            ``by_domain.<domain>.source``, which is what lets a reported domain
            be traced to the corpus it measured.
        synthetic: True when the texts are the bundled samples rather than real
            data. This is load-bearing rather than decoration: with no
            ``--code-ast-config`` the AST metric runs on synthetic code while
            reconstruction fidelity gets no code domain at all, and a reader
            cannot tell a synthetic domain from a real one without it.
        caps: a ``CorpusCaps`` recording what the file and character caps
            removed, or None when no cap was applied. The loader that did the
            work is built and discarded inside ``resolve_code_corpus``, so
            without this the counters were unreachable from the corpus every
            metric actually reads.
    """

    name: str
    #: Read-only after construction: __post_init__ replaces whatever was passed
    #: with a MappingProxyType of tuples, so the annotation states what the
    #: class guarantees rather than what the caller may hand in.
    texts: Mapping[str, Tuple[str, ...]]
    source: str
    synthetic: bool
    #: What the caps did while these texts were built, or None for a corpus
    #: no cap was applied to. Defaulted so the many call sites that build a
    #: Corpus by hand, tests included, are unaffected.
    caps: Optional['CorpusCaps'] = None

    def __post_init__(self):
        """Freeze *texts*, so ``frozen=True`` means what it says.

        The dataclass being frozen stops the four fields being reassigned and
        does nothing about the dict and the lists inside one of them. Copying
        on construction stopped a caller's later append reaching a corpus
        through the reference they still held, and left
        ``provider.get_corpus('code').texts['python'].append(...)`` working,
        which changes a corpus the provider may already have encoded and
        memoized: ``stats()`` then reports one size while the published numbers
        were measured on other contents.

        So the labels map to tuples behind a read-only view, and both routes
        raise instead. Consumers already build their own list or dict from this
        (``dict(corpus.texts)``, ``list(texts)``), so they are unaffected.

        A Corpus is still not hashable, because a mapping is not.
        """
        for label, texts in self.texts.items():
            if isinstance(texts, str):
                # tuple() over a string iterates its characters, so a bare
                # string became one text per character: the corpus reported
                # n_texts 5 for "x = 1" and every per-text metric scored single
                # characters. A caller who means one text has to say so.
                raise TypeError(
                    f"The {label!r} label of the {self.name!r} corpus was given "
                    "a bare string. That would be split into one text per "
                    f"character. Pass [{texts[:20]!r}...] to mean one text."
                    if len(texts) > 20 else
                    f"The {label!r} label of the {self.name!r} corpus was given "
                    "a bare string. That would be split into one text per "
                    f"character. Pass [{texts!r}] to mean one text."
                )
        object.__setattr__(
            self, "texts",
            MappingProxyType(
                {label: tuple(texts) for label, texts in self.texts.items()}
            ),
        )

    def stats(self) -> Dict[str, Any]:
        """Size of this corpus, so a reported domain can be traced to what it measured.

        The pooled summary is a micro-average, so the domain that contributes
        the most operators sets it. Publishing each domain's size is what lets
        a reader see which corpus is doing the work.
        """
        return corpus_size(self.texts)


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer objects."""
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        ...
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...


class VocabularyProvider(Protocol):
    """Protocol for objects that provide vocabulary information."""
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping (optional)."""
        ...


@dataclass
class InputSpecification:
    """Specification for input data to the analysis pipeline."""
    
    # For raw tokenization mode
    tokenizer: Optional['TokenizerWrapper'] = None
    texts: Optional[Dict[str, Union[str, List[str]]]] = None  # language -> text or list of texts
    
    # For pre-tokenized mode
    tokenizer_name: Optional[str] = None
    vocabulary: Optional[VocabularyProvider] = None  # Kept for backward compatibility, but tokenizer is preferred
    tokenized_data: Optional[List[TokenizedData]] = None
    
    # Common
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate input specification."""
        if self.metadata is None:
            self.metadata = {}
        
        # Validate mode consistency
        has_raw_inputs = self.tokenizer is not None and self.texts is not None
        has_tokenized_inputs = (self.tokenizer is not None and 
                               self.tokenized_data is not None)
        
        # Support legacy mode with tokenizer_name + vocabulary
        has_legacy_tokenized_inputs = (self.tokenizer_name is not None and 
                                     self.vocabulary is not None and 
                                     self.tokenized_data is not None)
        
        if not has_raw_inputs and not has_tokenized_inputs and not has_legacy_tokenized_inputs:
            raise ValueError(
                "Must provide either (tokenizer + texts) for raw mode or "
                "(tokenizer + tokenized_data) for pre-tokenized mode"
            )
        
        if has_raw_inputs and (has_tokenized_inputs or has_legacy_tokenized_inputs):
            raise ValueError(
                "Cannot provide both raw and pre-tokenized inputs simultaneously. "
                "Use separate InputSpecification objects."
            )
    
    @property
    def is_raw_mode(self) -> bool:
        """Check if this is raw tokenization mode."""
        return self.tokenizer is not None and self.texts is not None
    
    @property
    def is_pretokenized_mode(self) -> bool:
        """Check if this is pre-tokenized mode."""
        return (self.tokenizer is not None and self.tokenized_data is not None) or \
               (self.tokenizer_name is not None and 
                self.vocabulary is not None and 
                self.tokenized_data is not None)
    
    def get_tokenizer_name(self) -> str:
        """Get tokenizer name for both modes."""
        if self.is_raw_mode:
            return self.tokenizer.get_name() if hasattr(self.tokenizer, 'get_name') else getattr(self.tokenizer, 'name', 'unknown')
        elif self.tokenizer is not None:
            return self.tokenizer.get_name()
        else:
            return self.tokenizer_name or 'unknown'
    
    def get_languages(self) -> List[str]:
        """Get list of languages in this specification."""
        if self.is_raw_mode:
            return list(self.texts.keys())
        # sorted, for the reason PreTokenizedProvider.get_languages is: set
        # order made the language order hash-dependent between runs of one
        # dump. Same defect, same file family, missed by that sweep.
        return sorted({td.language for td in self.tokenized_data})
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size, from whichever source this specification carries.

        Both branches were wrong. The raw branch read ``tokenizer.vocab_size``,
        an attribute ``TokenizerWrapper`` does not have (it exposes
        ``get_vocab_size()``), and the pre-tokenized branch read
        ``self.vocabulary``, which is None for the ``tokenizer +
        tokenized_data`` shape that
        ``main.create_analyzer_from_tokenized_data`` builds. So the method
        raised ``AttributeError`` on every specification the package itself
        constructs. The tokenizer is the preferred source in both modes, with
        the legacy ``vocabulary`` provider, whose protocol declares
        ``vocab_size`` as a property, as the fallback.
        """
        if self.tokenizer is not None:
            return self.tokenizer.get_vocab_size()
        if self.vocabulary is not None:
            return self.vocabulary.vocab_size
        raise ValueError(
            "This InputSpecification carries neither a tokenizer nor a "
            "vocabulary provider, so its vocabulary size cannot be read."
        )


class InputProvider(ABC):
    """Abstract base class for providing tokenized data to analysis pipeline."""
    
    @abstractmethod
    def get_tokenized_data(self) -> Dict[str, List[TokenizedData]]:
        """
        Get the provider's own prose data, organized by tokenizer name.

        This method serves only the texts the provider was constructed with.
        The corpora registered with ``add_corpus`` are read through
        ``get_corpus_data``, which this ABC implements once for every provider.
        The two are kept apart because they are produced differently: this one
        is the subclass's own data, while a registered corpus is encoded by
        ``_encode_corpus`` here. Taking a corpus name here instead would also
        have changed this abstract signature, which every subclass outside this
        package implements.

        Returns:
            Dictionary mapping tokenizer names to lists of TokenizedData objects
        """
        pass
    
    @abstractmethod
    def get_tokenizer_names(self) -> List[str]:
        """Get list of tokenizer names."""
        pass
    
    @abstractmethod
    def get_vocab_size(self, tokenizer_name: str) -> int:
        """Get vocabulary size for a tokenizer."""
        pass
    
    @abstractmethod
    def get_languages(self, tokenizer_name: str = None) -> List[str]:
        """Get list of languages. If tokenizer_name is None, return all languages."""
        pass
    
    # ------------------------------------------------------------------
    # Named corpora
    # ------------------------------------------------------------------
    #
    # These are concrete rather than abstract, for the same reason
    # validate_data below is. Four classes subclass this ABC: the two providers
    # in input_providers.py, _AstOnlyProvider in scripts/run_ast_only.py, and
    # SimpleProvider in the test suite. A new abstract method would stop every
    # one of them from being instantiated.
    #
    # The registry is built on first use rather than in an __init__. This ABC
    # has no __init__ today and no subclass calls super().__init__(), so an
    # __init__ added here would run for none of them and every method below
    # would raise AttributeError on the missing attribute.

    def _corpus_registry(self) -> Dict[str, 'Corpus']:
        """The registered corpora, created on first use."""
        # self.__dict__, not getattr: getattr finds a class attribute as
        # readily as an instance one, so a subclass written as
        # `class P(InputProvider): _corpora = {}` would give every instance the
        # same registry. add_corpus on one provider would then refuse a name
        # another provider had registered, and get_corpus_data would hand back
        # encodings made with a different provider's tokenizers.
        registry = self.__dict__.get('_corpora')
        if registry is None:
            registry = {}
            self._corpora = registry
        return registry

    def add_corpus(self, corpus: 'Corpus') -> None:
        """Register *corpus* under its own name.

        A name that is already registered is refused rather than overwritten.
        Two loaders registering "code" would otherwise mean whichever ran
        second measured a different corpus from the one the first one reported,
        with nothing in the output saying which corpus produced which number.

        The prose corpus cannot be registered. It is served from the provider's
        own data, so a registered corpus under that name would have recorded a
        source and a set of texts that nothing reads, while the numbers came
        from somewhere else.
        """
        if corpus.name == PROSE_CORPUS:
            raise ValueError(
                f"The {PROSE_CORPUS!r} corpus cannot be registered: it is "
                "served from the provider's own texts, through "
                "get_tokenized_data(). Registering it here would record a "
                "source that no metric reads. Register a corpus under its own "
                "name instead."
            )
        registry = self._corpus_registry()
        if corpus.name in registry:
            raise ValueError(
                f"A corpus named {corpus.name!r} is already registered, with "
                f"source {registry[corpus.name].source!r}. Refusing to replace "
                f"it with one from {corpus.source!r}: the metrics that already "
                "read the first one would then report numbers measured on the "
                "second."
            )
        registry[corpus.name] = corpus

    def corpus_names(self) -> List[str]:
        """Names of the registered corpora, in registration order."""
        return list(self._corpus_registry())

    def get_corpus(self, name: str) -> 'Corpus':
        """The corpus registered under *name*."""
        registry = self._corpus_registry()
        if name not in registry:
            raise ValueError(
                f"No corpus named {name!r} is registered. Registered corpora: "
                f"{sorted(registry)}."
            )
        return registry[name]

    def get_tokenizer(self, tokenizer_name: str) -> 'TokenizerWrapper':
        """The tokenizer object for *tokenizer_name*.

        Declared here because eight call sites, in main.py and five of the
        metric modules, already required it while this ABC never named it, so a
        provider that omitted it failed at the call site rather than at the
        class. Both providers in input_providers.py implement it and keep their
        own version.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not supply tokenizer objects, so the "
            "metrics that encode their own corpora cannot run against it."
        )

    def get_corpus_data(self, name: str) -> Dict[str, List[TokenizedData]]:
        """A registered corpus, encoded with every tokenizer this provider has.

        This is the counterpart to ``get_tokenized_data()`` for the corpora a
        run registers with ``add_corpus``: code and math. It is concrete, so
        every provider gets it, including ones written before this registry
        existed.

        Asking for the prose corpus here is refused rather than served, because
        prose does not come from the registry.
        """
        if name == PROSE_CORPUS:
            raise ValueError(
                f"The {PROSE_CORPUS!r} corpus is not part of the corpus "
                "registry: it is the provider's own data. Call "
                "get_tokenized_data() for it."
            )
        return self._tokenized_corpus(name)

    def _tokenized_corpus(self, name: str) -> Dict[str, List[TokenizedData]]:
        """A registered corpus, encoded with every tokenizer, memoized by name.

        ``compute()`` is called once per language group and a registered corpus
        does not change between those calls, so it is encoded once rather than
        re-encoded for every group.
        """
        # self.__dict__ for the same reason as _corpus_registry above.
        cache = self.__dict__.get('_encoded_corpora')
        if cache is None:
            cache = {}
            self._encoded_corpora = cache
        if name not in cache:
            cache[name] = self._encode_corpus(self.get_corpus(name))
        return cache[name]

    @staticmethod
    def can_encode_raw_text(tokenizer_obj: Any) -> bool:
        """Whether *tokenizer_obj* can encode raw text.

        Public, and named without a leading underscore, because three metric
        modules call it and two error messages quote it by name. A predicate
        that many callers depend on is part of the interface whatever its name
        suggests, and an underscore only hid that renaming it would break them.

        ``can_encode()`` is the predicate, not ``hasattr(tok, "encode")``:
        ``PreTokenizedDataTokenizer`` *defines* ``encode`` and raises from it.

        Used to tell two situations apart when no character offsets are
        available: a tokenizer that encodes text and reported none is a defect
        the caller has to know about, while a provider that only supplies
        pre-tokenized ids never had offsets to give.
        """
        can_encode = getattr(tokenizer_obj, "can_encode", None)
        encode = getattr(tokenizer_obj, "encode", None)
        if callable(can_encode) and not can_encode():
            return False
        return callable(encode)

    def _encode_corpus(self, corpus: 'Corpus') -> Dict[str, List[TokenizedData]]:
        """Encode a registered corpus with every tokenizer this provider carries.

        A tokenizer that cannot encode raw text (a provider that only supplies
        pre-tokenized data) is omitted from the returned corpus and logged; it
        then simply has no code or math domain rather than crashing the whole
        metric.

        This is deliberately not the prose loop in
        ``RawTokenizationProvider.get_tokenized_data``. The differences that
        move published numbers are:

        1. The prose loop raises when a tokenizer cannot encode raw text; this
           one skips that tokenizer with a warning. This covers the check made
           before encoding starts. If ``encode`` itself raises partway through a
           corpus, it propagates out of here exactly as it does out of the prose
           loop.
        2. The prose loop records per-sample encode times, published as
           ``encoding_speed``; this one records none, so a derived corpus does
           not enter that measurement.

        Unifying them would change which tokenizers are skipped and what
        ``encoding_speed`` measures. The list above is the set that moves
        numbers, not the set of all differences: the two also differ in whether
        batch pre-validation runs, in the pairing label they pass to
        ``check_batch_pairing``, and in whether ``metadata`` is populated.
        Neither loop falls back to a second encode path any more, which was the
        largest of the differences until this release.
        """
        out: Dict[str, List[TokenizedData]] = {}
        for tok_name in self.get_tokenizer_names():
            try:
                tokenizer_obj = self.get_tokenizer(tok_name)
            except NO_TOKENIZER_ERRORS as exc:
                logger.warning(
                    "No tokenizer available for %r (%s); it gets no %s corpus.",
                    tok_name, exc, corpus.name,
                )
                continue
            encode = getattr(tokenizer_obj, "encode", None)
            if not self.can_encode_raw_text(tokenizer_obj):
                logger.warning(
                    "Tokenizer %r cannot encode raw text; it gets no %s corpus.",
                    tok_name, corpus.name,
                )
                continue
            # Encode with offsets where the wrapper supports it. The prose
            # corpus is loaded with offsets, and operator isolation and AST
            # alignment resolve a source span to tokens through offsets, so a
            # derived corpus without them would be skipped entirely and its
            # domain would silently vanish from the results.
            encode_batch = getattr(tokenizer_obj, "encode_batch_with_offsets", None)
            encode_offsets = getattr(tokenizer_obj, "encode_with_offsets", None)
            items: List[TokenizedData] = []
            for lang, texts in corpus.texts.items():
                usable = [text for text in texts if text and text.strip()]
                if not usable:
                    continue
                # One batch call per label rather than one call per text. The
                # Rust backends encode a batch across threads, which the
                # per-text loop this replaced gave up: 4.14 s against 0.98 s
                # over 300 files of the benchmark code corpus with gpt2, for
                # the same ids.
                #
                # A batch is paired with its texts by position, which is the
                # batch API's contract and what the prose loop already relies
                # on. check_batch_pairing below verifies the count for both
                # corpora; a backend that returned the right number of
                # encodings in the wrong order is caught by neither.
                #
                # A failure here aborts. It used to fall back to the per-text
                # path under a warning asserting "The ids and offsets are the
                # same either way; only the speed changes", which nothing
                # verified and which a wrapper whose two methods disagree
                # falsifies: the run would then publish numbers measured
                # through a different encode path from the one it reported
                # using, which is the substitution this package refuses
                # everywhere else. No shipped wrapper can reach it.
                encoded = None
                if callable(encode_batch):
                    try:
                        encoded = [(ids, offsets)
                                   for ids, offsets in encode_batch(usable)]
                    except Exception as exc:
                        raise RuntimeError(
                            f"encode_batch_with_offsets failed for {tok_name!r} "
                            f"on the {corpus.name!r} corpus at label {lang!r}. "
                            "Encoding one text at a time instead would measure "
                            "this corpus through a different path from the one "
                            "the run reports, and nothing here can check that "
                            "the two agree. Fix the wrapper, or remove its "
                            "encode_batch_with_offsets so the per-text path is "
                            "the declared one."
                        ) from exc
                    for ids, _offsets in encoded:
                        if ids is None:
                            raise RuntimeError(
                                f"encode_batch_with_offsets returned None ids for "
                                f"{tok_name!r} on the {corpus.name!r} corpus at "
                                f"label {lang!r}. The per-text path checks this; "
                                "the batch path did not, and the None reached "
                                "TokenizedData as a type error naming neither "
                                "the tokenizer nor the method."
                            )
                if encoded is None:
                    encoded = []
                    for text in usable:
                        ids, offsets = None, None
                        if callable(encode_offsets):
                            # No fallback to ids-only here. A wrapper that has
                            # no offsets to give returns (ids, None) from the
                            # TokenizerWrapper default rather than raising, so
                            # an exception out of this call is a defect in the
                            # wrapper, not a tokenizer declining to supply
                            # offsets. Substituting an ids-only encoding for it
                            # published operator-isolation and AST numbers
                            # measured through a different path, and said so
                            # only at debug level; the AST metric then failed
                            # further down reporting that the wrapper had
                            # "returned none", which is not what happened.
                            try:
                                ids, offsets = encode_offsets(text)
                            except Exception as exc:
                                raise RuntimeError(
                                    f"encode_with_offsets raised for tokenizer "
                                    f"{tok_name!r} on a {lang!r} text of the "
                                    f"{corpus.name} corpus: {exc!r}. This "
                                    "method is expected to return (ids, None) "
                                    "when a tokenizer has no offsets, so "
                                    "raising is a defect in the wrapper. "
                                    "Encoding the text without offsets instead "
                                    "would measure it through a different path "
                                    "from the rest of the corpus. Text starts: "
                                    f"{text[:60]!r}"
                                ) from exc
                        if ids is None:
                            if callable(encode_offsets):
                                # encode_with_offsets ran and gave back no ids.
                                # Falling through to encode() would measure this
                                # text through a different path from the rest of
                                # the corpus, with no log line saying so.
                                raise RuntimeError(
                                    f"encode_with_offsets returned None ids for "
                                    f"{tok_name!r} on the {corpus.name!r} corpus "
                                    f"at label {lang!r}. Encoding it without "
                                    "offsets instead would measure one text "
                                    "through a different path from the others. "
                                    f"Text starts: {text[:60]!r}"
                                )
                            # No encode_with_offsets at all: this is the primary
                            # and only encode path for such a tokenizer, not a
                            # fallback from one that failed.
                            ids, offsets = encode(text), None
                            if ids is None:
                                # Checked because this is the only encode this
                                # tokenizer gets. Without it the None reached
                                # TokenizedData and failed as "tokens must be
                                # a list of integers", naming neither the
                                # tokenizer nor the method that produced it.
                                # An empty list is a different thing and is
                                # kept: a tokenizer that encodes a text to
                                # nothing has measured something.
                                raise RuntimeError(
                                    f"encode returned None for {tok_name!r} on "
                                    f"the {corpus.name!r} corpus at label "
                                    f"{lang!r}. This tokenizer has no "
                                    "encode_with_offsets, so encode is the "
                                    "only path it has and there is nothing to "
                                    f"fall back to. Text starts: {text[:60]!r}"
                                )
                        encoded.append((ids, offsets))
                check_batch_pairing(
                    tok_name, lang, usable, encoded, f"{corpus.name} corpus",
                )
                for text, (ids, offsets) in zip(usable, encoded):
                    items.append(
                        TokenizedData(
                            tokenizer_name=tok_name,
                            language=lang,
                            tokens=ids,
                            text=text,
                            offsets=offsets,
                        )
                    )
            out[tok_name] = items
        return out

    def validate_data(self) -> bool:
        """Validate the provided data."""
        try:
            tokenized_data = self.get_tokenized_data()
            tokenizer_names = self.get_tokenizer_names()
            
            # Check consistency
            if set(tokenized_data.keys()) != set(tokenizer_names):
                logger.error("Mismatch between tokenized_data keys and tokenizer_names")
                return False
            
            # Validate each tokenizer's data
            for tok_name, data_list in tokenized_data.items():
                if not data_list:
                    logger.warning(f"No data for tokenizer {tok_name}")
                    continue
                
                for data in data_list:
                    if not isinstance(data, TokenizedData):
                        logger.error(f"Invalid data type for {tok_name}: {type(data)}")
                        return False
                    
                    if data.tokenizer_name != tok_name:
                        logger.error(f"Tokenizer name mismatch: expected {tok_name}, got {data.tokenizer_name}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False

def check_batch_pairing(tokenizer_name: str, language: str, texts, encoded, corpus: str) -> None:
    """Raise unless a batch result has one entry per text it was given.

    Both corpora pair a batch with its texts by position, which is the batch
    API's contract. Nothing verifies it. A backend returning fewer results than
    it was given would otherwise be consumed by ``zip``, which stops at the
    shorter side: the trailing texts vanish and every remaining pairing is
    still correct, so the loss is invisible in the output.

    This checks the count and nothing else. A backend that returned the right
    number of encodings in a different order would attach one text's offsets to
    another text, and neither this nor anything else in the pipeline catches
    that.

    Args:
        tokenizer_name: named in the error, so a multi-tokenizer run says which.
        language: named in the error for the same reason.
        texts: what was handed to the backend.
        encoded: what the backend returned.
        corpus: which corpus this is, for the error message.

    Raises:
        ValueError: naming both counts.
    """
    if len(encoded) == len(texts):
        return
    raise ValueError(
        f"Tokenizer {tokenizer_name!r} returned {len(encoded)} encodings for "
        f"the {len(texts)} {language!r} texts of the {corpus}. Pairing them by "
        "position would attach one text's offsets to another text, so the "
        "metric would be computed against the wrong source. Only the count is "
        "checked: a backend that reordered a batch of the right length would "
        "not be caught here."
    )
