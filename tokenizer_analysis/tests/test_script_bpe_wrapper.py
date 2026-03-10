from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig
from tokenizer_analysis.core.tokenizer_wrapper import create_tokenizer_wrapper


def test_script_bpe_wrapper_loads_dense_ids(tmp_path):
    pretokenizer = get_pretokenizer("scriptenc_cb")
    text = "\n".join(
        [
            "hello world 0011",
            "hello tokenizer world 223344",
            "tokenizer world hello 556677",
        ]
    )
    corpus = PretokenizedCorpus.from_texts(
        "script_bpe_wrapper",
        texts=[text],
        pretokenizer=pretokenizer,
        base_path=str(tmp_path),
    )
    model = UnigramTrainer(
        pretokenizer,
        corpus,
        UnigramTrainerConfig(additional_vocab_size=8, num_workers=1, verbose=False),
    ).train()
    model_path = model.save(str(tmp_path / "unigram.json.gz"))

    wrapper = create_tokenizer_wrapper("test_script_bpe", {"class": "script_bpe", "path": model_path})
    ids = wrapper.encode(text)

    assert ids
    assert max(ids) < wrapper.get_vocab_size()
    assert wrapper.convert_ids_to_tokens(ids)
from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig
from tokenizer_analysis.core.tokenizer_wrapper import create_tokenizer_wrapper


def test_script_bpe_wrapper_loads_dense_ids(tmp_path):
    pretokenizer = get_pretokenizer("scriptenc_cb")
    text = "\n".join(
        [
            "hello world 0011",
            "hello tokenizer world 223344",
            "tokenizer world hello 556677",
        ]
    )
    corpus = PretokenizedCorpus.from_texts(
        "script_bpe_wrapper",
        texts=[text],
        pretokenizer=pretokenizer,
        base_path=str(tmp_path),
    )
    model = UnigramTrainer(
        pretokenizer,
        corpus,
        UnigramTrainerConfig(additional_vocab_size=8, num_workers=1, verbose=False),
    ).train()
    model_path = model.save(str(tmp_path / "unigram.json.gz"))

    wrapper = create_tokenizer_wrapper("test_script_bpe", {"class": "script_bpe", "path": model_path})
    ids = wrapper.encode(text)

    assert ids
    assert max(ids) < wrapper.get_vocab_size()
    assert wrapper.convert_ids_to_tokens(ids)
