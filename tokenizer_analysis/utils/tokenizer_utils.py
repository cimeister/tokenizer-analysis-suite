"""
Utility functions for tokenizer analysis.
Attempts to import from parent codebase first, falls back to standalone versions.
"""

import math
import json
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

from tokenizers import Tokenizer
from tokenizers.models import Unigram, BPE  
from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Sequence
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

_ORIGINAL_FUNCTIONS_AVAILABLE = False
_original_encode_text = None
_original_load_tokenizer_from_config = None
# Try to import original functions using multiple strategies
try:
    from unimixlm.code.utils import encode_text_minimal as _original_encode_text
    from unimixlm.code.utils import load_tokenizer_from_config as _original_load_tokenizer_from_config
    _ORIGINAL_FUNCTIONS_AVAILABLE = True
except ImportError:
    logger.info("Unable to pull original encoding functions, will use new definitions")

def _as_batch(txt):
    """Return (list version, was_single_flag)."""
    if isinstance(txt, (str, tuple)):
        return [txt], True
    if isinstance(txt, list):
        return txt, False
    raise TypeError(type(txt))


def _apply_trunc_pad(
    ids_batch, toks_batch, offs_batch,
    *,
    truncation: bool, max_length: Optional[int],
    padding: Optional[str], pad_id: int, pad_tok: str,
    return_attention_mask: bool,
):
    """Apply truncation and padding to token sequences."""
    out_ids, out_toks, out_offs = [], [], []
    masks = [] if return_attention_mask else None

    for ids, toks, offs in zip(ids_batch, toks_batch,
                               offs_batch or [None] * len(ids_batch)):
        # Truncation
        if truncation and max_length:
            ids, toks = ids[:max_length], toks[:max_length]
            if offs is not None:
                offs = offs[:max_length]

        # Padding
        if padding == "max_length" and max_length:
            pad_len = max_length - len(ids)
            ids += [pad_id] * pad_len
            toks += [pad_tok] * pad_len
            if offs is not None:
                offs += [(0, 0)] * pad_len
        else:
            pad_len = 0

        # Attention mask
        if masks is not None:
            real_len = len(ids) - pad_len
            masks.append([1] * real_len + [0] * pad_len)

        out_ids.append(ids)
        out_toks.append(toks)
        if offs is not None:
            out_offs.append(offs)

    return (
        out_ids,
        out_toks,
        masks,
        out_offs if offs_batch is not None else None,
    )


def encode_text(
    tokenizer: Any,
    text: Union[str, List[str], Tuple[str, str], List[Tuple[str, str]]],
    *,
    add_special_tokens: bool = False,
    padding: Optional[str] = None,
    truncation: Union[bool, str] = False,
    max_length: Optional[int] = None,
    return_attention_mask: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Encode text using various tokenizer types.
    
    Tries to use original implementation first, falls back to simplified version.
    """
    if isinstance(tokenizer, AutoTokenizer):
        return tokenizer(text, add_special_tokens=add_special_tokens)
    elif isinstance(tokenizer, Tokenizer):
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if _ORIGINAL_FUNCTIONS_AVAILABLE:
        try:
            return _original_encode_text(
                tokenizer, text,
                add_special_tokens=add_special_tokens
            )
        except Exception as e:
            logger.warning(f"Original encode_text failed ({e}), using fallback")
    
    # Fallback implementation
    batch_txt, was_single = _as_batch(text)
    
    # Try to get pad token
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token:
        pad_token = tokenizer.pad_token
    else:
        pad_token = "<pad>"
    
    # Get pad_id
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        pad_id = vocab.get(pad_token, 0)
    else:
        pad_id = 0

    # Use tokenizer's encode_batch if available
    if hasattr(tokenizer, "encode_batch"):
        encodings = tokenizer.encode_batch(batch_txt, add_special_tokens=add_special_tokens)
        logger.debug("Using batch tokenization...")

        ids_batch = [enc.ids for enc in encodings]
        toks_batch = [enc.tokens for enc in encodings]
        offsets_b = [enc.offsets for enc in encodings] if hasattr(encodings[0], 'offsets') else None

        ids_batch, toks_batch, masks, offsets_b = _apply_trunc_pad(
            ids_batch, toks_batch, offsets_b,
            truncation=truncation, max_length=max_length,
            padding=padding, pad_id=pad_id,
            pad_tok=pad_token,
            return_attention_mask=return_attention_mask,
        )

        result = {"input_ids": ids_batch, "tokens": toks_batch}
        if masks is not None:
            result["attention_mask"] = masks
        if offsets_b is not None:
            result["offset_mapping"] = offsets_b

    else:
        # Fallback to HuggingFace tokenizer
        logger.debug("Using HuggingFace tokenization...")
        hf_kwargs = dict(
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_attention_mask=return_attention_mask,
        )
        enc = tokenizer(batch_txt, **hf_kwargs)
        result = enc.to_dict() if hasattr(enc, "to_dict") else dict(enc)
        if "tokens" not in result and hasattr(tokenizer, "convert_ids_to_tokens"):
            result["tokens"] = [tokenizer.convert_ids_to_tokens(ids) for ids in result["input_ids"]]

    # Unwrap single example
    if was_single:
        for k, v in result.items():
            if isinstance(v, list) and len(v) == 1:
                result[k] = v[0]
    
    return result


def load_tokenizer_from_config(config):
    """
    Load tokenizer from configuration.
    
    Tries to use original implementation first, falls back to simplified version.
    """
    if _ORIGINAL_FUNCTIONS_AVAILABLE:
        try:
            return _original_load_tokenizer_from_config(config)
        except Exception as e:
            logger.warning(f"Original load_tokenizer_from_config failed ({e}), using fallback")
    
    # Fallback implementation
    tokenizer_class = config.get('class', 'standard')
    
    if tokenizer_class == "custom_bpe":
        # Custom BPE tokenizer loading
        vocab_file = os.path.join(config['path'], "vocab.json")
        merges_file = os.path.join(config['path'], "merges.txt")
    
        # Load vocab and merges from files
        with open(vocab_file, "r", encoding="utf-8") as vf:
            vocab = json.load(vf)
        with open(merges_file, "r", encoding="utf-8") as mf:
            merges = [tuple(line.strip().split()) for line in mf if not line.startswith("#")]
    
        # Initialize tokenizer
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges))
    
        # Set pre-tokenizer
        tokenizer.pre_tokenizer = Sequence([Whitespace(), ByteLevel(use_regex=False)])
    
        # Set special tokens
        tokenizer.add_special_tokens(["<s>", "</s>", "<unk>", "<pad>"])
        tokenizer.model.unk_token = "<unk>"
    
        # Set post-processor
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", tokenizer.token_to_id("<s>")),
                ("</s>", tokenizer.token_to_id("</s>")),
            ]
        )
        
        return tokenizer
    
    else:
        path = config['path']
        
        # Strategy 1: If path points to a JSON file, use Tokenizer.from_file
        if path.endswith('.json') or os.path.isfile(path):
            try:
                logger.info(f"Loading tokenizer from file: {path}")
                tokenizer = Tokenizer.from_file(path)
                return tokenizer
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from file {path}: {e}")
        
        # Strategy 2: Try loading as HuggingFace tokenizer (directory or model name)
        try:
            logger.info(f"Loading tokenizer from HuggingFace: {path}")
            tokenizer = AutoTokenizer.from_pretrained(path)
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace tokenizer from {path}: {e}")
    
        # Strategy 3: If path is a directory, look for tokenizer files
        if os.path.isdir(path):
            # Look for common tokenizer file names
            for filename in ['tokenizer.json', 'vocab.json', 'merges.txt']:
                file_path = os.path.join(path, filename)
                if os.path.exists(file_path):
                    try:
                        if filename == 'tokenizer.json':
                            logger.info(f"Loading tokenizer from {file_path}")
                            return Tokenizer.from_file(file_path)
                        elif filename == 'vocab.json' and os.path.exists(os.path.join(path, 'merges.txt')):
                            # Load as BPE tokenizer
                            logger.info(f"Loading BPE tokenizer from directory: {path}")
                            return _load_bpe_from_directory(path)
                    except Exception as e:
                        logger.warning(f"Failed to load tokenizer from {file_path}: {e}")
                        continue
        
        raise ValueError(f"Could not load tokenizer from {path}.")


def _load_bpe_from_directory(directory_path):
    """Helper function to load BPE tokenizer from directory with vocab.json and merges.txt"""
    vocab_file = os.path.join(directory_path, "vocab.json")
    merges_file = os.path.join(directory_path, "merges.txt")
    
    # Load vocab and merges from files
    with open(vocab_file, "r", encoding="utf-8") as vf:
        vocab = json.load(vf)
    with open(merges_file, "r", encoding="utf-8") as mf:
        merges = [tuple(line.strip().split()) for line in mf if not line.startswith("#")]

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges))

    # Set pre-tokenizer
    tokenizer.pre_tokenizer = Sequence([Whitespace(), ByteLevel(use_regex=False)])

    # Set special tokens
    tokenizer.add_special_tokens(["<s>", "</s>", "<unk>", "<pad>"])
    tokenizer.model.unk_token = "<unk>"

    # Set post-processor
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ]
    )
    
    return tokenizer


def setup_environment():
    """Setup environment for tokenizer analysis."""
    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
