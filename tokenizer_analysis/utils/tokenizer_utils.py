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


def detect_unk_token_id(tokenizer_wrapper) -> Optional[int]:
    """
    Detect UNK token ID using multiple strategies.

    Args:
        tokenizer_wrapper: TokenizerWrapper instance

    Returns:
        UNK token ID or None if not found
    """
    # Strategy 1: Direct method from wrapper
    if hasattr(tokenizer_wrapper, 'get_unk_token_id'):
        unk_id = tokenizer_wrapper.get_unk_token_id()
        if unk_id is not None:
            return unk_id

    # Strategy 2: Check vocabulary for common UNK tokens
    vocab = tokenizer_wrapper.get_vocab()
    if vocab:
        unk_candidates = ['<unk>', '[UNK]', '<UNK>', 'unk', 'UNK', '⁇', '<|endoftext|>']
        for candidate in unk_candidates:
            if candidate in vocab:
                return vocab[candidate]

    # Strategy 3: Check underlying tokenizer
    underlying = tokenizer_wrapper.get_underlying_tokenizer()
    if underlying:
        # HuggingFace transformers tokenizer
        if hasattr(underlying, 'unk_token_id') and underlying.unk_token_id is not None:
            return underlying.unk_token_id

        # HuggingFace tokenizers library
        if hasattr(underlying, 'token_to_id') and hasattr(underlying, 'get_vocab'):
            # Try common UNK token strings
            unk_candidates = ['<unk>', '[UNK]', '<UNK>', 'unk', 'UNK', '⁇']
            for candidate in unk_candidates:
                try:
                    unk_id = underlying.token_to_id(candidate)
                    if unk_id is not None:
                        return unk_id
                except:
                    continue

    # Strategy 4: Check for typical UNK behavior (ID 0, 1, 2, 3)
    # This is a last resort and might not be reliable
    for potential_unk_id in [0, 1, 2, 3]:
        if potential_unk_id < tokenizer_wrapper.get_vocab_size():
            # We can't easily test if this is actually UNK without encoding,
            # so we'll skip this strategy for now
            pass

    logger.debug(f"No UNK token found for tokenizer {tokenizer_wrapper.get_name()}")
    return None

def load_tokenizer_from_config(config, name: str = "tokenizer"):
    """
    Load tokenizer wrapper from configuration (here for backwards compatibility)
    
    Args:
        config: Configuration dictionary
        name: Tokenizer name (for the wrapper)
        
    Returns:
        TokenizerWrapper instance
    """
    from ..core.tokenizer_wrapper import create_tokenizer_wrapper
    logger.warning("Deprecated function; Use `create_tokenizer_wrapper` in core.tokenizer_wrapper instead")
    return create_tokenizer_wrapper(name, config)


def _load_huggingface_tokenizer(config):
    """
    Internal function to load raw HuggingFace tokenizer from configuration.
    
    This function is used by the HuggingFaceTokenizer wrapper.
    Tries to use original implementation first, falls back to simplified version.
    """
 
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
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
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


def _load_custom_bpe_from_directory(config):
    """Helper function to load BPE tokenizer from directory with vocab.json and merges.txt"""
    directory_path = config['path']
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
