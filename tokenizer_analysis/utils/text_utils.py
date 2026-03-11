"""
Shared text processing utilities for tokenizer analysis.

This module provides common text processing functions that are used across
multiple data loading and processing components to eliminate code duplication.
"""

import json
import os
import re
import random
from typing import List, Optional
from ..constants import (
    MIN_PARAGRAPH_LENGTH,
    MIN_LINE_LENGTH,
    MIN_SENTENCE_LENGTH,
    MIN_CONTENT_LENGTH,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RANDOM_SEED
)

BUILTIN_MATH_SAMPLES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "sample_data",
    "math_samples.json",
)


def load_math_data(path: str) -> List[str]:
    """Load math-rich text from a file.

    Supported formats:

    * ``.json`` -- expects ``{"texts": ["...", ...]}`` or a bare list
      of strings.
    * ``.txt`` / other -- reads non-empty lines.
    """
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            texts = [str(t) for t in data if str(t).strip()]
        elif isinstance(data, dict) and "texts" in data:
            texts = [str(t) for t in data["texts"] if str(t).strip()]
        else:
            raise ValueError(
                f"JSON math data must be a list of strings or "
                f'{{"texts": [...]}}; got {type(data).__name__}'
            )
    else:
        with open(path, "r", encoding="utf-8") as f:
            texts = [line.rstrip() for line in f if line.strip()]
    return texts


def _split_and_filter(text: str, split_pattern: Optional[str], min_length: int) -> List[str]:
    """Split *text* by *split_pattern* and keep fragments >= *min_length*.

    When *split_pattern* is ``None``, splits on single newlines (``'\\n'``).
    When it is a plain string (no regex metacharacters intended), it is used
    with ``str.split``.  Otherwise it is compiled as a regex via ``re.split``.
    """
    if not text or not text.strip():
        return []
    if split_pattern is None:
        raw = text.split('\n')
    elif split_pattern == '\n\n':
        if '\n\n' not in text:
            return []
        raw = text.split('\n\n')
    else:
        raw = re.split(split_pattern, text)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) >= min_length]


def split_into_paragraphs(text: str, min_length: int = MIN_PARAGRAPH_LENGTH) -> List[str]:
    """Split text into paragraphs with minimum length filtering."""
    return _split_and_filter(text, '\n\n', min_length)


def split_into_lines(text: str, min_length: int = MIN_LINE_LENGTH) -> List[str]:
    """Split text into lines with minimum length filtering."""
    return _split_and_filter(text, None, min_length)


def split_into_sentences(text: str, min_length: int = MIN_SENTENCE_LENGTH) -> List[str]:
    """Split text into sentences with minimum length filtering."""
    return _split_and_filter(text, r'[.!?]+\s+', min_length)


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, max_chunks: int = 100) -> List[str]:
    """
    Chunk text into smaller pieces of specified size.
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        max_chunks: Maximum number of chunks to create
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    for i in range(0, len(text), chunk_size):
        if len(chunks) >= max_chunks:
            break
        
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    
    return chunks


def extract_texts_with_fallback_strategies(content: str, max_texts: int) -> List[str]:
    """
    Extract texts from content using multiple fallback strategies.
    
    This function implements the same text extraction logic that was duplicated
    across multiple loading functions.
    
    Args:
        content: Raw text content to process
        max_texts: Maximum number of texts to extract
        
    Returns:
        List of extracted text strings
    """
    if not content or not content.strip():
        return []
    
    texts = []
    
    # Strategy 1: Split by double newlines (paragraph-like)
    if len(texts) < max_texts:
        paragraphs = split_into_paragraphs(content)
        for para in paragraphs:
            if len(texts) >= max_texts:
                break
            texts.append(para)
    
    # Strategy 2: Split by single newlines if we don't have enough texts
    if len(texts) < max_texts:
        lines = split_into_lines(content)
        for line in lines:
            if len(texts) >= max_texts:
                break
            if line not in texts:  # Avoid duplicates
                texts.append(line)
    
    # Strategy 3: Split by sentences if we still don't have enough
    if len(texts) < max_texts and len(texts) < 10:
        sentences = split_into_sentences(content)
        for sentence in sentences:
            if len(texts) >= max_texts:
                break
            if sentence not in texts:  # Avoid duplicates
                texts.append(sentence)
    
    # Strategy 4: If still no luck, chunk the text
    if len(texts) == 0 and len(content) > MIN_CONTENT_LENGTH:
        chunk_size = min(DEFAULT_CHUNK_SIZE, len(content) // max(1, max_texts))
        chunks = chunk_text(content, chunk_size, max_texts)
        texts.extend(chunks)
    
    return texts[:max_texts]



def normalize_text_for_processing(text: str) -> str:
    """
    Normalize text for consistent processing across the pipeline.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text ready for processing
    """
    if not text:
        return ""
    
    # Remove extra whitespace but preserve line breaks where meaningful
    normalized = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    normalized = re.sub(r'\n +', '\n', normalized)  # Remove spaces after newlines
    normalized = re.sub(r' +\n', '\n', normalized)  # Remove spaces before newlines
    
    return normalized.strip()


