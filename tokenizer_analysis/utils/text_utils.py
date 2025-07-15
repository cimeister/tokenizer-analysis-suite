"""
Shared text processing utilities for tokenizer analysis.

This module provides common text processing functions that are used across
multiple data loading and processing components to eliminate code duplication.
"""

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


def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing extra whitespace.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text with normalized whitespace
    """
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    return " ".join(text.split())


def split_into_paragraphs(text: str, min_length: int = MIN_PARAGRAPH_LENGTH) -> List[str]:
    """
    Split text into paragraphs with minimum length filtering.
    
    Args:
        text: Input text to split
        min_length: Minimum paragraph length to include
        
    Returns:
        List of paragraph strings meeting minimum length requirement
    """
    if not text or not text.strip():
        return []
    
    paragraphs = []
    
    # Split by double newlines (paragraph-like)
    if '\n\n' in text:
        raw_paragraphs = text.split('\n\n')
        for para in raw_paragraphs:
            para = para.strip()
            if para and len(para) >= min_length:
                paragraphs.append(para)
    
    return paragraphs


def split_into_lines(text: str, min_length: int = MIN_LINE_LENGTH) -> List[str]:
    """
    Split text into lines with minimum length filtering.
    
    Args:
        text: Input text to split
        min_length: Minimum line length to include
        
    Returns:
        List of line strings meeting minimum length requirement
    """
    if not text or not text.strip():
        return []
    
    lines = []
    raw_lines = text.split('\n')
    
    for line in raw_lines:
        line = line.strip()
        if line and len(line) >= min_length:
            lines.append(line)
    
    return lines


def split_into_sentences(text: str, min_length: int = MIN_SENTENCE_LENGTH) -> List[str]:
    """
    Split text into sentences with minimum length filtering.
    
    Args:
        text: Input text to split
        min_length: Minimum sentence length to include
        
    Returns:
        List of sentence strings meeting minimum length requirement
    """
    if not text or not text.strip():
        return []
    
    sentences = []
    
    # Use simple sentence splitting based on punctuation
    raw_sentences = re.split(r'[.!?]+\s+', text)
    
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) >= min_length:
            sentences.append(sentence)
    
    return sentences


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


