"""
Multilingual data loader supporting JSON, Parquet, and text file formats.
Handles both directories and direct file paths specified in config.
"""

import os
import json
import glob
import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def load_multilingual_data(language_config_path: str, max_texts_per_language: int = 1000) -> Dict[str, List[str]]:
    """
    Load multilingual data from directories or files (JSON, Parquet, or text files).
    
    Args:
        language_config_path: Path to JSON config file with language->directory/file mappings
        max_texts_per_language: Maximum number of texts to load per language
        
    Returns:
        Dictionary mapping language codes to lists of text samples
    """
    with open(language_config_path, 'r') as f:
        language_config = json.load(f)
    
    language_texts = {}
    
    for lang_code, data_path in language_config['languages'].items():
        logger.info(f"Loading data for {lang_code} from {data_path}")
        
        try:
            texts = load_language_data(data_path, max_texts_per_language)
            if texts:
                language_texts[lang_code] = texts
                logger.info(f"✅ Loaded {len(texts)} texts for {lang_code}")
            else:
                logger.warning(f"❌ No texts found for {lang_code}")
        
        except Exception as e:
            logger.error(f"❌ Failed to load data for {lang_code}: {e}")
            continue
    
    logger.info(f"Successfully loaded data for {len(language_texts)} languages")
    return language_texts


def load_language_data(data_path: str, max_texts: int) -> List[str]:
    """
    Load text data from a directory or file (JSON, Parquet, or text file).
    
    Args:
        data_path: Directory containing data files OR path to a specific file
        max_texts: Maximum number of texts to load
        
    Returns:
        List of text samples
    """
    if not os.path.exists(data_path):
        logger.warning(f"Path does not exist: {data_path}")
        return []
    
    texts = []
    
    if os.path.isfile(data_path):
        # Handle single file
        logger.debug(f"Processing single file: {data_path}")
        texts = load_single_file(data_path, max_texts)
    elif os.path.isdir(data_path):
        # Handle directory - look for JSON, Parquet, and text files
        json_files = glob.glob(os.path.join(data_path, "*.json"))
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
        text_files = glob.glob(os.path.join(data_path, "*.txt"))
        
        # Process JSON files first
        for json_file in json_files:
            if len(texts) >= max_texts:
                break
            
            logger.debug(f"Processing JSON file: {json_file}")
            try:
                texts.extend(load_from_json(json_file, max_texts - len(texts)))
            except Exception as e:
                logger.error(f"Error processing JSON file {json_file}: {e}")
                continue
        
        # Process Parquet files if we need more texts
        for parquet_file in parquet_files:
            if len(texts) >= max_texts:
                break
            
            logger.debug(f"Processing Parquet file: {parquet_file}")
            try:
                texts.extend(load_from_parquet(parquet_file, max_texts - len(texts)))
            except Exception as e:
                logger.error(f"Error processing Parquet file {parquet_file}: {e}")
                continue
        
        # Process text files if we need more texts
        for text_file in text_files:
            if len(texts) >= max_texts:
                break
            
            logger.debug(f"Processing text file: {text_file}")
            try:
                texts.extend(load_from_text(text_file, max_texts - len(texts)))
            except Exception as e:
                logger.error(f"Error processing text file {text_file}: {e}")
                continue
    else:
        logger.warning(f"Path is neither file nor directory: {data_path}")
        return []
    
    return texts[:max_texts]


def load_from_json(json_file: str, max_texts: int) -> List[str]:
    """
    Load texts from a JSON file.
    
    Supports two formats:
    1. JSON Lines: each line is a JSON object with 'text' field
    2. Single JSON: array of objects with 'text' field
    
    Args:
        json_file: Path to JSON file
        max_texts: Maximum number of texts to load
        
    Returns:
        List of text samples
    """
    texts = []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            # Try to load as single JSON first
            data = json.load(f)
            if isinstance(data, list):
                # Array of objects
                for item in data:
                    if len(texts) >= max_texts:
                        break
                    if isinstance(item, dict) and 'text' in item:
                        text = item['text'].strip()
                        if text:
                            texts.append(text)
            elif isinstance(data, dict) and 'text' in data:
                # Single object
                text = data['text'].strip()
                if text:
                    texts.append(text)
        
        except json.JSONDecodeError:
            # Try JSON Lines format
            f.seek(0)
            for line_num, line in enumerate(f):
                if len(texts) >= max_texts:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and 'text' in data:
                        text = data['text'].strip()
                        if text:
                            texts.append(text)
                except json.JSONDecodeError as e:
                    logger.debug(f"Skipping invalid JSON line {line_num} in {json_file}: {e}")
                    continue
    
    return texts


def load_from_parquet(parquet_file: str, max_texts: int) -> List[str]:
    """
    Load texts from a Parquet file.
    
    Args:
        parquet_file: Path to Parquet file
        max_texts: Maximum number of texts to load
        
    Returns:
        List of text samples
    """
    texts = []
    
    try:
        # Read parquet file
        df = pd.read_parquet(parquet_file)
        
        # Look for text column (try common names)
        text_column = None
        for col_name in ['text', 'content', 'sentence', 'document', 'passage']:
            if col_name in df.columns:
                text_column = col_name
                break
        
        if text_column is None:
            # If no standard column found, use first string column
            string_columns = df.select_dtypes(include=['object', 'string']).columns
            if len(string_columns) > 0:
                text_column = string_columns[0]
                logger.info(f"Using column '{text_column}' as text column in {parquet_file}")
            else:
                logger.warning(f"No text column found in {parquet_file}")
                return []
        
        # Extract texts
        for idx, row in df.iterrows():
            if len(texts) >= max_texts:
                break
            
            text = str(row[text_column]).strip()
            if text and text != 'nan':
                texts.append(text)
    
    except Exception as e:
        logger.error(f"Error reading Parquet file {parquet_file}: {e}")
        return []
    
    return texts


def load_single_file(file_path: str, max_texts: int) -> List[str]:
    """
    Load texts from a single file, auto-detecting the format.
    
    Args:
        file_path: Path to the file
        max_texts: Maximum number of texts to load
        
    Returns:
        List of text samples
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.json':
        return load_from_json(file_path, max_texts)
    elif file_ext == '.parquet':
        return load_from_parquet(file_path, max_texts)
    elif file_ext in ['.txt', '.text']:
        return load_from_text(file_path, max_texts)
    else:
        # Try to detect format by attempting to load
        logger.info(f"Unknown file extension '{file_ext}', attempting auto-detection for {file_path}")
        
        # Try JSON first
        try:
            texts = load_from_json(file_path, max_texts)
            if texts:
                return texts
        except Exception:
            pass
        
        # Try text file
        try:
            texts = load_from_text(file_path, max_texts)
            if texts:
                return texts
        except Exception:
            pass
        
        logger.warning(f"Could not determine format for file: {file_path}")
        return []


def load_from_text(text_file: str, max_texts: int) -> List[str]:
    """
    Load texts from a plain text file.
    
    Supports multiple formats:
    1. One text per line
    2. Texts separated by double newlines
    3. Single large text (split into sentences)
    
    Args:
        text_file: Path to text file
        max_texts: Maximum number of texts to load
        
    Returns:
        List of text samples
    """
    texts = []
    
    try:
        with open(text_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read().strip()
            
            if not content:
                return []
            
            # Try different splitting strategies
            
            # Strategy 1: Split by double newlines (paragraph-like)
            if '\n\n' in content:
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    if len(texts) >= max_texts:
                        break
                    para = para.strip()
                    if para and len(para) > 10:  # Skip very short paragraphs
                        texts.append(para)
            
            # Strategy 2: Split by single newlines if we don't have enough texts
            if len(texts) < max_texts:
                lines = content.split('\n')
                for line in lines:
                    if len(texts) >= max_texts:
                        break
                    line = line.strip()
                    if line and len(line) > 10:  # Skip very short lines
                        texts.append(line)
            
            # Strategy 3: Split by sentences if we still don't have enough
            if len(texts) < max_texts and len(texts) < 10:
                # Use simple sentence splitting
                import re
                sentences = re.split(r'[.!?]+\s+', content)
                for sentence in sentences:
                    if len(texts) >= max_texts:
                        break
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 20:  # Skip very short sentences
                        texts.append(sentence)
            
            # Strategy 4: If still no luck, chunk the text
            if len(texts) == 0 and len(content) > 50:
                chunk_size = min(500, len(content) // max(1, max_texts))
                for i in range(0, len(content), chunk_size):
                    if len(texts) >= max_texts:
                        break
                    chunk = content[i:i + chunk_size].strip()
                    if chunk:
                        texts.append(chunk)
    
    except Exception as e:
        logger.error(f"Error reading text file {text_file}: {e}")
        return []
    
    return texts


def sample_texts(texts: List[str], max_samples: int, random_seed: int = 42) -> List[str]:
    """
    Sample texts randomly if there are more than max_samples.
    
    Args:
        texts: List of all texts
        max_samples: Maximum number of texts to return
        random_seed: Random seed for reproducibility
        
    Returns:
        Sampled list of texts
    """
    if len(texts) <= max_samples:
        return texts
    
    import random
    random.seed(random_seed)
    return random.sample(texts, max_samples)