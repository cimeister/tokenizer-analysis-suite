"""
Multilingual data loader supporting JSON, Parquet, and text file formats.
Handles both directories and direct file paths specified in config.
"""

import os
import json
import glob
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
from ..config.language_metadata import LanguageMetadata
from ..constants import (
    DEFAULT_MAX_TEXTS_PER_LANGUAGE,
    TEXT_COLUMN_NAMES,
    DEFAULT_ENCODING,
    ERROR_HANDLING,
    JSON_EXTENSIONS,
    PARQUET_EXTENSIONS,
    TEXT_EXTENSIONS,
)
from ..utils.text_utils import (
    extract_texts_with_fallback_strategies,
    normalize_text_for_processing
)

logger = logging.getLogger(__name__)

class ParquetEngineMissing(RuntimeError):
    """No parquet engine is installed, so a parquet corpus cannot be read.

    A distinct class because the loaders wrap parquet reads in several layers of
    ``except Exception`` that log and continue. Those handlers exist to skip one
    unreadable file among many, which is reasonable, but a missing engine makes
    EVERY parquet file unreadable, so skipping them all yields an empty corpus
    and a run that completes normally. Each handler re-raises this one type.
    """


def _reraise_if_no_parquet_engine(exc: Exception, path: str) -> None:
    """Turn a missing parquet engine into a loud error naming the extra.

    pandas raises an ImportError that already names pyarrow, but every parquet
    call site caught it and returned an empty list, so a user without the
    optional extra got a run that completed with silently zero data for every
    parquet source. A corpus that failed to load is not an empty corpus.
    """
    if isinstance(exc, ParquetEngineMissing):
        raise exc
    if isinstance(exc, ImportError) or 'usable engine' in str(exc).lower():
        raise ParquetEngineMissing(
            f"Cannot read the parquet file {path!r}: no parquet engine is "
            "installed. Install the optional extra with "
            "`uv sync --extra parquet` (or `pip install pyarrow`). Reading "
            "zero rows here would silently shrink the corpus, so this is an "
            "error rather than an empty result."
        ) from exc



def load_multilingual_data(language_metadata: LanguageMetadata, 
                          max_texts_per_language: int = DEFAULT_MAX_TEXTS_PER_LANGUAGE,
                          filter_by_group: Optional[Tuple[str, str]] = None) -> Dict[str, List[str]]:
    """
    Load multilingual data using LanguageMetadata configuration.
    
    Args:
        language_metadata: LanguageMetadata instance with language configurations
        max_texts_per_language: Maximum number of texts to load per language
        filter_by_group: Optional tuple of (group_type, group_name) to filter languages
        
    Returns:
        Dictionary mapping language codes to lists of text samples
    """
    # Get language paths
    if filter_by_group:
        group_type, group_name = filter_by_group
        if group_type == 'script_families':
            target_languages = language_metadata.get_languages_by_script_family(group_name)
        elif group_type == 'resource_levels':
            target_languages = language_metadata.get_languages_by_resource_level(group_name)
        else:
            # Support any group type defined in analysis_groups
            all_groups = language_metadata.get_all_analysis_groups()
            groups = all_groups.get(group_type, {})
            target_languages = groups.get(group_name, [])

        # An unknown group name used to reach the loader as an empty language
        # list, and the run then failed several steps later with "No valid
        # language texts loaded", which does not say that the name was the
        # problem or which names the config defines.
        if not target_languages:
            known = language_metadata.get_all_analysis_groups().get(group_type, {})
            if known:
                raise ValueError(
                    f"No language in the config has {group_type}={group_name}. "
                    f"The config defines: {', '.join(sorted(known))}."
                )
            raise ValueError(
                f"The config defines no {group_type} groups, so "
                f"{group_type}={group_name} selects nothing. Groups present: "
                f"{', '.join(sorted(language_metadata.get_all_analysis_groups())) or 'none'}."
            )

        language_paths = {}
        no_path = []
        for lang_code in target_languages:
            data_path = language_metadata.get_data_path(lang_code)
            if data_path:
                language_paths[lang_code] = data_path
            else:
                no_path.append(lang_code)
        if no_path:
            raise ValueError(
                f"{len(no_path)} language(s) in {group_type}={group_name} have "
                f"no data path in the config: {', '.join(sorted(no_path))}. "
                "Dropping them would change what the cross-language metrics are "
                "computed over without recording it."
            )
    else:
        # Load all languages
        language_paths = language_metadata.get_language_paths()
    
    language_texts = {}
    
    # A language that was asked for and could not be loaded aborts the run.
    # Continuing produced a complete-looking results file silently missing that
    # language: on a 13-language config a single typo'd path yielded a
    # 12-language result that read as finished, and every cross-language metric
    # (Gini, the utilization CoV) was then computed over a different set than
    # the one requested.
    failures: Dict[str, str] = {}

    for lang_code, data_path in language_paths.items():
        lang_name = language_metadata.get_language_name(lang_code)
        logger.info(f"Loading data for {lang_name} ({lang_code}) from {data_path}")

        try:
            texts = load_language_data(data_path, max_texts_per_language)
        except ParquetEngineMissing:
            # Not a property of this language's path: no parquet source in the
            # config can be read until the extra is installed, so listing it as
            # one language's failure buries a message about the environment
            # inside a per-language report. Propagate it; the CLI prints it
            # without a traceback.
            raise
        except Exception as e:
            failures[lang_code] = f"{type(e).__name__}: {e}"
            continue

        if texts:
            language_texts[lang_code] = texts
            logger.info(f"Loaded {len(texts)} texts for {lang_name} ({lang_code})")
        else:
            failures[lang_code] = f"no texts read from {data_path}"

    if failures:
        detail = "; ".join(
            f"{lang} ({language_metadata.get_language_name(lang)}): {reason}"
            for lang, reason in sorted(failures.items())
        )
        raise ValueError(
            f"Could not load {len(failures)} of {len(language_paths)} requested "
            f"language(s): {detail}. Every language named in the config must "
            "load, because dropping one silently changes what the "
            "cross-language metrics are computed over. Fix the path, or remove "
            "the language from the config."
        )

    if filter_by_group:
        logger.info(f"Successfully loaded data for {len(language_texts)} languages in {group_type}={group_name}")
    else:
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
        # Sorted, because max_texts truncates: with bare glob.glob the files
        # arrive in filesystem order, so *which* texts get analyzed would depend
        # on the directory layout and the same corpus could give different
        # numbers on two machines.
        json_files = sorted(glob.glob(os.path.join(data_path, "*.json")))
        parquet_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
        text_files = sorted(glob.glob(os.path.join(data_path, "*.txt")))
        
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
            except ParquetEngineMissing:
                raise
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
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as exc:
            _reraise_if_no_parquet_engine(exc, parquet_file)
            raise
        
        # Look for text column (try common names)
        text_column = None
        for col_name in TEXT_COLUMN_NAMES:
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
    
    except ParquetEngineMissing:
        raise
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
    
    if file_ext in JSON_EXTENSIONS:
        return load_from_json(file_path, max_texts)
    elif file_ext in PARQUET_EXTENSIONS:
        return load_from_parquet(file_path, max_texts)
    elif file_ext in TEXT_EXTENSIONS:
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
        with open(text_file, 'r', encoding=DEFAULT_ENCODING, errors=ERROR_HANDLING) as f:
            content = f.read().strip()
            
            if not content:
                return []
            
            # Normalize content for consistent processing
            content = normalize_text_for_processing(content)
            
            # Use shared text extraction logic
            texts = extract_texts_with_fallback_strategies(content, max_texts)
    
    except Exception as e:
        logger.error(f"Error reading text file {text_file}: {e}")
        return []
    
    return texts