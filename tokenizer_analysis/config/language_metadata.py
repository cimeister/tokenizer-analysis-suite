"""
Language metadata and grouping configuration for tokenizer analysis.

This module provides functionality to load and manage language metadata,
including analytical groupings by script family and resource level.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Anchor for relative `data_path` values in a language config. This is the
# directory that holds the `tokenizer_analysis` package, which in a source
# checkout is the repository root, so `parallel/eng_Latn.txt` names the same
# file whatever directory the process was started from. Resolving against the
# process working directory instead meant the same absolute --language-config
# loaded 5 languages from the repository root and 0 from configs/, logged as
# warnings, exit 0. An absolute data_path is used unchanged.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_corpus_path(data_path: str) -> str:
    """Return *data_path* anchored at PACKAGE_ROOT when it is relative."""
    path = Path(data_path)
    if path.is_absolute():
        return str(path)
    return str(PACKAGE_ROOT / path)


class LanguageMetadata:
    """
    Manages language metadata and analytical groupings.
    
    Provides access to language information and groupings for
    script families, resource levels, and other analytical dimensions.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize LanguageMetadata from configuration file.
        
        Args:
            config_path: Path to the language metadata JSON configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.languages = self.config.get('languages', {})
        self.analysis_groups = self.config.get('analysis_groups', {})
        
        # Create reverse mappings for efficient lookups
        self._build_reverse_mappings()
        
        # Validate configuration consistency
        self._validate_configuration()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Language metadata configuration file not found: {config_path}")
        except IsADirectoryError:
            # A comment in cli/run_analysis.py used to claim directories were
            # supported here; no code path ever read one, and the unhandled
            # IsADirectoryError named the path but not what was expected of it.
            raise IsADirectoryError(
                f"Language metadata configuration must be a JSON file, but "
                f"{config_path} is a directory. Name the file inside it."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in language metadata configuration: {e}")
    
    # Group-type names accepted in `analysis_groups`, most-preferred first.
    # Every config shipped in configs/ uses the singular form, but the accessors
    # here originally read only the plural, so get_script_family() returned
    # 'Unknown' for every language and get_script_families() returned []. Accept
    # both rather than breaking existing configs either way.
    _SCRIPT_GROUP_KEYS = ('script_family', 'script_families')
    _RESOURCE_GROUP_KEYS = ('resource_level', 'resource_levels')

    def _group(self, keys) -> Dict[str, List[str]]:
        """Return the first present group mapping among *keys*, else empty."""
        for key in keys:
            group = self.analysis_groups.get(key)
            if group:
                return group
        return {}

    def _build_reverse_mappings(self):
        """Build reverse mappings from language codes to groups."""
        self.lang_to_script_family = {}
        self.lang_to_resource_level = {}

        # Built ONLY from analysis_groups, never inferred from the language code.
        for script_family, languages in self._group(self._SCRIPT_GROUP_KEYS).items():
            for lang in languages:
                self.lang_to_script_family[lang] = script_family

        for resource_level, languages in self._group(self._RESOURCE_GROUP_KEYS).items():
            for lang in languages:
                self.lang_to_resource_level[lang] = resource_level
    
    def _validate_configuration(self):
        """Ensure all languages in analysis_groups exist in languages section."""
        all_analysis_languages = set()
        
        for group_type, groups in self.analysis_groups.items():
            if isinstance(groups, dict):
                for group_name, languages in groups.items():
                    if isinstance(languages, list):
                        all_analysis_languages.update(languages)
        
        # Check that all languages in analysis_groups exist in languages section
        missing_languages = all_analysis_languages - set(self.languages.keys())
        if missing_languages:
            raise ValueError(f"Languages in analysis_groups not found in languages section: {missing_languages}")
    
    # Language information methods
    def get_language_info(self, language_code: str) -> Dict[str, Any]:
        """Get full language information, normalized to the dict form."""
        info = self.languages.get(language_code, {})
        if isinstance(info, str):
            return {'data_path': info}
        return info if isinstance(info, dict) else {}

    def get_language_name(self, language_code: str) -> str:
        """Get the display name for a language, falling back to its code."""
        return self.get_language_info(language_code).get('name', language_code)
    
    def get_available_languages(self) -> List[str]:
        """Get list of all available language codes."""
        return list(self.languages.keys())
    
    # Script family methods
    def get_script_families(self) -> List[str]:
        """Get list of all script families."""
        return list(self._group(self._SCRIPT_GROUP_KEYS).keys())
    
    def get_languages_by_script_family(self, script_family: str) -> List[str]:
        """Get languages belonging to a specific script family."""
        return self._group(self._SCRIPT_GROUP_KEYS).get(script_family, [])
    
    def get_script_family(self, language_code: str) -> str:
        """Get script family for a language from analysis_groups."""
        return self.lang_to_script_family.get(language_code, 'Unknown')
    
    # Resource level methods
    def get_resource_levels(self) -> List[str]:
        """Get list of all resource levels."""
        return list(self._group(self._RESOURCE_GROUP_KEYS).keys())
    
    def get_languages_by_resource_level(self, resource_level: str) -> List[str]:
        """Get languages belonging to a specific resource level."""
        return self._group(self._RESOURCE_GROUP_KEYS).get(resource_level, [])
    
    def get_resource_level(self, language_code: str) -> str:
        """Get resource level for a language from analysis_groups."""
        return self.lang_to_resource_level.get(language_code, 'Unknown')
    
    # Group analysis methods
    def get_all_analysis_groups(self) -> Dict[str, Dict[str, List[str]]]:
        """Get all analysis groups."""
        return self.analysis_groups
    
    def get_group_type_names(self) -> List[str]:
        """Get list of all group types (e.g., 'script_families', 'resource_levels')."""
        return list(self.analysis_groups.keys())
    
    def filter_languages_by_availability(self, language_codes: List[str]) -> List[str]:
        """Filter language codes to only include those available in the configuration."""
        return [lang for lang in language_codes if lang in self.languages]
    
    # Statistics methods
    def get_group_statistics(self) -> Dict[str, Any]:
        """Get statistics about language groups."""
        stats = {
            'total_languages': len(self.languages),
            'script_families': {},
            'resource_levels': {}
        }
        
        # Script family statistics
        for script_family, languages in self._group(self._SCRIPT_GROUP_KEYS).items():
            stats['script_families'][script_family] = {
                'count': len(languages),
                'languages': languages
            }
        
        # Resource level statistics
        for resource_level, languages in self._group(self._RESOURCE_GROUP_KEYS).items():
            stats['resource_levels'][resource_level] = {
                'count': len(languages),
                'languages': languages
            }
        
        return stats
    
    # Data path methods
    @staticmethod
    def _data_path_of(lang_info: Any) -> Optional[str]:
        """Extract the corpus path from one `languages` entry.

        Two shapes are accepted. The full form is a dict carrying `data_path`
        alongside `name` and `iso_code`. The short form maps the code straight
        to a path string, `{"en": "/path/to/data"}`, which the README has always
        documented; before this it raised `AttributeError: 'str' object has no
        attribute 'get'`, so the documented form did not work.

        A relative path is anchored at PACKAGE_ROOT, so the same config names
        the same files from any working directory. See resolve_corpus_path.
        """
        if isinstance(lang_info, str):
            raw = lang_info or None
        elif isinstance(lang_info, dict):
            raw = lang_info.get('data_path')
        else:
            raw = None
        return resolve_corpus_path(raw) if raw else None

    def get_data_path(self, language_code: str) -> Optional[str]:
        """Get data path for a specific language."""
        return self._data_path_of(self.languages.get(language_code))

    def get_language_paths(self) -> Dict[str, str]:
        """Get all language code to data path mappings."""
        paths = {}
        for lang_code, lang_info in self.languages.items():
            data_path = self._data_path_of(lang_info)
            if data_path:
                paths[lang_code] = data_path
        return paths


# Nothing in the package reads this. LanguageMetadata requires a config file
# path and raises FileNotFoundError when the file is absent; it never falls
# back to this dict or to any other default. It is kept as a written record of
# the two top-level keys a language metadata config has. Note that the group
# names below are the plural forms, which _SCRIPT_GROUP_KEYS and
# _RESOURCE_GROUP_KEYS accept but do not prefer; the configs in configs/ use
# the singular script_family and resource_level.
DEFAULT_LANGUAGE_METADATA = {
    "languages": {},
    "analysis_groups": {
        "script_families": {},
        "resource_levels": {}
    }
}