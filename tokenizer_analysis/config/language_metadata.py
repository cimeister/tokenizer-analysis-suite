"""
Language metadata and grouping configuration for tokenizer analysis.

This module provides functionality to load and manage language metadata,
including analytical groupings by script family and resource level.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


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
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in language metadata configuration: {e}")
    
    def _build_reverse_mappings(self):
        """Build reverse mappings from language codes to groups."""
        self.lang_to_script_family = {}
        self.lang_to_resource_level = {}
        
        # Build script family mappings - ONLY from analysis_groups
        for script_family, languages in self.analysis_groups.get('script_families', {}).items():
            for lang in languages:
                self.lang_to_script_family[lang] = script_family
        
        # Build resource level mappings - ONLY from analysis_groups
        for resource_level, languages in self.analysis_groups.get('resource_levels', {}).items():
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
        """Get full language information."""
        return self.languages.get(language_code, {})
    
    def get_language_name(self, language_code: str) -> str:
        """Get the display name for a language."""
        return self.languages.get(language_code, {}).get('name', language_code)
    
    def get_available_languages(self) -> List[str]:
        """Get list of all available language codes."""
        return list(self.languages.keys())
    
    # Script family methods
    def get_script_families(self) -> List[str]:
        """Get list of all script families."""
        return list(self.analysis_groups.get('script_families', {}).keys())
    
    def get_languages_by_script_family(self, script_family: str) -> List[str]:
        """Get languages belonging to a specific script family."""
        return self.analysis_groups.get('script_families', {}).get(script_family, [])
    
    def get_script_family(self, language_code: str) -> str:
        """Get script family for a language from analysis_groups."""
        return self.lang_to_script_family.get(language_code, 'Unknown')
    
    # Resource level methods
    def get_resource_levels(self) -> List[str]:
        """Get list of all resource levels."""
        return list(self.analysis_groups.get('resource_levels', {}).keys())
    
    def get_languages_by_resource_level(self, resource_level: str) -> List[str]:
        """Get languages belonging to a specific resource level."""
        return self.analysis_groups.get('resource_levels', {}).get(resource_level, [])
    
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
        for script_family, languages in self.analysis_groups.get('script_families', {}).items():
            stats['script_families'][script_family] = {
                'count': len(languages),
                'languages': languages
            }
        
        # Resource level statistics
        for resource_level, languages in self.analysis_groups.get('resource_levels', {}).items():
            stats['resource_levels'][resource_level] = {
                'count': len(languages),
                'languages': languages
            }
        
        return stats
    
    # Data path methods
    def get_data_path(self, language_code: str) -> Optional[str]:
        """Get data path for a specific language."""
        lang_info = self.languages.get(language_code, {})
        return lang_info.get('data_path')
    
    def get_language_paths(self) -> Dict[str, str]:
        """Get all language code to data path mappings."""
        paths = {}
        for lang_code, lang_info in self.languages.items():
            data_path = lang_info.get('data_path')
            if data_path:
                paths[lang_code] = data_path
        return paths


def load_language_metadata(config_path: str) -> LanguageMetadata:
    """
    Convenience function to load language metadata.
    
    Args:
        config_path: Path to the language metadata configuration file
        
    Returns:
        LanguageMetadata instance
    """
    return LanguageMetadata(config_path)


# Default configuration for backward compatibility
DEFAULT_LANGUAGE_METADATA = {
    "languages": {},
    "analysis_groups": {
        "script_families": {},
        "resource_levels": {}
    }
}