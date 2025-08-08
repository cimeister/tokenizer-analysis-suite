"""
Simple data extraction utilities for visualization.
"""

from typing import Any


def extract_value(data: Any, key: str, default: float = 0.0) -> float:
    """Extract numeric value from nested data structures."""
    if isinstance(data, (int, float)):
        return data
    
    if isinstance(data, dict):
        if key in data:
            val = data[key]
            if isinstance(val, (int, float)):
                return val
            elif isinstance(val, dict) and 'mean' in val:
                return val['mean']
        
        # Common fallbacks
        if 'mean' in data:
            return data['mean']
        if 'value' in data:
            return data['value']
    
    return default