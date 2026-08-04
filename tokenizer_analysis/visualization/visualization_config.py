"""
Simple configuration for visualization.
"""

import matplotlib.pyplot as plt


# LaTeX table formatting options
class LaTeXFormatting:
    BOLD_BEST = True
    INCLUDE_STD_ERR = False
    STD_ERROR_SIZE = "footnotesize"


# Plot configuration constants
class PlotConfig:
    # Regular plot settings
    REGULAR_XTICK_SIZE = 10
    REGULAR_YTICK_SIZE = 10
    
    # Faceted plot settings  
    FACETED_XTICK_SIZE = 8
    FACETED_YTICK_SIZE = 9

