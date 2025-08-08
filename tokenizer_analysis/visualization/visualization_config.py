"""
Simple configuration for visualization.
"""

import matplotlib.pyplot as plt


# LaTeX table formatting options
class LaTeXFormatting:
    BOLD_BEST = True
    INCLUDE_STD_ERR = False
    STD_ERROR_SIZE = "footnotesize"


def setup_default_style():
    """Setup basic matplotlib styling."""
    plt.rcParams.update({
        'font.family': 'serif',
        'figure.figsize': (10, 6),
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })