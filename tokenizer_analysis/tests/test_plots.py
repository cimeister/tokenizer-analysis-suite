"""Plot tests.

Most of these assert on the figure object rather than on the written file.
A written SVG has a non-zero size whether or not any panel drew anything, so
``out.exists() and out.stat().st_size > 0`` passes on a blank panel. That is
how a plot reading a key the 1.0 metric merge had moved kept shipping.
"""
import copy
import os

import matplotlib
matplotlib.use("Agg")  # headless; must precede pyplot import in plots.py

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tokenizer_analysis.metrics.redundancy import merge_redundant_metrics
from tokenizer_analysis.visualization import plots
from tokenizer_analysis.visualization.plots import (
    generate_all_plots,
    plot_grouped_analysis,
    plot_lorenz_curves,
    plot_morphscore,
    plot_utf8_integrity,
    plot_vocab_util_cross_lingual_cov,
)

_TOKS = ['A', 'B']

_LORENZ = {
    'A': {'x_values': [0.0, 0.5, 1.0], 'y_values': [0.0, 0.4, 1.0]},
    'B': {'x_values': [0.0, 0.5, 1.0], 'y_values': [0.0, 0.3, 1.0]},
}


@pytest.fixture
def captured_plots(monkeypatch):
    """Capture ``(path, fig)`` instead of writing an image.

    Every plot function ends in ``save_plot``, so replacing it hands the test
    the Figure that would have been written. Asserting on ``fig.axes`` is what
    lets a test see an empty panel, a NaN bar height, or a bar colour.
    """
    captured = []

    def _capture(fig, filepath):
        captured.append((filepath, fig))
        plt.close(fig)   # keep the pyplot figure registry from filling up

    monkeypatch.setattr(plots, "save_plot", _capture)
    return captured


def _labelled_bars(ax):
    """Pair each bar with its category label, in draw order."""
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    return list(zip(tick_labels, ax.patches))


def _bar_colors(ax):
    """Map category label to the hex colour of that category's bar."""
    return {
        label: matplotlib.colors.to_hex(patch.get_facecolor())
        for label, patch in _labelled_bars(ax)
    }


def _full_results():
    """Results shaped like a real run, covering every plot generate_all_plots draws."""
    return {
        'fertility': {
            'per_tokenizer': {
                'A': {'global': {'mean': 1.4, 'std': 0.2}},
                'B': {'global': {'mean': 1.8, 'std': 0.3}},
            },
            'metadata': {'normalization_method': 'words'},
        },
        'vocabulary_utilization': {
            'per_tokenizer': {
                'A': {'global_utilization': 0.61, 'per_language_cov': 0.20},
                'B': {'global_utilization': 0.44, 'per_language_cov': 0.35},
            },
            'metadata': {},
        },
        'compression_rate': {
            'per_tokenizer': {
                'A': {'global': {'compression_rate': 3.9}},
                'B': {'global': {'compression_rate': 3.1}},
            },
            'metadata': {'normalization_method': 'bytes'},
        },
        'bigram_entropy': {
            'per_tokenizer': {
                'A': {'global_bigram_entropy': 0.72},
                'B': {'global_bigram_entropy': 0.65},
            },
            'metadata': {},
        },
        'tokenizer_fairness_gini': {
            'per_tokenizer': {
                'A': {'gini_coefficient': 0.11},
                'B': {'gini_coefficient': 0.19},
            },
            'metadata': {},
        },
        'lorenz_curve_data': {
            'per_tokenizer': copy.deepcopy(_LORENZ),
            'metadata': {},
        },
        'morphscore': {
            'per_tokenizer': {
                'A': {'summary': {'avg_morphscore_recall': 0.55,
                                  'avg_morphscore_precision': 0.61}},
                'B': {'summary': {'avg_morphscore_recall': 0.48,
                                  'avg_morphscore_precision': 0.52}},
            },
            'metadata': {},
        },
        'utf8_token_integrity': {
            'summary': {'A': {'completeness_rate': 0.99},
                        'B': {'completeness_rate': 0.93}},
            'per_tokenizer': {
                'A': {'global': {'completeness_rate': 0.99}},
                'B': {'global': {'completeness_rate': 0.93}},
            },
            'metadata': {},
        },
        'utf8_char_split': {
            'summary': {'A': {'splits_per_1k_tokens': 1.2},
                        'B': {'splits_per_1k_tokens': 4.7}},
            'per_tokenizer': {
                'A': {'global': {'splits_per_1k_tokens': 1.2}},
                'B': {'global': {'splits_per_1k_tokens': 4.7}},
            },
            'metadata': {},
        },
    }


def test_plot_vocab_util_cov_smoke_and_none_skip(tmp_path):
    """The CoV plot renders for normal tokenizers and silently skips the
    bar for a None-CoV (single-language) tokenizer instead of crashing
    (exercises the plot_metric_bar_chart None-skip guard)."""
    results = {
        'vocabulary_utilization': {
            'per_tokenizer': {
                'A': {'global_utilization': 0.5, 'per_language_cov': 0.20},
                'B': {'global_utilization': 0.5, 'per_language_cov': 0.35},
                'C': {'global_utilization': 0.5, 'per_language_cov': None},
            },
            'metadata': {},
        }
    }
    out = tmp_path / "vocab_util_cov.svg"
    plot_vocab_util_cross_lingual_cov(results, str(out), ['A', 'B', 'C'])
    assert out.exists() and out.stat().st_size > 0


_EXPECTED_PLOTS = {
    'bigram_entropy_individual.svg',
    'compression_rate_individual.svg',
    'fertility_individual.svg',
    'lorenz_curves_individual.svg',
    'morphscore_individual.svg',
    'tokenizer_fairness_gini_individual.svg',
    'utf8_integrity.svg',
    'vocab_util_cross_lingual_cov_individual.svg',
    'vocabulary_utilization_individual.svg',
}


def test_every_generated_plot_draws_content(captured_plots, tmp_path):
    """Catch a plot reading a key that a results-schema change has moved.

    A moved key shows up in one of two ways, so the test checks both. Some
    plots write a figure regardless and leave the panel blank, which the
    per-axes assertion catches. Others find nothing to draw and return before
    writing anything at all, which no per-figure assertion can see: that is why
    the set of files written is compared against _EXPECTED_PLOTS rather than
    only asserted non-empty. Adding a plot to generate_all_plots means adding
    its filename there.

    The 1.0 merge (metrics/redundancy.py) folds utf8_char_split into
    utf8_token_integrity and lorenz_curve_data into tokenizer_fairness_gini, so
    both the pre-merge and post-merge shapes must draw the same set. The merge
    runs here for real, not as a hand-written imitation of its output, so a
    change to MERGES that a plot does not follow fails this test.
    """
    cases = [
        ("pre-merge", _full_results()),
        ("post-merge", merge_redundant_metrics(_full_results())),
    ]
    for label, results in cases:
        start = len(captured_plots)
        generate_all_plots(results, str(tmp_path), _TOKS)
        produced = captured_plots[start:]
        names = {os.path.basename(path) for path, _ in produced}
        assert names == _EXPECTED_PLOTS, (
            f"{label}: missing {sorted(_EXPECTED_PLOTS - names)}, "
            f"unexpected {sorted(names - _EXPECTED_PLOTS)}"
        )
        for path, fig in produced:
            for i, ax in enumerate(fig.axes):
                if not ax.get_visible():
                    continue
                assert ax.patches or ax.lines, (
                    f"{label}: {os.path.basename(path)} axes {i} drew nothing"
                )


def _utf8_results_b_unmeasured():
    """utf8 results where tokenizer B has an empty summary entry."""
    return {
        'utf8_token_integrity': {
            'summary': {'A': {'completeness_rate': 0.99}, 'B': {}},
            'per_tokenizer': {
                'A': {'char_split': {'global': {'splits_per_1k_tokens': 1.2}}},
                'B': {},
            },
        }
    }


def _morphscore_results_b_unmeasured():
    """morphscore results where tokenizer B has an empty summary entry."""
    return {
        'morphscore': {
            'per_tokenizer': {
                'A': {'summary': {'avg_morphscore_recall': 0.55,
                                  'avg_morphscore_precision': 0.61}},
                'B': {'summary': {}},
            }
        }
    }


@pytest.mark.parametrize("plot_fn, build_results", [
    (plot_utf8_integrity, _utf8_results_b_unmeasured),
    (plot_morphscore, _morphscore_results_b_unmeasured),
], ids=["utf8_integrity", "morphscore"])
def test_unmeasured_value_is_absent_not_a_finite_bar(
    plot_fn, build_results, captured_plots, tmp_path
):
    """A tokenizer with no measurement must not get a drawable bar height.

    Both of these plots previously defaulted a missing value: morphscore to
    0.0 and the integrity rate to 1.0. A 0.0 bar is indistinguishable from a
    measured 0.0, and 1.0 is the best possible completeness rate, so an
    unmeasured tokenizer was drawn as the best one on the panel. Absent
    (no bar) or NaN (matplotlib draws nothing) are the two acceptable
    renderings; a finite height for B is not.
    """
    plot_fn(build_results(), str(tmp_path / "p.svg"), ['A', 'B'])
    assert captured_plots, "plot function produced no figure"

    finite_bars_for_a = 0
    for _, fig in captured_plots:
        for ax in fig.axes:
            for label, patch in _labelled_bars(ax):
                height = patch.get_height()
                if label == 'B':
                    assert np.isnan(height), (
                        f"B has no measurement but was drawn at height {height}"
                    )
                elif label == 'A' and not np.isnan(height):
                    finite_bars_for_a += 1
    assert finite_bars_for_a, "A's measured bars are missing; the panel is empty"


def test_grouped_extractor_failure_is_nan_not_zero(captured_plots, tmp_path):
    """A per-group extractor that raises must leave a gap, not a zero bar.

    plot_grouped_analysis is the only plot call generate_all_plots wraps in
    try/except, so a failure inside it is logged at WARNING and nothing else.
    If the failure produced a 0.0 bar the reader would see a measured zero for
    a tokenizer that was never measured in that group.
    """
    grouped = {
        'domain': {
            'code': {
                'fertility': {
                    'per_tokenizer': {
                        'A': {'global': {'mean': 1.5}},
                        # No 'global' key: the default extractor raises KeyError.
                        'B': {},
                    },
                    'metadata': {},
                }
            }
        }
    }
    plot_grouped_analysis(grouped, str(tmp_path), 'fertility', 'domain',
                          tokenizer_names=['A', 'B'])
    assert len(captured_plots) == 1
    _, fig = captured_plots[0]
    heights = {c.get_label(): [p.get_height() for p in c]
               for c in fig.axes[0].containers}
    assert heights['A'] == [1.5]
    assert np.isnan(heights['B'][0]), (
        f"failed extractor drew B at {heights['B'][0]} instead of NaN"
    )


@pytest.mark.parametrize("results", [
    pytest.param({'lorenz_curve_data': {'per_tokenizer': copy.deepcopy(_LORENZ)}},
                 id="legacy-top-level"),
    pytest.param({'tokenizer_fairness_gini': {'per_tokenizer': {
        tok: {'gini_coefficient': 0.1, 'lorenz_curve': copy.deepcopy(curve)}
        for tok, curve in _LORENZ.items()
    }}}, id="merged-under-gini"),
])
def test_lorenz_reads_both_result_layouts(results, captured_plots, tmp_path):
    """Both documented locations for the curve data must plot.

    plot_lorenz_curves' docstring promises that a results file written before
    the 1.0 merge still plots from the top-level lorenz_curve_data key. Nothing
    else covers that branch, and a silent failure there is an empty figure
    written under the expected filename.
    """
    plot_lorenz_curves(results, str(tmp_path / "lorenz.svg"), ['A', 'B'])
    assert len(captured_plots) == 1
    _, fig = captured_plots[0]
    drawn = {line.get_label() for line in fig.axes[0].lines}
    assert drawn == {'A', 'B', 'Perfect Equality'}


def test_utf8_panel_colours_are_keyed_to_the_tokenizer(captured_plots, tmp_path):
    """One tokenizer keeps one colour across both UTF-8 panels.

    The panels used to slice ``colors[:len(filtered_labels)]`` from a shared
    palette, so a tokenizer missing from one panel shifted every colour in the
    other. Measured with A absent from the integrity panel: B was #ee7733 on
    the left and #0077bb on the right of the same figure, which reads as two
    different tokenizers.
    """
    results = {
        'utf8_token_integrity': {
            # A is absent here and present in the split panel below.
            'summary': {'B': {'completeness_rate': 0.9},
                        'C': {'completeness_rate': 0.8}},
            'per_tokenizer': {
                'A': {'char_split': {'global': {'splits_per_1k_tokens': 3.0}}},
                'B': {'char_split': {'global': {'splits_per_1k_tokens': 2.0}}},
                'C': {'char_split': {'global': {'splits_per_1k_tokens': 1.0}}},
            },
        }
    }
    plot_utf8_integrity(results, str(tmp_path / "utf8.svg"), ['A', 'B', 'C'])
    assert len(captured_plots) == 1
    _, fig = captured_plots[0]
    integrity_colors = _bar_colors(fig.axes[0])
    split_colors = _bar_colors(fig.axes[1])

    assert set(integrity_colors) == {'B', 'C'}
    assert set(split_colors) == {'A', 'B', 'C'}
    for tok in ('B', 'C'):
        assert integrity_colors[tok] == split_colors[tok], (
            f"{tok} is {integrity_colors[tok]} on the integrity panel and "
            f"{split_colors[tok]} on the split panel of the same figure"
        )
    # Distinct tokenizers still get distinct colours.
    assert len(set(split_colors.values())) == 3


class TestFacetedAndPerLanguageFlagsAreIndependent:
    """`--faceted-plots` help text claimed a dependency the code does not have.

    Until 1.0.3 it read "for grouped analysis (--run-grouped-analysis) and
    per-language plots (--per-language-plots)". Both halves were wrong.
    `generate_all_plots` gates the two subdirectories on separate `if`
    statements, and `plotter.plot_grouped_analysis` passes both as `False` on
    the grouped path, so the flag reaches neither. These assert the behaviour
    the corrected help text now describes, so the two cannot drift apart again
    without a test failing.
    """

    def _dirs(self, captured):
        return {os.path.basename(os.path.dirname(path)) for path, _ in captured}

    def _results(self):
        """`_full_results` plus the per-language blocks those plots need.

        Without them every per-language plot returns before drawing, so a test
        asserting the subdirectory is absent would pass for the wrong reason.
        """
        results = _full_results()
        results['fertility']['per_tokenizer']['A']['per_language'] = {
            'eng_Latn': {'mean': 1.3}, 'deu_Latn': {'mean': 1.6}}
        results['fertility']['per_tokenizer']['B']['per_language'] = {
            'eng_Latn': {'mean': 1.7}, 'deu_Latn': {'mean': 2.1}}
        return results

    def test_faceted_alone_writes_faceted_and_no_per_language(self, captured_plots, tmp_path):
        generate_all_plots(self._results(), str(tmp_path), _TOKS,
                           per_language_plots=False, faceted_plots=True)
        dirs = self._dirs(captured_plots)
        assert "faceted_plots" in dirs
        assert "per-language" not in dirs

    def test_per_language_alone_writes_no_faceted(self, captured_plots, tmp_path):
        generate_all_plots(self._results(), str(tmp_path), _TOKS,
                           per_language_plots=True, faceted_plots=False)
        dirs = self._dirs(captured_plots)
        assert "per-language" in dirs, "the fixture drew no per-language plot at all"
        assert "faceted_plots" not in dirs

    def test_grouped_analysis_gets_neither(self, captured_plots, tmp_path):
        """The outcome, which two separate things in plotter.py guarantee.

        It passes `per_language_plots=False, faceted_plots=False`, and it also
        passes `{}` as the results dict, so both branches would find nothing to
        draw even if the flags were forwarded. This asserts the outcome rather
        than either mechanism, so flipping only the hardcoded `False` leaves it
        passing.
        """
        from tokenizer_analysis.visualization.plotter import TokenizerVisualizer

        viz = TokenizerVisualizer(_TOKS, str(tmp_path),
                                  per_language_plots=True, faceted_plots=True)
        viz.plot_grouped_analysis({"script": {"latin": self._results()}})
        dirs = self._dirs(captured_plots)
        assert "faceted_plots" not in dirs
        assert "per-language" not in dirs

    def test_the_help_text_does_not_promise_either_link(self):
        """The help string is the thing that was wrong; assert on it directly."""
        from tokenizer_analysis.cli.run_analysis import build_parser

        help_text = next(
            a.help for a in build_parser()._actions
            if "--faceted-plots" in (a.option_strings or [])
        )
        lowered = help_text.lower()
        assert "independent of --per-language-plots" in lowered
        assert "does not apply to grouped analysis" in lowered
