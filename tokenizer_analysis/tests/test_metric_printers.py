"""The console printers must not turn a published null into 0.000.

Several metrics publish None for a value they could not compute, so that a
reader cannot mistake it for a measured zero: MorphScore on zero languages,
operator isolation with no compound operator in the corpus, the Gini
coefficient when every language cost 0. The printers formatted those with
``{value:.3f}``, guarded at most by ``d.get(key, 0.0)``. Neither works: the
default never fires when the key is present holding None, so the format raises
TypeError and takes the run's summary down, and where the key is missing it
prints 0.000, which is the same misleading zero the results file avoids.
"""

import logging

from tokenizer_analysis.constants import MISSING_VALUE_DISPLAY
from tokenizer_analysis.metrics.base import format_optional
from tokenizer_analysis.metrics.gini import TokenizerGiniMetrics
from tokenizer_analysis.metrics.math import DigitBoundaryMetrics
from tokenizer_analysis.metrics.morphscore import MorphScoreMetrics

from .conftest import SimpleProvider as _SimpleProvider


def _null_morphscore_summary():
    """What MorphScore.compute publishes for a tokenizer it evaluated on
    zero languages."""
    return {
        'languages_evaluated': 0,
        'total_samples': 0,
        'avg_morphscore_recall': None,
        'avg_morphscore_precision': None,
        'avg_micro_f1': None,
        'avg_macro_f1': None,
    }


class TestNullsPrintAsAbsent:

    def test_format_optional_handles_none_and_numbers(self):
        assert format_optional(None) == MISSING_VALUE_DISPLAY
        assert format_optional(None, '.4f') == MISSING_VALUE_DISPLAY
        assert format_optional(0.5) == '0.500'
        assert format_optional(0.0) == '0.000'

    def test_morphscore_prints_no_zeros_for_a_tokenizer_it_could_not_score(
        self, capsys
    ):
        inst = object.__new__(MorphScoreMetrics)
        inst.tokenizer_names = ['tok']
        inst.print_results({'morphscore': {
            'metadata': {},
            'per_tokenizer': {'tok': {'summary': _null_morphscore_summary()}},
        }})
        out = capsys.readouterr().out
        for label in ('Recall', 'Precision', 'Micro F1', 'Macro F1'):
            line = next(l for l in out.splitlines() if l.strip().startswith(label))
            assert line.endswith(MISSING_VALUE_DISPLAY), line
        assert '0.000' not in out

    def test_morphscore_missing_scores_are_absent_not_zero(self, capsys):
        """A results file whose block is not the one the printer reads.

        The printed value used to come from ``.get(key, 0.0)``, so a block
        without the four score keys printed four zeros that looked measured.
        """
        inst = object.__new__(MorphScoreMetrics)
        inst.tokenizer_names = ['tok']
        inst.print_results({'morphscore': {
            'metadata': {},
            'per_tokenizer': {'tok': {'global': _null_morphscore_summary()}},
        }})
        assert '0.000' not in capsys.readouterr().out

    def test_operator_isolation_prints_null_compound_rate_as_absent(self, capsys):
        inst = object.__new__(DigitBoundaryMetrics)
        inst.tokenizer_names = ['tok']
        inst.print_results({'operator_isolation_rate': {
            'summary': {'tok': {
                'overall_isolation_rate': 0.8,
                # None because the corpus held no compound operator.
                'overall_compound_preservation_rate': None,
                'total_operators': 10,
                'total_compound_operators': 0,
            }},
            'per_tokenizer': {'tok': {'by_category': {
                'arithmetic': {'isolation_rate': 0.8,
                               'compound_preservation_rate': None,
                               'total': 10, 'compound_total': 0},
            }}},
        }})
        out = capsys.readouterr().out
        assert f'Compound Preservation         : {MISSING_VALUE_DISPLAY}' in out
        assert f'compound={MISSING_VALUE_DISPLAY}' in out
        assert '0.800' in out          # the rate that was measured still prints
        assert '0.000' not in out

    def test_gini_logs_its_own_nulls_without_raising(self, caplog, monkeypatch):
        """The Gini log line formats two values that same function publishes null.

        This one is not a printer: the line sits inside
        compute_tokenizer_fairness_gini, so formatting a null there aborted the
        whole metric rather than misreporting one number. The zero costs are
        injected because TokenizedData rejects an empty token list, which is
        what currently keeps a zero-cost language out of real data; the branch
        that produces the nulls is reached exactly this way.
        """
        provider = _SimpleProvider('tok')
        inst = TokenizerGiniMetrics(provider)
        monkeypatch.setattr(
            TokenizerGiniMetrics, '_compute_language_costs',
            lambda self, tok_data, languages=None: {'a': 0.0, 'b': 0.0},
        )
        with caplog.at_level(logging.INFO,
                             logger='tokenizer_analysis.metrics.gini'):
            result = inst.compute_tokenizer_fairness_gini({'tok': []})
        entry = result['per_tokenizer']['tok']
        assert entry['gini_coefficient'] is None
        assert entry['cost_ratio'] is None
        assert MISSING_VALUE_DISPLAY in caplog.text
