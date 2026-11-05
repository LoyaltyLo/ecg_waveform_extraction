"""Unit tests for the public calling interface (src/api.py).

Pure-function coverage only — the heavy stage chain is exercised by
the real runs documented in the stage docstrings.
"""
from pathlib import Path

import pandas as pd
import pytest

from ecg_waveform_extraction.src.api import (
    STAGES, SUBDIRS, _ordered_stages, _records_csv, _read_csv, _stage_argv,
)


# ------------------------------------------------------------------ stages

def test_stages_table_is_consistent():
    """Every stage has a module and an output subdir; order is canonical."""
    assert STAGES == tuple(SUBDIRS.keys())
    for stage in STAGES:
        argv = _stage_argv(stage, 'DATA', Path('OUT'), None, True)
        if stage != 'overview':  # overview reads only the four stage dirs
            assert argv[:2] == ['--data-dir', 'DATA']
        assert str(Path('OUT') / SUBDIRS[stage]) in argv


def test_ordered_stages_validates_and_sorts():
    assert _ordered_stages(None) == list(STAGES)
    # subset comes back in canonical order regardless of input order
    assert _ordered_stages(['reversal', 'polarity']) == ['polarity', 'reversal']
    with pytest.raises(ValueError, match='unknown stage'):
        _ordered_stages(['rpeaks', 'nope'])


# ------------------------------------------------------------------ records

def test_records_csv_accepts_str_iterable_none():
    assert _records_csv(None) is None
    assert _records_csv('a,b') == 'a,b'
    assert _records_csv(' a , , b ') == 'a,b'
    assert _records_csv(['x.aECG', 'y']) == 'x,y'
    assert _records_csv([]) is None


# -------------------------------------------------------------------- argv

def test_stage_argv_wiring():
    out = Path('OUT')
    # pwave reads the T-stage dir
    argv = _stage_argv('pwave', 'D', out, None, True)
    assert '--t-stage-dir' in argv and str(out / SUBDIRS['twave']) in argv
    # polarity joins the three upstream dirs
    argv = _stage_argv('polarity', 'D', out, None, True)
    for d in ('pwave', 'twave', 'qrs'):
        assert str(out / SUBDIRS[d]) in argv
    # overview reads the four stage dirs (no --data-dir of its own)
    argv = _stage_argv('overview', 'D', out, None, True)
    assert '--data-dir' not in argv and '--pol-dir' in argv
    # reversal reads polarity + overview and honors --no-plots
    argv = _stage_argv('reversal', 'D', out, None, False)
    assert '--no-plots' in argv
    assert str(out / SUBDIRS['overview'] / 'record_overview.csv') in argv
    assert '--no-plots' not in _stage_argv('reversal', 'D', out, None, True)
    # stage 1 has no --records flag; the others forward the filter
    assert '--records' not in _stage_argv('rpeaks', 'D', out, 'a,b', True)
    assert '--records' in _stage_argv('qrs', 'D', out, 'a,b', True)


# ---------------------------------------------------------------- read csv

def test_read_csv_new_fallback(tmp_path):
    """A locked-write .new.csv is picked up when the main file is absent."""
    pd.DataFrame({'a': [1]}).to_csv(tmp_path / 'table.new.csv', index=False)
    got = _read_csv(tmp_path, 'table.csv')
    assert list(got['a']) == [1]
    with pytest.raises(FileNotFoundError):
        _read_csv(tmp_path, 'missing.csv')
