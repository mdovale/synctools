"""Tests for signal-combination containers."""

import numpy as np
import pytest

from synctools.frequency import FrequencyData
from synctools.signals import (
    ThreeSignals,
    ThreeSignalsElement,
    TwoSignals,
    TwoSignalsElement,
)


def _frequency_data(values, fs=10.0, name="signal"):
    return FrequencyData(np.asarray(values, dtype=np.float64), fs=fs, name=name)


def test_two_signals_element_rejects_nonfinite_frequency():
    freqs = np.ones((8, 2))
    freqs[0, 0] = np.nan

    with pytest.raises(ValueError, match="freqs must contain only finite values"):
        TwoSignalsElement(freqs, fs=10.0)


def test_two_signals_element_handles_single_sample_phase():
    element = TwoSignalsElement(np.array([[2.0, 1.0]]), fs=10.0)

    assert element.freq.shape == (1,)
    assert element.phase.shape == (1,)
    assert np.all(np.isfinite(element.phase))


def test_two_signals_rejects_mismatched_lengths(default_lpsd_params):
    sig_a = _frequency_data(np.ones(8), name="a")
    sig_b = _frequency_data(np.ones(9), name="b")

    with pytest.raises(ValueError, match="All signals must have same length"):
        TwoSignals([sig_a, sig_b], default_lpsd_params)


def test_three_signals_element_rejects_nonfinite_weights():
    weights = np.ones((8, 3))
    weights[0, 0] = np.inf

    with pytest.raises(ValueError, match="weights must contain only finite values"):
        ThreeSignalsElement(np.ones((8, 3)), fs=10.0, signs=[1, 1, -1], weights=weights)


def test_three_signals_validates_balancing_signal_length(default_lpsd_params):
    sigs = [
        _frequency_data(np.ones(8), name="a"),
        _frequency_data(np.ones(8), name="b"),
        _frequency_data(np.ones(8), name="c"),
    ]
    bsigs = [_frequency_data(np.ones(7), name="a_balanced"), None, None]

    with pytest.raises(ValueError, match="bsigs\\[0\\].total must have length 8"):
        ThreeSignals(sigs, default_lpsd_params, bsigs=bsigs)


def test_three_signals_name_and_tau_are_stable(default_lpsd_params):
    n_samples = 128
    tau = np.arange(n_samples) / 10.0
    sigs = [
        _frequency_data(np.sin(tau), name="a"),
        _frequency_data(np.cos(tau), name="b"),
        _frequency_data(np.sin(tau) + np.cos(tau), name="c"),
    ]
    bsigs = [None, _frequency_data(np.zeros(n_samples), name="b_balanced"), None]

    result = ThreeSignals(sigs, default_lpsd_params, bsigs=bsigs)

    assert result.name == "(a,b+b_balanced,c)"
    assert result.tau.shape == sigs[0].tau.shape
    assert not np.shares_memory(result.tau, sigs[0].tau)
