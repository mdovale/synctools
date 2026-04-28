"""
Tests for Clock class.
"""
import numpy as np
import pytest

from synctools.clock import Clock
from synctools.frequency import FrequencyData


class TestClockBasic:
    """Basic tests for Clock."""

    def test_clock_creation_copies_frequency_data(self, sample_fs):
        """Clock should isolate its mutable FrequencyData from caller-owned data."""
        fd = FrequencyData(np.ones(100) * 1e-12, sample_fs)

        clock = Clock(fd, timer_offset=1e-6, timer_sign=1, name="clock")

        assert clock.name == "clock"
        assert clock.primary_stamped is True
        assert clock.timer_sign == 1
        assert clock.timer_offset == pytest.approx(1e-6)
        assert clock.rd is not fd
        assert not np.shares_memory(clock.rd.total, fd.total)
        assert clock.tshift.total["raw"][0] == pytest.approx(1e-6 + 1e-12 / sample_fs)

    def test_clock_validation(self, sample_fs):
        """Clock should reject invalid construction inputs."""
        fd = FrequencyData(np.zeros(100), sample_fs)

        with pytest.raises(ValueError, match="timer_sign must be -1 or \\+1"):
            Clock(fd, timer_sign=0)

        with pytest.raises(ValueError, match="timer_sign must be -1 or \\+1"):
            Clock(fd, timer_sign=True)

        with pytest.raises(ValueError, match="timer_offset must be finite"):
            Clock(fd, timer_offset=np.inf)

        with pytest.raises(ValueError, match="primary_stamped must be a bool"):
            Clock(fd, primary_stamped="yes")

        with pytest.raises(ValueError, match="name must be a string"):
            Clock(fd, name=object())

        with pytest.raises(ValueError, match="rd must be a FrequencyData-like object"):
            Clock(object())


class TestClockOperations:
    """Tests for Clock mutation methods."""

    def test_add_timer_offset_refreshes_timer(self, sample_fs):
        """Adding an offset should update both state and derived timer data."""
        fd = FrequencyData(np.zeros(100), sample_fs)
        clock = Clock(fd, timer_offset=1.0)

        clock.add_timer_offset(2.0)

        assert clock.timer_offset == pytest.approx(3.0)
        assert np.allclose(clock.tshift.total["raw"], 3.0)

        with pytest.raises(ValueError, match="addition must be finite"):
            clock.add_timer_offset(np.nan)

    def test_time_stamping_refreshes_timer(self, sample_fs):
        """Time-stamping rd should keep tshift consistent with transformed rd."""
        n_samples = 100
        fd = FrequencyData(np.ones(n_samples) * 1e-12, sample_fs)
        clock = Clock(fd, timer_sign=1)
        original_timer = clock.tshift.total["raw"].copy()

        clock.time_stamping(np.zeros(n_samples), factor=2.0, order=1)

        assert np.allclose(clock.rd.total, 2.0 * fd.total)
        assert np.allclose(clock.tshift.total["raw"], 2.0 * original_timer)

        with pytest.raises(ValueError, match="order must be positive"):
            clock.time_stamping(np.zeros(n_samples), order=0)

    def test_truncation(self, sample_fs):
        """Truncation should keep frequency and timer arrays aligned."""
        n_samples = 100
        fd = FrequencyData(np.zeros(n_samples), sample_fs)
        clock = Clock(fd)

        clock.truncation(10)

        assert len(clock.rd.tau) == n_samples - 20
        assert len(clock.tshift.tau) == len(clock.rd.tau)
        assert len(clock.tshift.total["raw"]) == len(clock.rd.tau)
        assert len(clock.tshift.timer_dev_err["estimate"]) == len(clock.rd.tau)

    def test_zero_truncation_is_noop(self, sample_fs):
        """Zero truncation should preserve arrays."""
        n_samples = 100
        fd = FrequencyData(np.zeros(n_samples), sample_fs)
        clock = Clock(fd)

        clock.truncation(0)

        assert len(clock.rd.tau) == n_samples
        assert len(clock.tshift.tau) == n_samples

    def test_truncation_validation(self, sample_fs):
        """Invalid truncation inputs should fail before mutation."""
        fd = FrequencyData(np.zeros(100), sample_fs)
        clock = Clock(fd)

        with pytest.raises(ValueError, match="must be < len\\(data\\) // 2"):
            clock.truncation(50)

        with pytest.raises(ValueError, match="must be non-negative"):
            clock.truncation(-1)
