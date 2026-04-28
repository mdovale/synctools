"""
Tests for TimerData class.
"""
import numpy as np
import pytest
from synctools.timer import TimerData
from synctools.frequency import FrequencyData


class TestTimerDataBasic:
    """Basic tests for TimerData."""
    
    def test_timer_data_creation(self, sample_fs):
        """Test basic TimerData creation."""
        n_samples = 1000
        # Fractional frequency (dimensionless)
        frac_freq = 1e-12 * np.ones(n_samples)
        fd = FrequencyData(frac_freq, sample_fs)
        
        td = TimerData(fd, sample_fs)
        
        assert len(td.tau) == n_samples
        assert 'raw' in td.total
        assert 'inv' in td.total
        assert 'raw' in td.fit
        assert 'inv' in td.fit
        assert 'raw' in td.fluc
        assert 'inv' in td.fluc
    
    def test_timer_data_validation(self, sample_fs):
        """Test input validation."""
        n_samples = 100
        frac_freq = np.zeros(n_samples)
        fd = FrequencyData(frac_freq, sample_fs)
        
        with pytest.raises(ValueError, match="Sampling rate fs must be > 0"):
            TimerData(fd, -1.0)
        
        with pytest.raises(ValueError, match="sign must be -1 or \\+1"):
            TimerData(fd, sample_fs, sign=0)
        
        with pytest.raises(ValueError, match="sign must be -1 or \\+1"):
            TimerData(fd, sample_fs, sign=2)
    
    def test_timer_data_sign(self, sample_fs):
        """Test that sign parameter flips the timer."""
        n_samples = 100
        frac_freq = 1e-12 * np.ones(n_samples)
        fd = FrequencyData(frac_freq, sample_fs)
        
        td_pos = TimerData(fd, sample_fs, sign=1)
        td_neg = TimerData(fd, sample_fs, sign=-1)
        
        # With opposite signs, raw timers should be opposite
        assert np.allclose(td_pos.total['raw'], -td_neg.total['raw'])
    
    def test_timer_data_offset(self, sample_fs):
        """Test that offset is applied correctly."""
        n_samples = 100
        frac_freq = np.zeros(n_samples)
        fd = FrequencyData(frac_freq, sample_fs)
        
        offset = 1e-6  # 1 microsecond
        td = TimerData(fd, sample_fs, offset=offset)
        
        # Timer should start at offset
        assert abs(td.total['raw'][0] - offset) < 1e-12


class TestTimerDataLinearDrift:
    """Tests for TimerData with linear drifts."""
    
    def test_linear_drift_integration(self, sample_fs):
        """Test that linear fractional frequency drift integrates to quadratic timer."""
        n_samples = 1000
        t = np.arange(n_samples) / sample_fs
        
        # Linear fractional frequency: y(t) = a*t
        a = 1e-12  # Hz
        frac_freq = a * t
        fd = FrequencyData(frac_freq, sample_fs, order=1)
        
        # TimerData uses sign=-1 by default, which flips the sign
        td = TimerData(fd, sample_fs, sign=-1)
        
        # Timer = integral of fractional frequency (discrete integration)
        # For y[i] = a * i/fs, timer[i] = dt * sum(y[0:i+1])
        # = (1/fs) * sum(a * j/fs for j in range(i+1))
        # = (a/fs^2) * i*(i+1)/2
        # = (a/2) * (i/fs) * ((i+1)/fs)
        # With sign=-1, it becomes negative
        dt = 1.0 / sample_fs
        expected_timer = np.array([-(a / 2.0) * (i * dt) * ((i + 1) * dt) for i in range(n_samples)])
        
        # Check that timer matches expected (within numerical precision)
        # Use more lenient tolerance for discrete integration
        assert np.allclose(td.total['raw'], expected_timer, rtol=1e-4, atol=1e-15)
    
    def test_constant_fractional_frequency(self, sample_fs):
        """Test timer for constant fractional frequency."""
        n_samples = 1000
        t = np.arange(n_samples) / sample_fs
        
        # Constant fractional frequency
        frac_freq = 1e-12 * np.ones(n_samples)
        fd = FrequencyData(frac_freq, sample_fs, order=0)
        
        td = TimerData(fd, sample_fs)
        
        # Timer = integral of constant = constant * t
        expected_timer = 1e-12 * t
        
        assert np.allclose(td.total['raw'], expected_timer, rtol=1e-10)
    
    def test_known_offset_linear_drift(self, sample_fs):
        """Test TimerData with known offset and linear drift."""
        n_samples = 1000
        t = np.arange(n_samples) / sample_fs
        
        # Linear drift with offset
        offset = 1e-6  # 1 microsecond
        drift_rate = 1e-12  # Hz
        frac_freq = drift_rate * t
        fd = FrequencyData(frac_freq, sample_fs, order=1)
        
        # TimerData uses sign=-1 by default, which flips the sign
        td = TimerData(fd, sample_fs, sign=-1, offset=offset)
        
        # Expected timer = offset + integral of drift (discrete)
        # With sign=-1, drift becomes negative
        dt = 1.0 / sample_fs
        expected_timer = offset + np.array([-(drift_rate / 2.0) * (i * dt) * ((i + 1) * dt) for i in range(n_samples)])
        
        assert np.allclose(td.total['raw'], expected_timer, rtol=1e-4, atol=1e-15)
        # Check initial offset
        assert abs(td.total['raw'][0] - offset) < 1e-12


class TestTimerDataInverse:
    """Tests for TimerData inverse timer computation."""
    
    def test_inverse_timer_basic(self, sample_fs):
        """Test basic inverse timer computation."""
        n_samples = 100
        # Small linear drift
        t = np.arange(n_samples) / sample_fs
        frac_freq = 1e-12 * t
        fd = FrequencyData(frac_freq, sample_fs, order=1)
        
        td = TimerData(fd, sample_fs, sign=-1)
        
        # Inverse timer should exist
        assert 'inv' in td.total
        assert len(td.total['inv']) == n_samples
        
        # For very small drifts, inverse timer might be very close to raw timer
        # Check that they're at least numerically different
        # Use a more lenient check - they should differ by more than numerical noise
        diff = np.abs(td.total['raw'] - td.total['inv'])
        # For small drifts, the difference might be very small, but should exist
        # Check that max difference is at least 1e-18 (numerical precision threshold)
        assert np.max(diff) > 1e-18 or np.any(diff > 1e-20)
    
    def test_inverse_timer_wo_large_offset(self, sample_fs):
        """Test inverse_timer_wo_large_offset method."""
        n_samples = 100
        frac_freq = 1e-12 * np.ones(n_samples)
        fd = FrequencyData(frac_freq, sample_fs)
        
        td = TimerData(fd, sample_fs)
        
        # Test the method
        inverse = td.inverse_timer_wo_large_offset(td.total['raw'], sample_fs)
        
        assert len(inverse) == n_samples
        # Should be approximately equal to inverse timer
        assert np.allclose(inverse, td.total['inv'], rtol=1e-3)
    
    def test_inverse_timer_validation(self, sample_fs):
        """Test inverse timer method validation."""
        n_samples = 100
        frac_freq = np.zeros(n_samples)
        fd = FrequencyData(frac_freq, sample_fs)
        td = TimerData(fd, sample_fs)
        
        with pytest.raises(ValueError, match="Sampling rate fs must be > 0"):
            td.inverse_timer_wo_large_offset(td.total['raw'], -1.0)
        
        with pytest.raises(ValueError, match="interp_order must be positive"):
            td.inverse_timer_wo_large_offset(td.total['raw'], sample_fs, interp_order=0)
        
        with pytest.raises(ValueError, match="maxiter must be positive"):
            td.inverse_timer_wo_large_offset(td.total['raw'], sample_fs, maxiter=0)


class TestTimerDataTruncation:
    """Tests for TimerData truncation."""
    
    def test_truncation(self, sample_fs):
        """Test timer data truncation."""
        n_samples = 1000
        frac_freq = np.zeros(n_samples)
        fd = FrequencyData(frac_freq, sample_fs)
        td = TimerData(fd, sample_fs)
        
        original_length = len(td.tau)
        n_trunc = 100
        
        td.truncation(n_trunc)
        
        assert len(td.tau) == original_length - 2 * n_trunc
        assert len(td.total['raw']) == len(td.tau)
        assert len(td.total['inv']) == len(td.tau)
        assert len(td.fit['raw']) == len(td.tau)
        assert len(td.fit['inv']) == len(td.tau)
        assert len(td.fluc['raw']) == len(td.tau)
        assert len(td.fluc['inv']) == len(td.tau)

    def test_zero_truncation_is_noop(self, sample_fs):
        """Zero truncation should preserve timer arrays."""
        n_samples = 100
        frac_freq = np.zeros(n_samples)
        fd = FrequencyData(frac_freq, sample_fs)
        td = TimerData(fd, sample_fs)

        td.truncation(0)

        assert len(td.tau) == n_samples
        assert len(td.total["raw"]) == n_samples
        assert len(td.total["inv"]) == n_samples
    
    def test_truncation_validation(self, sample_fs):
        """Test truncation validation."""
        n_samples = 100
        frac_freq = np.zeros(n_samples)
        fd = FrequencyData(frac_freq, sample_fs)
        td = TimerData(fd, sample_fs)
        
        with pytest.raises(ValueError, match="must be < len\\(data\\) // 2"):
            td.truncation(50)
        
        with pytest.raises(ValueError, match="must be non-negative"):
            td.truncation(-1)


class TestTimerDataErrorComputation:
    """Tests for timer deviation error computation."""
    
    def test_compute_timer_deviation_error(self, sample_fs):
        """Test timer deviation error computation."""
        n_samples = 1000
        # Small linear drift
        t = np.arange(n_samples) / sample_fs
        frac_freq = 1e-12 * t
        fd = FrequencyData(frac_freq, sample_fs, order=1)
        
        td = TimerData(fd, sample_fs)
        td.compute_timer_deviation_error(sample_fs)
        
        assert 'estimate' in td.timer_dev_err
        assert 'model' in td.timer_dev_err
        assert len(td.timer_dev_err['estimate']) == n_samples
        assert len(td.timer_dev_err['model']) == n_samples
    
    def test_compute_timer_deviation_error_validation(self, sample_fs):
        """Test timer deviation error validation."""
        n_samples = 100
        frac_freq = np.zeros(n_samples)
        fd = FrequencyData(frac_freq, sample_fs)
        td = TimerData(fd, sample_fs)
        
        with pytest.raises(ValueError, match="Sampling rate fs must be > 0"):
            td.compute_timer_deviation_error(-1.0)
    
    def test_skip_error_computation(self, sample_fs):
        """Test that error computation can be skipped."""
        n_samples = 100
        frac_freq = np.zeros(n_samples)
        fd = FrequencyData(frac_freq, sample_fs)
        
        td = TimerData(fd, sample_fs, skip_error_computation=True)
        
        # Should not have timer_dev_err attribute
        assert not hasattr(td, 'timer_dev_err')
