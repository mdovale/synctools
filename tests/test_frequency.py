"""
Tests for FrequencyData class.
"""
import numpy as np
import pytest
from synctools.frequency import FrequencyData
from synctools.auxiliary import spectra


class TestFrequencyDataBasic:
    """Basic tests for FrequencyData."""
    
    def test_frequency_data_creation(self, sample_fs, synthetic_frequency_data):
        """Test basic FrequencyData creation."""
        fd = FrequencyData(synthetic_frequency_data, sample_fs)
        
        assert len(fd.total) == len(synthetic_frequency_data)
        assert fd.fs == sample_fs
        assert len(fd.tau) == len(fd.total)
        assert len(fd.fit) == len(fd.total)
        assert len(fd.fluc) == len(fd.total)
    
    def test_frequency_data_validation(self, sample_fs):
        """Test input validation."""
        with pytest.raises(ValueError, match="Sampling rate fs must be > 0"):
            FrequencyData(np.array([1.0, 2.0]), -1.0)
        
        with pytest.raises(ValueError, match="main_tot must be 1D array"):
            FrequencyData(np.array([[1.0, 2.0], [3.0, 4.0]]), sample_fs)
        
        with pytest.raises(ValueError, match="main_tot cannot be empty"):
            FrequencyData(np.array([]), sample_fs)
        
        with pytest.raises(ValueError, match="order must be non-negative"):
            FrequencyData(np.array([1.0, 2.0]), sample_fs, order=-1)
    
class TestFrequencyDataLinearDrift:
    """Tests for FrequencyData with linear drifts."""
    
    def test_linear_drift_detrending(self, sample_fs, sample_time_array):
        """Test that linear drift is properly detrended."""
        # Create signal with linear drift
        slope = 1e-3  # Hz/s
        offset = 1e6  # Hz
        freq_data = offset + slope * sample_time_array
        
        fd = FrequencyData(freq_data, sample_fs, order=1)
        
        # Fit should capture the linear trend
        # Check that fit is approximately linear
        fit_slope = np.polyfit(fd.tau, fd.fit, 1)[0]
        assert abs(fit_slope - slope) < 0.01 * abs(slope)
        
        # Fluctuation should have zero mean (approximately)
        # Use more lenient tolerance due to numerical precision in polynomial fitting
        assert abs(np.mean(fd.fluc)) < 0.3 * np.std(fd.fluc)
    
    def test_linear_drift_known_offset(self, sample_fs):
        """Test FrequencyData with known constant offset."""
        n_samples = 1000
        constant_freq = 1e6  # Hz
        freq_data = constant_freq * np.ones(n_samples)
        
        fd = FrequencyData(freq_data, sample_fs, order=0)
        
        # For constant frequency, fit should be constant
        assert np.allclose(fd.fit, constant_freq, rtol=1e-10)
        # Fluctuation should be approximately zero
        assert np.std(fd.fluc) < 1e-10
    
    def test_quadratic_drift(self, sample_fs):
        """Test FrequencyData with quadratic drift."""
        n_samples = 1000
        t = np.arange(n_samples) / sample_fs
        # Quadratic: f(t) = a*t^2 + b*t + c
        a, b, c = 1e-6, 1e-3, 1e6
        freq_data = a * t**2 + b * t + c
        
        fd = FrequencyData(freq_data, sample_fs, order=2)
        
        # Fit should capture the quadratic trend
        p_fit = np.polyfit(fd.tau, fd.fit, 2)
        assert abs(p_fit[0] - a) < 0.01 * abs(a)
        assert abs(p_fit[1] - b) < 0.01 * abs(b)
        assert abs(p_fit[2] - c) < 0.01 * abs(c)


class TestFrequencyDataWhiteNoise:
    """Tests for FrequencyData with white noise."""
    
    def test_white_noise_psd(self, sample_fs, default_lpsd_params, fixed_seed):
        """Test that white noise produces flat PSD."""
        np.random.seed(fixed_seed)
        n_samples = 10000
        fs = 1.0
        
        # White noise with unit variance
        white_noise = np.random.randn(n_samples)
        # Scale to reasonable frequency units (Hz)
        freq_data = 1e6 + 1.0 * white_noise
        
        fd = FrequencyData(freq_data, fs, order=0)
        fd.compute_spectrum(default_lpsd_params)
        
        assert fd.fourier_freq is not None
        assert fd.asd is not None
        assert len(fd.fourier_freq) == len(fd.asd)
        
        # For white noise, PSD should be approximately flat
        # Check that ASD doesn't vary too much (within factor of 2-3)
        asd_mean = np.mean(fd.asd)
        asd_std = np.std(fd.asd)
        # Coefficient of variation should be reasonable
        assert asd_std / asd_mean < 1.0
    
    def test_white_noise_analytical_check(self, sample_fs, default_lpsd_params, fixed_seed):
        """Test white noise PSD against analytical expectation."""
        np.random.seed(fixed_seed)
        n_samples = 50000
        fs = 1.0
        noise_level = 1.0  # Hz
        
        # White noise
        freq_data = noise_level * np.random.randn(n_samples)
        
        fd = FrequencyData(freq_data, fs, order=0)
        fd.compute_spectrum(default_lpsd_params)
        
        # For white noise with variance sigma^2, PSD should be approximately sigma^2/fs
        # ASD = sqrt(PSD) = sigma/sqrt(fs)
        expected_asd = noise_level / np.sqrt(fs)
        
        # Check that mean ASD is close to expected (within factor of 2)
        mean_asd = np.mean(fd.asd)
        assert 0.5 * expected_asd < mean_asd < 2.0 * expected_asd
    
    def test_white_noise_rms_integration(self, sample_fs, default_lpsd_params, fixed_seed):
        """Test RMS integration for white noise."""
        np.random.seed(fixed_seed)
        n_samples = 10000
        fs = 1.0
        noise_level = 1.0
        
        freq_data = noise_level * np.random.randn(n_samples)
        fd = FrequencyData(freq_data, fs, order=0)
        fd.compute_spectrum(default_lpsd_params)
        
        # RMS from time domain
        rms_time = np.std(fd.total)
        
        # RMS from frequency domain (integrate PSD)
        from synctools.auxiliary import integral_rms
        rms_freq = integral_rms(fd.fourier_freq, fd.asd)
        
        # Should be approximately equal (within reasonable tolerance)
        assert abs(rms_time - rms_freq) < 0.2 * rms_time


class TestFrequencyDataOperations:
    """Tests for FrequencyData arithmetic operations."""
    
    def test_frequency_data_addition(self, sample_fs):
        """Test FrequencyData addition."""
        n_samples = 100
        fd1 = FrequencyData(np.ones(n_samples) * 1e6, sample_fs)
        fd2 = FrequencyData(np.ones(n_samples) * 2e6, sample_fs)
        
        fd_sum = fd1 + fd2
        
        assert np.allclose(fd_sum.total, 3e6)
    
    def test_frequency_data_subtraction(self, sample_fs):
        """Test FrequencyData subtraction."""
        n_samples = 100
        fd1 = FrequencyData(np.ones(n_samples) * 3e6, sample_fs)
        fd2 = FrequencyData(np.ones(n_samples) * 1e6, sample_fs)
        
        fd_diff = fd1 - fd2
        
        assert np.allclose(fd_diff.total, 2e6)
    
    def test_frequency_data_scalar_operations(self, sample_fs):
        """Test FrequencyData scalar operations."""
        n_samples = 100
        fd = FrequencyData(np.ones(n_samples) * 1e6, sample_fs)
        
        # Multiplication
        fd_mult = fd * 2.0
        assert np.allclose(fd_mult.total, 2e6)
        
        # Division
        fd_div = fd / 2.0
        assert np.allclose(fd_div.total, 0.5e6)
        
        # Addition
        fd_add = fd + 1e5
        assert np.allclose(fd_add.total, 1.1e6)


class TestFrequencyDataTruncation:
    """Tests for FrequencyData truncation."""
    
    def test_truncation(self, sample_fs):
        """Test data truncation."""
        n_samples = 1000
        fd = FrequencyData(np.ones(n_samples) * 1e6, sample_fs)
        
        original_length = len(fd.total)
        n_trunc = 100
        
        fd.truncation(n_trunc)
        
        assert len(fd.total) == original_length - 2 * n_trunc
        assert len(fd.tau) == len(fd.total)
        assert len(fd.fit) == len(fd.total)
        assert len(fd.fluc) == len(fd.total)
    
    def test_truncation_validation(self, sample_fs):
        """Test truncation validation."""
        n_samples = 100
        fd = FrequencyData(np.ones(n_samples) * 1e6, sample_fs)
        
        with pytest.raises(ValueError, match="must be < len\\(data\\) // 2"):
            fd.truncation(50)  # n_trunc >= len(data) // 2
        
        with pytest.raises(ValueError, match="must be non-negative"):
            fd.truncation(-1)
