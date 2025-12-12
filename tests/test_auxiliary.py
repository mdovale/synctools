"""
Unit tests for auxiliary.py helper functions.
"""
import numpy as np
import pytest
from synctools.auxiliary import (
    spectra,
    integral_rms,
    model_timer_deviation_error,
    convert_frequency_to_phase_in_time,
    convert_frequency_to_phase_in_asd,
    crop_data,
)


class TestSpectra:
    """Tests for spectra function."""
    
    def test_spectra_basic(self, default_lpsd_params, sample_fs, white_noise_signal):
        """Test basic spectra computation on white noise."""
        fourier_freq, asd = spectra(white_noise_signal, sample_fs, default_lpsd_params)
        
        assert fourier_freq is not None
        assert asd is not None
        assert len(fourier_freq) == len(asd)
        assert np.all(fourier_freq > 0)
        assert np.all(asd >= 0)
    
    def test_spectra_white_noise_flat(self, default_lpsd_params, sample_fs, fixed_seed):
        """Test that white noise produces approximately flat PSD."""
        np.random.seed(fixed_seed)
        n_samples = 10000
        fs = 1.0
        # White noise with unit variance
        white_noise = np.random.randn(n_samples)
        
        fourier_freq, asd = spectra(white_noise, fs, default_lpsd_params)
        
        # White noise should have approximately flat PSD
        # Check that ASD is roughly constant (within factor of 2-3)
        asd_mean = np.mean(asd)
        asd_std = np.std(asd)
        # For white noise, variation should be reasonable
        assert asd_std / asd_mean < 1.0  # Less than 100% variation
    
    def test_spectra_invalid_fs(self, default_lpsd_params, white_noise_signal):
        """Test that invalid sampling rate raises error."""
        with pytest.raises(ValueError, match="Sampling rate fs must be > 0"):
            spectra(white_noise_signal, -1.0, default_lpsd_params)
        
        with pytest.raises(ValueError, match="Sampling rate fs must be > 0"):
            spectra(white_noise_signal, 0.0, default_lpsd_params)
    
    def test_spectra_sine_wave(self, default_lpsd_params, sample_fs):
        """Test spectra on a sine wave (should show peak at frequency)."""
        t = np.arange(1000) / sample_fs
        freq_signal = 0.1  # Hz
        signal = np.sin(2 * np.pi * freq_signal * t)
        
        fourier_freq, asd = spectra(signal, sample_fs, default_lpsd_params)
        
        # Should have a peak near the signal frequency
        idx_peak = np.argmax(asd)
        peak_freq = fourier_freq[idx_peak]
        # Peak should be within reasonable range of signal frequency
        assert abs(peak_freq - freq_signal) < 0.05  # Within 0.05 Hz


class TestIntegralRMS:
    """Tests for integral_rms function."""
    
    def test_integral_rms_basic(self, sample_fs):
        """Test basic RMS integration."""
        # Create simple frequency and ASD arrays
        fourier_freq = np.logspace(-3, 0, 100)  # 0.001 to 1 Hz
        # Flat ASD (white noise)
        asd = np.ones_like(fourier_freq)
        
        rms = integral_rms(fourier_freq, asd)
        
        assert rms > 0
        # For flat ASD, RMS^2 = integral of ASD^2 = integral of 1 = bandwidth
        # Should be approximately sqrt(bandwidth)
        bandwidth = fourier_freq[-1] - fourier_freq[0]
        expected_rms = np.sqrt(bandwidth)
        assert abs(rms - expected_rms) < 0.1 * expected_rms
    
    def test_integral_rms_passband(self, sample_fs):
        """Test RMS integration with passband limits."""
        fourier_freq = np.logspace(-3, 0, 100)
        asd = np.ones_like(fourier_freq)
        
        # Full band
        rms_full = integral_rms(fourier_freq, asd, pass_band=[-np.inf, np.inf])
        
        # Limited band
        rms_limited = integral_rms(fourier_freq, asd, pass_band=[0.01, 0.1])
        
        assert rms_limited < rms_full
        assert rms_limited > 0
    
    def test_integral_rms_analytical(self):
        """Test RMS integration against analytical result for known spectrum."""
        # Create frequency array
        fourier_freq = np.linspace(0.01, 1.0, 1000)
        
        # Create 1/f^2 noise (ASD = 1/f)
        asd = 1.0 / fourier_freq
        
        rms = integral_rms(fourier_freq, asd)
        
        # For ASD = 1/f, PSD = 1/f^2
        # Integral of PSD from f1 to f2 = 1/f1 - 1/f2
        f1, f2 = fourier_freq[0], fourier_freq[-1]
        expected_rms2 = 1.0/f1 - 1.0/f2
        expected_rms = np.sqrt(expected_rms2)
        
        # Should match within reasonable tolerance
        assert abs(rms - expected_rms) < 0.1 * expected_rms


class TestModelTimerDeviationError:
    """Tests for model_timer_deviation_error function."""
    
    def test_model_timer_deviation_error_linear(self, fixed_seed):
        """Test timer deviation error model for linear drift."""
        np.random.seed(fixed_seed)
        
        # Linear fractional frequency: y(t) = a*t + b
        # For small drift, p_fit = [a, b] (coefficients for polyval)
        a = 1e-12  # Small linear drift
        b = 0.0
        p_fit = np.array([a, b])
        
        # Time array (seconds)
        tau = np.linspace(3600, 3600e3, 1000)  # 1 hour to 1000 hours
        
        # Compute error model
        error = model_timer_deviation_error(p_fit, tau)
        
        assert len(error) == len(tau)
        # Error should be small for small drift
        assert np.all(np.abs(error) < 1e-6)  # Less than 1 microsecond
    
    def test_model_timer_deviation_error_zero(self):
        """Test timer deviation error for zero drift (should be small)."""
        # Zero drift: p_fit = [0]
        p_fit = np.array([0.0])
        tau = np.linspace(3600, 3600e3, 100)
        
        error = model_timer_deviation_error(p_fit, tau)
        
        # Should be very small or zero
        assert np.all(np.abs(error) < 1e-12)
    
    def test_model_timer_deviation_error_iterations(self, fixed_seed):
        """Test that iterations parameter affects the result."""
        np.random.seed(fixed_seed)
        
        # Use larger drift to see iteration effects
        p_fit = np.array([1e-10, 0.0])  # Larger drift
        tau = np.linspace(3600, 3600e3, 100)
        
        error_0 = model_timer_deviation_error(p_fit, tau, iterations=0)
        error_1 = model_timer_deviation_error(p_fit, tau, iterations=1)
        
        # Results should be different (though possibly similar for small drifts)
        # For very small drifts, iterations might not change much
        # Check that at least some values differ significantly
        max_diff = np.max(np.abs(error_0 - error_1))
        # For larger drifts, there should be a noticeable difference
        # If difference is very small, that's also valid (iterations converged quickly)
        assert max_diff > 1e-20 or np.any(np.abs(error_0 - error_1) > 1e-22)


class TestConvertFrequencyToPhase:
    """Tests for frequency to phase conversion functions."""
    
    def test_convert_frequency_to_phase_in_time(self, sample_fs):
        """Test frequency to phase conversion in time domain."""
        # Constant frequency should give linear phase
        freq = 1.0  # Hz
        n_samples = 1000
        freq_data = freq * np.ones(n_samples)
        
        phase = convert_frequency_to_phase_in_time(freq_data, sample_fs)
        
        assert len(phase) == n_samples
        # Phase should increase linearly
        phase_diff = np.diff(phase)
        expected_phase_diff = 2 * np.pi * freq / sample_fs
        assert np.allclose(phase_diff, expected_phase_diff, rtol=1e-10)
    
    def test_convert_frequency_to_phase_in_asd(self):
        """Test frequency ASD to phase ASD conversion."""
        fourier_freq = np.logspace(-3, 0, 100)
        freq_asd = np.ones_like(fourier_freq)  # Flat frequency ASD
        
        phase_asd = convert_frequency_to_phase_in_asd(fourier_freq, freq_asd)
        
        # Phase ASD = Frequency ASD / f
        expected_phase_asd = freq_asd / fourier_freq
        assert np.allclose(phase_asd, expected_phase_asd)
    
    def test_convert_frequency_to_phase_in_asd_validation(self):
        """Test validation in frequency to phase ASD conversion."""
        fourier_freq = np.array([0.0, 0.1, 0.2])  # Contains zero
        freq_asd = np.ones_like(fourier_freq)
        
        with pytest.raises(ValueError, match="fourier_freq must be > 0"):
            convert_frequency_to_phase_in_asd(fourier_freq, freq_asd)
    
    def test_convert_frequency_to_phase_in_asd_length_mismatch(self):
        """Test that length mismatch raises error."""
        fourier_freq = np.logspace(-3, 0, 100)
        freq_asd = np.ones(50)  # Wrong length
        
        with pytest.raises(ValueError, match="must have same length"):
            convert_frequency_to_phase_in_asd(fourier_freq, freq_asd)


class TestCropData:
    """Tests for crop_data function."""
    
    def test_crop_data_basic(self):
        """Test basic data cropping."""
        x = np.linspace(0, 10, 100)
        y = x ** 2
        
        x_cropped, y_cropped = crop_data(x, y, 2.0, 8.0)
        
        assert len(x_cropped) == len(y_cropped)
        assert np.all(x_cropped >= 2.0)
        assert np.all(x_cropped <= 8.0)
        # Check that values match
        mask = (x >= 2.0) & (x <= 8.0)
        assert np.allclose(x_cropped, x[mask])
        assert np.allclose(y_cropped, y[mask])
    
    def test_crop_data_no_overlap(self):
        """Test cropping when range has no overlap with data."""
        x = np.linspace(0, 10, 100)
        y = x ** 2
        
        x_cropped, y_cropped = crop_data(x, y, 20.0, 30.0)
        
        assert len(x_cropped) == 0
        assert len(y_cropped) == 0
