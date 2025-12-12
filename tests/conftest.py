"""
Shared fixtures and utilities for pytest tests.
"""
import numpy as np
import pytest
from typing import Dict, Any


@pytest.fixture
def default_lpsd_params():
    """Default SpecKit parameters for spectral analysis."""
    return {
        "olap": "default",
        "bmin": 1.0,
        "Lmin": 1,
        "Jdes": 100,
        "Kdes": 500,
        "order": 0,
        "win": "Kaiser",
        "psll": 200.0
    }


@pytest.fixture
def sample_fs():
    """Default sampling frequency (Hz)."""
    return 1.0  # 1 Hz


@pytest.fixture
def sample_time_array(sample_fs):
    """Sample time array for testing."""
    duration = 1000.0  # seconds
    n_samples = int(duration * sample_fs)
    return np.arange(n_samples) / sample_fs


@pytest.fixture
def fixed_seed():
    """Fix random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def white_noise_signal(sample_time_array, fixed_seed):
    """Generate white noise signal with known variance."""
    n_samples = len(sample_time_array)
    # White noise with unit variance
    noise = np.random.randn(n_samples)
    return noise


@pytest.fixture
def linear_drift_signal(sample_time_array):
    """Generate signal with linear drift."""
    slope = 1e-6  # Hz/s
    offset = 1e6  # Hz
    return offset + slope * sample_time_array


@pytest.fixture
def synthetic_frequency_data(sample_time_array, sample_fs):
    """Create synthetic frequency data with known characteristics."""
    n_samples = len(sample_time_array)
    # Constant frequency with small white noise
    base_freq = 1e6  # Hz
    noise_level = 1.0  # Hz
    noise = noise_level * np.random.randn(n_samples)
    return base_freq + noise


def generate_synthetic_signal_with_offset(
    fs: float,
    duration: float,
    base_freq: float,
    time_offset: float,
    doppler_shift: float = 0.0,
    noise_level: float = 0.0,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic frequency signal with known time offset and Doppler shift.
    
    Args:
        fs: Sampling frequency (Hz)
        duration: Signal duration (s)
        base_freq: Base frequency (Hz)
        time_offset: Time offset to apply (s)
        doppler_shift: Doppler shift factor (dimensionless, e.g., 1e-9)
        noise_level: White noise level (Hz)
        seed: Random seed
    
    Returns:
        Frequency signal array (Hz)
    """
    np.random.seed(seed)
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    
    # Generate base signal
    signal = base_freq * np.ones(n_samples)
    
    # Apply Doppler shift (frequency scaling)
    signal *= (1.0 + doppler_shift)
    
    # Add noise
    if noise_level > 0:
        signal += noise_level * np.random.randn(n_samples)
    
    return signal


@pytest.fixture
def synthetic_signal_generator():
    """Fixture that returns the signal generator function."""
    return generate_synthetic_signal_with_offset
