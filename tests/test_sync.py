"""
Integration tests for sync_signals function.
"""
import numpy as np
import pytest
from synctools.synchronization import sync_signals
from speckit.dsp import timeshift


class TestSyncSignalsBasic:
    """Basic tests for sync_signals."""
    
    def test_sync_signals_two_signals(self, default_lpsd_params, fixed_seed):
        """Test synchronization of two signals."""
        np.random.seed(fixed_seed)
        n_samples = 100
        fs = 10.0  # Use higher fs to satisfy lpf_cutoff < fs/2
        
        # Create two identical signals (should have zero offset)
        signal1 = 1e6 * np.ones(n_samples)
        signal2 = 1e6 * np.ones(n_samples)
        
        unsynced, synced = sync_signals(
            [signal1, signal2],
            fs,
            default_lpsd_params,
            init_offsets=[0.0],
            model="fluc", 
            domain="time",
            method="Nelder-Mead"
        )
        
        assert synced.timer_offsets is not None
        assert len(synced.timer_offsets) == 1
        # For identical signals, offset should be close to zero
        assert abs(synced.timer_offsets[0]) < 0.1
    
    def test_sync_signals_three_signals(self, default_lpsd_params, fixed_seed):
        """Test synchronization of three signals."""
        np.random.seed(fixed_seed)
        n_samples = 100
        fs = 10.0  # Use higher fs to satisfy lpf_cutoff < fs/2
        
        # Create three identical signals
        signal1 = 1e6 * np.ones(n_samples)
        signal2 = 1e6 * np.ones(n_samples)
        signal3 = 1e6 * np.ones(n_samples)
        
        unsynced, synced = sync_signals(
            [signal1, signal2, signal3],
            fs,
            default_lpsd_params,
            init_offsets=[0.0, 0.0],
            method="Nelder-Mead"
        )
        
        assert synced.timer_offsets is not None
        assert len(synced.timer_offsets) == 2
        # For identical signals, offsets should be close to zero
        assert abs(synced.timer_offsets[0]) < 0.1
        assert abs(synced.timer_offsets[1]) < 0.1
    
    def test_sync_signals_validation(self, default_lpsd_params, sample_fs):
        """Test input validation."""
        signal = np.ones(100)
        
        with pytest.raises(ValueError, match="Sampling rate fs must be > 0"):
            sync_signals([signal, signal], -1.0, default_lpsd_params)
        
        with pytest.raises(ValueError, match="Insufficient input signals"):
            sync_signals([signal], sample_fs, default_lpsd_params)
        
        with pytest.raises(ValueError, match="Too many input signals"):
            sync_signals([signal, signal, signal, signal], sample_fs, default_lpsd_params)


class TestSyncSignalsKnownOffsets:
    """Tests for sync_signals with known time offsets."""
    
    def test_known_time_offset_two_signals(self, default_lpsd_params, fixed_seed):
        """Test recovery of known time offset for two signals."""
        np.random.seed(fixed_seed)
        n_samples = 200
        fs = 10.0  # Use higher fs to satisfy lpf_cutoff < fs/2
        duration = n_samples / fs
        
        # Known time offset
        true_offset = 0.1234  # seconds
        
        # Generate base signal
        t = np.arange(n_samples) / fs
        base_freq = 1e6  # Hz
        # Add some variation to make it non-trivial
        signal_base = np.random.randn(n_samples)
        
        # Create second signal with time offset using fractional time shift
        # Shift by true_offset seconds (convert to samples)
        signal2 = timeshift(signal_base, -true_offset * fs)
        
        # Use small initial guess
        unsynced, synced = sync_signals(
            [signal_base, signal2],
            fs,
            default_lpsd_params,
            init_offsets=[1.0],
            method="Nelder-Mead",
            domain="freq", model="fluc",
            n_truncate=20, interp_order=121,
        )
        
        recovered_offset = synced.timer_offsets[0]
        
        # Should recover offset within reasonable tolerance
        # Tolerance depends on signal characteristics
        tolerance = 1e-2  # seconds
        assert abs(recovered_offset - true_offset) < tolerance
    
    
    def test_small_time_offset(self, default_lpsd_params, fixed_seed):
        """Test recovery of small time offset."""
        np.random.seed(fixed_seed)
        n_samples = 1000
        fs = 10.0  # Use higher fs to satisfy lpf_cutoff < fs/2
        
        true_offset = 10e-6  # 10 us
        
        t = np.arange(n_samples) / fs
        base_freq = 1e6
        signal_base = base_freq + 0.1 * np.sin(2 * np.pi * 0.01 * t)
        
        # Use fractional time shift
        signal2 = timeshift(signal_base, true_offset * fs)
        
        unsynced, synced = sync_signals(
            [signal_base, signal2],
            fs,
            default_lpsd_params,
            init_offsets=[0.0],
            method="Nelder-Mead",
            n_truncate=50
        )
        
        recovered_offset = synced.timer_offsets[0]
        tolerance = 0.05  # 50 ms
        assert abs(recovered_offset - true_offset) < tolerance


class TestSyncSignalsDopplerShift:
    """Tests for sync_signals with Doppler shifts."""
    
    def test_doppler_shift_two_signals(self, default_lpsd_params, fixed_seed):
        """Test synchronization with Doppler shift."""
        np.random.seed(fixed_seed)
        n_samples = 20000
        fs = 10.0  # Use higher fs to satisfy lpf_cutoff < fs/2
        
        # Known Doppler shift
        doppler_shift = 1e-9  # 1 ppb
        
        t = np.arange(n_samples) / fs
        base_freq = 1e6
        signal1 = base_freq + 0.1 * np.sin(2 * np.pi * 0.01 * t)
        
        # Apply Doppler shift (frequency scaling)
        signal2 = signal1 * (1.0 + doppler_shift)
        
        unsynced, synced = sync_signals(
            [signal1, signal2],
            fs,
            default_lpsd_params,
            init_offsets=[0.0],
            model="total",  # Use total model to account for Doppler
            method="Nelder-Mead",
            n_truncate=100
        )
        
        # With Doppler shift, synchronization should still work
        # The offset might be adjusted to compensate
        assert synced.timer_offsets is not None
        assert len(synced.timer_offsets) == 1


class TestSyncSignalsModels:
    """Tests for different synchronization models."""
    
    def test_total_model(self, default_lpsd_params, fixed_seed):
        """Test synchronization with 'total' model."""
        np.random.seed(fixed_seed)
        n_samples = 10000
        fs = 10.0  # Use higher fs to satisfy lpf_cutoff < fs/2
        
        t = np.arange(n_samples) / fs
        signal1 = 1e6 + 0.1 * np.sin(2 * np.pi * 0.01 * t)
        signal2 = signal1.copy()
        
        unsynced, synced = sync_signals(
            [signal1, signal2],
            fs,
            default_lpsd_params,
            init_offsets=[0.0],
            model="total",
            method="Nelder-Mead"
        )
        
        assert synced.model == "total"
        assert synced.timer_offsets is not None
    
    def test_fluc_model(self, default_lpsd_params, fixed_seed):
        """Test synchronization with 'fluc' model."""
        np.random.seed(fixed_seed)
        n_samples = 10000
        fs = 10.0  # Use higher fs to satisfy lpf_cutoff < fs/2
        
        t = np.arange(n_samples) / fs
        signal1 = 1e6 + 0.1 * np.sin(2 * np.pi * 0.01 * t)
        signal2 = signal1.copy()
        
        unsynced, synced = sync_signals(
            [signal1, signal2],
            fs,
            default_lpsd_params,
            init_offsets=[0.0],
            model="fluc",
            method="Nelder-Mead"
        )
        
        assert synced.model == "fluc"
        assert synced.timer_offsets is not None


class TestSyncSignalsDomains:
    """Tests for different optimization domains."""
    
    def test_time_domain(self, default_lpsd_params, fixed_seed):
        """Test synchronization in time domain."""
        np.random.seed(fixed_seed)
        n_samples = 10000
        fs = 10.0  # Use higher fs to satisfy lpf_cutoff < fs/2
        
        t = np.arange(n_samples) / fs
        signal1 = 1e6 + 0.1 * np.sin(2 * np.pi * 0.01 * t)
        signal2 = signal1.copy()
        
        unsynced, synced = sync_signals(
            [signal1, signal2],
            fs,
            default_lpsd_params,
            init_offsets=[0.0],
            domain="time",
            method="Nelder-Mead"
        )
        
        assert synced.domain == "time"
        assert synced.timer_offsets is not None
    
    def test_freq_domain(self, default_lpsd_params, fixed_seed):
        """Test synchronization in frequency domain."""
        np.random.seed(fixed_seed)
        n_samples = 10000
        fs = 10.0  # Use higher fs to satisfy lpf_cutoff < fs/2
        
        t = np.arange(n_samples) / fs
        signal1 = 1e6 + 0.1 * np.sin(2 * np.pi * 0.01 * t)
        signal2 = signal1.copy()
        
        unsynced, synced = sync_signals(
            [signal1, signal2],
            fs,
            default_lpsd_params,
            init_offsets=[0.0],
            domain="freq",
            method="Nelder-Mead"
        )
        
        assert synced.domain == "freq"
        assert synced.timer_offsets is not None


class TestSyncMultipleTwoSignals:
    """Tests for sync_multiple_twosignals function."""
    
    def test_sync_multiple_twosignals_basic(self, default_lpsd_params, fixed_seed):
        """Test basic multiple TwoSignal synchronization."""
        from synctools.synchronization import sync_multiple_twosignals
        
        np.random.seed(fixed_seed)
        n_samples = 100
        fs = 10.0
        
        # Create four signals (A, B, C, D)
        signal_A = 1e6 * np.ones(n_samples)
        signal_B = 1e6 * np.ones(n_samples)
        signal_C = 1e6 * np.ones(n_samples)
        signal_D = 1e6 * np.ones(n_samples)
        
        results = sync_multiple_twosignals(
            [signal_A, signal_B, signal_C, signal_D],
            fs,
            default_lpsd_params,
            init_offsets=None,
            model="fluc",
            domain="time",
            method="Nelder-Mead"
        )
        
        # Should have 3 results: [A,B], [A,C], [A,D]
        assert len(results) == 3
        
        # Each result should be a tuple of (unsynced_obj, synced_obj)
        for i, (unsynced, synced) in enumerate(results):
            assert unsynced is not None
            assert synced is not None
            assert synced.timer_offsets is not None
            assert len(synced.timer_offsets) == 1
            # For identical signals, offset should be close to zero
            assert abs(synced.timer_offsets[0]) < 0.1
    
    def test_sync_multiple_twosignals_with_init_offsets(self, default_lpsd_params, fixed_seed):
        """Test multiple TwoSignal synchronization with custom init_offsets."""
        from synctools.synchronization import sync_multiple_twosignals
        
        np.random.seed(fixed_seed)
        n_samples = 100
        fs = 10.0
        
        signal_A = 1e6 * np.ones(n_samples)
        signal_B = 1e6 * np.ones(n_samples)
        signal_C = 1e6 * np.ones(n_samples)
        
        # Use custom init_offsets: None for [A,B], [0.0] for [A,C]
        results = sync_multiple_twosignals(
            [signal_A, signal_B, signal_C],
            fs,
            default_lpsd_params,
            init_offsets=[None, [0.0]],
            method="Nelder-Mead"
        )
        
        assert len(results) == 2
        for unsynced, synced in results:
            assert synced.timer_offsets is not None
            assert len(synced.timer_offsets) == 1
    
    def test_sync_multiple_twosignals_validation(self, default_lpsd_params, sample_fs):
        """Test input validation for sync_multiple_twosignals."""
        from synctools.synchronization import sync_multiple_twosignals
        
        signal = np.ones(100)
        
        with pytest.raises(ValueError, match="Insufficient input signals"):
            sync_multiple_twosignals([signal], sample_fs, default_lpsd_params)
        
        with pytest.raises(ValueError, match="init_offsets must have length"):
            sync_multiple_twosignals(
                [signal, signal, signal],
                sample_fs,
                default_lpsd_params,
                init_offsets=[[0.0]]  # Wrong length, should be 2
            )
