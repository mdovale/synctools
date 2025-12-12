# synctools

Python tools for synchronization of multi-channel measurements.

## Overview

`synctools` is a Python package for synchronizing misaligned time series data. It corrects for static time offsets and clock jitter between different measurement systems or chanels, enabling accurate combination and analysis of multi-channel measurements.

The package is designed for applications in precision metrology, interferometry, and gravitational wave detection, where multiple instruments measure signals that need to be synchronized before combination.

## Features

- **Multi-signal synchronization**: Synchronize combinations of 2 or 3 signals 
- **Time offset correction**: Automatically determine and correct time offsets between signals
- **Clock jitter handling**: Account for clock jitter using optional clock reference signals
- **Flexible models**: Support for "total" and "fluc" synchronization models
- **Time and frequency domain optimization**: Optimize in either time or frequency domain
- **Spectral analysis**: Integrated spectral analysis tools

## Installation

### Requirements

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0
- speckit (for spectral analysis)
- pytdi (for time-shifting operations)

### Install from source

```bash
git clone <repository-url>
cd synctools
pip install -e .
```

### Development dependencies

For development and testing:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Synchronizing Two Signals

```python
import numpy as np
from synctools import sync_signals

# Define parameters for spectral analysis
from speckit import compute_spectrum as lpsd
from multiprocessing import Pool

p_lpsd = {
    "olap": "default",
    "bmin": 1,
    "Lmin": 0,
    "Jdes": 500,
    "Kdes": 100,
    "order": 2,
    "win": np.kaiser,
    "psll": 250,
    "pool": Pool()
}

# Create two frequency time series (in Hz)
fs = 10.0  # Sampling rate (Hz)
n_samples = 10000
signal1 = 1e6 + 0.1 * np.sin(2 * np.pi * 0.01 * np.arange(n_samples) / fs)
signal2 = signal1.copy()  # Identical signals (zero offset expected)

# Synchronize the signals
unsynced, synced = sync_signals(
    [signal1, signal2],
    fs=fs,
    p_lpsd=p_lpsd,
    init_offsets=[0.0],  # Initial guess for time offset (seconds)
    model="fluc",        # Use fluctuation model
    domain="time",       # Optimize in time domain
    method="Nelder-Mead"
)

# Access results
print(f"Recovered time offset: {synced.timer_offsets[0]:.6f} seconds")
print(f"Synchronized frequency: {synced.freq['time'][:5]} Hz")
```

### Synchronizing Three Signals

```python
import numpy as np
from synctools import sync_signals

# Create three frequency time series
fs = 10.0
n_samples = 10000
t = np.arange(n_samples) / fs

signal1 = 1e6 + 0.1 * np.sin(2 * np.pi * 0.01 * t)
signal2 = 1e6 + 0.1 * np.sin(2 * np.pi * 0.01 * t)
signal3 = 1e6 + 0.1 * np.sin(2 * np.pi * 0.01 * t)

# Synchronize three signals
unsynced, synced = sync_signals(
    [signal1, signal2, signal3],
    fs=fs,
    p_lpsd=p_lpsd,
    init_offsets=[0.0, 0.0],  # Two offsets for three signals
    model="total",
    domain="freq",
    method="Powell"
)

# Access results
print(f"Time offsets: {synced.timer_offsets} seconds")
print(f"TDIR precision: {synced.TDIR_precision:.2e}")
```

### With Clock References

If you have clock reference signals (differential clock measurements), you can provide them to account for clock jitter:

```python
# Create clock reference arrays (differential clock frequency in Hz)
clock_ref1 = np.random.randn(n_samples) * 1e-9  # Small clock jitter
clock_ref2 = np.random.randn(n_samples) * 1e-9

unsynced, synced = sync_signals(
    [signal1, signal2, signal3],
    fs=fs,
    p_lpsd=p_lpsd,
    clock_refs=[clock_ref1, clock_ref2],  # Clock references for secondary signals
    init_offsets=[0.0, 0.0],
    model="fluc"
)
```

## Core Classes

### FrequencyData

Represents a frequency time series with detrending capabilities:

```python
from synctools import FrequencyData

fd = FrequencyData(
    main_tot=signal_array,  # 1D array, shape (n_samples,), units: Hz
    fs=10.0,                # Sampling rate, units: Hz
    name='signal1',         # Optional name
    order=1                  # Polynomial detrending order
)

# Access attributes
fd.total  # Total frequency (Hz), shape (n_samples,)
fd.fit    # Deterministic (fitted) frequency (Hz), shape (n_samples,)
fd.fluc   # Stochastic (fluctuation) frequency (Hz), shape (n_samples,)
fd.tau    # Time array (s), shape (n_samples,)
fd.fs     # Sampling rate (Hz)
```

### Synchronization

The main synchronization result object:

```python
# After calling sync_signals(), access synced object:
synced.timer_offsets      # Optimized time offsets (s), shape (n_signals-1,)
synced.freq['time']       # Synchronized frequency in time domain (Hz), shape (n_samples,)
synced.freq['asd']        # Frequency ASD (Hz/√Hz), shape (n_freq,)
synced.phase['time']       # Synchronized phase in time domain (rad), shape (n_samples,)
synced.phase['asd']       # Phase ASD (rad/√Hz), shape (n_freq,)
synced.fourier_freq       # Fourier frequencies (Hz), shape (n_freq,)
synced.TDIR_precision     # TDIR precision estimate (dimensionless)
synced.tau                # Time array after truncation (s), shape (n_samples_trunc,)
```

## API Reference

### sync_signals()

Main entry point for signal synchronization.

**Parameters:**
- `in_signals` (List[np.ndarray]): List of frequency signal arrays. Each array must be 1D with shape `(n_samples,)`. Units: Hz. Must contain 2 or 3 signals.
- `fs` (float): Sampling rate. Units: Hz. Must be > 0.
- `p_lpsd` (Dict[str, Any]): SpecKit parameters dictionary for spectral analysis.
- `init_offsets` (Optional[List[float]]): Initial guesses for time offsets. Units: seconds. Length must be `len(in_signals) - 1`. If None, defaults to zeros.
- `model` (str): Synchronization model, either `"total"` or `"fluc"`. Default: `"total"`.
- `domain` (str): Optimization domain, either `"time"` or `"freq"`. Default: `"time"`.
- `method` (str): Optimization method for `scipy.optimize.minimize`. Default: `"Nelder-Mead"`.
- `interp_order` (int): Interpolation order for time-shifting. Must be positive. Default: 121.
- `n_truncate` (Optional[int]): Number of points to truncate at each end. Must satisfy `n_truncate < len(data) // 2`. If None, auto-calculated.
- `clock_refs` (Optional[List[np.ndarray]]): Optional clock reference arrays. Each array must be 1D with shape `(n_samples,)`. Units: Hz. Length must be `len(in_signals) - 1`.
- `logger` (Optional[logging.Logger]): Optional logger instance.

**Returns:**
- `unsynced_obj` (TwoSignals or ThreeSignals): Object containing unsynchronized signal combination.
- `synced_obj` (Synchronization): Object containing synchronized results and optimization output.

**Raises:**
- `ValueError`: If input validation fails (invalid array shapes, incompatible lengths, etc.).

## Examples

See the `notebooks/` directory for detailed examples:
- `0.1_synctools-testing.ipynb`: Basic synchronization examples
- `0.2_synctools-testing.ipynb`: Advanced usage examples

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=synctools
```

## License

BSD 3-Clause License

## Authors

- Miguel Dovale (University of Arizona)

## Citation

If you use synctools in your research, please cite appropriately.
