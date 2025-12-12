# Test Suite for synctools

This directory contains comprehensive tests for the synctools package.

## Test Structure

- `conftest.py`: Shared fixtures and utilities for all tests
- `test_auxiliary.py`: Unit tests for helper functions in `auxiliary.py`
  - `spectra`: Spectral analysis function
  - `integral_rms`: RMS integration function
  - `model_timer_deviation_error`: Timer deviation error model
  - Frequency/phase conversion functions
- `test_frequency.py`: Tests for `FrequencyData` class
  - Linear drifts with known offsets
  - White noise cases with PSD/ASD validation
  - Arithmetic operations
- `test_timer.py`: Tests for `TimerData` class
  - Linear drifts and known offsets
  - White noise integration
  - Inverse timer computation
- `test_sync.py`: Integration tests for `sync_signals` function
  - Known time offsets recovery
  - Doppler shift handling
  - Deterministic datasets with fixed seeds for optimizer testing
  - Different models (total, fluc) and domains (time, freq)

## Running Tests

To run all tests:
```bash
pytest
```

To run specific test files:
```bash
pytest tests/test_auxiliary.py
pytest tests/test_frequency.py
pytest tests/test_timer.py
pytest tests/test_sync.py
```

To run with verbose output:
```bash
pytest -v
```

To run a specific test:
```bash
pytest tests/test_sync.py::TestSyncSignalsKnownOffsets::test_known_time_offset_two_signals
```

## Test Design Principles

1. **Deterministic Testing**: Tests use fixed random seeds for reproducibility
2. **Synthetic Data**: Tests use synthetic signals with known characteristics
3. **Tolerance-Based Assertions**: Numerical tests use appropriate tolerances
4. **Comprehensive Coverage**: Tests cover both basic functionality and edge cases
5. **Integration Testing**: End-to-end tests verify synchronization with known offsets

## Notes

- Tests for numerically tricky parts (optimizers) use small deterministic datasets
- Random seeds are fixed to ensure reproducibility
- White noise tests validate PSD/ASD against analytical expectations
- Integration tests verify recovered offsets within tolerance
