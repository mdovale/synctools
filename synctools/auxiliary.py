# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale (University of Arizona).

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
"""Auxiliary numerical helpers for synctools.

The root ``synctools`` namespace is the stable public API. Helpers exported
from this module are available for expert workflows, examples, and tests, but
they remain provisional until the v1.0 API policy is finalized.
"""

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import scipy.optimize as optimize
from scipy import integrate, signal
from speckit import compute_spectrum as lpsd

logger = logging.getLogger(__name__)

__all__ = [
    "build_kaiser_lpf_taps",
    "combination_2sig",
    "combination_3sig",
    "components_for_balancing",
    "convert_frequency_to_detrended_phase_in_time",
    "convert_frequency_to_phase_in_asd",
    "convert_frequency_to_phase_in_time",
    "convert_phase_to_frequency_in_time",
    "crop_data",
    "derive_sign_pairs",
    "get_asd_delay_factor",
    "integral_rms",
    "model_timer_deviation_error",
    "spectra",
]

_REQUIRED_LPSD_KEYS = ("olap", "bmin", "Lmin", "Jdes", "Kdes", "order", "win", "psll")
_DEFAULT_2SIG_SIGNS = (1, -1)
_DEFAULT_3SIG_SIGNS = (1, 1, -1)


def _empty_float_array() -> npt.NDArray[np.float64]:
    return np.array([], dtype=np.float64)


def _empty_spectrum() -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    return _empty_float_array(), _empty_float_array()


def _validate_positive_float(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}") from exc
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return number


def _as_1d_float_array(data: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_2d_float_array(
    data: npt.ArrayLike,
    name: str,
    expected_columns: int,
) -> npt.NDArray[np.float64]:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != expected_columns:
        raise ValueError(
            f"{name} must have shape (n_samples, {expected_columns}), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_sign_array(
    signs: Optional[npt.ArrayLike],
    default: Sequence[int],
) -> npt.NDArray[np.int_]:
    sign_array = np.asarray(default if signs is None else signs, dtype=np.int_)
    if sign_array.shape != (len(default),):
        raise ValueError(f"signs must have shape ({len(default)},), got {sign_array.shape}")
    if not np.all(np.isin(sign_array, (-1, 1))):
        raise ValueError("signs must contain only -1 or +1 values")
    return sign_array


def spectra(
    data: npt.ArrayLike,
    fs: float,
    p_lpsd: Dict[str, Any]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute amplitude spectral density (ASD) using SpecKit.
    
    Args:
        data: Input time series data. Units depend on input type.
        fs: Sampling rate (Hz). Must be > 0.
        p_lpsd: SpecKit parameters dictionary.
    
    Returns:
        Tuple of (fourier_freq, asd) where:
        - fourier_freq: Fourier frequency array (Hz)
        - asd: Amplitude spectral density array (same units as input / √Hz)
    """
    sampling_rate = _validate_positive_float(fs, "Sampling rate fs")
    missing_keys = [key for key in _REQUIRED_LPSD_KEYS if key not in p_lpsd]
    if missing_keys:
        raise ValueError(f"p_lpsd is missing required keys: {missing_keys}")

    time_series = _as_1d_float_array(data, "data")
    data_len = time_series.size
    
    # Handle empty data
    if data_len == 0:
        logger.warning("Empty data array provided to spectra, returning empty arrays")
        return _empty_spectrum()
    
    # Warn for suspicious but not fatal situations
    if sampling_rate < 0.1:
        logger.warning(
            "Very small sampling rate fs=%s Hz may lead to poor spectral estimation",
            sampling_rate,
        )
    
    if data_len < 100:
        logger.warning(
            "Short data sequence (length=%s) may lead to poor spectral estimation",
            data_len,
        )
    
    # Check if data length is too short for SpecKit parameters
    # Lmin is the minimum segment length, need at least one segment
    Lmin = p_lpsd.get("Lmin", 100)
    if data_len < Lmin:
        logger.warning(
            "Data length (%s) is shorter than Lmin (%s), returning empty arrays",
            data_len,
            Lmin,
        )
        return _empty_spectrum()
    
    try:
        psd = lpsd(
            time_series,
            sampling_rate,
            olap=p_lpsd["olap"],
            bmin=p_lpsd["bmin"],
            Lmin=p_lpsd["Lmin"],
            Jdes=p_lpsd["Jdes"],
            Kdes=p_lpsd["Kdes"],
            order=p_lpsd["order"],
            win=p_lpsd["win"],
            psll=p_lpsd["psll"],
            verbose=False,
        )
    except (ZeroDivisionError, ValueError, FloatingPointError) as exc:
        logger.warning("SpecKit computation failed: %s, returning empty arrays", exc)
        return _empty_spectrum()

    fourier_freq = np.asarray(psd.f, dtype=np.float64)
    psd_values = np.asarray(psd.Gxx, dtype=np.float64)
    if fourier_freq.ndim != 1 or psd_values.shape != fourier_freq.shape:
        logger.warning("SpecKit returned inconsistent spectrum shapes, returning empty arrays")
        return _empty_spectrum()
    if not np.all(np.isfinite(fourier_freq)) or not np.all(np.isfinite(psd_values)):
        logger.warning("SpecKit returned non-finite spectrum values, returning empty arrays")
        return _empty_spectrum()

    return fourier_freq, np.sqrt(np.maximum(psd_values, 0.0))


def derive_sign_pairs(
    freq1: npt.ArrayLike,
    freq2: npt.ArrayLike,
    freq3: npt.ArrayLike,
    threshold: float = 1e5,
) -> npt.NDArray[np.int_]:
    """Derive proper signs for three signals.

    Args:
        freq1: Frequency data 1 (Hz). Must have same length as freq2 and freq3.
        freq2: Frequency data 2 (Hz). Must have same length as freq1 and freq3.
        freq3: Frequency data 3 (Hz). Must have same length as freq1 and freq2.
        threshold: Residual frequency threshold (Hz).

    Returns:
        Sign array of shape (3,) with values -1 or +1.
    """
    freq_arrays = (
        _as_1d_float_array(freq1, "freq1"),
        _as_1d_float_array(freq2, "freq2"),
        _as_1d_float_array(freq3, "freq3"),
    )
    lengths = tuple(freq.size for freq in freq_arrays)
    if len(set(lengths)) != 1:
        raise ValueError(
            f"All frequency arrays must have same length: "
            f"freq1={lengths[0]}, freq2={lengths[1]}, freq3={lengths[2]}"
        )
    if lengths[0] == 0:
        raise ValueError("frequency arrays must not be empty")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"threshold must be a positive finite number, got {threshold}")

    candidate_signs = np.array(
        [
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ],
        dtype=np.int_,
    )
    freqs = np.column_stack(freq_arrays)
    residuals = np.min(np.abs(freqs @ candidate_signs.T), axis=0)
    matching = np.flatnonzero(residuals < threshold)

    output = np.ones(3, dtype=np.int_)
    if matching.size == 0:
        logger.warning("Signs for three-signal combination are all +1")
    else:
        output[matching[np.argmin(residuals[matching])]] = -1
    return output


def combination_3sig(
    freqs: npt.ArrayLike,
    weights: npt.ArrayLike,
    signs: Optional[npt.ArrayLike] = None,
) -> npt.NDArray[np.float64]:
    """Governing equation of the three-signal combination.

    Args:
        freqs: Frequency array with shape (n_samples, 3) (Hz).
        weights: Weight array with shape (n_samples, 3) (dimensionless).
        signs: Sign array with shape (3,), values -1 or +1 (dimensionless).
              Defaults to [1, 1, -1].

    Returns:
        The three-signal combination as an array of length n_samples (Hz).
    """
    freq_array = _as_2d_float_array(freqs, "freqs", expected_columns=3)
    weight_array = _as_2d_float_array(weights, "weights", expected_columns=3)
    sign_array = _as_sign_array(signs, _DEFAULT_3SIG_SIGNS)
    if weight_array.shape != freq_array.shape:
        raise ValueError(
            f"weights must have same shape as freqs ({freq_array.shape}), got {weight_array.shape}"
        )

    return np.sum(sign_array * weight_array * freq_array, axis=1)


def combination_2sig(
    freqs: npt.ArrayLike,
    weights: npt.ArrayLike,
    signs: Optional[npt.ArrayLike] = None,
) -> npt.NDArray[np.float64]:
    """Governing equation of the two-signal combination.

    Args:
        freqs: Frequency array with shape (n_samples, 2) (Hz).
        weights: Weight array with shape (n_samples, 2) (dimensionless).
        signs: Sign array with shape (2,), values -1 or +1 (dimensionless).
              Defaults to [1, -1].

    Returns:
        The two-signal combination as an array of length n_samples (Hz).
    """
    freq_array = _as_2d_float_array(freqs, "freqs", expected_columns=2)
    weight_array = _as_2d_float_array(weights, "weights", expected_columns=2)
    sign_array = _as_sign_array(signs, _DEFAULT_2SIG_SIGNS)
    if weight_array.shape != freq_array.shape:
        raise ValueError(
            f"weights must have same shape as freqs ({freq_array.shape}), got {weight_array.shape}"
        )

    return np.sum(sign_array * weight_array * freq_array, axis=1)


def build_kaiser_lpf_taps(
    fs: float,
    f_pass: float = 0.1,
    f_stop: float = 1.0,
    attenuation: float = 1000,
) -> npt.NDArray[np.float64]:
    """Build Kaiser-window low-pass FIR taps.

    Args:
        fs: Sampling frequency (Hz).
        f_pass: Beginning of transition band (Hz).
        f_stop: End of transition band (Hz).
        attenuation: Stop-band attenuation (dB).
    """
    sampling_rate = _validate_positive_float(fs, "fs")
    pass_frequency = float(f_pass)
    stop_frequency = float(f_stop)
    attenuation_db = _validate_positive_float(attenuation, "attenuation")
    nyquist = sampling_rate / 2.0
    if not 0 <= pass_frequency < stop_frequency < nyquist:
        raise ValueError(
            "filter frequencies must satisfy 0 <= f_pass < f_stop < fs / 2; "
            f"got f_pass={f_pass}, f_stop={f_stop}, fs={fs}"
        )

    transition_width = (stop_frequency - pass_frequency) / nyquist
    numtaps, beta = signal.kaiserord(attenuation_db, transition_width)
    taps = signal.firwin(
        numtaps,
        (pass_frequency + stop_frequency) / 2.0,
        fs=sampling_rate,
        window=("kaiser", beta),
    )
    return taps


def integral_rms(
    fourier_freq: npt.ArrayLike,
    asd: npt.ArrayLike,
    pass_band: Sequence[float] = (-np.inf, np.inf),
) -> float:
    """Compute RMS by integrating ASD squared over a pass band.

    Args:
        fourier_freq: Fourier frequency (Hz).
        asd: Amplitude spectral density from which RMS is computed.
        pass_band: Two-element sequence ``(min_frequency, max_frequency)``.
    """
    freq_array = _as_1d_float_array(fourier_freq, "fourier_freq")
    asd_array = _as_1d_float_array(asd, "asd")
    if freq_array.size != asd_array.size:
        raise ValueError(
            f"fourier_freq and asd must have same length "
            f"({freq_array.size} vs {asd_array.size})"
        )
    if freq_array.size == 0:
        return 0.0
    if len(pass_band) != 2:
        raise ValueError(f"pass_band must have two elements, got {pass_band}")
    pass_min = float(pass_band[0])
    pass_max = float(pass_band[1])
    if pass_min > pass_max:
        raise ValueError(f"pass_band min must be <= max, got {pass_band}")

    integral_range_min = max(float(np.min(freq_array)), pass_min)
    integral_range_max = min(float(np.max(freq_array)), pass_max)
    if integral_range_min > integral_range_max:
        return 0.0

    f_tmp, asd_tmp = crop_data(freq_array, asd_array, integral_range_min, integral_range_max)
    if f_tmp.size < 2:
        return 0.0
    integral_rms2 = integrate.cumulative_trapezoid(asd_tmp**2, f_tmp, initial=0)
    return float(np.sqrt(integral_rms2[-1]))


def crop_data(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    xmin: float,
    xmax: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Crop paired x/y data to an inclusive x range.

    Args:
        x: Data in x.
        y: Data in y.
        xmin: Lower bound of x.
        xmax: Upper bound of x.
    """
    x_arr = _as_1d_float_array(x, "x")
    y_arr = _as_1d_float_array(y, "y")
    if len(x_arr) != len(y_arr):
        raise ValueError(f"x and y must have same length ({len(x_arr)} vs {len(y_arr)})")
    if xmin > xmax:
        raise ValueError(f"xmin must be <= xmax, got xmin={xmin}, xmax={xmax}")

    mask = (x_arr >= xmin) & (x_arr <= xmax)
    return x_arr[mask], y_arr[mask]


def convert_frequency_to_phase_in_time(
    data: npt.ArrayLike,
    fs: float
) -> npt.NDArray[np.float64]:
    """Convert frequency to phase by integration.
    
    Args:
        data: Frequency data to be converted to phase (Hz).
        fs: Data rate (Hz). Must be > 0.
    
    Returns:
        Phase array (rad).
    """
    sampling_rate = _validate_positive_float(fs, "Sampling rate fs")
    freq = _as_1d_float_array(data, "data")

    dt = 1.0 / sampling_rate
    factor = 2.0 * np.pi * dt
    return factor * np.cumsum(freq)


def convert_phase_to_frequency_in_time(
    data: npt.ArrayLike,
    fs: float,
    prepend: float = np.nan
) -> npt.NDArray[np.float64]:
    """Convert phase to frequency by discrete differentiation.

    Args:
        data: Phase data to be converted to frequency (rad).
        fs: Data rate (Hz). Must be > 0.
        prepend: Value prepended before differencing. Defaults to NaN so the
                 first sample reflects the missing previous phase sample.

    Returns:
        Frequency array (Hz).
    """
    sampling_rate = _validate_positive_float(fs, "Sampling rate fs")
    phase = _as_1d_float_array(data, "data")

    return np.diff(phase, prepend=prepend) * sampling_rate / (2.0 * np.pi)


def convert_frequency_to_detrended_phase_in_time(
    data: npt.ArrayLike,
    fs: float
) -> npt.NDArray[np.float64]:
    """Convert frequency to phase and remove a linear phase trend.

    The mean frequency is removed before integration so the resulting phase is
    dominated by residual fluctuations. A final linear detrend removes numerical
    or low-order drift before plotting or comparing residual phase time series.
    """
    freq = _as_1d_float_array(data, "data")
    if len(freq) == 0:
        return _empty_float_array()

    phase = convert_frequency_to_phase_in_time(freq - np.mean(freq), fs)
    if len(phase) < 2:
        return phase - np.mean(phase)
    return signal.detrend(phase, type="linear")


def convert_frequency_to_phase_in_asd(
    fourier_freq: npt.ArrayLike,
    data: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Convert frequency ASD to phase ASD with Fourier frequency.
    
    Args:
        fourier_freq: Fourier frequency array (Hz). Must have same length as data.
        data: Frequency spectral density to be converted to phase spectral density (Hz/√Hz).
    
    Returns:
        Phase spectral density array (rad/√Hz).
    """
    freq_array = _as_1d_float_array(fourier_freq, "fourier_freq")
    data_array = _as_1d_float_array(data, "data")
    if freq_array.size != data_array.size:
        raise ValueError(
            f"fourier_freq and data must have same length "
            f"({freq_array.size} vs {data_array.size})"
        )
    if np.any(freq_array <= 0):
        raise ValueError("fourier_freq must be > 0 for all elements")
    
    return data_array / freq_array


def get_asd_delay_factor(
    fourier_freq: npt.ArrayLike,
    delay: float,
) -> npt.NDArray[np.float64]:
    """Get delay factor for ASD of split signals.

    Args:
        fourier_freq: Fourier frequency (Hz).
        delay: Time delay (sec).
    """
    freq_array = _as_1d_float_array(fourier_freq, "fourier_freq")
    delay_seconds = float(delay)
    if not np.isfinite(delay_seconds):
        raise ValueError(f"delay must be finite, got {delay}")
    if np.any(freq_array < 0):
        raise ValueError("fourier_freq must be >= 0 for all elements")

    z = -2.0j * np.pi * freq_array * delay_seconds
    return np.abs(np.expm1(z))


def model_timer_deviation_error(
    p_fit: npt.ArrayLike,
    tau: npt.ArrayLike,
    iterations: int = 0,
) -> npt.NDArray[np.float64]:
    """Model approximation error for timer deviation.

    Args:
        p_fit: Polynomial coefficients of the clock fractional frequency.
        tau: Time array (sec).
        iterations: Number of correction iterations.

    Notes:
        The notation is based on TPS <-> THE
    """
    coefficients = _as_1d_float_array(p_fit, "p_fit")
    if coefficients.size == 0:
        raise ValueError("p_fit must not be empty")
    tau_array = _as_1d_float_array(tau, "tau")
    if tau_array.size == 0:
        return _empty_float_array()
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")

    # : prepare a broad time array to avoid numerical error
    tarray = np.logspace(np.log10(3600), np.log10(3600e3), 1000)

    # : Compute THE in TPS
    the_in_tps = func_the_in_tps(coefficients, tarray)
    
    # : Compute the inverse function, i.e. TPS in THE
    tps_in_the = _invert_func_the_in_tps(coefficients, the_in_tps)
    
    # : Compute the exact delta tau (wo initial offset)
    exact_del_tau = np.polyval(coefficients, tps_in_the)

    # : Compute the approximated delta tau and its deviation from the exact
    approximate_del_tau = np.polyval(coefficients, the_in_tps)
    diff_apprx_exact_tau = approximate_del_tau - exact_del_tau
    for _ in range(iterations):
        approximate_del_tau = np.polyval(coefficients, the_in_tps - approximate_del_tau)
        diff_apprx_exact_tau = approximate_del_tau - exact_del_tau

    # : fit the model over a broad range and derive the model over the measurement time
    fit = np.polyfit(tarray, diff_apprx_exact_tau, coefficients.size - 1)
    diff_apprx_exact_tau_in_range = np.polyval(fit, tau_array)
    # Normalize to zero at the first measurement sample so the model represents
    # accumulated deviation over the provided measurement span.
    diff_apprx_exact_tau_in_range -= diff_apprx_exact_tau_in_range[0]

    return diff_apprx_exact_tau_in_range


def _invert_func_the_in_tps(
    p_fit: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    max_iterations: int = 50,
) -> npt.NDArray[np.float64]:
    """Invert ``func_the_in_tps`` for an array of THE values."""
    derivative_coefficients = np.polyder(p_fit)
    inverse = values - np.polyval(p_fit, values)
    absolute_tolerance = 1e-12
    relative_tolerance = 4.0 * np.finfo(np.float64).eps

    for _ in range(max_iterations):
        residual = func_the_in_tps(p_fit, inverse) - values
        derivative = 1.0 + np.polyval(derivative_coefficients, inverse)
        if (
            not np.all(np.isfinite(residual))
            or not np.all(np.isfinite(derivative))
            or np.any(np.isclose(derivative, 0.0, atol=absolute_tolerance))
        ):
            break

        step = residual / derivative
        next_inverse = inverse - step
        tolerance = absolute_tolerance + relative_tolerance * np.maximum(
            1.0,
            np.abs(next_inverse),
        )
        inverse = next_inverse
        if np.all(np.abs(step) <= tolerance):
            return inverse

    logger.debug("Newton inversion did not converge; falling back to scalar optimizer")
    fallback = np.empty_like(values)
    for idx, value in enumerate(values):
        result = optimize.minimize(diff, inverse[idx], args=(value, p_fit), method="Nelder-Mead")
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Failed to invert THE/TPS mapping at index {idx}: {result.message}")
        fallback[idx] = result.x[0]
    return fallback


def func_the_in_tps(
    p_fit: npt.ArrayLike,
    x: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Generate THE values from TPS values.

    Args:
        p_fit: Polynomial coefficients.
        x: Time values.
    """
    coefficients = np.asarray(p_fit, dtype=np.float64)
    values = np.asarray(x, dtype=np.float64)
    return np.polyval(coefficients, values) + values


def diff(
    inverse_x: npt.ArrayLike,
    original_the_in_tps: float,
    p_fit: npt.ArrayLike,
) -> float:
    """Squared error for inverting THE/TPS mapping."""
    new_the_in_tps = func_the_in_tps(p_fit, inverse_x)
    return float(np.sum(np.square(new_the_in_tps - original_the_in_tps)))


def components_for_balancing(
    bsigs: Sequence[Optional[Any]],
    size: int,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Prepare complementary frequencies and weights for balancing.

    Args:
        bsigs: List of the three one-signal classes for balanced detection.
        size: Size of signal array.
    """
    if not isinstance(size, (int, np.integer)) or size < 0:
        raise ValueError(f"size must be a non-negative integer, got {size}")
    size_int = int(size)
    balancing_signals = list(bsigs)
    if len(balancing_signals) != 3:
        raise ValueError(f"bsigs must contain exactly 3 elements, got {len(balancing_signals)}")

    bfreqs = []
    weights = np.ones((size_int, 3), dtype=np.float64)
    for index, bs in enumerate(balancing_signals):
        if bs is None:
            bf = np.zeros(size_int, dtype=np.float64)
        else:
            bf = _as_1d_float_array(getattr(bs, "total", None), f"bsigs[{index}].total")
            if bf.size != size_int:
                raise ValueError(
                    f"bsigs[{index}].total must have length {size_int}, got {bf.size}"
                )
            weights[:, index] *= 0.5
        bfreqs.append(bf)
    bfreqs_array = np.column_stack(bfreqs)

    return bfreqs_array, weights
