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
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import numpy.typing as npt
from pytdi.dsp import calculate_advancements

from synctools.auxiliary import model_timer_deviation_error

if TYPE_CHECKING:
    from synctools.frequency import FrequencyData


def _as_1d_float_array(
    values: npt.ArrayLike,
    name: str,
    *,
    copy_array: bool = False,
) -> npt.NDArray[np.float64]:
    """Return finite 1D data as ``float64``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D array, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if copy_array:
        array = array.copy()
    return array


def _validate_finite_float(value: float, name: str) -> float:
    """Coerce a numeric scalar to finite ``float``."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric scalar, got {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _validate_sampling_rate(fs: float) -> float:
    result = _validate_finite_float(fs, "fs")
    if result <= 0:
        raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
    return result


def _validate_positive_float(value: float, name: str) -> float:
    result = _validate_finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return result


def _validate_non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return result


def _validate_positive_int(value: int, name: str) -> int:
    result = _validate_non_negative_int(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return result


def _validate_sign(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"sign must be -1 or +1, got {value!r}")
    sign = int(value)
    if sign not in (-1, 1):
        raise ValueError(f"sign must be -1 or +1, got {value}")
    return sign


class TimerData:
    def __init__(
        self,
        rd: "FrequencyData",
        fs: float,
        sign: int = -1,
        offset: float = 0.0,
        skip_error_computation: bool = False,
    ) -> None:
        """Class for timer data (integrated clock signal).

        This class converts fractional frequency data to timer data through
        integration, providing both "raw" (direct integration) and "inv"
        (inverse/advancement) forms.

        Args:
            rd: FrequencyData instance containing fractional frequency.
                ``rd.total``, ``rd.fit``, ``rd.fluc``, and ``rd.tau`` must be
                finite 1D arrays with matching lengths.
            fs: Data rate (sampling frequency), in Hz. Must be finite and > 0.
            sign: Sign multiplier for frequency-to-time conversion. Must be
                exactly -1 or +1.
            offset: Initial timer offset, in seconds.
            skip_error_computation: If True, skip computation of timer deviation
                error for faster initialization.

        Attributes:
            tau: Time array copied from ``rd.tau``.
            total: Raw and inverse timer data, keyed by ``"raw"`` and ``"inv"``.
            fit: Fitted deterministic timer components.
            fluc: Stochastic timer components.
            p_fit_model: Frequency-fit coefficients padded with a zero constant
                term for timer deviation error modeling.
            timer_dev_err: Timer deviation error dictionary, set only when
                ``skip_error_computation`` is False or when
                ``compute_timer_deviation_error()`` is called explicitly.

        Raises:
            ValueError: If inputs are not finite, 1D, length-compatible, or if
                ``fs``/``sign``/``offset`` are invalid.
        """
        fs = _validate_sampling_rate(fs)
        sign = _validate_sign(sign)
        offset = _validate_finite_float(offset, "offset")

        tau = _as_1d_float_array(rd.tau, "rd.tau", copy_array=True)
        total_freq = _as_1d_float_array(rd.total, "rd.total")
        fit_freq = _as_1d_float_array(rd.fit, "rd.fit")
        fluc_freq = _as_1d_float_array(rd.fluc, "rd.fluc")
        order = _validate_non_negative_int(rd.order, "rd.order")
        p_fit = _as_1d_float_array(rd.p_fit, "rd.p_fit")

        data_len = tau.size
        for name, values in (
            ("rd.total", total_freq),
            ("rd.fit", fit_freq),
            ("rd.fluc", fluc_freq),
        ):
            if values.size != data_len:
                raise ValueError(
                    f"{name} must have same length as rd.tau ({values.size} vs {data_len})"
                )
        if p_fit.size != order + 1:
            raise ValueError(
                f"rd.p_fit length must equal rd.order + 1 ({p_fit.size} vs {order + 1})"
            )

        self.fs = fs
        self.tau = tau

        total = self.convert_frac_frequency_to_time(sign * total_freq, fs) + offset
        fit = self.convert_frac_frequency_to_time(sign * fit_freq, fs) + offset
        fluc = self.convert_frac_frequency_to_time(sign * fluc_freq, fs)

        # Timer is the integral of fractional frequency, so the fitted inverse
        # timer polynomial is one order higher than the source frequency fit.
        timer_fit_order = order + 1
        total_inv = calculate_advancements(total, fs)
        fit_inv, self.p_fit_inv = self.fit_timer(self.tau, total_inv, timer_fit_order)
        fluc_inv = total_inv - fit_inv

        self.total: Dict[str, npt.NDArray[np.float64]] = {"raw": total, "inv": total_inv}
        self.fit: Dict[str, npt.NDArray[np.float64]] = {"raw": fit, "inv": fit_inv}
        self.fluc: Dict[str, npt.NDArray[np.float64]] = {"raw": fluc, "inv": fluc_inv}

        self.p_fit_model = np.append(p_fit, 0.0)
        if not skip_error_computation:
            self.compute_timer_deviation_error(fs)

    def compute_timer_deviation_error(self, fs: float) -> None:
        """Compute the modeled and estimated non-inverted timer deviation error.

        Args:
            fs: Data rate (Hz). Must be finite and > 0.
        """
        fs = _validate_sampling_rate(fs)

        total_raw = _as_1d_float_array(self.total["raw"], 'self.total["raw"]')
        if total_raw.size != self.tau.size:
            raise ValueError(
                f"self.total['raw'] must have same length as tau "
                f"({total_raw.size} vs {self.tau.size})"
            )

        # Use inverse_timer_wo_large_offset instead of self.total["inv"] so large
        # initial offsets do not dominate the advancement calculation.
        total_inv = self.inverse_timer_wo_large_offset(total_raw, fs)
        timer_dev_err_estimate = total_inv - total_raw

        p_fit_model_tmp = self.p_fit_model.copy()
        degree = p_fit_model_tmp.size - 1
        for index in range(degree):
            p_fit_model_tmp[index] /= degree - index
        timer_dev_err_model = model_timer_deviation_error(p_fit_model_tmp, self.tau)

        self.timer_dev_err = {"estimate": timer_dev_err_estimate, "model": timer_dev_err_model}

    @staticmethod
    def fit_timer(
        tau: npt.ArrayLike,
        data: npt.ArrayLike,
        order: int = 0,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Fit timer data with a polynomial.

        Args:
            tau: Time array (s).
            data: Timer data array to detrend (s).
            order: Detrend order. Must be a non-negative integer less than the
                number of samples.

        Returns:
            Tuple of ``(data_fit, p_fit)`` where ``data_fit`` is the fitted timer
            data and ``p_fit`` contains NumPy polynomial coefficients.
        """
        tau = _as_1d_float_array(tau, "tau")
        data = _as_1d_float_array(data, "data")
        order = _validate_non_negative_int(order, "order")
        if tau.size != data.size:
            raise ValueError(f"tau and data must have same length ({tau.size} vs {data.size})")
        if order >= data.size:
            raise ValueError(f"order ({order}) must be less than number of samples ({data.size})")
        if order == 0:
            mean = float(np.mean(data, dtype=np.float64))
            return np.full(data.shape, mean, dtype=np.float64), np.array([mean])

        p_fit = np.polyfit(tau, data, deg=order)
        data_fit = np.polyval(p_fit, tau)
        return data_fit, p_fit

    @staticmethod
    def convert_frac_frequency_to_time(
        data: npt.ArrayLike,
        fs: float,
        zeroed: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Convert fractional frequency to time by discrete integration.

        Args:
            data: Fractional frequency data to convert to time (dimensionless).
            fs: Data rate (Hz). Must be finite and > 0.
            zeroed: If True, anchor the initial value to zero after integration.

        Returns:
            Timer data array (s).
        """
        data = _as_1d_float_array(data, "data")
        fs = _validate_sampling_rate(fs)

        timer = np.cumsum(data, dtype=np.float64) / fs
        if zeroed:
            timer = timer - timer[0]

        return timer

    @staticmethod
    def inverse_timer_wo_large_offset(
        data: npt.ArrayLike,
        fs: float,
        interp_order: int = 5,
        delta: float = 1e-12,
        maxiter: int = 100,
    ) -> npt.NDArray[np.float64]:
        """Invert timer deviation while removing large initial offsets first.

        To be compared with ``model_timer_deviation_error()``, which simulates
        only deterministic components.

        Args:
            data: Timer data to be inverted (s).
            fs: Data rate (Hz). Must be finite and > 0.
            interp_order: Interpolation order. Must be a positive integer.
            delta: Positive convergence threshold for numerical calculations.
            maxiter: Positive maximum number of numerical iterations.

        Returns:
            Inverse timer data array (s).
        """
        data = _as_1d_float_array(data, "data")
        fs = _validate_sampling_rate(fs)
        interp_order = _validate_positive_int(interp_order, "interp_order")
        delta = _validate_positive_float(delta, "delta")
        maxiter = _validate_positive_int(maxiter, "maxiter")

        initial_offset = data[0]
        inverse = calculate_advancements(
            data - initial_offset,
            fs,
            order=interp_order,
            delta=delta,
            maxiter=maxiter,
        )
        return inverse + initial_offset

    def truncation(self, n_trunc: int) -> None:
        """Truncate both ends of timer time-series in place.

        Args:
            n_trunc: Number of points to truncate at each end of each array.
                Must satisfy ``n_trunc < len(data) // 2``.
        """
        n_trunc = _validate_non_negative_int(n_trunc, "n_trunc")
        if n_trunc == 0:
            return
        data_len = self.tau.size
        if n_trunc >= data_len // 2:
            raise ValueError(
                f"n_trunc ({n_trunc}) must be < len(data) // 2 ({data_len // 2})"
            )

        slc = slice(n_trunc, -n_trunc)
        self.tau = self.tau[slc]
        for timer_dict in (self.total, self.fit, self.fluc):
            for key in ("raw", "inv"):
                timer_dict[key] = timer_dict[key][slc]
        if hasattr(self, "timer_dev_err"):
            for key in ("estimate", "model"):
                self.timer_dev_err[key] = self.timer_dev_err[key][slc]
