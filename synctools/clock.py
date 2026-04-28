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
import copy
from typing import TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt

from synctools.timer import TimerData

if TYPE_CHECKING:
    from synctools.frequency import FrequencyData


def _validate_finite_float(value: float, name: str) -> float:
    """Coerce a numeric scalar to finite ``float``."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric scalar, got {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
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


def _validate_timer_sign(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"timer_sign must be -1 or +1, got {value!r}")
    timer_sign = int(value)
    if timer_sign not in (-1, 1):
        raise ValueError(f"timer_sign must be -1 or +1, got {value}")
    return timer_sign


def _validate_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


def _validate_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"name must be a string, got {value!r}")
    return value


def _validate_frequency_data_like(rd: object) -> None:
    required_attributes = (
        "total",
        "fit",
        "fluc",
        "tau",
        "fs",
        "order",
        "p_fit",
        "time_stamping",
        "truncation",
    )
    missing = [
        attribute for attribute in required_attributes if not hasattr(rd, attribute)
    ]
    if missing:
        raise ValueError(
            "rd must be a FrequencyData-like object with attributes "
            f"{', '.join(required_attributes)}; missing {', '.join(missing)}"
        )


class Clock:
    """Class for differential clock measurements.

    This class wraps a FrequencyData object (rd) representing a differential clock
    signal and its corresponding TimerData (tshift), providing an interface for
    clock synchronization operations including time-stamping and truncation.
    """
    def __init__(
        self,
        rd: "FrequencyData",
        timer_offset: float = 0.0,
        timer_sign: int = 1,
        primary_stamped: bool = True,
        name: str = "diff_clock",
    ) -> None:
        """Initialize a Clock instance.

        Args:
            rd: FrequencyData instance representing the differential clock signal.
                The object is deep-copied because clock time-stamping mutates the
                frequency data in place.
            timer_offset: Initial timer offset, in seconds.
            timer_sign: Sign multiplier for timer conversion. Must be exactly
                -1 or +1.
            primary_stamped: Whether the clock is already stamped with the
                primary time reference.
            name: Name identifier for this clock instance.

        Attributes:
            rd: Deep copy of the input FrequencyData-like object.
            tshift: TimerData instance derived from ``rd``.
            primary_stamped: Whether the clock is primary-stamped.
            timer_sign: Sign multiplier for timer conversion.
            timer_offset: Current timer offset, in seconds.
            name: Name identifier.

        Raises:
            ValueError: If inputs are invalid.

        Example:
            >>> import numpy as np
            >>> from synctools import Clock, FrequencyData
            >>> # Create zero clock reference (no jitter)
            >>> rd = FrequencyData(np.zeros(1000), fs=10.0)
            >>> clock = Clock(rd, timer_offset=0.0)
            >>> print(f"Clock name: {clock.name}")
        """
        _validate_frequency_data_like(rd)
        self.name = _validate_name(name)
        self.rd = copy.deepcopy(rd)
        self.primary_stamped = _validate_bool(primary_stamped, "primary_stamped")
        self.timer_sign = _validate_timer_sign(timer_sign)
        self.timer_offset = _validate_finite_float(timer_offset, "timer_offset")
        self._refresh_timer(skip_error_computation=False)

    def _refresh_timer(self, *, skip_error_computation: bool) -> None:
        self.tshift = TimerData(
            self.rd,
            self.rd.fs,
            sign=self.timer_sign,
            offset=self.timer_offset,
            skip_error_computation=skip_error_computation,
        )

    def add_timer_offset(self, addition: float) -> None:
        """Add to the clock timer offset and refresh derived timer data.

        Args:
            addition: Correction added to the current timer offset (s).
        """
        addition = _validate_finite_float(addition, "addition")
        self.timer_offset += addition
        self._refresh_timer(skip_error_computation=True)

    def time_stamping(
        self,
        shifts: npt.ArrayLike,
        factor: Union[float, npt.ArrayLike] = 1.0,
        order: int = 121,
    ) -> None:
        """Time-stamp the clock frequency measurement in place.

        ``FrequencyData.time_stamping`` mutates ``rd``. The derived ``tshift``
        timer is rebuilt afterward so both representations remain consistent.

        Args:
            shifts: Number of fractional samples to shift. Must have same length
                as the underlying frequency data.
            factor: Scalar or per-sample scaling factor, e.g. Doppler factor due
                to clock bias.
            order: Interpolation order. Must be a positive integer.
        """
        order = _validate_positive_int(order, "order")
        self.rd.time_stamping(shifts, factor=factor, order=order)
        self._refresh_timer(skip_error_computation=True)

    def truncation(self, n_trunc: int) -> None:
        """Truncate both ends of frequency and timer time-series in place.

        Args:
            n_trunc: Number of points to be truncated at each end of array.
                Must satisfy ``n_trunc < len(data) // 2``.
        """
        n_trunc = _validate_non_negative_int(n_trunc, "n_trunc")
        if n_trunc == 0:
            return
        data_len = self.rd.tau.size
        if self.tshift.tau.size != data_len:
            raise ValueError(
                f"rd and tshift must have matching lengths "
                f"({data_len} vs {self.tshift.tau.size})"
            )
        if n_trunc >= data_len // 2:
            raise ValueError(
                f"n_trunc ({n_trunc}) must be < len(data) // 2 ({data_len // 2})"
            )

        self.rd.truncation(n_trunc)
        self.tshift.truncation(n_trunc)