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
"""Signal-combination containers for two- and three-channel measurements."""

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from synctools.auxiliary import (
    combination_2sig,
    combination_3sig,
    components_for_balancing,
    convert_frequency_to_detrended_phase_in_time,
    convert_frequency_to_phase_in_asd,
    derive_sign_pairs,
    spectra,
)

if TYPE_CHECKING:
    from synctools.frequency import FrequencyData


__all__ = [
    "TwoSignalsElement",
    "TwoSignals",
    "ThreeSignalsElement",
    "ThreeSignals",
]


def _validate_finite_float(value: float, name: str) -> float:
    """Return ``value`` as a finite float."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite numeric scalar, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric scalar, got {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _validate_positive_float(value: float, name: str) -> float:
    """Return ``value`` as a positive finite float."""
    result = _validate_finite_float(value, name)
    if result <= 0:
        raise ValueError(f"Sampling rate {name} must be > 0, got {value}")
    return result


def _as_1d_float_array(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return finite 1D data as ``float64``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D array, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_2d_float_array(
    values: npt.ArrayLike,
    name: str,
    expected_columns: int,
) -> npt.NDArray[np.float64]:
    """Return finite 2D data with the expected column count as ``float64``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != expected_columns:
        raise ValueError(
            f"{name} must have shape (n_samples, {expected_columns}), got {array.shape}"
        )
    if array.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _weights_or_ones(
    weights: Optional[npt.ArrayLike],
    freqs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return validated per-sample weights matching ``freqs``."""
    if weights is None:
        return np.ones_like(freqs, dtype=np.float64)

    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != freqs.shape:
        raise ValueError(
            f"weights must have same shape as freqs ({freqs.shape}), "
            f"got {weight_array.shape}"
        )
    if not np.all(np.isfinite(weight_array)):
        raise ValueError("weights must contain only finite values")
    return weight_array.copy()


def _as_sign_array(signs: npt.ArrayLike) -> npt.NDArray[np.int_]:
    """Return a validated sign vector with values -1 or +1."""
    sign_array = np.asarray(signs)
    if sign_array.shape != (3,):
        raise ValueError(f"signs must have shape (3,), got {sign_array.shape}")
    if np.issubdtype(sign_array.dtype, np.bool_):
        raise ValueError("signs must contain only integer -1 or +1 values")
    if not np.all(np.isin(sign_array, [-1, 1])):
        raise ValueError(f"signs must contain only -1 or +1, got {sign_array}")
    return sign_array.astype(np.int_, copy=True)


def _signal_name(sig: "FrequencyData") -> str:
    """Return a stable display name for a signal-like object."""
    return str(getattr(sig, "name", "signal"))


def _as_signal_tuple(
    sigs: Sequence["FrequencyData"],
    expected_count: int,
    name: str,
) -> Tuple["FrequencyData", ...]:
    """Validate a signal sequence has the expected number of entries."""
    try:
        sig_tuple = tuple(sigs)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of frequency signals") from exc
    if len(sig_tuple) != expected_count:
        raise ValueError(
            f"{name} must have exactly {expected_count} elements, got {len(sig_tuple)}"
        )
    return sig_tuple


def _validate_frequency_signals(
    sigs: Sequence["FrequencyData"],
    expected_count: int,
    name: str,
) -> Tuple[
    Tuple["FrequencyData", ...],
    Tuple[npt.NDArray[np.float64], ...],
    float,
    npt.NDArray[np.float64],
]:
    """Validate signal-like objects and return their total frequencies."""
    sig_tuple = _as_signal_tuple(sigs, expected_count, name)
    totals = []
    ref_total_shape = None
    ref_tau = None
    ref_fs = None

    for index, sig in enumerate(sig_tuple):
        prefix = f"{name}[{index}]"
        total = _as_1d_float_array(getattr(sig, "total", None), f"{prefix}.total")
        tau = _as_1d_float_array(getattr(sig, "tau", None), f"{prefix}.tau")
        fs = _validate_positive_float(getattr(sig, "fs", None), f"{prefix}.fs")

        if tau.shape != total.shape:
            raise ValueError(
                f"{prefix}.tau must have same length as {prefix}.total "
                f"({tau.size} vs {total.size})"
            )

        if ref_total_shape is None:
            ref_total_shape = total.shape
            ref_tau = tau
            ref_fs = fs
        else:
            if total.shape != ref_total_shape:
                raise ValueError(
                    f"All signals must have same length: expected {ref_total_shape[0]}, "
                    f"got {total.size} for {prefix}.total"
                )
            if not np.isclose(fs, ref_fs):
                raise ValueError(
                    f"All signals must have same sampling rate: "
                    f"expected {ref_fs}, got {fs} for {prefix}.fs"
                )
            if not np.allclose(tau, ref_tau):
                raise ValueError(f"{prefix}.tau does not match {name}[0].tau")

        totals.append(total)

    if ref_fs is None or ref_tau is None:
        raise ValueError(f"{name} cannot be empty")
    return sig_tuple, tuple(totals), ref_fs, ref_tau.copy()


def _validate_balancing_signals(
    bsigs: Optional[Sequence[Optional["FrequencyData"]]],
    expected_count: int,
    expected_length: int,
    expected_fs: float,
) -> Tuple[Optional["FrequencyData"], ...]:
    """Validate optional balanced-detection companion signals."""
    if bsigs is None:
        return (None,) * expected_count

    bsig_tuple = _as_signal_tuple(bsigs, expected_count, "bsigs")
    for index, sig in enumerate(bsig_tuple):
        if sig is None:
            continue

        prefix = f"bsigs[{index}]"
        total = _as_1d_float_array(getattr(sig, "total", None), f"{prefix}.total")
        fs = _validate_positive_float(getattr(sig, "fs", None), f"{prefix}.fs")
        if total.size != expected_length:
            raise ValueError(
                f"{prefix}.total must have length {expected_length}, got {total.size}"
            )
        if not np.isclose(fs, expected_fs):
            raise ValueError(
                f"{prefix}.fs must match primary sampling rate {expected_fs}, got {fs}"
            )

    return bsig_tuple


class TwoSignalsElement:
    def __init__(
        self,
        freqs: npt.ArrayLike,
        fs: float,
        scaling: float = 1.0,
        weights: Optional[npt.ArrayLike] = None
    ) -> None:
        """Class for a two-signal measurement.

        Args:
            freqs: Frequency array with shape (n_samples, 2) (Hz).
            fs: Data rate (Hz). Must be > 0.
            scaling: Overall scaling factor (dimensionless).
            weights: Optional weight array for each term. If None, uses equal weights.
                   Must have shape (n_samples, 2) if provided.
        
        Raises:
            ValueError: If validation fails (invalid shape, fs <= 0, etc.)
        """
        freqs = _as_2d_float_array(freqs, "freqs", expected_columns=2)

        self.fs = _validate_positive_float(fs, "fs")
        self.scaling = _validate_finite_float(scaling, "scaling")
        self.weights = _weights_or_ones(weights, freqs)
        self.fourier_freq: Optional[npt.NDArray[np.float64]] = None
        self.freq_asd: Optional[npt.NDArray[np.float64]] = None
        self.phase_asd: Optional[npt.NDArray[np.float64]] = None

        self.freq = self.scaling * combination_2sig(freqs, self.weights)
        self.phase = convert_frequency_to_detrended_phase_in_time(self.freq, self.fs)

    def compute_all_spectrum(self, p_lpsd: Dict[str, Any]) -> None:
        """Compute frequency and phase ASDs.
        
        Args:
            p_lpsd: SpecKit parameters dictionary.
        """
        self.fourier_freq, self.freq_asd = spectra(self.freq, self.fs, p_lpsd)
        self.phase_asd = convert_frequency_to_phase_in_asd(
            self.fourier_freq,
            self.freq_asd,
        )


class TwoSignals:
    def __init__(
        self,
        sigs: Sequence['FrequencyData'],
        p_lpsd: Dict[str, Any]
    ) -> None:
        """Class for two-signal combinations with all two-signal pairs.
        
        Args:
            sigs: List of the two one-signal classes (FrequencyData instances).
                 Must have exactly 2 elements.
            p_lpsd: SpecKit parameters dictionary.
        
        Raises:
            ValueError: If sigs does not have exactly 2 elements.
        """
        sigs, totals, self.fs, self.tau = _validate_frequency_signals(
            sigs,
            expected_count=2,
            name="sigs",
        )

        self.name = f"{_signal_name(sigs[0])} vs. {_signal_name(sigs[1])}"
        freqs = np.column_stack(totals)
        self.main = TwoSignalsElement(freqs, self.fs)
        self.main.compute_all_spectrum(p_lpsd)


class ThreeSignalsElement:
    def __init__(
        self,
        freqs: npt.ArrayLike,
        fs: float,
        signs: npt.ArrayLike,
        scaling: float = 1.0,
        weights: Optional[npt.ArrayLike] = None
    ) -> None:
        """Class for a three-signal measurement.
        
        Args:
            freqs: Frequency array with shape (n_samples, 3) (Hz).
            fs: Data rate (Hz). Must be > 0.
            signs: Sign array for each term, shape (3,). Values should be -1 or +1.
            scaling: Overall scaling factor (dimensionless).
            weights: Optional weight array for each term. If None, uses equal weights.
                   Must have shape (n_samples, 3) if provided.
        
        Raises:
            ValueError: If validation fails (invalid shape, fs <= 0, etc.)
        """
        freqs = _as_2d_float_array(freqs, "freqs", expected_columns=3)

        self.fs = _validate_positive_float(fs, "fs")
        self.scaling = _validate_finite_float(scaling, "scaling")
        self.signs = _as_sign_array(signs)
        self.weights = _weights_or_ones(weights, freqs)
        self.fourier_freq: Optional[npt.NDArray[np.float64]] = None
        self.freq_asd: Optional[npt.NDArray[np.float64]] = None
        self.phase_asd: Optional[npt.NDArray[np.float64]] = None

        self.freq = self.scaling * combination_3sig(freqs, self.weights, self.signs)
        self.phase = convert_frequency_to_detrended_phase_in_time(self.freq, self.fs)

    def compute_all_spectrum(self, p_lpsd: Dict[str, Any]) -> None:
        """Compute frequency and phase ASDs.
        
        Args:
            p_lpsd: SpecKit parameters dictionary.
        """
        self.fourier_freq, self.freq_asd = spectra(self.freq, self.fs, p_lpsd)
        self.phase_asd = convert_frequency_to_phase_in_asd(
            self.fourier_freq,
            self.freq_asd,
        )


class ThreeSignals:
    def __init__(
        self,
        sigs: Sequence['FrequencyData'],
        p_lpsd: Dict[str, Any],
        bsigs: Optional[Sequence[Optional['FrequencyData']]] = None
    ) -> None:
        """Class for three-signal combinations with all three-signal pairs.
        
        Args:
            sigs: List of the three one-signal classes (FrequencyData instances).
                 Must have exactly 3 elements.
            p_lpsd: SpecKit parameters dictionary.
            bsigs: Optional list of the three one-signal classes for balanced detection.
                  Must have the same order as sigs, heterodyne-frequency wise.
                  If None, defaults to [None, None, None]. Must have exactly 3 elements.
        
        Raises:
            ValueError: If sigs or bsigs do not have exactly 3 elements.
        """
        sigs, totals, self.fs, self.tau = _validate_frequency_signals(
            sigs,
            expected_count=3,
            name="sigs",
        )
        bsigs = _validate_balancing_signals(
            bsigs,
            expected_count=3,
            expected_length=totals[0].size,
            expected_fs=self.fs,
        )

        name_parts = []
        for sig, bsig in zip(sigs, bsigs):
            signal_name = _signal_name(sig)
            if bsig is not None:
                signal_name = f"{signal_name}+{_signal_name(bsig)}"
            name_parts.append(signal_name)
        self.name = f"({','.join(name_parts)})"

        self.signs = derive_sign_pairs(
            totals[0],
            totals[1],
            totals[2],
        )

        freqs = np.column_stack(totals)
        bfreqs, weights = components_for_balancing(bsigs, freqs[:, 0].size)
        self.main = ThreeSignalsElement(
            freqs + bfreqs,
            self.fs,
            self.signs,
            weights=weights,
        )

        self.main.compute_all_spectrum(p_lpsd)