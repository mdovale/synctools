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
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from scipy import signal
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from synctools.frequency import FrequencyData

from synctools.auxiliary import (
    combination_2sig,
    combination_3sig,
    convert_frequency_to_phase_in_time,
    convert_frequency_to_phase_in_asd,
    derive_sign_pairs,
    components_for_balancing,
    spectra,
)

class TwoSignalsElement:
    def __init__(
        self,
        freqs: npt.NDArray[np.float64],
        fs: float,
        scaling: float = 1.0,
        weights: Optional[npt.NDArray[np.float64]] = None
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
        if fs <= 0:
            raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
        freqs = np.asarray(freqs, dtype=np.float64)
        if freqs.ndim != 2 or freqs.shape[1] != 2:
            raise ValueError(
                f"freqs must have shape (n_samples, 2), got {freqs.shape}"
            )
        
        self.fs = fs
        self.scaling = scaling
        self.weights = np.ones((freqs.shape[0],freqs.shape[1])) if weights is None else weights
        if self.weights.shape != freqs.shape:
            raise ValueError(
                f"weights must have same shape as freqs ({freqs.shape}), "
                f"got {self.weights.shape}"
            )
        self.freq = self.scaling*combination_2sig(freqs, self.weights)
        self.phase = convert_frequency_to_phase_in_time(self.freq, fs)

        # : linear detrending to remove a residual ramp (basically just for plots)
        self.phase = signal.detrend(self.phase, type='linear')

    def compute_all_spectrum(self, p_lpsd: Dict[str, Any]) -> None:
        """Compute frequency and phase ASDs.
        
        Args:
            p_lpsd: SpecKit parameters dictionary.
        """
        self.fourier_freq, self.freq_asd = spectra(self.freq, self.fs, p_lpsd)
        self.phase_asd = convert_frequency_to_phase_in_asd(self.fourier_freq, self.freq_asd)
        
class TwoSignals:
    def __init__(
        self,
        sigs: List['FrequencyData'],
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
        if len(sigs) != 2:
            raise ValueError(
                f"sigs must have exactly 2 elements for TwoSignals, got {len(sigs)}"
            )
        
        _sigs = copy.deepcopy(sigs)

        self.name = f"{_sigs[0].name} vs. {_sigs[1].name}"
        self.tau = _sigs[0].tau
        self.fs = _sigs[0].fs

        # : main
        freqs = np.array([s.total for s in _sigs]).T
        self.main = TwoSignalsElement(freqs, self.fs)

        self.main.compute_all_spectrum(p_lpsd)


class ThreeSignalsElement:
    def __init__(
        self,
        freqs: npt.NDArray[np.float64],
        fs: float,
        signs: npt.NDArray[np.int_],
        scaling: float = 1.0,
        weights: Optional[npt.NDArray[np.float64]] = None
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
        if fs <= 0:
            raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
        freqs = np.asarray(freqs, dtype=np.float64)
        if freqs.ndim != 2 or freqs.shape[1] != 3:
            raise ValueError(
                f"freqs must have shape (n_samples, 3), got {freqs.shape}"
            )
        signs = np.asarray(signs, dtype=np.int_)
        if signs.shape != (3,):
            raise ValueError(f"signs must have shape (3,), got {signs.shape}")
        if not np.all(np.isin(signs, [-1, 1])):
            raise ValueError(f"signs must contain only -1 or +1, got {signs}")
        
        self.fs = fs
        self.signs = signs
        self.weights = np.ones((freqs.shape[0],freqs.shape[1])) if weights is None else weights
        if self.weights.shape != freqs.shape:
            raise ValueError(
                f"weights must have same shape as freqs ({freqs.shape}), "
                f"got {self.weights.shape}"
            )
        self.freq = scaling*combination_3sig(freqs, self.weights, self.signs)
        self.phase = convert_frequency_to_phase_in_time(self.freq, self.fs)

        # : linear detrending to remove a residual ramp (basically just for plots)
        self.phase = signal.detrend(self.phase, type='linear')

    def compute_all_spectrum(self, p_lpsd: Dict[str, Any]) -> None:
        """Compute frequency and phase ASDs.
        
        Args:
            p_lpsd: SpecKit parameters dictionary.
        """
        self.fourier_freq, self.freq_asd = spectra(self.freq, self.fs, p_lpsd)
        self.phase_asd = convert_frequency_to_phase_in_asd(self.fourier_freq, self.freq_asd)
        
class ThreeSignals:
    def __init__(
        self,
        sigs: List['FrequencyData'],
        p_lpsd: Dict[str, Any],
        bsigs: Optional[List[Optional['FrequencyData']]] = None
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
        if len(sigs) != 3:
            raise ValueError(
                f"sigs must have exactly 3 elements for ThreeSignals, got {len(sigs)}"
            )
        if bsigs is None:
            bsigs = [None, None, None]
        if len(bsigs) != 3:
            raise ValueError(
                f"bsigs must have exactly 3 elements, got {len(bsigs)}"
            )
        
        _sigs = copy.deepcopy(sigs)
        _bsigs = copy.deepcopy(bsigs)
        self.name = "("
        for _s, _b in zip(_sigs, _bsigs):
            _name = f"{_s.name}"
            if _b is not None:
                _name += f"+{_s.name}"
            _name += ","
            self.name += _name
        self.name += ')'

        self.signs = derive_sign_pairs(
            _sigs[0].total,
            _sigs[1].total,
            _sigs[2].total
            )
        self.tau = _sigs[0].tau
        self.fs = _sigs[0].fs

        # : main
        freqs = np.array([s.total for s in _sigs]).T
        bfreqs, weights = components_for_balancing(_bsigs, freqs[:,0].size)
        self.main = ThreeSignalsElement(freqs+bfreqs, self.fs,  self.signs, weights=weights)

        self.main.compute_all_spectrum(p_lpsd)