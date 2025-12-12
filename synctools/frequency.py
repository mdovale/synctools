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
from typing import Optional, Tuple
from pytdi.dsp import timeshift
import logging
logger = logging.getLogger(__name__)

from synctools.auxiliary import spectra

class FrequencyData:
    def __init__(
        self,
        main_tot: npt.NDArray[np.float64],
        fs: float,
        name: str = 'y',
        order: int = 0
    ) -> None:
        """Class for frequency time series data with detrending capabilities.
        
        This class represents a frequency measurement time series and provides
        methods to separate deterministic (fitted) and stochastic (fluctuation)
        components through polynomial fitting.
        
        Args:
            main_tot: Total frequency data array.
                     - Shape: 1D array, (n_samples,)
                     - Units: Hz (frequency)
                     - Must not be empty
            fs: Sampling rate.
                - Units: Hz
                - Must be > 0
            name: Name identifier for this frequency data instance.
                  Used for labeling in plots and debugging.
            order: Order of polynomial fit for detrending.
                  - Non-negative integer
                  - 0: Constant (mean removal)
                  - 1: Linear detrending
                  - 2: Quadratic detrending
                  - Higher orders fit higher-degree polynomials
        
        Attributes (after initialization):
            total: Total frequency array.
                   - Shape: (n_samples,)
                   - Units: Hz
                   - Same as input main_tot
            fit: Deterministic (fitted) frequency component.
                 - Shape: (n_samples,)
                 - Units: Hz
                 - Polynomial fit of order 'order'
            fluc: Stochastic (fluctuation) frequency component.
                  - Shape: (n_samples,)
                  - Units: Hz
                  - Calculated as: fluc = total - fit
                  - Has zero mean (corrected for numerical precision)
            tau: Time array.
                 - Shape: (n_samples,)
                 - Units: seconds
                 - Calculated as: tau = np.arange(n_samples) / fs
            fs: Sampling rate (same as input).
                - Units: Hz
            name: Name identifier (same as input).
            order: Polynomial fit order (same as input).
            p_fit: Polynomial coefficients from fit.
                   - Shape: (order+1,)
                   - Units: Hz for coefficient[0], Hz/s for coefficient[1], etc.
            clock_registered: Boolean flag indicating if a clock is registered.
            fourier_freq: Fourier frequencies (set after compute_spectrum()).
                          - Shape: (n_freq,)
                          - Units: Hz
            asd: Amplitude spectral density (set after compute_spectrum()).
                 - Shape: (n_freq,)
                 - Units: Hz/√Hz
        
        Example:
            >>> import numpy as np
            >>> from synctools import FrequencyData
            >>> fs = 10.0
            >>> freq_data = FrequencyData(np.ones(1000) * 1e6, fs, name='signal1')
            >>> print(f"Total: {freq_data.total[0]:.1f} Hz")
            >>> print(f"Fluctuation std: {np.std(freq_data.fluc):.2e} Hz")
        """
        # Validate inputs
        if fs <= 0:
            raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
        
        main_tot = np.asarray(main_tot, dtype=np.float64)
        if main_tot.ndim != 1:
            raise ValueError(f"main_tot must be 1D array, got shape {main_tot.shape}")
        
        if len(main_tot) == 0:
            raise ValueError("main_tot cannot be empty")
        
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")
        
        self.clock_registered = False
        self.fourier_freq = None
        self.asd = None
        self.total = main_tot 
        self.fs = fs
        self.name = name
        self.order = order
        self.tau = np.arange(len(self.total))/fs
        self.fit, self.p_fit = self.fit_frequency(self.tau, self.total, self.order)
        self.fluc = self.total - self.fit
        # Ensure fluc has zero mean to correct for numerical precision in polynomial fitting
        self.fluc = self.fluc - np.mean(self.fluc)

    def __add__(self, other) -> 'FrequencyData':
        if isinstance(other, FrequencyData):
            return FrequencyData(self.total+other.total, self.fs, order=self.order)
        elif isinstance(other, (float, int)):
            return FrequencyData(self.total+other, self.fs, order=self.order)
        else:
            raise ValueError(f'invalid data type for FrequencyData addition: {type(other)}')

    def __sub__(self, other) -> 'FrequencyData':
        if isinstance(other, FrequencyData):
            return FrequencyData(self.total-other.total, self.fs, order=self.order)
        elif isinstance(other, (float, int)):
            return FrequencyData(self.total-other, self.fs, order=self.order)
        else:
            raise ValueError(f'invalid data type for FrequencyData subtraction: {type(other)}')

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return FrequencyData(self.total*other, self.fs, order=self.order)
        else:
            raise ValueError(f"invalid data type {type(other)}")

    def __rmul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return FrequencyData(self.total*other, self.fs, order=self.order)
        else:
            raise ValueError(f"invalid data type {type(other)}")

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return FrequencyData(self.total/other, self.fs, order=self.order)
        else:
            raise ValueError(f"invalid data type {type(other)}")

    def fit_frequency(
        self,
        tau: npt.NDArray[np.float64],
        data: npt.NDArray[np.float64],
        order: int = 0
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Fit a total frequency with a polynomial function.
        
        Args:
            tau: Time array (s)
            data: Frequency data array to detrend (Hz)
            order: Detrend order (non-negative integer)
        
        Returns:
            Tuple of (data_fit, p_fit) where:
            - data_fit: Fitted frequency data (Hz)
            - p_fit: Polynomial coefficients
        """
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")
        if len(tau) != len(data):
            raise ValueError(
                f"tau and data must have same length ({len(tau)} vs {len(data)})"
            )
        p_fit = np.polyfit(tau, data, deg=order)
        data_fit = np.polyval(p_fit, tau)
        return data_fit, p_fit
    
    def register_differential_clock(self, clock) -> None:
        """Register a differential clock signal from this to the primary.
        
        Registers a Clock instance that represents the differential clock signal
        between this frequency signal and the primary time reference. This clock
        is used later in timing_transformation() to correct for time offsets and
        clock jitter.
        
        Args:
            clock: Clock class instance.
                  - Must be a Clock object
                  - Contains differential clock frequency signal (clock.rd)
                  - Will be deep-copied and stored as self.diff_clock
        
        Returns:
            None. Sets instance attributes:
            - self.diff_clock: Deep copy of the clock
            - self.clock_registered: Set to True
            - self.timer_type: "raw" or "inv" based on clock.primary_stamped
            - self.clock_sign: -1 or +1 based on clock.primary_stamped
        
        Example:
            >>> from synctools import Clock, FrequencyData
            >>> fd = FrequencyData(signal, fs=10.0)
            >>> clock = Clock(clock_rd)  # clock_rd is FrequencyData
            >>> fd.register_differential_clock(clock)
            >>> print(f"Clock registered: {fd.clock_registered}")
        """
        self.diff_clock = copy.deepcopy(clock)
        self.timer_type = "raw" if self.diff_clock.primary_stamped else "inv"
        self.clock_sign = -1 if self.diff_clock.primary_stamped else +1
        self.clock_registered = True

    def compute_spectrum(self, p_lpsd: dict) -> None:
        """Compute a frequency amplitude spectral density (ASD).
        
        Args:
            p_lpsd: SpecKit parameters dictionary
        """
        # use self.total to avoid 'double detrending' due to fluc and p_lpsd
        self.fourier_freq, self.asd = spectra(self.total, self.fs, p_lpsd)

    def timing_transformation(
        self,
        fs: float,
        timer_offset: float = 0.0,
        interp_order: int = 121,
        n_trunc: int = 150,
        Doppler_type: str = "total",
        shifts: Optional[npt.NDArray[np.float64]] = None
    ) -> None:
        """Transform reference time frame to primary using differential clock signal.
        
        This method performs time-stamping and Doppler correction to synchronize
        this frequency signal to the primary time reference using a registered
        differential clock signal.
        
        Args:
            fs: Data rate (sampling frequency).
                - Units: Hz
                - Must be > 0
                - Should match self.fs
            timer_offset: Timer offset to add to clock signal.
                         - Units: seconds
                         - Added to the clock timer before computing shifts
            interp_order: Interpolation order for time-shifting.
                         - Must be positive integer
                         - Typical range: 5-121
                         - Higher values: better accuracy, slower computation
            n_trunc: Number of points to truncate at each end of arrays.
                    - Must satisfy: n_trunc < len(data) // 2
                    - Removes edge effects from time-shifting
                    - Applied to both this signal and registered clock
            Doppler_type: Type of Doppler factor to use.
                         - "total": Use total frequency (includes deterministic drift)
                         - "fit": Use fitted (deterministic) frequency only
            shifts: Optional pre-computed shifts array.
                   - Shape: (n_samples,)
                   - Units: samples (fractional)
                   - If provided, timer_offset is ignored and this array is used directly
                   - Must have same length as self.total
        
        Returns:
            None. Modifies instance in place:
            - self.total, self.fit, self.fluc: Time-shifted and Doppler-corrected
            - self.tau: Truncated time array
            - self.clock_correction_term: Clock correction term computed
            - self.diff_clock: Also time-shifted and truncated
        
        Raises:
            ValueError: If validation fails (invalid fs, interp_order, n_trunc,
                       Doppler_type, or shifts length mismatch).
        
        Note:
            Requires that a clock has been registered via register_differential_clock().
            The clock signal is used to compute time shifts and Doppler factors.
        
        Example:
            >>> fd = FrequencyData(signal, fs=10.0)
            >>> fd.register_differential_clock(clock)
            >>> fd.timing_transformation(
            ...     fs=10.0, timer_offset=0.0, interp_order=121, n_trunc=150
            ... )
        """
        if fs <= 0:
            raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
        if interp_order <= 0:
            raise ValueError(f"interp_order must be positive, got {interp_order}")
        if n_trunc < 0:
            raise ValueError(f"n_trunc must be non-negative, got {n_trunc}")
        if n_trunc >= len(self.total) // 2:
            raise ValueError(
                f"n_trunc ({n_trunc}) must be < len(data) // 2 ({len(self.total) // 2})"
            )
        if Doppler_type not in ("total", "fit"):
            raise ValueError(f"Doppler_type must be 'total' or 'fit', got {Doppler_type}")
        if shifts is not None:
            shifts = np.asarray(shifts, dtype=np.float64)
            if len(shifts) != len(self.total):
                raise ValueError(
                    f"shifts must have same length as data "
                    f"({len(shifts)} vs {len(self.total)})"
                )

        if shifts is None:
            self.diff_clock.add_timer_offset(timer_offset)
            _shifts = fs*self.diff_clock.tshift.total[self.timer_type]
        else:
            _shifts = copy.copy(shifts)

        if self.diff_clock.primary_stamped:
            Doppler_factor = 1 + getattr(self.diff_clock.rd,Doppler_type)
        else:
            self.diff_clock.time_stamping(_shifts, factor=1, order=interp_order)
            Doppler_factor = 1 / (1 + getattr(self.diff_clock.rd,Doppler_type))
        self.time_stamping(_shifts, factor=Doppler_factor, order=interp_order)

        # : truncation
        self.diff_clock.truncation(n_trunc)
        self.truncation(n_trunc)

        # : compute a clock correction term for a stochastic analysis mode
        self.clock_correction_term = self.fit * self.clock_sign*self.diff_clock.rd.fluc / (1 + self.clock_sign*self.diff_clock.rd.fit)

    def time_stamping(
        self,
        shifts: npt.NDArray[np.float64],
        factor: float = 1.0,
        order: int = 121
    ) -> None:
        """Time-stamp all frequencies with interpolation.
        
        Args:
            shifts: Number of (fractional) samples to be shifted. 
                   Must have same length as data.
            factor: Scaling factor, e.g. Doppler factor due to clock bias (dimensionless)
            order: Interpolation order (positive integer)
        """
        shifts = np.asarray(shifts, dtype=np.float64)
        if len(shifts) != len(self.total):
            raise ValueError(
                f"shifts must have same length as data "
                f"({len(shifts)} vs {len(self.total)})"
            )
        if order <= 0:
            raise ValueError(f"order must be positive, got {order}")
        
        self.total = factor*timeshift(self.total, shifts, order=order)
        self.fit = factor*timeshift(self.fit, shifts, order=order)
        self.fluc = factor*timeshift(self.fluc, shifts, order=order)

    def truncation(self, n_trunc: int) -> None:
        """Truncate both ends of time-series.
        
        Args:
            n_trunc: Number of points to be truncated at each end of array.
                    Must satisfy n_trunc < len(data) // 2.
        """
        if n_trunc < 0:
            raise ValueError(f"n_trunc must be non-negative, got {n_trunc}")
        if n_trunc >= len(self.total) // 2:
            raise ValueError(
                f"n_trunc ({n_trunc}) must be < len(data) // 2 ({len(self.total) // 2})"
            )
        
        self.tau = self.tau[n_trunc:-n_trunc]
        self.total = self.total[n_trunc:-n_trunc]
        self.fit = self.fit[n_trunc:-n_trunc]
        self.fluc = self.fluc[n_trunc:-n_trunc]