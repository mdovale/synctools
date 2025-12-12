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
from typing import List, Optional, Dict, Any, Callable, Union, Tuple
from scipy import signal
import scipy.optimize as optimize
import logging
logger = logging.getLogger(__name__)

from synctools.auxiliary import (
    build_kaiser_lpf_taps,
    integral_rms,
    convert_frequency_to_phase_in_time,
    spectra,
    get_asd_delay_factor,
    combination_2sig,
    combination_3sig,
)
from synctools.frequency import FrequencyData
from synctools.clock import Clock
from synctools.signals import TwoSignals, ThreeSignals
from functools import partial


class Synchronization:
    def __init__(
        self,
        combination: Callable,
        desync: Union['TwoSignals', 'ThreeSignals'],
        fs: float,
        p_lpsd: Dict[str, Any],
        model: str = "total",
        domain: str = "time",
        method: str = "Powell",
        interp_order: int = 121,
        n_trunc: int = 150,
        lpf_cutoff: float = 0.8,
        myfolder: str = '/result_sync',
        name: str = ""
    ) -> None:
        """Class for clock synchronization between multiple phasemeters.
        
        This class performs optimization-based synchronization of frequency signals
        from multiple phasemeters, correcting for time offsets and clock jitter.
        
        Args:
            combination: A governing combination functor.
                        - Must be combination_2sig or combination_3sig
                        - Function signature: (freqs, weights) -> combined_freq
                        - freqs: shape (n_samples, n_signals), units: Hz
                        - weights: shape (n_samples, n_signals), dimensionless
                        - Returns: shape (n_samples,), units: Hz
            desync: Desynchronized combination instance.
                   - TwoSignals or ThreeSignals object
                   - Contains unsynchronized signal combination
            fs: Data rate (sampling frequency).
                - Units: Hz
                - Must be > 0
            p_lpsd: SpecKit parameters dictionary.
                   - Required for spectral analysis
                   - See speckit documentation for required keys
            model: Model of clock synchronization.
                  - "total": Use total frequency (includes deterministic drift)
                  - "fluc": Use fluctuation frequency (stochastic component only)
            domain: Domain in which the RMS is computed in TDIR.
                   - "time": Optimize in time domain
                   - "freq": Optimize in frequency domain
            method: Optimization method for scipy.optimize.minimize.
                   - Common choices: "Powell", "TNC", "Nelder-Mead", "L-BFGS-B"
            interp_order: Interpolation order for time-shifting.
                         - Must be positive integer
                         - Typical range: 5-121
                         - Higher values: better accuracy, slower computation
            n_trunc: Number of points to truncate at each end of arrays.
                    - Must satisfy: n_trunc < len(data) // 2
                    - Removes edge effects from time-shifting
            lpf_cutoff: Low-pass filter cutoff frequency.
                       - Units: Hz
                       - Must be > 0 and < fs/2
                       - Used for filtering synchronized results
            myfolder: Folder name for results (legacy parameter, currently unused).
            name: Name identifier for this instance.
        
        Attributes (after processing() is called):
            timer_offsets: Optimized time offsets.
                          - Shape: (n_signals - 1,)
                          - Units: seconds
                          - Time offsets for secondary signals relative to primary
            freq: Dictionary containing synchronized frequency data.
                  - freq['time']: Time domain frequency, shape (n_samples_trunc,), units: Hz
                  - freq['asd']: Frequency ASD, shape (n_freq,), units: Hz/√Hz
            phase: Dictionary containing synchronized phase data.
                   - phase['time']: Time domain phase, shape (n_samples_trunc,), units: rad
                   - phase['asd']: Phase ASD, shape (n_freq,), units: rad/√Hz
            fourier_freq: Fourier frequencies.
                          - Shape: (n_freq,)
                          - Units: Hz
            TDIR_precision: TDIR (Time Delay Interferometry Residual) precision estimate.
                           - Units: dimensionless
                           - Lower values indicate better synchronization
            TDIR_residual_asd: Residual ASD after synchronization.
                              - Shape: (n_freq,)
                              - Units: rad/√Hz
                              - Residual phase noise after correction
            tau: Time array after truncation.
                 - Shape: (n_samples_trunc,)
                 - Units: seconds
            freq_filt: Low-pass filtered frequency.
                       - Shape: (n_samples_trunc,)
                       - Units: Hz
            phase_filt: Low-pass filtered phase.
                        - Shape: (n_samples_trunc,)
                        - Units: rad
            ccs: List of synchronized FrequencyData instances (after processing).
            n_clocks: Number of registered clocks.
            is3S: Boolean indicating if this is a three-signal synchronization.
        
        Raises:
            ValueError: If input validation fails (invalid fs, model, domain, etc.).
        
        Example:
            >>> from synctools import Synchronization, ThreeSignals
            >>> from synctools.auxiliary import combination_3sig
            >>> # After creating desync object and combination function
            >>> sync = Synchronization(
            ...     combination_3sig, desync, fs=10.0, p_lpsd=p_lpsd,
            ...     model="fluc", domain="time", method="Nelder-Mead"
            ... )
            >>> sync.processing(signals, init_offsets=[0.0, 0.0])
            >>> print(f"Time offsets: {sync.timer_offsets} s")
        """
        # Validate inputs
        if fs <= 0:
            raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
        if model not in ("total", "fluc"):
            raise ValueError(f"model must be 'total' or 'fluc', got {model}")
        if domain not in ("time", "freq"):
            raise ValueError(f"domain must be 'time' or 'freq', got {domain}")
        if interp_order <= 0:
            raise ValueError(f"interp_order must be positive, got {interp_order}")
        if n_trunc < 0:
            raise ValueError(f"n_trunc must be non-negative, got {n_trunc}")
        if lpf_cutoff <= 0:
            raise ValueError(f"lpf_cutoff must be > 0, got {lpf_cutoff}")
        if lpf_cutoff >= fs / 2:
            raise ValueError(
                f"lpf_cutoff ({lpf_cutoff}) must be < fs/2 ({fs/2})"
            )

        # : === Register attributes =====
        self.combination = combination
        self.desync = desync
        self.fs = fs
        self.p_lpsd = p_lpsd
        self.model = model
        self.domain = domain
        self.method = method
        self.interp_order = interp_order
        self.n_trunc = n_trunc
        self.lpf_cutoff = lpf_cutoff
        self.design_filter_taps()
        self.myfolder = myfolder
        self.name = name

    def design_filter_taps(self) -> None:
        """Design filter taps for LPF (and potentially BPF).
        
        Creates Kaiser window low-pass filter taps based on lpf_cutoff and fs.
        """
        # : design low-pass filter taps
        # (generated in any case for a result in the time domain)
        width = 0.5*self.lpf_cutoff
        f_pass = 0.5*(2.0*self.lpf_cutoff - width)
        f_stop = 0.5*(2.0*self.lpf_cutoff + width)
        self.lpf_taps = build_kaiser_lpf_taps(fs=self.fs, f_pass=f_pass, f_stop=f_stop)
        self.lpf_taps = (self.lpf_taps,[1])
        self.lpf_size = len(self.lpf_taps[0])

    def processing(
        self,
        ccs: List,
        init_offsets: List[float],
        weights: Union[str, List[float]] = 'default',
        bypass: bool = False
    ) -> None:
        """Process synchronization of carrier signals.
        
        This method performs the actual synchronization optimization, correcting
        time offsets between signals and generating synchronized results.
        
        Args:
            ccs: List of carrier-carrier frequency instances.
                 - Must be FrequencyData objects
                 - Must have 2 or 3 elements, matching the combiner type
                 - ccs[0] is the primary signal, ccs[1:] are secondary signals
                 - All must have the same length and sampling rate
            init_offsets: List of initial guesses for timer offsets.
                         - Shape: (len(ccs) - 1,)
                         - Units: seconds
                         - Used as starting point for optimization
                         - One offset per secondary signal (relative to primary)
            weights: Weights for each carrier in the combination.
                    - If 'default': uses equal weights (1.0 for all)
                    - If List[float]: must have length len(ccs)
                    - Units: dimensionless
            bypass: If True, bypass optimization and use init_offsets directly.
                   - Useful for testing or when offsets are already known
                   - Still performs time-stamping and truncation
        
        Returns:
            None. Results are stored in instance attributes:
            - timer_offsets: Optimized (or bypassed) time offsets
            - freq: Synchronized frequency data
            - phase: Synchronized phase data
            - TDIR_precision: Synchronization quality metric
            - Other attributes as documented in __init__
        
        Raises:
            ValueError: If validation fails (number of carriers vs combiner mismatch,
                       invalid init_offsets length, incompatible signal lengths, etc.)
        
        Example:
            >>> sync = Synchronization(...)
            >>> signals = [fd1, fd2, fd3]  # FrequencyData objects
            >>> sync.processing(signals, init_offsets=[0.0, 0.0])
            >>> print(f"Optimized offsets: {sync.timer_offsets} s")
        """
        # Validate number of carriers
        if len(ccs) < 2 or len(ccs) > 3:
            raise ValueError(
                f"Number of carriers must be 2 or 3, got {len(ccs)}"
            )
        
        # Validate combiner consistency
        # Check if combiner expects 2 or 3 signals by inspecting the function
        from synctools.auxiliary import combination_2sig, combination_3sig
        from synctools.signals import TwoSignals, ThreeSignals
        
        # Determine expected number of signals from desync object type
        if isinstance(self.desync, ThreeSignals):
            expected_n_signals = 3
        elif isinstance(self.desync, TwoSignals):
            expected_n_signals = 2
        else:
            # Fallback: inspect the combiner function
            combiner_func = self.combination.func if hasattr(self.combination, 'func') else self.combination
            if combiner_func is combination_3sig or combiner_func.__name__ == 'combination_3sig':
                expected_n_signals = 3
            elif combiner_func is combination_2sig or combiner_func.__name__ == 'combination_2sig':
                expected_n_signals = 2
            else:
                # Cannot determine, skip validation
                expected_n_signals = None
        
        if expected_n_signals is not None and len(ccs) != expected_n_signals:
            raise ValueError(
                f"Number of carriers ({len(ccs)}) does not match combiner type: "
                f"expected {expected_n_signals} signals for "
                f"{'ThreeSignals' if expected_n_signals == 3 else 'TwoSignals'} combiner"
            )
        
        # Validate init_offsets length
        if len(init_offsets) != len(ccs) - 1:
            raise ValueError(
                f"init_offsets must have length {len(ccs) - 1} "
                f"(number of secondary signals), got {len(init_offsets)}"
            )
        
        # Validate weights
        if weights != 'default':
            if len(weights) != len(ccs):
                raise ValueError(
                    f"weights must have length {len(ccs)} (number of carriers), "
                    f"got {len(weights)}"
                )

        # : === Register signals =====
        self.ccs = copy.deepcopy(ccs)
        self.init_offsets = init_offsets
        self.n_clocks = len([cc.diff_clock for cc in self.ccs if cc.clock_registered]) # number of secondary clocks
        self.is3S = True if len(self.ccs)==3 else False # three signal sync or not
        if weights == 'default':
            self.weights = [1]*len(self.ccs)
        else:
            self.weights = weights

        # : === Synchronization =====
        self.timer_offsets = self.run_optimization(self.ccs) if not bypass else np.array(init_offsets)
        self.generate_performances()

    def run_optimization(self, ccs: List) -> npt.NDArray[np.float64]:
        """Run the whole optimization.
        
        Args:
            ccs: List of carrier-carrier frequency instances.
        
        Returns:
            Optimized timer offsets array (s).
        """
        # : optimization
        p_sync = optimize.minimize(fun=self.f_timer_offset, x0=self.init_offsets, args=(ccs), method=self.method)
        print(f"# ===== Synchronization Result ["+self.name+"] ==========")
        print(f"    TDIR result (sec) = {p_sync.x}")
        print(f"    TDIR success = {p_sync.success}")
        print(f"    TDIR message = {p_sync.message}")
        timer_offsets = p_sync.x

        # : update carriers with optimized timers
        self.update_timer_and_time_stamping_and_truncation(self.ccs, timer_offsets)
        self.tau = self.ccs[0].tau

        return timer_offsets

    def f_timer_offset(
        self,
        param: npt.NDArray[np.float64],
        ccs: List
    ) -> float:
        """Optimize the initial timer offset.
        
        Args:
            param: Parameter to be optimized, i.e. timer offset array (s).
            ccs: List of the carrier one-signal classes.
        
        Returns:
            RMS value (dimensionless).
        """

        _ccs = copy.deepcopy(ccs)
        timer_offsets = param
        self.update_timer_and_time_stamping_and_truncation(_ccs, timer_offsets)

        if self.domain=="time":
            _, _, phase = self.IO_compute_TDIR_output(_ccs, skip_asd_computation=True)
            phase_output_t = phase["time"]
            
            # Check if phase output is empty or too short
            if len(phase_output_t) == 0:
                # Return large penalty for empty data
                return 1e10
            
            phase_output_t = signal.detrend(phase_output_t, type='linear')
            RMS = 0
            tap_size = len(self.lpf_taps[0])*len(self.lpf_taps[1])
            start = tap_size*5 # long warm-up time
            stop = tap_size
            
            # Check if we have enough data after filtering
            if len(phase_output_t) <= start + stop:
                # Return large penalty for insufficient data
                return 1e10
            
            phase_filt_t = signal.lfilter(*self.lpf_taps, phase_output_t)[start:-stop]
            
            # Check if filtered data is empty
            if len(phase_filt_t) == 0:
                return 1e10
            
            RMS += np.mean(phase_filt_t**2)

        elif self.domain=="freq":
            frfr, _, phase = self.IO_compute_TDIR_output(_ccs, skip_asd_computation=False)
            
            # Check if phase output is empty
            if phase["time"] is not None and len(phase["time"]) == 0:
                return 1e10
            
            # Check if ASD computation succeeded
            if phase["asd"] is None or len(phase["asd"]) == 0:
                return 1e10
            
            if frfr is None or len(frfr) == 0:
                return 1e10
            
            RMS = 0
            _rms = integral_rms(frfr, phase["asd"], [0, self.lpf_cutoff])
            RMS += _rms**2
        else:
            raise ValueError(f"invalid domain name {self.domain}")

        return RMS

    def update_timer_and_time_stamping_and_truncation(
        self,
        ccs: List,
        timer_offsets: Union[str, npt.NDArray[np.float64]] = 'default',
        shifts: Union[str, List[npt.NDArray[np.float64]]] = 'default'
    ) -> None:
        """Update time, time-stamping both carrier and clock and truncate both.
        
        Args:
            ccs: List of the carrier one-signal classes.
            timer_offsets: Timer offsets (s). If 'default', uses self.timer_offsets.
            shifts: Optional pre-computed shifts arrays (samples). If 'default', computes from timer_offsets.
        """

        # : determine Doppler type
        if self.model == "total":
            Doppler_type = "total"
        elif self.model == "fluc":
            Doppler_type = "fit"
        else:
            raise ValueError(f"invalid model name {self.model}")

        idx = 0
        for cc in ccs:
            if cc.clock_registered:
                _timer_offsets = None if isinstance(timer_offsets, str) else timer_offsets[idx]
                _shifts = None if isinstance(shifts, str) else shifts[idx]
                cc.timing_transformation(
                    fs=self.fs, timer_offset=_timer_offsets,
                    interp_order=self.interp_order,
                    n_trunc=self.n_trunc, Doppler_type=Doppler_type,
                    shifts=_shifts
                    )
                idx += 1
            else: # only truncation if primary
                cc.truncation(self.n_trunc)

    def IO_compute_TDIR_output(
        self,
        ccs: List,
        skip_asd_computation: bool = False
    ) -> Tuple[Optional[npt.NDArray[np.float64]], Dict[str, Any], Dict[str, Any]]:
        """I/O interface for compute_TDIR_output().
        
        Args:
            ccs: List of the carrier one-signal classes.
            skip_asd_computation: If True, skip ASD computations (e.g., for time-domain optimization).
        
        Returns:
            Tuple of (fourier_freq, freq_dict, phase_dict) where:
            - fourier_freq: Fourier frequency array (Hz) or None if skip_asd_computation
            - freq_dict: Dictionary with 'time' (Hz) and 'asd' (Hz/√Hz) keys
            - phase_dict: Dictionary with 'time' (rad) and 'asd' (rad/√Hz) keys
        """

        if self.model == "total":
            _ccs = []
            for cc in ccs:
                _ccs.append(cc.total)
        elif self.model == "fluc":
            _ccs=[]
            for cc in ccs:
                if not cc.clock_registered:
                    _ccs.append(cc.fluc)
                else:
                    _cc = cc.fluc - cc.clock_correction_term
                    _ccs.append(_cc)
        else:
            raise ValueError(f"invalid model name {self.model}")
        _ccs = np.array(_ccs).T

        return self.compute_TDIR_output(_ccs, skip_asd_computation)

    def compute_TDIR_output(
        self,
        ccs: npt.NDArray[np.float64],
        skip_asd_computation: bool = False
    ) -> Tuple[Optional[npt.NDArray[np.float64]], Dict[str, Any], Dict[str, Any]]:
        """Compute the resulting performance of TDIR for both freq and phase ASDs.
        
        Args:
            ccs: Carrier-carrier beatnote frequencies array (Hz). Shape: (n_samples, n_signals).
            skip_asd_computation: If True, skip ASD computations (e.g., for time-domain optimization).
        
        Returns:
            Tuple of (fourier_freq, freq_dict, phase_dict) where:
            - fourier_freq: Fourier frequency array (Hz) or None if skip_asd_computation
            - freq_dict: Dictionary with 'time' (Hz) and 'asd' (Hz/√Hz) keys
            - phase_dict: Dictionary with 'time' (rad) and 'asd' (rad/√Hz) keys
        """

        weights = np.array([np.full(ccs.shape[0], w) for w in self.weights]).T
        freq_output_t = self.combination(ccs, weights=weights)
        
        # Check if combination resulted in empty array
        if len(freq_output_t) == 0:
            # Return empty results
            if not skip_asd_computation:
                # Return empty arrays with proper structure
                empty_freq = np.array([])
                empty_phase = np.array([])
                return empty_freq, {"time": empty_freq, "asd": empty_phase}, {"time": empty_phase, "asd": empty_phase}
            else:
                empty_freq = np.array([])
                empty_phase = np.array([])
                return None, {"time": empty_freq, "asd": None}, {"time": empty_phase, "asd": None}
        
        phase_output_t = convert_frequency_to_phase_in_time(freq_output_t, self.fs)

        if not skip_asd_computation:
            # Check if data is long enough for spectral analysis
            if len(freq_output_t) < 100:
                # Data too short, return empty ASD arrays
                empty_asd = np.array([])
                empty_frfr = np.array([])
                freq = {"time": freq_output_t, "asd": empty_asd}
                phase = {"time": phase_output_t, "asd": empty_asd}
                return empty_frfr, freq, phase
            
            try:
                frfr, freq_output_asd = spectra(freq_output_t, self.fs, p_lpsd=self.p_lpsd)
                frfr, phase_output_asd = spectra(phase_output_t, self.fs, p_lpsd=self.p_lpsd)
                freq = {"time": freq_output_t, "asd": freq_output_asd}
                phase = {"time": phase_output_t, "asd": phase_output_asd}
            except (ZeroDivisionError, ValueError) as e:
                # If spectra computation fails, return empty ASD arrays
                logger.warning(f"Spectra computation failed: {e}. Returning empty ASD arrays.")
                empty_asd = np.array([])
                empty_frfr = np.array([])
                freq = {"time": freq_output_t, "asd": empty_asd}
                phase = {"time": phase_output_t, "asd": empty_asd}
                return empty_frfr, freq, phase
        else:
            frfr = None
            freq = {"time": freq_output_t, "asd": None}
            phase = {"time": phase_output_t, "asd": None}

        return frfr, freq, phase

    def compute_tdir_accuracy(
        self,
        frfr: npt.NDArray[np.float64],
        ccs: List,
        combi_asd: npt.NDArray[np.float64],
        factor: float = 1.0,
        test_freq: Optional[float] = None
    ) -> Tuple[float, npt.NDArray[np.float64]]:
        """Compute TDIR precision.
        
        Args:
            frfr: Fourier frequency array (Hz).
            ccs: List of carrier-carrier frequency instances (Hz).
            combi_asd: Combination ASD array (rad/√Hz).
            factor: Factor considering the number of clocks to be synchronized
                   (dimensionless). E.g., 1 for one secondary and √2 for two secondaries,
                   assuming the same error for them.
            test_freq: Optional test frequency (Hz). If None, uses minimum value over band.
        
        Returns:
            Tuple of (TDIR_precision, TDIR_residual_asd) where:
            - TDIR_precision: TDIR accuracy (s)
            - TDIR_residual_asd: Residual phase noise ASD array (rad/√Hz)
        """

        # : choose a test Fourier frequency
        if test_freq is None: # use the minimum value over the band
            idx = np.argmin(combi_asd)
        else: # use the value at the tone
            idx = np.argmin(np.abs(np.array(frfr) - test_freq))

        # : compute a TDIR accuracy (sec)
        factor = np.sqrt(self.n_clocks)
        if ccs[0].asd == None:
            ccs[0].compute_spectrum(self.p_lpsd)
        input_phase_asd = ccs[0].asd/ccs[0].fourier_freq
        TDIR_precision = np.arcsin(combi_asd[idx]/(2*factor*input_phase_asd[idx]))/(np.pi*frfr[idx])

        # : derive a residual phase noise
        delay_factor = get_asd_delay_factor(ccs[0].fourier_freq, TDIR_precision)
        TDIR_residual_asd = factor*delay_factor*input_phase_asd
        if test_freq is None:
            print(f"    TDIR accuracy = {TDIR_precision} s (pass band = {[0,self.lpf_cutoff]})")
        else:
            print(f"    TDIR accuracy at {test_freq:.4f} Hz = {TDIR_precision} s")

        return TDIR_precision, TDIR_residual_asd

    def generate_performances(self) -> None:
        """Generate performance results with synchronized signals.
        
        Computes synchronized combination, filtered signals, and TDIR accuracy.
        """

        # : compute a synchronized combination
        self.fourier_freq, self.freq, self.phase = self.IO_compute_TDIR_output(self.ccs)
        
        # Check if data is empty
        if len(self.freq["time"]) == 0 or len(self.phase["time"]) == 0:
            # Set empty filtered arrays
            self.freq_filt = np.array([])
            self.phase_filt = np.array([])
            self.TDIR_precision = np.nan
            self.TDIR_residual_asd = np.array([])
            return
        
        self.freq_filt = signal.detrend(self.freq["time"], type='constant')
        
        # Check if we have enough data for filtering
        if len(self.freq_filt) > 0:
            self.freq_filt = signal.lfilter(*self.lpf_taps, self.freq_filt) # use LPF not to reject slow drifs in any case
        else:
            self.freq_filt = np.array([])
        
        self.phase_filt = signal.detrend(self.phase["time"], type='linear')
        
        # Check if we have enough data for filtering
        if len(self.phase_filt) > 0:
            self.phase_filt = signal.lfilter(*self.lpf_taps, self.phase_filt) # use LPF not to reject slow drifs in any case
        else:
            self.phase_filt = np.array([])

        # Only compute TDIR accuracy if we have valid ASD data
        if self.phase["asd"] is not None and len(self.phase["asd"]) > 0 and self.fourier_freq is not None and len(self.fourier_freq) > 0:
            self.TDIR_precision, self.TDIR_residual_asd = self.compute_tdir_accuracy(self.fourier_freq,
                self.ccs, combi_asd=self.phase["asd"])
        else:
            self.TDIR_precision = np.nan
            self.TDIR_residual_asd = np.array([])

        # : compute timer deviation errors for registered clocks
        for cc in self.ccs:
            if cc.clock_registered:
                cc.diff_clock.tshift.compute_timer_deviation_error(fs=self.fs)


def sync_signals(
    in_signals: List[npt.NDArray[np.float64]], 
    fs: float,
    p_lpsd: Dict[str, Any],
    init_offsets: Optional[List[float]] = None,
    model: str = "total", 
    domain: str = "time", 
    method: str = "Nelder-Mead",
    interp_order: int = 121,
    n_truncate: Optional[int] = None,
    clock_refs: Optional[List[npt.NDArray[np.float64]]] = None, 
    logger: Optional[logging.Logger] = None
) -> Tuple[Union[TwoSignals, ThreeSignals], Synchronization]:
    """Synchronize multiple frequency signals from different phasemeters.
    
    This function corrects for time offsets and clock jitter between multiple
    frequency time series, enabling accurate combination and analysis.
    
    Args:
        in_signals: List of input frequency signal arrays. 
                   - Shape: Each array must be 1D with shape (n_samples,)
                   - Units: Hz (frequency)
                   - Must contain exactly 2 or 3 signals
                   - All arrays must have the same length
        fs: Sampling rate.
            - Units: Hz
            - Must be > 0
        p_lpsd: SpecKit parameters dictionary.
                Required keys: "olap", "bmin", "Lmin", "Jdes", "Kdes", "order",
                "win", "psll", "pool". See speckit documentation for details.
        init_offsets: Optional list of initial timer offset guesses.
                     - Shape: (n_signals - 1,)
                     - Units: seconds
                     - If None, defaults to zeros
                     - Used as starting point for optimization
        model: Clock synchronization model.
               - "total": Synchronize using total frequency (includes deterministic drift)
               - "fluc": Synchronize using fluctuation frequency (stochastic component only)
        domain: Domain for RMS computation in TDIR (Time Delay Interferometry Residual).
                - "time": Optimize in time domain
                - "freq": Optimize in frequency domain
        method: Optimization method for scipy.optimize.minimize.
                Common choices: "Nelder-Mead", "Powell", "TNC", "L-BFGS-B"
        interp_order: Interpolation order for time-shifting operations.
                     - Must be positive integer
                     - Higher values give better accuracy but slower computation
                     - Typical range: 5-121
        n_truncate: Number of points to truncate at each end of arrays.
                   - Must satisfy: n_truncate < len(data) // 2
                   - If None, auto-calculated based on init_offsets or defaults to 150
                   - Truncation removes edge effects from time-shifting
        clock_refs: Optional list of clock reference arrays for clock jitter correction.
                   - Shape: Each array must be 1D with shape (n_samples,)
                   - Units: Hz (differential clock frequency)
                   - Length must be len(in_signals) - 1 (one per secondary signal)
                   - If None, assumes zero clock jitter
        logger: Optional logger instance for debug output.
                If None, uses module logger.
    
    Returns:
        Tuple of (unsynced_obj, synced_obj):
        
        - unsynced_obj (TwoSignals or ThreeSignals): Object containing the
          unsynchronized signal combination. Important attributes:
          * main: Main signal combination element
          * main.freq: Combined frequency (Hz), shape (n_samples,)
          * main.phase: Combined phase (rad), shape (n_samples,)
          * main.freq_asd: Frequency ASD (Hz/√Hz), shape (n_freq,)
          * fourier_freq: Fourier frequencies (Hz), shape (n_freq,)
        
        - synced_obj (Synchronization): Object containing synchronized results.
          Important attributes:
          * timer_offsets: Optimized time offsets (s), shape (n_signals-1,)
          * freq['time']: Synchronized frequency in time domain (Hz), shape (n_samples_trunc,)
          * freq['asd']: Frequency ASD (Hz/√Hz), shape (n_freq,)
          * phase['time']: Synchronized phase in time domain (rad), shape (n_samples_trunc,)
          * phase['asd']: Phase ASD (rad/√Hz), shape (n_freq,)
          * fourier_freq: Fourier frequencies (Hz), shape (n_freq,)
          * TDIR_precision: TDIR precision estimate (dimensionless)
          * TDIR_residual_asd: Residual ASD after synchronization (rad/√Hz), shape (n_freq,)
          * tau: Time array after truncation (s), shape (n_samples_trunc,)
          * freq_filt: Low-pass filtered frequency (Hz), shape (n_samples_trunc,)
          * phase_filt: Low-pass filtered phase (rad), shape (n_samples_trunc,)
    
    Raises:
        ValueError: If input validation fails (invalid array shapes, incompatible
                   lengths, invalid parameter values, etc.).
    
    Example:
        >>> import numpy as np
        >>> from synctools import sync_signals
        >>> fs = 10.0
        >>> signal1 = 1e6 * np.ones(1000)
        >>> signal2 = 1e6 * np.ones(1000)
        >>> p_lpsd = {...}  # SpecKit parameters
        >>> unsynced, synced = sync_signals(
        ...     [signal1, signal2], fs, p_lpsd, init_offsets=[0.0]
        ... )
        >>> print(f"Time offset: {synced.timer_offsets[0]:.6f} s")
    """
    # Validate inputs
    if fs <= 0:
        raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
    
    if len(in_signals) < 2:
        raise ValueError(
            f"Insufficient input signals for synchronization: got {len(in_signals)}, "
            f"need at least 2"
        )
    if len(in_signals) > 3:
        raise ValueError(
            f"Too many input signals: got {len(in_signals)}, maximum is 3"
        )
    
    # Convert to numpy arrays and validate
    in_signals_arrays = []
    for i, sig in enumerate(in_signals):
        sig_array = np.asarray(sig, dtype=np.float64)
        if sig_array.ndim != 1:
            raise ValueError(
                f"Signal {i} must be 1D array, got shape {sig_array.shape}"
            )
        if len(sig_array) == 0:
            raise ValueError(f"Signal {i} cannot be empty")
        in_signals_arrays.append(sig_array)
    
    # Check all signals have the same length
    first_len = len(in_signals_arrays[0])
    for i, sig in enumerate(in_signals_arrays[1:], 1):
        if len(sig) != first_len:
            raise ValueError(
                f"All input signals must have the same length: "
                f"signal 0 has length {first_len}, signal {i} has length {len(sig)}"
            )
    
    # Validate model and domain
    if model not in ("total", "fluc"):
        raise ValueError(f"model must be 'total' or 'fluc', got {model}")
    if domain not in ("time", "freq"):
        raise ValueError(f"domain must be 'time' or 'freq', got {domain}")
    
    if interp_order <= 0:
        raise ValueError(f"interp_order must be positive, got {interp_order}")
    
    # Validate n_truncate if provided
    if n_truncate is not None:
        if n_truncate < 0:
            raise ValueError(f"n_truncate must be non-negative, got {n_truncate}")
        if n_truncate >= first_len // 2:
            raise ValueError(
                f"n_truncate ({n_truncate}) must be < len(data) // 2 ({first_len // 2})"
            )
    
    signals = []
    clocks = []
    init_dt: List[float] = []

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.debug(f"Starting up...")

    if len(in_signals_arrays) == 2:
        signals.append(FrequencyData(in_signals_arrays[0], fs))
        signals.append(FrequencyData(in_signals_arrays[1], fs))
    elif len(in_signals_arrays) == 3:
        signals.append(FrequencyData(in_signals_arrays[0], fs))
        signals.append(FrequencyData(in_signals_arrays[1], fs))
        signals.append(FrequencyData(in_signals_arrays[2], fs))

    if init_offsets is not None:
        if len(init_offsets) != len(signals)-1:
            raise ValueError(
                f"init_offsets must have length {len(signals)-1} "
                f"(number of secondary signals), got {len(init_offsets)}"
            )
        init_dt = init_offsets
        if n_truncate is None:
            n_truncate = int(2*abs(max(init_dt) * fs))
            # Ensure n_truncate is valid
            if n_truncate >= first_len // 2:
                n_truncate = max(1, first_len // 2 - 1)
                logger.warning(
                    f"Auto-calculated n_truncate would be too large, "
                    f"setting to {n_truncate}"
                )
    else:
        for _ in range(len(signals)):
            init_dt.append(0.0)
        if n_truncate is None:
            n_truncate = 150
            # Ensure n_truncate is valid
            if n_truncate >= first_len // 2:
                n_truncate = max(1, first_len // 2 - 1)
                logger.warning(
                    f"Default n_truncate ({150}) is too large for data length "
                    f"({first_len}), setting to {n_truncate}"
                )

    init_dsamples = list(np.array(init_dt)*fs)
    logger.debug(f"Initial offsets of {init_dt} seconds ({init_dsamples} samples), truncation set to {n_truncate:d}")

    if clock_refs is not None:
        if len(clock_refs) != len(signals)-1:
            raise ValueError(
                f"clock_refs must have length {len(signals)-1} "
                f"(number of secondary signals), got {len(clock_refs)}"
            )
        # Validate clock_refs array lengths
        for i, ref in enumerate(clock_refs):
            ref_array = np.asarray(ref, dtype=np.float64)
            if ref_array.ndim != 1:
                raise ValueError(
                    f"clock_refs[{i}] must be 1D array, got shape {ref_array.shape}"
                )
            if len(ref_array) != first_len:
                raise ValueError(
                    f"clock_refs[{i}] must have same length as input signals "
                    f"({first_len}), got {len(ref_array)}"
                )

    logger.debug("Creating and registering clock objects...")

    if clock_refs is None:
        logger.debug("No clock reference provided, assuming zero clock jitter")
        for _ in range(len(signals)-1):
            # Create zero clock reference
            clock_rd = FrequencyData(np.zeros(first_len), fs)
            clocks.append(Clock(clock_rd))
    else:
        logger.debug("Clock reference provided, using custom clock jitter")
        for ref in clock_refs:
            # Create FrequencyData from clock reference array
            clock_rd = FrequencyData(ref, fs)
            clocks.append(Clock(clock_rd))

    for i, clk in enumerate(clocks):
        signals[i+1].register_differential_clock(clk)

    if len(signals) == 2:
        logger.debug("Creating TwoSignals object")
        unsynced_obj = TwoSignals([*signals], p_lpsd)
        synced_obj_name = "2-signal-sync"
    else:
        logger.debug("Creating ThreeSignals object")
        unsynced_obj = ThreeSignals([*signals], p_lpsd)
        logger.debug(f"Derived signs for the three-signal combination: {unsynced_obj.signs}")  
        synced_obj_name = "3-signal-sync"  

    if isinstance(unsynced_obj, ThreeSignals):
        signal_combiner = partial(combination_3sig, signs=unsynced_obj.signs)
    else: 
        signal_combiner = partial(combination_2sig)
    
    synced_obj = Synchronization(signal_combiner, unsynced_obj, fs, p_lpsd,
        model=model, domain=domain, method=method,
        interp_order=interp_order, n_trunc=n_truncate,
        myfolder='/result_sync/', name=synced_obj_name
    )

    logger.debug("Synchronizing...")
    synced_obj.processing(signals, init_offsets=init_dt)
    final_dsamples = list(np.array(synced_obj.timer_offsets)*fs)
    logger.debug(f"Synchronization finished with dt = {synced_obj.timer_offsets} seconds ({final_dsamples} samples)")

    return unsynced_obj, synced_obj


def sync_multiple_twosignals(
    in_signals: List[npt.NDArray[np.float64]], 
    fs: float,
    p_lpsd: Dict[str, Any],
    init_offsets: Optional[List[Optional[List[float]]]] = None,
    model: str = "total", 
    domain: str = "time", 
    method: str = "Nelder-Mead",
    interp_order: int = 121,
    n_truncate: Optional[int] = None,
    clock_refs: Optional[List[Optional[npt.NDArray[np.float64]]]] = None, 
    logger: Optional[logging.Logger] = None
) -> List[Tuple[TwoSignals, Synchronization]]:
    """Perform multiple TwoSignal synchronizations using the first signal as reference.
    
    This function takes a list of N input signals (A, B, C, D, ...) and performs
    TwoSignal synchronization for each pair [A,B], [A,C], [A,D], etc., where A
    is the first signal used as the reference.
    
    Args:
        in_signals: List of input frequency signal arrays. 
                   - Shape: Each array must be 1D with shape (n_samples,)
                   - Units: Hz (frequency)
                   - Must contain at least 2 signals
                   - All arrays must have the same length
                   - First signal (index 0) is used as the reference
        fs: Sampling rate.
            - Units: Hz
            - Must be > 0
        p_lpsd: SpecKit parameters dictionary.
                Required keys: "olap", "bmin", "Lmin", "Jdes", "Kdes", "order",
                "win", "psll", "pool". See speckit documentation for details.
        init_offsets: Optional list of initial timer offset guesses for each pair.
                     - Length: len(in_signals) - 1 (one per secondary signal)
                     - Each element can be None or a list of length 1
                     - If None for a pair, defaults to [0.0] for that pair
                     - Example: [None, [0.1], None] for signals [A,B,C,D] means:
                       - [A,B]: init_offset=[0.0] (default)
                       - [A,C]: init_offset=[0.1]
                       - [A,D]: init_offset=[0.0] (default)
        model: Clock synchronization model.
               - "total": Synchronize using total frequency (includes deterministic drift)
               - "fluc": Synchronize using fluctuation frequency (stochastic component only)
        domain: Domain for RMS computation in TDIR (Time Delay Interferometry Residual).
                - "time": Optimize in time domain
                - "freq": Optimize in frequency domain
        method: Optimization method for scipy.optimize.minimize.
                Common choices: "Nelder-Mead", "Powell", "TNC", "L-BFGS-B"
        interp_order: Interpolation order for time-shifting operations.
                     - Must be positive integer
                     - Higher values give better accuracy but slower computation
                     - Typical range: 5-121
        n_truncate: Number of points to truncate at each end of arrays.
                   - Must satisfy: n_truncate < len(data) // 2
                   - If None, auto-calculated based on init_offsets or defaults to 150
                   - Truncation removes edge effects from time-shifting
        clock_refs: Optional list of clock reference arrays for clock jitter correction.
                   - Length: len(in_signals) - 1 (one per secondary signal)
                   - Each element can be None or a 1D array with shape (n_samples,)
                   - Units: Hz (differential clock frequency)
                   - If None for a pair, assumes zero clock jitter for that pair
        logger: Optional logger instance for debug output.
                If None, uses module logger.
    
    Returns:
        List of tuples, one for each pair [A, B], [A, C], [A, D], etc.
        Each tuple contains (unsynced_obj, synced_obj):
        
        - unsynced_obj (TwoSignals): Object containing the unsynchronized signal
          combination for the pair. Important attributes:
          * main: Main signal combination element
          * main.freq: Combined frequency (Hz), shape (n_samples,)
          * main.phase: Combined phase (rad), shape (n_samples,)
          * main.freq_asd: Frequency ASD (Hz/√Hz), shape (n_freq,)
          * fourier_freq: Fourier frequencies (Hz), shape (n_freq,)
        
        - synced_obj (Synchronization): Object containing synchronized results for
          the pair. Important attributes:
          * timer_offsets: Optimized time offsets (s), shape (1,)
          * freq['time']: Synchronized frequency in time domain (Hz), shape (n_samples_trunc,)
          * freq['asd']: Frequency ASD (Hz/√Hz), shape (n_freq,)
          * phase['time']: Synchronized phase in time domain (rad), shape (n_samples_trunc,)
          * phase['asd']: Phase ASD (rad/√Hz), shape (n_freq,)
          * fourier_freq: Fourier frequencies (Hz), shape (n_freq,)
          * TDIR_precision: TDIR precision estimate (dimensionless)
          * TDIR_residual_asd: Residual ASD after synchronization (rad/√Hz), shape (n_freq,)
          * tau: Time array after truncation (s), shape (n_samples_trunc,)
          * freq_filt: Low-pass filtered frequency (Hz), shape (n_samples_trunc,)
          * phase_filt: Low-pass filtered phase (rad), shape (n_samples_trunc,)
    
    Raises:
        ValueError: If input validation fails (invalid array shapes, incompatible
                   lengths, invalid parameter values, etc.).
    
    Example:
        >>> import numpy as np
        >>> from synctools import sync_multiple_twosignals
        >>> fs = 10.0
        >>> signal_A = 1e6 * np.ones(1000)
        >>> signal_B = 1e6 * np.ones(1000)
        >>> signal_C = 1e6 * np.ones(1000)
        >>> signal_D = 1e6 * np.ones(1000)
        >>> p_lpsd = {...}  # SpecKit parameters
        >>> results = sync_multiple_twosignals(
        ...     [signal_A, signal_B, signal_C, signal_D], fs, p_lpsd
        ... )
        >>> # results[0] contains sync for [A,B]
        >>> # results[1] contains sync for [A,C]
        >>> # results[2] contains sync for [A,D]
        >>> print(f"Time offset A-B: {results[0][1].timer_offsets[0]:.6f} s")
        >>> print(f"Time offset A-C: {results[1][1].timer_offsets[0]:.6f} s")
        >>> print(f"Time offset A-D: {results[2][1].timer_offsets[0]:.6f} s")
    """
    # Validate inputs
    if fs <= 0:
        raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
    
    if len(in_signals) < 2:
        raise ValueError(
            f"Insufficient input signals for synchronization: got {len(in_signals)}, "
            f"need at least 2"
        )
    
    # Convert to numpy arrays and validate
    in_signals_arrays = []
    for i, sig in enumerate(in_signals):
        sig_array = np.asarray(sig, dtype=np.float64)
        if sig_array.ndim != 1:
            raise ValueError(
                f"Signal {i} must be 1D array, got shape {sig_array.shape}"
            )
        if len(sig_array) == 0:
            raise ValueError(f"Signal {i} cannot be empty")
        in_signals_arrays.append(sig_array)
    
    # Check all signals have the same length
    first_len = len(in_signals_arrays[0])
    for i, sig in enumerate(in_signals_arrays[1:], 1):
        if len(sig) != first_len:
            raise ValueError(
                f"All input signals must have the same length: "
                f"signal 0 has length {first_len}, signal {i} has length {len(sig)}"
            )
    
    # Validate model and domain
    if model not in ("total", "fluc"):
        raise ValueError(f"model must be 'total' or 'fluc', got {model}")
    if domain not in ("time", "freq"):
        raise ValueError(f"domain must be 'time' or 'freq', got {domain}")
    
    if interp_order <= 0:
        raise ValueError(f"interp_order must be positive, got {interp_order}")
    
    # Validate n_truncate if provided
    if n_truncate is not None:
        if n_truncate < 0:
            raise ValueError(f"n_truncate must be non-negative, got {n_truncate}")
        if n_truncate >= first_len // 2:
            raise ValueError(
                f"n_truncate ({n_truncate}) must be < len(data) // 2 ({first_len // 2})"
            )
    
    # Validate init_offsets if provided
    n_pairs = len(in_signals_arrays) - 1
    if init_offsets is not None:
        if len(init_offsets) != n_pairs:
            raise ValueError(
                f"init_offsets must have length {n_pairs} (number of pairs), "
                f"got {len(init_offsets)}"
            )
        for i, offset in enumerate(init_offsets):
            if offset is not None and len(offset) != 1:
                raise ValueError(
                    f"init_offsets[{i}] must be None or a list of length 1, "
                    f"got {offset}"
                )
    
    # Validate clock_refs if provided
    if clock_refs is not None:
        if len(clock_refs) != n_pairs:
            raise ValueError(
                f"clock_refs must have length {n_pairs} (number of pairs), "
                f"got {len(clock_refs)}"
            )
        for i, ref in enumerate(clock_refs):
            if ref is not None:
                ref_array = np.asarray(ref, dtype=np.float64)
                if ref_array.ndim != 1:
                    raise ValueError(
                        f"clock_refs[{i}] must be None or a 1D array, "
                        f"got shape {ref_array.shape}"
                    )
                if len(ref_array) != first_len:
                    raise ValueError(
                        f"clock_refs[{i}] must have same length as input signals "
                        f"({first_len}), got {len(ref_array)}"
                    )
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.debug(f"Starting multiple TwoSignal synchronizations with {len(in_signals_arrays)} signals")
    
    # Prepare default values
    if init_offsets is None:
        init_offsets = [None] * n_pairs
    if clock_refs is None:
        clock_refs = [None] * n_pairs
    
    # Perform synchronization for each pair
    results = []
    reference_signal = in_signals_arrays[0]
    
    for i in range(1, len(in_signals_arrays)):
        secondary_signal = in_signals_arrays[i]
        pair_index = i - 1
        
        logger.debug(f"Synchronizing pair [A, signal_{i}] (pair {pair_index + 1}/{n_pairs})")
        
        # Prepare init_offset for this pair
        pair_init_offset = init_offsets[pair_index]
        if pair_init_offset is None:
            pair_init_offset = [0.0]
        
        # Prepare clock_ref for this pair
        pair_clock_ref = clock_refs[pair_index]
        pair_clock_refs = None if pair_clock_ref is None else [pair_clock_ref]
        
        # Perform TwoSignal synchronization
        unsynced_obj, synced_obj = sync_signals(
            [reference_signal, secondary_signal],
            fs,
            p_lpsd,
            init_offsets=pair_init_offset,
            model=model,
            domain=domain,
            method=method,
            interp_order=interp_order,
            n_truncate=n_truncate,
            clock_refs=pair_clock_refs,
            logger=logger
        )
        
        results.append((unsynced_obj, synced_obj))
        logger.debug(
            f"Pair [A, signal_{i}] synchronized with offset = "
            f"{synced_obj.timer_offsets[0]:.6f} s"
        )
    
    logger.debug(f"Completed {n_pairs} TwoSignal synchronizations")
    
    return results