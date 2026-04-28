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
from typing import TYPE_CHECKING, Tuple, Dict
from pytdi.dsp import calculate_advancements
import logging
logger = logging.getLogger(__name__)

from synctools.auxiliary import model_timer_deviation_error

if TYPE_CHECKING:
    from synctools.frequency import FrequencyData

class TimerData:
	def __init__(
		self,
		rd: 'FrequencyData',
		fs: float,
		sign: int = -1,
		offset: float = 0.0,
		skip_error_computation: bool = False
	) -> None:
		"""Class for timer data (integrated clock signal).
		
		This class converts fractional frequency data to timer data through integration,
		providing both "raw" (direct integration) and "inv" (inverse/advancement) forms.
		
		Args:
			rd: The underlying FrequencyData instance containing fractional frequency.
			    - rd.total: Total fractional frequency, shape (n_samples,)
			    - Units: dimensionless (fractional frequency)
			    - Typically represents clock jitter or differential clock frequency
			fs: Data rate (sampling frequency).
			   - Units: Hz
			   - Must be > 0
			sign: Sign multiplier for frequency-to-time conversion.
			     - Must be -1 or +1
			     - If -1, flips the sign of input frequency before integration
			     - Convention: -1 typically used for timer sign against frequency
			offset: Initial timer offset.
			       - Units: seconds
			       - Added to the integrated timer signal
			skip_error_computation: If True, skip computation of timer deviation error.
			                       - Set to True for faster initialization when error
			                         will be computed later
        
		Attributes (after initialization):
			tau: Time array.
			     - Shape: (n_samples,)
			     - Units: seconds
			     - Same as rd.tau
			total: Dictionary containing timer data.
			       - total['raw']: Raw timer (direct integration), shape (n_samples,), units: s
			       - total['inv']: Inverse timer (advancement), shape (n_samples,), units: s
			fit: Dictionary containing fitted (deterministic) timer components.
			     - fit['raw']: Fitted raw timer, shape (n_samples,), units: s
			     - fit['inv']: Fitted inverse timer, shape (n_samples,), units: s
			     - Polynomial fit order is rd.order + 1
			fluc: Dictionary containing fluctuation (stochastic) timer components.
			      - fluc['raw']: Fluctuation raw timer, shape (n_samples,), units: s
			      - fluc['inv']: Fluctuation inverse timer, shape (n_samples,), units: s
			p_fit_model: Polynomial coefficients for timer deviation error model.
			             - Shape: (rd.order + 2,)
			             - Used for modeling timer deviation error
			timer_dev_err: Dictionary containing timer deviation error (if computed).
			               - timer_dev_err['estimate']: Estimated error, shape (n_samples,), units: s
			               - timer_dev_err['model']: Modeled error, shape (n_samples,), units: s
			               - Only set if skip_error_computation=False
        
		Raises:
			ValueError: If fs <= 0 or sign is not -1 or +1.
		
		Example:
			>>> import numpy as np
			>>> from synctools import TimerData, FrequencyData
			>>> rd = FrequencyData(np.ones(1000) * 1e-9, fs=10.0)  # 1 ppb fractional freq
			>>> timer = TimerData(rd, fs=10.0, sign=-1, offset=0.0)
			>>> print(f"Timer range: {timer.total['raw'].min():.2e} to {timer.total['raw'].max():.2e} s")
		"""
		if fs <= 0:
			raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
		if sign not in (-1, 1):
			raise ValueError(f"sign must be -1 or +1, got {sign}")
		
		_rd = copy.deepcopy(rd)
		self.tau = _rd.tau

		# : === create timers ================
		# : raw timers
		total = self.convert_frac_frequency_to_time(sign*_rd.total, fs) + offset
		fit = self.convert_frac_frequency_to_time(sign*_rd.fit, fs) + offset
		fluc = self.convert_frac_frequency_to_time(sign*_rd.fluc, fs)
		# : inverse timers
		order = _rd.order + 1 # + 1 is because timer is the integral of frac. freq
		total_inv = calculate_advancements(total, fs)
		fit_inv, p_fit_inv = self.fit_timer(self.tau, total_inv, order)
		fluc_inv = total_inv - fit_inv
		# : packaging
		self.total = {"raw": total, "inv": total_inv}
		self.fit = {"raw": fit, "inv": fit_inv}
		self.fluc = {"raw": fluc, "inv": fluc_inv}

		# : === compute timer deviation error ================
		self.p_fit_model = np.append(_rd.p_fit, 0.0)
		if not skip_error_computation:
			self.compute_timer_deviation_error(fs)

	def compute_timer_deviation_error(self, fs: float) -> None:
		"""Compute the error of non-inverted timer deviation.
		
		Args:
			fs: Data rate (Hz). Must be > 0.
		"""
		if fs <= 0:
			raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
		
		p_fit_model_tmp = self.p_fit_model # do not update self.p_fit_model itself
		# : estimate (use inverse_timer_wo_large_offset, instead of self.total["inv"])
		total_inv = self.inverse_timer_wo_large_offset(self.total["raw"], fs)
		timer_dev_err_estimate = total_inv - self.total["raw"]
		# : model
		for i in range(p_fit_model_tmp.shape[0] - 1):
			scale = (p_fit_model_tmp.shape[0]-1) - i
			p_fit_model_tmp[i] /= scale
		timer_dev_err_model = model_timer_deviation_error(p_fit_model_tmp, self.tau)
		# : packaging
		self.timer_dev_err = {"estimate": timer_dev_err_estimate, "model":timer_dev_err_model}

	def fit_timer(
		self,
		tau: npt.NDArray[np.float64],
		data: npt.NDArray[np.float64],
		order: int = 0
	) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
		"""Polynomial fit for timer.
		
		Args:
			tau: Time array (s)
			data: Timer data array to detrend (s)
			order: Detrend order (non-negative integer)
		
		Returns:
			Tuple of (data_fit, p_fit) where:
			- data_fit: Fitted timer data (s)
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

	def convert_frac_frequency_to_time(
		self,
		data: npt.NDArray[np.float64],
		fs: float,
		zeroed: bool = False
	) -> npt.NDArray[np.float64]:
		"""Convert fractional frequency to time by integration.
		
		Args:
			data: Fractional frequency data to be converted to time (dimensionless)
			fs: Data rate (Hz). Must be > 0.
			zeroed: If True, anchor initial value to zero.
		
		Returns:
			Timer data array (s)
		"""
		if fs <= 0:
			raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
		
		dt = 1.0/fs
		timer = dt*np.cumsum(data)
		if zeroed:
			timer -= timer[0]

		return timer

	def inverse_timer_wo_large_offset(
		self,
		data: npt.NDArray[np.float64],
		fs: float,
		interp_order: int = 5,
		delta: float = 1e-12,
		maxiter: int = 100
	) -> npt.NDArray[np.float64]:
		"""Inverse timer deviation without a large initial offset for advancements.
		
		To be compared with model_timer_deviation_error(), which simulates only 
		deterministic components.
		
		Args:
			data: Timer data to be inverted (s)
			fs: Data rate (Hz). Must be > 0.
			interp_order: Interpolation order (positive integer)
			delta: Error threshold of the numerical derivation
			maxiter: Maximum number of iterations of numerical computations
		
		Returns:
			Inverse timer data array (s)
		"""
		if fs <= 0:
			raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
		if interp_order <= 0:
			raise ValueError(f"interp_order must be positive, got {interp_order}")
		if maxiter <= 0:
			raise ValueError(f"maxiter must be positive, got {maxiter}")
		
		inverse = calculate_advancements(data-data[0], fs, order=interp_order, delta=delta, maxiter=maxiter)
		inverse += data[0]
		return inverse

	def truncation(self, n_trunc: int) -> None:
		"""Truncate both ends of time-series.
		
		Args:
			n_trunc: Number of points to be truncated at each end of array.
			        Must satisfy n_trunc < len(data) // 2.
		"""
		if n_trunc < 0:
			raise ValueError(f"n_trunc must be non-negative, got {n_trunc}")
		if n_trunc == 0:
			return
		data_len = len(self.tau)
		if n_trunc >= data_len // 2:
			raise ValueError(
				f"n_trunc ({n_trunc}) must be < len(data) // 2 ({data_len // 2})"
			)
		
		self.tau = self.tau[n_trunc:-n_trunc]
		self.total["raw"] = self.total["raw"][n_trunc:-n_trunc]
		self.total["inv"] = self.total["inv"][n_trunc:-n_trunc]
		self.fit["raw"] = self.fit["raw"][n_trunc:-n_trunc]
		self.fit["inv"] = self.fit["inv"][n_trunc:-n_trunc]
		self.fluc["raw"] = self.fluc["raw"][n_trunc:-n_trunc]
		self.fluc["inv"] = self.fluc["inv"][n_trunc:-n_trunc]