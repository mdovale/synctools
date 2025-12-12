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
from typing import TYPE_CHECKING
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from synctools.frequency import FrequencyData

from synctools.timer import TimerData

class Clock:
    """Class for differential clock measurements.
    
    This class wraps a FrequencyData object (rd) representing a differential clock
    signal and its corresponding TimerData (tshift), providing an interface for
    clock synchronization operations including time-stamping and truncation.
    """
    def __init__(
        self,
        rd: 'FrequencyData',
        timer_offset: float = 0.0,
        timer_sign: int = 1,
        primary_stamped: bool = True,
        name: str = 'diff_clock'
    ) -> None:
        """Initialize a Clock instance.
        
        Args: 
            rd: FrequencyData instance representing the differential clock signal.
                - Shape: rd.total has shape (n_samples,)
                - Units: Hz (fractional frequency)
                - Typically represents clock jitter or differential clock frequency
            timer_offset: Initial timer offset.
                         - Units: seconds
                         - Can be updated later with add_timer_offset()
            timer_sign: Sign multiplier for timer conversion.
                       - Must be -1 or +1
                       - Determines sign convention for converting frequency to time
            primary_stamped: Whether the clock is primary-stamped.
                           - True: Clock is stamped with primary time reference
                           - False: Clock needs time-stamping transformation
            name: Name identifier for this clock instance.
                  Used for debugging and logging.
        
        Attributes (after initialization):
            rd: FrequencyData instance (deep copy of input).
                - Contains differential clock frequency signal
            tshift: TimerData instance derived from rd.
                    - Contains timer data in both "raw" and "inv" (inverse) forms
                    - Access via: tshift.total['raw'] or tshift.total['inv']
                    - Units: seconds
            primary_stamped: Boolean flag (same as input).
            timer_sign: Sign multiplier (same as input).
            timer_offset: Current timer offset (same as input initially).
                          - Units: seconds
            name: Name identifier (same as input).
        
        Raises:
            ValueError: If timer_sign is not -1 or +1.
        
        Example:
            >>> import numpy as np
            >>> from synctools import Clock, FrequencyData
            >>> # Create zero clock reference (no jitter)
            >>> rd = FrequencyData(np.zeros(1000), fs=10.0)
            >>> clock = Clock(rd, timer_offset=0.0)
            >>> print(f"Clock name: {clock.name}")
        """
        if timer_sign not in (-1, 1):
            raise ValueError(f"timer_sign must be -1 or +1, got {timer_sign}")
        
        self.name = name
        self.rd = copy.deepcopy(rd)
        self.primary_stamped = primary_stamped
        self.timer_sign = timer_sign
        self.timer_offset = timer_offset
        self.tshift = TimerData(self.rd, self.rd.fs, sign=self.timer_sign, offset=self.timer_offset)

    def add_timer_offset(self, addition: float) -> None:
        """Add an initial timer offset.
        
        Args:
            addition: Correction added to the current timer offset (s).
        """
        self.timer_offset += addition
        self.tshift = TimerData(self.rd, self.rd.fs, sign=self.timer_sign, offset=self.timer_offset, skip_error_computation=True)

    def time_stamping(
        self,
        shifts: npt.NDArray[np.float64],
        factor: float = 1.0,
        order: int = 121
    ) -> None:
        """Time-stamp a clock measurement.
        
        Args:
            shifts: Number of (fractional) samples to be shifted. Must have same length as data.
            factor: Scaling factor, e.g. Doppler factor due to clock bias (dimensionless).
            order: Interpolation order (positive integer).
        """
        if order <= 0:
            raise ValueError(f"order must be positive, got {order}")
        self.rd.time_stamping(shifts, factor=factor, order=order)

    def truncation(self, n_trunc: int) -> None:
        """Truncate both ends of time-series.
        
        Args:
            n_trunc: Number of points to be truncated at each end of array.
                    Must satisfy n_trunc < len(data) // 2.
        """
        if n_trunc < 0:
            raise ValueError(f"n_trunc must be non-negative, got {n_trunc}")
        data_len = len(self.rd.tau)
        if n_trunc >= data_len // 2:
            raise ValueError(
                f"n_trunc ({n_trunc}) must be < len(data) // 2 ({data_len // 2})"
            )
        
        self.rd.truncation(n_trunc)
        self.tshift.truncation(n_trunc)