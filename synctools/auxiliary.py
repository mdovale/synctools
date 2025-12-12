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
import numpy as np
import numpy.typing as npt
from typing import Dict, Any, Tuple, List, Optional
from scipy import signal
import scipy.optimize as optimize
from scipy import integrate
from speckit import compute_spectrum as lpsd
import logging
logger = logging.getLogger(__name__)


def spectra(
    data: npt.NDArray[np.float64],
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
    if fs <= 0:
        raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
    
    data_len = len(data)
    
    # Handle empty data
    if data_len == 0:
        logger.warning("Empty data array provided to spectra, returning empty arrays")
        return np.array([]), np.array([])
    
    # Warn for suspicious but not fatal situations
    if fs < 0.1:
        logger.warning(f"Very small sampling rate fs={fs} Hz may lead to poor spectral estimation")
    
    if data_len < 100:
        logger.warning(f"Short data sequence (length={data_len}) may lead to poor spectral estimation")
    
    # Check if data length is too short for SpecKit parameters
    # Lmin is the minimum segment length, need at least one segment
    Lmin = p_lpsd.get("Lmin", 100)
    if data_len < Lmin:
        logger.warning(f"Data length ({data_len}) is shorter than Lmin ({Lmin}), returning empty arrays")
        return np.array([]), np.array([])
    
    try:
        psd = lpsd(data, fs, olap=p_lpsd["olap"], 
            bmin=p_lpsd["bmin"], Lmin=p_lpsd["Lmin"], Jdes=p_lpsd["Jdes"], 
            Kdes=p_lpsd["Kdes"], order=p_lpsd["order"], win=p_lpsd["win"], 
            psll=p_lpsd["psll"], verbose=False)
        
        return psd.f, np.sqrt(psd.Gxx)
    except (ZeroDivisionError, ValueError) as e:
        logger.warning(f"SpecKit computation failed: {e}, returning empty arrays")
        return np.array([]), np.array([])

def derive_sign_pairs(
	freq1: npt.NDArray[np.float64],
	freq2: npt.NDArray[np.float64],
	freq3: npt.NDArray[np.float64],
	threshold: float = 1e5
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
	if len(freq1) != len(freq2) or len(freq1) != len(freq3):
		raise ValueError(
			f"All frequency arrays must have same length: "
			f"freq1={len(freq1)}, freq2={len(freq2)}, freq3={len(freq3)}"
		)
	index = -1
	for i in range(3):
		signs = np.full(3,1)
		signs[i] = - signs[i]
		combi = np.min(np.abs(signs[0]*freq1+signs[1]*freq2+signs[2]*freq3))
		if combi < threshold:
			index = i

	output = np.full(3,1)
	if index == -1:
		logger.warning('WARNING: signs for three-signal combination are all +1')
	else:
		output[index] = -output[index]
	return output

def combination_3sig(
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    signs: npt.NDArray[np.int_] = np.array([1, 1, -1])
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
    if freqs.shape[1] != 3:
        raise ValueError(f"freqs must have shape (n_samples, 3), got {freqs.shape}")
    if weights.shape != freqs.shape:
        raise ValueError(
            f"weights must have same shape as freqs ({freqs.shape}), got {weights.shape}"
        )
    if signs.shape != (3,):
        raise ValueError(f"signs must have shape (3,), got {signs.shape}")
    
    return signs[0]*weights[:,0]*freqs[:,0] + signs[1]*weights[:,1]*freqs[:,1] + signs[2]*weights[:,2]*freqs[:,2]

def combination_2sig(
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    signs: npt.NDArray[np.int_] = np.array([1, -1])
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
    if freqs.shape[1] != 2:
        raise ValueError(f"freqs must have shape (n_samples, 2), got {freqs.shape}")
    if weights.shape != freqs.shape:
        raise ValueError(
            f"weights must have same shape as freqs ({freqs.shape}), got {weights.shape}"
        )
    if signs.shape != (2,):
        raise ValueError(f"signs must have shape (2,), got {signs.shape}")
    
    return signs[0]*weights[:,0]*freqs[:,0] + signs[1]*weights[:,1]*freqs[:,1]

def build_kaiser_lpf_taps(fs, f_pass=0.1, f_stop=1.0, attenuation=1000):
    """ filter taps for lpf with Kaiser
    Args:
        fs: sampling frequency (Hz)
        f_pass: beginning of transition band (Hz)
        f_stop: end of transition band (Hz)
        attenuation: attenuation in dB
    """

    fn = fs / 2
    numtaps, beta = signal.kaiserord(attenuation, (f_stop - f_pass) / fn)
    taps = signal.firwin(numtaps, (f_pass + f_stop) / 2, fs=fs, window=('kaiser', beta))
    return taps

def integral_rms(fourier_freq, asd, pass_band=[-np.inf,np.inf]):
    """ compute an integral RMS
    Args:
        fourier_freq: fourier frequency (Hz)
        asd: amplitude spectral density from which RMS is computed
        pass_band: [0] = min, [1] = max
    """

    integral_range_min = max(np.min(fourier_freq), pass_band[0])
    integral_range_max = min(np.max(fourier_freq), pass_band[1])
    f_tmp, asd_tmp = crop_data(fourier_freq, asd, integral_range_min, integral_range_max)
    integral_rms2 = integrate.cumulative_trapezoid(asd_tmp**2, f_tmp, initial=0)
    return np.sqrt(integral_rms2[-1])

def crop_data(x,y,xmin,xmax):
    """ crop data
    Args:
        x: data in x
        y: data in y
        xmin: lower bound of x
        xmax: upper bound of x
    """

    x_tmp = []
    y_tmp = []
    for i in range(len(x)):
        if(x[i] >= xmin and x[i] <= xmax):
            x_tmp.append(x[i])
            y_tmp.append(y[i])
            
    return np.array(x_tmp), np.array(y_tmp)

def convert_frequency_to_phase_in_time(
    data: npt.NDArray[np.float64],
    fs: float
) -> npt.NDArray[np.float64]:
    """Convert frequency to phase by integration.
    
    Args:
        data: Frequency data to be converted to phase (Hz).
        fs: Data rate (Hz). Must be > 0.
    
    Returns:
        Phase array (rad).
    """
    if fs <= 0:
        raise ValueError(f"Sampling rate fs must be > 0, got {fs}")
    
    dt = 1.0/fs
    factor = 2*np.pi*dt
    return factor*np.cumsum(data)

def convert_frequency_to_phase_in_asd(
    fourier_freq: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Convert frequency ASD to phase ASD with Fourier frequency.
    
    Args:
        fourier_freq: Fourier frequency array (Hz). Must have same length as data.
        data: Frequency spectral density to be converted to phase spectral density (Hz/√Hz).
    
    Returns:
        Phase spectral density array (rad/√Hz).
    """
    if len(fourier_freq) != len(data):
        raise ValueError(
            f"fourier_freq and data must have same length "
            f"({len(fourier_freq)} vs {len(data)})"
        )
    if np.any(fourier_freq <= 0):
        raise ValueError("fourier_freq must be > 0 for all elements")
    
    asd = data/fourier_freq
    return asd

def get_asd_delay_factor(fourier_freq, delay):
    """ get delay factor for ASD of split signals
    Args:
        fourier_freq: Fourier frequency (Hz)
        delay: time delay (sec)
    """

    z = -np.pi*np.array(fourier_freq)*delay*2.0j
    return np.abs(1.0 - np.exp(z))

def model_timer_deviation_error(p_fit, tau, iterations=0):
    """ modelling the error of the approximation of the timer deviation
    Args:
        p_fit: polynomial coefficients of a fractional frequency of clock ()
        tau: time array (sec)
        iterations: the number of iterations
    Notes:
        The notation is based on TPS <-> THE
    """

    # : prepare a broad time array to avoid numerical error
    tarray = np.logspace(np.log10(3600), np.log10(3600e3), 1000)

    # : Compute THE in TPS
    the_in_tps = func_the_in_tps(p_fit, tarray)
    
    # : Compute the inverse function, i.e. TPS in THE
    tps_in_the = np.zeros(the_in_tps.size)
    for idx, tauvalue in enumerate(the_in_tps):
        res = optimize.minimize(diff, 1, args=(tauvalue, p_fit), method='Nelder-Mead')
        tps_in_the[idx] = res.x[0]
    
    # : Compute the exact delta tau (wo initial offset)
    exact_del_tau = np.polyval(p_fit, tps_in_the)

    # : Compute the approximated delta tau and its deviation from the exact
    approximate_del_tau = np.polyval(p_fit, the_in_tps)
    diff_apprx_exact_tau = approximate_del_tau - exact_del_tau
    for i in range(iterations):
        approximate_del_tau = np.polyval(p_fit, the_in_tps - approximate_del_tau)
        diff_apprx_exact_tau = approximate_del_tau - exact_del_tau

    # : fit the model over a broad range and derive the model over the measurement time
    fit = np.polyfit(tarray, diff_apprx_exact_tau, p_fit.size-1)
    diff_apprx_exact_tau_in_range = np.polyval(fit, tau)
    diff_apprx_exact_tau_in_range -= diff_apprx_exact_tau_in_range[0] # not clear why this is needed (not needed before the refactoring)

    return diff_apprx_exact_tau_in_range

def func_the_in_tps(p_fit, x):
    """ function to generate THE in TPS
    Args:
        p_fit: polynomial coefficient
        x: data in time
    """

    return np.polyval(p_fit,x) + x

def diff(inverse_x, original_the_in_tps, p_fit):
    """ function to generate THE in TPS
    """

    new_the_in_tps = func_the_in_tps(p_fit, inverse_x)
    return (new_the_in_tps - original_the_in_tps)**2

def components_for_balancing(bsigs, size):
    """ Prepare complementary frequencies and weights for balancing
    Args:
        bsigs: list of the three one-signal classes for balanced detection
        size: size of signal array
    """

    bfreqs = []
    weights = np.ones((size,3))
    for i, bs in enumerate(bsigs):
        bf = bs.total if bs is not None else np.zeros(size)
        bfreqs.append(bf)

        if bs is not None:
            weights[:,i] /= 2
    bfreqs = np.array(bfreqs).T

    return bfreqs, weights
