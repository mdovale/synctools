# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale

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
VERSION = '1.03'
"""
mokutools.sync: Synchronization of two Liquid Instruments Moku phasemeter data streams

The two Moku share a clock, but their data streams are 
misaligned by a non-integer number of samples
"""
from mokutools.filetools import *
from mokutools.phasemeter import MokuPhasemeterObject
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytdi.dsp import timeshift
from spectools import lpsd
import multiprocessing
from synctools.sync import sync_signals
import logging
logger = logging.getLogger(__name__)

# : ===== User variables ====================
MASTER_COL = '1_freq' # Quantity on master device to use for sync (<channel>_<freq/phase>)
SLAVE_COL = '1_freq' # Quantity on slave device to use for sync (<channel>_<freq/phase>)
"""
The options for MASTER_COL and SLAVE_COL are '<ch>_<signal>':
    '1_phase', '2_phase', '3_phase', '4_phase'
    '1_freq', '2_freq', '3_freq', '4_freq'
"""
MODEL = 'fluc' # Model for TDIR-like synchronization ('fluc' or 'total', default: 'fluc')
DOMAIN = 'freq' # Domain for TDIR-like synchronization ('freq' or 'time', default: 'time')
SOLVER = 'Nelder-Mead' # Solver to use in scipy.optimize.minimize for TDIR-like synchronization
INTERP_ORDER = 121 # Interpolation order for TDIR-like synchronization (default: 121)
RESDIR = './results' # Where to store outputs
FILENAME = 'synced-data.csv' # Name of output data file
PLOTS = True # Generate plots or not
# Plot options
title = 'Moku-synchronization' # Title to use in plots
figsize = (6,4) # Figure size (inches)
dpi = 300 # Pixel density
fontsize = 8 # Font size
linewidth = 1.5 # Linewidth

# : ===== LPSD parameters ====================
p_lpsd = {"olap":"default",
          "bmin":1.0,
          "Lmin":1,
          "Jdes":500,
          "Kdes":100,
          "order":2,
          "win":np.kaiser,
          "psll":250}

# : ===== Main program ====================
def main():
    logger.debug(f"mokusync v{VERSION} - Starting up...")
    parser = argparse.ArgumentParser(description="Liquid Instruments Phasemeter Synchronization: mokusync")
    parser.add_argument('-d', '--debug', action='store_true', help="Enable debug messages")
    parser.add_argument('-s', '--start', type=float, default=0.0, help="Start time in seconds")
    parser.add_argument('-t', '--synctime', type=float, default=0.0, help="Duration in seconds of the data chunk for synchronization")
    parser.add_argument('-T', '--totaltime', type=float, default=0.0, help="Total time in seconds to process from the input files")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Multiprocessing
    pool = multiprocessing.Pool()
    p_lpsd["pool"] = pool

    # The program only looks for files in the current directory
    logger.debug(f"Looking for CSV files in {os.getcwd()}:")
    files_in_directory = os.listdir()
    file_list = sorted([file for file in files_in_directory if (file.endswith('.csv') or file.endswith('.mat'))])

    # Make results directory if it does not exist
    if not os.path.exists(RESDIR):
        os.makedirs(RESDIR)

    # Did we find files?
    if not file_list:
        logger.error("Error: No CSV or MAT files found in the current directory.")
        pool.close()
        pool.join()
        sys.exit(1)
    elif len(file_list) == 1:
        logger.error("Error: Not enough CSV or MAT files found in the current directory.")
        pool.close()
        pool.join()
        sys.exit(1)
    else:
        while True:
            display_menu(file_list)
            action, selected_files = get_two_file_choice(file_list)
            
            if action == 'Q':
                pool.close()
                pool.join()
                logger.debug('Done!')
                sys.exit(0)
            elif action == 'F':
                file1 = selected_files[0]
                file2 = selected_files[1]
                break

        logger.debug('Files selected:')
        logger.debug('    ' + file1 + ' (Master)')
        logger.debug('    ' + file2 + ' (Slave)')

    if file1 == file2:
        logger.error(f"Error: Two different output files are needed")
        pool.close()
        pool.join()
        sys.exit(1)

    start_time = float(args.start)
    synctime = float(args.synctime)
    totaltime = float(args.totaltime)

    if totaltime == 0.0:
        totaltime = None

    logger.debug("Loading data, please wait...")
    mo1 = MokuPhasemeterObject(file1, start_time=start_time, duration=totaltime, prefix='1_', logger=logger)
    fs = mo1.fs
    logger.debug("    * Master device data loaded successfully")

    mo2 = MokuPhasemeterObject(file2, start_time=start_time, duration=totaltime, prefix='2_', logger=logger)
    logger.debug("    * Slave device data loaded successfully")

    # Check: are the files sampled at the same frequency?
    if mo1.fs != mo2.fs:
        logger.error("Error: The input files report different sampling frequencies")
        pool.close()
        pool.join()
        sys.exit(1)

    sig_master = '1_' + MASTER_COL
    sig_slave  = '2_' + SLAVE_COL

    if sig_master not in mo1.df:
        logger.error(f"Error: {MASTER_COL} not found in the Master device data file")
        pool.close()
        pool.join()
        sys.exit(1)

    if sig_slave not in mo2.df:
        logger.error(f"Error: {SLAVE_COL} not found in the Master device data file")
        pool.close()
        pool.join()
        sys.exit(1)

    record_lengths = [(len(mo1.df) / fs), (len(mo2.df) / fs)] # Data stream durations in seconds
    max_totaltime = min(record_lengths) - start_time  # Maximum overlapping time between datasets in seconds

    if (totaltime is None) or (totaltime >= max_totaltime):
        logger.debug(f"Truncating data streams to maximum overlap section of {max_totaltime:.2f} seconds")
        end_row = int(max_totaltime * fs)  # Calculate the end row based on max overlap
        mo1.df = mo1.df.iloc[:end_row].copy()
        mo1.df.reset_index(drop=True, inplace=True)
        mo2.df = mo2.df.iloc[:end_row].copy()
        mo2.df.reset_index(drop=True, inplace=True)
        logger.debug(f"Master stream length: {len(mo1.df)}, Slave stream length: {len(mo2.df)}")

    logger.debug(f"Working on {mo1.duration} seconds of data sampled at {mo1.fs} Hz ({mo1.nrows} points)")

    # Figure out initial time offset from file metadata
    dt_datetime = mo1.date - mo2.date # Slave - Master
    dt_seconds = dt_datetime.total_seconds()
    if dt_seconds < 0:
        logger.debug(f"Time offset from metadata: {dt_seconds:.2f} s, Master device ahead")
    else:
        logger.debug(f"Time offset from metadata: {dt_seconds:.2f} s, Slave device ahead")
    if abs(dt_seconds) > 100.0:
        logger.warning(f"Warning: The initial time offset of {dt_seconds:.2f} seconds is very high!")

    # Assert that the DataFrames are of the same length
    if len(mo1.df) > len(mo2.df):
        logger.error(f"Error: {file2} is shorter than required")
        pool.close()
        pool.join()
        sys.exit(1)
    elif len(mo1.df) < len(mo2.df):
        logger.error(f"Error: {file1} is shorter than required")
        pool.close()
        pool.join()
        sys.exit(1)

    # Find columns with NaNs in both DataFrames
    logger.debug("Looking for NaNs in data...")
    df1_nans = get_columns_with_nans(mo1.df)
    df2_nans = get_columns_with_nans(mo2.df)

    if len(df1_nans) > 0:
        for col, num in df1_nans.items():
            logger.warning(f"Warning: NaNs detected in {file1} column {num} ({col}) ")

    if len(df2_nans) > 0:
        for col, num in df2_nans.items():
            logger.warning(f"Warning: NaNs detected in {file2} column {num} ({col}) ")

    if sig_master in df1_nans:
        logger.error("Error: The specified master device column contains NaNs")
        pool.close()
        pool.join()
        sys.exit(-1)

    if sig_slave in df2_nans:
        logger.error("Error: The specified slave device column contains NaNs")
        pool.close()
        pool.join()
        sys.exit(-1)
    
    # Identify columns without NaNs in both DataFrames
    non_nan_columns_df1 = [col for col in mo1.df.columns if col not in df1_nans]
    non_nan_columns_df2 = [col for col in mo2.df.columns if col not in df2_nans]

    # Subset the DataFrames to only include non-NaN columns
    df1_non_nan = mo1.df[non_nan_columns_df1].copy()
    df2_non_nan = mo2.df[non_nan_columns_df2].copy()

    # Concatenate the DataFrames along columns (axis=1)
    logger.debug("Creating single DataFrame...")
    df = pd.concat([df1_non_nan.reset_index(drop=True), df2_non_nan.reset_index(drop=True)], axis=1)

    file1_filename = mo1.filename
    file2_filename = mo2.filename
    file1_header_rows = mo1.header_rows
    file2_header_rows = mo2.header_rows

    # Free up memory
    del mo1, mo2, df1_non_nan, df2_non_nan

    # When using phase signals for sync, convert them to frequency in Hertz by differentiation
    if 'phase' in sig_master:
        df[sig_master+'_to_freq'] = np.diff(df[sig_master], prepend=np.nan) / (2*np.pi/fs)
        sig_master = sig_master+'_to_freq'
        non_nan_columns_df1.append(sig_master)

    if 'phase' in sig_slave:
        df[sig_slave+'_to_freq'] = np.diff(df[sig_slave], prepend=np.nan) / (2*np.pi/fs)
        sig_slave = sig_slave+'_to_freq'
        non_nan_columns_df2.append(sig_slave)

    df = df.iloc[1:] # Remove first row of the DataFrame, it may contain NaNs from differentiation

    # Calculate and print RMS values of the two signals, useful for debugging
    rms1 = np.sqrt(np.mean(np.square(df[sig_master]-np.mean(df[sig_master]))))
    rms2 = np.sqrt(np.mean(np.square(df[sig_slave]-np.mean(df[sig_slave]))))
    logger.debug(f"    * RMS value of master signal: {rms1:.6}")
    logger.debug(f"    * RMS value of slave signal: {rms2:.6}")

    if (synctime > 0.0) and (synctime >= max_totaltime):
        logger.warning(f"Cannot use {synctime:.2f} seconds for synchronization when the maximum overlap is {max_totaltime:.2f} seconds")

    if (synctime > 0.0) and (synctime < max_totaltime):
        in1 = np.array(df[sig_master].iloc[:int(synctime*fs)])
        in2 = np.array(df[sig_slave].iloc[:int(synctime*fs)])
    else:
        in1 = np.array(df[sig_master])
        in2 = np.array(df[sig_slave])

    logger.debug(f"Calling synctools::sync_signals with inputs of length {len(in1):d}...")
    n_truncate = int(2*abs(dt_seconds*fs))
    unsync, sync = sync_signals(
                        in_signals=[in1, in2],
                        fs=fs, 
                        p_lpsd=p_lpsd, 
                        init_offsets=[dt_seconds], 
                        model=MODEL, 
                        domain=DOMAIN, 
                        method=SOLVER, 
                        interp_order=INTERP_ORDER, 
                        n_truncate=n_truncate,
                        clock_refs=None,
                        pt_list=None,
                        logger=logger)

    dt = sync.timer_offsets[0]

    # Timeshift all signals from slave device
    logger.debug('Generating timeshifted outputs...')
    for sig in non_nan_columns_df2:
        df[sig+'_shifted'] = timeshift(np.array(df[sig]), dt*fs)

    # Truncate output
    df = df.iloc[n_truncate:-n_truncate] 

    # Form unsynced and synced signal combinations
    df['freq_unsynced'] = df[sig_master] - df[sig_slave]
    df['freq_synced'] = df[sig_master] - df[sig_slave+'_shifted']

    # Convert phase in radians to frequency in Hertz via integration
    df['phase_unsynced'] = (2*np.pi/fs)*np.cumsum(np.array(df['freq_unsynced']-np.mean(df['freq_unsynced'])))
    df['phase_synced'] = (2*np.pi/fs)*np.cumsum(np.array(df['freq_synced']-np.mean(df['freq_synced'])))

    # Linear detrend just in case
    df['phase_unsynced'] = df['phase_unsynced'] - np.polyval(np.polyfit(np.arange(len(df)), df['phase_unsynced'], 1), np.arange(len(df)))
    df['phase_synced'] = df['phase_synced'] - np.polyval(np.polyfit(np.arange(len(df)), df['phase_synced'], 1), np.arange(len(df)))

    df.insert(0, 'time', np.arange(len(df))/fs) # Absolute time

    # Save data to file
    logger.debug('Saving data...')

    metadata1 = read_lines(file1_filename, file1_header_rows)
    metadata2 = read_lines(file2_filename, file2_header_rows)

    metadata1[0] += ' (Master)'
    metadata2[0] += ' (Slave)'

    metadata = []
    for line in metadata1:
        metadata.append(line)
    for line in metadata2:
        metadata.append(line)
    metadata.append(f"% Synchronized with moku-sync v{VERSION}, dt = {dt:.14f} seconds ({dt*fs:.14f} samples)")

    output_file_path = os.path.join(RESDIR, FILENAME)

    # Ensure the directory exists
    os.makedirs(RESDIR, exist_ok=True)

    with open(output_file_path, 'w') as file:
        for line in metadata:
            file.write(line + '\n')

    with open(output_file_path, 'a') as file:
        df.to_csv(file, index=False)
    
    if PLOTS:
        # Time domain plots
        logger.debug('Plotting time series data...')
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(df['time'], df['freq_unsynced'], linewidth=linewidth, label=r"Unsynced", color="gray")
        ax.plot(df['time'], df['freq_synced'], linewidth=linewidth, label=r"Synced", color="tomato")
        ax.set_xlabel("Time (s)", fontsize=fontsize)
        ax.set_ylabel("Frequency (Hz)", fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid()
        ax.legend(loc='best', edgecolor='black', fancybox=True, shadow=True, framealpha=1, fontsize=fontsize, handlelength=2.5)
        fig.tight_layout()
        fig.savefig(os.path.join(RESDIR,'fig_freq_t.pdf'))
        logger.debug(f"    * Plot saved to {os.path.join(RESDIR,'fig_freq_t.pdf')}")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(df['time'], df['phase_unsynced'], linewidth=linewidth, label=r"Unsynced", color="gray")
        ax.plot(df['time'], df['phase_synced'], linewidth=linewidth, label=r"Synced", color="tomato")
        ax.set_xlabel("Time (s)", fontsize=fontsize)
        ax.set_ylabel("Phase (rad)", fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid()
        ax.legend(loc='best', edgecolor='black', fancybox=True, shadow=True, framealpha=1, fontsize=fontsize, handlelength=2.5)
        fig.tight_layout()
        fig.savefig(os.path.join(RESDIR, 'fig_phase_t.pdf'))
        logger.debug(f"    * Plot saved to {os.path.join(RESDIR, 'fig_phase_t.pdf')}")

        # Compute spectrums with spectools
        logger.debug('Computing unsynced frequency spectrum...')
        psd_unsynced = lpsd.lpsd(np.array(df['freq_unsynced']), fs=fs,
            olap=p_lpsd['olap'], bmin=p_lpsd['bmin'], Lmin=p_lpsd['Lmin'], 
            Jdes=p_lpsd['Jdes'], Kdes=p_lpsd['Kdes'],
            order=p_lpsd['order'], win=p_lpsd['win'], psll=p_lpsd['psll'],
            return_type='object', pool=pool)
        
        logger.debug('Computing synced frequency spectrum...')
        psd_synced = lpsd.lpsd(np.array(df['freq_synced']), fs=fs,
            olap=p_lpsd['olap'], bmin=p_lpsd['bmin'], Lmin=p_lpsd['Lmin'], 
            Jdes=p_lpsd['Jdes'], Kdes=p_lpsd['Kdes'],
            order=p_lpsd['order'], win=p_lpsd['win'], psll=p_lpsd['psll'],
            return_type='object', pool=pool)
        logger.debug('Plotting spectrums...')
    
        # ASD plots
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.loglog(psd_unsynced.f, np.sqrt(psd_unsynced.Gxx), linewidth=linewidth, label=r"Unsynced", color="gray")
        ax.loglog(psd_synced.f, np.sqrt(psd_synced.Gxx), linewidth=linewidth, label=r"Synced", color="tomato")
        ax.set_xlim(psd_synced.f[0], psd_synced.f[-1])
        ax.set_xlabel("Fourier frequency (Hz)", fontsize=fontsize)
        ax.set_ylabel(r"Frequency ASD $\rm (Hz/Hz^{1/2})$", fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(which='both')
        ax.legend(loc='best', edgecolor='black', fancybox=True, shadow=True, framealpha=1, fontsize=fontsize, handlelength=2.5)
        fig.tight_layout()
        fig.savefig(os.path.join(RESDIR, 'fig_freq_asd.pdf'));
        logger.debug(f"    * Plot saved to {os.path.join(RESDIR, 'fig_freq_asd.pdf')}")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.loglog(psd_unsynced.f, np.sqrt(psd_unsynced.Gxx)/psd_unsynced.f, linewidth=linewidth, label=r"Unsynced", color="gray")
        ax.loglog(psd_synced.f, np.sqrt(psd_synced.Gxx)/psd_synced.f, linewidth=linewidth, label=r"Synced", color="tomato")
        ax.set_xlim(psd_synced.f[0], psd_synced.f[-1])
        ax.set_xlabel("Fourier frequency (Hz)", fontsize=fontsize)
        ax.set_ylabel(r"Phase ASD $\rm (rad/Hz^{1/2})$", fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(which='both')
        ax.legend(loc='best', edgecolor='black', fancybox=True, shadow=True, framealpha=1, fontsize=fontsize, handlelength=2.5)
        fig.tight_layout()
        fig.savefig(os.path.join(RESDIR, 'fig_phase_asd.pdf'))
        logger.debug(f"    * Plot saved to {os.path.join(RESDIR, 'fig_phase_asd.pdf')}")

    pool.close()
    pool.join()
    logger.debug('Done!')




if __name__ == "__main__":
    main()