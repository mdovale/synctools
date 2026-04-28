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
"""Example: synchronize two Liquid Instruments Moku phasemeter streams.

The two Moku instruments share a clock, but their data streams are misaligned
by a non-integer number of samples. This script prepares Moku data and calls
synctools for offset recovery.
"""
from __future__ import annotations

VERSION = "1.05"

import argparse
import logging
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import List, NoReturn, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mokutools.moku_io.cli import pick_two_files, print_menu
from mokutools.moku_io.core import get_columns_with_nans
from mokutools.phasemeter import MokuPhasemeterObject
from pytdi.dsp import timeshift
from synctools import (
    convert_frequency_to_detrended_phase_in_time,
    convert_phase_to_frequency_in_time,
    sync_signals,
)
from synctools.auxiliary import convert_frequency_to_phase_in_asd, spectra

logger = logging.getLogger(__name__)

# Column token after 1_ prefix, e.g. 1_freq, 1_phase, 2_freq, ...
_DEFAULT_MASTER_COL = "1_freq"
_DATA_EXTENSIONS = (".csv", ".csv.zip", ".mat")


class MokuSyncError(RuntimeError):
    """Recoverable run-time error reported cleanly by the CLI."""


class UserCancelled(Exception):
    """Raised when interactive file selection is cancelled."""


@dataclass(frozen=True)
class PlotSettings:
    title: str
    figsize: Tuple[float, float]
    dpi: int
    fontsize: float
    linewidth: float
    enabled: bool


@dataclass(frozen=True)
class LoadedStreams:
    master_df: pd.DataFrame
    slave_df: pd.DataFrame
    fs: float
    metadata_offset: float
    duration_seconds: float
    master_header: List[str]
    slave_header: List[str]


@dataclass(frozen=True)
class PreparedSyncData:
    df: pd.DataFrame
    master_sync_col: str
    slave_sync_col: str
    slave_columns_to_shift: List[str]
    duration_seconds: float


def _resolve_in_data_dir(data_dir: str, name: str) -> str:
    if os.path.isabs(name):
        return os.path.abspath(name)
    return os.path.abspath(os.path.join(data_dir, name))


def build_parser() -> argparse.ArgumentParser:
    epilog = textwrap.dedent("""\
        examples:
          Interactive file selection, files under ./data:
            %(prog)s --data-dir ./data

          Non-interactive, explicit master and slave (paths relative to --data-dir
          unless you pass absolute paths):
            %(prog)s -C /path/to/sets --master run_a.csv --slave run_b.csv

          Use only 30 s to estimate sync, skip figure generation, custom result folder:
            %(prog)s -C . --master a.csv --slave b.csv -t 30 --no-plots -o /tmp/moku-out

          Explicit initial offset guess (seconds) instead of file metadata:
            %(prog)s -C . --master a.csv --slave b.csv --init-offset -1.25e-3

        Column names: each device uses prefix 1_ or 2_ in the file; --master-col and
        --slave-col are the part after the prefix, e.g. 1_freq -> column 1_1_freq
        in the Moku data frame; use 1_freq for a channel-1 frequency column, etc.
    """)
    p = argparse.ArgumentParser(
        prog="mokusync",
        description="Synchronize two Moku phasemeter streams (see synctools sync_signals) "
        "and write combined CSV, optional PDF diagnostics.",
        epilog=epilog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s (mokutools) {VERSION}",
    )

    # --- Data locations & selection ---
    g_in = p.add_argument_group("input (data & file selection)")
    g_in.add_argument(
        "-C", "--data-dir",
        type=str,
        default=".",
        metavar="DIR",
        help="Directory containing Moku .csv (or .mat) files. Used for listing, "
        "and as the base path for --master / --slave when those are not absolute.",
    )
    g_in.add_argument(
        "--master",
        metavar="FILE",
        help="Master stream file (basename in --data-dir, or absolute). "
        "If set, --slave is required; skips interactive file menu.",
    )
    g_in.add_argument(
        "--slave",
        metavar="FILE",
        help="Slave stream file (same rules as --master).",
    )
    g_in.add_argument(
        "--master-col",
        type=str,
        default=_DEFAULT_MASTER_COL,
        help="Column token used with master 1_ prefix, e.g. 1_freq, 2_phase, 3_freq, 4_freq.",
    )
    g_in.add_argument(
        "--slave-col",
        type=str,
        default=_DEFAULT_MASTER_COL,
        help="Column token used with slave 2_ prefix, same form as --master-col.",
    )

    # --- Windows & timing ---
    g_time = p.add_argument_group("time range (seconds, relative to loaded files)")
    g_time.add_argument(
        "-s", "--start", type=float, default=0.0,
        help="Start time to load from each file (seconds).",
    )
    g_time.add_argument(
        "-t", "--synctime", type=float, default=0.0,
        metavar="T",
        help="If > 0, use only the first T seconds to estimate the offset (faster, "
        "must be less than the overlap of both streams). If 0, use full length.",
    )
    g_time.add_argument(
        "-T", "--totaltime", type=float, default=0.0,
        help="If > 0, load at most this many seconds from the start. If 0, load the "
        "maximum available subject to file limits.",
    )

    # --- synctools ---
    g_sync = p.add_argument_group("synchronization (synctools.sync_signals)")
    g_sync.add_argument(
        "--model", choices=("fluc", "total"), default="fluc",
        help="TDIR model: fluctuation (fluc) or total (includes mean drift).",
    )
    g_sync.add_argument(
        "--domain", choices=("time", "freq"), default="freq",
        help="RMS / optimization domain: time or frequency.",
    )
    g_sync.add_argument(
        "--method", default="Nelder-Mead", metavar="NAME",
        help="scipy.optimize.minimize method, e.g. Nelder-Mead, Powell, L-BFGS-B, TNC.",
    )
    g_sync.add_argument(
        "--interp-order", type=int, default=121,
        help="Interpolation order for time shifting (e.g. 5-121).",
    )
    g_sync.add_argument(
        "--n-truncate", type=int, default=None, metavar="N",
        help="Override edge truncation in samples (two-sided). If omitted, uses "
        "2*|metadata offset|*fs, consistent with the library init-offset scaling.",
    )
    g_init = g_sync.add_mutually_exclusive_group()
    g_init.add_argument(
        "--no-init-offset", action="store_true",
        help="Start optimization from 0 s instead of the metadata (file timestamp) offset.",
    )
    g_init.add_argument(
        "--init-offset",
        type=float,
        metavar="SEC",
        default=None,
        help="Initial time-offset guess for sync optimization (seconds). "
        "If omitted (and not --no-init-offset), uses file timestamp metadata.",
    )

    # --- Output ---
    g_out = p.add_argument_group("output")
    g_out.add_argument(
        "-o", "--output-dir", type=str, default="./results", metavar="DIR",
        help="Directory for the CSV and any PDFs.",
    )
    g_out.add_argument(
        "--output-file", type=str, default="synced-data.csv", metavar="NAME",
        help="Name of the output combined CSV in --output-dir.",
    )

    # --- Plots ---
    g_plot = p.add_argument_group("plotting (matplotlib, optional PDFs)")
    g_plot.add_argument(
        "--no-plots", action="store_true",
        help="Do not create time-domain or spectrum figures.",
    )
    g_plot.add_argument(
        "--plot-title", type=str, default="Moku-synchronization",
        help="Figure title.",
    )
    g_plot.add_argument(
        "--figsize", nargs=2, type=float, default=[6, 4], metavar=("W", "H"),
        help="Figure width and height in inches.",
    )
    g_plot.add_argument("--dpi", type=int, default=300, help="Figure resolution (dots per inch).")
    g_plot.add_argument("--fontsize", type=float, default=8, help="Axis and legend font size.")
    g_plot.add_argument(
        "--linewidth", type=float, default=1.5,
        help="Line width in spectrum and time series plots.",
    )

    # --- speckit / LPSD ---
    g_lpsd = p.add_argument_group("LPSD (speckit / spectrum estimation)")
    g_lpsd.add_argument(
        "--lpsd-olap", type=str, default="default",
        help="Overlap control string passed to speckit (see speckit docs).",
    )
    g_lpsd.add_argument("--lpsd-bmin", type=float, default=1.0, help="Segment bandwidth (Hz).")
    g_lpsd.add_argument("--lpsd-lmin", type=int, default=1, help="Minimum segment length.")
    g_lpsd.add_argument(
        "--lpsd-jdes", type=int, default=500,
        help="Desired number of frequency bins (segment count parameter).",
    )
    g_lpsd.add_argument(
        "--lpsd-kdes", type=int, default=100,
        help="Desired number of periodogram averages (parameter K).",
    )
    g_lpsd.add_argument(
        "--lpsd-order", type=int, default=2, help="Differentiation or synthesis order in LPSD.",
    )
    g_lpsd.add_argument(
        "--lpsd-psll", type=int, default=250, help="Kaiser side-lobe suppression (dB).",
    )

    g_run = p.add_argument_group("run time")
    g_run.add_argument(
        "-d", "--debug", action="store_true",
        help="Verbose debug logging to stderr.",
    )
    g_run.add_argument(
        "-q", "--quiet", action="store_true",
        help="Only errors and critical messages.",
    )

    return p


def configure_logging(args: argparse.Namespace) -> None:
    if args.debug:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def parse_args(
    parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None
) -> Tuple[argparse.Namespace, str]:
    args = parser.parse_args(argv)
    if bool(args.master) != bool(args.slave):
        parser.error(
            "use --master and --slave together, or omit both for interactive file selection"
        )
    if args.debug and args.quiet:
        parser.error("--debug and --quiet are mutually exclusive")
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    if not os.path.isdir(data_dir):
        parser.error(f"not a directory: {data_dir}")
    return args, data_dir


def _fail(message: str) -> NoReturn:
    raise MokuSyncError(message)


def _interactive_pick_two(
    file_list: List[str],
) -> Tuple[str, str]:
    while True:
        print_menu(file_list)
        action, selected_files = pick_two_files(file_list)
        if action == "Q":
            raise UserCancelled()
        if action == "F" and selected_files is not None:
            return selected_files[0], selected_files[1]
    raise RuntimeError("unreachable")


def _validate_runtime_options(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    if any(x <= 0 for x in args.figsize):
        parser.error("--figsize W and H must be positive")
    for label, value in [
        ("--lpsd-bmin", args.lpsd_bmin),
        ("--lpsd-lmin", args.lpsd_lmin),
        ("--lpsd-jdes", args.lpsd_jdes),
        ("--lpsd-kdes", args.lpsd_kdes),
        ("--lpsd-order", args.lpsd_order),
        ("--lpsd-psll", args.lpsd_psll),
    ]:
        if value <= 0:
            parser.error(f"{label} must be positive, got {value!r}")
    if args.n_truncate is not None and args.n_truncate < 0:
        parser.error("--n-truncate must be non-negative")


def _build_lpsd_params(args: argparse.Namespace) -> dict:
    return {
        "olap": args.lpsd_olap,
        "bmin": args.lpsd_bmin,
        "Lmin": args.lpsd_lmin,
        "Jdes": args.lpsd_jdes,
        "Kdes": args.lpsd_kdes,
        "order": args.lpsd_order,
        "win": np.kaiser,
        "psll": args.lpsd_psll,
    }


def _build_plot_settings(args: argparse.Namespace) -> PlotSettings:
    return PlotSettings(
        title=str(args.plot_title),
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        dpi=int(args.dpi),
        fontsize=float(args.fontsize),
        linewidth=float(args.linewidth),
        enabled=not bool(args.no_plots),
    )


def _list_data_files(data_dir: str) -> List[str]:
    return sorted(
        entry.name
        for entry in os.scandir(data_dir)
        if entry.is_file() and entry.name.endswith(_DATA_EXTENSIONS)
    )


def _select_input_files(args: argparse.Namespace, data_dir: str) -> Tuple[str, str]:
    if args.master and args.slave:
        file1, file2 = str(args.master), str(args.slave)
        logger.info("Non-interactive: master %s, slave %s", file1, file2)
        return file1, file2

    logger.info("Scanning for Moku data files under: %s", data_dir)
    file_list = _list_data_files(data_dir)
    if not file_list:
        _fail("No CSV, CSV.ZIP, or MAT files in the data directory.")
    if len(file_list) < 2:
        _fail("Need at least two data files, or use --master and --slave with paths.")
    return _interactive_pick_two(file_list)


def _resolve_input_paths(data_dir: str, file1: str, file2: str) -> Tuple[str, str]:
    if file1 == file2:
        _fail("Master and slave must be two different file names/paths.")

    path1 = _resolve_in_data_dir(data_dir, file1)
    path2 = _resolve_in_data_dir(data_dir, file2)
    if os.path.abspath(path1) == os.path.abspath(path2):
        _fail("Master and slave resolve to the same file path.")

    for label, path in (("master", path1), ("slave", path2)):
        if not os.path.isfile(path):
            _fail(f"Not a file ({label}): {path}")
    return path1, path2


def _load_streams(
    path1: str,
    path2: str,
    start_time: float,
    totaltime: Optional[float],
) -> LoadedStreams:
    logger.info("Loading master: %s", path1)
    mo1 = MokuPhasemeterObject(
        path1, start_time=start_time, duration=totaltime, prefix="1_", logger=logger
    )
    fs = float(mo1.fs)

    logger.info("Loading slave: %s", path2)
    mo2 = MokuPhasemeterObject(
        path2, start_time=start_time, duration=totaltime, prefix="2_", logger=logger
    )

    if float(mo1.fs) != float(mo2.fs):
        _fail("The two input files do not have the same sampling rate.")

    common_rows = min(len(mo1.df), len(mo2.df))
    if common_rows < 2:
        _fail("The loaded overlap is too short; need at least two rows.")

    if len(mo1.df) != len(mo2.df):
        logger.warning(
            "Loaded streams have different lengths; trimming to common overlap of %d rows.",
            common_rows,
        )

    duration_seconds = common_rows / fs
    logger.info(
        "Using common overlap: %.2f s (%d rows) at fs = %.3f Hz",
        duration_seconds,
        common_rows,
        fs,
    )

    metadata_offset = float((mo2.date - mo1.date).total_seconds())
    logger.info(
        "Metadata offset (slave start - master start): %.6e s (%.6e samples at fs).",
        metadata_offset,
        metadata_offset * fs,
    )
    if abs(metadata_offset) > 100.0:
        logger.warning(
            "File timestamp gap is very large (%.0f s). Check recording metadata.",
            abs(metadata_offset),
        )

    master_header = [line.strip() for line in mo1.header[: int(mo1.header_rows)]]
    slave_header = [line.strip() for line in mo2.header[: int(mo2.header_rows)]]

    return LoadedStreams(
        master_df=mo1.df.iloc[:common_rows].reset_index(drop=True).copy(),
        slave_df=mo2.df.iloc[:common_rows].reset_index(drop=True).copy(),
        fs=fs,
        metadata_offset=metadata_offset,
        duration_seconds=duration_seconds,
        master_header=master_header,
        slave_header=slave_header,
    )


def _sync_column(prefix: str, token: str) -> str:
    return prefix + str(token)


def _warn_about_nans(label: str, filename: str, nan_columns: dict) -> None:
    for col, column_index in nan_columns.items():
        logger.warning("NaNs in %s %r column index %r: %r", label, filename, column_index, col)


def _append_frequency_column_if_phase(
    df: pd.DataFrame,
    sync_col: str,
    fs: float,
) -> Tuple[str, bool]:
    if "phase" not in sync_col.lower():
        return sync_col, False

    freq_col = sync_col + "_to_freq"
    df[freq_col] = convert_phase_to_frequency_in_time(
        np.asarray(df[sync_col], dtype=np.float64),
        fs,
    )
    return freq_col, True


def _prepare_sync_data(
    streams: LoadedStreams,
    file1: str,
    file2: str,
    master_col_token: str,
    slave_col_token: str,
) -> PreparedSyncData:
    master_sync_col = _sync_column("1_", master_col_token)
    slave_sync_col = _sync_column("2_", slave_col_token)

    if master_sync_col not in streams.master_df:
        _fail(f"Column {master_sync_col!r} not in master. Try --master-col (e.g. 1_freq).")
    if slave_sync_col not in streams.slave_df:
        _fail(f"Column {slave_sync_col!r} not in slave. Try --slave-col (e.g. 1_freq).")

    logger.info("Merging and checking NaNs ...")
    master_nans = get_columns_with_nans(streams.master_df)
    slave_nans = get_columns_with_nans(streams.slave_df)
    _warn_about_nans("master", file1, master_nans)
    _warn_about_nans("slave", file2, slave_nans)

    if master_sync_col in master_nans:
        _fail("The master sync column contains NaNs. Choose a different --master-col.")
    if slave_sync_col in slave_nans:
        _fail("The slave sync column contains NaNs. Choose a different --slave-col.")

    master_columns = [col for col in streams.master_df.columns if col not in master_nans]
    slave_columns = [col for col in streams.slave_df.columns if col not in slave_nans]
    df = pd.concat(
        [
            streams.master_df[master_columns].reset_index(drop=True),
            streams.slave_df[slave_columns].reset_index(drop=True),
        ],
        axis=1,
    )

    master_sync_col, dropped_first = _append_frequency_column_if_phase(
        df,
        master_sync_col,
        streams.fs,
    )
    slave_sync_col, dropped_first_slave = _append_frequency_column_if_phase(
        df,
        slave_sync_col,
        streams.fs,
    )
    dropped_first = dropped_first or dropped_first_slave
    if master_sync_col not in master_columns:
        master_columns.append(master_sync_col)
    if slave_sync_col not in slave_columns:
        slave_columns.append(slave_sync_col)

    if dropped_first:
        df = df.iloc[1:].reset_index(drop=True).copy()
    if len(df) < 2:
        _fail("Prepared data is too short after phase-to-frequency conversion.")

    return PreparedSyncData(
        df=df,
        master_sync_col=master_sync_col,
        slave_sync_col=slave_sync_col,
        slave_columns_to_shift=slave_columns,
        duration_seconds=len(df) / streams.fs,
    )


def _log_sync_channel_rms(prepared: PreparedSyncData) -> None:
    master = np.asarray(prepared.df[prepared.master_sync_col], dtype=np.float64)
    slave = np.asarray(prepared.df[prepared.slave_sync_col], dtype=np.float64)
    master_rms = float(np.sqrt(np.mean(np.square(master - np.mean(master)))))
    slave_rms = float(np.sqrt(np.mean(np.square(slave - np.mean(slave)))))
    logger.debug(
        "RMS (sync ch.) master: %.5g, slave: %.5g (same units as input)",
        master_rms,
        slave_rms,
    )


def _select_estimation_signals(
    prepared: PreparedSyncData,
    fs: float,
    synctime: float,
) -> Tuple[np.ndarray, np.ndarray]:
    nrows = len(prepared.df)
    if 0.0 < synctime < prepared.duration_seconds:
        nseg = int(synctime * fs)
        if nseg < 2:
            _fail("--synctime selects fewer than two samples; increase it or use the full record.")
        logger.info("Using first %.3f s (%d samples) to estimate the offset.", synctime, nseg)
        return (
            np.asarray(prepared.df[prepared.master_sync_col].iloc[:nseg], dtype=np.float64),
            np.asarray(prepared.df[prepared.slave_sync_col].iloc[:nseg], dtype=np.float64),
        )

    if 0.0 < synctime:
        logger.warning(
            "synctime %.2f s is not < overlap %.2f s; using the full %d rows.",
            synctime,
            prepared.duration_seconds,
            nrows,
        )
    logger.info("Using the full %d rows for offset estimation.", nrows)
    return (
        np.asarray(prepared.df[prepared.master_sync_col], dtype=np.float64),
        np.asarray(prepared.df[prepared.slave_sync_col], dtype=np.float64),
    )


def _resolve_n_truncate(
    requested_n_truncate: Optional[int],
    metadata_offset: float,
    fs: float,
    nrows: int,
) -> Tuple[int, int]:
    n_auto = int(2.0 * abs(metadata_offset * fs))
    n_truncate = n_auto if requested_n_truncate is None else int(requested_n_truncate)
    if n_truncate * 2 >= nrows:
        _fail(
            f"n_truncate={n_truncate} is too large for the merged length {nrows}. "
            "Lower --n-truncate or use more data."
        )
    return n_auto, n_truncate


def _estimate_timer_offset(
    in1: np.ndarray,
    in2: np.ndarray,
    fs: float,
    p_lpsd: dict,
    init_offset: float,
    n_truncate: int,
    args: argparse.Namespace,
) -> float:
    logger.info(
        "sync_signals: model=%s, domain=%s, method=%s, interp=%d, "
        "n_truncate=%d, init=%.3e s",
        args.model,
        args.domain,
        args.method,
        args.interp_order,
        n_truncate,
        init_offset,
    )

    try:
        _unsynced, sync = sync_signals(
            in_signals=[in1, in2],
            fs=fs,
            p_lpsd=p_lpsd,
            init_offsets=[init_offset],
            model=str(args.model),
            domain=str(args.domain),
            method=str(args.method),
            interp_order=int(args.interp_order),
            n_truncate=n_truncate,
            clock_refs=None,
            logger=logger,
        )
    except ValueError as exc:
        raise MokuSyncError(f"sync_signals failed: {exc}") from exc
    del _unsynced

    offset = float(sync.timer_offsets[0])
    logger.info(
        "Resulting offset: %.6e s (%.6e samples at fs); correction from init: %.6e s.",
        offset,
        offset * fs,
        offset - init_offset,
    )
    return offset


def _shift_slave_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    offset_seconds: float,
    fs: float,
    interp_order: int,
) -> None:
    sample_shift = offset_seconds * fs
    for name in columns:
        values = np.asarray(df[name], dtype=np.float64)
        df[name + "_shifted"] = timeshift(values, sample_shift, order=interp_order)


def _trim_edges(df: pd.DataFrame, n_truncate: int) -> pd.DataFrame:
    if n_truncate == 0:
        return df.reset_index(drop=True).copy()
    return df.iloc[n_truncate:-n_truncate].reset_index(drop=True).copy()


def _build_output_dataframe(
    prepared: PreparedSyncData,
    fs: float,
    n_truncate: int,
) -> pd.DataFrame:
    df = _trim_edges(prepared.df, n_truncate)
    if len(df) < 1:
        _fail("No rows remain after edge truncation.")

    df["freq_unsynced"] = df[prepared.master_sync_col] - df[prepared.slave_sync_col]
    df["freq_synced"] = df[prepared.master_sync_col] - df[prepared.slave_sync_col + "_shifted"]
    df["phase_unsynced"] = convert_frequency_to_detrended_phase_in_time(
        np.asarray(df["freq_unsynced"], dtype=np.float64),
        fs,
    )
    df["phase_synced"] = convert_frequency_to_detrended_phase_in_time(
        np.asarray(df["freq_synced"], dtype=np.float64),
        fs,
    )
    df.insert(0, "time", np.arange(len(df), dtype=np.float64) / float(fs))
    return df


def _output_path(output_dir: str, output_file: str) -> str:
    resdir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(resdir, exist_ok=True)
    out = str(output_file)
    path = out if os.path.isabs(out) else os.path.join(resdir, out)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def _metadata_lines(
    master_header: Sequence[str],
    slave_header: Sequence[str],
    offset_seconds: float,
    fs: float,
    data_dir: str,
    file1: str,
    file2: str,
    args: argparse.Namespace,
) -> List[str]:
    meta1 = list(master_header)
    meta2 = list(slave_header)
    if meta1 and isinstance(meta1[0], str) and not meta1[0].rstrip().endswith("(Master)"):
        meta1[0] = meta1[0] + " (Master)"
    if meta2 and isinstance(meta2[0], str) and not meta2[0].rstrip().endswith("(Slave)"):
        meta2[0] = meta2[0] + " (Slave)"

    return meta1 + meta2 + [
        f"% moku-sync v{VERSION} -- dt = {offset_seconds:.14f} s, "
        f"{offset_seconds * fs:.14f} samples @ fs; data-dir={data_dir!r} "
        f"master={file1!r} slave={file2!r} model={args.model} domain={args.domain}"
    ]


def _write_output_csv(
    output_path: str,
    df: pd.DataFrame,
    header_lines: Sequence[str],
) -> None:
    logger.info("Writing %s", output_path)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        for line in header_lines:
            fh.write(line if line.endswith("\n") else line + "\n")
    with open(output_path, "a", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False)


def _plot_pdfs(
    resdir: str,
    df: pd.DataFrame,
    fs: float,
    p_lpsd: dict,
    title: str,
    figsize: Tuple[float, float],
    dpi: int,
    fontsize: float,
    linewidth: float,
) -> None:
    # Time series
    logger.debug("Plotting time series...")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(df["time"], df["freq_unsynced"], linewidth=linewidth, label=r"Unsynced", color="gray")
    ax.plot(df["time"], df["freq_synced"], linewidth=linewidth, label=r"Synced", color="tomato")
    ax.set_xlabel("Time (s)", fontsize=fontsize)
    ax.set_ylabel("Frequency (Hz)", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid()
    ax.legend(
        loc="best",
        edgecolor="black",
        fancybox=True,
        shadow=True,
        framealpha=1.0,
        fontsize=fontsize,
        handlelength=2.5,
    )
    fig.tight_layout()
    fpath = os.path.join(resdir, "fig_freq_t.pdf")
    fig.savefig(fpath)
    logger.info("Wrote %s", fpath)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(df["time"], df["phase_unsynced"], linewidth=linewidth, label=r"Unsynced", color="gray")
    ax.plot(df["time"], df["phase_synced"], linewidth=linewidth, label=r"Synced", color="tomato")
    ax.set_xlabel("Time (s)", fontsize=fontsize)
    ax.set_ylabel("Phase (rad)", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid()
    ax.legend(
        loc="best",
        edgecolor="black",
        fancybox=True,
        shadow=True,
        framealpha=1.0,
        fontsize=fontsize,
        handlelength=2.5,
    )
    fig.tight_layout()
    fpath = os.path.join(resdir, "fig_phase_t.pdf")
    fig.savefig(fpath)
    logger.info("Wrote %s", fpath)

    logger.debug("Computing frequency spectra...")
    freq_unsynced, asd_unsynced = spectra(
        np.asarray(df["freq_unsynced"], dtype=np.float64),
        fs,
        p_lpsd,
    )
    freq_synced, asd_synced = spectra(
        np.asarray(df["freq_synced"], dtype=np.float64),
        fs,
        p_lpsd,
    )
    if len(freq_unsynced) == 0 or len(freq_synced) == 0:
        logger.warning("Skipping ASD plots because spectrum estimation returned no data.")
        plt.close("all")
        return

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.loglog(freq_unsynced, asd_unsynced, linewidth=linewidth, label=r"Unsynced", color="gray")
    ax.loglog(freq_synced, asd_synced, linewidth=linewidth, label=r"Synced", color="tomato")
    ax.set_xlim(freq_synced[0], freq_synced[-1])
    ax.set_xlabel("Fourier frequency (Hz)", fontsize=fontsize)
    ax.set_ylabel(r"Frequency ASD $\rm (Hz/Hz^{1/2})$", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(which="both")
    ax.legend(
        loc="best",
        edgecolor="black",
        fancybox=True,
        shadow=True,
        framealpha=1.0,
        fontsize=fontsize,
        handlelength=2.5,
    )
    fig.tight_layout()
    fpath = os.path.join(resdir, "fig_freq_asd.pdf")
    fig.savefig(fpath)
    logger.info("Wrote %s", fpath)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.loglog(
        freq_unsynced,
        convert_frequency_to_phase_in_asd(freq_unsynced, asd_unsynced),
        linewidth=linewidth,
        label=r"Unsynced",
        color="gray",
    )
    ax.loglog(
        freq_synced,
        convert_frequency_to_phase_in_asd(freq_synced, asd_synced),
        linewidth=linewidth,
        label=r"Synced",
        color="tomato",
    )
    ax.set_xlim(freq_synced[0], freq_synced[-1])
    ax.set_xlabel("Fourier frequency (Hz)", fontsize=fontsize)
    ax.set_ylabel(r"Phase ASD $\rm (rad/Hz^{1/2})$", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(which="both")
    ax.legend(
        loc="best",
        edgecolor="black",
        fancybox=True,
        shadow=True,
        framealpha=1.0,
        fontsize=fontsize,
        handlelength=2.5,
    )
    fig.tight_layout()
    fpath = os.path.join(resdir, "fig_phase_asd.pdf")
    fig.savefig(fpath)
    logger.info("Wrote %s", fpath)
    plt.close("all")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args, data_dir = parse_args(parser, argv)
    configure_logging(args)
    _validate_runtime_options(args, parser)

    logger.info("mokutools moku-sync v%s | data directory: %s", VERSION, data_dir)

    try:
        p_lpsd = _build_lpsd_params(args)
        plot_settings = _build_plot_settings(args)
        output_path = _output_path(args.output_dir, args.output_file)
        resdir = os.path.abspath(os.path.expanduser(args.output_dir))

        file1, file2 = _select_input_files(args, data_dir)
        path1, path2 = _resolve_input_paths(data_dir, file1, file2)
        totaltime = None if float(args.totaltime) == 0.0 else float(args.totaltime)
        streams = _load_streams(path1, path2, float(args.start), totaltime)

        prepared = _prepare_sync_data(
            streams,
            file1,
            file2,
            str(args.master_col),
            str(args.slave_col),
        )
        _log_sync_channel_rms(prepared)

        in1, in2 = _select_estimation_signals(prepared, streams.fs, float(args.synctime))
        n_auto, n_truncate = _resolve_n_truncate(
            args.n_truncate,
            streams.metadata_offset,
            streams.fs,
            len(prepared.df),
        )
        logger.info("Edge truncation: n_truncate=%d (auto was %d)", n_truncate, n_auto)

        if args.init_offset is not None:
            init_offset = float(args.init_offset)
        elif args.no_init_offset:
            init_offset = 0.0
        else:
            init_offset = streams.metadata_offset
        offset = _estimate_timer_offset(
            in1,
            in2,
            streams.fs,
            p_lpsd,
            init_offset,
            n_truncate,
            args,
        )

        _shift_slave_columns(
            prepared.df,
            prepared.slave_columns_to_shift,
            offset,
            streams.fs,
            int(args.interp_order),
        )
        df = _build_output_dataframe(prepared, streams.fs, n_truncate)
        header_lines = _metadata_lines(
            streams.master_header,
            streams.slave_header,
            offset,
            streams.fs,
            data_dir,
            file1,
            file2,
            args,
        )
        _write_output_csv(output_path, df, header_lines)

        if plot_settings.enabled:
            _plot_pdfs(
                resdir,
                df,
                streams.fs,
                p_lpsd,
                plot_settings.title,
                plot_settings.figsize,
                plot_settings.dpi,
                plot_settings.fontsize,
                plot_settings.linewidth,
            )
        else:
            logger.info("Skipping plots (--no-plots).")

        logger.info("Finished. Primary output: %s", output_path)
        return 0
    except UserCancelled:
        logger.info("Cancelled.")
        return 0
    except MokuSyncError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)