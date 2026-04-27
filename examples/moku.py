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
"""
mokutools.sync: Synchronization of two Liquid Instruments Moku phasemeter data streams

The two Moku share a clock, but their data streams are
misaligned by a non-integer number of samples
"""
from __future__ import annotations

VERSION = "1.04"

from mokutools.filetools import *
from mokutools.phasemeter import MokuPhasemeterObject
import os
import sys
import argparse
import textwrap
from typing import List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytdi.dsp import timeshift
from speckit import compute_spectrum as lpsd
import multiprocessing
from synctools import sync_signals
import logging
logger = logging.getLogger(__name__)

_DEFAULT_MASTER_COL = "1_freq"  # column token after 1_ prefix, e.g. 1_freq, 1_phase, 2_freq, ...


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
    g_sync.add_argument(
        "--no-init-offset", action="store_true",
        help="Start optimization from 0 s instead of the metadata (file timestamp) offset.",
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

    # --- Pool & logging ---
    g_run = p.add_argument_group("run time")
    g_run.add_argument(
        "-j", "--processes", type=int, default=None, metavar="N",
        help="Number of worker processes in the speckit Pool (default: all CPUs).",
    )
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
        parser.error("use --master and --slave together, or omit both for interactive file selection")
    if args.debug and args.quiet:
        parser.error("--debug and --quiet are mutually exclusive")
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    if not os.path.isdir(data_dir):
        parser.error(f"not a directory: {data_dir}")
    return args, data_dir


def _fail(message: str, pool: object) -> "None":  # noqa: UP037
    logger.error("%s", message)
    pool.close()
    pool.join()
    sys.exit(1)


def _interactive_pick_two(
    _data_dir: str, file_list: List[str], pool: object
) -> Tuple[str, str]:
    while True:
        display_menu(file_list)
        action, selected_files = get_two_file_choice(file_list)
        if action == "Q":
            pool.close()
            pool.join()
            sys.exit(0)
        if action == "F":
            return selected_files[0], selected_files[1]
    raise RuntimeError("unreachable")


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

    logger.debug("Computing unsynced frequency spectrum...")
    psd_unsynced = lpsd(
        np.asarray(df["freq_unsynced"], dtype=np.float64),
        fs=fs,
        olap=p_lpsd["olap"],
        bmin=p_lpsd["bmin"],
        Lmin=p_lpsd["Lmin"],
        Jdes=p_lpsd["Jdes"],
        Kdes=p_lpsd["Kdes"],
        order=p_lpsd["order"],
        win=p_lpsd["win"],
        psll=p_lpsd["psll"],
    )
    psd_synced = lpsd(
        np.asarray(df["freq_synced"], dtype=np.float64),
        fs=fs,
        olap=p_lpsd["olap"],
        bmin=p_lpsd["bmin"],
        Lmin=p_lpsd["Lmin"],
        Jdes=p_lpsd["Jdes"],
        Kdes=p_lpsd["Kdes"],
        order=p_lpsd["order"],
        win=p_lpsd["win"],
        psll=p_lpsd["psll"],
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.loglog(psd_unsynced.f, np.sqrt(psd_unsynced.Gxx), linewidth=linewidth, label=r"Unsynced", color="gray")
    ax.loglog(psd_synced.f, np.sqrt(psd_synced.Gxx), linewidth=linewidth, label=r"Synced", color="tomato")
    ax.set_xlim(psd_synced.f[0], psd_synced.f[-1])
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
        psd_unsynced.f,
        np.sqrt(psd_unsynced.Gxx) / psd_unsynced.f,
        linewidth=linewidth,
        label=r"Unsynced",
        color="gray",
    )
    ax.loglog(
        psd_synced.f,
        np.sqrt(psd_synced.Gxx) / psd_synced.f,
        linewidth=linewidth,
        label=r"Synced",
        color="tomato",
    )
    ax.set_xlim(psd_synced.f[0], psd_synced.f[-1])
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
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    if any(x <= 0 for x in args.figsize):
        parser.error("--figsize W and H must be positive")
    for label, n in [
        ("--lpsd-bmin", args.lpsd_bmin),
        ("--lpsd-lmin", args.lpsd_lmin),
        ("--lpsd-jdes", args.lpsd_jdes),
        ("--lpsd-kdes", args.lpsd_kdes),
        ("--lpsd-order", args.lpsd_order),
        ("--lpsd-psll", args.lpsd_psll),
    ]:
        if n <= 0:
            parser.error(f"{label} must be positive, got {n!r}")
    if args.n_truncate is not None and args.n_truncate < 0:
        parser.error("--n-truncate must be non-negative")
    if args.processes is not None and args.processes < 1:
        parser.error("--processes must be at least 1")

    logger.info("mokutools moku-sync v%s | data directory: %s", VERSION, data_dir)

    pool: multiprocessing.Pool = multiprocessing.Pool(processes=args.processes)
    p_lpsd = {
        "olap": args.lpsd_olap,
        "bmin": args.lpsd_bmin,
        "Lmin": args.lpsd_lmin,
        "Jdes": args.lpsd_jdes,
        "Kdes": args.lpsd_kdes,
        "order": args.lpsd_order,
        "win": np.kaiser,
        "psll": args.lpsd_psll,
        "pool": pool,
    }

    resdir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(resdir, exist_ok=True)
    figsize = (args.figsize[0], args.figsize[1])
    dpi, fontsize, linewidth = int(args.dpi), float(args.fontsize), float(args.linewidth)
    title = str(args.plot_title)
    do_plots = not bool(args.no_plots)

    logger.info("Scanning for .csv / .mat under: %s", data_dir)
    file_list = sorted(
        f for f in os.listdir(data_dir) if f.endswith((".csv", ".mat"))
    )
    if not file_list and not (args.master and args.slave):
        _fail("No CSV or MAT files in the data directory.", pool)
    if len(file_list) < 2 and not (args.master and args.slave):
        _fail("Need at least two data files, or use --master and --slave with paths.", pool)

    if args.master and args.slave:
        file1, file2 = str(args.master), str(args.slave)
        logger.info("Non-interactive: master %s, slave %s", file1, file2)
    else:
        file1, file2 = _interactive_pick_two(data_dir, file_list, pool)

    if file1 == file2:
        _fail("Master and slave must be two different file names/paths.", pool)

    start_time = float(args.start)
    synctime = float(args.synctime)
    totaltime: Optional[float] = None if float(args.totaltime) == 0.0 else float(args.totaltime)

    path1 = _resolve_in_data_dir(data_dir, file1)
    path2 = _resolve_in_data_dir(data_dir, file2)
    for label, pth in (("master", path1), ("slave", path2)):
        if not os.path.isfile(pth):
            _fail(f"Not a file ({label}): {pth}", pool)

    logger.info("Loading master: %s", path1)
    mo1 = MokuPhasemeterObject(
        path1, start_time=start_time, duration=totaltime, prefix="1_", logger=logger
    )
    fs = float(mo1.fs)

    logger.info("Loading slave: %s", path2)
    mo2 = MokuPhasemeterObject(
        path2, start_time=start_time, duration=totaltime, prefix="2_", logger=logger
    )

    if mo1.fs != mo2.fs:
        _fail("The two input files do not have the same sampling rate.", pool)

    master_col = str(args.master_col)
    slave_col = str(args.slave_col)
    sig_master = "1_" + master_col
    sig_slave = "2_" + slave_col

    if sig_master not in mo1.df:
        _fail(
            f"Column {sig_master!r} not in master. Try --master-col (e.g. 1_freq).", pool
        )
    if sig_slave not in mo2.df:
        _fail(
            f"Column {sig_slave!r} not in slave. Try --slave-col (e.g. 1_freq).", pool
        )

    record_lengths = [len(mo1.df) / fs, len(mo2.df) / fs]
    max_totaltime = min(record_lengths) - start_time

    if (totaltime is None) or (totaltime >= max_totaltime):
        end_row = int(max_totaltime * fs)
        mo1.df = mo1.df.iloc[:end_row].copy()
        mo1.df.reset_index(drop=True, inplace=True)
        mo2.df = mo2.df.iloc[:end_row].copy()
        mo2.df.reset_index(drop=True, inplace=True)
        logger.info(
            "Using maximum overlap: %.2f s (%d rows) at fs = %.3f Hz",
            max_totaltime, end_row, float(mo1.fs),
        )
    else:
        logger.info(
            "Load window: ~%.2f s, fs = %.3f Hz, rows = %d",
            float(mo1.duration),
            float(mo1.fs),
            int(mo1.nrows),
        )

    dt_meta = float((mo1.date - mo2.date).total_seconds())
    if dt_meta < 0.0:
        logger.debug("Metadata: master clock is ahead of slave (%.2f s).", abs(dt_meta))
    else:
        logger.debug("Metadata: slave is ahead of master (%.2f s).", dt_meta)
    if abs(dt_meta) > 100.0:
        logger.warning(
            "File timestamp gap is very large (%.0f s). Check recording metadata.", abs(dt_meta)
        )

    if len(mo1.df) != len(mo2.df):
        if len(mo1.df) > len(mo2.df):
            _fail(f"Slave is shorter than master after windowing: {os.path.basename(path2)}", pool)
        _fail(f"Master is shorter than slave after windowing: {os.path.basename(path1)}", pool)

    init_offset = 0.0 if args.no_init_offset else dt_meta

    logger.info("Merging and checking NaNs ...")
    df1_nans = get_columns_with_nans(mo1.df)
    df2_nans = get_columns_with_nans(mo2.df)

    if df1_nans:
        for col, num in df1_nans.items():
            logger.warning("NaNs in master %r column index %r: %r", file1, num, col)
    if df2_nans:
        for col, num in df2_nans.items():
            logger.warning("NaNs in slave  %r column index %r: %r", file2, num, col)

    if sig_master in df1_nans:
        _fail("The master sync column contains NaNs. Choose a different --master-col.", pool)
    if sig_slave in df2_nans:
        _fail("The slave sync column contains NaNs. Choose a different --slave-col.", pool)

    non_nan_columns_df1 = [c for c in mo1.df.columns if c not in df1_nans]
    non_nan_columns_df2 = [c for c in mo2.df.columns if c not in df2_nans]
    df1_non_nan = mo1.df[non_nan_columns_df1].copy()
    df2_non_nan = mo2.df[non_nan_columns_df2].copy()
    h1, h2 = int(mo1.header_rows), int(mo2.header_rows)
    # Header text for the output CSV (read_lines() cannot open .csv.zip as plain text).
    meta1 = [ln.strip() for ln in mo1.header[:h1]]
    meta2 = [ln.strip() for ln in mo2.header[:h2]]

    del mo1, mo2

    df = pd.concat(
        [df1_non_nan.reset_index(drop=True), df2_non_nan.reset_index(drop=True)],
        axis=1,
    )
    del df1_non_nan, df2_non_nan

    if "phase" in sig_master:
        df[sig_master + "_to_freq"] = np.diff(df[sig_master], prepend=np.nan) / (2.0 * np.pi / fs)
        sig_master = sig_master + "_to_freq"
        non_nan_columns_df1.append(sig_master)
    if "phase" in sig_slave:
        df[sig_slave + "_to_freq"] = np.diff(df[sig_slave], prepend=np.nan) / (2.0 * np.pi / fs)
        sig_slave = sig_slave + "_to_freq"
        non_nan_columns_df2.append(sig_slave)

    df = df.iloc[1:].copy()
    nrows = int(len(df))

    r1 = float(np.sqrt(np.mean(np.square(df[sig_master] - np.mean(df[sig_master])))))
    r2 = float(np.sqrt(np.mean(np.square(df[sig_slave] - np.mean(df[sig_slave])))))
    logger.debug("RMS (sync ch.) master: %.5g, slave: %.5g (same units as input)", r1, r2)

    if 0.0 < synctime < max_totaltime:
        nseg = int(synctime * fs)
        in1 = np.asarray(df[sig_master].iloc[:nseg], dtype=np.float64)
        in2 = np.asarray(df[sig_slave].iloc[:nseg], dtype=np.float64)
        logger.info("Using first %.3f s (%d samples) to estimate the offset.", synctime, nseg)
    else:
        if 0.0 < synctime:
            logger.warning(
                "synctime %.2f s is not < overlap %.2f s; using the full %d rows.",
                synctime,
                max_totaltime,
                nrows,
            )
        in1 = np.asarray(df[sig_master], dtype=np.float64)
        in2 = np.asarray(df[sig_slave], dtype=np.float64)
        logger.info("Using the full %d rows for offset estimation.", nrows)

    n_auto = int(2.0 * abs(dt_meta * fs))
    n_truncate = n_auto if args.n_truncate is None else int(args.n_truncate)
    if n_truncate * 2 >= nrows:
        pool.close()
        pool.join()
        logger.error(
            "n_truncate=%d is too large for the merged length %d. Lower --n-truncate or use more data.",
            n_truncate, nrows,
        )
        sys.exit(1)

    logger.info(
        "sync_signals: model=%s, domain=%s, method=%s, interp=%d, n_truncate=%d (auto was %d), init=%.3e s",
        args.model, args.domain, args.method, args.interp_order, n_truncate, n_auto, init_offset,
    )

    _u, sync = sync_signals(
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
    del _u

    dt = float(sync.timer_offsets[0])
    logger.info("Resulting offset: %.6e s (%.6e samples at fs).", dt, dt * fs)

    for sname in non_nan_columns_df2:
        v = np.asarray(df[sname], dtype=np.float64)
        df[sname + "_shifted"] = timeshift(v, dt * fs)

    df = df.iloc[n_truncate:-n_truncate].copy()
    nrows = int(len(df))

    df["freq_unsynced"] = df[sig_master] - df[sig_slave]
    df["freq_synced"] = df[sig_master] - df[sig_slave + "_shifted"]
    m_u = float(np.mean(df["freq_unsynced"]))
    m_s = float(np.mean(df["freq_synced"]))
    df["phase_unsynced"] = (2.0 * np.pi / fs) * np.cumsum(np.asarray(df["freq_unsynced"] - m_u, dtype=np.float64))
    df["phase_synced"] = (2.0 * np.pi / fs) * np.cumsum(np.asarray(df["freq_synced"] - m_s, dtype=np.float64))
    ax_t = np.arange(nrows, dtype=np.float64)
    df["phase_unsynced"] = df["phase_unsynced"] - np.polyval(
        np.polyfit(ax_t, np.asarray(df["phase_unsynced"], dtype=np.float64), 1), ax_t
    )
    df["phase_synced"] = df["phase_synced"] - np.polyval(
        np.polyfit(ax_t, np.asarray(df["phase_synced"], dtype=np.float64), 1), ax_t
    )
    df.insert(0, "time", ax_t / float(fs))

    out = str(args.output_file)
    output_path = out if os.path.isabs(out) else os.path.join(resdir, out)
    out_parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_parent, exist_ok=True)

    logger.info("Writing %s", output_path)
    if meta1 and len(meta1) and isinstance(meta1[0], str) and not meta1[0].rstrip().endswith("(Master)"):
        meta1[0] = meta1[0] + " (Master)"
    if meta2 and len(meta2) and isinstance(meta2[0], str) and not meta2[0].rstrip().endswith("(Slave)"):
        meta2[0] = meta2[0] + " (Slave)"
    out_lines: List[str] = list(meta1) + list(meta2) + [
        f"% moku-sync v{VERSION} -- dt = {dt:.14f} s, {dt * fs:.14f} samples @ fs; "
        f"data-dir={data_dir!r} master={file1!r} slave={file2!r} model={args.model} domain={args.domain}"
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        for line in out_lines:
            s = line if line.endswith("\n") else (line + "\n")
            fh.write(s)
    with open(output_path, "a", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False)

    if do_plots:
        _plot_pdfs(resdir, df, fs, p_lpsd, title, figsize, dpi, fontsize, linewidth)
    else:
        logger.info("Skipping plots (--no-plots).")

    pool.close()
    pool.join()
    logger.info("Finished. Primary output: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)