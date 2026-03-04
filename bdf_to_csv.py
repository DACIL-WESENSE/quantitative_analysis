#!/usr/bin/env python3
"""Batch convert Biosemi BDF files to timeseries CSV or TSV.

Each output file contains one row per sample with the following columns:

- ``time_s``  : elapsed time in seconds from the start of the recording
- one column per EEG/ECG/EMG/EOG channel, in microvolts (uV)
- one column per other signal channel, in its native physical unit
- ``status``  : integer trigger / event codes from the BDF Status channel

Usage
-----
    python bdf_to_csv.py                          # convert all .bdf under ./data
    python bdf_to_csv.py --data-dir ./data        # explicit directory
    python bdf_to_csv.py --format tsv             # write tab-separated files
    python bdf_to_csv.py --output-dir ./output    # put all outputs in one folder
    python bdf_to_csv.py --channels Fp1 Fp2       # export only selected channels
    python bdf_to_csv.py --overwrite              # re-convert even if output exists
    python bdf_to_csv.py --dry-run                # preview without writing files

The BDF format specification is documented at:
    https://www.biosemi.com/faq/file_format.htm
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from typing import Optional

import mne
import numpy as np
import pandas as pd

mne.set_log_level("WARNING")

logger = logging.getLogger(__name__)

# Channel types for which digital values (V) are converted to microvolts (uV).
_UV_TYPES: frozenset[str] = frozenset({"eeg", "ecg", "emg", "eog"})


def read_bdf(
    bdf_path: str,
    channels: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Read a Biosemi BDF file and return a timeseries DataFrame.

    Parameters
    ----------
    bdf_path : str
        Path to the ``.bdf`` file.
    channels : list of str, optional
        Channel names to include.  When ``None`` all data channels are
        included.  The Status channel is always included as ``status``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``time_s``, one column per selected signal channel, ``status``.
        EEG/ECG/EMG/EOG channels are in microvolts; all other types retain their
        native physical unit as returned by MNE.
    """
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)

    # --- stim (Status) channel -------------------------------------------
    stim_picks = mne.pick_types(raw.info, stim=True, exclude=[])
    if stim_picks.size > 0:
        status_data = raw.get_data(picks=stim_picks)[0].astype(np.int32)
    else:
        status_data = np.zeros(raw.n_times, dtype=np.int32)

    # --- signal channels --------------------------------------------------
    data_picks = mne.pick_types(raw.info, meg=False, eeg=True, ecg=True,
                                emg=True, eog=True, misc=True, stim=False,
                                exclude=[])
    if channels is not None:
        requested = set(channels)
        data_picks = [
            p for p in data_picks
            if raw.ch_names[p] in requested
        ]

    ch_names = [raw.ch_names[p] for p in data_picks]
    ch_types = [mne.channel_type(raw.info, p) for p in data_picks]
    data_matrix = raw.get_data(picks=data_picks)  # shape: (n_ch, n_samples)

    # --- build DataFrame --------------------------------------------------
    df = pd.DataFrame({"time_s": raw.times})

    for name, ch_type, row in zip(ch_names, ch_types, data_matrix):
        scale = 1e6 if ch_type in _UV_TYPES else 1.0
        df[name] = row * scale

    df["status"] = status_data

    return df


def convert_bdf(
    bdf_path: str,
    out_path: str,
    sep: str,
    channels: Optional[list[str]] = None,
) -> None:
    """Read *bdf_path* and write a delimited file to *out_path*.

    Parameters
    ----------
    bdf_path : str
        Source ``.bdf`` file.
    out_path : str
        Destination CSV or TSV file.
    sep : str
        Column separator (``","`` or ``"\\t"``).
    channels : list of str, optional
        Channel names to include; ``None`` means all data channels.
    """
    df = read_bdf(bdf_path, channels=channels)
    df.to_csv(out_path, sep=sep, index=False)


def main() -> None:
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch convert Biosemi BDF files to timeseries CSV or TSV. "
            "Each output file has one row per sample: time_s, signal channels, status."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        help="Root directory to search recursively for .bdf files (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write output files. "
            "Defaults to the same directory as each source .bdf file."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["csv", "tsv"],
        default="csv",
        help="Output format: csv (comma-separated) or tsv (tab-separated). Default: csv",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        metavar="CH",
        default=None,
        help="Channel names to include. Omit to export all data channels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert files even when an output file already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files",
    )
    args = parser.parse_args()

    sep = "," if args.format == "csv" else "\t"
    ext = f".{args.format}"

    bdf_files = sorted(
        glob.glob(os.path.join(args.data_dir, "**", "*.bdf"), recursive=True)
    )

    if not bdf_files:
        print(f"No .bdf files found under '{args.data_dir}'")
        return

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    converted = skipped = errors = 0

    for bdf_path in bdf_files:
        stem = os.path.splitext(os.path.basename(bdf_path))[0]
        if args.output_dir:
            out_path = os.path.join(args.output_dir, stem + ext)
        else:
            out_path = os.path.join(os.path.dirname(bdf_path), stem + ext)

        rel_bdf = os.path.relpath(bdf_path)
        rel_out = os.path.relpath(out_path)

        if os.path.exists(out_path) and not args.overwrite:
            print(
                f"  SKIP   {rel_bdf}  "
                f"(output exists; use --overwrite to re-convert)"
            )
            skipped += 1
            continue

        if args.dry_run:
            print(f"  DRY    {rel_bdf} -> {rel_out}")
            converted += 1
            continue

        try:
            convert_bdf(bdf_path, out_path, sep=sep, channels=args.channels)
            print(f"  OK     {rel_bdf} -> {rel_out}")
            converted += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR  {rel_bdf}: {exc}")
            errors += 1

    prefix = "Would convert" if args.dry_run else "Converted"
    print(f"\nDone. {prefix} {converted}, skipped {skipped}, errors {errors}.")


if __name__ == "__main__":
    main()
