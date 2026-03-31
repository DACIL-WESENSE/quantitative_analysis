"""CLI entrypoint for ECG-focused analysis without notebooks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger("run_ecg")

plt = None
np = None
pd = None
tqdm = None
fn = None


def _load_runtime_dependencies() -> None:
    """Import runtime dependencies lazily so --help works without installed packages."""
    global plt, np, pd, tqdm, fn
    if fn is not None:
        return
    try:
        import matplotlib.pyplot as _plt
        import numpy as _np
        import pandas as _pd
        from tqdm.auto import tqdm as _tqdm

        import functions as _fn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install requirements with "
            "'pip install -r requirements.txt'."
        ) from exc

    plt = _plt
    np = _np
    pd = _pd
    tqdm = _tqdm
    fn = _fn


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ECG feature extraction and plots without notebooks."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory containing patient sub-folders.",
    )
    parser.add_argument(
        "--output-root",
        default="output",
        help="Directory where ECG outputs are written.",
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Folder name of the patient to analyse (default: first available).",
    )
    parser.add_argument(
        "--window-duration",
        type=float,
        default=30.0,
        help="Sliding window duration in seconds for ECG feature extraction.",
    )
    parser.add_argument(
        "--preview-seconds",
        type=int,
        default=300,
        help="Duration (seconds) used for raw ECG preview plots.",
    )
    parser.add_argument(
        "--run-batch-summary",
        action="store_true",
        help="Also compute mean ECG metrics for all patients with available BDF files.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path (default: <output-root>/pipeline.log).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _select_patient_folder(data_root: Path, patient_id: Optional[str]) -> Path:
    """Return the selected patient folder, defaulting to the first available."""
    folders = fn.discover_patient_folders(str(data_root))
    if not folders:
        raise FileNotFoundError(f"No patient folders found under {data_root}")

    if patient_id is None:
        return folders[0]

    for folder in folders:
        if folder.name == patient_id:
            return folder
    raise ValueError(f"Patient folder not found: {patient_id}")


def _extract_patient_timeseries(
    patient_folder: Path,
    window_duration: float,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """Load L1/L2 BDF files and extract time-resolved ECG features."""
    l1, l2 = fn.find_bdf_files(patient_folder)
    sensor_paths = [("L1", l1), ("L2", l2)]

    sensors: Dict[str, Dict[str, object]] = {}
    frames: List[pd.DataFrame] = []

    for sensor_label, bdf_path in tqdm(
        sensor_paths,
        desc="Processing sensors",
        unit="sensor",
        leave=False,
    ):
        if bdf_path is None:
            continue
        raw = fn.load_ecg(bdf_path)
        if raw is None:
            continue

        data = raw.get_data()
        sfreq = int(raw.info["sfreq"])
        ch_names = raw.ch_names

        sensors[sensor_label] = {
            "data": data,
            "sfreq": sfreq,
            "ch_names": ch_names,
        }

        ts = fn.extract_ecg_timeseries_array(
            data,
            sfreq,
            ch_names,
            window_duration=window_duration,
        )
        ts.insert(0, "sensor", sensor_label)
        frames.append(ts)

    if not frames:
        raise RuntimeError(f"No usable ECG BDF files found for {patient_folder.name}")

    return pd.concat(frames, ignore_index=True), sensors


def _save_patient_ecg_plots(
    ts_all: pd.DataFrame,
    sensors: Dict[str, Dict[str, object]],
    patient_id: str,
    out_dir: Path,
    preview_seconds: int,
) -> None:
    """Generate and save patient ECG plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_stages = [
        ("Heart rate", "ecg_hr"),
        ("HRV time-domain", "ecg_hrv_timedomain"),
        ("HRV frequency-domain", "ecg_hrv_freqdomain"),
        ("Breathing rate", "ecg_breathing"),
        ("Raw ECG preview", "ecg_raw_preview"),
    ]
    pbar = tqdm(total=len(plot_stages), desc="Generating ECG plots", unit="plot", leave=False)

    fig_hr, ax_hr = plt.subplots(figsize=(13, 4))
    for (sensor, channel), group_df in ts_all.groupby(["sensor", "channel"], sort=False):
        t_min = group_df["time_s"] / 60
        ax_hr.plot(t_min, group_df["hr_bpm"], label=f"{sensor} – {channel}", linewidth=1.2)
    ax_hr.set_xlabel("Time (min)")
    ax_hr.set_ylabel("Heart rate (bpm)")
    ax_hr.set_title(f"{patient_id} — Heart rate")
    ax_hr.legend(fontsize=8)
    ax_hr.grid(True, alpha=0.3)
    fig_hr.tight_layout()
    fn._save_figure(fig_hr, out_dir, f"{patient_id}_ecg_hr.png")
    plt.close(fig_hr)
    pbar.update(1)

    fig_td, axes_td = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for (sensor, channel), group_df in ts_all.groupby(["sensor", "channel"], sort=False):
        label = f"{sensor} – {channel}"
        t_min = group_df["time_s"] / 60
        axes_td[0].plot(t_min, group_df["rmssd_ms"], label=label, linewidth=1.2)
        axes_td[1].plot(t_min, group_df["sdnn_ms"], label=label, linewidth=1.2)
    axes_td[0].set_title(f"{patient_id} — HRV time-domain")
    axes_td[1].set_xlabel("Time (min)")
    for axis, ylabel in zip(axes_td, ["RMSSD (ms)", "SDNN (ms)"]):
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.3)
    fig_td.tight_layout()
    fn._save_figure(fig_td, out_dir, f"{patient_id}_ecg_hrv_timedomain.png")
    plt.close(fig_td)
    pbar.update(1)

    freq_metrics = [
        ("lf_ms2", "LF power (ms²)"),
        ("hf_ms2", "HF power (ms²)"),
        ("lf_hf_ratio", "LF/HF ratio"),
    ]
    fig_fd, axes_fd = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for (sensor, channel), group_df in ts_all.groupby(["sensor", "channel"], sort=False):
        label = f"{sensor} – {channel}"
        t_min = group_df["time_s"] / 60
        for axis, (column, _) in zip(axes_fd, freq_metrics):
            axis.plot(t_min, group_df[column], label=label, linewidth=1.2)
    axes_fd[0].set_title(f"{patient_id} — HRV frequency-domain")
    axes_fd[-1].set_xlabel("Time (min)")
    for axis, (_, ylabel) in zip(axes_fd, freq_metrics):
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.3)
    fig_fd.tight_layout()
    fn._save_figure(fig_fd, out_dir, f"{patient_id}_ecg_hrv_freqdomain.png")
    plt.close(fig_fd)
    pbar.update(1)

    fig_br, ax_br = plt.subplots(figsize=(13, 4))
    for (sensor, channel), group_df in ts_all.groupby(["sensor", "channel"], sort=False):
        ax_br.plot(
            group_df["time_s"] / 60,
            group_df["breathing_rate_bpm"],
            label=f"{sensor} – {channel} (EDR)",
            linewidth=1.2,
        )
    ax_br.set_xlabel("Time (min)")
    ax_br.set_ylabel("Breathing rate (breaths/min, EDR)")
    ax_br.set_ylim(0, 45)
    ax_br.set_title(f"{patient_id} — Breathing rate")
    ax_br.legend(fontsize=8)
    ax_br.grid(True, alpha=0.3)
    fig_br.tight_layout()
    fn._save_figure(fig_br, out_dir, f"{patient_id}_ecg_breathing.png")
    plt.close(fig_br)
    pbar.update(1)

    n_panels = sum(
        int(np.asarray(sensor_info["data"]).shape[0]) for sensor_info in sensors.values()
    )
    if n_panels == 0:
        pbar.close()
        return

    fig_preview, axes_preview = plt.subplots(
        n_panels, 1, figsize=(13, 3 * n_panels), sharex=True
    )
    if n_panels == 1:
        axes_preview = [axes_preview]

    panel = 0
    for sensor_label, sensor_info in sensors.items():
        data = np.asarray(sensor_info["data"])
        sfreq = int(sensor_info["sfreq"])
        ch_names = list(sensor_info["ch_names"])
        n_show = min(int(preview_seconds * sfreq), data.shape[1])
        t_show = np.arange(n_show) / sfreq

        for ch_idx, ch_name in enumerate(ch_names):
            signal = data[ch_idx, :n_show].astype(float)
            peaks = fn._detect_r_peaks(signal, sfreq)
            axis = axes_preview[panel]
            axis.plot(t_show, signal, linewidth=0.8, color="steelblue", label="ECG")
            if len(peaks):
                axis.scatter(
                    peaks / sfreq,
                    signal[peaks],
                    color="crimson",
                    s=25,
                    zorder=3,
                    label="R-peaks",
                )
            axis.set_ylabel("Amplitude")
            axis.set_title(f"{sensor_label} — {ch_name}")
            axis.legend(fontsize=8, loc="upper right")
            axis.grid(True, alpha=0.3)
            panel += 1

    axes_preview[-1].set_xlabel("Time (s)")
    fig_preview.suptitle(
        f"{patient_id} — Raw ECG preview (first {preview_seconds} s)", fontsize=12
    )
    fig_preview.tight_layout()
    fn._save_figure(fig_preview, out_dir, f"{patient_id}_ecg_raw_preview.png")
    plt.close(fig_preview)
    pbar.update(1)
    pbar.close()


def _save_raw_ecg_exports(
    sensors: Dict[str, Dict[str, object]],
    patient_id: str,
    out_dir: Path,
) -> List[Path]:
    """Persist full raw ECG samples for dashboard-level inspection.

    Each export contains the full recording for one sensor in a compact
    long-form layout with ``time_s`` plus the raw ECG and respiration channels
    converted to microvolts when available.
    """
    export_paths: List[Path] = []

    for sensor_label, sensor_info in sensors.items():
        data = np.asarray(sensor_info["data"], dtype=float)
        sfreq = float(sensor_info["sfreq"])
        ch_names = list(sensor_info["ch_names"])
        if data.ndim != 2 or not ch_names:
            continue

        time_s = np.arange(data.shape[1], dtype=float) / sfreq
        raw_df = pd.DataFrame({"time_s": time_s})
        raw_df.insert(0, "sensor", sensor_label)

        channel_lookup = {name.lower(): idx for idx, name in enumerate(ch_names)}
        for channel_name, output_name in [("ecg", "ecg_uV"), ("resp", "resp_uV")]:
            ch_idx = channel_lookup.get(channel_name)
            if ch_idx is None:
                continue
            raw_df[output_name] = data[ch_idx].astype(float) * 1e6

        if raw_df.shape[1] <= 2:
            continue

        out_path = out_dir / f"{patient_id}_ecg_raw_{sensor_label}.csv.gz"
        raw_df.to_csv(out_path, index=False, compression="gzip")
        export_paths.append(out_path)
        LOGGER.info("Saved raw ECG samples: %s", out_path)

    return export_paths


def _run_batch_summary(
    data_root: Path,
    output_root: Path,
    window_duration: float,
) -> Optional[pd.DataFrame]:
    """Compute mean ECG metrics for all patients with usable BDF data."""
    all_folders = fn.discover_patient_folders(str(data_root))
    rows: List[Dict[str, float | str]] = []

    for folder in tqdm(
        all_folders,
        desc="Processing patients (batch ECG)",
        unit="patient",
    ):
        l1, l2 = fn.find_bdf_files(folder)
        frames: List[pd.DataFrame] = []
        for sensor_label, bdf_path in [("L1", l1), ("L2", l2)]:
            if bdf_path is None:
                continue
            raw = fn.load_ecg(bdf_path)
            if raw is None:
                continue
            ts = fn.extract_ecg_timeseries_array(
                raw.get_data(),
                int(raw.info["sfreq"]),
                raw.ch_names,
                window_duration=window_duration,
            )
            ts.insert(0, "sensor", sensor_label)
            frames.append(ts)

        if not frames:
            continue

        patient_df = pd.concat(frames, ignore_index=True)
        rows.append(
            {
                "patient_id": folder.name,
                "mean_hr_bpm": float(patient_df["hr_bpm"].mean()),
                "mean_rmssd_ms": float(patient_df["rmssd_ms"].mean()),
                "mean_sdnn_ms": float(patient_df["sdnn_ms"].mean()),
                "mean_lf_ms2": float(patient_df["lf_ms2"].mean()),
                "mean_hf_ms2": float(patient_df["hf_ms2"].mean()),
                "mean_lf_hf": float(patient_df["lf_hf_ratio"].mean()),
                "mean_breathing_bpm": float(patient_df["breathing_rate_bpm"].mean()),
            }
        )

    if not rows:
        return None

    batch_df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    batch_csv = output_root / "ecg_batch_summary.csv"
    batch_df.to_csv(batch_csv, index=False)
    LOGGER.info("Saved ECG batch summary: %s", batch_csv)

    metrics_batch = [
        ("mean_hr_bpm", "Mean HR (bpm)"),
        ("mean_rmssd_ms", "Mean RMSSD (ms)"),
        ("mean_sdnn_ms", "Mean SDNN (ms)"),
        ("mean_lf_ms2", "Mean LF power (ms²)"),
        ("mean_hf_ms2", "Mean HF power (ms²)"),
        ("mean_lf_hf", "Mean LF/HF ratio"),
        ("mean_breathing_bpm", "Mean breathing rate (bpm, EDR)"),
    ]
    fig, axes = plt.subplots(1, len(metrics_batch), figsize=(4 * len(metrics_batch), 5))
    for axis, (column, ylabel) in zip(axes, metrics_batch):
        values = batch_df[column].dropna().values
        if len(values):
            axis.boxplot(values, widths=0.5)
            axis.scatter(
                np.ones(len(values)),
                values,
                color="steelblue",
                s=30,
                zorder=3,
                alpha=0.75,
            )
        axis.set_ylabel(ylabel, fontsize=8)
        axis.set_xticks([])
        axis.grid(True, alpha=0.3, axis="y")
    fig.suptitle("ECG feature summary — all patients", fontsize=12)
    fig.tight_layout()
    fn._save_figure(fig, output_root, "ecg_batch_summary.png")
    plt.close(fig)

    return batch_df


def main() -> int:
    """Run ECG analysis from command line."""
    args = parse_args()
    _load_runtime_dependencies()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else output_root / "pipeline.log"
    fn.setup_logging(log_file=str(log_path), level=getattr(logging, args.log_level))

    if not data_root.exists():
        LOGGER.error("Data root does not exist: %s", data_root)
        return 1

    patient_folder = _select_patient_folder(data_root, args.patient_id)
    patient_id = patient_folder.name
    out_dir = output_root / "ecg_analysis" / patient_id

    ts_all, sensors = _extract_patient_timeseries(
        patient_folder=patient_folder,
        window_duration=args.window_duration,
    )
    ts_csv = out_dir / f"{patient_id}_ecg_timeseries.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_all.to_csv(ts_csv, index=False)
    LOGGER.info("Saved ECG timeseries: %s", ts_csv)
    _save_raw_ecg_exports(sensors, patient_id, out_dir)

    summary_columns = [
        "hr_bpm",
        "rmssd_ms",
        "sdnn_ms",
        "lf_ms2",
        "hf_ms2",
        "lf_hf_ratio",
        "breathing_rate_bpm",
    ]
    summary_table = ts_all[summary_columns].describe().round(2)
    summary_csv = out_dir / f"{patient_id}_ecg_summary_stats.csv"
    summary_table.to_csv(summary_csv)
    LOGGER.info("Saved ECG summary stats: %s", summary_csv)

    _save_patient_ecg_plots(
        ts_all=ts_all,
        sensors=sensors,
        patient_id=patient_id,
        out_dir=out_dir,
        preview_seconds=args.preview_seconds,
    )

    if args.run_batch_summary:
        _run_batch_summary(
            data_root=data_root,
            output_root=output_root,
            window_duration=args.window_duration,
        )

    LOGGER.info("ECG analysis finished for patient %s.", patient_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
