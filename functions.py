"""
functions.py
============
Reusable helper functions for the DACIL-WESENSE quantitative analysis pipeline.

This module provides utilities for:
- Directory parsing and file discovery
- Data loading (CSV telemetry, BDF ECG)
- Data synchronization and cleaning
- Exploratory data analysis (EDA)
- Unsupervised machine learning (PCA, K-Means, DBSCAN)
- Output export

All functions follow PEP 8, include NumPy-style docstrings, and use type hints.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# Use non-interactive backend so plots can be saved in batch mode.
matplotlib.use("Agg")

mne.set_log_level("WARNING")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage annotation helpers
# ---------------------------------------------------------------------------

STAGE_ORDER: List[str] = ["Opwarmen", "Test", "VT1", "VT2", "Herstel"]

# ---------------------------------------------------------------------------
# 1. Directory parsing
# ---------------------------------------------------------------------------


def discover_patient_folders(root_dir: str) -> List[Path]:
    """Return all immediate sub-directories of *root_dir*.

    Parameters
    ----------
    root_dir : str
        Path to the root data directory that contains one sub-folder per
        patient/trial.

    Returns
    -------
    List[Path]
        Sorted list of patient folder paths.
    """
    root = Path(root_dir)
    folders = sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    logger.info("Discovered %d patient folder(s) under '%s'.", len(folders), root_dir)
    return folders


def find_bdf_files(folder: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the L1 and L2 BDF ECG files inside *folder*.

    The expected naming pattern is::

        WESENSETEST_<id>_L1_ECG*.bdf
        WESENSETEST_<id>_L2_ECG*.bdf

    Parameters
    ----------
    folder : Path
        Patient folder to search.

    Returns
    -------
    Tuple[Optional[Path], Optional[Path]]
        (path_to_L1, path_to_L2).  Either may be ``None`` when not found.
    """
    l1_path: Optional[Path] = None
    l2_path: Optional[Path] = None

    for bdf in folder.glob("*.bdf"):
        name = bdf.name.upper()
        if "_L1_ECG" in name:
            l1_path = bdf
        elif "_L2_ECG" in name:
            l2_path = bdf

    return l1_path, l2_path


def find_csv_file(folder: Path) -> Optional[Path]:
    """Return the first ``.csv`` file found in *folder*.

    Parameters
    ----------
    folder : Path
        Patient folder to search.

    Returns
    -------
    Optional[Path]
        Path to the CSV file, or ``None`` if not found.
    """
    files = list(folder.glob("*.csv"))
    return files[0] if files else None


# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------


def load_telemetry(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load patient metadata and telemetry data from a WESENSE CPET CSV file.

    WESENSE exports begin with a 5-row metadata header (patient info), followed
    by the column-name row, a units row, and then the time-series data.  This
    function reads both sections separately.

    Parameters
    ----------
    csv_path : Path
        Path to the ``.csv`` file.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(info_df, telemetry_df)`` where *info_df* contains the raw metadata
        rows (columns are integer-indexed, no header) and *telemetry_df* has
        named metric columns plus a ``Stage`` column.
    """
    # ── Metadata (rows 0-3) ───────────────────────────────────────────────
    info_df = pd.read_csv(
        csv_path, header=None, nrows=4, encoding_errors="replace"
    )

    # ── Telemetry ─────────────────────────────────────────────────────────
    # Skip rows 0-4 (metadata + blank); row 5 becomes the header.
    # The units row (original row 6) ends up as the first data row; it
    # contains strings in numeric columns and is dropped by _parse_telemetry.
    raw = pd.read_csv(
        csv_path,
        skiprows=list(range(5)),
        header=0,
        encoding_errors="replace",
    )

    logger.debug("CSV rows loaded: %d", len(raw))
    telemetry_df = _parse_telemetry(raw)

    return info_df, telemetry_df


def _parse_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw telemetry DataFrame and add a ``Stage`` column.

    Rows that contain a stage annotation keyword (see :data:`STAGE_ORDER`) are
    used to label subsequent rows until the next annotation is encountered.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as read directly from Excel.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with an additional ``Stage`` string column.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Identify stage-annotation rows by checking every cell.
    stage_col: List[Optional[str]] = []
    current_stage: Optional[str] = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing telemetry", leave=False):
        row_text = " ".join(str(v) for v in row.values)
        matched = None
        for stage in STAGE_ORDER:
            if stage.lower() in row_text.lower():
                matched = stage
                break
        if matched:
            current_stage = matched
        stage_col.append(current_stage)

    df["Stage"] = stage_col

    # Drop rows that are purely annotation rows (non-numeric first column).
    first_col = df.columns[0]
    df[first_col] = pd.to_numeric(df[first_col], errors="coerce")
    df = df.dropna(subset=[first_col])

    df.reset_index(drop=True, inplace=True)
    return df


def load_ecg(bdf_path: Path) -> Optional[mne.io.BaseRaw]:
    """Load a BDF ECG file with MNE and return the Raw object.

    Parameters
    ----------
    bdf_path : Path
        Path to the ``.bdf`` file.

    Returns
    -------
    Optional[mne.io.BaseRaw]
        MNE Raw object, or ``None`` if loading fails.
    """
    try:
        raw = mne.io.read_raw_bdf(str(bdf_path), preload=False, verbose=False)
        logger.info("Loaded BDF: %s  (%.1f s)", bdf_path.name, raw.times[-1])
        return raw
    except (ValueError, OSError, RuntimeError) as exc:
        logger.warning("Could not load BDF '%s': %s", bdf_path, exc)
        return None


def extract_ecg_features(raw: mne.io.BaseRaw) -> pd.DataFrame:
    """Compute basic ECG summary statistics from a Raw object.

    The function picks ECG channels (or the first EEG channel as a fallback),
    computes per-channel mean absolute amplitude and estimated heart rate from
    R-peak detection.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Loaded MNE Raw object.

    Returns
    -------
    pd.DataFrame
        One row per ECG channel with columns ``channel``, ``mean_abs_uV``,
        ``estimated_hr_bpm``.
    """
    raw.load_data(verbose=False)

    # Pick ECG channels; fall back to first available channel.
    ecg_picks = mne.pick_types(raw.info, ecg=True)
    if len(ecg_picks) == 0:
        ecg_picks = mne.pick_types(raw.info, eeg=True, meg=False)
    if len(ecg_picks) == 0:
        ecg_picks = np.arange(min(2, len(raw.ch_names)))

    records = []
    for ch_idx in ecg_picks:
        ch_name = raw.ch_names[ch_idx]
        data, _ = raw[ch_idx, :]
        data = data.squeeze()
        mean_abs = float(np.mean(np.abs(data)) * 1e6)  # convert V to uV

        # Estimate HR via autocorrelation of the signal envelope.
        hr_bpm = _estimate_hr_from_signal(data, raw.info["sfreq"])
        records.append(
            {"channel": ch_name, "mean_abs_uV": mean_abs, "estimated_hr_bpm": hr_bpm}
        )

    return pd.DataFrame(records)


def _estimate_hr_from_signal(signal: np.ndarray, sfreq: float) -> float:
    """Estimate heart rate from a 1-D ECG signal via autocorrelation.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG amplitude array.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    float
        Estimated heart rate in beats per minute, or ``float('nan')`` on
        failure.
    """
    try:
        # Band-pass rough filtering using diff as a quick proxy.
        diff_signal = np.diff(signal)
        acorr = np.correlate(diff_signal, diff_signal, mode="full")
        acorr = acorr[len(acorr) // 2 :]

        # Search for the first peak beyond 0.3 s (200 bpm max).
        min_lag = int(0.3 * sfreq)
        max_lag = int(2.0 * sfreq)  # 30 bpm min
        search_region = acorr[min_lag:max_lag]
        if len(search_region) == 0:
            return float("nan")
        peak_idx = int(np.argmax(search_region)) + min_lag
        period_s = peak_idx / sfreq
        return 60.0 / period_s if period_s > 0 else float("nan")
    except (ValueError, IndexError, ZeroDivisionError):
        return float("nan")


def sync_ecg_with_telemetry(
    ecg_features: pd.DataFrame,
    telemetry_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ECG summary features into the telemetry DataFrame.

    Since ECG is recorded at a much higher frequency, the extracted summary
    statistics are appended as metadata columns to the entire telemetry
    DataFrame.

    Parameters
    ----------
    ecg_features : pd.DataFrame
        Output from :func:`extract_ecg_features`.
    telemetry_df : pd.DataFrame
        Telemetry data with a ``Stage`` column.

    Returns
    -------
    pd.DataFrame
        Telemetry DataFrame augmented with ``ecg_mean_abs_uV`` and
        ``ecg_estimated_hr_bpm`` columns (mean across all ECG channels).
    """
    out = telemetry_df.copy()

    if not ecg_features.empty:
        out["ecg_mean_abs_uV"] = ecg_features["mean_abs_uV"].mean()
        out["ecg_estimated_hr_bpm"] = ecg_features["estimated_hr_bpm"].mean()

    return out


# ---------------------------------------------------------------------------
# 3. EDA helpers
# ---------------------------------------------------------------------------


def compute_stage_summary(
    telemetry_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute mean +/- std of selected metrics grouped by trial stage.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame with a ``Stage`` column.
    metrics : Optional[List[str]]
        Column names to summarise.  Defaults to all numeric columns.

    Returns
    -------
    pd.DataFrame
        MultiIndex summary with (mean, std) per stage.
    """
    if metrics is None:
        metrics = telemetry_df.select_dtypes(include=[np.number]).columns.tolist()

    available = [m for m in metrics if m in telemetry_df.columns]
    summary = (
        telemetry_df.groupby("Stage", observed=True)[available]
        .agg(["mean", "std"])
        .round(3)
    )
    return summary


def plot_metrics_by_stage(
    telemetry_df: pd.DataFrame,
    metrics: List[str],
    patient_id: str,
    output_dir: Optional[Path] = None,
) -> Figure:
    """Plot the time course of *metrics* coloured by trial stage.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame with a ``Stage`` column.
    metrics : List[str]
        Column names to plot.
    patient_id : str
        Label used in the figure title and saved filename.
    output_dir : Optional[Path]
        If provided, the figure is saved as a PNG in this directory.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    available = [m for m in metrics if m in telemetry_df.columns]
    if not available:
        logger.warning("No requested metrics found for patient '%s'.", patient_id)
        fig, ax = plt.subplots()
        ax.set_title(f"No data - {patient_id}")
        return fig

    n_rows = len(available)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    stage_palette = dict(zip(STAGE_ORDER, sns.color_palette("tab10", len(STAGE_ORDER))))

    for ax, metric in zip(axes, available):
        for stage, grp in telemetry_df.groupby("Stage", observed=True):
            color = stage_palette.get(str(stage), "grey")
            ax.plot(grp.index, grp[metric], label=str(stage), color=color)
        ax.set_ylabel(metric, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Observation index")
    fig.suptitle(f"Patient {patient_id} - Metrics by Stage", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f"{patient_id}_metrics_by_stage.png"
        fig.savefig(fig_path, dpi=150)
        logger.info("Saved plot: %s", fig_path)

    return fig


def plot_batch_aggregates(
    summary_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> List[Figure]:
    """Generate batch-level aggregate visualisations.

    Creates:
    1. Box plot of ``VO2`` (or first available VO2-like column) grouped by
       ``Gender``.
    2. Scatter plot of ``BMI`` vs peak VO2.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Master summary DataFrame with one row per patient and columns
        including ``Gender``, ``BMI``, and any VO2-related metric.
    output_dir : Optional[Path]
        If provided, figures are saved to this directory.

    Returns
    -------
    List[Figure]
        List of generated Matplotlib figures.
    """
    figures: List[Figure] = []

    vo2_col = _find_column(summary_df, ["VO2", "vo2", "VO2max", "vo2_max"])
    gender_col = _find_column(summary_df, ["Gender", "gender", "Geslacht"])
    bmi_col = _find_column(summary_df, ["BMI", "bmi"])

    if vo2_col and gender_col:
        fig, ax = plt.subplots(figsize=(7, 5))
        order = sorted(summary_df[gender_col].dropna().unique())
        sns.boxplot(data=summary_df, x=gender_col, y=vo2_col, order=order, ax=ax)
        ax.set_title(f"Distribution of {vo2_col} by {gender_col}")
        ax.set_xlabel(gender_col)
        ax.set_ylabel(vo2_col)
        fig.tight_layout()
        _save_figure(fig, output_dir, "batch_vo2_by_gender.png")
        figures.append(fig)

    if bmi_col and vo2_col:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(summary_df[bmi_col], summary_df[vo2_col], alpha=0.7, edgecolors="k")
        ax.set_xlabel(bmi_col)
        ax.set_ylabel(vo2_col)
        ax.set_title(f"{bmi_col} vs Peak {vo2_col}")
        fig.tight_layout()
        _save_figure(fig, output_dir, "batch_bmi_vs_vo2.png")
        figures.append(fig)

    return figures


# ---------------------------------------------------------------------------
# 4. Machine learning helpers
# ---------------------------------------------------------------------------


def prepare_features(
    telemetry_df: pd.DataFrame,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    """Prepare a scaled feature matrix from telemetry data.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame.
    drop_cols : Optional[List[str]]
        Additional column names to exclude from features.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, StandardScaler]
        ``(X_scaled, feature_df, scaler)`` where ``feature_df`` is the
        imputed (but unscaled) numeric DataFrame and ``scaler`` is the fitted
        :class:`~sklearn.preprocessing.StandardScaler`.
    """
    exclude = set(["Stage"] + (drop_cols or []))
    num_df = telemetry_df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in exclude if c in telemetry_df.columns],
        errors="ignore",
    )

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(num_df)
    feature_df = pd.DataFrame(X_imputed, columns=num_df.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, feature_df, scaler


def run_pca(
    X_scaled: np.ndarray,
    n_components: int = 10,
) -> Tuple[PCA, np.ndarray]:
    """Fit PCA on scaled data.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix of shape ``(n_samples, n_features)``.
    n_components : int
        Number of principal components to retain (capped at
        ``min(n_samples, n_features)``).

    Returns
    -------
    Tuple[PCA, np.ndarray]
        ``(pca_model, X_pca)`` where ``X_pca`` has shape
        ``(n_samples, n_components)``.
    """
    n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca


def plot_scree(
    pca: PCA,
    patient_id: str = "",
    output_dir: Optional[Path] = None,
) -> Figure:
    """Generate a Scree plot for a fitted PCA model.

    Parameters
    ----------
    pca : PCA
        Fitted :class:`~sklearn.decomposition.PCA` instance.
    patient_id : str
        Label used in the title and saved filename.
    output_dir : Optional[Path]
        If provided, the figure is saved here.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    ev = pca.explained_variance_ratio_
    cum_ev = np.cumsum(ev)
    components = np.arange(1, len(ev) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(components, ev * 100, color="steelblue", alpha=0.8, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title(f"Scree Plot - {patient_id}" if patient_id else "Scree Plot")

    ax2 = ax1.twinx()
    ax2.plot(components, cum_ev * 100, color="tomato", marker="o", label="Cumulative")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.axhline(y=90, color="tomato", linestyle="--", alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    _save_figure(fig, output_dir, f"{patient_id}_scree.png" if patient_id else "scree.png")
    return fig


def plot_pca_scatter(
    X_pca: np.ndarray,
    labels: pd.Series,
    label_name: str = "Stage",
    patient_id: str = "",
    output_dir: Optional[Path] = None,
    three_d: bool = False,
) -> Figure:
    """2-D or 3-D scatter plot of the first two/three principal components.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-transformed data of shape ``(n_samples, >= 2)``.
    labels : pd.Series
        Categorical labels for colouring (e.g., stage or gender).
    label_name : str
        Display name for the colour legend.
    patient_id : str
        Label used in the title and filename.
    output_dir : Optional[Path]
        If provided, the figure is saved here.
    three_d : bool
        If ``True`` and at least 3 PCs are available, produces a 3-D plot.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    unique_labels = labels.unique()
    palette = dict(
        zip(unique_labels, sns.color_palette("tab10", len(unique_labels)))
    )
    colors = labels.map(palette).values

    if three_d and X_pca.shape[1] >= 3:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for lbl, color in palette.items():
            mask = labels == lbl
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                X_pca[mask, 2],
                label=str(lbl),
                color=color,
                alpha=0.7,
                s=20,
            )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        title_suffix = "3D"
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        for lbl, color in palette.items():
            mask = labels == lbl
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                label=str(lbl),
                color=color,
                alpha=0.7,
                s=20,
            )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        title_suffix = "2D"
        ax.grid(True, alpha=0.3)

    title = (
        f"PCA {title_suffix} - {patient_id} ({label_name})"
        if patient_id
        else f"PCA {title_suffix} ({label_name})"
    )
    if three_d and X_pca.shape[1] >= 3:
        fig.suptitle(title)
    else:
        ax.set_title(title)  # type: ignore[union-attr]
    plt.legend(title=label_name, fontsize=8)
    fig.tight_layout()

    fname = (
        f"{patient_id}_pca_{title_suffix.lower()}_{label_name.lower()}.png"
        if patient_id
        else f"pca_{title_suffix.lower()}_{label_name.lower()}.png"
    )
    _save_figure(fig, output_dir, fname)
    return fig


def run_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 4,
) -> Tuple[KMeans, np.ndarray]:
    """Fit K-Means clustering.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    Tuple[KMeans, np.ndarray]
        ``(model, cluster_labels)``
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def run_dbscan(
    X_scaled: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> Tuple[DBSCAN, np.ndarray]:
    """Fit DBSCAN clustering.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    eps : float
        Maximum distance between two samples in the same neighbourhood.
    min_samples : int
        Minimum number of samples in a neighbourhood.

    Returns
    -------
    Tuple[DBSCAN, np.ndarray]
        ``(model, cluster_labels)``
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    return db, labels


def plot_cluster_scatter(
    feature_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    x_col: str,
    y_col: str,
    patient_id: str = "",
    algorithm_name: str = "Cluster",
    output_dir: Optional[Path] = None,
) -> Figure:
    """Scatter plot of two physiological metrics coloured by cluster assignment.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Un-scaled feature DataFrame (for axis labels and values).
    cluster_labels : np.ndarray
        Integer cluster assignments from a clustering model.
    x_col : str
        Column to use as the X axis.
    y_col : str
        Column to use as the Y axis.
    patient_id : str
        Label used in the title.
    algorithm_name : str
        Name of the clustering algorithm (for the title).
    output_dir : Optional[Path]
        If provided, the figure is saved here.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if x_col not in feature_df.columns or y_col not in feature_df.columns:
        ax.set_title("Requested columns not available")
        return fig

    unique_clusters = np.unique(cluster_labels)
    palette = dict(
        zip(unique_clusters, sns.color_palette("tab10", len(unique_clusters)))
    )

    for clust in unique_clusters:
        mask = cluster_labels == clust
        lbl = f"Cluster {clust}" if clust >= 0 else "Noise"
        color = palette[clust]
        ax.scatter(
            feature_df.loc[mask, x_col],
            feature_df.loc[mask, y_col],
            label=lbl,
            color=color,
            alpha=0.7,
            s=25,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(
        f"{algorithm_name} Clusters - {patient_id}" if patient_id else f"{algorithm_name} Clusters"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = (
        f"{patient_id}_{algorithm_name.lower()}_clusters.png"
        if patient_id
        else f"{algorithm_name.lower()}_clusters.png"
    )
    _save_figure(fig, output_dir, fname)
    return fig


def elbow_plot(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    patient_id: str = "",
    output_dir: Optional[Path] = None,
) -> Figure:
    """Generate an elbow plot to aid K-Means cluster selection.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k_range : range
        Range of K values to evaluate.
    patient_id : str
        Label used in the title and filename.
    output_dir : Optional[Path]
        If provided, the figure is saved here.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    inertias = []
    for k in tqdm(k_range, desc="Elbow (K-Means)"):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, marker="o", color="steelblue")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title(f"Elbow Plot - {patient_id}" if patient_id else "Elbow Plot")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = f"{patient_id}_elbow.png" if patient_id else "elbow.png"
    _save_figure(fig, output_dir, fname)
    return fig


# ---------------------------------------------------------------------------
# 5. Export helpers
# ---------------------------------------------------------------------------


def save_telemetry_csv(
    telemetry_df: pd.DataFrame,
    patient_id: str,
    output_dir: Path,
) -> Path:
    """Save cleaned telemetry data as a CSV file.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame.
    patient_id : str
        Used to construct the output filename.
    output_dir : Path
        Target directory (created if it does not exist).

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{patient_id}_telemetry.csv"
    telemetry_df.to_csv(out_path, index=False)
    logger.info("Saved telemetry CSV: %s", out_path)
    return out_path


def build_patient_summary(
    patient_id: str,
    info_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    ecg_features: Optional[pd.DataFrame] = None,
) -> Dict:
    """Compile a single-row summary dict for one patient.

    Parameters
    ----------
    patient_id : str
        Unique patient / folder identifier.
    info_df : pd.DataFrame
        Patient metadata DataFrame.
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame.
    ecg_features : Optional[pd.DataFrame]
        ECG summary features (optional).

    Returns
    -------
    Dict
        Flat dictionary suitable for appending to a master summary DataFrame.
    """
    row: Dict = {"patient_id": patient_id}

    # --- Patient info (first row) ---
    if not info_df.empty:
        meta_row = info_df.iloc[0]
        for col in ["Age", "BMI", "Weight", "Gender"]:
            match = _find_column_in_series(meta_row, [col])
            if match:
                row[col.lower()] = meta_row[match]

    # --- Peak telemetry values per numeric column ---
    numeric_cols = telemetry_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        row[f"peak_{col}"] = telemetry_df[col].max()
        row[f"mean_{col}"] = telemetry_df[col].mean()

    # --- ECG summary ---
    if ecg_features is not None and not ecg_features.empty:
        row["ecg_mean_abs_uV"] = ecg_features["mean_abs_uV"].mean()
        row["ecg_estimated_hr_bpm"] = ecg_features["estimated_hr_bpm"].mean()

    return row


# ---------------------------------------------------------------------------
# COPD risk scoring
# ---------------------------------------------------------------------------

#: Clinical thresholds derived from ATS/ERS CPET guidelines and COPD literature.
#: Sources: ATS/ACCP Statement on CPET (Am J Respir Crit Care Med 2003);
#:          Puente-Maestu et al. ERJ 2016; Wasserman et al. "Principles of
#:          Exercise Testing and Interpretation" 5th ed.
_COPD_THRESHOLDS: Dict[str, float] = {
    "peak_vo2_per_kg_low": 20.0,   # mL/min/kg  — below = elevated risk
    "ve_vco2_slope_high": 34.0,    # above = elevated ventilatory inefficiency
    "spo2_nadir_low": 95.0,        # % — below = exercise-induced desaturation
    "spo2_drop_high": 4.0,         # % — drop ≥ 4 pp = clinically significant
    "bf_peak_high": 40.0,          # breaths/min — above = severe hyperventilation
    "rer_peak_low": 1.0,           # below at exhaustion = effort-limited test
    "o2_pulse_peak_low": 10.0,     # mL/beat — below = reduced cardiac output
}


def parse_patient_meta(raw_df: pd.DataFrame) -> Dict:
    """Extract patient metadata from the 5-row CSV header block.

    The first five rows of a WESENSE CPET export contain key–value pairs
    encoded across alternating columns, for example::

        Identificatie: | WESENSETEST04 | Naam: | ... | Voornaam: | ...
        Bezoekdatum:   | 09.01.2026    | Geboortedatum: | 1-1-1997 | Leeftijd: | 29 Jaar
        Geslacht:      | vrouw         | Gewicht: | 68.0 kg | BMI: | 24 kg/m²
        Gebruiker:     | ...

    Parameters
    ----------
    raw_df : pd.DataFrame
        DataFrame as returned by ``pd.read_csv(path, header=0)`` — i.e. the
        first data row becomes the header and subsequent rows are data.

    Returns
    -------
    Dict
        Keys: ``patient_id``, ``visit_date``, ``birth_date``, ``age_years``,
        ``sex``, ``weight_kg``, ``bmi``.  Missing values are ``None``.
    """
    meta: Dict = {
        "patient_id": None,
        "visit_date": None,
        "birth_date": None,
        "age_years": None,
        "sex": None,
        "weight_kg": None,
        "bmi": None,
    }

    # The header itself (row 0 of the file) becomes column names; rows 0-2 of
    # raw_df correspond to file rows 1-3 (0-indexed).
    def _cell(row_idx: int, col_idx: int) -> str:
        try:
            return str(raw_df.iloc[row_idx, col_idx]).strip()
        except (IndexError, KeyError):
            return ""

    # Row 0 (file row 1): Identificatie / Naam / Voornaam
    meta["patient_id"] = _cell(0, 1) or None

    # Row 1 (file row 2): Bezoekdatum / Geboortedatum / Leeftijd
    meta["visit_date"] = _cell(1, 1) or None
    meta["birth_date"] = _cell(1, 3) or None
    age_raw = _cell(1, 5)
    try:
        meta["age_years"] = float(age_raw.split()[0])
    except (ValueError, IndexError):
        pass

    # Row 2 (file row 3): Geslacht / Gewicht / BMI
    meta["sex"] = _cell(2, 1) or None
    weight_raw = _cell(2, 3)
    try:
        meta["weight_kg"] = float(weight_raw.split()[0])
    except (ValueError, IndexError):
        pass
    bmi_raw = _cell(2, 5)
    try:
        meta["bmi"] = float(bmi_raw.split()[0])
    except (ValueError, IndexError):
        pass

    return meta


def extract_copd_features(
    telemetry_df: pd.DataFrame,
    weight_kg: Optional[float] = None,
) -> pd.Series:
    """Compute CPET-based markers associated with COPD severity.

    All markers are derived from the cleaned telemetry DataFrame that
    ``load_telemetry`` returns.  The ``Stage`` column is used to restrict
    certain calculations to the relevant exercise phase.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame with a ``Stage`` column.
    weight_kg : Optional[float]
        Patient body weight in kg.  Required to compute peak VO₂/kg; if
        ``None`` the marker is ``NaN``.

    Returns
    -------
    pd.Series
        Named Series with the following keys:

        * ``peak_vo2_ml_min``      — peak oxygen uptake (mL/min)
        * ``peak_vo2_per_kg``      — peak VO₂ normalised by body weight
        * ``ve_vco2_slope``        — slope of V′E vs V′CO₂ (linear regression)
        * ``spo2_nadir``           — minimum SpO₂ during the test (%)
        * ``spo2_drop``            — baseline SpO₂ − nadir (percentage points)
        * ``bf_peak``              — peak breathing frequency (breaths/min)
        * ``rer_peak``             — peak respiratory exchange ratio
        * ``o2_pulse_peak``        — peak oxygen pulse (mL/beat)
        * ``vt1_present``          — 1.0 if VT1 stage present, else 0.0
    """
    df = telemetry_df.copy()

    def _col(candidates: List[str]) -> Optional[str]:
        return _find_column(df, candidates)

    vo2_col = _col(["V'O2", "VO2", "V.O2", "vo2"])
    vco2_col = _col(["V'CO2", "VCO2", "V.CO2", "vco2"])
    ve_col = _col(["V'E", "VE", "V.E", "ve"])
    spo2_col = _col(["SpO2", "SPO2", "spo2"])
    bf_col = _col(["BF", "bf", "RR"])
    rer_col = _col(["RER", "rer"])
    o2_pulse_col = _col(["VO2/HR", "VO2/hr", "o2_pulse", "O2pulse"])

    # Exercise phases to use for "peak" values (exclude warm-up and recovery)
    active_stages = {"Test", "VT1", "VT2"}
    if "Stage" in df.columns:
        active_mask = df["Stage"].isin(active_stages)
        active_df = df[active_mask] if active_mask.any() else df
    else:
        active_df = df

    # ── Peak VO₂ ──────────────────────────────────────────────────────────
    peak_vo2 = float(
        pd.to_numeric(active_df[vo2_col], errors="coerce").max()
    ) if vo2_col else float("nan")
    if not np.isnan(peak_vo2) and weight_kg and weight_kg > 0:
        peak_vo2_per_kg = peak_vo2 / weight_kg
    else:
        peak_vo2_per_kg = float("nan")

    # ── VE/VCO₂ slope (linear regression over active stages) ──────────────
    ve_vco2_slope = float("nan")
    if ve_col and vco2_col:
        subset = active_df[[ve_col, vco2_col]].copy()
        subset[ve_col] = pd.to_numeric(subset[ve_col], errors="coerce")
        subset[vco2_col] = pd.to_numeric(subset[vco2_col], errors="coerce")
        subset = subset.dropna()
        if len(subset) >= 5:
            x = subset[vco2_col].values.astype(float)
            y = subset[ve_col].values.astype(float)
            # Least-squares slope via normal equations (avoid sklearn dependency)
            x_mean, y_mean = x.mean(), y.mean()
            denom = float(np.sum((x - x_mean) ** 2))
            if denom > 0:
                ve_vco2_slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)

    # ── SpO₂ nadir and drop ───────────────────────────────────────────────
    spo2_nadir = float("nan")
    spo2_drop = float("nan")
    if spo2_col:
        spo2_series = pd.to_numeric(df[spo2_col], errors="coerce")
        if not spo2_series.dropna().empty:
            # Baseline: median of first 10 chronological rows (resting / warm-up)
            baseline_spo2 = float(spo2_series.head(10).dropna().median())
            spo2_nadir = float(
                pd.to_numeric(active_df[spo2_col], errors="coerce").min()
            )
            spo2_drop = max(0.0, baseline_spo2 - spo2_nadir)

    # ── Peak breathing frequency ───────────────────────────────────────────
    bf_peak = (
        float(pd.to_numeric(active_df[bf_col], errors="coerce").max())
        if bf_col else float("nan")
    )

    # ── Peak RER ──────────────────────────────────────────────────────────
    rer_peak = (
        float(pd.to_numeric(active_df[rer_col], errors="coerce").max())
        if rer_col else float("nan")
    )

    # ── Peak oxygen pulse ─────────────────────────────────────────────────
    o2_pulse_peak = (
        float(pd.to_numeric(active_df[o2_pulse_col], errors="coerce").max())
        if o2_pulse_col else float("nan")
    )

    # ── VT1 presence ──────────────────────────────────────────────────────
    vt1_present = (
        1.0
        if "Stage" in df.columns and (df["Stage"] == "VT1").any()
        else 0.0
    )

    return pd.Series(
        {
            "peak_vo2_ml_min": peak_vo2,
            "peak_vo2_per_kg": peak_vo2_per_kg,
            "ve_vco2_slope": ve_vco2_slope,
            "spo2_nadir": spo2_nadir,
            "spo2_drop": spo2_drop,
            "bf_peak": bf_peak,
            "rer_peak": rer_peak,
            "o2_pulse_peak": o2_pulse_peak,
            "vt1_present": vt1_present,
        }
    )


def score_copd_risk(features: pd.Series) -> pd.Series:
    """Apply clinical thresholds to CPET features and compute a composite risk score.

    Each marker is compared against the thresholds in :data:`_COPD_THRESHOLDS`.
    A flag of ``1`` indicates an abnormal value associated with elevated risk.
    The composite score is the sum of all flags, mapped to an ordinal label.

    Parameters
    ----------
    features : pd.Series
        As returned by :func:`extract_copd_features`.

    Returns
    -------
    pd.Series
        Original feature values plus flag columns (``flag_*``) and:

        * ``n_flags``       — total number of raised flags (int)
        * ``risk_score``    — ``"Low"``, ``"Moderate"``, or ``"High"``
    """
    result = features.copy()
    t = _COPD_THRESHOLDS

    result["flag_low_peak_vo2"] = int(
        not np.isnan(features["peak_vo2_per_kg"])
        and features["peak_vo2_per_kg"] < t["peak_vo2_per_kg_low"]
    )
    result["flag_high_ve_vco2"] = int(
        not np.isnan(features["ve_vco2_slope"])
        and features["ve_vco2_slope"] > t["ve_vco2_slope_high"]
    )
    result["flag_low_spo2"] = int(
        not np.isnan(features["spo2_nadir"])
        and features["spo2_nadir"] < t["spo2_nadir_low"]
    )
    result["flag_spo2_drop"] = int(
        not np.isnan(features["spo2_drop"])
        and features["spo2_drop"] >= t["spo2_drop_high"]
    )
    result["flag_high_bf"] = int(
        not np.isnan(features["bf_peak"])
        and features["bf_peak"] > t["bf_peak_high"]
    )
    result["flag_low_rer"] = int(
        not np.isnan(features["rer_peak"])
        and features["rer_peak"] < t["rer_peak_low"]
    )
    result["flag_low_o2_pulse"] = int(
        not np.isnan(features["o2_pulse_peak"])
        and features["o2_pulse_peak"] < t["o2_pulse_peak_low"]
    )

    flag_cols = [c for c in result.index if c.startswith("flag_")]
    n_flags = int(sum(result[c] for c in flag_cols))
    result["n_flags"] = n_flags

    if n_flags == 0:
        risk = "Low"
    elif n_flags <= 2:
        risk = "Moderate"
    else:
        risk = "High"
    result["risk_score"] = risk

    return result


def plot_copd_radar(
    features: pd.Series,
    patient_id: str = "",
    output_dir: Optional[Path] = None,
) -> Figure:
    """Spider / radar chart of normalised COPD marker values.

    Each marker is scaled to [0, 1] relative to its clinical threshold so
    that values beyond the threshold appear in the outer (red) zone.

    Parameters
    ----------
    features : pd.Series
        As returned by :func:`extract_copd_features` (raw values, not flags).
    patient_id : str
        Used in the figure title and output filename.
    output_dir : Optional[Path]
        If provided, the figure is saved here.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    t = _COPD_THRESHOLDS

    # Each tuple: (label, value, threshold, higher_is_worse)
    markers = [
        ("Peak VO₂/kg\n(mL/min/kg)", features.get("peak_vo2_per_kg", float("nan")),
         t["peak_vo2_per_kg_low"], False),
        ("VE/VCO₂\nslope", features.get("ve_vco2_slope", float("nan")),
         t["ve_vco2_slope_high"], True),
        ("SpO₂ nadir\n(%)", features.get("spo2_nadir", float("nan")),
         t["spo2_nadir_low"], False),
        ("SpO₂ drop\n(pp)", features.get("spo2_drop", float("nan")),
         t["spo2_drop_high"], True),
        ("Peak BF\n(br/min)", features.get("bf_peak", float("nan")),
         t["bf_peak_high"], True),
        ("Peak RER", features.get("rer_peak", float("nan")),
         t["rer_peak_low"], False),
        ("O₂ pulse\n(mL/beat)", features.get("o2_pulse_peak", float("nan")),
         t["o2_pulse_peak_low"], False),
    ]

    labels = [m[0] for m in markers]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Normalise: threshold maps to 0.5; clip to [0, 1]
    norm_values: List[float] = []
    for _, val, thresh, higher_worse in markers:
        if np.isnan(val) or thresh == 0:
            norm_values.append(0.5)
            continue
        if higher_worse:
            # value/threshold * 0.5  →  1.0 means 2× threshold
            norm = np.clip(val / thresh * 0.5, 0.0, 1.0)
        else:
            # thresh/value * 0.5  →  1.0 means value = 0.5 × threshold
            norm = np.clip(thresh / val * 0.5, 0.0, 1.0) if val > 0 else 1.0
        norm_values.append(float(norm))
    norm_values += norm_values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "threshold", "", ""], fontsize=7)

    # Reference circle at threshold (0.5)
    ax.plot([a for a in angles], [0.5] * len(angles), color="grey",
            linestyle="--", linewidth=0.8, alpha=0.6)

    ax.plot(angles, norm_values, color="steelblue", linewidth=2)
    ax.fill(angles, norm_values, color="steelblue", alpha=0.25)

    title = f"COPD Marker Profile — {patient_id}" if patient_id else "COPD Marker Profile"
    ax.set_title(title, pad=20, fontsize=10)
    fig.tight_layout()

    fname = f"{patient_id}_copd_radar.png" if patient_id else "copd_radar.png"
    _save_figure(fig, output_dir, fname)
    return fig


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column from *candidates* that exists in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    candidates : List[str]
        Candidate column names in priority order.

    Returns
    -------
    Optional[str]
        Matched column name, or ``None``.
    """
    for c in candidates:
        if c in df.columns:
            return c
    # Case-insensitive fallback.
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _find_column_in_series(series: pd.Series, candidates: List[str]) -> Optional[str]:
    """Return the first matching index label from *candidates* in *series*.

    Parameters
    ----------
    series : pd.Series
        Series whose index to search.
    candidates : List[str]
        Candidate label names in priority order.

    Returns
    -------
    Optional[str]
        Matched label, or ``None``.
    """
    for c in candidates:
        if c in series.index:
            return c
    lower_map = {str(idx).lower(): idx for idx in series.index}
    for c in candidates:
        if c.lower() in lower_map:
            return str(lower_map[c.lower()])
    return None


def _save_figure(fig: Figure, output_dir: Optional[Path], filename: str) -> None:
    """Save *fig* to *output_dir/filename* if *output_dir* is not ``None``.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    output_dir : Optional[Path]
        Target directory.
    filename : str
        Output filename including extension.
    """
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure: %s", out_path)


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure the root logger with a consistent format.

    Parameters
    ----------
    log_file : Optional[str]
        If provided, log messages are also written to this file.
    level : int
        Logging level (default: ``logging.INFO``).
    """
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )
