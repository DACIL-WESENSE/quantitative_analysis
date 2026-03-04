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
    folders = sorted([p for p in root.iterdir() if p.is_dir()])
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
    """Load patient metadata and telemetry data from a CSV file.

    The function reads the CSV and uses the first row as headers.
    Both ``info_df`` and ``telemetry_df`` are derived from the same file.

    Parameters
    ----------
    csv_path : Path
        Path to the ``.csv`` file.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(patient_info_df, telemetry_df)`` where *telemetry_df* has columns
        for every metric and a ``Stage`` column parsed from annotation rows.
    """
    raw = pd.read_csv(csv_path, header=0)

    logger.debug("CSV rows loaded: %d", len(raw))

    info_df = raw

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

    for _, row in df.iterrows():
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

    # Drop rows that are purely annotation rows (non-numeric first numeric col).
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df = df.dropna(subset=[numeric_cols[0]])

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
    for k in k_range:
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
