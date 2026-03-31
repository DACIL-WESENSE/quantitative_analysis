"""Streamlit dashboard for the DACIL-WESENSE output folder."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import functions as fn

LOGGER = logging.getLogger(__name__)

APP_TITLE = "DACIL-WESENSE results dashboard"
APP_ICON = ":material/monitor_heart:"
DEFAULT_OUTPUT_ROOT = Path("output")
SPECIAL_OUTPUT_DIRS = {"ecg_analysis", "copd_risk"}

STAGE_COLORS = {
    "Opwarmen": "#1f77b4",
    "Test": "#ff7f0e",
    "VT1": "#2ca02c",
    "VT2": "#d62728",
    "Herstel": "#7f7f7f",
    "Unknown": "#9ca3af",
}

TELEMETRY_METRIC_CHOICES = {
    "Heart rate": ["HR", "Heart Rate", "hr"],
    "VO2": ["V'O2", "VO2", "V.O2"],
    "VCO2": ["V'CO2", "VCO2", "V.CO2"],
    "Ventilation": ["V'E", "VE", "V.E"],
    "Breathing frequency": ["BF", "RR"],
    "RER": ["RER"],
    "SpO2": ["SpO2", "SPO2"],
    "Oxygen pulse": ["VO2/HR", "VO2/hr", "o2_pulse", "O2pulse"],
    "EqCO2": ["EqCO2"],
    "EqO2": ["EqO2"],
    "PETCO2": ["PETCO2"],
    "PETO2": ["PETO2"],
}


def _configure_page() -> None:
    """Set the page configuration."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _apply_styles() -> None:
    """Reserved hook for future layout tweaks."""
    return None


def _file_token(path: Path) -> tuple[int, int]:
    """Return a fingerprint for cache invalidation."""
    if not path.exists():
        return (-1, -1)
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first matching column name from ``candidates``."""
    if df.empty:
        return None
    return fn._find_column(df, list(candidates))


def _pick_series(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    numeric: bool = True,
) -> tuple[Optional[str], Optional[pd.Series]]:
    """Return a matching column and its values."""
    col = _pick_column(df, candidates)
    if col is None:
        return None, None
    series = df[col]
    if numeric:
        series = pd.to_numeric(series, errors="coerce")
    return col, series


@st.cache_data(show_spinner=False)
def _read_csv(
    path_str: str,
    fingerprint: tuple[int, int],
    header_rows: tuple[int, ...] = (0,),
    index_col: int | None = None,
) -> pd.DataFrame:
    """Read a CSV file with lightweight cache invalidation."""
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()

    header_arg: int | list[int]
    header_arg = header_rows[0] if len(header_rows) == 1 else list(header_rows)
    try:
        return pd.read_csv(path, header=header_arg, index_col=index_col)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        LOGGER.warning("Could not read CSV '%s': %s", path, exc)
        return pd.DataFrame()


def _read_optional_csv(
    path: Path,
    *,
    header_rows: tuple[int, ...] = (0,),
    index_col: int | None = None,
) -> pd.DataFrame:
    """Read a CSV when it exists, otherwise return an empty frame."""
    return _read_csv(
        path.as_posix(),
        _file_token(path),
        header_rows=header_rows,
        index_col=index_col,
    )


def _flatten_stage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the MultiIndex layout used by the stage summary export."""
    if df.empty:
        return df.copy()
    if not isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        if out.index.name is None and "Stage" not in out.columns:
            out = out.reset_index().rename(columns={"index": "Stage"})
        if "Stage" in out.columns:
            mask = out["Stage"].astype(str).str.strip().eq("Stage")
            if len(out.columns) > 1:
                mask &= out.drop(columns=["Stage"], errors="ignore").isna().all(axis=1)
            out = out.loc[~mask].copy()
        return out

    out = df.copy()
    out.columns = [
        "_".join(str(part) for part in col if str(part) not in {"", "nan"}).strip("_")
        for col in out.columns.to_flat_index()
    ]
    out = out.reset_index()
    if "index" in out.columns and "Stage" not in out.columns:
        out = out.rename(columns={"index": "Stage"})
    if "Stage" in out.columns:
        mask = out["Stage"].astype(str).str.strip().eq("Stage")
        if len(out.columns) > 1:
            mask &= out.drop(columns=["Stage"], errors="ignore").isna().all(axis=1)
        out = out.loc[~mask].copy()
    return out


def _collect_patient_ids(output_root: Path, tables: Sequence[pd.DataFrame]) -> list[str]:
    """Collect patient IDs from tables and output folders."""
    patient_ids: list[str] = []

    for table in tables:
        if table.empty or "patient_id" not in table.columns:
            continue
        patient_ids.extend(table["patient_id"].astype(str).tolist())

    if output_root.exists():
        for child in output_root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            if child.name in SPECIAL_OUTPUT_DIRS:
                continue
            patient_ids.append(child.name)

    unique_ids: list[str] = []
    seen: set[str] = set()
    for raw in patient_ids:
        canonical = fn.canonicalize_patient_id(raw) or raw
        if canonical not in seen:
            seen.add(canonical)
            unique_ids.append(canonical)

    return sorted(unique_ids, key=fn._natural_sort_key)


@st.cache_data(show_spinner=False)
def _load_core_tables(output_root_str: str) -> dict[str, object]:
    """Load the root-level output tables and patient list."""
    output_root = Path(output_root_str)
    master_summary = _read_optional_csv(output_root / "master_summary.csv")
    patient_status = _read_optional_csv(output_root / "patient_analysis_status.csv")
    breathing_reserve = _read_optional_csv(output_root / "breathing_reserve_summary.csv")
    ecg_batch = _read_optional_csv(output_root / "ecg_batch_summary.csv")
    copd_summary = _read_optional_csv(output_root / "copd_risk_summary.csv")

    patient_ids = _collect_patient_ids(
        output_root,
        [master_summary, patient_status, breathing_reserve, ecg_batch, copd_summary],
    )

    return {
        "master_summary": master_summary,
        "patient_status": patient_status,
        "breathing_reserve": breathing_reserve,
        "ecg_batch": ecg_batch,
        "copd_summary": copd_summary,
        "patient_ids": patient_ids,
    }


def _patient_paths(output_root: Path, patient_id: str) -> dict[str, Path]:
    """Return the file paths used by the patient explorer."""
    patient_dir = output_root / patient_id
    ecg_dir = output_root / "ecg_analysis" / patient_id
    copd_dir = output_root / "copd_risk"

    return {
        "patient_dir": patient_dir,
        "telemetry": patient_dir / f"{patient_id}_telemetry.csv",
        "stage_summary": patient_dir / f"{patient_id}_stage_summary.csv",
        "validation": patient_dir / f"{patient_id}_breathing_rate_validation.csv",
        "tasks_markers": patient_dir / f"{patient_id}_tasks_markers.csv",
        "metrics_plot": patient_dir / f"{patient_id}_metrics_by_stage.png",
        "ve_vco2_plot": patient_dir / f"{patient_id}_ve_vco2_slope.png",
        "oxygen_pulse_plot": patient_dir / f"{patient_id}_oxygen_pulse.png",
        "ventilatory_efficiency_plot": patient_dir / f"{patient_id}_ventilatory_efficiency.png",
        "ecg_dir": ecg_dir,
        "ecg_timeseries": ecg_dir / f"{patient_id}_ecg_timeseries.csv",
        "ecg_summary": ecg_dir / f"{patient_id}_ecg_summary_stats.csv",
        "ecg_hr_plot": ecg_dir / f"{patient_id}_ecg_hr.png",
        "ecg_hrv_time_plot": ecg_dir / f"{patient_id}_ecg_hrv_timedomain.png",
        "ecg_hrv_freq_plot": ecg_dir / f"{patient_id}_ecg_hrv_freqdomain.png",
        "ecg_breathing_plot": ecg_dir / f"{patient_id}_ecg_breathing.png",
        "ecg_raw_plot": ecg_dir / f"{patient_id}_ecg_raw_preview.png",
        "copd_radar": copd_dir / f"{patient_id}_copd_radar.png",
    }


def _load_patient_bundle(output_root: Path, patient_id: str) -> dict[str, object]:
    """Load all known output artifacts for a patient."""
    paths = _patient_paths(output_root, patient_id)

    telemetry = _read_optional_csv(paths["telemetry"])
    stage_summary = _read_optional_csv(
        paths["stage_summary"],
        header_rows=(0, 1),
        index_col=0,
    )
    if stage_summary.empty or not isinstance(stage_summary.columns, pd.MultiIndex):
        stage_summary = _read_optional_csv(paths["stage_summary"])
    stage_summary = _flatten_stage_summary(stage_summary)

    validation = _read_optional_csv(paths["validation"])
    tasks_markers = _read_optional_csv(paths["tasks_markers"])
    ecg_timeseries = _read_optional_csv(paths["ecg_timeseries"])
    ecg_summary = _read_optional_csv(paths["ecg_summary"])

    image_paths = [
        paths["metrics_plot"],
        paths["ve_vco2_plot"],
        paths["oxygen_pulse_plot"],
        paths["ventilatory_efficiency_plot"],
        paths["ecg_hr_plot"],
        paths["ecg_hrv_time_plot"],
        paths["ecg_hrv_freq_plot"],
        paths["ecg_breathing_plot"],
        paths["ecg_raw_plot"],
        paths["copd_radar"],
    ]

    return {
        "paths": paths,
        "telemetry": telemetry,
        "stage_summary": stage_summary,
        "validation": validation,
        "tasks_markers": tasks_markers,
        "ecg_timeseries": ecg_timeseries,
        "ecg_summary": ecg_summary,
        "image_paths": [path for path in image_paths if path.exists()],
    }


def _build_preview_table(
    df: pd.DataFrame,
    specs: Sequence[tuple[str, Sequence[str]]],
) -> pd.DataFrame:
    """Select and rename a handful of readable columns."""
    if df.empty:
        return pd.DataFrame()

    preview = pd.DataFrame(index=df.index)
    if "patient_id" in df.columns:
        preview["Patient"] = df["patient_id"].astype(str)
    if "Stage" in df.columns and "Stage" not in preview.columns:
        preview["Stage"] = df["Stage"]

    for label, candidates in specs:
        col = _pick_column(df, candidates)
        if col is not None:
            preview[label] = df[col]

    if preview.empty:
        return pd.DataFrame()
    return preview


def _format_value(value: object, digits: int = 1) -> str:
    """Format numbers and fall back to a readable placeholder."""
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer() and digits == 0:
        return f"{int(number)}"
    return f"{number:.{digits}f}"


def _series_value(row: pd.Series, candidates: Sequence[str]) -> object:
    """Pick the first available value from a row."""
    col = fn._find_column_in_series(row, list(candidates))
    return row[col] if col is not None else None


def _filtered_patient_ids(
    patient_ids: Sequence[str],
    patient_status: pd.DataFrame,
    *,
    only_ecg: bool,
    only_processed: bool,
    only_validated: bool,
    only_tasks: bool,
) -> list[str]:
    """Apply sidebar filters to the patient ID list."""
    if patient_status.empty:
        return list(patient_ids)

    lookup = patient_status.set_index("patient_id", drop=False)
    filtered: list[str] = []
    for patient_id in patient_ids:
        row = lookup.loc[patient_id] if patient_id in lookup.index else None
        if row is None or isinstance(row, pd.DataFrame):
            if only_ecg or only_processed or only_validated or only_tasks:
                continue
            filtered.append(patient_id)
            continue

        if only_ecg and not bool(row.get("analysis_ecg_file_present", False)):
            continue
        if only_processed and not bool(row.get("analysis_ecg_processed", False)):
            continue
        if only_validated and not bool(row.get("analysis_breathing_validation_ok", False)):
            continue
        if only_tasks and not bool(row.get("analysis_tasks_log_found", False)):
            continue
        filtered.append(patient_id)

    return filtered


def _make_boolean_heatmap(patient_status: pd.DataFrame) -> Optional[go.Figure]:
    """Build a heatmap of analysis flags."""
    if patient_status.empty or "patient_id" not in patient_status.columns:
        return None

    flag_cols = [
        col
        for col in [
            "analysis_telemetry_present",
            "analysis_ecg_file_present",
            "analysis_ecg_processed",
            "analysis_tasks_log_found",
            "analysis_tasks_markers_parsed",
            "analysis_tasks_start_marker_found",
            "analysis_breathing_validation_ok",
        ]
        if col in patient_status.columns
    ]
    if not flag_cols:
        return None

    plot_df = patient_status[["patient_id", *flag_cols]].copy()
    plot_df["patient_id"] = plot_df["patient_id"].astype(str)
    plot_df = plot_df.set_index("patient_id").fillna(False)
    plot_df = plot_df[flag_cols].astype(int)

    fig = go.Figure(
        data=go.Heatmap(
            z=plot_df.values,
            x=flag_cols,
            y=plot_df.index.tolist(),
            colorscale=[[0.0, "#e5e7eb"], [1.0, "#1f77b4"]],
            showscale=False,
            hovertemplate="Patient=%{y}<br>Flag=%{x}<br>Value=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Analysis completeness heatmap",
        xaxis_title="Flag",
        yaxis_title="Patient",
        height=max(360, 40 * len(plot_df)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _make_bar_chart(
    df: pd.DataFrame,
    *,
    value_candidates: Sequence[str],
    title: str,
    y_label: str,
    patient_order: Optional[list[str]] = None,
) -> Optional[go.Figure]:
    """Build a simple patient bar chart from a table."""
    if df.empty or "patient_id" not in df.columns:
        return None

    value_col = _pick_column(df, value_candidates)
    if value_col is None:
        return None

    plot_df = df[["patient_id", value_col]].copy()
    plot_df["patient_id"] = plot_df["patient_id"].astype(str)
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col])
    if plot_df.empty:
        return None

    if patient_order:
        plot_df["patient_id"] = pd.Categorical(
            plot_df["patient_id"], categories=patient_order, ordered=True
        )
        plot_df = plot_df.sort_values("patient_id")

    fig = px.bar(
        plot_df,
        x="patient_id",
        y=value_col,
        text_auto=".1f",
        color=value_col,
        color_continuous_scale=["#dbeafe", "#1f77b4"],
        title=title,
    )
    fig.update_layout(
        xaxis_title="Patient",
        yaxis_title=y_label,
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_showscale=False,
    )
    return fig


def _make_validation_scatter(validation: pd.DataFrame) -> Optional[go.Figure]:
    """Build an ECG-vs-telemetry breathing-rate scatter plot."""
    if validation.empty:
        return None
    calc_col = _pick_column(validation, ["calculated_br_bpm"])
    meas_col = _pick_column(validation, ["measured_br_bpm"])
    if calc_col is None or meas_col is None:
        return None

    plot_df = validation[[calc_col, meas_col]].copy()
    plot_df[calc_col] = pd.to_numeric(plot_df[calc_col], errors="coerce")
    plot_df[meas_col] = pd.to_numeric(plot_df[meas_col], errors="coerce")
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return None

    fig = px.scatter(
        plot_df,
        x=calc_col,
        y=meas_col,
        title="Breathing-rate validation",
        labels={calc_col: "ECG-derived breathing rate (bpm)", meas_col: "Measured breathing rate (bpm)"},
    )
    min_val = float(min(plot_df[calc_col].min(), plot_df[meas_col].min()))
    max_val = float(max(plot_df[calc_col].max(), plot_df[meas_col].max()))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Identity",
            line=dict(color="#7f7f7f", dash="dash"),
        )
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _make_telemetry_chart(
    telemetry: pd.DataFrame,
    metric_label: str,
) -> Optional[go.Figure]:
    """Plot one telemetry metric over observation index and stage."""
    candidates = TELEMETRY_METRIC_CHOICES.get(metric_label, [metric_label])
    col = _pick_column(telemetry, candidates)
    if col is None:
        return None

    plot_df = telemetry.copy().reset_index(drop=True)
    plot_df["Observation"] = plot_df.index + 1
    if "Stage" in plot_df.columns:
        plot_df["Stage"] = plot_df["Stage"].fillna("Unknown").astype(str)
    else:
        plot_df["Stage"] = pd.Series(["Unknown"] * len(plot_df), index=plot_df.index)
    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=[col])
    if plot_df.empty:
        return None

    fig = px.line(
        plot_df,
        x="Observation",
        y=col,
        color="Stage",
        color_discrete_map=STAGE_COLORS,
        title=metric_label,
        labels={col: metric_label},
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _make_telemetry_scatter(telemetry: pd.DataFrame) -> Optional[go.Figure]:
    """Plot VO2 against HR for the selected patient."""
    hr_col = _pick_column(telemetry, ["HR", "Heart Rate", "hr"])
    vo2_col = _pick_column(telemetry, ["V'O2", "VO2", "V.O2"])
    if hr_col is None or vo2_col is None:
        return None

    plot_df = telemetry[[hr_col, vo2_col]].copy()
    if "Stage" in telemetry.columns:
        plot_df["Stage"] = telemetry["Stage"].fillna("Unknown").astype(str).values
    else:
        plot_df["Stage"] = "Unknown"
    plot_df[hr_col] = pd.to_numeric(plot_df[hr_col], errors="coerce")
    plot_df[vo2_col] = pd.to_numeric(plot_df[vo2_col], errors="coerce")
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return None

    fig = px.scatter(
        plot_df,
        x=vo2_col,
        y=hr_col,
        color="Stage",
        color_discrete_map=STAGE_COLORS,
        title="VO2 vs HR",
        labels={vo2_col: "VO2", hr_col: "Heart rate (bpm)"},
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _render_image_grid(image_paths: Sequence[Path], *, columns: int = 2) -> None:
    """Render a simple image gallery."""
    if not image_paths:
        st.info("No image artifacts were found for this selection.")
        return

    for start in range(0, len(image_paths), columns):
        row = image_paths[start : start + columns]
        cols = st.columns(columns)
        for col, image_path in zip(cols, row):
            with col:
                st.image(
                    str(image_path),
                    caption=image_path.name,
                    use_container_width=True,
                )


def _render_cohort_overview(
    output_root: Path,
    core: dict[str, object],
    patient_ids: Sequence[str],
) -> None:
    """Render the cohort-level dashboard."""
    master = core["master_summary"]
    status = core["patient_status"]
    breathing_reserve = core["breathing_reserve"]
    ecg_batch = core["ecg_batch"]
    copd_summary = core["copd_summary"]

    master_view = (
        master[master["patient_id"].astype(str).isin(patient_ids)].copy()
        if not master.empty and "patient_id" in master.columns
        else pd.DataFrame()
    )
    status_view = (
        status[status["patient_id"].astype(str).isin(patient_ids)].copy()
        if not status.empty and "patient_id" in status.columns
        else pd.DataFrame()
    )
    br_view = (
        breathing_reserve[breathing_reserve["patient_id"].astype(str).isin(patient_ids)].copy()
        if not breathing_reserve.empty and "patient_id" in breathing_reserve.columns
        else pd.DataFrame()
    )
    ecg_view = (
        ecg_batch[ecg_batch["patient_id"].astype(str).isin(patient_ids)].copy()
        if not ecg_batch.empty and "patient_id" in ecg_batch.columns
        else pd.DataFrame()
    )
    copd_view = (
        copd_summary[copd_summary["patient_id"].astype(str).isin(patient_ids)].copy()
        if not copd_summary.empty and "patient_id" in copd_summary.columns
        else pd.DataFrame()
    )

    metrics = [
        ("Patients in view", len(patient_ids), "Patient folders with generated output"),
        (
            "ECG files",
            int(status_view.get("analysis_ecg_file_present", pd.Series(dtype=bool)).sum())
            if not status_view.empty
            else 0,
            "Patients with at least one ECG BDF",
        ),
        (
            "ECG processed",
            int(status_view.get("analysis_ecg_processed", pd.Series(dtype=bool)).sum())
            if not status_view.empty
            else 0,
            "Patients with ECG extraction completed",
        ),
        (
            "Validation ok",
            int(status_view.get("analysis_breathing_validation_ok", pd.Series(dtype=bool)).sum())
            if not status_view.empty
            else 0,
            "Patients with aligned breathing-rate validation",
        ),
    ]

    st.subheader("Cohort snapshot")
    metric_cols = st.columns(len(metrics))
    for col, (label, value, help_text) in zip(metric_cols, metrics):
        with col:
            with st.container(border=True):
                st.metric(label, value, help=help_text)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        with st.container(border=True):
            st.markdown("**Peak VO2 by patient**")
            fig = _make_bar_chart(
                master_view,
                value_candidates=["peak_V'O2", "peak_VO2", "peak_vo2_ml_min"],
                title="Peak VO2 by patient",
                y_label="Peak VO2",
                patient_order=list(patient_ids),
            )
            if fig is None:
                st.info("No peak VO2 column was found in `master_summary.csv`.")
            else:
                st.plotly_chart(fig, use_container_width=True)

    with chart_cols[1]:
        with st.container(border=True):
            st.markdown("**Analysis completeness heatmap**")
            fig = _make_boolean_heatmap(status_view)
            if fig is None:
                st.info("No status table was found for the current selection.")
            else:
                st.plotly_chart(fig, use_container_width=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        with st.container(border=True):
            st.markdown("**Breathing reserve by patient**")
            fig = _make_bar_chart(
                br_view,
                value_candidates=["BR_pct"],
                title="Breathing reserve percentage",
                y_label="BR %",
                patient_order=list(patient_ids),
            )
            if fig is None:
                st.info("No breathing reserve summary was found.")
            else:
                st.plotly_chart(fig, use_container_width=True)

    with chart_cols[1]:
        with st.container(border=True):
            st.markdown("**ECG batch summary**")
            batch_image = output_root / "ecg_batch_summary.png"
            if batch_image.exists():
                st.image(str(batch_image), use_container_width=True)
            elif not ecg_view.empty:
                preview = _build_preview_table(
                    ecg_view,
                    [
                        ("Mean HR", ["mean_hr_bpm"]),
                        ("RMSSD", ["mean_rmssd_ms"]),
                        ("SDNN", ["mean_sdnn_ms"]),
                        ("Breathing rate", ["mean_breathing_bpm"]),
                    ],
                )
                if preview.empty:
                    st.info("The ECG batch summary file is present but could not be previewed.")
                else:
                    st.dataframe(preview, hide_index=True, use_container_width=True)
            else:
                st.info("No ECG batch summary was found.")

    st.subheader("Key tables")
    table_cols = st.columns(2)
    with table_cols[0]:
        with st.container(border=True):
            st.markdown("**Master summary preview**")
            preview = _build_preview_table(
                master_view,
                [
                    ("Peak HR", ["peak_HR", "peak_hr"]),
                    ("Mean HR", ["mean_HR", "mean_hr_bpm"]),
                    ("Peak VO2", ["peak_V'O2", "peak_VO2"]),
                    ("Mean VO2", ["mean_V'O2", "mean_VO2"]),
                    ("Peak VE", ["peak_V'E", "peak_VE"]),
                    ("Peak SpO2", ["peak_SpO2", "peak_SPO2"]),
                    ("RMSSD", ["mean_rmssd_ms"]),
                    ("SDNN", ["mean_sdnn_ms"]),
                    ("Breathing MAE", ["breathing_rate_mae_bpm"]),
                ],
            )
            if preview.empty:
                st.info("No master summary table was found.")
            else:
                st.dataframe(preview, hide_index=True, use_container_width=True)

    with table_cols[1]:
        with st.container(border=True):
            st.markdown("**Analysis status preview**")
            preview = _build_preview_table(
                status_view,
                [
                    ("ECG file", ["analysis_ecg_file_present"]),
                    ("ECG processed", ["analysis_ecg_processed"]),
                    ("Tasks log", ["analysis_tasks_log_found"]),
                    ("Validation ok", ["analysis_breathing_validation_ok"]),
                    ("Mean HR", ["mean_hr_bpm"]),
                    ("RMSSD", ["mean_rmssd_ms"]),
                    ("SDNN", ["mean_sdnn_ms"]),
                    ("Breathing MAE", ["breathing_rate_mae_bpm"]),
                ],
            )
            if preview.empty:
                st.info("No patient analysis status table was found.")
            else:
                st.dataframe(preview, hide_index=True, use_container_width=True)

    if not copd_view.empty:
        st.subheader("COPD risk summary")
        risk_cols = st.columns(2)
        with risk_cols[0]:
            with st.container(border=True):
                st.markdown("**COPD risk scores**")
                risk_col = _pick_column(copd_view, ["risk_score"])
                if risk_col is None:
                    st.info("The COPD risk file is missing a `risk_score` column.")
                else:
                    counts = (
                        copd_view[risk_col]
                        .astype(str)
                        .value_counts()
                        .rename_axis("risk_score")
                        .reset_index(name="count")
                    )
                    fig = px.bar(counts, x="risk_score", y="count", color="risk_score")
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)
        with risk_cols[1]:
            with st.container(border=True):
                st.markdown("**COPD risk preview**")
                st.dataframe(copd_view, hide_index=True, use_container_width=True)


def _render_telemetry_section(
    telemetry: pd.DataFrame,
    selected_metric_labels: Sequence[str],
) -> None:
    """Render the telemetry charts for a patient."""
    if telemetry.empty:
        st.info("No telemetry CSV was found for this patient.")
        return

    available_metrics = [
        label
        for label in selected_metric_labels
        if _pick_column(telemetry, TELEMETRY_METRIC_CHOICES.get(label, [label])) is not None
    ]

    if not available_metrics:
        st.info("None of the selected telemetry metrics are available in this patient file.")
    else:
        chart_cols = st.columns(2)
        for index, metric_label in enumerate(available_metrics):
            with chart_cols[index % 2]:
                with st.container(border=True):
                    fig = _make_telemetry_chart(telemetry, metric_label)
                    if fig is None:
                        st.info(f"{metric_label} is not available for this patient.")
                    else:
                        st.plotly_chart(fig, use_container_width=True)

    scatter_fig = _make_telemetry_scatter(telemetry)
    if scatter_fig is not None:
        with st.container(border=True):
            st.markdown("**VO2 vs HR scatter**")
            st.plotly_chart(scatter_fig, use_container_width=True)

    with st.expander("Telemetry table", expanded=False):
        st.dataframe(telemetry, hide_index=True, use_container_width=True, height=420)


def _render_stage_summary_section(stage_summary: pd.DataFrame, telemetry: pd.DataFrame) -> None:
    """Render the stage summary table and chart."""
    if stage_summary.empty and telemetry.empty:
        st.info("No stage summary or telemetry table was found.")
        return

    if stage_summary.empty and not telemetry.empty and "Stage" in telemetry.columns:
        try:
            stage_summary = fn.compute_stage_summary(
                telemetry,
                metrics=[
                    col
                    for col in ["HR", "V'O2", "V'CO2", "V'E", "BF", "SpO2", "RER"]
                    if col in telemetry.columns
                ],
            )
            stage_summary = _flatten_stage_summary(stage_summary)
        except ValueError:
            stage_summary = pd.DataFrame()

    if stage_summary.empty:
        st.info("No stage summary was available for this patient.")
        return

    if "Stage" not in stage_summary.columns:
        stage_summary = stage_summary.reset_index().rename(columns={"index": "Stage"})

    mean_cols = [col for col in stage_summary.columns if col.endswith("_mean")]
    if mean_cols:
        plot_df = stage_summary[["Stage", *mean_cols]].copy()
        plot_df = plot_df.melt(id_vars="Stage", var_name="metric", value_name="value")
        plot_df["metric"] = plot_df["metric"].str.replace("_mean", "", regex=False)
        fig = px.bar(
            plot_df,
            x="Stage",
            y="value",
            color="metric",
            barmode="group",
            title="Stage means",
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"],
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(stage_summary, hide_index=True, use_container_width=True)


def _render_ecg_section(ecg_timeseries: pd.DataFrame, ecg_summary: pd.DataFrame) -> None:
    """Render ECG timeseries and summary outputs."""
    if ecg_summary.empty and ecg_timeseries.empty:
        st.info("No ECG analysis files were found for this patient.")
        return

    if not ecg_summary.empty:
        st.dataframe(ecg_summary, use_container_width=True)

    if ecg_timeseries.empty:
        st.info("No ECG timeseries export is available.")
        return

    metric_options = [
        col
        for col in ["hr_bpm", "rmssd_ms", "sdnn_ms", "lf_ms2", "hf_ms2", "breathing_rate_bpm"]
        if col in ecg_timeseries.columns
    ]
    if metric_options:
        chosen_metric = st.selectbox(
            "ECG timeseries metric",
            metric_options,
            index=0,
            key="ecg_metric_select",
        )
        plot_df = ecg_timeseries.copy()
        plot_df["channel_label"] = plot_df.get("sensor", "sensor").astype(str)
        if "channel" in plot_df.columns:
            plot_df["channel_label"] = plot_df["channel_label"] + " - " + plot_df["channel"].astype(str)
        if "time_s" not in plot_df.columns:
            plot_df["time_s"] = range(1, len(plot_df) + 1)
        plot_df[chosen_metric] = pd.to_numeric(plot_df[chosen_metric], errors="coerce")
        plot_df = plot_df.dropna(subset=[chosen_metric])
        if not plot_df.empty:
            fig = px.line(
                plot_df,
                x="time_s",
                y=chosen_metric,
                color="channel_label",
                title=chosen_metric,
                labels={"time_s": "Time (s)", chosen_metric: chosen_metric},
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("ECG timeseries table", expanded=False):
        st.dataframe(ecg_timeseries, hide_index=True, use_container_width=True, height=420)


def _render_validation_section(validation: pd.DataFrame) -> None:
    """Render breathing-rate validation output."""
    if validation.empty:
        st.info("No aligned breathing-rate samples were exported for this patient.")
        return

    metric_cols = st.columns(4)
    metrics = [
        ("Aligned samples", validation.get("time_s_calc", pd.Series(dtype=float)).notna().sum()),
        ("MAE", None),
        ("RMSE", None),
        ("Correlation", None),
    ]
    if {"calculated_br_bpm", "measured_br_bpm"}.issubset(validation.columns):
        calc = pd.to_numeric(validation["calculated_br_bpm"], errors="coerce")
        meas = pd.to_numeric(validation["measured_br_bpm"], errors="coerce")
        valid = pd.DataFrame({"calc": calc, "meas": meas}).dropna()
        if not valid.empty:
            err = valid["calc"] - valid["meas"]
            metrics[1] = ("MAE", float(err.abs().mean()))
            metrics[2] = ("RMSE", float((err ** 2).mean() ** 0.5))
            if len(valid) >= 2 and valid["calc"].nunique() > 1 and valid["meas"].nunique() > 1:
                metrics[3] = ("Correlation", float(valid["calc"].corr(valid["meas"])))

    for col, (label, value) in zip(metric_cols, metrics):
        with col:
            with st.container(border=True):
                st.metric(
                    label,
                    _format_value(value, digits=2 if label != "Aligned samples" else 0),
                )

    fig = _make_validation_scatter(validation)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(validation, hide_index=True, use_container_width=True)


def _render_file_inventory(output_root: Path) -> None:
    """Render a simple inventory of generated files."""
    if not output_root.exists():
        st.info("The output folder does not exist yet.")
        return

    records: list[dict[str, object]] = []
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(output_root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if path.parent == output_root:
            patient_id = ""
        elif rel_parts and rel_parts[0] in SPECIAL_OUTPUT_DIRS:
            patient_id = rel_parts[1] if len(rel_parts) > 1 else ""
        else:
            patient_id = rel_parts[0] if rel_parts else ""

        stat = path.stat()
        records.append(
            {
                "relative_path": path.relative_to(output_root).as_posix(),
                "patient_id": patient_id,
                "kind": path.suffix.lower().lstrip(".") or "file",
                "size_kb": round(stat.st_size / 1024.0, 1),
                "modified": pd.to_datetime(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
            }
        )

    if not records:
        st.info("No files were found in the output folder.")
        return

    inventory = pd.DataFrame(records).sort_values(
        ["patient_id", "relative_path"],
        kind="stable",
        ignore_index=True,
    )
    kind_counts = inventory["kind"].value_counts().rename_axis("kind").reset_index(name="count")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        with st.container(border=True):
            st.markdown("**File counts by type**")
            fig = px.bar(kind_counts, x="kind", y="count", color="kind", title="Files by type")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with chart_cols[1]:
        with st.container(border=True):
            st.markdown("**Inventory preview**")
            st.dataframe(inventory, hide_index=True, use_container_width=True, height=320)


def _patient_badges(row: Optional[pd.Series]) -> None:
    """Render a compact row of status badges."""
    if row is None or row.empty:
        st.info("No patient status row was found for this patient.")
        return

    badges = [
        (
            "Telemetry",
            bool(row.get("analysis_telemetry_present", False)),
            "green",
            ":material/check_circle:",
        ),
        (
            "ECG file",
            bool(row.get("analysis_ecg_file_present", False)),
            "blue",
            ":material/monitor_heart:",
        ),
        (
            "ECG processed",
            bool(row.get("analysis_ecg_processed", False)),
            "green",
            ":material/analytics:",
        ),
        (
            "Tasks log",
            bool(row.get("analysis_tasks_log_found", False)),
            "orange",
            ":material/description:",
        ),
        (
            "Validation ok",
            bool(row.get("analysis_breathing_validation_ok", False)),
            "green",
            ":material/check:",
        ),
    ]
    cols = st.columns(len(badges))
    for col, (label, value, color, icon) in zip(cols, badges):
        with col:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                if value:
                    st.markdown(f"{icon} Available")
                else:
                    st.markdown(":material/warning: Missing")


def _patient_metric_cards(
    master_row: Optional[pd.Series],
    status_row: Optional[pd.Series],
    breathing_row: Optional[pd.Series],
) -> None:
    """Render patient-level KPI cards."""
    metrics = [
        ("Peak HR", _series_value(master_row, ["peak_HR", "peak_hr"]) if master_row is not None else None, "bpm"),
        ("Peak VO2", _series_value(master_row, ["peak_V'O2", "peak_VO2"]) if master_row is not None else None, ""),
        ("Mean HRV RMSSD", _series_value(status_row, ["mean_rmssd_ms"]) if status_row is not None else None, "ms"),
        (
            "Breathing reserve",
            _series_value(breathing_row, ["BR_pct"]) if breathing_row is not None else None,
            "%",
        ),
    ]

    metric_cols = st.columns(len(metrics))
    for col, (label, value, unit) in zip(metric_cols, metrics):
        with col:
            with st.container(border=True):
                text = _format_value(value, digits=1)
                if value is not None and pd.notna(value) and unit:
                    text = f"{text} {unit}"
                st.metric(label, text)

    if status_row is not None and not status_row.empty:
        validation_status = str(status_row.get("analysis_breathing_validation_status", "n/a"))
        validation_ok = bool(status_row.get("analysis_breathing_validation_ok", False))
        with st.container(border=True):
            st.markdown("**Breathing validation**")
            if validation_ok:
                st.markdown(f":material/check_circle: {validation_status}")
            else:
                st.markdown(f":material/warning: {validation_status}")


def _render_patient_explorer(
    output_root: Path,
    patient_id: str,
    bundle: dict[str, object],
    core: dict[str, object],
    selected_metric_labels: Sequence[str],
) -> None:
    """Render the per-patient dashboard."""
    telemetry = bundle["telemetry"]
    stage_summary = bundle["stage_summary"]
    validation = bundle["validation"]
    tasks_markers = bundle["tasks_markers"]
    ecg_timeseries = bundle["ecg_timeseries"]
    ecg_summary = bundle["ecg_summary"]
    image_paths = bundle["image_paths"]

    status = core["patient_status"]
    master = core["master_summary"]
    breathing_reserve = core["breathing_reserve"]

    status_row = None
    if not status.empty and "patient_id" in status.columns:
        match = status[status["patient_id"].astype(str) == patient_id]
        if not match.empty:
            status_row = match.iloc[0]

    master_row = None
    if not master.empty and "patient_id" in master.columns:
        match = master[master["patient_id"].astype(str) == patient_id]
        if not match.empty:
            master_row = match.iloc[0]

    breathing_row = None
    if not breathing_reserve.empty and "patient_id" in breathing_reserve.columns:
        match = breathing_reserve[breathing_reserve["patient_id"].astype(str) == patient_id]
        if not match.empty:
            breathing_row = match.iloc[0]

    st.subheader(f"Patient {patient_id}")
    _patient_metric_cards(master_row, status_row, breathing_row)
    _patient_badges(status_row)

    telemetry_tab, stage_tab, ecg_tab, validation_tab, files_tab = st.tabs(
        ["Telemetry", "Stage summary", "ECG", "Validation", "Files"]
    )

    with telemetry_tab:
        _render_telemetry_section(telemetry, selected_metric_labels)

    with stage_tab:
        _render_stage_summary_section(stage_summary, telemetry)

    with ecg_tab:
        _render_ecg_section(ecg_timeseries, ecg_summary)
        image_subset = [path for path in image_paths if path.name.startswith(patient_id) and "ecg" in path.name.lower()]
        if image_subset:
            with st.container(border=True):
                st.markdown("**ECG image gallery**")
                _render_image_grid(image_subset, columns=2)

    with validation_tab:
        _render_validation_section(validation)

    with files_tab:
        paths = _patient_paths(output_root, patient_id)
        patient_files: list[Path] = [
            paths["telemetry"],
            paths["stage_summary"],
            paths["validation"],
            paths["tasks_markers"],
            paths["metrics_plot"],
            paths["ve_vco2_plot"],
            paths["oxygen_pulse_plot"],
            paths["ventilatory_efficiency_plot"],
            paths["ecg_timeseries"],
            paths["ecg_summary"],
        ]
        patient_files = [path for path in patient_files if path.exists()]
        if not patient_files:
            st.info("No patient-specific files were found.")
        else:
            with st.container(border=True):
                st.markdown("**Patient file gallery**")
                _render_image_grid([path for path in patient_files if path.suffix.lower() == ".png"], columns=2)
            with st.expander("Patient file table", expanded=False):
                rows = []
                for path in patient_files:
                    stat = path.stat()
                    rows.append(
                        {
                            "file": path.name,
                            "relative_path": path.relative_to(output_root).as_posix(),
                            "kind": path.suffix.lower().lstrip(".") or "file",
                            "size_kb": round(stat.st_size / 1024.0, 1),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def main() -> int:
    """Run the Streamlit dashboard."""
    _configure_page()
    _apply_styles()
    st.title(APP_TITLE)
    st.caption("A viewer for the generated `output/` artifacts. The dashboard does not rerun analysis.")

    if "output_root" not in st.session_state:
        st.session_state.output_root = str(DEFAULT_OUTPUT_ROOT)
    if "selected_patient" not in st.session_state:
        st.session_state.selected_patient = ""
    if "telemetry_metrics" not in st.session_state:
        st.session_state.telemetry_metrics = [
            "Heart rate",
            "VO2",
            "VCO2",
            "Ventilation",
        ]

    with st.sidebar:
        st.header("Filters")
        output_root_input = st.text_input(
            "Output folder",
            value=st.session_state.output_root,
            key="output_root",
            help="Folder that contains the generated analysis results.",
        )
        if st.button("Refresh data", icon=":material/refresh:", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("### Display options")
        st.session_state.telemetry_metrics = st.multiselect(
            "Telemetry metrics",
            list(TELEMETRY_METRIC_CHOICES.keys()),
            default=st.session_state.telemetry_metrics,
        )
        only_ecg = st.checkbox("Show only patients with ECG files", value=False)
        only_processed = st.checkbox("Show only patients with ECG processing", value=False)
        only_validated = st.checkbox("Show only patients with breathing validation", value=False)
        only_tasks = st.checkbox("Show only patients with tasks logs", value=False)

    output_root = Path(output_root_input).expanduser()
    core = _load_core_tables(output_root.as_posix())
    patient_ids = core["patient_ids"]
    patient_status = core["patient_status"]
    filtered_patient_ids = _filtered_patient_ids(
        patient_ids,
        patient_status,
        only_ecg=only_ecg,
        only_processed=only_processed,
        only_validated=only_validated,
        only_tasks=only_tasks,
    )
    if not filtered_patient_ids:
        st.warning("No patients matched the active filters; showing all available patients.")
        filtered_patient_ids = list(patient_ids)

    if not filtered_patient_ids:
        st.error("No patient outputs were found. Run the pipeline first to generate the output folder.")
        files_tab = st.tabs(["Artifact browser"])[0]
        with files_tab:
            _render_file_inventory(output_root)
        return 0

    selected_index = 0
    if st.session_state.selected_patient in filtered_patient_ids:
        selected_index = filtered_patient_ids.index(st.session_state.selected_patient)
    selected_patient = st.sidebar.selectbox(
        "Patient",
        filtered_patient_ids,
        index=selected_index,
    )
    st.session_state.selected_patient = selected_patient

    bundle = _load_patient_bundle(output_root, selected_patient)

    overview_tab, patient_tab, files_tab = st.tabs(
        ["Cohort overview", "Patient explorer", "Artifact browser"]
    )

    with overview_tab:
        _render_cohort_overview(output_root, core, filtered_patient_ids)

    with patient_tab:
        _render_patient_explorer(
            output_root,
            selected_patient,
            bundle,
            core,
            st.session_state.telemetry_metrics,
        )

    with files_tab:
        _render_file_inventory(output_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
