"""
Streamlit application for the DACIL-WESENSE quantitative analysis pipeline.

Provides interactive dashboards for:
- Pipeline status overview
- Data flow visualization
- Task/stage explorer
- Processing control
- Results inspection
- Configuration management
- Logs viewing
"""

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import functions as fn
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

# ============================================================================
# Streamlit page configuration
# ============================================================================

st.set_page_config(
    page_title="DACIL-WESENSE Analysis Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# Session state initialization
# ============================================================================


def init_session_state():
    """Initialize session state variables."""
    if "data_root" not in st.session_state:
        st.session_state.data_root = "data"
    if "output_root" not in st.session_state:
        st.session_state.output_root = "output"
    if "selected_patients" not in st.session_state:
        st.session_state.selected_patients = []
    if "pipeline_config" not in st.session_state:
        st.session_state.pipeline_config = {
            "include_ecg": True,
            "parallel_processing": False,
            "num_workers": 1,
        }
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()


init_session_state()

# ============================================================================
# Utility functions
# ============================================================================


def get_pipeline_status() -> Dict:
    """Get current pipeline status and statistics."""
    data_root = Path(st.session_state.data_root)
    output_root = Path(st.session_state.output_root)

    status = {
        "data_root_exists": data_root.exists(),
        "output_root_exists": output_root.exists(),
        "patient_count": 0,
        "csv_count": 0,
        "bdf_count": 0,
        "processed_count": 0,
        "last_run": None,
    }

    if data_root.exists():
        try:
            patient_folders = fn.discover_patient_folders(str(data_root))
            status["patient_count"] = len(patient_folders)

            for folder in patient_folders:
                csv_file = fn.find_csv_file(folder)
                if csv_file:
                    status["csv_count"] += 1

                bdf_l1, bdf_l2 = fn.find_bdf_files(folder)
                if bdf_l1 or bdf_l2:
                    status["bdf_count"] += 1

                # Check for processed output
                output_patient = output_root / folder.name
                if output_patient.exists() and any(output_patient.glob("*.csv")):
                    status["processed_count"] += 1
        except Exception as e:
            st.warning(f"Error scanning data directory: {e}")

    # Check last run time from log file
    log_file = Path("pipeline.log")
    if log_file.exists():
        status["last_run"] = datetime.fromtimestamp(log_file.stat().st_mtime)

    return status


def get_patient_details(patient_folder: Path) -> Dict:
    """Get detailed information about a patient folder."""
    details = {
        "folder_name": patient_folder.name,
        "csv_file": None,
        "bdf_l1": None,
        "bdf_l2": None,
        "tasks_log": None,
        "output_dir": None,
        "processed": False,
        "csv_rows": 0,
        "ecg_samples": 0,
    }

    # Find CSV file
    csv_file = fn.find_csv_file(patient_folder)
    if csv_file:
        details["csv_file"] = csv_file
        try:
            df, _ = fn.load_telemetry(csv_file)
            details["csv_rows"] = len(df)
        except Exception as e:
            st.warning(f"Error loading CSV for {patient_folder.name}: {e}")

    # Find BDF files
    bdf_l1, bdf_l2 = fn.find_bdf_files(patient_folder)
    details["bdf_l1"] = bdf_l1
    details["bdf_l2"] = bdf_l2

    # Find tasks log
    tasks_log = fn.find_tasks_log_file(patient_folder)
    if tasks_log:
        details["tasks_log"] = tasks_log

    # Check output directory
    output_root = Path(st.session_state.output_root)
    output_patient = output_root / patient_folder.name
    if output_patient.exists():
        details["output_dir"] = output_patient
        if any(output_patient.glob("*.csv")):
            details["processed"] = True

    return details


def read_log_file(log_path: Path, num_lines: int = 100) -> str:
    """Read last N lines from a log file."""
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
            return "".join(lines[-num_lines:]) if lines else "Log file is empty."
    except Exception as e:
        return f"Error reading log file: {e}"


def create_pipeline_diagram() -> go.Figure:
    """Create an interactive pipeline flow diagram."""
    fig = go.Figure()

    # Define stages
    stages = [
        ("Data\nLoading", 0, 0.5),
        ("Preprocessing", 1, 0.5),
        ("ECG\nProcessing", 2, 0.5),
        ("Alignment", 3, 0.5),
        ("Output\nGeneration", 4, 0.5),
    ]

    # Add nodes
    for i, (stage, x, y) in enumerate(stages):
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(
                    size=40,
                    color=["#667eea", "#764ba2", "#f093fb", "#4facfe", "#00f2fe"][i],
                ),
                text=stage,
                textposition="middle center",
                textfont=dict(color="white", size=12, family="Arial Black"),
                hovertemplate="<b>%{text}</b><extra></extra>",
                showlegend=False,
            )
        )

    # Add arrows between stages
    for i in range(len(stages) - 1):
        x0, y0 = stages[i][1], stages[i][2]
        x1, y1 = stages[i + 1][1], stages[i + 1][2]
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#cccccc",
        )

    fig.update_layout(
        title="Pipeline Data Flow",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(240, 240, 240, 1)",
        height=300,
    )

    return fig


# ============================================================================
# Main app layout
# ============================================================================


def main():
    """Main Streamlit application."""
    # Sidebar configuration
    with st.sidebar:
        st.title("️ Configuration")

        # Data paths
        st.session_state.data_root = st.text_input(
            "Data Root Directory",
            value=st.session_state.data_root,
            help="Root folder containing patient subfolders",
        )
        st.session_state.output_root = st.text_input(
            "Output Root Directory",
            value=st.session_state.output_root,
            help="Directory for generated outputs",
        )

        st.divider()

        # Pipeline options
        st.subheader("Pipeline Options")
        st.session_state.pipeline_config["include_ecg"] = st.checkbox(
            "Include ECG Processing",
            value=st.session_state.pipeline_config["include_ecg"],
        )
        st.session_state.pipeline_config["parallel_processing"] = st.checkbox(
            "Parallel Processing",
            value=st.session_state.pipeline_config["parallel_processing"],
        )
        if st.session_state.pipeline_config["parallel_processing"]:
            st.session_state.pipeline_config["num_workers"] = st.slider(
                "Number of Workers", 1, 8, st.session_state.pipeline_config["num_workers"]
            )

        st.divider()

        # Refresh button
        if st.button(" Refresh All Data"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            " Dashboard",
            " Data Flow",
            " Task Explorer",
            "️ Processing",
            " Results",
            "️ Config Panel",
            " Logs",
        ]
    )

    # ========================================================================
    # Tab 1: Dashboard Overview
    # ========================================================================
    with tab1:
        st.title(" Dashboard Overview")

        status = get_pipeline_status()

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                " Patients",
                status["patient_count"],
                help="Total patient folders discovered",
            )

        with col2:
            st.metric(
                " Telemetry Files",
                status["csv_count"],
                help="CSV telemetry files found",
            )

        with col3:
            st.metric(
                " ECG Files",
                status["bdf_count"],
                help="BDF ECG files found",
            )

        with col4:
            st.metric(
                " Processed",
                status["processed_count"],
                help="Patients with completed analysis",
            )

        st.divider()

        # Status indicators
        col1, col2, col3 = st.columns(3)

        with col1:
            data_root_status = " Exists" if status["data_root_exists"] else " Missing"
            st.write(f"**Data Root**: {data_root_status}")

        with col2:
            output_root_status = " Exists" if status["output_root_exists"] else " Missing"
            st.write(f"**Output Root**: {output_root_status}")

        with col3:
            if status["last_run"]:
                st.write(f"**Last Run**: {status['last_run'].strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.write("**Last Run**: Never")

        st.divider()

        # Analysis completion progress
        if status["patient_count"] > 0:
            progress = status["processed_count"] / status["patient_count"]
            st.progress(progress, text=f"Analysis Progress: {progress:.1%}")

    # ========================================================================
    # Tab 2: Data Flow Visualization
    # ========================================================================
    with tab2:
        st.title(" Data Flow Visualization")
        st.write("Interactive pipeline architecture diagram showing data flow through stages.")

        fig = create_pipeline_diagram()
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Pipeline Stages")
        stages_info = {
            " Data Loading": "Discover patient folders, load telemetry CSV and ECG BDF files.",
            " Preprocessing": "Parse metadata, normalize column names, align timestamps.",
            " ECG Processing": "Extract ECG features, compute HRV metrics, breathing rate estimation.",
            " Alignment": "Synchronize ECG with telemetry data, validate breathing rates.",
            " Output Generation": "Export cleaned telemetry, compute stage summaries, generate plots.",
        }

        for stage, description in stages_info.items():
            with st.expander(stage):
                st.write(description)

    # ========================================================================
    # Tab 3: Task/Stage Explorer
    # ========================================================================
    with tab3:
        st.title(" Task/Stage Explorer")

        data_root = Path(st.session_state.data_root)

        if not data_root.exists():
            st.error(f"Data root directory not found: {data_root}")
            st.stop()

        patient_folders = fn.discover_patient_folders(str(data_root))

        if not patient_folders:
            st.warning("No patient folders found.")
            st.stop()

        # Patient selection
        patient_names = [f.name for f in patient_folders]
        selected_patient = st.selectbox(
            "Select Patient", patient_names, help="Browse patient details"
        )

        patient_folder = data_root / selected_patient
        details = get_patient_details(patient_folder)

        # Display patient details
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(" Files Found")
            st.write(f"**Folder**: {details['folder_name']}")

            if details["csv_file"]:
                st.success(f" Telemetry: {details['csv_file'].name}")
                st.caption(f"Rows: {details['csv_rows']}")
            else:
                st.error(" No telemetry CSV found")

            if details["bdf_l1"]:
                st.success(f" ECG L1: {details['bdf_l1'].name}")
            if details["bdf_l2"]:
                st.success(f" ECG L2: {details['bdf_l2'].name}")
            if not details["bdf_l1"] and not details["bdf_l2"]:
                st.info("ℹ️ No ECG files found")

            if details["tasks_log"]:
                st.success(f" Tasks Log: {details['tasks_log'].name}")
            else:
                st.info("ℹ️ No tasks.log found")

        with col2:
            st.subheader(" Processing Status")

            if details["processed"]:
                st.success(" **Processed** - Output files generated")
            else:
                st.warning("⏳ **Pending** - Not yet processed")

            if details["output_dir"]:
                output_files = list(details["output_dir"].glob("*.csv"))
                if output_files:
                    st.write(f"**Output Files**: {len(output_files)}")
                    for f in output_files:
                        st.caption(f"• {f.name}")

        # Load and preview CSV data
        if details["csv_file"]:
            st.divider()
            if st.checkbox("Preview Telemetry Data", key=f"preview_{selected_patient}"):
                try:
                    df, meta = fn.load_telemetry(details["csv_file"])
                    st.write(f"**Rows**: {len(df)}, **Columns**: {len(df.columns)}")
                    st.dataframe(df.head(20), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading telemetry: {e}")

    # ========================================================================
    # Tab 4: Processing Control
    # ========================================================================
    with tab4:
        st.title("️ Processing Control")

        data_root = Path(st.session_state.data_root)

        if not data_root.exists():
            st.error(f"Data root directory not found: {data_root}")
            st.stop()

        patient_folders = fn.discover_patient_folders(str(data_root))

        if not patient_folders:
            st.warning("No patient folders found.")
            st.stop()

        patient_names = [f.name for f in patient_folders]

        st.subheader("Select Patients for Analysis")

        # Select all / deselect all buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button(" Select All"):
                st.session_state.selected_patients = patient_names
                st.rerun()
        with col2:
            if st.button(" Clear All"):
                st.session_state.selected_patients = []
                st.rerun()

        # Multi-select for patients
        st.session_state.selected_patients = st.multiselect(
            "Patients to Process",
            patient_names,
            default=st.session_state.selected_patients,
            help="Select which patients to include in the analysis",
        )

        st.divider()

        st.subheader("Pipeline Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Include ECG Processing**")
            st.info(
                f"ECG: {'Enabled ' if st.session_state.pipeline_config['include_ecg'] else 'Disabled '}"
            )

        with col2:
            st.write("**Parallel Processing**")
            st.info(
                f"Parallel: {'Enabled ' if st.session_state.pipeline_config['parallel_processing'] else 'Disabled '}"
            )
            if st.session_state.pipeline_config["parallel_processing"]:
                st.caption(f"Workers: {st.session_state.pipeline_config['num_workers']}")

        st.divider()

        # Processing summary
        if st.session_state.selected_patients:
            st.subheader("Analysis Summary")
            summary_data = {
                "Patient": st.session_state.selected_patients,
                "Status": ["Pending"] * len(st.session_state.selected_patients),
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

        st.divider()

        # Process button
        col1, col2 = st.columns([3, 1])

        with col1:
            st.info(
                f"Selected {len(st.session_state.selected_patients)} patient(s) for analysis."
            )

        with col2:
            if st.button("️ Start Processing", type="primary"):
                if not st.session_state.selected_patients:
                    st.error("Please select at least one patient.")
                else:
                    st.info("Processing would run the pipeline here. In production, this would:")
                    st.write(
                        """
                        1. Iterate through selected patients
                        2. Load telemetry and ECG data
                        3. Perform alignment and cleaning
                        4. Generate outputs
                        5. Update status in results tab
                        """
                    )

    # ========================================================================
    # Tab 5: Results Inspector
    # ========================================================================
    with tab5:
        st.title(" Results Inspector")

        output_root = Path(st.session_state.output_root)

        if not output_root.exists():
            st.warning(f"Output directory not found: {output_root}")
            st.stop()

        # Find all patient output directories
        patient_output_dirs = sorted([d for d in output_root.iterdir() if d.is_dir()])

        if not patient_output_dirs:
            st.info("No results available yet. Run the pipeline first.")
            st.stop()

        # Patient selector
        patient_names = [d.name for d in patient_output_dirs]
        selected_patient = st.selectbox(
            "Select Patient Results", patient_names, help="View analysis results for a patient"
        )

        patient_output = output_root / selected_patient
        csv_files = sorted(patient_output.glob("*.csv"))

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader(" Available Files")
            if csv_files:
                st.write(f"Found {len(csv_files)} output files:")
                for f in csv_files:
                    st.caption(f"• {f.name}")
            else:
                st.info("No output files found for this patient.")

        with col2:
            st.subheader(" File Preview")
            if csv_files:
                selected_file = st.selectbox(
                    "Select File to View",
                    [f.name for f in csv_files],
                    key=f"results_file_{selected_patient}",
                )

                try:
                    file_path = patient_output / selected_file
                    df = pd.read_csv(file_path)

                    st.write(f"**Rows**: {len(df)}, **Columns**: {len(df.columns)}")
                    st.dataframe(df, use_container_width=True)

                    # Download button
                    csv_buffer = df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_buffer,
                        file_name=selected_file,
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        st.divider()

        # Cross-patient summary
        st.subheader(" Cross-Patient Summary")

        try:
            summary_rows = []
            for patient_dir in patient_output_dirs:
                summary_file = patient_dir / "summary.csv"
                if summary_file.exists():
                    df = pd.read_csv(summary_file)
                    summary_rows.append(df)

            if summary_rows:
                combined_summary = pd.concat(summary_rows, ignore_index=True)
                st.dataframe(combined_summary, use_container_width=True)

                # Download combined summary
                csv_buffer = combined_summary.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Combined Summary",
                    data=csv_buffer,
                    file_name="combined_summary.csv",
                    mime="text/csv",
                )
            else:
                st.info("No summary files found across patient outputs.")
        except Exception as e:
            st.warning(f"Error loading cross-patient summary: {e}")

    # ========================================================================
    # Tab 6: Configuration Panel
    # ========================================================================
    with tab6:
        st.title("️ Configuration Panel")

        st.subheader("Pipeline Paths")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Data Root**: `{st.session_state.data_root}`")
            if Path(st.session_state.data_root).exists():
                st.success(" Directory exists")
            else:
                st.error(" Directory not found")

        with col2:
            st.write(f"**Output Root**: `{st.session_state.output_root}`")
            if Path(st.session_state.output_root).exists():
                st.success(" Directory exists")
            else:
                st.info("ℹ️ Will be created during processing")

        st.divider()

        st.subheader("Stage Labels")
        st.info(f"**Stages**: {', '.join(fn.STAGE_ORDER)}")
        st.caption("Edit stage order in functions.py (STAGE_ORDER)")

        st.divider()

        st.subheader("COPD Risk Thresholds")

        # Read thresholds from functions.py
        thresholds_info = {
            "Peak VO2": "≤ 20.0 mL/min/kg → elevated risk",
            "VE/VCO2 Slope": "> 34.0 → ventilatory inefficiency",
            "SpO2 Nadir": "< 95.0% → desaturation",
            "SpO2 Drop": "≥ 4.0% → clinically significant",
            "Peak Breathing Freq": "> 40 breaths/min → hyperventilation",
            "Peak RER": "< 1.0 → effort-limited test",
            "Peak O2 Pulse": "< 10.0 mL/beat → reduced cardiac output",
        }

        for metric, threshold in thresholds_info.items():
            st.write(f"**{metric}**: {threshold}")

        st.caption("Edit thresholds in functions.py (_COPD_THRESHOLDS)")

        st.divider()

        st.subheader("Telemetry Column Candidates")

        columns_info = {
            "Time": fn.TELEMETRY_TIME_CANDIDATES[:3],
            "Heart Rate": fn.VYNTUS_HR_CANDIDATES[:3] if hasattr(fn, "VYNTUS_HR_CANDIDATES") else ["HR"],
            "Breathing Rate": fn.VYNTUS_BREATHING_RATE_CANDIDATES[:3],
        }

        for col_type, candidates in columns_info.items():
            st.write(f"**{col_type}**: {', '.join(candidates)}")

    # ========================================================================
    # Tab 7: Logs Viewer
    # ========================================================================
    with tab7:
        st.title(" Logs Viewer")

        log_file = Path("pipeline.log")

        if not log_file.exists():
            st.info("No pipeline.log file found yet.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(" Create Sample Log"):
                    st.info("Log file will be created during pipeline execution.")
        else:
            st.success(" pipeline.log found")

            # Log controls
            col1, col2 = st.columns([1, 1])

            with col1:
                num_lines = st.slider("Lines to Display", 10, 1000, 100)

            with col2:
                if st.button(" Refresh Logs"):
                    st.rerun()

            st.divider()

            # Display logs
            log_content = read_log_file(log_file, num_lines)

            st.subheader(f"Last {num_lines} lines from pipeline.log")

            # Use a code block for better formatting
            st.code(log_content, language="log")

            # Search logs
            st.divider()

            search_term = st.text_input(" Search logs", help="Find specific text in logs")

            if search_term:
                log_lines = log_content.split("\n")
                matching_lines = [line for line in log_lines if search_term.lower() in line.lower()]

                if matching_lines:
                    st.write(f"Found {len(matching_lines)} matching lines:")
                    st.code("\n".join(matching_lines), language="log")
                else:
                    st.info(f"No matches found for '{search_term}'")

            # Log file info
            st.divider()
            st.subheader(" Log File Info")

            file_stat = log_file.stat()
            col1, col2, col3 = st.columns(3)

            with col1:
                size_kb = file_stat.st_size / 1024
                st.metric("File Size", f"{size_kb:.1f} KB")

            with col2:
                mod_time = datetime.fromtimestamp(file_stat.st_mtime)
                st.metric("Last Modified", mod_time.strftime("%Y-%m-%d %H:%M"))

            with col3:
                line_count = len(log_content.split("\n"))
                st.metric("Total Lines", line_count)

            # Download button
            with open(log_file, "r") as f:
                log_text = f.read()

            st.download_button(
                label="⬇️ Download Log File",
                data=log_text,
                file_name="pipeline.log",
                mime="text/plain",
            )

    # ========================================================================
    # Footer
    # ========================================================================
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

    with col2:
        st.caption("DACIL-WESENSE Quantitative Analysis Pipeline")

    with col3:
        st.caption(
            """
             [README](README.md) | 
             [Documentation](STREAMLIT_USAGE.md)
            """
        )


if __name__ == "__main__":
    main()
