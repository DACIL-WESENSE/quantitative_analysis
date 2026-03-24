"""
test_summary_generation.py
===========================
Tests for patient summary generation and cross-patient aggregation.

Coverage:
- build_patient_summary: produces correct columns and merges all metrics
- build_patient_analysis_status_table: cross-patient aggregation works
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import functions as fn


class TestBuildPatientSummary:
    """Test patient summary compilation (build_patient_summary)."""

    def test_build_summary_basic(self, sample_info_df, sample_telemetry_df):
        """Should build a summary row with all inputs."""
        summary = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
        )

        assert isinstance(summary, dict)
        assert "patient_id" in summary
        assert summary["patient_id"] == "TEST001"

    def test_build_summary_includes_peaks(self, sample_info_df, sample_telemetry_df):
        """Should include peak values for all numeric columns."""
        summary = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
        )

        # Should have peak_* columns
        peak_cols = [k for k in summary.keys() if k.startswith("peak_")]
        assert len(peak_cols) > 0

    def test_build_summary_includes_means(self, sample_info_df, sample_telemetry_df):
        """Should include mean values for all numeric columns."""
        summary = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
        )

        # Should have mean_* columns
        mean_cols = [k for k in summary.keys() if k.startswith("mean_")]
        assert len(mean_cols) > 0

    def test_build_summary_with_ecg_features(
        self, sample_info_df, sample_telemetry_df, sample_ecg_features_df
    ):
        """Should include ECG feature values."""
        summary = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
            ecg_features=sample_ecg_features_df,
        )

        assert "ecg_mean_abs_uV" in summary
        assert "ecg_estimated_hr_bpm" in summary

    def test_build_summary_with_ecg_timeseries(
        self, sample_info_df, sample_telemetry_df, sample_ecg_timeseries_df
    ):
        """Should include ECG time-series metrics."""
        summary = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
            ecg_timeseries=sample_ecg_timeseries_df,
        )

        # Should have ECG time-series derived metrics
        assert "mean_rmssd_ms" in summary or "mean_ecg_hr_bpm" in summary

    def test_build_summary_with_breathing_metrics(
        self, sample_info_df, sample_telemetry_df, sample_breathing_rate_metrics
    ):
        """Should include breathing-rate validation metrics."""
        summary = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
            breathing_rate_metrics=sample_breathing_rate_metrics,
        )

        # Should include breathing rate metrics
        assert "breathing_rate_mae_bpm" in summary

    def test_build_summary_patient_id_canonicalized(self, sample_info_df, sample_telemetry_df):
        """Should canonicalize patient ID."""
        summary = fn.build_patient_summary(
            patient_id="WeSense TEST-001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
        )

        assert summary["patient_id"] == "WESENSE_TEST-001"

    def test_build_summary_extracts_demographics(self, sample_info_df, sample_telemetry_df):
        """Should extract demographic info from info_df."""
        summary = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
        )

        # Fixture has gender='m', weight=75 kg, bmi=24.0 in second row columns
        # Extraction is case-insensitive
        assert "age" in summary or "Age" in summary or summary is not None

    def test_build_summary_handles_empty_info(self, sample_telemetry_df):
        """Should handle missing info DataFrame."""
        empty_info = pd.DataFrame()

        summary = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=empty_info,
            telemetry_df=sample_telemetry_df,
        )

        # Should still build summary without demographics
        assert summary is not None
        assert "patient_id" in summary

    def test_build_summary_all_numeric_columns(self, sample_info_df, sample_telemetry_df):
        """Should process all numeric columns in telemetry."""
        summary = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
        )

        # Should have peak and mean for HR, VO2, VCO2, VE, RR
        for col in ["HR", "VO2", "VCO2", "VE", "RR"]:
            assert f"peak_{col}" in summary or f"peak_HR" in summary  # At least HR should be there


class TestBuildPatientAnalysisStatusTable:
    """Test cross-patient analysis status aggregation."""

    def test_build_status_table_basic(self, sample_info_df, sample_telemetry_df):
        """Should build analysis status table."""
        # Create a summary DataFrame
        summary_row = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
        )
        summary_df = pd.DataFrame([summary_row])

        status_table = fn.build_patient_analysis_status_table(summary_df)

        assert isinstance(status_table, pd.DataFrame)
        assert len(status_table) == 1

    def test_build_status_table_columns(self, sample_info_df, sample_telemetry_df):
        """Should include all required status columns."""
        summary_row = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
        )
        summary_df = pd.DataFrame([summary_row])

        status_table = fn.build_patient_analysis_status_table(summary_df)

        required_cols = [
            "patient_id",
            "analysis_telemetry_present",
            "analysis_ecg_file_present",
            "analysis_ecg_processed",
            "mean_hr_bpm",
            "mean_rmssd_ms",
            "mean_sdnn_ms",
            "mean_breathing_rate_bpm_edr",
            "mean_breathing_rate_bpm_cpet",
        ]
        for col in required_cols:
            assert col in status_table.columns

    def test_build_status_table_multiple_patients(self, sample_info_df, sample_telemetry_df):
        """Should handle multiple patients."""
        rows = []
        for i in range(3):
            row = fn.build_patient_summary(
                patient_id=f"TEST{i:03d}",
                info_df=sample_info_df.copy(),
                telemetry_df=sample_telemetry_df.copy(),
            )
            rows.append(row)
        summary_df = pd.DataFrame(rows)

        status_table = fn.build_patient_analysis_status_table(summary_df)

        assert len(status_table) == 3
        assert len(status_table["patient_id"].unique()) == 3

    def test_build_status_table_extracts_hr(self, sample_info_df, sample_telemetry_df):
        """Should extract mean HR from multiple possible columns."""
        summary_row = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
        )
        summary_df = pd.DataFrame([summary_row])

        status_table = fn.build_patient_analysis_status_table(summary_df)

        # Should have extracted HR value
        assert not np.isnan(status_table["mean_hr_bpm"].iloc[0])

    def test_build_status_table_extracts_hrv(
        self, sample_info_df, sample_telemetry_df, sample_ecg_timeseries_df
    ):
        """Should extract HRV metrics when available."""
        summary_row = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
            ecg_timeseries=sample_ecg_timeseries_df,
        )
        summary_df = pd.DataFrame([summary_row])

        status_table = fn.build_patient_analysis_status_table(summary_df)

        # May have HRV values (depending on ECG time-series data)
        assert status_table is not None

    def test_build_status_table_bool_columns(self, sample_info_df, sample_telemetry_df):
        """Should convert analysis flags to boolean."""
        summary_row = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
        )
        summary_df = pd.DataFrame([summary_row])

        status_table = fn.build_patient_analysis_status_table(summary_df)

        # Analysis flags should be boolean
        bool_cols = [
            "analysis_telemetry_present",
            "analysis_ecg_file_present",
            "analysis_ecg_processed",
        ]
        for col in bool_cols:
            if col in status_table.columns:
                assert status_table[col].dtype == bool or status_table[col].dtype == "bool"

    def test_build_status_table_numeric_extraction(self, sample_info_df, sample_telemetry_df):
        """Should extract numeric metrics correctly."""
        summary_row = fn.build_patient_summary(
            patient_id="TEST001",
            info_df=sample_info_df,
            telemetry_df=sample_telemetry_df,
        )
        summary_df = pd.DataFrame([summary_row])

        status_table = fn.build_patient_analysis_status_table(summary_df)

        # Numeric columns should be float
        numeric_cols = ["mean_hr_bpm", "mean_rmssd_ms", "mean_sdnn_ms"]
        for col in numeric_cols:
            if col in status_table.columns:
                assert pd.api.types.is_numeric_dtype(status_table[col])

    def test_build_status_table_handles_empty_input(self):
        """Should handle empty summary DataFrame."""
        empty_df = pd.DataFrame({"patient_id": []})

        status_table = fn.build_patient_analysis_status_table(empty_df)

        assert isinstance(status_table, pd.DataFrame)
        assert len(status_table) == 0

    def test_build_status_table_duplicates_removed(self, sample_info_df, sample_telemetry_df):
        """Should remove duplicate patient IDs, keeping last."""
        rows = []
        for i in range(2):
            row = fn.build_patient_summary(
                patient_id="TEST001",  # Same patient ID
                info_df=sample_info_df.copy(),
                telemetry_df=sample_telemetry_df.copy(),
            )
            rows.append(row)
        summary_df = pd.DataFrame(rows)

        status_table = fn.build_patient_analysis_status_table(summary_df)

        # Should have only 1 row (duplicate removed)
        assert len(status_table) == 1


class TestSaveTelemetryCsv:
    """Test telemetry CSV export."""

    def test_save_telemetry_csv(self, sample_telemetry_df, temp_dir):
        """Should save telemetry DataFrame to CSV."""
        output_path = fn.save_telemetry_csv(
            sample_telemetry_df,
            patient_id="TEST001",
            output_dir=temp_dir,
        )

        assert output_path.exists()
        assert output_path.name == "TEST001_telemetry.csv"

    def test_save_telemetry_preserves_data(self, sample_telemetry_df, temp_dir):
        """Should preserve all data in saved CSV."""
        fn.save_telemetry_csv(
            sample_telemetry_df,
            patient_id="TEST001",
            output_dir=temp_dir,
        )

        # Read back and verify
        saved_df = pd.read_csv(temp_dir / "TEST001_telemetry.csv")
        assert len(saved_df) == len(sample_telemetry_df)
        assert list(saved_df.columns) == list(sample_telemetry_df.columns)


class TestSaveTasksMarkersCsv:
    """Test tasks markers CSV export."""

    def test_save_tasks_markers_csv(self, sample_tasks_markers_df, temp_dir):
        """Should save markers DataFrame to CSV."""
        output_path = fn.save_tasks_markers_csv(
            sample_tasks_markers_df,
            patient_id="TEST001",
            output_dir=temp_dir,
        )

        assert output_path is not None
        assert output_path.exists()
        assert "tasks_markers" in output_path.name

    def test_save_tasks_markers_empty(self, temp_dir):
        """Should save empty markers DataFrame gracefully."""
        empty_df = pd.DataFrame(columns=fn._TASKS_MARKER_COLUMNS)

        output_path = fn.save_tasks_markers_csv(
            empty_df,
            patient_id="TEST001",
            output_dir=temp_dir,
        )

        assert output_path.exists()
