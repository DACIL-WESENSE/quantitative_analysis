"""
test_breathing_rate_validation.py
==================================
Tests for breathing-rate validation and ECG-Vyntus comparison.

Coverage:
- align_breathing_rate_series: merges calculated vs measured series
- compute_breathing_rate_comparison: calculates agreement metrics (MAE, RMSE, bias, corr)
- evaluate_breathing_rate_validation: full validation pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import functions as fn


class TestAlignBreathingRateSeries:
    """Test breathing-rate alignment (align_breathing_rate_series)."""

    def test_align_basic(self, sample_breathing_rate_aligned_df):
        """Should align two breathing-rate time series by nearest timestamp."""
        calc_df = pd.DataFrame({
            "time_s": [10.0, 20.0, 30.0, 40.0, 50.0],
            "calculated_br_bpm": [16.5, 17.2, 18.0, 19.5, 16.8],
        })
        meas_df = pd.DataFrame({
            "time_s": [10.2, 20.1, 29.8, 40.3, 50.1],
            "measured_br_bpm": [16.0, 17.0, 18.5, 19.0, 17.0],
        })

        aligned = fn.align_breathing_rate_series(calc_df, meas_df, tolerance_s=20.0)

        assert len(aligned) == 5
        assert "time_s_calc" in aligned.columns
        assert "time_s_measured" in aligned.columns
        assert "time_delta_s" in aligned.columns

    def test_align_calculates_time_delta(self):
        """Should calculate time difference between aligned samples."""
        calc_df = pd.DataFrame({
            "time_s": [10.0],
            "calculated_br_bpm": [16.5],
        })
        meas_df = pd.DataFrame({
            "time_s": [10.5],
            "measured_br_bpm": [16.0],
        })

        aligned = fn.align_breathing_rate_series(calc_df, meas_df, tolerance_s=20.0)

        assert aligned["time_delta_s"].iloc[0] == 0.5

    def test_align_respects_tolerance(self):
        """Should only align samples within tolerance."""
        calc_df = pd.DataFrame({
            "time_s": [10.0, 100.0],
            "calculated_br_bpm": [16.5, 18.0],
        })
        meas_df = pd.DataFrame({
            "time_s": [10.1, 200.0],  # Second pair exceeds tolerance
            "measured_br_bpm": [16.0, 17.0],
        })

        aligned = fn.align_breathing_rate_series(calc_df, meas_df, tolerance_s=1.0)

        # Only first pair should align within 1.0s tolerance
        assert len(aligned) == 1

    def test_align_empty_calculated(self):
        """Should return empty DataFrame when calculated is empty."""
        calc_df = pd.DataFrame(columns=["time_s", "calculated_br_bpm"])
        meas_df = pd.DataFrame({
            "time_s": [10.0],
            "measured_br_bpm": [16.0],
        })

        aligned = fn.align_breathing_rate_series(calc_df, meas_df)

        assert len(aligned) == 0

    def test_align_empty_measured(self):
        """Should return empty DataFrame when measured is empty."""
        calc_df = pd.DataFrame({
            "time_s": [10.0],
            "calculated_br_bpm": [16.5],
        })
        meas_df = pd.DataFrame(columns=["time_s", "measured_br_bpm"])

        aligned = fn.align_breathing_rate_series(calc_df, meas_df)

        assert len(aligned) == 0

    def test_align_sorts_by_calculated_time(self):
        """Should sort output by calculated time."""
        calc_df = pd.DataFrame({
            "time_s": [50.0, 10.0, 30.0],
            "calculated_br_bpm": [16.8, 16.5, 18.0],
        })
        meas_df = pd.DataFrame({
            "time_s": [10.1, 50.1, 30.1],
            "measured_br_bpm": [16.0, 16.7, 18.1],
        })

        aligned = fn.align_breathing_rate_series(calc_df, meas_df, tolerance_s=20.0)

        # Should be sorted by time_s_calc
        calc_times = aligned["time_s_calc"].tolist()
        assert calc_times == sorted(calc_times)


class TestComputeBreathingRateComparison:
    """Test breathing-rate comparison metrics (compute_breathing_rate_comparison)."""

    def test_compute_metrics_basic(self, sample_breathing_rate_aligned_df):
        """Should compute agreement metrics."""
        metrics = fn.compute_breathing_rate_comparison(sample_breathing_rate_aligned_df)

        assert "breathing_rate_n_aligned" in metrics
        assert "breathing_rate_mean_calculated_bpm" in metrics
        assert "breathing_rate_mean_measured_bpm" in metrics
        assert "breathing_rate_bias_bpm" in metrics
        assert "breathing_rate_mae_bpm" in metrics
        assert "breathing_rate_rmse_bpm" in metrics

    def test_compute_n_aligned(self, sample_breathing_rate_aligned_df):
        """Should compute correct sample count."""
        metrics = fn.compute_breathing_rate_comparison(sample_breathing_rate_aligned_df)

        assert metrics["breathing_rate_n_aligned"] == 5.0

    def test_compute_mean_values(self):
        """Should compute correct mean values."""
        df = pd.DataFrame({
            "calculated_br_bpm": [16.0, 18.0],
            "measured_br_bpm": [16.0, 18.0],
        })

        metrics = fn.compute_breathing_rate_comparison(df)

        assert metrics["breathing_rate_mean_calculated_bpm"] == 17.0
        assert metrics["breathing_rate_mean_measured_bpm"] == 17.0

    def test_compute_bias(self):
        """Should compute bias (mean error)."""
        df = pd.DataFrame({
            "calculated_br_bpm": [17.0, 19.0],  # consistently higher
            "measured_br_bpm": [16.0, 18.0],
        })

        metrics = fn.compute_breathing_rate_comparison(df)

        # Expected bias: (1.0 + 1.0) / 2 = 1.0
        assert metrics["breathing_rate_bias_bpm"] == 1.0

    def test_compute_mae(self):
        """Should compute Mean Absolute Error."""
        df = pd.DataFrame({
            "calculated_br_bpm": [16.0, 18.0],
            "measured_br_bpm": [17.0, 19.0],  # all errors are 1.0
        })

        metrics = fn.compute_breathing_rate_comparison(df)

        assert metrics["breathing_rate_mae_bpm"] == 1.0

    def test_compute_rmse(self):
        """Should compute Root Mean Square Error."""
        df = pd.DataFrame({
            "calculated_br_bpm": [16.0, 18.0],
            "measured_br_bpm": [17.0, 19.0],  # errors: [1.0, 1.0]
        })

        metrics = fn.compute_breathing_rate_comparison(df)

        # RMSE = sqrt(mean([1.0^2, 1.0^2])) = 1.0
        assert metrics["breathing_rate_rmse_bpm"] == 1.0

    def test_compute_correlation(self):
        """Should compute Pearson correlation."""
        df = pd.DataFrame({
            "calculated_br_bpm": [16.0, 17.0, 18.0, 19.0, 20.0],
            "measured_br_bpm": [16.0, 17.0, 18.0, 19.0, 20.0],  # perfect match
        })

        metrics = fn.compute_breathing_rate_comparison(df)

        # Perfect correlation (or very close to 1.0)
        assert metrics["breathing_rate_corr"] >= 0.99

    def test_compute_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        df = pd.DataFrame(columns=["calculated_br_bpm", "measured_br_bpm"])

        metrics = fn.compute_breathing_rate_comparison(df)

        assert metrics["breathing_rate_n_aligned"] == 0.0
        assert np.isnan(metrics["breathing_rate_mae_bpm"])

    def test_compute_single_sample(self):
        """Should handle single sample (no correlation possible)."""
        df = pd.DataFrame({
            "calculated_br_bpm": [17.0],
            "measured_br_bpm": [16.0],
        })

        metrics = fn.compute_breathing_rate_comparison(df)

        assert metrics["breathing_rate_n_aligned"] == 1.0
        assert metrics["breathing_rate_mae_bpm"] == 1.0
        # Correlation needs at least 2 unique values
        assert np.isnan(metrics.get("breathing_rate_corr", float("nan")))


class TestPrepareBreathingSeries:
    """Test breathing series preparation helpers."""

    def test_prepare_vyntus_breathing_series(self, sample_telemetry_df):
        """Should extract measured Vyntus breathing rate."""
        col, series_df = fn._prepare_vyntus_breathing_series(sample_telemetry_df, "TEST001")

        # RR is the breathing rate column in the fixture
        assert col in sample_telemetry_df.columns
        assert "time_s" in series_df.columns
        assert "measured_br_bpm" in series_df.columns

    def test_prepare_ecg_breathing_series(self, sample_ecg_timeseries_df):
        """Should extract ECG-derived breathing rate."""
        series_df = fn._prepare_ecg_breathing_series(sample_ecg_timeseries_df, "TEST001")

        assert "time_s" in series_df.columns
        assert "calculated_br_bpm" in series_df.columns
        assert len(series_df) > 0

    def test_prepare_vyntus_missing_column(self, sample_telemetry_df):
        """Should return empty DataFrame when breathing rate column missing."""
        # Remove all breathing rate candidates
        bad_df = sample_telemetry_df[["Tijd", "HR", "Stage"]].copy()
        
        col, series_df = fn._prepare_vyntus_breathing_series(bad_df, "TEST001")

        assert col is None
        assert len(series_df) == 0

    def test_prepare_ecg_missing_column(self):
        """Should return empty DataFrame when required columns missing."""
        bad_df = pd.DataFrame({
            "time_s": [1.0, 2.0, 3.0],
        })
        
        series_df = fn._prepare_ecg_breathing_series(bad_df, "TEST001")

        assert len(series_df) == 0


class TestEvaluateBreathingRateValidation:
    """Test full breathing-rate validation pipeline."""

    def test_evaluate_basic(self, sample_telemetry_df, sample_ecg_timeseries_df):
        """Should run full validation pipeline."""
        metrics, aligned = fn.evaluate_breathing_rate_validation(
            sample_telemetry_df,
            sample_ecg_timeseries_df,
            patient_id="TEST001",
        )

        assert "breathing_rate_validation_status" in metrics
        assert metrics["breathing_rate_validation_status"] in [
            "ok", "missing_data", "no_aligned_samples", "skipped_ecg"
        ]

    def test_evaluate_returns_metrics_dict(self, sample_telemetry_df, sample_ecg_timeseries_df):
        """Should return metrics dictionary with expected keys."""
        metrics, _ = fn.evaluate_breathing_rate_validation(
            sample_telemetry_df,
            sample_ecg_timeseries_df,
            patient_id="TEST001",
        )

        expected_keys = [
            "breathing_rate_calc_source",
            "breathing_rate_measured_source_column",
            "breathing_rate_validation_status",
        ]
        for key in expected_keys:
            assert key in metrics

    def test_evaluate_returns_aligned_dataframe(self, sample_telemetry_df, sample_ecg_timeseries_df):
        """Should return aligned comparison DataFrame."""
        _, aligned = fn.evaluate_breathing_rate_validation(
            sample_telemetry_df,
            sample_ecg_timeseries_df,
            patient_id="TEST001",
        )

        # Should be DataFrame (possibly empty)
        assert isinstance(aligned, pd.DataFrame)

    def test_evaluate_missing_ecg(self, sample_telemetry_df):
        """Should handle missing ECG gracefully."""
        empty_ecg = pd.DataFrame()

        metrics, _ = fn.evaluate_breathing_rate_validation(
            sample_telemetry_df,
            empty_ecg,
            patient_id="TEST001",
        )

        assert metrics["breathing_rate_validation_status"] == "missing_data"

    def test_evaluate_saves_csv(self, sample_telemetry_df, sample_ecg_timeseries_df, temp_dir):
        """Should save aligned comparison to CSV."""
        metrics, _ = fn.evaluate_breathing_rate_validation(
            sample_telemetry_df,
            sample_ecg_timeseries_df,
            patient_id="TEST001",
            output_dir=temp_dir,
        )

        # Should have created a CSV file
        csv_files = list(temp_dir.glob("*breathing_rate_validation.csv"))
        assert len(csv_files) > 0
