"""
test_tasks_log_parsing.py
==========================
Tests for tasks.log marker parsing and timestamp extraction.

Coverage:
- parse_tasks_log: parses marker rows with various timestamp formats
- _parse_tasks_timestamp: handles multiple timestamp formats (seconds, HH:MM:SS, datetime)
- find_tasks_marker_timestamp: finds earliest match for event + label
- trim_raw_before_timestamp: removes pre-marker samples correctly
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import functions as fn


class TestParseTasksTimestamp:
    """Test timestamp parsing (_parse_tasks_timestamp)."""

    def test_parse_numeric_seconds(self):
        """Should parse numeric seconds directly."""
        seconds, dt = fn._parse_tasks_timestamp("12.5")
        assert seconds == 12.5
        assert dt is None

    def test_parse_negative_seconds(self):
        """Should parse negative seconds (shouldn't occur in practice)."""
        seconds, dt = fn._parse_tasks_timestamp("-5.0")
        assert seconds == -5.0

    def test_parse_comma_decimal(self):
        """Should handle comma as decimal separator."""
        seconds, dt = fn._parse_tasks_timestamp("12,5")
        assert seconds == 12.5

    def test_parse_mmss_format(self):
        """Should parse MM:SS format to seconds."""
        seconds, dt = fn._parse_tasks_timestamp("1:30")
        assert seconds == 90.0
        assert dt is None

    def test_parse_hhmmss_format(self):
        """Should parse HH:MM:SS format to seconds."""
        seconds, dt = fn._parse_tasks_timestamp("1:2:30")
        assert seconds == 3600 + 120 + 30  # 1h 2m 30s
        assert dt is None

    def test_parse_hhmmss_with_decimal(self):
        """Should parse HH:MM:SS.milliseconds format."""
        seconds, dt = fn._parse_tasks_timestamp("0:0:30.5")
        assert seconds == 30.5

    def test_parse_mmss_with_decimal(self):
        """Should parse MM:SS.milliseconds format."""
        seconds, dt = fn._parse_tasks_timestamp("1:30.5")
        assert seconds == 90.5

    def test_parse_datetime_iso(self):
        """Should parse ISO 8601 datetime strings."""
        seconds, dt = fn._parse_tasks_timestamp("2024-01-15T12:30:45")
        assert seconds is None
        assert dt is not None

    def test_parse_datetime_with_timezone(self):
        """Should parse datetime with timezone info."""
        seconds, dt = fn._parse_tasks_timestamp("2024-01-15T12:30:45Z")
        assert seconds is None
        assert dt is not None

    def test_parse_key_value_with_prefix(self):
        """Should strip key= prefix before parsing."""
        seconds, dt = fn._parse_tasks_timestamp("time=12.5")
        assert seconds == 12.5

    def test_parse_invalid_returns_none(self):
        """Should return (None, None) for invalid timestamps."""
        seconds, dt = fn._parse_tasks_timestamp("invalid_time")
        assert seconds is None
        assert dt is None

    def test_parse_empty_string(self):
        """Should return (None, None) for empty string."""
        seconds, dt = fn._parse_tasks_timestamp("")
        assert seconds is None
        assert dt is None


class TestParseTasksLog:
    """Test tasks.log parsing (parse_tasks_log)."""

    def test_parse_csv_format(self, sample_tasks_log_csv):
        """Should parse CSV-format tasks.log file."""
        markers = fn.parse_tasks_log(sample_tasks_log_csv)
        
        assert markers is not None
        assert len(markers) > 0
        assert "timestamp_seconds" in markers.columns
        assert "event_code" in markers.columns
        assert "label" in markers.columns

    def test_parse_comments_ignored(self, sample_tasks_log_csv):
        """Should ignore lines starting with # or //."""
        markers = fn.parse_tasks_log(sample_tasks_log_csv)
        
        # Should not have comments in parsed results
        for raw_line in markers["raw_line"]:
            assert not raw_line.startswith("#")
            assert not raw_line.startswith("//")

    def test_parse_multiple_delimiters(self, sample_tasks_log_csv):
        """Should handle comma, tab, semicolon, pipe delimiters."""
        markers = fn.parse_tasks_log(sample_tasks_log_csv)
        
        # Should have parsed all non-comment lines with timestamps
        # The fixture has multiple delimiters in alternative format section
        assert len(markers) >= 3

    def test_parse_extracts_events(self, sample_tasks_log_csv):
        """Should extract event codes correctly."""
        markers = fn.parse_tasks_log(sample_tasks_log_csv)
        
        event_codes = markers["event_code"].tolist()
        assert "START" in event_codes
        assert any("EVENT" in code for code in event_codes)

    def test_parse_extracts_labels(self, sample_tasks_log_csv):
        """Should extract label text correctly."""
        markers = fn.parse_tasks_log(sample_tasks_log_csv)
        
        labels = markers["label"].tolist()
        # Should have some labels with descriptive text
        assert any(len(label) > 5 for label in labels if isinstance(label, str))

    def test_parse_sets_status_ok(self, sample_tasks_log_csv):
        """Should mark successfully parsed rows with 'ok' status."""
        markers = fn.parse_tasks_log(sample_tasks_log_csv)
        
        ok_count = (markers["parse_status"] == "ok").sum()
        assert ok_count > 0  # At least some should parse successfully

    def test_parse_returns_dataframe(self, sample_tasks_log_csv):
        """Should return DataFrame with all required columns."""
        markers = fn.parse_tasks_log(sample_tasks_log_csv)
        
        required_cols = [
            "line_number", "timestamp_raw", "timestamp_seconds",
            "timestamp_datetime", "event_code", "label", "marker_text",
            "raw_line", "parse_status", "source_file",
        ]
        for col in required_cols:
            assert col in markers.columns

    def test_parse_empty_file(self, temp_dir):
        """Should return empty DataFrame for empty log file."""
        empty_log = temp_dir / "empty.log"
        empty_log.write_text("")
        
        markers = fn.parse_tasks_log(empty_log)
        assert len(markers) == 0

    def test_parse_handles_missing_timestamp(self, temp_dir):
        """Should mark rows without timestamps as missing_timestamp."""
        log_path = temp_dir / "no_timestamp.log"
        log_path.write_text("event,label\nEVENT_NO_TIME,Test marker\n")
        
        markers = fn.parse_tasks_log(log_path)
        if len(markers) > 0:
            assert "missing_timestamp" in markers["parse_status"].values


class TestFindTasksMarkerTimestamp:
    """Test marker timestamp finding (find_tasks_marker_timestamp)."""

    def test_find_start_marker(self, sample_tasks_markers_df):
        """Should find 'start' marker with specific label."""
        result = fn.find_tasks_marker_timestamp(
            sample_tasks_markers_df,
            event_code="START",
            label_contains="CPX",
        )
        assert result == 0.5

    def test_find_marker_case_insensitive(self, sample_tasks_markers_df):
        """Should match markers case-insensitively."""
        result = fn.find_tasks_marker_timestamp(
            sample_tasks_markers_df,
            event_code="start",  # lowercase
            label_contains="cpx",  # lowercase
        )
        assert result == 0.5

    def test_find_event_only(self, sample_tasks_markers_df):
        """Should match on event code even without label."""
        result = fn.find_tasks_marker_timestamp(
            sample_tasks_markers_df,
            event_code="EVENT",
            label_contains="",
        )
        assert result == 5.2

    def test_find_earliest_match(self, sample_tasks_markers_df):
        """Should return earliest matching timestamp."""
        # Add a second START marker at later time
        new_row = {
            "timestamp_seconds": 20.0,
            "event_code": "START",
            "label": "Start again",
            "marker_text": "START, Start again",
            "parse_status": "ok",
        }
        extended_df = pd.concat([
            sample_tasks_markers_df,
            pd.DataFrame([new_row])
        ], ignore_index=True)
        
        result = fn.find_tasks_marker_timestamp(
            extended_df,
            event_code="START",
            label_contains="",
        )
        # Should return first START marker
        assert result == 0.5

    def test_find_no_match_returns_none(self, sample_tasks_markers_df):
        """Should return None when no match is found."""
        result = fn.find_tasks_marker_timestamp(
            sample_tasks_markers_df,
            event_code="NONEXISTENT",
            label_contains="",
        )
        assert result is None

    def test_find_empty_dataframe(self):
        """Should return None for empty DataFrame."""
        empty_df = pd.DataFrame(columns=["event_code", "label", "timestamp_seconds"])
        
        result = fn.find_tasks_marker_timestamp(
            empty_df,
            event_code="ANY",
            label_contains="",
        )
        assert result is None

    def test_find_label_substring_match(self, sample_tasks_markers_df):
        """Should match label as substring."""
        result = fn.find_tasks_marker_timestamp(
            sample_tasks_markers_df,
            event_code="VT1_DETECTED",
            label_contains="ventilatory",  # substring of full label
        )
        assert result == 10.0


class TestTrimRawBeforeTimestamp:
    """Test ECG trimming (trim_raw_before_timestamp)."""

    def test_trim_valid_marker(self, sample_raw_bdf_like):
        """Should trim ECG data at valid start marker."""
        original_duration = sample_raw_bdf_like.times[-1]
        
        trimmed = fn.trim_raw_before_timestamp(
            sample_raw_bdf_like,
            start_seconds=30.0,
        )
        
        trimmed_duration = trimmed.times[-1]
        assert trimmed_duration < original_duration
        # New start should be close to 30 seconds from original end
        expected_new_duration = original_duration - 30.0
        assert abs(trimmed_duration - expected_new_duration) < 1.0

    def test_trim_zero_marker(self, sample_raw_bdf_like):
        """Should not trim when start marker is <= 0."""
        original_duration = sample_raw_bdf_like.times[-1]
        
        trimmed = fn.trim_raw_before_timestamp(
            sample_raw_bdf_like,
            start_seconds=0.0,
        )
        
        assert len(trimmed.times) == len(sample_raw_bdf_like.times)

    def test_trim_none_marker(self, sample_raw_bdf_like):
        """Should not trim when start_seconds is None."""
        original_duration = sample_raw_bdf_like.times[-1]
        
        result = fn.trim_raw_before_timestamp(
            sample_raw_bdf_like,
            start_seconds=None,
        )
        
        assert result.times[-1] == original_duration

    def test_trim_invalid_marker(self, sample_raw_bdf_like):
        """Should not trim when start_seconds is invalid (inf, nan)."""
        original_duration = sample_raw_bdf_like.times[-1]
        
        result = fn.trim_raw_before_timestamp(
            sample_raw_bdf_like,
            start_seconds=float('inf'),
        )
        
        assert result.times[-1] == original_duration

    def test_trim_marker_beyond_duration(self, sample_raw_bdf_like):
        """Should not trim when marker is beyond recording duration."""
        original_duration = sample_raw_bdf_like.times[-1]
        
        result = fn.trim_raw_before_timestamp(
            sample_raw_bdf_like,
            start_seconds=original_duration + 100.0,  # way beyond
        )
        
        # Should return untrimmed
        assert result.times[-1] == original_duration

    def test_trim_negative_marker(self, sample_raw_bdf_like):
        """Should handle negative start markers gracefully."""
        result = fn.trim_raw_before_timestamp(
            sample_raw_bdf_like,
            start_seconds=-5.0,
        )
        
        # Should return untrimmed for negative
        assert result is not None


class TestBuildTasksMarkerMatchMask:
    """Test internal marker matching (internal function)."""

    def test_match_event_exact(self):
        """Should match exact event codes."""
        df = pd.DataFrame({
            "event_code": ["START", "EVENT", "END"],
            "label": ["begin", "middle", "finish"],
            "marker_text": ["START, begin", "EVENT, middle", "END, finish"],
        })
        
        mask = fn._build_tasks_marker_match_mask(df, "START", "")
        assert mask.sum() == 1
        assert mask.iloc[0] == True

    def test_match_label_substring(self):
        """Should match label as substring."""
        df = pd.DataFrame({
            "event_code": ["EVENT", "EVENT"],
            "label": ["ventilatory threshold 1", "ventilatory threshold 2"],
            "marker_text": ["EVENT, ventilatory threshold 1", "EVENT, ventilatory threshold 2"],
        })
        
        mask = fn._build_tasks_marker_match_mask(df, "EVENT", "threshold 1")
        assert mask.sum() == 1
        assert mask.iloc[0] == True

    def test_match_empty_dataframe(self):
        """Should handle empty DataFrame."""
        df = pd.DataFrame(columns=["event_code", "label", "marker_text"])
        
        mask = fn._build_tasks_marker_match_mask(df, "ANY", "")
        assert mask.empty or mask.sum() == 0
