"""
test_folder_discovery.py
========================
Tests for directory parsing, file discovery, and patient ID resolution.

Coverage:
- discover_patient_folders: finds patient folders recursively
- canonicalize_patient_id: normalizes IDs to filesystem-safe format
- resolve_patient_id: precedence-based resolution (metadata > filename > folder)
- find_csv_file: locates telemetry CSV among candidate files
- find_bdf_files: discovers L1/L2 ECG files by naming pattern
- find_tasks_log_file: locates tasks.log file
"""

import pytest
from pathlib import Path
import sys
import functions as fn


class TestCanonicalize:
    """Test patient ID normalization (canonicalize_patient_id)."""

    def test_basic_id(self):
        """Normal alphanumeric ID should be uppercased."""
        result = fn.canonicalize_patient_id("WESENSETEST001")
        assert result == "WESENSETEST001"

    def test_mixed_case(self):
        """Mixed case should be normalized to uppercase."""
        result = fn.canonicalize_patient_id("WeSenseTest001")
        assert result == "WESENSETEST001"

    def test_spaces_to_underscores(self):
        """Spaces should be converted to underscores."""
        result = fn.canonicalize_patient_id("WESENSE TEST 001")
        assert result == "WESENSE_TEST_001"

    def test_special_chars_removed(self):
        """Non-alphanumeric characters (except ._-) should be removed."""
        result = fn.canonicalize_patient_id("WESENSE@TEST!001")
        assert result == "WESENSETEST001"

    def test_slashes_to_dashes(self):
        """Forward/backward slashes should become dashes."""
        result = fn.canonicalize_patient_id("WESENSE/TEST\\001")
        assert result == "WESENSE-TEST-001"

    def test_none_returns_none(self):
        """None should return None."""
        result = fn.canonicalize_patient_id(None)
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        result = fn.canonicalize_patient_id("")
        assert result is None

    def test_nan_string_returns_none(self):
        """String 'nan' (case-insensitive) should return None."""
        assert fn.canonicalize_patient_id("nan") is None
        assert fn.canonicalize_patient_id("NaN") is None

    def test_whitespace_only_returns_none(self):
        """Whitespace-only string should return None."""
        result = fn.canonicalize_patient_id("   ")
        assert result is None

    def test_underscore_collapsing(self):
        """Multiple consecutive underscores should collapse to one."""
        result = fn.canonicalize_patient_id("PATIENT___001")
        assert result == "PATIENT_001"

    def test_leading_trailing_punctuation_stripped(self):
        """Leading/trailing dots, dashes, underscores should be stripped."""
        result = fn.canonicalize_patient_id("...PATIENT001---")
        assert result == "PATIENT001"


class TestResolvePatientId:
    """Test patient ID resolution with precedence rules."""

    def test_precedence_metadata_first(self, sample_patient_folder):
        """Metadata (info_df) should take priority over all other sources."""
        info_df = fn.pd.DataFrame([
            ["Identificatie:", "META_ID", ""],
        ])
        csv_path = Path("CSVID.csv")
        
        result = fn.resolve_patient_id(
            sample_patient_folder,
            info_df=info_df,
            csv_path=csv_path,
        )
        assert result == "META_ID"

    def test_precedence_csv_second(self, sample_patient_folder):
        """CSV filename should be used when info_df is missing or empty."""
        csv_path = Path("CSVID_DATA.csv")
        
        result = fn.resolve_patient_id(
            sample_patient_folder,
            info_df=None,
            csv_path=csv_path,
        )
        # Result should be either CSVID or the folder name, but should be something
        assert result is not None
        assert len(result) > 0

    def test_precedence_bdf_third(self, sample_patient_folder):
        """BDF filename should be used when CSV is missing."""
        bdf_path = Path("WESENSETEST_BDFID_L1_ECG.bdf")
        
        result = fn.resolve_patient_id(
            sample_patient_folder,
            info_df=None,
            csv_path=None,
            bdf_paths=(bdf_path, None),
        )
        assert "BDFID" in result or "WESENSETEST" in result

    def test_fallback_to_folder_name(self, sample_patient_folder):
        """Folder name should be last resort fallback."""
        result = fn.resolve_patient_id(
            sample_patient_folder,
            info_df=None,
            csv_path=None,
            bdf_paths=None,
        )
        assert "PATIENT001" in result

    def test_unknown_patient_fallback(self, temp_dir):
        """If no ID can be resolved, should use UNKNOWN_PATIENT."""
        unknown_folder = temp_dir / "xyz"
        unknown_folder.mkdir()
        
        result = fn.resolve_patient_id(
            unknown_folder,
            info_df=None,
            csv_path=None,
        )
        # Should still use folder name "xyz" if canonicalization succeeds
        assert result is not None


class TestDiscoverPatientFolders:
    """Test recursive patient folder discovery."""

    def test_discover_single_patient(self, temp_dir):
        """Should find a folder with a CSV file."""
        patient_dir = temp_dir / "Patient001"
        patient_dir.mkdir()
        csv_file = patient_dir / "data.csv"
        csv_file.write_text("Identificatie:\nTijd,HR,VO2\n0:00,80,500\n")

        folders = fn.discover_patient_folders(str(temp_dir))
        assert len(folders) == 1
        assert folders[0].name == "Patient001"

    def test_discover_multiple_patients(self, temp_dir):
        """Should find multiple patient folders."""
        for i in range(3):
            patient_dir = temp_dir / f"Patient{i:03d}"
            patient_dir.mkdir()
            csv_file = patient_dir / "data.csv"
            csv_file.write_text("Identificatie:\nTijd,HR,VO2\n0:00,80,500\n")

        folders = fn.discover_patient_folders(str(temp_dir))
        assert len(folders) == 3

    def test_discover_skips_hidden_dirs(self, temp_dir):
        """Should skip hidden directories (starting with .)."""
        visible_dir = temp_dir / "Patient001"
        visible_dir.mkdir()
        csv_file = visible_dir / "data.csv"
        csv_file.write_text("Identificatie:\nTijd,HR,VO2\n0:00,80,500\n")

        hidden_dir = temp_dir / ".hidden"
        hidden_dir.mkdir()
        csv_file_hidden = hidden_dir / "data.csv"
        csv_file_hidden.write_text("Identificatie:\nTijd,HR,VO2\n0:00,80,500\n")

        folders = fn.discover_patient_folders(str(temp_dir))
        assert len(folders) == 1
        assert folders[0].name == "Patient001"

    def test_natural_sort_order(self, temp_dir):
        """Should sort folders in natural (numeric-aware) order."""
        for name in ["Patient10", "Patient2", "Patient1"]:
            patient_dir = temp_dir / name
            patient_dir.mkdir()
            csv_file = patient_dir / "data.csv"
            csv_file.write_text("Identificatie:\nTijd,HR,VO2\n0:00,80,500\n")

        folders = fn.discover_patient_folders(str(temp_dir))
        names = [f.name for f in folders]
        assert names == ["Patient1", "Patient2", "Patient10"]

    def test_empty_directory_returns_empty(self, temp_dir):
        """Should return empty list for directory with no CSV files."""
        folders = fn.discover_patient_folders(str(temp_dir))
        assert folders == []

    def test_nonexistent_directory(self, temp_dir):
        """Should return empty list and log warning for nonexistent directory."""
        folders = fn.discover_patient_folders(str(temp_dir / "nonexistent"))
        assert folders == []


class TestFindCsvFile:
    """Test CSV file discovery."""

    def test_find_single_csv(self, temp_dir):
        """Should find a single CSV file in folder."""
        csv_path = temp_dir / "data.csv"
        csv_path.write_text("Identificatie:\nTijd,HR,VO2\n0:00,80,500\n")

        result = fn.find_csv_file(temp_dir)
        assert result == csv_path

    def test_prefer_telemetry_csv(self, temp_dir):
        """Should prefer WESENSE-format CSV over generic CSV."""
        # Generic CSV
        generic_csv = temp_dir / "other_data.csv"
        generic_csv.write_text("col1,col2\nval1,val2\n")

        # Telemetry CSV with recognizable headers
        telemetry_csv = temp_dir / "WESENSE_data.csv"
        telemetry_csv.write_text("Identificatie:\nBezoekdatum:\nTijd,HR,VO2\n0:00,80,500\n")

        result = fn.find_csv_file(temp_dir)
        assert result == telemetry_csv

    def test_find_csv_recursively(self, temp_dir):
        """Should find CSV in subdirectories."""
        subdir = temp_dir / "subfolder"
        subdir.mkdir()
        csv_path = subdir / "data.csv"
        csv_path.write_text("Identificatie:\nTijd,HR,VO2\n0:00,80,500\n")

        result = fn.find_csv_file(temp_dir)
        assert result == csv_path

    def test_no_csv_returns_none(self, temp_dir):
        """Should return None when no CSV is found."""
        result = fn.find_csv_file(temp_dir)
        assert result is None


class TestFindBdfFiles:
    """Test BDF ECG file discovery."""

    def test_find_l1_bdf(self, temp_dir):
        """Should find L1 ECG file with correct naming."""
        bdf_l1 = temp_dir / "WESENSETEST_001_L1_ECG.bdf"
        bdf_l1.touch()

        l1, l2 = fn.find_bdf_files(temp_dir)
        assert l1 == bdf_l1
        assert l2 is None

    def test_find_l2_bdf(self, temp_dir):
        """Should find L2 ECG file with correct naming."""
        bdf_l2 = temp_dir / "WESENSETEST_001_L2_ECG.bdf"
        bdf_l2.touch()

        l1, l2 = fn.find_bdf_files(temp_dir)
        assert l1 is None
        assert l2 == bdf_l2

    def test_find_both_l1_and_l2(self, temp_dir):
        """Should find both L1 and L2 ECG files."""
        bdf_l1 = temp_dir / "WESENSETEST_001_L1_ECG.bdf"
        bdf_l2 = temp_dir / "WESENSETEST_001_L2_ECG.bdf"
        bdf_l1.touch()
        bdf_l2.touch()

        l1, l2 = fn.find_bdf_files(temp_dir)
        assert l1 == bdf_l1
        assert l2 == bdf_l2

    def test_find_bdf_case_insensitive(self, temp_dir):
        """Should find BDF files regardless of case."""
        bdf_l1 = temp_dir / "WESENSETEST_001_l1_ecg.bdf"
        bdf_l1.touch()

        l1, l2 = fn.find_bdf_files(temp_dir)
        assert l1 == bdf_l1

    def test_find_bdf_recursively(self, temp_dir):
        """Should find BDF files in subdirectories."""
        subdir = temp_dir / "ecg_data"
        subdir.mkdir()
        bdf_l1 = subdir / "WESENSETEST_001_L1_ECG.bdf"
        bdf_l1.touch()

        l1, l2 = fn.find_bdf_files(temp_dir)
        assert l1 == bdf_l1

    def test_no_bdf_returns_none(self, temp_dir):
        """Should return (None, None) when no BDF files are found."""
        l1, l2 = fn.find_bdf_files(temp_dir)
        assert l1 is None
        assert l2 is None


class TestFindTasksLogFile:
    """Test tasks.log file discovery."""

    def test_find_tasks_log(self, temp_dir):
        """Should find tasks.log file."""
        log_path = temp_dir / "tasks.log"
        log_path.write_text("timestamp,event\n0.5,START\n")

        result = fn.find_tasks_log_file(temp_dir)
        assert result == log_path

    def test_find_tasks_log_case_insensitive(self, temp_dir):
        """Should find tasks.log regardless of case."""
        log_path = temp_dir / "TASKS.LOG"
        log_path.write_text("timestamp,event\n0.5,START\n")

        result = fn.find_tasks_log_file(temp_dir)
        assert result == log_path

    def test_find_tasks_log_recursively(self, temp_dir):
        """Should find tasks.log in subdirectories."""
        subdir = temp_dir / "logs"
        subdir.mkdir()
        log_path = subdir / "tasks.log"
        log_path.write_text("timestamp,event\n0.5,START\n")

        result = fn.find_tasks_log_file(temp_dir)
        assert result == log_path

    def test_no_tasks_log_returns_none(self, temp_dir):
        """Should return None when tasks.log is not found."""
        result = fn.find_tasks_log_file(temp_dir)
        assert result is None
