"""
test_xls_conversion.py
======================
Tests for XLS file discovery and automatic CSV conversion.

Coverage:
- find_xls_file: locates telemetry XLS files by naming pattern
- find_csv_file: now automatically converts XLS to CSV when no CSV found
- convert_xls_to_csv: roundtrip validation
"""

import pytest
from pathlib import Path
import sys
import functions as fn


class TestFindXlsFile:
    """Test XLS file discovery (find_xls_file)."""

    def test_find_xls_file(self, temp_dir):
        """Should find a basic XLS file."""
        xls_path = temp_dir / "telemetry.xls"
        xls_path.write_text("Tijd\tHR\tVO2\n0:00\t80\t500\n")

        result = fn.find_xls_file(temp_dir)
        assert result == xls_path

    def test_find_xls_prefer_wesense(self, temp_dir):
        """Should prefer WESENSE-named XLS files."""
        generic_xls = temp_dir / "data.xls"
        generic_xls.write_text("Tijd\tHR\tVO2\n0:00\t80\t500\n")

        wesense_xls = temp_dir / "WESENSETEST_001.xls"
        wesense_xls.write_text("Tijd\tHR\tVO2\n0:00\t80\t500\n")

        result = fn.find_xls_file(temp_dir)
        assert result == wesense_xls

    def test_find_xls_avoid_ecg(self, temp_dir):
        """Should prefer non-ECG XLS files."""
        ecg_xls = temp_dir / "WESENSETEST_001_ECG.xls"
        ecg_xls.write_text("Data\tValue\n1\t100\n")

        telemetry_xls = temp_dir / "WESENSETEST_001.xls"
        telemetry_xls.write_text("Tijd\tHR\tVO2\n0:00\t80\t500\n")

        result = fn.find_xls_file(temp_dir)
        assert result == telemetry_xls

    def test_find_xls_recursively(self, temp_dir):
        """Should find XLS files in subdirectories."""
        subdir = temp_dir / "subfolder"
        subdir.mkdir()
        xls_path = subdir / "telemetry.xls"
        xls_path.write_text("Tijd\tHR\tVO2\n0:00\t80\t500\n")

        result = fn.find_xls_file(temp_dir)
        assert result == xls_path

    def test_no_xls_returns_none(self, temp_dir):
        """Should return None when no XLS files are found."""
        result = fn.find_xls_file(temp_dir)
        assert result is None


class TestFindCsvFileWithXlsConversion:
    """Test CSV file discovery with automatic XLS conversion."""

    def test_find_csv_existing(self, temp_dir):
        """Should find existing CSV without touching XLS."""
        csv_path = temp_dir / "telemetry.csv"
        csv_path.write_text("Identificatie:\nTijd,HR,VO2\n0:00,80,500\n")

        xls_path = temp_dir / "telemetry.xls"
        xls_path.write_text("Tijd\tHR\tVO2\n0:00\t80\t500\n")

        result = fn.find_csv_file(temp_dir)
        assert result == csv_path

    def test_find_csv_no_csv_but_xls_exists(self, temp_dir):
        """Should convert XLS to CSV when no CSV found."""
        xls_path = temp_dir / "telemetry.xls"
        xls_path.write_text("Identificatie:\tValue\nTijd\tHR\tVO2\n0:00\t80\t500\n")

        result = fn.find_csv_file(temp_dir)
        
        # Should have created a CSV file
        assert result is not None
        assert result.suffix == ".csv"
        assert result.parent == temp_dir
        
        # Should be readable
        content = result.read_text()
        assert "Tid" in content or "HR" in content or "0:00" in content

    def test_csv_conversion_preserves_data(self, temp_dir):
        """Converted CSV should preserve data from XLS."""
        xls_path = temp_dir / "telemetry.xls"
        xls_content = "Identificatie:\tWESENSETEST_001\nTijd\tHR\tVO2\n0:00\t80\t500\n0:01\t82\t510\n"
        xls_path.write_text(xls_content)

        csv_path = fn.find_csv_file(temp_dir)
        
        # Should have created the CSV
        assert csv_path is not None
        assert csv_path.exists()
        
        # CSV should be readable
        csv_content = csv_path.read_text()
        assert "80" in csv_content or "500" in csv_content

    def test_find_csv_prefer_csv_over_xls(self, temp_dir):
        """Should prefer CSV over XLS even if both exist."""
        csv_path = temp_dir / "telemetry.csv"
        csv_path.write_text("Identificatie:\nTijd,HR,VO2\n0:00,80,500\n")

        xls_path = temp_dir / "telemetry.xls"
        xls_path.write_text("Tijd\tHR\tVO2\n0:00\t85\t550\n")

        result = fn.find_csv_file(temp_dir)
        assert result == csv_path

    def test_find_csv_no_files_returns_none(self, temp_dir):
        """Should return None when neither CSV nor XLS found."""
        result = fn.find_csv_file(temp_dir)
        assert result is None

    def test_find_csv_conversion_creates_file_in_correct_location(self, temp_dir):
        """Converted CSV should be in same directory as XLS."""
        subdir = temp_dir / "subfolder"
        subdir.mkdir()
        xls_path = subdir / "telemetry.xls"
        xls_path.write_text("Identificatie:\tValue\nTijd\tHR\tVO2\n0:00\t80\t500\n")

        result = fn.find_csv_file(temp_dir)
        
        assert result is not None
        assert result.parent == subdir
        assert result.name == "telemetry.csv"

    def test_find_csv_xls_conversion_iso_to_utf8(self, temp_dir):
        """XLS files with ISO-8859-1 encoding should be converted correctly."""
        xls_path = temp_dir / "telemetry.xls"
        # Write with ISO-8859-1 encoding
        xls_path.write_text(
            "Identificatie:\tValue\nTijd\tHR\tVO2\n0:00\t80\t500\n",
            encoding="iso-8859-1"
        )

        result = fn.find_csv_file(temp_dir)
        
        assert result is not None
        assert result.suffix == ".csv"
        
        # CSV should be readable with UTF-8 (default)
        content = result.read_text(encoding="utf-8")
        assert len(content) > 0


class TestConvertXlsToCsv:
    """Test the convert_xls_to_csv function."""

    def test_convert_basic_xls(self, temp_dir):
        """Should convert basic tab-separated XLS to CSV."""
        xls_path = temp_dir / "input.xls"
        csv_path = temp_dir / "output.csv"
        
        xls_path.write_text(
            "Identificatie:\tValue\nTijd\tHR\tVO2\n0:00\t80\t500\n",
            encoding="iso-8859-1"
        )

        fn.convert_xls_to_csv(xls_path, csv_path)
        
        assert csv_path.exists()
        content = csv_path.read_text(encoding="utf-8")
        assert "Identificatie" in content
        assert "Tijd" in content

    def test_convert_preserves_all_rows(self, temp_dir):
        """Conversion should preserve all rows from input."""
        xls_path = temp_dir / "input.xls"
        csv_path = temp_dir / "output.csv"
        
        lines = ["Header1\tHeader2\tHeader3"]
        for i in range(5):
            lines.append(f"Row{i}\tValue{i}\tData{i}")
        
        xls_path.write_text("\n".join(lines), encoding="iso-8859-1")
        fn.convert_xls_to_csv(xls_path, csv_path)
        
        csv_lines = csv_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(csv_lines) >= 6  # header + 5 rows
