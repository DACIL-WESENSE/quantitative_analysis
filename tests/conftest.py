"""
conftest.py
===========
Shared fixtures for pytest test suite.

Provides:
- Temporary directories for test data
- Synthetic telemetry DataFrames
- Mock folder structures
- Sample BDF-like MNE Raw objects
- Parsed tasks.log data
"""

import pytest
import pandas as pd
import numpy as np
import mne
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple


@pytest.fixture(scope="session")
def temp_base_dir():
    """Create a persistent temporary directory for the test session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_patient_folder(temp_dir) -> Path:
    """Create a mock patient folder structure with basic files."""
    patient_dir = temp_dir / "Patient001"
    patient_dir.mkdir(parents=True, exist_ok=True)
    return patient_dir


@pytest.fixture
def sample_telemetry_csv(sample_patient_folder) -> Path:
    """Create a sample WESENSE telemetry CSV with metadata header."""
    csv_path = sample_patient_folder / "WESENSETEST_001.csv"

    # WESENSE format: 5-row header, then column names, units row, then data
    # Simplified to avoid parsing issues
    lines = []
    
    # Metadata header (5 rows)
    lines.append("Identificatie:,WESENSETEST_001,Naam:,Test,Voornaam:,Subject")
    lines.append("Bezoekdatum:,2024-01-15,Geboortedatum:,1990-05-20,Leeftijd:,33")
    lines.append("Geslacht:,m,Gewicht:,75,BMI:,24.0")
    lines.append("Gebruiker:,TestUser,,,,")
    lines.append(",,,,,,")  # Blank line
    
    # Column names
    lines.append("Tijd,HR,VO2,VCO2,VE,RR")
    
    # Units row (will be filtered)
    lines.append("mm:ss,bpm,mL/min,mL/min,L/min,breaths/min")
    
    # Data rows
    for i in range(5):
        lines.append(f"0:{i:02d},{80+i*2},{500+i*50},{400+i*40},{15+i*0.5},{14}")
    for i in range(5, 15):
        lines.append(f"0:{i:02d},{100+i*2},{700+i*50},{600+i*40},{20+i*0.5},{16}")
    for i in range(15, 25):
        lines.append(f"0:{i:02d},{130+i},{1000+i*30},{850+i*25},{30+i*0.3},{18}")
    for i in range(25, 35):
        lines.append(f"0:{i:02d},{160+i},{1500+i*20},{1200+i*15},{45+i*0.2},{22}")
    for i in range(35, 40):
        lines.append(f"0:{i:02d},{130-(i-35)*5},{800-(i-35)*50},{600-(i-35)*40},{25-(i-35)*0.5},{16}")
    
    csv_path.write_text("\n".join(lines))
    return csv_path


@pytest.fixture
def sample_telemetry_df() -> pd.DataFrame:
    """Create a clean telemetry DataFrame with Stage column and numeric data."""
    n_rows = 50
    times = np.linspace(0, 60, n_rows)
    
    stages = []
    for t in times:
        if t < 10:
            stages.append("Opwarmen")
        elif t < 20:
            stages.append("Test")
        elif t < 30:
            stages.append("VT1")
        elif t < 40:
            stages.append("VT2")
        else:
            stages.append("Herstel")
    
    df = pd.DataFrame({
        "Tijd": [f"0:{int(t):02d}" for t in times],
        "HR": 80 + times * 2,
        "VO2": 500 + times * 30,
        "VCO2": 400 + times * 25,
        "VE": 15 + times * 0.5,
        "RR": 14 + times * 0.1,
        "Stage": stages,
    })
    return df


@pytest.fixture
def sample_info_df() -> pd.DataFrame:
    """Create a mock info DataFrame from CSV metadata header."""
    return pd.DataFrame([
        ["Identificatie:", "WESENSETEST_001", "Naam:", "Test", "Voornaam:", "Subject"],
        ["Bezoekdatum:", "2024-01-15", "Geboortedatum:", "1990-05-20", "Leeftijd:", "33"],
        ["Geslacht:", "m", "Gewicht:", "75", "BMI:", "24.0"],
        ["Gebruiker:", "TestUser", "", "", "", ""],
    ])


@pytest.fixture
def sample_tasks_log_csv(sample_patient_folder) -> Path:
    """Create a sample tasks.log file with various timestamp formats."""
    tasks_log_path = sample_patient_folder / "tasks.log"
    
    content = """# Test tasks log file
timestamp,event,label
0.5,START,Start on Vyntus CPX
5.2,EVENT,Exercise begin
10.0,VT1_DETECTED,First ventilatory threshold
20.5,VT2_DETECTED,Second ventilatory threshold
45.0,END,Exercise stop
50.0,RECOVERY_START,Recovery phase begins
60.5,RECOVERY_END,End of recovery

# Alternative formats (key-value style)
time=12.5 | event=marker1 | label=test marker
time:15.0, event:marker2, label:another marker
t=30.0; event=marker3; label=third marker
"""
    
    tasks_log_path.write_text(content)
    return tasks_log_path


@pytest.fixture
def sample_tasks_markers_df() -> pd.DataFrame:
    """Create a mock parsed tasks.log DataFrame."""
    return pd.DataFrame([
        {
            "line_number": 1,
            "timestamp_raw": "0.5",
            "timestamp_seconds": 0.5,
            "timestamp_datetime": None,
            "event_code": "START",
            "label": "Start on Vyntus CPX",
            "marker_text": "START, Start on Vyntus CPX",
            "raw_line": "0.5,START,Start on Vyntus CPX",
            "parse_status": "ok",
            "source_file": "tasks.log",
        },
        {
            "line_number": 2,
            "timestamp_raw": "5.2",
            "timestamp_seconds": 5.2,
            "timestamp_datetime": None,
            "event_code": "EVENT",
            "label": "Exercise begin",
            "marker_text": "EVENT, Exercise begin",
            "raw_line": "5.2,EVENT,Exercise begin",
            "parse_status": "ok",
            "source_file": "tasks.log",
        },
        {
            "line_number": 3,
            "timestamp_raw": "10.0",
            "timestamp_seconds": 10.0,
            "timestamp_datetime": None,
            "event_code": "VT1_DETECTED",
            "label": "First ventilatory threshold",
            "marker_text": "VT1_DETECTED, First ventilatory threshold",
            "raw_line": "10.0,VT1_DETECTED,First ventilatory threshold",
            "parse_status": "ok",
            "source_file": "tasks.log",
        },
    ])


@pytest.fixture
def sample_ecg_features_df() -> pd.DataFrame:
    """Create a mock ECG features DataFrame from extract_ecg_features()."""
    return pd.DataFrame([
        {
            "channel": "ECG_L1",
            "mean_abs_uV": 450.5,
            "estimated_hr_bpm": 75.2,
        },
        {
            "channel": "ECG_L2",
            "mean_abs_uV": 480.3,
            "estimated_hr_bpm": 75.5,
        },
    ])


@pytest.fixture
def sample_ecg_timeseries_df() -> pd.DataFrame:
    """Create a mock ECG time-series DataFrame from extract_ecg_timeseries()."""
    times = np.linspace(0, 60, 20)
    return pd.DataFrame({
        "time_s": times,
        "channel": "ECG_L1",
        "hr_bpm": 75 + 10 * np.sin(times / 10),
        "rmssd_ms": 50 + 5 * np.random.randn(20),
        "sdnn_ms": 80 + 8 * np.random.randn(20),
        "lf_ms2": 100 + 10 * np.random.randn(20),
        "hf_ms2": 50 + 5 * np.random.randn(20),
        "lf_hf_ratio": 2.0 + 0.5 * np.random.randn(20),
        "breathing_rate_bpm": 16 + 2 * np.random.randn(20),
    })


@pytest.fixture
def sample_raw_bdf_like():
    """Create a mock MNE Raw object mimicking a BDF ECG recording."""
    # Create synthetic ECG-like data
    sfreq = 256  # sampling frequency in Hz
    duration = 120  # seconds
    n_samples = sfreq * duration

    # Two ECG channels with synthetic heart-rate-like oscillations
    t = np.linspace(0, duration, n_samples)
    # Simulate ECG with ~70 bpm baseline plus some noise
    ch1_signal = 100 * np.sin(2 * np.pi * (70/60) * t / 60) + 30 * np.random.randn(n_samples)
    ch2_signal = 90 * np.sin(2 * np.pi * (71/60) * t / 60 + 0.5) + 25 * np.random.randn(n_samples)

    data = np.array([ch1_signal, ch2_signal])

    # Create MNE Info object
    info = mne.create_info(
        ch_names=["ECG_L1", "ECG_L2"],
        sfreq=sfreq,
        ch_types=["ecg", "ecg"],
    )

    # Create Raw object
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture
def sample_breathing_rate_aligned_df() -> pd.DataFrame:
    """Create a mock aligned breathing-rate comparison DataFrame."""
    return pd.DataFrame({
        "time_s_calc": [10.0, 20.0, 30.0, 40.0, 50.0],
        "time_s_measured": [10.2, 20.1, 29.8, 40.3, 50.1],
        "time_delta_s": [0.2, 0.1, 0.2, 0.3, 0.1],
        "calculated_br_bpm": [16.5, 17.2, 18.0, 19.5, 16.8],
        "measured_br_bpm": [16.0, 17.0, 18.5, 19.0, 17.0],
    })


@pytest.fixture
def sample_breathing_rate_metrics() -> dict:
    """Create mock breathing-rate validation metrics."""
    return {
        "breathing_rate_n_aligned": 5.0,
        "breathing_rate_mean_calculated_bpm": 17.6,
        "breathing_rate_mean_measured_bpm": 17.5,
        "breathing_rate_bias_bpm": 0.1,
        "breathing_rate_mae_bpm": 0.36,
        "breathing_rate_rmse_bpm": 0.42,
        "breathing_rate_mape_pct": 2.05,
        "breathing_rate_corr": 0.95,
        "breathing_rate_mean_abs_time_delta_s": 0.18,
    }
