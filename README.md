# DACIL-WESENSE Quantitative Analysis

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Input data format](#input-data-format)
3. [Quickstart](#quickstart)

## Prerequisites

**Conda environment (recommended)**

```bash
conda create -n quantitative_analysis python=3.13
conda activate quantitative_analysis
pip install -r requirements.txt
```

---

## Input data format

### Folder structure

Place one sub-folder per patient/trial inside `./data`. Nested patient folders
are supported; discovery is recursive and deterministic. A canonical patient ID
is resolved per dataset (metadata-first, with filename/folder fallback) and is
used consistently in outputs.

```
data/
└── Patient 1/                          ← any name; becomes the patient ID
    ├── WESENSET<id>.csv                ← CPET telemetry (preferred)
    ├── WESENSET<id>.xls                ← or CPET telemetry (auto-converted to CSV)
    ├── L1.bdf                          ← ECG recording, sensor L1 (optional)
    ├── L2.bdf                          ← ECG recording, sensor L2 (optional)
    └── tasks.log                       ← Raw tasklog

```

Multiple patient folders are supported:

```
data/
├── Patient 1/
├── Patient 2/
└── Patient 3/
```

Any extra files in a patient folder (`.dat`, `.log`, etc.) are ignored.

### CPET telemetry file (`.csv` or `.xls`)

After conversion (automatic or manual), the CSV has the following structure:

```
Row 0  Identificatie: | <id>  | Naam: | <name> | Voornaam: | <first>
Row 1  Bezoekdatum:   | <date>| Geboortedatum: | <dob> | Leeftijd: | 29 Jaar
Row 2  Geslacht:      | vrouw | Gewicht: | 68.0 kg | BMI: | 24 kg/m²
Row 3  Gebruiker:     | ...
Row 4  (empty)
Row 5  Tijd | Belasting | RPM | V'E | BF | VTex | V'O2 | V'CO2 | RER | HR | …  ← column headers
Row 6  min  | W         | … (units row — skipped automatically)
Row 7+ <time-series data>
```

Stage annotation rows are interspersed in the data.  Each annotation row
contains one of the following keywords and labels all subsequent rows until
the next annotation:

| Keyword    | Meaning                 |
|------------|-------------------------|
| `Opwarmen` | Warm-up                 |
| `Test`     | Incremental test        |
| `VT1`      | Ventilatory threshold 1 |
| `VT2`      | Ventilatory threshold 2 |
| `Herstel`  | Recovery                |

The pipeline tolerates missing stages — if `VT1` or `VT2` rows are absent the
corresponding features will be `NaN` or flagged accordingly.

**Per-patient, the pipeline recursively searches for telemetry `.csv` files and
selects one deterministically** (preferring files that match the expected
WESENSE telemetry header pattern). If a folder contains multiple telemetry
exports, keep only the intended file.

### BDF ECG files (optional)

BDF files are loaded with MNE.  Both L1 and L2 are optional — missing files
generate a warning and the pipeline continues without ECG features.

Naming convention: ECG files are simply named `L1` and `L2` within each patient
folder (case-insensitive). If your export still includes a `.bdf` extension,
the loader accepts it, and legacy naming patterns (e.g.,
`WESENSETEST_<id>_L1_ECG*.bdf`) are still supported for backwards compatibility.

---

## Quickstart

```bash
# 1. Activate the environment
conda activate quantitative_analysis

# 2. Place patient folders under ./data (see Input data format above)
#    Files can be either .csv or .xls format; the pipeline handles both

# 3. Run the CPET pipeline (automatic XLS conversion if needed)
python run_pipeline.py --data-root data --output-root output

# 4. (Optional) Run ECG deep-dive for one patient
python run_ecg.py --data-root data --output-root output
```

Results are written to `./output/`.

**Note**: If you prefer to convert XLS files manually before running the pipeline:
```bash
python xls_to_csv.py   # Pre-convert all .xls to .csv (optional)
```

---

## Step-by-step usage

### 1. Prepare data (XLS files optional)

The pipeline automatically converts XLS files to CSV during processing, so you can provide either format:

```bash
# Option A: Let the pipeline handle XLS → CSV conversion automatically
# (No manual conversion step needed—just place .xls files in patient folders)
python run_pipeline.py --data-root data --output-root output

# Option B: Pre-convert XLS files manually (optional)
# Useful if you want to inspect/validate CSVs before running the pipeline
python xls_to_csv.py

# With options:
python xls_to_csv.py --overwrite       # Re-convert even if CSV exists
python xls_to_csv.py --dry-run         # Preview without writing
python xls_to_csv.py --data-dir /path/to/data
```

**Note**: The pipeline calls `find_csv_file()`, which automatically locates and
converts XLS files if no CSV is already present. Manual conversion is optional.

### 2. Convert BDF files to CSV/TSV

Biosemi BDF files (ECG recordings) can be converted to human-readable
timeseries files with `bdf_to_csv.py`:

```bash
# Convert all .bdf files under ./data to CSV (one file per BDF)
python bdf_to_csv.py

# Write tab-separated files instead
python bdf_to_csv.py --format tsv

# Place all output files in a single directory
python bdf_to_csv.py --output-dir ./output/ecg_raw

# Export only specific channels
python bdf_to_csv.py --channels Fp1 Fp2 Status

# Re-convert even if an output file already exists
python bdf_to_csv.py --overwrite

# Preview what would be converted without writing any files
python bdf_to_csv.py --dry-run
```

Each output file has one row per sample and the following columns:

| Column     | Contents                                                                |
|------------|-------------------------------------------------------------------------|
| `time_s`   | Elapsed time in seconds from the start of the recording                 |
| `<ch_name>`| Signal value in **µV** for EEG/ECG/EMG/EOG channels; native unit otherwise |
| `status`   | Integer trigger / event code from the BDF Status channel                |

The output file is placed alongside the source `.bdf` unless `--output-dir`
is specified.

### 3. Run the main analysis pipeline (Python script)

```bash
# Basic run (telemetry-focused; ECG extraction skipped)
python run_pipeline.py --data-root data --output-root output

# Include ECG features from BDF files (if available)
python run_pipeline.py --data-root data --output-root output --include-ecg

# Parallel patient processing and custom workers
python run_pipeline.py --data-root data --output-root output --parallel --workers 8
```

The script runs the following steps for every patient folder:

| Step           | What it does                                                               |
|----------------|----------------------------------------------------------------------------|
| Load telemetry | Reads the CSV, parses stage annotations, returns a cleaned DataFrame       |
| Parse tasks.log | If present, parses timestamped task markers and exports per-patient marker CSV |
| Load ECG       | Optional (`--include-ecg`): reads L1 / L2 BDF files with MNE; trims pre-start data before marker `start,Start on Vyntus CPX` when available |
| Sync           | Merges ECG features into the telemetry DataFrame                           |
| Validate BR    | Compares ECG-derived breathing rate with measured Vyntus BF/RR (time-aligned metrics) |
| Save           | Writes `<patient_id>_telemetry.csv` to `output/<patient_id>/`              |
| Summarise      | Appends a row to the master summary                                        |

After batch processing, the script produces EDA plots (stage profiles,
V-slope, RER trajectory, HRR), a cross-patient correlation heatmap, and
unsupervised ML analysis (PCA, K-Means, DBSCAN).

It also writes a cross-patient status table (`patient_analysis_status.csv`)
with canonical patient IDs, analysis-run flags, and key metrics:
`mean_hr_bpm`, `mean_rmssd_ms` (SSDM request mapped to RMSSD naming),
`mean_sdnn_ms`, `mean_breathing_rate_bpm_edr`,
`mean_breathing_rate_bpm_cpet`, and `breathing_rate_mae_bpm`.

Progress bars show throughput for each batch step.  Failed patients are
logged to `pipeline.log` and skipped — they do not interrupt the pipeline.

### 4. Run the ECG analysis script (optional)

```bash
# Single-patient ECG analysis (defaults to first patient folder)
python run_ecg.py --data-root data --output-root output

# Explicit patient + custom ECG window length
python run_ecg.py --data-root data --output-root output \
  --patient-id "Patient 1" --window-duration 30

# Also compute a cross-patient ECG batch summary
python run_ecg.py --data-root data --output-root output --run-batch-summary
```

This script reuses the ECG helpers in `functions.py` and writes outputs under
`output/ecg_analysis/<patient_id>/`.

### 5. Run the COPD risk scoring notebook

Open `copd_risk.ipynb` and set the same two path variables:

```python
DATA_ROOT  = "data"
OUTPUT_ROOT = "output"
```

Then **Run All**.  For each patient it:

1. Extracts 9 CPET biomarkers from the telemetry (see [COPD risk scoring](#copd-risk-scoring))
2. Compares each marker against a clinical threshold
3. Assigns a composite risk level: **Low** (0 flags), **Moderate** (1–2 flags), **High** (≥ 3 flags)
4. Generates a radar chart, a cross-patient bar chart, and (if ≥ 3 patients) a PCA scatter

---

## Configuration reference

### `run_pipeline.py` (primary CPET entrypoint)

| Flag                  | Default   | Description |
|-----------------------|-----------|-------------|
| `--data-root`         | `data`    | Root directory containing patient sub-folders |
| `--output-root`       | `output`  | Base output directory |
| `--include-ecg`       | off       | Include ECG extraction/sync during per-patient processing |
| `--parallel`          | off       | Process patient folders concurrently |
| `--workers`           | `4`       | Worker threads when `--parallel` is set |
| `--skip-ml`           | off       | Skip pooled PCA / clustering section |
| `--kmeans-clusters`   | `4`       | Number of K-Means clusters |
| `--dbscan-eps`        | `1.0`     | DBSCAN neighbourhood radius |
| `--dbscan-min-samples`| `5`       | DBSCAN minimum neighbourhood size |
| `--use-gpu`           | off       | Enable optional CuPy acceleration |
| `--log-file`          | `<output-root>/pipeline.log` | Override log file path |
| `--log-level`         | `INFO`    | Logging verbosity |

### `run_ecg.py` (primary ECG entrypoint)

| Flag                | Default   | Description |
|---------------------|-----------|-------------|
| `--data-root`       | `data`    | Root directory containing patient sub-folders |
| `--output-root`     | `output`  | Base output directory |
| `--patient-id`      | first patient folder | Target patient folder name |
| `--window-duration` | `30`      | Sliding window duration (seconds) |
| `--preview-seconds` | `300`     | Seconds shown in raw ECG preview plot |
| `--run-batch-summary` | off     | Build batch-level ECG summary CSV + figure |
| `--log-file`        | `<output-root>/pipeline.log` | Override log file path |
| `--log-level`       | `INFO`    | Logging verbosity |

### `main.ipynb`

| Variable          | Location            | Default                                         | Description                                                         |
|-------------------|---------------------|-------------------------------------------------|---------------------------------------------------------------------|
| `DATA_ROOT`       | Cell 2              | `"data"`                                        | Root directory containing patient sub-folders                       |
| `OUTPUT_ROOT`     | Cell 2              | `"output"`                                      | Base output directory                                               |
| `EDA_METRICS`     | EDA section         | `["HR", "VO2", "VCO2", "Power", "RER", "SpO2"]` | Metrics shown in per-patient stage plots                            |
| `CORR_CANDIDATES` | Correlation section | `["HR", "VO2", ...]`                            | Columns included in the pooled correlation heatmap                  |
| `N_CLUSTERS`      | K-Means cell        | `4`                                             | Number of K-Means clusters — adjust after inspecting the elbow plot |
| `eps`             | DBSCAN cell         | `1.0`                                           | DBSCAN neighbourhood radius (in standard-deviation units)           |
| `min_samples`     | DBSCAN cell         | `5`                                             | DBSCAN minimum neighbourhood size                                   |

### `copd_risk.ipynb`

| Variable      | Location | Default    | Description                                   |
|---------------|----------|------------|-----------------------------------------------|
| `DATA_ROOT`   | Cell 2   | `"data"`   | Root directory containing patient sub-folders |
| `OUTPUT_ROOT` | Cell 2   | `"output"` | Base output directory                         |

To change the clinical thresholds used for risk flagging, edit the
`_COPD_THRESHOLDS` dictionary in `functions.py` (see
[COPD risk scoring](#copd-risk-scoring)).

### `xls_to_csv.py`

| Flag          | Default  | Description                                      |
|---------------|----------|--------------------------------------------------|
| `--data-dir`  | `./data` | Directory to search recursively for `.xls` files |
| `--overwrite` | off      | Re-convert even if a `.csv` already exists       |
| `--dry-run`   | off      | Print what would be done without writing files   |

### `bdf_to_csv.py`

| Flag             | Default              | Description                                                              |
|------------------|----------------------|--------------------------------------------------------------------------|
| `--data-dir`     | `./data`             | Directory to search recursively for `.bdf` files                         |
| `--output-dir`   | same as source BDF   | Directory to write output files                                          |
| `--format`       | `csv`                | Output format: `csv` (comma-separated) or `tsv` (tab-separated)         |
| `--channels`     | all data channels    | Whitelist of channel names to include in the output                      |
| `--overwrite`    | off                  | Re-convert even if an output file already exists                         |
| `--dry-run`      | off                  | Print what would be done without writing files                           |

---

## Output reference

```
output/
├── pipeline.log                        ← combined log from all notebook runs
├── master_summary.csv                  ← one row per patient, peak/mean metrics + breathing-rate validation metrics
├── patient_analysis_status.csv         ← cross-patient analysis flags + key metrics (HR/RMSSD/SDNN/breathing + MAE)
├── batch_vo2_by_gender.png             ← box plot: VO2 distribution by sex
├── batch_bmi_vs_vo2.png                ← scatter: BMI vs peak VO2
├── scree.png                           ← PCA explained variance
├── pca_2d_stage.png                    ← PCA scatter coloured by exercise stage
├── elbow.png                           ← K-Means inertia vs K
├── kmeans_clusters.png                 ← K-Means cluster scatter
├── dbscan_clusters.png                 ← DBSCAN cluster scatter
├── ecg_batch_summary.csv               ← optional cross-patient ECG summary from `run_ecg.py`
├── ecg_batch_summary.png               ← optional ECG summary plot from `run_ecg.py`
├── ecg_analysis/                       ← ECG-focused outputs from `run_ecg.py`
│   └── <patient_id>/
│       ├── <patient_id>_ecg_timeseries.csv      ← windowed ECG features
│       ├── <patient_id>_ecg_summary_stats.csv   ← aggregate ECG metrics
│       ├── <patient_id>_ecg_hr.png              ← heart-rate trace
│       ├── <patient_id>_ecg_hrv_timedomain.png  ← RMSSD/SDNN traces
│       ├── <patient_id>_ecg_hrv_freqdomain.png   ← LF/HF traces
│       ├── <patient_id>_ecg_breathing.png       ← breathing-rate trace
│       └── <patient_id>_ecg_raw_preview.png     ← raw ECG preview with R-peaks
│
├── <patient_id>/                       ← one sub-folder per patient
│   ├── <patient_id>_telemetry.csv      ← cleaned telemetry with Stage column
│   ├── <patient_id>_tasks_markers.csv  ← parsed tasks.log markers with timestamps
│   ├── <patient_id>_breathing_rate_validation.csv  ← aligned ECG-vs-Vyntus breathing-rate samples
│   └── <patient_id>_metrics_by_stage.png
│
├── copd_risk_summary.csv               ← one row per patient: markers + flags + risk score
└── copd_risk/
    ├── <patient_id>_copd_radar.png     ← radar chart of normalised CPET markers
    ├── copd_risk_bar.png               ← flag-count bar chart across patients
    └── all_patients_pca_2d_risk.png    ← PCA scatter coloured by risk level (≥3 patients)
```

`patient_analysis_status.csv` uses explicit, stable column names. For HRV,
the requested "SSDM" metric is represented as `mean_rmssd_ms` (primary), and
`mean_sdnn_ms` is also included to make the RMSSD/SDNN mapping explicit.

`pipeline.log` uses ISO 8601 timestamps and is appended on every script/notebook run. To
start a fresh log, delete the file before running.

---

# AI disclosure

GitHub Copilot was used to aid in the development of this codebase. However, at no point in time was the Copilot process physically able to interact with patient data. All Copilot outputs were manually checked by a human.
