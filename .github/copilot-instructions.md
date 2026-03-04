# Copilot Instructions — DACIL-WESENSE Quantitative Analysis

## Project overview

This is a batch analysis pipeline for exercise physiology data from the DACIL-WESENSE study. It processes per-patient folders containing:
- A `.csv` telemetry file (exercise metrics like HR, VO2, VCO2, Power, RER, SpO2)
- Optional `.bdf` ECG files (L1 and L2 sensors, named `WESENSETEST_<id>_L1_ECG*.bdf` / `_L2_ECG*.bdf`)

## Running the pipeline

The main entry point is `main.ipynb`. Before running, set two variables near the top of the notebook:

```python
DATA_ROOT = "data"    # root folder with one sub-folder per patient
OUTPUT_ROOT = "output"
```

Then run all cells. Logs are written to both the notebook output and `pipeline.log`.

For COPD risk scoring, run `copd_risk.ipynb` (same path config, same output folder).

## Data preparation

Raw `.xls` files from the device are **tab-separated text** with ISO-8859-1 encoding (not real Excel binary). Convert them first:

```bash
python xls_to_csv.py                   # convert all .xls under ./data (skip existing)
python xls_to_csv.py --overwrite       # re-convert even if .csv already exists
python xls_to_csv.py --dry-run         # preview without writing
python xls_to_csv.py --data-dir ./path # custom data directory
```

## Architecture

```
main.ipynb          # full pipeline: load → EDA → ML → report
copd_risk.ipynb     # COPD exacerbation risk scoring (thin orchestration, no logic inline)
functions.py        # all reusable helpers (imported as `fn`)
xls_to_csv.py       # data preparation utility (CLI: --data-dir, --overwrite, --dry-run)
data/<Patient N>/   # one sub-folder per patient; gitignored
output/             # generated CSVs, plots, HTML report; gitignored
```

`functions.py` is structured in logical sections:
1. Directory parsing (`discover_patient_folders`, `find_csv_file`, `find_bdf_files`)
2. Data loading (`load_telemetry`, `load_ecg`, `extract_ecg_features`)
3. Sync & cleaning (`sync_ecg_with_telemetry`, `compute_stage_summary`)
4. EDA plotting (`plot_metrics_by_stage`, `plot_batch_aggregates`)
5. ML (`prepare_features`, `run_pca`, `run_kmeans`, `run_dbscan`, `elbow_plot`, `plot_scree`, `plot_cluster_scatter`, `plot_pca_scatter`)
6. COPD risk scoring (`parse_patient_meta`, `extract_copd_features`, `score_copd_risk`, `plot_copd_radar`)
7. Export (`save_telemetry_csv`, `build_patient_summary`)
8. Utilities (`setup_logging`, `_save_figure`, `_find_column`)

## Key conventions

- **Coding style**: PEP 8, NumPy-style docstrings, type hints on all public functions. No emojis should be used.
- **Private helpers** are prefixed with `_` (e.g. `_parse_telemetry`, `_estimate_hr_from_signal`).
- **Matplotlib backend** is set to `"Agg"` at import time in `functions.py` — plots are never shown interactively, only saved to disk via `_save_figure`.
- **MNE log level** is globally set to `"WARNING"` in `functions.py`.
- **Stage labels** follow a fixed order defined by `STAGE_ORDER = ["Opwarmen", "Test", "VT1", "VT2", "Herstel"]`. Stage detection is case-insensitive text matching across all row cells.
- **Column lookup** for telemetry metrics uses fuzzy candidate matching via `_find_column` / `_find_column_in_series` to tolerate variations in column names across data exports.
- **COPD risk thresholds** are defined in the module-level dict `_COPD_THRESHOLDS` in `functions.py`. Sources: ATS/ERS CPET guidelines; Wasserman 5th ed. Adjust there to change all scoring behaviour.
- **WESENSE CSV format**: files have a 4-row metadata header before the actual column-name row; `load_telemetry` handles this with `skiprows=list(range(5))`. A units row follows the column names and is automatically dropped by `_parse_telemetry` (non-numeric values coerce to NaN).
- **Logging** uses the standard `logging` module. Call `fn.setup_logging()` once at startup; individual modules get `logging.getLogger(__name__)`.
- ECG feature extraction falls back gracefully: ECG channels → EEG channels → first available channels.
