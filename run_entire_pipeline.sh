#!/bin/bash
# Orchestration script for the DACIL-WESENSE quantitative analysis pipeline.
# Coordinates all analysis steps: data prep → main pipeline → ECG analysis.
# Usage: ./run_entire_pipeline.sh [OPTIONS]
#   --data-root DIR          Root directory with patient sub-folders (default: ./data)
#   --output-root DIR        Output directory (default: ./output)
#   --include-ecg            Include ECG feature extraction (default: off)
#   --run-batch-ecg          Also run ECG batch summary (default: off)
#   --skip-ml                Skip ML clustering analysis (default: off)
#   --parallel               Process patients in parallel (default: off)
#   --workers N              Number of parallel workers (default: 4)
#   --dry-run                Show what would be run without executing
#   --help                   Show this help message

set -o errexit
set -o pipefail

# ============================================================================
# Parse command-line arguments
# ============================================================================

DATA_ROOT="data"
OUTPUT_ROOT="output"
INCLUDE_ECG=false
RUN_BATCH_ECG=false
SKIP_ML=false
PARALLEL=false
WORKERS=4
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --include-ecg)
            INCLUDE_ECG=true
            shift
            ;;
        --run-batch-ecg)
            RUN_BATCH_ECG=true
            shift
            ;;
        --skip-ml)
            SKIP_ML=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            grep "^#" "$0" | head -15 | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ============================================================================
# Configuration and logging
# ============================================================================

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")"; pwd )"
VENV_PATH="${SCRIPT_DIR}/.venv"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${OUTPUT_ROOT}/orchestration_${TIMESTAMP}.log"

# Function to log messages
log() {
    local level="$1"
    shift
    local message="$@"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

# Function to log commands being executed
log_command() {
    local cmd="$1"
    log "INFO" "Executing: ${cmd}"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] Would execute: ${cmd}"
    fi
}

# Trap errors and log them
trap_error() {
    local line_no=$1
    log "ERROR" "Script failed at line ${line_no}. See ${LOG_FILE} for details."
    exit 1
}

trap 'trap_error ${LINENO}' ERR

# ============================================================================
# Print startup information
# ============================================================================

mkdir -p "${OUTPUT_ROOT}"

log "INFO" "════════════════════════════════════════════════════════════"
log "INFO" "DACIL-WESENSE Pipeline Orchestration"
log "INFO" "════════════════════════════════════════════════════════════"
log "INFO" "Start time: $(date)"
log "INFO" "Script directory: ${SCRIPT_DIR}"
log "INFO" "Data root: ${DATA_ROOT}"
log "INFO" "Output root: ${OUTPUT_ROOT}"
log "INFO" "Log file: ${LOG_FILE}"
log "INFO" "Include ECG: ${INCLUDE_ECG}"
log "INFO" "Run batch ECG: ${RUN_BATCH_ECG}"
log "INFO" "Skip ML: ${SKIP_ML}"
log "INFO" "Parallel: ${PARALLEL} (workers: ${WORKERS})"
log "INFO" "Dry-run mode: ${DRY_RUN}"
log "INFO" "════════════════════════════════════════════════════════════"

# ============================================================================
# Check prerequisites
# ============================================================================

log "INFO" "Checking prerequisites..."

if [[ ! -d "$VENV_PATH" ]]; then
    log "ERROR" "Virtual environment not found: ${VENV_PATH}"
    log "ERROR" "Create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    log "ERROR" "Data root directory not found: ${DATA_ROOT}"
    exit 1
fi

log "INFO" "Prerequisites OK"

# ============================================================================
# Activate virtual environment
# ============================================================================

log "INFO" "Activating virtual environment: ${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

# Verify activation
if [[ "$VIRTUAL_ENV" == "" ]]; then
    log "ERROR" "Failed to activate virtual environment"
    exit 1
fi

log "INFO" "Virtual environment activated: ${VIRTUAL_ENV}"

# ============================================================================
# Step 1: Convert XLS files to CSV (optional)
# ============================================================================

if [[ -f "${SCRIPT_DIR}/xls_to_csv.py" ]]; then
    log "INFO" "Step 1: Converting XLS files to CSV..."
    CMD="python \"${SCRIPT_DIR}/xls_to_csv.py\" --data-dir \"${DATA_ROOT}\""
    log_command "$CMD"

    if [[ "$DRY_RUN" != "true" ]]; then
        if python "${SCRIPT_DIR}/xls_to_csv.py" --data-dir "${DATA_ROOT}" >> "${LOG_FILE}" 2>&1; then
            log "INFO" "XLS conversion completed successfully"
        else
            log "WARNING" "XLS conversion encountered issues (continuing)"
        fi
    else
        eval "$CMD"
    fi
else
    log "INFO" "Step 1: Skipping XLS conversion (xls_to_csv.py not found)"
fi

# ============================================================================
# Step 2: Run main CPET pipeline
# ============================================================================

log "INFO" "Step 2: Running main CPET pipeline (run_pipeline.py)..."

PIPELINE_ARGS=(
    "--data-root" "${DATA_ROOT}"
    "--output-root" "${OUTPUT_ROOT}"
    "--log-file" "${LOG_FILE}"
    "--log-level" "INFO"
)

if [[ "$INCLUDE_ECG" == "true" ]]; then
    PIPELINE_ARGS+=("--include-ecg")
fi

if [[ "$SKIP_ML" == "true" ]]; then
    PIPELINE_ARGS+=("--skip-ml")
fi

if [[ "$PARALLEL" == "true" ]]; then
    PIPELINE_ARGS+=("--parallel" "--workers" "${WORKERS}")
fi

CMD="python \"${SCRIPT_DIR}/run_pipeline.py\" ${PIPELINE_ARGS[@]}"
log_command "$CMD"

if [[ "$DRY_RUN" != "true" ]]; then
    if python "${SCRIPT_DIR}/run_pipeline.py" "${PIPELINE_ARGS[@]}" >> "${LOG_FILE}" 2>&1; then
        log "INFO" "Main pipeline completed successfully"
    else
        log "ERROR" "Main pipeline failed"
        exit 1
    fi
else
    python "${SCRIPT_DIR}/run_pipeline.py" "${PIPELINE_ARGS[@]}" 2>&1 | head -20
fi

# ============================================================================
# Step 3: Run ECG analysis (optional)
# ============================================================================

if [[ "$INCLUDE_ECG" == "true" ]]; then
    log "INFO" "Step 3: Running ECG analysis (run_ecg.py)..."

    ECG_ARGS=(
        "--data-root" "${DATA_ROOT}"
        "--output-root" "${OUTPUT_ROOT}"
        "--log-file" "${LOG_FILE}"
        "--log-level" "INFO"
    )

    if [[ "$RUN_BATCH_ECG" == "true" ]]; then
        ECG_ARGS+=("--run-batch-summary")
    fi

    CMD="python \"${SCRIPT_DIR}/run_ecg.py\" ${ECG_ARGS[@]}"
    log_command "$CMD"

    if [[ "$DRY_RUN" != "true" ]]; then
        if python "${SCRIPT_DIR}/run_ecg.py" "${ECG_ARGS[@]}" >> "${LOG_FILE}" 2>&1; then
            log "INFO" "ECG analysis completed successfully"
        else
            log "WARNING" "ECG analysis encountered issues (non-fatal)"
        fi
    else
        python "${SCRIPT_DIR}/run_ecg.py" "${ECG_ARGS[@]}" 2>&1 | head -20
    fi
else
    log "INFO" "Step 3: Skipping ECG analysis (use --include-ecg to enable)"
fi

# ============================================================================
# Final summary
# ============================================================================

log "INFO" "════════════════════════════════════════════════════════════"
log "INFO" "Pipeline completed successfully!"
log "INFO" "End time: $(date)"
log "INFO" "════════════════════════════════════════════════════════════"
log "INFO" "Output directory: ${OUTPUT_ROOT}"
log "INFO" "Full log: ${LOG_FILE}"
log "INFO" "════════════════════════════════════════════════════════════"

exit 0
