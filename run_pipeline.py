"""CLI entrypoint for the CPET batch pipeline (notebook-free execution)."""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger("run_pipeline")

DEFAULT_EDA_METRICS: List[str] = ["HR", "VO2", "VCO2", "Power", "RER", "SpO2"]

plt = None
np = None
pd = None
tqdm = None
fn = None


def _load_runtime_dependencies() -> None:
    """Import runtime dependencies lazily so --help works without installed packages."""
    global plt, np, pd, tqdm, fn
    if fn is not None:
        return
    try:
        import matplotlib.pyplot as _plt
        import numpy as _np
        import pandas as _pd
        from tqdm.auto import tqdm as _tqdm

        import functions as _fn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install requirements with "
            "'pip install -r requirements.txt'."
        ) from exc

    plt = _plt
    np = _np
    pd = _pd
    tqdm = _tqdm
    fn = _fn


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the DACIL-WESENSE CPET pipeline without notebooks."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory that contains one sub-folder per patient.",
    )
    parser.add_argument(
        "--output-root",
        default="output",
        help="Directory where all generated files are written.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path (default: <output-root>/pipeline.log).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable optional CuPy acceleration when available.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process patient folders in parallel.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads used with --parallel.",
    )
    parser.add_argument(
        "--include-ecg",
        action="store_true",
        help="Include ECG feature extraction during per-patient processing.",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip PCA / clustering section.",
    )
    parser.add_argument(
        "--kmeans-clusters",
        type=int,
        default=4,
        help="Number of clusters for K-Means.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=1.0,
        help="DBSCAN eps parameter.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples parameter.",
    )
    return parser.parse_args()


def _process_patients(
    patient_folders: List[Path],
    output_base: Path,
    include_ecg: bool,
    use_parallel: bool,
    workers: int,
) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
    """Process all patient folders and return summary rows + telemetry by patient."""
    master_records: List[Dict] = []
    processed_telemetry: Dict[str, pd.DataFrame] = {}

    def _collect_result(result: Optional[Dict]) -> None:
        if result is None:
            return
        master_records.append(result["summary_row"])
        processed_telemetry[result["patient_id"]] = result["telemetry_df"]

    if use_parallel and len(patient_folders) > 1:
        n_workers = max(1, workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    fn.process_patient,
                    folder,
                    output_base,
                    skip_ecg=not include_ecg,
                ): folder
                for folder in patient_folders
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing patients",
                unit="patient",
            ):
                _collect_result(future.result())
    else:
        for folder in tqdm(
            patient_folders,
            desc="Processing patients",
            unit="patient",
        ):
            _collect_result(
                fn.process_patient(
                    folder,
                    output_base,
                    skip_ecg=not include_ecg,
                )
            )

    return master_records, processed_telemetry


def _run_eda_and_patient_plots(
    processed_telemetry: Dict[str, pd.DataFrame],
    output_base: Path,
) -> None:
    """Run per-patient EDA and physiology plots."""
    br_records: List[Dict[str, float]] = []

    for patient_id, telemetry_df in tqdm(
        processed_telemetry.items(),
        desc="Generating patient plots",
        unit="patient",
    ):
        patient_out = output_base / patient_id

        stage_summary = fn.compute_stage_summary(telemetry_df, metrics=DEFAULT_EDA_METRICS)
        stage_summary.to_csv(patient_out / f"{patient_id}_stage_summary.csv")

        fig = fn.plot_metrics_by_stage(
            telemetry_df,
            metrics=DEFAULT_EDA_METRICS,
            patient_id=patient_id,
            output_dir=patient_out,
        )
        plt.close(fig)

        for plot_fn in (
            fn.plot_ve_vco2_slope,
            fn.plot_oxygen_pulse,
            fn.plot_ventilatory_efficiency,
        ):
            fig_opt = plot_fn(telemetry_df, patient_id=patient_id, output_dir=patient_out)
            if fig_opt is not None:
                plt.close(fig_opt)

        br = fn.compute_breathing_reserve(telemetry_df)
        br["patient_id"] = patient_id
        br_records.append(br)

    if br_records:
        br_df = pd.DataFrame(br_records).set_index("patient_id")
        br_df["BR_flag"] = br_df["BR_pct"].apply(
            lambda value: (
                "ventilatory limitation"
                if (not np.isnan(value) and value < 15.0)
                else "normal"
            )
        )
        br_df.to_csv(output_base / "breathing_reserve_summary.csv")


def _run_batch_aggregates(summary_df: pd.DataFrame, output_base: Path) -> None:
    """Generate and save batch-level aggregate plots."""
    if summary_df.empty:
        return
    figures = fn.plot_batch_aggregates(summary_df, output_dir=output_base)
    for fig in figures:
        plt.close(fig)


def _run_ml_sections(
    processed_telemetry: Dict[str, pd.DataFrame],
    output_base: Path,
    kmeans_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> None:
    """Run pooled telemetry ML analysis (PCA + clustering)."""
    if not processed_telemetry:
        LOGGER.warning("No telemetry data available, skipping ML section.")
        return

    all_tele = pd.concat(
        [df.assign(patient_id=patient_id) for patient_id, df in processed_telemetry.items()],
        ignore_index=True,
    )

    try:
        x_scaled, feature_df, _ = fn.prepare_features(all_tele, drop_cols=["patient_id"])
    except ValueError as exc:
        LOGGER.warning("Unable to prepare features for ML section: %s", exc)
        return

    if x_scaled.shape[0] <= 1:
        LOGGER.warning("Insufficient samples for ML section.")
        return

    stage_labels = all_tele["Stage"].fillna("Unknown").reset_index(drop=True)

    LOGGER.info("Running PCA...")
    pca_model, x_pca = fn.run_pca(x_scaled, n_components=min(10, x_scaled.shape[1]))
    fig_scree = fn.plot_scree(pca_model, output_dir=output_base)
    fig_pca_2d = fn.plot_pca_scatter(
        x_pca, stage_labels, label_name="Stage", output_dir=output_base
    )
    plt.close(fig_scree)
    plt.close(fig_pca_2d)

    if x_pca.shape[1] >= 3:
        fig_pca_3d = fn.plot_pca_scatter(
            x_pca,
            stage_labels,
            label_name="Stage",
            output_dir=output_base,
            three_d=True,
        )
        plt.close(fig_pca_3d)

    if x_scaled.shape[0] > 10:
        LOGGER.info("Generating elbow plot...")
        max_k = min(10, x_scaled.shape[0] - 1)
        fig_elbow = fn.elbow_plot(
            x_scaled,
            k_range=range(2, max_k + 1),
            output_dir=output_base,
        )
        plt.close(fig_elbow)

    if feature_df.shape[1] < 2:
        LOGGER.warning("Insufficient feature columns for cluster scatter plots.")
        return

    vo2_col = fn._find_column(feature_df, ["VO2", "vo2", "VO2_max"])
    hr_col = fn._find_column(feature_df, ["HR", "hr", "Heart Rate"])
    x_vis = vo2_col or feature_df.columns[0]
    y_vis = hr_col or feature_df.columns[min(1, len(feature_df.columns) - 1)]

    if x_scaled.shape[0] >= kmeans_clusters:
        LOGGER.info("Running K-Means clustering...")
        _, km_labels = fn.run_kmeans(x_scaled, n_clusters=kmeans_clusters)
        fig_km = fn.plot_cluster_scatter(
            feature_df,
            km_labels,
            x_col=x_vis,
            y_col=y_vis,
            algorithm_name="K-Means",
            output_dir=output_base,
        )
        plt.close(fig_km)

        km_series = pd.Series(km_labels.astype(str), name="KMeans_Cluster")
        fig_pca_km = fn.plot_pca_scatter(
            x_pca,
            km_series,
            label_name="K-Means Cluster",
            output_dir=output_base,
        )
        plt.close(fig_pca_km)

    if x_scaled.shape[0] >= 10:
        LOGGER.info("Running DBSCAN clustering...")
        _, db_labels = fn.run_dbscan(
            x_scaled, eps=dbscan_eps, min_samples=dbscan_min_samples
        )
        fig_db = fn.plot_cluster_scatter(
            feature_df,
            db_labels,
            x_col=x_vis,
            y_col=y_vis,
            algorithm_name="DBSCAN",
            output_dir=output_base,
        )
        plt.close(fig_db)


def main() -> int:
    """Run the CPET pipeline from command line."""
    args = parse_args()
    _load_runtime_dependencies()

    data_root = Path(args.data_root)
    output_base = Path(args.output_root)
    output_base.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else output_base / "pipeline.log"
    fn.setup_logging(log_file=str(log_path), level=getattr(logging, args.log_level))
    fn.configure(use_gpu=args.use_gpu)

    if not data_root.exists():
        LOGGER.error("Data root does not exist: %s", data_root)
        return 1

    patient_folders = fn.discover_patient_folders(str(data_root))
    LOGGER.info("Found %d patient folder(s).", len(patient_folders))
    if not patient_folders:
        LOGGER.warning("No patient folders found under %s.", data_root)
        return 0

    master_records, processed_telemetry = _process_patients(
        patient_folders=patient_folders,
        output_base=output_base,
        include_ecg=args.include_ecg,
        use_parallel=args.parallel,
        workers=args.workers,
    )
    LOGGER.info(
        "Batch complete. Successfully processed %d patient(s).", len(master_records)
    )
    if not master_records:
        LOGGER.warning("No patients were processed successfully.")
        return 1

    summary_df = pd.DataFrame(master_records)
    master_csv = output_base / "master_summary.csv"
    summary_df.to_csv(master_csv, index=False)
    LOGGER.info("Saved master summary: %s", master_csv)
    status_df = fn.build_patient_analysis_status_table(summary_df)
    status_csv = output_base / "patient_analysis_status.csv"
    status_df.to_csv(status_csv, index=False)
    LOGGER.info("Saved patient analysis status table: %s", status_csv)

    _run_eda_and_patient_plots(processed_telemetry, output_base)
    _run_batch_aggregates(summary_df, output_base)

    if not args.skip_ml:
        _run_ml_sections(
            processed_telemetry=processed_telemetry,
            output_base=output_base,
            kmeans_clusters=args.kmeans_clusters,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
        )

    LOGGER.info("Pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
