"""Prepare modelling splits and write a concise EDA/data-quality report."""

from pathlib import Path

from complaint_app.data_pipeline import prepare_project


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    report = prepare_project(
        raw_path=ROOT / "data" / "complaints_processed.csv",
        output_dir=ROOT / "data" / "prepared",
        report_path=ROOT / "reports" / "data_summary.json",
    )
    print("Prepared splits:", report["split_sizes"])
    print("Class weighting recommended:", report["eda"]["class_weighting_recommended"])
