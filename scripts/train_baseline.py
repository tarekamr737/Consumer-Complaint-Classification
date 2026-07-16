"""Train the lightweight baseline used for the working Gradio application."""

from pathlib import Path

from complaint_app.baseline import train_baseline


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    metrics = train_baseline(
        ROOT / "data" / "prepared" / "train.csv",
        ROOT / "data" / "prepared" / "test.csv",
        ROOT / "artifacts" / "baseline_classifier.joblib",
        ROOT / "reports",
    )
    print("Baseline evaluation:", metrics)
