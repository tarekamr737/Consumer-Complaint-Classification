"""A fast, probability-producing baseline for the local Gradio experience."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib

from complaint_app.preprocessing import clean_text, load_split


def train_baseline(train_path: Path, test_path: Path, artifact_path: Path, report_dir: Path) -> dict[str, float]:
    """Fit a balanced TF-IDF logistic-regression model and persist evaluation evidence."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    from sklearn.pipeline import Pipeline

    train = load_split(train_path)
    test = load_split(test_path)
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    token_pattern=r"(?u)\b\w+\b",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=350,
                    solver="saga",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(train["narrative"], train["product"])
    predictions = pipeline.predict(test["narrative"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        test["product"], predictions, average="weighted", zero_division=0
    )
    metrics = {
        "model": "tfidf_logistic_regression_baseline",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "test_accuracy": round(float(accuracy_score(test["product"], predictions)), 4),
        "test_precision_weighted": round(float(precision), 4),
        "test_recall_weighted": round(float(recall), 4),
        "test_f1_weighted": round(float(f1), 4),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics}, artifact_path)

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (report_dir / "baseline_classification_report.json").write_text(
        json.dumps(classification_report(test["product"], predictions, output_dict=True, zero_division=0), indent=2),
        encoding="utf-8",
    )
    labels = list(pipeline.classes_)
    matrix = confusion_matrix(test["product"], predictions, labels=labels)
    (report_dir / "baseline_confusion_matrix.csv").write_text(
        "product," + ",".join(labels) + "\n"
        + "\n".join(f"{label}," + ",".join(map(str, row)) for label, row in zip(labels, matrix))
        + "\n",
        encoding="utf-8",
    )
    return metrics


class BaselineClassifier:
    """Loads a fitted local baseline and returns ranked predictions."""

    def __init__(self, artifact_path: Path):
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        artifact = joblib.load(artifact_path)
        self.pipeline = artifact["pipeline"]
        self.metrics = artifact["metrics"]

    def predict(self, narrative: str) -> list[dict[str, float | str]]:
        probabilities = self.pipeline.predict_proba([narrative])[0]
        ranked = sorted(
            zip(self.pipeline.classes_, probabilities, strict=True), key=lambda item: item[1], reverse=True
        )
        return [
            {"category": str(category), "confidence": round(float(probability) * 100, 1)}
            for category, probability in ranked
        ]
