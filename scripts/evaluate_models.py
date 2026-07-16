"""Evaluate saved baseline and recurrent models on the held-out test split."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from complaint_app.baseline import BaselineClassifier
from complaint_app.preprocessing import CustomPreprocessor, load_split


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")


def _write_evaluation(name: str, actual, predicted, labels: list[str]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="weighted", zero_division=0)
    metrics = {
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / f"{name}_classification_report.json").write_text(
        json.dumps(classification_report(actual, predicted, output_dict=True, zero_division=0), indent=2),
        encoding="utf-8",
    )
    matrix = confusion_matrix(actual, predicted, labels=labels)
    with (report_dir / f"{name}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["product", *labels])
        writer.writerows(zip(labels, matrix))
    return metrics


def main(max_test_samples: int | None = None) -> None:
    import numpy as np
    from tensorflow import keras

    test = load_split(ROOT / "data" / "prepared" / "test.csv")
    if max_test_samples:
        test = test.sample(n=min(max_test_samples, len(test)), random_state=42)
    actual = test["product"].tolist()
    comparisons: dict[str, dict[str, float]] = {}

    baseline = BaselineClassifier(ROOT / "artifacts" / "baseline_classifier.joblib")
    baseline_predictions = baseline.pipeline.predict(test["narrative"]).tolist()
    baseline_labels = sorted({*actual, *baseline_predictions})
    comparisons["tfidf_logistic_regression_baseline"] = _write_evaluation(
        "tfidf_logistic_regression_baseline", actual, baseline_predictions, baseline_labels
    )

    preprocessor = CustomPreprocessor.load(ROOT / "artifacts" / "recurrent_preprocessor")
    x_test = preprocessor.transform_texts(test["clean_narrative"])
    recurrent_labels = preprocessor.decode_labels(np.arange(preprocessor.num_classes))
    for architecture in ("simple_rnn", "lstm", "gru"):
        model_path = ROOT / "artifacts" / f"{architecture}.keras"
        if not model_path.exists():
            continue
        model = keras.models.load_model(model_path, compile=False)
        encoded = model.predict(x_test, batch_size=256, verbose=0).argmax(axis=1)
        predicted = preprocessor.decode_labels(encoded)
        comparisons[architecture] = _write_evaluation(architecture, actual, predicted, recurrent_labels)

    distilbert_dir = ROOT / "artifacts" / "distilbert"
    if distilbert_dir.exists():
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(distilbert_dir)
        model = AutoModelForSequenceClassification.from_pretrained(distilbert_dir)
        model.eval()
        predicted_ids: list[int] = []
        narratives = test["narrative"].tolist()
        for start in range(0, len(narratives), 8):
            batch = tokenizer(narratives[start : start + 8], return_tensors="pt", padding=True, truncation=True, max_length=256)
            with torch.no_grad():
                predicted_ids.extend(model(**batch).logits.argmax(dim=1).tolist())
        predicted = [model.config.id2label[str(index)] if isinstance(model.config.id2label, dict) and str(index) in model.config.id2label else model.config.id2label[index] for index in predicted_ids]
        distilbert_labels = sorted({*actual, *predicted})
        comparisons["distilbert"] = _write_evaluation("distilbert", actual, predicted, distilbert_labels)

    best_name = max(comparisons, key=lambda name: comparisons[name]["f1_weighted"])
    summary = {"models": comparisons, "selected_model": best_name, "test_samples": len(test)}
    (ROOT / "reports" / "model_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    deployment = {
        "model_name": best_name,
        "artifact": "baseline_classifier.joblib" if best_name == "tfidf_logistic_regression_baseline" else f"{best_name}.keras",
        "preprocessing": "recurrent_preprocessor" if best_name != "tfidf_logistic_regression_baseline" else None,
        "selection_metric": "f1_weighted",
        "selection_score": comparisons[best_name]["f1_weighted"],
    }
    (ROOT / "artifacts" / "deployment_model.json").write_text(json.dumps(deployment, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()
    main(args.max_test_samples)
