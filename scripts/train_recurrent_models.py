"""Train the required SimpleRNN, LSTM and GRU models on prepared splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from complaint_app.models import build_recurrent_model
from complaint_app.preprocessing import CustomPreprocessor, load_split


ROOT = Path(__file__).resolve().parents[1]


def main(epochs: int, batch_size: int, architectures: tuple[str, ...]) -> None:
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow import keras

    train = load_split(ROOT / "data" / "prepared" / "train.csv")
    validation = load_split(ROOT / "data" / "prepared" / "validation.csv")
    preprocessor = CustomPreprocessor().fit(train["clean_narrative"], train["product"])
    x_train = preprocessor.transform_texts(train["clean_narrative"])
    x_validation = preprocessor.transform_texts(validation["clean_narrative"])
    y_train = preprocessor.encode_labels(train["product"])
    y_validation = preprocessor.encode_labels(validation["product"])
    classes = np.arange(preprocessor.num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {index: float(weight) for index, weight in enumerate(weights)}

    preprocessor.save(ROOT / "artifacts" / "recurrent_preprocessor")
    report_file = ROOT / "reports" / "recurrent_training.json"
    if report_file.exists():
        metrics: dict[str, dict[str, float]] = json.loads(report_file.read_text(encoding="utf-8"))
    else:
        metrics = {}
    for architecture in architectures:
        model = build_recurrent_model(
            architecture, preprocessor.vocabulary_size, preprocessor.max_length, preprocessor.num_classes
        )
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ]
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            class_weight=class_weight,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        model.save(ROOT / "artifacts" / f"{architecture}.keras")
        metrics[architecture] = {
            key: float(values[-1]) for key, values in history.history.items() if values
        }

        (ROOT / "reports").mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("simple_rnn", "lstm", "gru"),
        default=("simple_rnn", "lstm", "gru"),
    )
    args = parser.parse_args()
    main(args.epochs, args.batch_size, tuple(args.architectures))
