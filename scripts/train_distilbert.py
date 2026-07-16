"""Fine-tune DistilBERT for complaint-category classification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")


def main(
    epochs: float, batch_size: int, max_train_samples: int | None, max_validation_samples: int | None
) -> None:
    import numpy as np
    import torch
    from datasets import Dataset
    from sklearn.preprocessing import LabelEncoder
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    from complaint_app.preprocessing import load_split

    model_name = "distilbert-base-uncased"
    train = load_split(ROOT / "data" / "prepared" / "train.csv")
    validation = load_split(ROOT / "data" / "prepared" / "validation.csv")
    if max_train_samples:
        train = train.sample(n=min(max_train_samples, len(train)), random_state=42)
    if max_validation_samples:
        validation = validation.sample(n=min(max_validation_samples, len(validation)), random_state=42)
    encoder = LabelEncoder().fit(train["product"])
    train = train.assign(labels=encoder.transform(train["product"]))
    validation = validation.assign(labels=encoder.transform(validation["product"]))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["narrative"], truncation=True, max_length=256)

    train_dataset = Dataset.from_pandas(train[["narrative", "labels"]], preserve_index=False).map(tokenize, batched=True)
    validation_dataset = Dataset.from_pandas(validation[["narrative", "labels"]], preserve_index=False).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(encoder.classes_),
        id2label={index: label for index, label in enumerate(encoder.classes_)},
        label2id={label: index for index, label in enumerate(encoder.classes_)},
    )
    counts = np.bincount(train["labels"], minlength=len(encoder.classes_))
    class_weights = torch.tensor(len(train) / (len(counts) * counts), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    output_dir = ROOT / "artifacts" / "distilbert"
    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=25,
        report_to=[],
        fp16=False,
    )
    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    (output_dir / "label_classes.json").write_text(json.dumps(encoder.classes_.tolist()), encoding="utf-8")
    (ROOT / "reports" / "distilbert_training.json").write_text(
        json.dumps({"model": model_name, "epochs": epochs, "train_rows": len(train), "validation_rows": len(validation)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.max_train_samples, args.max_validation_samples)
