"""Text preprocessing shared by recurrent and transformer model workflows."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WHITESPACE = re.compile(r"\s+")
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_WORD = re.compile(r"[^a-z0-9\s']")


def clean_text(text: str) -> str:
    """Normalize a complaint without discarding meaningful word boundaries."""
    text = URL.sub(" ", str(text).lower())
    text = NON_WORD.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def load_split(path: str | Path):
    """Read a prepared split, applying the custom-model text cleaning step."""
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"narrative", "product"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prepared split is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["narrative", "product"]).copy()
    frame["clean_narrative"] = frame["narrative"].map(clean_text)
    return frame[frame["clean_narrative"].str.len() > 0].reset_index(drop=True)


@dataclass
class CustomPreprocessor:
    """Tokenizer, sequence padding and label encoding for recurrent networks."""

    vocabulary_size: int = 20_000
    max_length: int = 256
    tokenizer: object | None = None
    label_encoder: object | None = None

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "CustomPreprocessor":
        from sklearn.preprocessing import LabelEncoder
        from tensorflow.keras.preprocessing.text import Tokenizer

        cleaned_texts = [clean_text(text) for text in texts]
        self.tokenizer = Tokenizer(num_words=self.vocabulary_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(cleaned_texts)
        self.label_encoder = LabelEncoder().fit(list(labels))
        return self

    def transform_texts(self, texts: Iterable[str]):
        """Tokenize and post-pad text to a fixed-length matrix."""
        if self.tokenizer is None:
            raise RuntimeError("Fit or load the preprocessor before transforming text.")
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        sequences = self.tokenizer.texts_to_sequences([clean_text(text) for text in texts])
        return pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post")

    def encode_labels(self, labels: Iterable[str]):
        if self.label_encoder is None:
            raise RuntimeError("Fit or load the preprocessor before encoding labels.")
        return self.label_encoder.transform(list(labels))

    def decode_labels(self, encoded_labels: Iterable[int]) -> list[str]:
        if self.label_encoder is None:
            raise RuntimeError("Fit or load the preprocessor before decoding labels.")
        return self.label_encoder.inverse_transform(list(encoded_labels)).tolist()

    @property
    def num_classes(self) -> int:
        if self.label_encoder is None:
            raise RuntimeError("Fit or load the preprocessor before requesting class count.")
        return len(self.label_encoder.classes_)

    def save(self, directory: str | Path) -> None:
        if self.tokenizer is None or self.label_encoder is None:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text(self.tokenizer.to_json(), encoding="utf-8")
        (directory / "preprocessing_config.json").write_text(
            json.dumps({"vocabulary_size": self.vocabulary_size, "max_length": self.max_length}, indent=2),
            encoding="utf-8",
        )
        joblib.dump(self.label_encoder, directory / "label_encoder.joblib")

    @classmethod
    def load(cls, directory: str | Path) -> "CustomPreprocessor":
        import joblib
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        directory = Path(directory)
        config = json.loads((directory / "preprocessing_config.json").read_text(encoding="utf-8"))
        instance = cls(**config)
        instance.tokenizer = tokenizer_from_json((directory / "tokenizer.json").read_text(encoding="utf-8"))
        instance.label_encoder = joblib.load(directory / "label_encoder.joblib")
        return instance


def build_transformer_tokenizer(model_name: str = "distilbert-base-uncased"):
    """Load the Hugging Face tokenizer used by the DistilBERT workflow."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)
