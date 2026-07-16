"""Reproducible data preparation and lightweight exploratory analysis."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


TEXT_ALIASES = ("consumer complaint narrative", "narrative")
LABEL_ALIASES = ("product",)
TOKEN_PATTERN = re.compile(r"[a-zA-Z]{2,}")


def _find_column(columns: Iterable[str], aliases: tuple[str, ...]) -> str:
    normalized = {column.strip().casefold(): column for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    raise ValueError(f"Required column not found. Expected one of: {aliases}")


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def load_and_clean(raw_path: Path) -> tuple[list[dict[str, str]], dict[str, int | str]]:
    """Select source fields, remove blank rows and de-duplicate selected records."""
    with raw_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        text_column = _find_column(fieldnames, TEXT_ALIASES)
        label_column = _find_column(fieldnames, LABEL_ALIASES)
        raw_rows = list(reader)

    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    dropped_null = 0
    dropped_duplicates = 0
    for row in raw_rows:
        narrative = _clean_text(row.get(text_column, ""))
        product = _clean_text(row.get(label_column, ""))
        if not narrative or not product:
            dropped_null += 1
            continue
        key = (narrative, product)
        if key in seen:
            dropped_duplicates += 1
            continue
        seen.add(key)
        cleaned.append({"narrative": narrative, "product": product})

    return cleaned, {
        "source_rows": len(raw_rows),
        "kept_rows": len(cleaned),
        "dropped_null_rows": dropped_null,
        "dropped_duplicate_rows": dropped_duplicates,
        "text_source_column": text_column,
        "label_source_column": label_column,
    }


def stratified_split(
    rows: list[dict[str, str]], seed: int = 42
) -> dict[str, list[dict[str, str]]]:
    """Create deterministic 70/15/15 splits while retaining every class in each split."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["product"], []).append(row)

    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for label, group in grouped.items():
        if len(group) < 3:
            raise ValueError(f"Class {label!r} has fewer than three rows and cannot be stratified.")
        group = group.copy()
        rng.shuffle(group)
        validation_count = max(1, round(len(group) * 0.15))
        test_count = max(1, round(len(group) * 0.15))
        train_count = len(group) - validation_count - test_count
        if train_count < 1:
            raise ValueError(f"Class {label!r} has too few rows for train/validation/test splits.")
        splits["train"].extend(group[:train_count])
        splits["validation"].extend(group[train_count : train_count + validation_count])
        splits["test"].extend(group[train_count + validation_count :])

    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


def make_eda(rows: list[dict[str, str]]) -> dict[str, object]:
    """Build JSON-serializable distribution, length and frequency summaries."""
    counts = Counter(row["product"] for row in rows)
    lengths = [len(row["narrative"].split()) for row in rows]
    words = Counter(
        word.lower()
        for row in rows
        for word in TOKEN_PATTERN.findall(row["narrative"])
    )
    largest = max(counts.values())
    smallest = min(counts.values())
    return {
        "class_distribution": dict(counts.most_common()),
        "class_count": len(counts),
        "imbalance_ratio_max_to_min": round(largest / smallest, 3),
        "class_weighting_recommended": smallest / largest < 0.5,
        "balanced_class_weights": {
            label: round(len(rows) / (len(counts) * count), 6)
            for label, count in sorted(counts.items())
        },
        "complaint_length_words": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 2),
            "median": sorted(lengths)[len(lengths) // 2],
        },
        "top_30_words": words.most_common(30),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["narrative", "product"])
        writer.writeheader()
        writer.writerows(rows)


def prepare_project(raw_path: Path, output_dir: Path, report_path: Path) -> dict[str, object]:
    rows, quality = load_and_clean(raw_path)
    splits = stratified_split(rows)
    for name, split_rows in splits.items():
        write_csv(output_dir / f"{name}.csv", split_rows)

    report: dict[str, object] = {
        "data_quality": quality,
        "eda": make_eda(rows),
        "split_sizes": {name: len(split_rows) for name, split_rows in splits.items()},
        "split_seed": 42,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
