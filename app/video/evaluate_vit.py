"""
Quantitative evaluation for the independent ViT emotion classifier.

Expected dataset layout:

    dataset_root/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/

Run:

    python -m app.video.evaluate_vit --data-root path\to\dataset

The evaluator uses the checkpoint's own image processor, raw logits, and
standard softmax. It never uses webcam frames and does not fine-tune models.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "mo-thecreator/vit-Facial-Expression-Recognition"
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_SYNONYMS = {"anger": "angry"}


def _discover_images(data_root: Path) -> Tuple[List[Path], List[int]]:
    paths: List[Path] = []
    targets: List[int] = []
    for label_index, label in enumerate(LABELS):
        label_dir = data_root / label
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)
                targets.append(label_index)
    return paths, targets


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_visualizations(
    output_dir: Path,
    matrix: np.ndarray,
    per_class: Dict[str, Dict[str, float]],
    fear_rows: List[Dict[str, object]],
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(
        title="ViT Emotion Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(len(LABELS)),
        yticks=range(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(LABELS))
    width = 0.25
    for offset, metric in enumerate(("precision", "recall", "f1-score")):
        axis.bar(
            x + (offset - 1) * width,
            [per_class[label][metric] for label in LABELS],
            width,
            label=metric,
        )
    axis.set(
        title="Per-Class Precision, Recall, and F1",
        ylabel="Score",
        xticks=x,
        xticklabels=LABELS,
        ylim=(0, 1),
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_metrics.png", dpi=160)
    plt.close(fig)

    fear_probabilities = [
        float(row["fear_probability"])
        for row in fear_rows
        if row["true_label"] == "fear"
    ]
    if fear_probabilities:
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.hist(fear_probabilities, bins=20, range=(0, 1))
        axis.set(
            title="Fear Probability Distribution for Actual Fear Samples",
            xlabel="Raw fear probability",
            ylabel="Number of samples",
        )
        fig.tight_layout()
        fig.savefig(output_dir / "fear_probability_distribution.png", dpi=160)
        plt.close(fig)
    return True


def evaluate(
    data_root: Path,
    output_dir: Path,
    batch_size: int,
    local_files_only: bool,
) -> Dict[str, object]:
    image_paths, targets = _discover_images(data_root)
    if not image_paths:
        raise ValueError(
            f"No labeled images found under {data_root}. Expected one directory "
            f"per class: {', '.join(LABELS)}."
        )

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        local_files_only=local_files_only,
    )
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        local_files_only=local_files_only,
    )
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    model_mapping = {
        int(index): LABEL_SYNONYMS.get(label.strip().lower(), label.strip().lower())
        for index, label in model.config.id2label.items()
    }
    if set(model_mapping.values()) != set(LABELS):
        raise ValueError(f"Checkpoint labels {model_mapping} do not match {LABELS}.")
    label_to_model_index = {label: index for index, label in model_mapping.items()}

    predictions: List[int] = []
    probabilities: List[np.ndarray] = []
    skipped: List[Dict[str, str]] = []

    for start in range(0, len(image_paths), max(1, batch_size)):
        batch_paths = image_paths[start:start + max(1, batch_size)]
        images = []
        valid_paths = []
        for path in batch_paths:
            try:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
                valid_paths.append(path)
            except (OSError, UnidentifiedImageError) as exc:
                skipped.append({"path": str(path), "error": str(exc)})

        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            batch_probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        for row, path in zip(batch_probabilities, valid_paths):
            probabilities.append(row)
            predictions.append(int(np.argmax(row)))

    valid_targets = [
        target for path, target in zip(image_paths, targets)
        if str(path) not in {entry["path"] for entry in skipped}
    ]
    if len(valid_targets) != len(predictions):
        raise RuntimeError("Prediction and target counts diverged.")

    ordered_model_indices = [label_to_model_index[label] for label in LABELS]
    normalized_probabilities = np.asarray(probabilities)[:, ordered_model_indices]
    normalized_predictions = np.argmax(normalized_probabilities, axis=1)
    matrix = confusion_matrix(valid_targets, normalized_predictions, labels=range(len(LABELS)))
    report = classification_report(
        valid_targets,
        normalized_predictions,
        labels=range(len(LABELS)),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        valid_targets,
        normalized_predictions,
        labels=range(len(LABELS)),
        zero_division=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fear_rows = []
    for path, target, prediction, row in zip(
        image_paths,
        valid_targets,
        normalized_predictions,
        normalized_probabilities,
    ):
        fear_rows.append({
            "path": str(path),
            "true_label": LABELS[target],
            "predicted_label": LABELS[prediction],
            "fear_probability": float(row[LABELS.index("fear")]),
            "surprise_probability": float(row[LABELS.index("surprise")]),
            "sad_probability": float(row[LABELS.index("sad")]),
            "neutral_probability": float(row[LABELS.index("neutral")]),
            "disgust_probability": float(row[LABELS.index("disgust")]),
            "correct": bool(target == prediction),
        })

    metrics = {
        "model_id": MODEL_ID,
        "dataset_root": str(data_root),
        "labels": LABELS,
        "model_label_mapping": model_mapping,
        "preprocessing": {
            "size": {"height": 224, "width": 224},
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "softmax": "standard softmax over model logits",
        },
        "sample_count": len(valid_targets),
        "skipped_count": len(skipped),
        "samples_per_class": {
            label: int(sum(target == index for target in valid_targets))
            for index, label in enumerate(LABELS)
        },
        "accuracy": float(accuracy_score(valid_targets, normalized_predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(valid_targets, normalized_predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(valid_targets, normalized_predictions, average="weighted", zero_division=0)),
        "fear": {
            "precision": float(precision[LABELS.index("fear")]),
            "recall": float(recall[LABELS.index("fear")]),
            "f1": float(f1[LABELS.index("fear")]),
            "support": int(support[LABELS.index("fear")]),
            "false_positives": int(matrix[:, LABELS.index("fear")].sum() - matrix[LABELS.index("fear"), LABELS.index("fear")]),
            "false_negatives": int(matrix[LABELS.index("fear"), :].sum() - matrix[LABELS.index("fear"), LABELS.index("fear")]),
            "predicted_as_surprise": int(matrix[LABELS.index("fear"), LABELS.index("surprise")]),
            "predicted_as_sad": int(matrix[LABELS.index("fear"), LABELS.index("sad")]),
            "predicted_as_neutral": int(matrix[LABELS.index("fear"), LABELS.index("neutral")]),
            "predicted_as_disgust": int(matrix[LABELS.index("fear"), LABELS.index("disgust")]),
        },
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1-score": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": matrix.tolist(),
        "skipped": skipped,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    _write_csv(
        output_dir / "sample_predictions.csv",
        fear_rows,
        [
            "path", "true_label", "predicted_label", "fear_probability",
            "surprise_probability", "sad_probability", "neutral_probability",
            "disgust_probability", "correct",
        ],
    )
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted"] + LABELS)
        writer.writerows(
            [[LABELS[index]] + list(row) for index, row in enumerate(matrix)]
        )
    if not _save_visualizations(output_dir, matrix, report, fear_rows):
        print("matplotlib is unavailable; numeric artifacts were still written.")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the MindBloom ViT on labeled emotion images.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("vit_evaluation"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face to download missing model files.",
    )
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        metrics = evaluate(
            args.data_root,
            args.output_dir,
            args.batch_size,
            local_files_only=not args.allow_download,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
