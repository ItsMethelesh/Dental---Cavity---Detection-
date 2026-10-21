from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    torch = None

    class Dataset:  # type: ignore[override]
        pass

from .preprocessing import detect_cavity_candidates, preprocess_for_model, read_xray_grayscale


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(slots=True)
class SampleRecord:
    path: Path
    label: int
    score: float


def natural_key(path: Path) -> tuple:
    name = path.stem
    prefix = "".join(char for char in name if not char.isdigit())
    suffix = "".join(char for char in name if char.isdigit())
    number = int(suffix) if suffix else 0
    return prefix, number, path.name


def list_image_paths(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    images = [p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort(key=natural_key)
    return images


def deduplicate_images(paths: Iterable[Path]) -> list[Path]:
    seen_hashes: set[str] = set()
    unique_paths: list[Path] = []
    for path in paths:
        content = path.read_bytes()
        digest = hashlib.md5(content).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        unique_paths.append(path)
    return unique_paths


def _read_labels_csv(labels_csv: Path) -> dict[str, int]:
    label_map: dict[str, int] = {}
    with labels_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        original_fieldnames = reader.fieldnames or []
        if not original_fieldnames:
            raise ValueError(f"labels file is empty: {labels_csv}")
        normalized_fields = {
            field.lower().strip(): field for field in original_fieldnames if field is not None
        }

        image_field_candidates = ("image", "filename", "file", "path")
        label_field_candidates = ("label", "target", "class", "has_cavity")
        image_field = next(
            (normalized_fields[f] for f in image_field_candidates if f in normalized_fields),
            None,
        )
        label_field = next(
            (normalized_fields[f] for f in label_field_candidates if f in normalized_fields),
            None,
        )
        if image_field is None or label_field is None:
            raise ValueError(
                "labels.csv needs columns for image and label, e.g. image,label."
            )

        for row in reader:
            image_name = row.get(image_field, "").strip()
            raw_label = row.get(label_field, "").strip().lower()
            if not image_name or raw_label == "":
                continue
            if raw_label in {"1", "true", "yes", "cavity", "caries"}:
                label = 1
            elif raw_label in {"0", "false", "no", "healthy", "normal"}:
                label = 0
            else:
                try:
                    label = int(float(raw_label))
                    label = 1 if label >= 1 else 0
                except ValueError as exc:
                    raise ValueError(
                        f"Unsupported label value '{raw_label}' for image '{image_name}'."
                    ) from exc
            label_map[Path(image_name).name] = label
    return label_map


def build_records(
    data_dir: str | Path,
    labels_csv: str | Path | None = None,
    pseudo_label_quantile: float = 0.62,
    dedupe: bool = True,
    max_samples: int | None = None,
    random_seed: int = 42,
) -> tuple[list[SampleRecord], dict]:
    image_paths = list_image_paths(data_dir)
    if dedupe:
        image_paths = deduplicate_images(image_paths)
    if not image_paths:
        return [], {"label_source": "none", "threshold": None, "class_balance": {}}

    if max_samples is not None and max_samples > 0 and len(image_paths) > max_samples:
        rng = random.Random(random_seed)
        image_paths = rng.sample(image_paths, k=max_samples)
        image_paths.sort(key=natural_key)

    labels_csv_path = Path(labels_csv) if labels_csv else None
    if labels_csv_path and labels_csv_path.exists():
        label_map = _read_labels_csv(labels_csv_path)
        records: list[SampleRecord] = []
        unmatched = 0
        for path in image_paths:
            filename = path.name
            if filename not in label_map:
                unmatched += 1
                continue
            label = int(label_map[filename])
            gray = read_xray_grayscale(path)
            _, score = detect_cavity_candidates(gray)
            records.append(SampleRecord(path=path, label=label, score=score))
        metadata = {
            "label_source": "labels_csv",
            "threshold": None,
            "unmatched_images": unmatched,
        }
    else:
        scores: list[float] = []
        for path in image_paths:
            gray = read_xray_grayscale(path)
            _, score = detect_cavity_candidates(gray)
            scores.append(score)
        threshold = float(np.quantile(np.array(scores, dtype=np.float32), pseudo_label_quantile))
        records = [
            SampleRecord(path=path, label=int(score >= threshold), score=score)
            for path, score in zip(image_paths, scores)
        ]
        metadata = {
            "label_source": "pseudo_labels",
            "threshold": threshold,
            "quantile": pseudo_label_quantile,
        }

    positives = sum(record.label for record in records)
    negatives = len(records) - positives
    metadata["class_balance"] = {"positive": positives, "negative": negatives}
    return records, metadata


def stratified_split(
    records: list[SampleRecord], val_ratio: float = 0.2, random_seed: int = 42
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if len(records) < 2:
        return records, []

    positives = [record for record in records if record.label == 1]
    negatives = [record for record in records if record.label == 0]

    rng = random.Random(random_seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    if positives and negatives:
        val_pos = max(1, int(len(positives) * val_ratio))
        val_neg = max(1, int(len(negatives) * val_ratio))
        val = positives[:val_pos] + negatives[:val_neg]
        train = positives[val_pos:] + negatives[val_neg:]
    else:
        shuffled = records.copy()
        rng.shuffle(shuffled)
        pivot = max(1, int(len(shuffled) * (1.0 - val_ratio)))
        train = shuffled[:pivot]
        val = shuffled[pivot:]

    if not train or not val:
        pivot = max(1, int(len(records) * (1.0 - val_ratio)))
        train = records[:pivot]
        val = records[pivot:]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


class XrayDataset(Dataset):
    def __init__(self, records: list[SampleRecord], image_size: int, augment: bool = False):
        if torch is None:
            raise RuntimeError(
                "PyTorch is required to use XrayDataset. Install dependencies from requirements.txt."
            )
        self.records = records
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
        if random.random() < 0.4:
            image = _rotate_image(image, random.uniform(-7.0, 7.0))
        if random.random() < 0.3:
            gamma = random.uniform(0.85, 1.2)
            image = np.clip(image ** gamma, 0.0, 1.0)
        if random.random() < 0.3:
            noise = np.random.normal(loc=0.0, scale=0.02, size=image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)
        return image

    def __getitem__(self, index: int):
        record = self.records[index]
        gray = read_xray_grayscale(record.path)
        image = preprocess_for_model(gray, self.image_size)
        if self.augment:
            image = self._augment(image)

        tensor = torch.from_numpy(image).unsqueeze(0).float()
        label = torch.tensor([float(record.label)], dtype=torch.float32)
        return tensor, label
