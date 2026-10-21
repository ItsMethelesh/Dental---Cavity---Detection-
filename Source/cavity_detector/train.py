from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import XrayDataset, build_records, stratified_split
from .model import CavityCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _binary_metrics(probabilities: np.ndarray, targets: np.ndarray, threshold: float) -> dict:
    preds = (probabilities >= threshold).astype(np.int32)
    targets_i = targets.astype(np.int32)

    tp = int(np.sum((preds == 1) & (targets_i == 1)))
    tn = int(np.sum((preds == 0) & (targets_i == 0)))
    fp = int(np.sum((preds == 1) & (targets_i == 0)))
    fn = int(np.sum((preds == 0) & (targets_i == 1)))
    total = max(1, tp + tn + fp + fn)

    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _evaluate(
    model: CavityCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> dict:
    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            probs = torch.sigmoid(logits).cpu().numpy()
            probabilities.append(probs)
            targets.append(labels.cpu().numpy())

    if not probabilities:
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    prob_array = np.concatenate(probabilities, axis=0).ravel()
    target_array = np.concatenate(targets, axis=0).ravel()
    metrics = _binary_metrics(prob_array, target_array, threshold=threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def _write_training_records(records_path: Path, train_records, val_records) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "image", "label", "score"])
        for split_name, split_records in (("train", train_records), ("val", val_records)):
            for record in split_records:
                writer.writerow([split_name, record.path.name, int(record.label), float(record.score)])


def train_model(
    data_dir: str | Path = "data",
    labels_csv: str | Path | None = None,
    model_output_path: str | Path = "models/cavity_cnn.pt",
    image_size: int = 224,
    epochs: int = 14,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.2,
    pseudo_label_quantile: float = 0.62,
    max_samples: int | None = None,
    random_seed: int = 42,
    threshold: float = 0.5,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    set_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records, metadata = build_records(
        data_dir=data_dir,
        labels_csv=labels_csv,
        pseudo_label_quantile=pseudo_label_quantile,
        max_samples=max_samples,
        random_seed=random_seed,
    )
    if len(records) < 10:
        raise ValueError("Not enough images to train. Add more samples to the data folder.")

    train_records, val_records = stratified_split(
        records=records, val_ratio=val_ratio, random_seed=random_seed
    )
    if not val_records:
        raise ValueError("Validation split is empty. Increase dataset size or val_ratio.")

    train_dataset = XrayDataset(records=train_records, image_size=image_size, augment=True)
    val_dataset = XrayDataset(records=val_records, image_size=image_size, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    model = CavityCNN().to(device)

    positives = max(1, sum(record.label for record in train_records))
    negatives = max(1, len(train_records) - positives)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_f1 = -1.0
    best_state: dict | None = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_metrics = _evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
        )
        scheduler.step(val_metrics["f1"])

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_stats)

        if progress_callback is not None:
            progress_callback(epoch_stats)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {
                "state_dict": model.state_dict(),
                "image_size": image_size,
                "threshold": threshold,
                "metadata": metadata,
            }

    if best_state is None:
        best_state = {
            "state_dict": model.state_dict(),
            "image_size": image_size,
            "threshold": threshold,
            "metadata": metadata,
        }

    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_output_path)

    training_log_path = model_output_path.with_suffix(".history.json")
    with training_log_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    records_path = model_output_path.with_suffix(".records.csv")
    _write_training_records(records_path, train_records=train_records, val_records=val_records)

    final_summary = {
        "model_path": str(model_output_path),
        "history_path": str(training_log_path),
        "records_path": str(records_path),
        "best_val_f1": float(best_f1),
        "epochs": epochs,
        "num_train": len(train_records),
        "num_val": len(val_records),
        "label_source": metadata.get("label_source", "unknown"),
        "class_balance": metadata.get("class_balance", {}),
        "device": str(device),
    }
    return final_summary

