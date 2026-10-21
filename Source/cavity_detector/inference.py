from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    torch = None

from .preprocessing import (
    apply_heatmap_overlay,
    apply_mask_overlay,
    detect_cavity_candidates,
    preprocess_for_model,
    read_xray_color,
    read_xray_grayscale,
)

if TYPE_CHECKING:
    from .model import CavityCNN


@dataclass(slots=True)
class ModelBundle:
    model: Any
    image_size: int
    threshold: float
    device: Any


def load_model_bundle(model_path: str | Path) -> ModelBundle | None:
    if torch is None:
        return None

    from .model import CavityCNN

    model_path = Path(model_path)
    if not model_path.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    image_size = int(checkpoint.get("image_size", 224))
    threshold = float(checkpoint.get("threshold", 0.5))
    state_dict = checkpoint["state_dict"]

    model = CavityCNN().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return ModelBundle(model=model, image_size=image_size, threshold=threshold, device=device)


def _compute_gradcam(bundle: ModelBundle, gray: np.ndarray) -> tuple[float, np.ndarray]:
    if torch is None:
        raise RuntimeError("PyTorch is required for Grad-CAM inference.")

    input_image = preprocess_for_model(gray, image_size=bundle.image_size)
    tensor = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0).to(bundle.device)
    tensor = tensor.float()

    bundle.model.zero_grad(set_to_none=True)

    logits, features = bundle.model.forward_with_features(tensor)
    features.retain_grad()
    logits[:, 0].backward(retain_graph=False)
    gradients = features.grad
    if gradients is None:
        prob = float(torch.sigmoid(logits)[0, 0].item())
        return prob, np.zeros_like(gray, dtype=np.float32)

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * features).sum(dim=1, keepdim=False))
    cam = cam[0].detach().cpu().numpy()

    cam = cv2.resize(cam, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    probability = float(torch.sigmoid(logits)[0, 0].item())
    return probability, cam.astype(np.float32)


def _count_regions(mask: np.ndarray) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def predict_scan(
    image_path: str | Path,
    model_bundle: ModelBundle | None = None,
    cam_threshold: float = 0.55,
) -> dict[str, Any]:
    gray = read_xray_grayscale(image_path)
    bgr = read_xray_color(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    cv_mask, cv_score = detect_cavity_candidates(gray)

    if model_bundle is not None:
        model_prob, heatmap = _compute_gradcam(model_bundle, gray)
        cam_mask = np.uint8(heatmap >= cam_threshold) * 255
    else:
        model_prob = None
        heatmap = np.zeros_like(gray, dtype=np.float32)
        cam_mask = np.zeros_like(gray, dtype=np.uint8)

    combined_mask = cv2.bitwise_or(cv_mask, cam_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    tooth_pixels = max(1, int(np.count_nonzero(gray > 0)))
    lesion_pixels = int(np.count_nonzero(combined_mask))
    area_ratio = lesion_pixels / tooth_pixels

    fallback_prob = min(1.0, cv_score * 9.0)
    risk_score = model_prob if model_prob is not None else fallback_prob

    heatmap_overlay = apply_heatmap_overlay(rgb, heatmap, alpha=0.42)
    candidate_overlay = apply_mask_overlay(rgb, cv_mask, color=(254, 123, 53), alpha=0.45)
    combined_overlay = apply_mask_overlay(rgb, combined_mask, color=(228, 42, 42), alpha=0.50)

    return {
        "image_path": str(image_path),
        "probability": model_prob,
        "risk_score": float(risk_score),
        "opencv_score": float(cv_score),
        "area_ratio": float(area_ratio),
        "region_count": _count_regions(combined_mask),
        "original_rgb": rgb,
        "heatmap": heatmap,
        "heatmap_overlay": heatmap_overlay,
        "candidate_overlay": candidate_overlay,
        "combined_overlay": combined_overlay,
        "candidate_mask": cv_mask,
        "combined_mask": combined_mask,
    }
