from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def read_xray_grayscale(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return image


def read_xray_color(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return image


def enhance_xray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return enhanced


def build_tooth_mask(enhanced_gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(
        enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _filter_contours(mask: np.ndarray) -> np.ndarray:
    filtered = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = mask.shape[0] * mask.shape[1]
    min_area = image_area * 0.00002
    max_area = image_area * 0.02

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.03:
            continue
        cv2.drawContours(filtered, [contour], -1, 255, thickness=cv2.FILLED)
    return filtered


def detect_cavity_candidates(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    enhanced = enhance_xray(gray)
    tooth_mask = build_tooth_mask(enhanced)

    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, blackhat_kernel)
    _, dark_spots = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    laplacian = cv2.Laplacian(enhanced, cv2.CV_32F, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian)
    _, texture_edges = cv2.threshold(
        laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    candidate_mask = cv2.bitwise_and(dark_spots, texture_edges)
    candidate_mask = cv2.bitwise_and(candidate_mask, tooth_mask)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel_small)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel_medium)

    candidate_mask = _filter_contours(candidate_mask)

    tooth_area = max(1, int(np.count_nonzero(tooth_mask)))
    lesion_area = int(np.count_nonzero(candidate_mask))
    ratio = lesion_area / tooth_area
    darkness = (
        float(blackhat[tooth_mask > 0].mean()) / 255.0 if np.any(tooth_mask > 0) else 0.0
    )
    score = 0.75 * ratio + 0.25 * darkness
    return candidate_mask, float(score)


def preprocess_for_model(gray: np.ndarray, image_size: int) -> np.ndarray:
    enhanced = enhance_xray(gray)
    resized = cv2.resize(enhanced, (image_size, image_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def apply_mask_overlay(
    image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.4
) -> np.ndarray:
    if image_rgb.ndim != 3:
        raise ValueError("Expected RGB image for overlay.")
    if mask.shape[:2] != image_rgb.shape[:2]:
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))

    output = image_rgb.copy()
    layer = np.zeros_like(output, dtype=np.uint8)
    layer[:, :] = np.array(color, dtype=np.uint8)
    active = mask > 0
    output[active] = cv2.addWeighted(output[active], 1.0 - alpha, layer[active], alpha, 0)
    return output


def apply_heatmap_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    heatmap_uint8 = np.uint8(np.clip(heatmap * 255.0, 0, 255))
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, colored, alpha, 0)
    return blended

