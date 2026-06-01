"""Image preprocessing for scanned Thai documents.

Pipeline:
  VLM backends  (Gemini, Typhoon): deskew → resize to 1800 px
  Tesseract backend (Gemma):       deskew → grayscale → denoise → binarise
"""
from __future__ import annotations

import cv2
import numpy as np
import PIL.Image as PILModule
from PIL.Image import Image as PILImage


def pil_to_cv2(img: PILImage) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(arr: np.ndarray) -> PILImage:
    return PILModule.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, method: str = "nlm") -> np.ndarray:
    if method == "nlm":
        return cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def binarise(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )


def detect_skew_angle(gray: np.ndarray) -> float:
    """Detect document skew via dilated text-line contours. Returns degrees."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    angles = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]
        if angle < -45:
            angle += 90
        angles.append(angle)

    return float(np.median(angles)) if angles else 0.0


def deskew(arr: np.ndarray, angle: float | None = None) -> np.ndarray:
    gray = to_grayscale(arr) if len(arr.shape) == 3 else arr
    if angle is None:
        angle = detect_skew_angle(gray)

    if abs(angle) < 0.3:
        return arr

    h, w = arr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        arr,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def resize_for_vlm(img: PILImage, max_side: int = 1800) -> PILImage:
    """Resize so the longest side is at most max_side (Typhoon OCR trained on 1800 px)."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), PILModule.Resampling.LANCZOS)


def clean_for_vlm(img: PILImage, deskew_enabled: bool = True) -> PILImage:
    """Preprocess for VLM backends (Typhoon, Gemini). No binarisation."""
    arr = pil_to_cv2(img)
    if deskew_enabled:
        arr = deskew(arr)
    return resize_for_vlm(cv2_to_pil(arr))


def clean_for_tesseract(img: PILImage, deskew_enabled: bool = True) -> PILImage:
    """Preprocess for Tesseract: deskew + denoise + binarise."""
    arr = pil_to_cv2(img)
    if deskew_enabled:
        arr = deskew(arr)
    gray = to_grayscale(arr)
    denoised = denoise(gray, method="nlm")
    binary = binarise(denoised)
    rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return PILModule.fromarray(rgb)
