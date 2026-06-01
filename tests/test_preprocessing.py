"""Tests for image preprocessing utilities (no PDF required)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILModule

from document_ocr.preprocessing.image_cleaner import (
    binarise,
    clean_for_tesseract,
    clean_for_vlm,
    cv2_to_pil,
    denoise,
    detect_skew_angle,
    deskew,
    pil_to_cv2,
    resize_for_vlm,
    to_grayscale,
)


def _make_white_image(width: int = 400, height: int = 300) -> PILModule.Image:
    return PILModule.fromarray(np.full((height, width, 3), 255, dtype=np.uint8))


def _make_grey_image(width: int = 400, height: int = 300) -> PILModule.Image:
    return PILModule.fromarray(np.full((height, width, 3), 128, dtype=np.uint8))


class TestConversions:
    def test_pil_to_cv2_and_back(self) -> None:
        img = _make_grey_image()
        arr = pil_to_cv2(img)
        assert arr.shape == (300, 400, 3)
        restored = cv2_to_pil(arr)
        assert restored.size == (400, 300)

    def test_to_grayscale(self) -> None:
        arr = pil_to_cv2(_make_grey_image())
        gray = to_grayscale(arr)
        assert len(gray.shape) == 2


class TestDenoise:
    def test_nlm_denoise_returns_same_shape(self) -> None:
        arr = pil_to_cv2(_make_grey_image())
        gray = to_grayscale(arr)
        result = denoise(gray, method="nlm")
        assert result.shape == gray.shape

    def test_gaussian_denoise(self) -> None:
        arr = pil_to_cv2(_make_grey_image())
        gray = to_grayscale(arr)
        result = denoise(gray, method="gaussian")
        assert result.shape == gray.shape


class TestBinarise:
    def test_binarise_returns_binary_image(self) -> None:
        arr = pil_to_cv2(_make_grey_image())
        gray = to_grayscale(arr)
        binary = binarise(gray)
        unique = set(np.unique(binary))
        assert unique.issubset({0, 255})


class TestSkewDetection:
    def test_white_image_returns_zero_angle(self) -> None:
        arr = pil_to_cv2(_make_white_image())
        gray = to_grayscale(arr)
        angle = detect_skew_angle(gray)
        assert angle == 0.0

    def test_deskew_trivial(self) -> None:
        arr = pil_to_cv2(_make_grey_image())
        result = deskew(arr, angle=0.0)
        assert result.shape == arr.shape


class TestResizeForVLM:
    def test_small_image_unchanged(self) -> None:
        img = _make_white_image(800, 600)
        result = resize_for_vlm(img, max_side=1800)
        assert result.size == (800, 600)

    def test_large_image_scaled_down(self) -> None:
        img = _make_white_image(3600, 2700)
        result = resize_for_vlm(img, max_side=1800)
        assert max(result.size) <= 1800

    def test_aspect_ratio_preserved(self) -> None:
        img = _make_white_image(3600, 1800)
        result = resize_for_vlm(img, max_side=1800)
        assert result.size[0] == 1800
        assert result.size[1] == 900


class TestCleanPipelines:
    def test_clean_for_vlm_returns_pil(self) -> None:
        img = _make_grey_image()
        result = clean_for_vlm(img, deskew_enabled=False)
        assert isinstance(result, PILModule.Image)

    def test_clean_for_tesseract_returns_pil(self) -> None:
        img = _make_grey_image()
        result = clean_for_tesseract(img, deskew_enabled=False)
        assert isinstance(result, PILModule.Image)

    def test_clean_for_vlm_with_deskew(self) -> None:
        img = _make_grey_image(800, 600)
        result = clean_for_vlm(img, deskew_enabled=True)
        assert isinstance(result, PILModule.Image)
