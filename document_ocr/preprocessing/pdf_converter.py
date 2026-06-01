from __future__ import annotations

from pathlib import Path

from pdf2image import convert_from_path
from PIL.Image import Image as PILImage


def pdf_to_images(
    pdf_path: Path | str,
    dpi: int = 300,
    first_page: int | None = None,
    last_page: int | None = None,
    fmt: str = "JPEG",
    thread_count: int = 2,
) -> list[PILImage]:
    """Convert a scanned PDF to a list of PIL Images.

    Args:
        pdf_path: Path to the scanned PDF.
        dpi: Render resolution. 300 DPI is the minimum for reliable Thai OCR.
        first_page: 1-indexed start page; None means first page.
        last_page: 1-indexed end page; None means last page.
        fmt: Intermediate format (JPEG is faster than PNG for large scans).
        thread_count: Poppler render threads (keep ≤4 to avoid I/O contention).

    Returns:
        List of PIL Images, one per page.
    """
    return convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=first_page,
        last_page=last_page,
        fmt=fmt,
        thread_count=thread_count,
    )
