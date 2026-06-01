"""ExtractionPipeline: top-level orchestrator.

Connects: PDF path → preprocessing → extractor → validated ExtractionResult.
"""
from __future__ import annotations

from pathlib import Path

from PIL.Image import Image as PILImage

from document_ocr.config import settings
from document_ocr.extractors.base import BaseExtractor, DocType, ExtractionResult
from document_ocr.preprocessing.image_cleaner import clean_for_tesseract, clean_for_vlm
from document_ocr.preprocessing.pdf_converter import pdf_to_images

_TESSERACT_BACKENDS = {"gemma"}


def _build_extractor(backend: str) -> BaseExtractor:
    if backend == "gemini":
        from document_ocr.extractors.gemini import GeminiExtractor
        return GeminiExtractor()
    if backend == "typhoon-api":
        from document_ocr.extractors.typhoon_api import TyphoonAPIExtractor
        return TyphoonAPIExtractor()
    if backend == "typhoon":
        from document_ocr.extractors.typhoon import TyphoonExtractor
        return TyphoonExtractor()
    if backend == "gemma":
        from document_ocr.extractors.gemma import GemmaExtractor
        return GemmaExtractor()
    raise ValueError(f"Unknown backend: {backend!r}. Choose from: gemini, typhoon-api, typhoon, gemma")


class ExtractionPipeline:
    def load_and_preprocess(
        self,
        pdf_path: Path,
        backend: str = "vlm",
        first_page: int | None = None,
        last_page: int | None = None,
    ) -> list[PILImage]:
        """Load PDF pages and apply backend-appropriate preprocessing."""
        images = pdf_to_images(
            pdf_path,
            dpi=settings.pdf_dpi,
            first_page=first_page,
            last_page=last_page,
        )
        use_tesseract = backend in _TESSERACT_BACKENDS
        clean_fn = clean_for_tesseract if use_tesseract else clean_for_vlm
        return [clean_fn(img, deskew_enabled=settings.deskew_enabled) for img in images]

    def run(
        self,
        pdf_path: Path,
        doc_type: DocType,
        backend: str = "typhoon",
        first_page: int | None = None,
        last_page: int | None = None,
    ) -> ExtractionResult:
        """Full pipeline: PDF → structured data."""
        images = self.load_and_preprocess(
            pdf_path, backend=backend, first_page=first_page, last_page=last_page
        )
        extractor = _build_extractor(backend)
        return extractor.extract(images, doc_type)
