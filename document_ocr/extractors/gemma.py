"""Gemma backend — two-stage via Tesseract + Ollama.

Stage 1: Tesseract (tha+eng tessdata) → raw Thai/English text
Stage 2: gemma3:4b (Ollama structured output) → validated JSON

Use as a lightweight fallback when Typhoon OCR is unavailable or too slow.
Tesseract quality on complex Thai scans is lower than Typhoon VLM.

System requirements:
  sudo apt-get install tesseract-ocr tesseract-ocr-tha
  ollama pull gemma3:4b
"""
from __future__ import annotations

import time

import pytesseract
from PIL.Image import Image as PILImage

from document_ocr.config import settings
from document_ocr.extractors._struct_helper import struct_extract_ollama
from document_ocr.extractors.base import BaseExtractor, DocType, ExtractionResult


class GemmaExtractor(BaseExtractor):
    backend_name = "gemma3-4b+tesseract"

    def __init__(
        self,
        struct_model: str | None = None,
        tesseract_lang: str = "tha+eng",
        ollama_base_url: str | None = None,
    ) -> None:
        self._struct_model = struct_model or settings.gemma_struct_model
        self._tesseract_lang = tesseract_lang
        self._base_url = ollama_base_url or settings.ollama_base_url

    def _run_tesseract(self, images: list[PILImage]) -> str:
        pages: list[str] = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(
                img,
                lang=self._tesseract_lang,
                config="--psm 6",
            )
            pages.append(f"--- Page {i + 1} ---\n{text}")
        return "\n\n".join(pages)

    def extract(self, images: list[PILImage], doc_type: DocType) -> ExtractionResult:
        schema_cls = self._schema_cls(doc_type)

        t0 = time.perf_counter()
        raw_text = self._run_tesseract(images)
        data = struct_extract_ollama(
            raw_text,
            doc_type,
            schema_cls,
            model=self._struct_model,
            base_url=self._base_url,
        )
        elapsed = time.perf_counter() - t0

        data.raw_markdown = raw_text
        return ExtractionResult(
            data=data,
            backend=self.backend_name,
            pages_processed=len(images),
            elapsed_seconds=elapsed,
            raw_markdown=raw_text,
        )

    def health_check(self) -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
