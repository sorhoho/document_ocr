"""Typhoon OCR 1.5 self-hosted backend — 1.5-stage via Ollama.

Stage 1: scb10x/typhoon-ocr1.5-3b (Ollama VLM) → raw Markdown per page
Stage 2: gemma3:4b (Ollama structured output) → validated JSON

Why two stages:
  Typhoon OCR 1.5 is fine-tuned for Markdown transcription, not JSON generation.
  Forcing JSON output degrades OCR quality. Keeping OCR and structuring separate
  gives best accuracy per stage. Gemma enforces the schema via Ollama's format param.

Setup:
  ollama pull scb10x/typhoon-ocr1.5-3b
  ollama pull gemma3:4b

CPU performance: ~60–120s per page on a modern multi-core CPU (Q4_K_M model).
"""
from __future__ import annotations

import base64
import io
import time

import ollama
from PIL.Image import Image as PILImage

from document_ocr.config import settings
from document_ocr.extractors._struct_helper import struct_extract_ollama
from document_ocr.extractors.base import BaseExtractor, DocType, ExtractionResult

_TYPHOON_OCR_PROMPT = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Page Numbers: Wrap page numbers in <page_number>...</page_number>.
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."""


class TyphoonExtractor(BaseExtractor):
    backend_name = "typhoon-ocr1.5"

    def __init__(
        self,
        ocr_model: str | None = None,
        struct_model: str | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        self._base_url = ollama_base_url or settings.ollama_base_url
        self._ocr_model = ocr_model or settings.typhoon_ocr_model
        self._struct_model = struct_model or settings.gemma_struct_model
        self._client = ollama.Client(host=self._base_url)

    def _ocr_pages(self, images: list[PILImage]) -> str:
        """Run Typhoon OCR on each page image and return concatenated Markdown."""
        pages_md: list[str] = []
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode()

            response = self._client.chat(
                model=self._ocr_model,
                messages=[
                    {
                        "role": "user",
                        "content": _TYPHOON_OCR_PROMPT,
                        "images": [b64],
                    }
                ],
                options={"temperature": 0.0, "top_p": 0.6, "repeat_penalty": 1.1},
            )
            pages_md.append(f"<!-- Page {i + 1} -->\n{response.message.content}")
        return "\n\n".join(pages_md)

    def extract(self, images: list[PILImage], doc_type: DocType) -> ExtractionResult:
        schema_cls = self._schema_cls(doc_type)

        t0 = time.perf_counter()
        markdown = self._ocr_pages(images)
        data = struct_extract_ollama(
            markdown,
            doc_type,
            schema_cls,
            model=self._struct_model,
            base_url=self._base_url,
        )
        elapsed = time.perf_counter() - t0

        data.raw_markdown = markdown
        return ExtractionResult(
            data=data,
            backend=self.backend_name,
            pages_processed=len(images),
            elapsed_seconds=elapsed,
            raw_markdown=markdown,
        )

    def health_check(self) -> bool:
        try:
            model_names = [m.model for m in self._client.list().models]
            return self._ocr_model in model_names
        except Exception:
            return False
