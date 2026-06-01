"""OpenTyphoon cloud API backend — one-stage, free for research.

Uses the OpenTyphoon API (OpenAI-compatible) at api.opentyphoon.ai.
Typhoon OCR outputs Markdown; a second Gemma struct pass converts to JSON.
This backend provides Typhoon accuracy without local GPU/CPU inference.

Sign up for a free API key at: https://opentyphoon.ai
"""
from __future__ import annotations

import base64
import io
import time

from openai import OpenAI
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


class TyphoonAPIExtractor(BaseExtractor):
    backend_name = "typhoon-ocr-api"

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.typhoon_api_key,
            base_url=settings.typhoon_api_base_url,
        )

    def _ocr_pages(self, images: list[PILImage]) -> str:
        pages_md: list[str] = []
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode()

            response = self._client.chat.completions.create(
                model=settings.typhoon_api_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _TYPHOON_OCR_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    }
                ],
                temperature=0.0,
                top_p=0.6,
            )
            pages_md.append(
                f"<!-- Page {i + 1} -->\n{response.choices[0].message.content}"
            )
        return "\n\n".join(pages_md)

    def extract(self, images: list[PILImage], doc_type: DocType) -> ExtractionResult:
        schema_cls = self._schema_cls(doc_type)

        t0 = time.perf_counter()
        markdown = self._ocr_pages(images)
        data = struct_extract_ollama(markdown, doc_type, schema_cls)
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
            self._client.models.list()
            return True
        except Exception:
            return False
