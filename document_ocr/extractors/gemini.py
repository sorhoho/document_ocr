"""Gemini 2.5 Flash backend — one-stage cloud API extraction.

Sends page images directly to Gemini with response_schema enforcement for
native structured JSON output. This is the accuracy baseline.
"""
from __future__ import annotations

import io
import time

from google import genai
from google.genai import types
from PIL.Image import Image as PILImage

from document_ocr.config import settings
from document_ocr.extractors._prompts import get_extraction_prompt
from document_ocr.extractors.base import BaseExtractor, DocType, ExtractionResult

# Pricing USD per 1M tokens (Gemini 2.5 Flash, as of 2025)
_INPUT_PRICE_PER_1M = 0.075
_OUTPUT_PRICE_PER_1M = 0.30


class GeminiExtractor(BaseExtractor):
    backend_name = "gemini-2.5-flash"

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def extract(self, images: list[PILImage], doc_type: DocType) -> ExtractionResult:
        schema_cls = self._schema_cls(doc_type)
        prompt = get_extraction_prompt(doc_type)

        parts: list[types.Part] = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            parts.append(
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
            )
        parts.append(types.Part.from_text(text=prompt))

        t0 = time.perf_counter()
        response = self._client.models.generate_content(
            model=settings.gemini_model,
            contents=parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema_cls,
                temperature=0.0,
            ),
        )
        elapsed = time.perf_counter() - t0

        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count or 0
        output_tokens = usage.candidates_token_count or 0
        cost = (
            input_tokens * _INPUT_PRICE_PER_1M + output_tokens * _OUTPUT_PRICE_PER_1M
        ) / 1_000_000

        data = schema_cls.model_validate_json(response.text)
        return ExtractionResult(
            data=data,
            backend=self.backend_name,
            pages_processed=len(images),
            elapsed_seconds=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        )

    def health_check(self) -> bool:
        try:
            self._client.models.generate_content(
                model=settings.gemini_model,
                contents="ping",
            )
            return True
        except Exception:
            return False
