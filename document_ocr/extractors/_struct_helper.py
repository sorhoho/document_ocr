"""Shared Ollama structured-output extraction helper.

Used by TyphoonExtractor and TyphoonAPIExtractor to convert OCR Markdown
into validated Pydantic models via Gemma 3 4B with schema enforcement.
"""
from __future__ import annotations

import ollama

from document_ocr.config import settings
from document_ocr.extractors._prompts import get_struct_prompt
from document_ocr.extractors.base import DocType
from document_ocr.schemas.contract import ThaiContract
from document_ocr.schemas.invoice import SupplierInvoice


def struct_extract_ollama(
    ocr_text: str,
    doc_type: DocType,
    schema_cls: type[ThaiContract | SupplierInvoice],
    model: str | None = None,
    base_url: str | None = None,
) -> ThaiContract | SupplierInvoice:
    """Convert OCR text to a structured Pydantic model via Ollama schema enforcement.

    Args:
        ocr_text: Raw OCR output (Markdown or plain text).
        doc_type: CONTRACT or INVOICE.
        schema_cls: Target Pydantic model class.
        model: Ollama model name (defaults to settings.gemma_struct_model).
        base_url: Ollama server URL (defaults to settings.ollama_base_url).

    Returns:
        Validated Pydantic model instance.
    """
    client = ollama.Client(host=base_url or settings.ollama_base_url)
    prompt = get_struct_prompt(ocr_text, doc_type)

    response = client.chat(
        model=model or settings.gemma_struct_model,
        messages=[{"role": "user", "content": prompt}],
        format=schema_cls.model_json_schema(),
        options={"temperature": 0.0},
    )
    return schema_cls.model_validate_json(response.message.content)
