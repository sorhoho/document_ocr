"""Shared Ollama structured-output extraction helper.

Uses format="json" (Ollama's JSON mode) rather than format=schema dict,
because schema-dict enforcement causes Gemma 3 4B to return all-null defaults
for Thai text. format="json" with a template-based prompt produces correct output.

The JSON response is validated with Pydantic after extraction.
"""
from __future__ import annotations

import json

import ollama

from document_ocr.config import settings
from document_ocr.extractors.base import DocType
from document_ocr.schemas.contract import ThaiContract
from document_ocr.schemas.invoice import SupplierInvoice

_INVOICE_TEMPLATE = """{
  "document_type": "invoice",
  "vendor_name": "<company name of the SELLER>",
  "vendor_address": "<seller address or null>",
  "vendor_tax_id": "<seller tax ID number or null>",
  "invoice_number": "<invoice/document number>",
  "invoice_date": "<date as YYYY-MM-DD in CE, null if absent>",
  "payment_due_date": "<due date as YYYY-MM-DD in CE, null if absent>",
  "subtotal": "<pre-tax amount as number string, e.g. '13000', null if absent>",
  "vat_amount": "<VAT amount as number string, e.g. '910', null if absent>",
  "vat_rate": "<VAT percentage as number string, e.g. '7.0'>",
  "total_amount": "<grand total as number string, e.g. '13910', null if absent>",
  "buyer": {
    "name": "<buyer/customer company name or null>",
    "address": "<buyer address or null>",
    "tax_id": "<buyer tax ID or null>"
  },
  "line_items": [
    {
      "description": "<item description>",
      "quantity": "<quantity as number string>",
      "unit_price": "<unit price as number string>",
      "total_price": "<line total as number string>"
    }
  ],
  "confidence_notes": "<any notes about uncertain fields or null>"
}"""

_CONTRACT_TEMPLATE = """{
  "document_type": "contract",
  "contract_reference": "<contract/document number or null>",
  "parties": [
    {
      "name": "<party full legal name>",
      "address": "<party address or null>",
      "role": "<role in Thai, e.g. ผู้ว่าจ้าง or ผู้รับจ้าง, or null>",
      "tax_id": "<party tax ID or null>"
    }
  ],
  "contract_date": "<date as YYYY-MM-DD in CE, null if absent>",
  "effective_date": "<start date as YYYY-MM-DD in CE, null if absent>",
  "expiry_date": "<end date as YYYY-MM-DD in CE, null if absent>",
  "contract_value": "<contract amount as number string or null>",
  "currency": "THB",
  "payment_terms": "<payment terms description or null>",
  "key_obligations": ["<obligation 1>", "<obligation 2>"],
  "penalty_clauses": [
    {
      "description": "<penalty description>",
      "amount": "<penalty amount as number string or null>",
      "rate_per_day": "<daily rate as number string or null>"
    }
  ],
  "confidence_notes": "<any notes about uncertain fields or null>"
}"""

_INVOICE_STRUCT_PROMPT = """You are a Thai document data extraction assistant. Extract invoice fields from the OCR text below.

Thai → English field mappings:
- เลขที่ / Invoice No → invoice_number
- วันที่ → invoice_date (convert BE year: subtract 543, format YYYY-MM-DD)
- บริษัท...จำกัด (with tax ID เลขประจำตัวผู้เสียภาษี) → vendor_name + vendor_tax_id (this is the SELLER)
- ผู้ซื้อ / ลูกค้า → buyer
- ราคาก่อนภาษี → subtotal (number only, no commas or บาท)
- ภาษีมูลค่าเพิ่ม / VAT → vat_amount (number only)
- รวมทั้งสิ้น / ยอดรวม → total_amount (number only)
- รายการสินค้า → line_items

OCR TEXT:
{ocr_text}

Fill in the JSON template below with the extracted values. Replace placeholder strings with actual values from the text. Use null (not "null") for missing fields. Numbers must be strings without commas or currency symbols.

{template}"""

_CONTRACT_STRUCT_PROMPT = """You are a Thai document data extraction assistant. Extract contract fields from the OCR text below.

Thai → English field mappings:
- สัญญาเลขที่ / Contract No → contract_reference
- วันที่ทำสัญญา → contract_date (convert BE year: subtract 543, format YYYY-MM-DD)
- ผู้ว่าจ้าง = employer/client, ผู้รับจ้าง = contractor → both are parties
- มูลค่าสัญญา / ราคาจ้าง → contract_value (number only)
- เงื่อนไขการชำระเงิน → payment_terms
- ค่าปรับ / เบี้ยปรับ → penalty_clauses

OCR TEXT:
{ocr_text}

Fill in the JSON template below with the extracted values. Replace placeholder strings with actual values. Use null for missing fields. Numbers as strings without commas or currency symbols.

{template}"""


def struct_extract_ollama(
    ocr_text: str,
    doc_type: DocType,
    schema_cls: type[ThaiContract | SupplierInvoice],
    model: str | None = None,
    base_url: str | None = None,
) -> ThaiContract | SupplierInvoice:
    """Convert OCR text to a structured Pydantic model using Ollama JSON mode.

    Uses format="json" (not format=schema) because Gemma 3 4B returns all-null
    when given a schema dict with nullable fields. The template-based prompt
    guides field extraction reliably on Thai text.

    Args:
        ocr_text: Raw OCR output (Markdown or plain text).
        doc_type: CONTRACT or INVOICE.
        schema_cls: Target Pydantic model class.
        model: Ollama model name (defaults to settings.gemma_struct_model).
        base_url: Ollama server URL (defaults to settings.ollama_base_url).

    Returns:
        Validated Pydantic model instance.
    """
    client = ollama.Client(host=base_url or settings.ollama_base_url, timeout=600)

    if doc_type == DocType.INVOICE:
        prompt = _INVOICE_STRUCT_PROMPT.format(
            ocr_text=ocr_text, template=_INVOICE_TEMPLATE
        )
    else:
        prompt = _CONTRACT_STRUCT_PROMPT.format(
            ocr_text=ocr_text, template=_CONTRACT_TEMPLATE
        )

    response = client.chat(
        model=model or settings.gemma_struct_model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.0},
    )

    raw = response.message.content
    parsed = json.loads(raw)

    # Remove template placeholder strings (values still containing "<...>")
    _strip_placeholders(parsed)

    return schema_cls.model_validate(parsed)


def _strip_placeholders(obj: object) -> None:
    """Recursively replace unfilled template placeholder strings with None."""
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
                obj[key] = None  # type: ignore[index]
            else:
                _strip_placeholders(value)
    elif isinstance(obj, list):
        for item in obj:
            _strip_placeholders(item)
