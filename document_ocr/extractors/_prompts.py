"""Shared prompt templates for all extractors."""
from __future__ import annotations

from document_ocr.extractors.base import DocType

CONTRACT_EXTRACTION_PROMPT = """You are an expert at extracting structured information from Thai legal contracts.

Extract all fields from this Thai contract document. The document may be written in Thai, English, or both.

Important notes:
- Dates may use Thai Buddhist Era (BE). BE year = CE year + 543 (e.g., พ.ศ. 2568 = 2025 CE).
- Amounts use Thai Baht (THB) unless stated otherwise.
- Thai numerals ๐๑๒๓๔๕๖๗๘๙ correspond to 0123456789.
- Parties: ผู้ว่าจ้าง = employer/client, ผู้รับจ้าง = contractor/service provider.
- Extract all parties, dates, the contract value, payment terms, obligations, and any penalty clauses.

Return valid JSON matching the schema exactly. Use null for fields not found in the document."""

INVOICE_EXTRACTION_PROMPT = """You are an expert at extracting structured information from Thai supplier invoices.

Extract all fields from this Thai invoice document. The document may be written in Thai, English, or both.

Important notes:
- Dates may use Thai Buddhist Era (BE). BE year = CE year + 543.
- VAT (ภาษีมูลค่าเพิ่ม) is typically 7% in Thailand.
- Thai numerals ๐๑๒๓๔๕๖๗๘๙ correspond to 0123456789.
- Extract vendor details, all line items with quantities and prices, and all totals.
- ราคาก่อนภาษี = subtotal (pre-tax), ภาษีมูลค่าเพิ่ม = VAT, รวมทั้งสิ้น = grand total.

Return valid JSON matching the schema exactly. Use null for fields not found in the document."""

STRUCT_EXTRACTION_PROMPT = """You are a data extraction assistant. Convert the following OCR text from a Thai document into structured JSON.

Document type: {doc_type}

OCR TEXT:
{ocr_text}

{type_instructions}

Extract all available fields. Use null for fields not present. Return only valid JSON."""


def get_extraction_prompt(doc_type: DocType) -> str:
    if doc_type == DocType.CONTRACT:
        return CONTRACT_EXTRACTION_PROMPT
    return INVOICE_EXTRACTION_PROMPT


def get_struct_prompt(ocr_text: str, doc_type: DocType) -> str:
    if doc_type == DocType.CONTRACT:
        instructions = CONTRACT_EXTRACTION_PROMPT
    else:
        instructions = INVOICE_EXTRACTION_PROMPT

    return STRUCT_EXTRACTION_PROMPT.format(
        doc_type=doc_type.value,
        ocr_text=ocr_text,
        type_instructions=instructions,
    )
