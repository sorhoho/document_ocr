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

Extract the following fields from the Thai invoice text. The document may be in Thai, English, or both.

Field extraction rules:
- vendor_name: The SELLER/SUPPLIER company name (ผู้ขาย, ผู้ออกใบกำกับภาษี). NOT the buyer.
- vendor_tax_id: The SELLER'S tax ID (เลขประจำตัวผู้เสียภาษีของผู้ขาย).
- invoice_number: The invoice/document number (เลขที่, Invoice No).
- invoice_date: The document date (วันที่). Convert Thai Buddhist Era to CE: subtract 543 from year. Format as YYYY-MM-DD.
- subtotal: Pre-tax amount (ราคาก่อนภาษี, ราคาสินค้า). Write as a plain number, e.g. 13000.
- vat_amount: VAT amount (ภาษีมูลค่าเพิ่ม). Write as a plain number, e.g. 910.
- total_amount: Grand total including VAT (รวมทั้งสิ้น, ยอดรวมสุทธิ). Write as a plain number, e.g. 13910.
- vat_rate: VAT percentage, typically 7.
- line_items: List of line items. Each item has: description (product/service name), quantity (จำนวน), unit_price (ราคาหน่วย), total_price (รวม). Write all prices as plain numbers.
- buyer: The BUYER/CUSTOMER details (ผู้ซื้อ, ลูกค้า).

Important:
- Thai Buddhist Era (BE) year: subtract 543 to get CE year. Example: 2568 → 2025, 15/06/2568 → 2025-06-15.
- Write all amounts as numeric strings without currency symbols or commas.
- If a field is not present in the document, use null.
- Do NOT confuse the vendor (seller) with the buyer (customer).

Return valid JSON matching the schema exactly."""

STRUCT_EXTRACTION_PROMPT = """You are a data extraction assistant. Read the OCR text below from a Thai document and extract the requested fields into JSON.

Document type: {doc_type}

{type_instructions}

---
OCR TEXT:
{ocr_text}
---

Now extract all fields listed above from the OCR text. Return ONLY valid JSON, no explanation."""


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

