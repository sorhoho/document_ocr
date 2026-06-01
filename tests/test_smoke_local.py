"""End-to-end smoke test for Typhoon OCR 1.5 and Gemma backends.

Automatically skipped if Ollama is not running or models are not pulled.
Tests the full extraction pipeline on a synthetically generated Thai invoice.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Skip guard: skip the entire module if Ollama is unreachable
# ---------------------------------------------------------------------------
def _ollama_ready() -> bool:
    try:
        import ollama
        client = ollama.Client(timeout=5)
        models = [m.model for m in client.list().models]
        return "scb10x/typhoon-ocr1.5-3b:latest" in models
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_ready(),
    reason="Ollama not running or scb10x/typhoon-ocr1.5-3b model not pulled",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _thai_font_path() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
        "/usr/share/fonts/truetype/thai-tlwg/Garuda.ttf",
        "/usr/share/fonts/truetype/tlwg/Norasi.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


@pytest.fixture(scope="module")
def synthetic_invoice_image() -> Image.Image:
    """Generate a synthetic Thai invoice as a PIL Image."""
    lines = [
        ("ใบแจ้งหนี้ / Invoice", 32),
        ("เลขที่: INV-2025-001", 20),
        ("วันที่: 15/06/2568", 20),
        ("", 10),
        ("บริษัท ไทย ซัพพลาย จำกัด", 20),
        ("เลขประจำตัวผู้เสียภาษี: 0105560001234", 18),
        ("", 10),
        ("1. สินค้า A  2 ชิ้น  5,000 บาท  รวม 10,000 บาท", 18),
        ("2. บริการ B  1 รายการ  3,000 บาท  รวม 3,000 บาท", 18),
        ("", 10),
        ("ราคาก่อนภาษี: 13,000 บาท", 20),
        ("ภาษีมูลค่าเพิ่ม 7%: 910 บาท", 20),
        ("รวมทั้งสิ้น: 13,910 บาท", 22),
    ]

    font_path = _thai_font_path()
    total_h = sum(sz + 6 for _, sz in lines) + 80
    img = Image.new("RGB", (800, max(total_h, 600)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = 40
    for text, size in lines:
        try:
            font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        draw.text((40, y), text, fill=(0, 0, 0), font=font)
        y += size + 6

    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTyphoonExtractorSmoke:
    def test_ocr_returns_markdown(self, synthetic_invoice_image: Image.Image) -> None:
        """Typhoon OCR stage should return non-empty Markdown text."""
        import ollama
        from document_ocr.extractors.typhoon import TyphoonExtractor

        extractor = TyphoonExtractor()

        # Only test the OCR stage to keep this test fast
        markdown = extractor._ocr_pages([synthetic_invoice_image])

        assert isinstance(markdown, str)
        assert len(markdown) > 50, "Expected meaningful OCR output"
        print(f"\n--- Typhoon OCR Markdown ---\n{markdown[:500]}")

    def test_full_extraction_invoice(self, synthetic_invoice_image: Image.Image) -> None:
        """Full pipeline: image → Typhoon OCR → Gemma struct → SupplierInvoice."""
        from document_ocr.extractors.base import DocType
        from document_ocr.extractors.typhoon import TyphoonExtractor
        from document_ocr.schemas.invoice import SupplierInvoice

        extractor = TyphoonExtractor()
        result = extractor.extract([synthetic_invoice_image], DocType.INVOICE)

        assert isinstance(result.data, SupplierInvoice), "Expected SupplierInvoice schema"
        assert result.pages_processed == 1
        assert result.elapsed_seconds > 0
        assert result.raw_markdown is not None and len(result.raw_markdown) > 0

        inv: SupplierInvoice = result.data  # type: ignore[assignment]
        print(f"\n--- Extracted SupplierInvoice ---\n{inv.model_dump_json(indent=2)}")
        print(f"\nOCR time: {result.elapsed_seconds:.1f}s")

        # At minimum, some fields should be populated
        populated = [
            f for f in ["invoice_number", "total_amount", "vendor_name",
                        "subtotal", "vat_amount", "vat_rate"]
            if getattr(inv, f, None) is not None
        ]
        assert len(populated) >= 2, (
            f"Expected at least 2 extracted fields, got {len(populated)}: {populated}"
        )


class TestGemmaExtractorSmoke:
    @pytest.mark.skipif(
        not Path("/usr/bin/tesseract").exists() and not Path("/usr/local/bin/tesseract").exists(),
        reason="Tesseract not installed",
    )
    def test_full_extraction_invoice(self, synthetic_invoice_image: Image.Image) -> None:
        """Tesseract Thai OCR → Gemma struct → SupplierInvoice."""
        from document_ocr.extractors.base import DocType
        from document_ocr.extractors.gemma import GemmaExtractor
        from document_ocr.schemas.invoice import SupplierInvoice

        extractor = GemmaExtractor()
        result = extractor.extract([synthetic_invoice_image], DocType.INVOICE)

        assert isinstance(result.data, SupplierInvoice)
        inv: SupplierInvoice = result.data  # type: ignore[assignment]
        print(f"\n--- Gemma Extracted Invoice ---\n{inv.model_dump_json(indent=2)}")
        print(f"Gemma time: {result.elapsed_seconds:.1f}s")
