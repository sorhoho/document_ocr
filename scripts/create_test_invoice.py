"""Generate a synthetic Thai supplier invoice as an image and PDF for smoke testing.

Uses the fonts-thai-tlwg package (Garuda font) to render Thai text.
Saves:
  data/samples/test_invoice.png  — 300 DPI image
  data/samples/test_invoice.pdf  — single-page PDF wrapping the image
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

INVOICE_LINES = [
    ("ใบแจ้งหนี้ / Invoice", 36, True),
    ("", 12, False),
    ("เลขที่ / Invoice No: INV-2025-001", 20, False),
    ("วันที่ / Date: 15/06/2568", 20, False),
    ("กำหนดชำระ / Due: 30/06/2568", 20, False),
    ("", 12, False),
    ("ผู้ขาย / Vendor:", 22, True),
    ("  บริษัท ไทย ซัพพลาย จำกัด", 20, False),
    ("  เลขประจำตัวผู้เสียภาษี: 0105560001234", 20, False),
    ("  123 ถนนสุขุมวิท กรุงเทพฯ 10110", 20, False),
    ("", 12, False),
    ("ผู้ซื้อ / Buyer:", 22, True),
    ("  บริษัท ลูกค้า จำกัด", 20, False),
    ("  เลขประจำตัวผู้เสียภาษี: 0105565009999", 20, False),
    ("", 12, False),
    ("รายการสินค้า / Line Items:", 22, True),
    ("  1. สินค้า A  จำนวน 2 ชิ้น  ราคาหน่วย 5,000 บาท  รวม 10,000 บาท", 18, False),
    ("  2. บริการ B  จำนวน 1 รายการ  ราคาหน่วย 3,000 บาท  รวม 3,000 บาท", 18, False),
    ("", 12, False),
    ("ราคาก่อนภาษี / Subtotal:    13,000 บาท", 20, False),
    ("ภาษีมูลค่าเพิ่ม 7% / VAT 7%:     910 บาท", 20, False),
    ("─" * 48, 20, False),
    ("รวมทั้งสิ้น / Grand Total:  13,910 บาท", 22, True),
]

# Thai font paths (fonts-thai-tlwg package)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/thai-tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/tlwg/Norasi.ttf",
]


def _find_thai_font() -> str | None:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def create_invoice_image(output_dir: Path, dpi: int = 150) -> Path:
    """Render a synthetic Thai invoice image.

    Args:
        output_dir: Directory to write test_invoice.png and test_invoice.pdf.
        dpi: Image DPI (150 is fine for a smoke test; use 300 for production realism).

    Returns:
        Path to the generated PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    font_path = _find_thai_font()

    # Calculate canvas height
    line_heights = sum(size + 4 for _, size, _ in INVOICE_LINES) + 60
    w, h = 900, max(line_heights, 800)

    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = 30
    for text, size, bold in INVOICE_LINES:
        if font_path:
            try:
                font = ImageFont.truetype(font_path, size)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        color = (0, 0, 0) if not bold else (20, 20, 80)
        draw.text((40, y), text, fill=color, font=font)
        y += size + 6

    # Add a simple border
    draw.rectangle([10, 10, w - 10, h - 10], outline=(100, 100, 100), width=2)

    png_path = output_dir / "test_invoice.png"
    img.save(str(png_path), dpi=(dpi, dpi))
    print(f"Saved PNG: {png_path}  ({w}×{h} px @ {dpi} DPI)")

    # Save as single-page PDF via PIL
    pdf_path = output_dir / "test_invoice.pdf"
    img.save(str(pdf_path), "PDF", resolution=dpi)
    print(f"Saved PDF: {pdf_path}")

    return png_path


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "data" / "samples"
    create_invoice_image(out)
