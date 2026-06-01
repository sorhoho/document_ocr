from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LineItem(BaseModel):
    description: str
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    total_price: Optional[Decimal] = None


class BuyerDetails(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    tax_id: Optional[str] = None


class SupplierInvoice(BaseModel):
    """Structured extraction output for a Thai supplier invoice."""

    document_type: str = Field(default="invoice")

    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    vendor_tax_id: Optional[str] = None

    invoice_number: Optional[str] = None
    invoice_date: Optional[date] = None
    payment_due_date: Optional[date] = None

    line_items: list[LineItem] = Field(default_factory=list)

    subtotal: Optional[Decimal] = None
    vat_amount: Optional[Decimal] = None
    vat_rate: Optional[Decimal] = Field(default=Decimal("7.0"))
    total_amount: Optional[Decimal] = None

    buyer: Optional[BuyerDetails] = None

    confidence_notes: Optional[str] = None

    # Intermediate OCR text; excluded from serialisation
    raw_markdown: Optional[str] = Field(default=None, exclude=True)

    @field_validator("invoice_date", "payment_due_date", mode="before")
    @classmethod
    def normalise_be_date(cls, v: object) -> object:
        """Convert Thai Buddhist Era year (BE) to CE. BE = CE + 543."""
        if not isinstance(v, str):
            return v

        import re

        m = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", v)
        if m:
            year = int(m.group(1))
            if year > 2500:
                year -= 543
            return f"{year:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

        m = re.match(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", v)
        if m:
            year = int(m.group(3))
            if year > 2500:
                year -= 543
            return f"{year:04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"

        return v
