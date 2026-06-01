"""Tests for Pydantic schema validation and Thai date normalisation."""
from __future__ import annotations

from decimal import Decimal

import pytest

from document_ocr.schemas.contract import Party, PenaltyClause, ThaiContract
from document_ocr.schemas.invoice import BuyerDetails, LineItem, SupplierInvoice


class TestThaiContractSchema:
    def test_basic_contract(self) -> None:
        c = ThaiContract(
            contract_reference="CTR-2025-001",
            contract_value=Decimal("1500000.00"),
            currency="THB",
        )
        assert c.contract_reference == "CTR-2025-001"
        assert c.contract_value == Decimal("1500000.00")
        assert c.currency == "THB"
        assert c.parties == []

    def test_be_date_conversion_iso(self) -> None:
        """Buddhist Era year 2568 → CE year 2025."""
        c = ThaiContract(contract_date="2568-03-15")
        assert c.contract_date is not None
        assert c.contract_date.year == 2025
        assert c.contract_date.month == 3
        assert c.contract_date.day == 15

    def test_be_date_conversion_slash(self) -> None:
        c = ThaiContract(effective_date="15/03/2568")
        assert c.effective_date is not None
        assert c.effective_date.year == 2025

    def test_ce_date_unchanged(self) -> None:
        c = ThaiContract(contract_date="2025-06-01")
        assert c.contract_date is not None
        assert c.contract_date.year == 2025

    def test_parties(self) -> None:
        c = ThaiContract(
            parties=[
                Party(name="บริษัท ตัวอย่าง จำกัด", role="ผู้ว่าจ้าง"),
                Party(name="นาย ตัวอย่าง", role="ผู้รับจ้าง"),
            ]
        )
        assert len(c.parties) == 2
        assert c.parties[0].role == "ผู้ว่าจ้าง"

    def test_raw_markdown_excluded_from_dump(self) -> None:
        c = ThaiContract(raw_markdown="# Contract\nsome text")
        dumped = c.model_dump()
        assert "raw_markdown" not in dumped

    def test_json_roundtrip(self) -> None:
        c = ThaiContract(
            contract_reference="REF-001",
            contract_value=Decimal("50000"),
            parties=[Party(name="Test Co Ltd", role="ผู้ว่าจ้าง")],
        )
        json_str = c.model_dump_json()
        restored = ThaiContract.model_validate_json(json_str)
        assert restored.contract_reference == c.contract_reference
        assert restored.contract_value == c.contract_value


class TestSupplierInvoiceSchema:
    def test_basic_invoice(self) -> None:
        inv = SupplierInvoice(
            vendor_name="Supplier Co Ltd",
            invoice_number="INV-2025-0042",
            total_amount=Decimal("10700.00"),
            vat_rate=Decimal("7.0"),
        )
        assert inv.invoice_number == "INV-2025-0042"
        assert inv.total_amount == Decimal("10700.00")

    def test_be_date_on_invoice(self) -> None:
        inv = SupplierInvoice(invoice_date="2568-01-20")
        assert inv.invoice_date is not None
        assert inv.invoice_date.year == 2025
        assert inv.invoice_date.month == 1

    def test_line_items(self) -> None:
        inv = SupplierInvoice(
            line_items=[
                LineItem(description="สินค้า A", quantity=Decimal("2"), unit_price=Decimal("500")),
                LineItem(description="บริการ B", quantity=Decimal("1"), unit_price=Decimal("9000")),
            ]
        )
        assert len(inv.line_items) == 2
        assert inv.line_items[0].quantity == Decimal("2")

    def test_default_vat_rate(self) -> None:
        inv = SupplierInvoice()
        assert inv.vat_rate == Decimal("7.0")

    def test_raw_markdown_excluded(self) -> None:
        inv = SupplierInvoice(raw_markdown="some ocr text")
        assert "raw_markdown" not in inv.model_dump()

    def test_json_roundtrip(self) -> None:
        inv = SupplierInvoice(
            vendor_name="Thai Supplier",
            total_amount=Decimal("10700"),
            buyer=BuyerDetails(name="Buyer Corp", tax_id="0105560001234"),
        )
        restored = SupplierInvoice.model_validate_json(inv.model_dump_json())
        assert restored.vendor_name == inv.vendor_name
        assert restored.buyer is not None
        assert restored.buyer.tax_id == "0105560001234"
