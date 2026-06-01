"""Tests for evaluation metrics and agreement scoring."""
from __future__ import annotations

import pytest

from document_ocr.evaluation.agreement import compute_agreement
from document_ocr.evaluation.cost_tracker import CostTracker
from document_ocr.evaluation.metrics import (
    exact_match,
    fuzzy_text_match,
    numeric_match,
    compare_fields,
    FIELD_TYPES_INVOICE,
    FIELD_TYPES_CONTRACT,
)


class TestExactMatch:
    def test_equal_strings(self) -> None:
        assert exact_match("INV-001", "INV-001") == 1.0

    def test_different_strings(self) -> None:
        assert exact_match("INV-001", "INV-002") == 0.0

    def test_both_none(self) -> None:
        assert exact_match(None, None) == 1.0

    def test_one_none(self) -> None:
        assert exact_match("value", None) == 0.0

    def test_thai_numerals_normalised(self) -> None:
        assert exact_match("๒๐๒๕-๐๑-๑๕", "2025-01-15") == 1.0

    def test_case_insensitive(self) -> None:
        assert exact_match("ABC", "abc") == 1.0


class TestNumericMatch:
    def test_equal_values(self) -> None:
        assert numeric_match(1000.0, 1000.0) == 1.0

    def test_within_tolerance(self) -> None:
        assert numeric_match(1000.0, 1005.0) == 1.0  # 0.5% diff

    def test_outside_tolerance(self) -> None:
        assert numeric_match(1000.0, 1200.0) == 0.0  # 20% diff

    def test_both_none(self) -> None:
        assert numeric_match(None, None) == 1.0

    def test_one_none(self) -> None:
        assert numeric_match(1000.0, None) == 0.0

    def test_both_zero(self) -> None:
        assert numeric_match(0, 0) == 1.0

    def test_comma_formatted(self) -> None:
        assert numeric_match("1,000.00", "1000") == 1.0

    def test_decimal_types(self) -> None:
        from decimal import Decimal
        assert numeric_match(Decimal("107000.00"), Decimal("107000")) == 1.0


class TestFuzzyTextMatch:
    def test_identical(self) -> None:
        assert fuzzy_text_match("บริษัท ตัวอย่าง จำกัด", "บริษัท ตัวอย่าง จำกัด") == 1.0

    def test_partial_match(self) -> None:
        # token_set_ratio returns 1.0 when one string's tokens are a complete
        # subset of the other — correct for Thai name matching (e.g. short name vs full name).
        score = fuzzy_text_match("บริษัท ตัวอย่าง จำกัด", "ตัวอย่าง")
        assert score > 0.3

    def test_unrelated_strings(self) -> None:
        score = fuzzy_text_match("บริษัท ตัวอย่าง จำกัด", "ห้างหุ้นส่วนจำกัด ไทยซัพพลาย")
        assert 0.0 <= score < 1.0

    def test_both_none(self) -> None:
        assert fuzzy_text_match(None, None) == 1.0

    def test_one_none(self) -> None:
        assert fuzzy_text_match("text", None) == 0.0

    def test_different_word_order(self) -> None:
        score = fuzzy_text_match("Apple Mango Banana", "Banana Apple Mango")
        assert score >= 0.9


class TestCompareFields:
    def test_invoice_fields(self) -> None:
        a = {"invoice_number": "INV-001", "total_amount": "10700.00", "vendor_name": "Thai Co"}
        b = {"invoice_number": "INV-001", "total_amount": "10700", "vendor_name": "Thai Co Ltd"}
        scores = compare_fields(a, b, FIELD_TYPES_INVOICE)
        assert scores["invoice_number"] == 1.0
        assert scores["total_amount"] == 1.0
        assert scores["vendor_name"] > 0.5


class TestAgreement:
    def test_perfect_agreement(self) -> None:
        results = {
            "gemini": {"invoice_number": "INV-001", "total_amount": "10700"},
            "typhoon": {"invoice_number": "INV-001", "total_amount": "10700"},
        }
        report = compute_agreement(results, doc_type="invoice")
        assert report["overall"] == 1.0
        assert "invoice_number" in report["high_confidence_fields"]

    def test_disagreement(self) -> None:
        results = {
            "gemini": {"invoice_number": "INV-001", "total_amount": "10700"},
            "typhoon": {"invoice_number": "INV-002", "total_amount": "9999"},
        }
        report = compute_agreement(results, doc_type="invoice")
        assert report["overall"] < 1.0
        assert "invoice_number" in report["low_confidence_fields"]

    def test_consensus_majority(self) -> None:
        results = {
            "gemini": {"invoice_number": "INV-001"},
            "typhoon": {"invoice_number": "INV-001"},
            "gemma": {"invoice_number": "INV-999"},
        }
        report = compute_agreement(results, doc_type="invoice")
        assert report["consensus"]["invoice_number"] == "INV-001"

    def test_three_backends_high_confidence(self) -> None:
        results = {
            "gemini": {"total_amount": "50000"},
            "typhoon": {"total_amount": "50000"},
            "gemma": {"total_amount": "50000"},
        }
        report = compute_agreement(results, doc_type="invoice")
        assert report["per_field"]["total_amount"] == 1.0


class TestCostTracker:
    def test_api_cost_recording(self) -> None:
        tracker = CostTracker(cpu_cost_per_hour_usd=0.0)
        tracker.record(
            backend="gemini",
            document_id="doc1",
            doc_type="invoice",
            pages=2,
            input_tokens=1000,
            output_tokens=200,
            api_cost_usd=0.005,
            elapsed_seconds=5.0,
        )
        summary = tracker.summary()
        assert "gemini" in summary
        assert summary["gemini"]["count"] == 1
        assert summary["gemini"]["total_cost_usd"] == pytest.approx(0.005)

    def test_cpu_cost_calculation(self) -> None:
        tracker = CostTracker(cpu_cost_per_hour_usd=1.0)
        tracker.record(
            backend="typhoon",
            document_id="doc1",
            doc_type="contract",
            pages=1,
            input_tokens=0,
            output_tokens=0,
            api_cost_usd=0.0,
            elapsed_seconds=3600.0,  # 1 hour
        )
        summary = tracker.summary()
        assert summary["typhoon"]["total_cost_usd"] == pytest.approx(1.0)

    def test_multiple_records_avg(self) -> None:
        tracker = CostTracker(cpu_cost_per_hour_usd=0.0)
        for i in range(3):
            tracker.record(
                backend="gemini",
                document_id=f"doc{i}",
                doc_type="invoice",
                pages=1,
                input_tokens=500,
                output_tokens=100,
                api_cost_usd=float(i + 1) * 0.001,
                elapsed_seconds=3.0,
            )
        summary = tracker.summary()
        assert summary["gemini"]["count"] == 3
        assert summary["gemini"]["avg_cost_usd"] == pytest.approx(0.002)
