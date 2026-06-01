"""Field-level accuracy metrics for Thai document extraction.

Designed for no-ground-truth evaluation:
  - exact_match: dates, IDs, invoice/contract numbers
  - numeric_match: monetary amounts, quantities (1% tolerance)
  - fuzzy_text_match: names, addresses (rapidfuzz token_set_ratio)

Thai numeral normalisation (๐–๙ → 0–9) is applied before all comparisons.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation

from rapidfuzz import fuzz

THAI_DIGIT_MAP = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# Field registries — used by agreement.py and harness.py
FIELD_TYPES_CONTRACT: dict[str, str] = {
    "contract_reference": "exact",
    "contract_date": "exact",
    "effective_date": "exact",
    "expiry_date": "exact",
    "contract_value": "numeric",
    "currency": "exact",
    "payment_terms": "fuzzy",
}

FIELD_TYPES_INVOICE: dict[str, str] = {
    "vendor_name": "fuzzy",
    "vendor_tax_id": "exact",
    "invoice_number": "exact",
    "invoice_date": "exact",
    "payment_due_date": "exact",
    "subtotal": "numeric",
    "vat_amount": "numeric",
    "vat_rate": "numeric",
    "total_amount": "numeric",
}


def _normalise_text(text: str | None) -> str:
    if text is None:
        return ""
    return text.strip().translate(THAI_DIGIT_MAP).lower()


def _normalise_number(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        return None


def exact_match(a: object, b: object) -> float:
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    return 1.0 if _normalise_text(str(a)) == _normalise_text(str(b)) else 0.0


def numeric_match(a: object, b: object, tolerance: float = 0.01) -> float:
    da, db = _normalise_number(a), _normalise_number(b)
    if da is None and db is None:
        return 1.0
    if da is None or db is None:
        return 0.0
    if da == 0 and db == 0:
        return 1.0
    rel_diff = abs(da - db) / max(abs(da), abs(db))
    return 1.0 if rel_diff <= tolerance else 0.0


def fuzzy_text_match(a: str | None, b: str | None) -> float:
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    score = fuzz.token_set_ratio(_normalise_text(a), _normalise_text(b))
    return score / 100.0


def compare_fields(
    result_a: dict,
    result_b: dict,
    field_types: dict[str, str],
) -> dict[str, float]:
    """Compare two extraction dicts field-by-field. Returns per-field scores [0, 1]."""
    scores: dict[str, float] = {}
    for field, ftype in field_types.items():
        va = result_a.get(field)
        vb = result_b.get(field)
        if ftype == "numeric":
            scores[field] = numeric_match(va, vb)
        elif ftype == "exact":
            scores[field] = exact_match(va, vb)
        else:
            scores[field] = fuzzy_text_match(
                str(va) if va is not None else None,
                str(vb) if vb is not None else None,
            )
    return scores
