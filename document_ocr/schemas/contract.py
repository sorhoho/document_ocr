from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Party(BaseModel):
    name: str = Field(description="Full legal name of the party (Thai or English)")
    address: Optional[str] = None
    role: Optional[str] = Field(
        default=None,
        description="Role in Thai: e.g. ผู้ว่าจ้าง (employer), ผู้รับจ้าง (contractor)",
    )
    tax_id: Optional[str] = None


class PenaltyClause(BaseModel):
    description: str
    amount: Optional[Decimal] = None
    rate_per_day: Optional[Decimal] = None


class ThaiContract(BaseModel):
    """Structured extraction output for a Thai contract document."""

    document_type: str = Field(default="contract")

    contract_reference: Optional[str] = Field(
        default=None, description="Contract reference or document number"
    )

    parties: list[Party] = Field(default_factory=list)

    contract_date: Optional[date] = None
    effective_date: Optional[date] = None
    expiry_date: Optional[date] = None

    contract_value: Optional[Decimal] = None
    currency: str = Field(default="THB")
    payment_terms: Optional[str] = None

    key_obligations: list[str] = Field(default_factory=list)
    penalty_clauses: list[PenaltyClause] = Field(default_factory=list)

    confidence_notes: Optional[str] = None

    # Intermediate OCR text; excluded from serialisation
    raw_markdown: Optional[str] = Field(default=None, exclude=True)

    @field_validator("contract_date", "effective_date", "expiry_date", mode="before")
    @classmethod
    def normalise_be_date(cls, v: object) -> object:
        """Convert Thai Buddhist Era year (BE) to CE before Pydantic parses the date.

        BE year = CE year + 543. Dates with year > 2500 are treated as BE.
        Accepts ISO strings like '2568-01-15' or '15/01/2568'.
        """
        if not isinstance(v, str):
            return v

        import re

        # Try YYYY-MM-DD or YYYY/MM/DD
        m = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", v)
        if m:
            year = int(m.group(1))
            if year > 2500:
                year -= 543
            return f"{year:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

        # Try DD/MM/YYYY or DD-MM-YYYY
        m = re.match(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", v)
        if m:
            year = int(m.group(3))
            if year > 2500:
                year -= 543
            return f"{year:04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"

        return v
