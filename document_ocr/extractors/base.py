from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

from PIL.Image import Image as PILImage

from document_ocr.schemas.contract import ThaiContract
from document_ocr.schemas.invoice import SupplierInvoice

DocumentOutput = Union[ThaiContract, SupplierInvoice]


class DocType(str, Enum):
    CONTRACT = "contract"
    INVOICE = "invoice"


@dataclass
class ExtractionResult:
    data: DocumentOutput
    backend: str
    pages_processed: int
    elapsed_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    raw_markdown: str | None = None


class BaseExtractor(abc.ABC):
    """Common interface for all extraction backends."""

    backend_name: str

    @abc.abstractmethod
    def extract(self, images: list[PILImage], doc_type: DocType) -> ExtractionResult:
        """Extract structured data from preprocessed page images."""
        ...

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Return True if the backend is reachable and ready."""
        ...

    def _schema_cls(self, doc_type: DocType) -> type[ThaiContract | SupplierInvoice]:
        return ThaiContract if doc_type == DocType.CONTRACT else SupplierInvoice
