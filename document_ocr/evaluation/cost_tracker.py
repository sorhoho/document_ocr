"""Per-extraction cost tracker.

Tracks both API costs (token-based) and CPU infrastructure costs (time-based)
so self-hosted and cloud backends can be compared on a common dollar basis.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class ExtractionCostRecord:
    backend: str
    document_id: str
    doc_type: str
    timestamp: datetime
    pages: int
    input_tokens: int
    output_tokens: int
    api_cost_usd: float
    elapsed_seconds: float
    cpu_cost_usd: float
    total_cost_usd: float


class CostTracker:
    def __init__(self, cpu_cost_per_hour_usd: float = 0.05) -> None:
        self._cpu_cost_per_hour = cpu_cost_per_hour_usd
        self._records: list[ExtractionCostRecord] = []

    def record(
        self,
        backend: str,
        document_id: str,
        doc_type: str,
        pages: int,
        input_tokens: int,
        output_tokens: int,
        api_cost_usd: float,
        elapsed_seconds: float,
    ) -> ExtractionCostRecord:
        cpu_cost = (elapsed_seconds / 3600.0) * self._cpu_cost_per_hour
        rec = ExtractionCostRecord(
            backend=backend,
            document_id=document_id,
            doc_type=doc_type,
            timestamp=datetime.now(tz=timezone.utc),
            pages=pages,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            api_cost_usd=api_cost_usd,
            elapsed_seconds=elapsed_seconds,
            cpu_cost_usd=cpu_cost,
            total_cost_usd=api_cost_usd + cpu_cost,
        )
        self._records.append(rec)
        return rec

    def summary(self) -> dict[str, dict]:
        by_backend: dict[str, list[ExtractionCostRecord]] = {}
        for r in self._records:
            by_backend.setdefault(r.backend, []).append(r)

        return {
            backend: {
                "count": len(recs),
                "total_cost_usd": round(sum(r.total_cost_usd for r in recs), 6),
                "avg_cost_usd": round(
                    sum(r.total_cost_usd for r in recs) / len(recs), 6
                ),
                "avg_elapsed_seconds": round(
                    sum(r.elapsed_seconds for r in recs) / len(recs), 2
                ),
                "total_pages": sum(r.pages for r in recs),
            }
            for backend, recs in by_backend.items()
        }
