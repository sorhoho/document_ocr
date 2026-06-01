"""Cross-model agreement scoring.

When no ground truth exists, pairwise agreement between independent backends
is used as a proxy for correctness: high agreement → high confidence field value.

Returns:
  - per_field: mean pairwise agreement score per field
  - overall: mean across all fields
  - consensus: majority-vote value per field
  - high_confidence_fields: fields with score >= 0.8
  - low_confidence_fields: fields with score < 0.8
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

from document_ocr.evaluation.metrics import (
    FIELD_TYPES_CONTRACT,
    FIELD_TYPES_INVOICE,
    compare_fields,
)

HIGH_CONFIDENCE_THRESHOLD = 0.8


def compute_agreement(
    results: dict[str, dict],
    doc_type: str = "contract",
) -> dict:
    """Compute cross-model agreement across N backend result dicts.

    Args:
        results: Mapping of backend_name → extracted-fields dict.
        doc_type: "contract" or "invoice".

    Returns:
        Agreement report dict.
    """
    field_types = FIELD_TYPES_CONTRACT if doc_type == "contract" else FIELD_TYPES_INVOICE
    backend_names = list(results.keys())
    pairs = list(combinations(backend_names, 2))

    per_field_scores: dict[str, list[float]] = {f: [] for f in field_types}

    for b1, b2 in pairs:
        pair_scores = compare_fields(results[b1], results[b2], field_types)
        for field, score in pair_scores.items():
            per_field_scores[field].append(score)

    per_field_mean: dict[str, float] = {
        f: (sum(scores) / len(scores) if scores else 0.0)
        for f, scores in per_field_scores.items()
    }

    consensus: dict[str, object] = {}
    for field in field_types:
        values = [
            results[b].get(field)
            for b in backend_names
            if results[b].get(field) is not None
        ]
        if values:
            most_common, _ = Counter(map(str, values)).most_common(1)[0]
            consensus[field] = most_common
        else:
            consensus[field] = None

    overall = (
        sum(per_field_mean.values()) / len(per_field_mean) if per_field_mean else 0.0
    )

    return {
        "per_field": per_field_mean,
        "overall": overall,
        "consensus": consensus,
        "high_confidence_fields": [
            f for f, s in per_field_mean.items() if s >= HIGH_CONFIDENCE_THRESHOLD
        ],
        "low_confidence_fields": [
            f for f, s in per_field_mean.items() if s < HIGH_CONFIDENCE_THRESHOLD
        ],
    }
