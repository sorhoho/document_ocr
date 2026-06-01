"""LLM-as-judge evaluation using Gemini 2.5 Flash as the reference model.

Strategy:
  - Use Gemini's own extraction as the reference (production baseline).
  - For each candidate backend, judge scores low-confidence fields.
  - Categorical verdicts for reproducibility: CORRECT / PARTIALLY_CORRECT /
    INCORRECT / CANNOT_DETERMINE.
  - Thai-aware rubric: accepts BE/CE equivalents, name variants, number formats.
"""
from __future__ import annotations

import base64
import io
import json
from enum import Enum

from google import genai
from google.genai import types
from PIL.Image import Image as PILImage

from document_ocr.config import settings

_JUDGE_SYSTEM_PROMPT = """You are a strict but fair Thai document extraction evaluator.

You will be given:
1. A reference extraction (from a high-accuracy model, used as the expected answer)
2. A candidate extraction to evaluate against the reference
3. The first page of the original document image (for grounding)

Evaluate each field listed and assign exactly one verdict:
- CORRECT: The candidate matches the reference (allowing for format variants).
- PARTIALLY_CORRECT: The candidate has the right information but with minor errors.
- INCORRECT: The candidate value is wrong or meaningfully different.
- CANNOT_DETERMINE: Cannot judge without more context.

Thai-specific rules:
- Thai names and their romanised equivalents count as CORRECT.
- Buddhist Era (BE) and CE dates referring to the same calendar day count as CORRECT.
- Numbers formatted differently (1,000.00 vs 1000) count as CORRECT if numerically equal.
- Thai numerals (๐–๙) and Arabic numerals count as CORRECT if equal.
- A null/None candidate when the reference is also null counts as CORRECT.
- A null/None candidate when the reference is non-null counts as INCORRECT.

Return ONLY a JSON object with field names as keys and verdicts as values.
Example: {"vendor_name": "CORRECT", "invoice_date": "PARTIALLY_CORRECT"}"""


class Verdict(str, Enum):
    CORRECT = "CORRECT"
    PARTIALLY_CORRECT = "PARTIALLY_CORRECT"
    INCORRECT = "INCORRECT"
    CANNOT_DETERMINE = "CANNOT_DETERMINE"

    @property
    def score(self) -> float:
        return {
            "CORRECT": 1.0,
            "PARTIALLY_CORRECT": 0.5,
            "INCORRECT": 0.0,
            "CANNOT_DETERMINE": 0.0,
        }[self.value]


class LLMJudge:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def evaluate(
        self,
        reference: dict,
        candidate: dict,
        page_images: list[PILImage],
        fields_to_evaluate: list[str],
    ) -> dict[str, Verdict]:
        """Score candidate extraction against reference for the given fields.

        Args:
            reference: Reference extraction dict (from Gemini baseline).
            candidate: Candidate extraction dict to evaluate.
            page_images: Document page images (first page used for grounding).
            fields_to_evaluate: Field names to judge.

        Returns:
            Mapping of field_name → Verdict.
        """
        if not fields_to_evaluate:
            return {}

        ref_subset = {k: reference.get(k) for k in fields_to_evaluate}
        cand_subset = {k: candidate.get(k) for k in fields_to_evaluate}

        prompt_text = (
            f"Reference extraction:\n{json.dumps(ref_subset, ensure_ascii=False, indent=2)}\n\n"
            f"Candidate extraction:\n{json.dumps(cand_subset, ensure_ascii=False, indent=2)}\n\n"
            "Evaluate all fields listed above. Return JSON only."
        )

        parts: list[types.Part] = []
        if page_images:
            buf = io.BytesIO()
            page_images[0].save(buf, format="JPEG", quality=85)
            parts.append(
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
            )
        parts.append(types.Part.from_text(text=prompt_text))

        response = self._client.models.generate_content(
            model=settings.gemini_model,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=_JUDGE_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        raw: dict = json.loads(response.text)
        return {
            k: Verdict(v)
            for k, v in raw.items()
            if k in fields_to_evaluate and v in Verdict._value2member_map_
        }
