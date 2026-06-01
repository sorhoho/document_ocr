"""EvaluationHarness: orchestrates multi-backend evaluation of a single document.

Workflow:
  1. Run all specified backends on the same document.
  2. Compute cross-model agreement scores.
  3. Optionally run LLM-as-judge on low-confidence fields (uses Gemini as reference).
  4. Return a structured report with extractions, agreement, judge scores, and costs.
"""
from __future__ import annotations

from pathlib import Path

from PIL.Image import Image as PILImage

from document_ocr.evaluation.agreement import compute_agreement
from document_ocr.evaluation.cost_tracker import CostTracker
from document_ocr.evaluation.judge import LLMJudge
from document_ocr.extractors.base import DocType, ExtractionResult
from document_ocr.pipeline import ExtractionPipeline, _build_extractor

_VALID_BACKENDS = ("gemini", "typhoon-api", "typhoon", "gemma")


class EvaluationHarness:
    def __init__(
        self,
        backends: list[str] | None = None,
        use_llm_judge: bool = True,
        cpu_cost_per_hour_usd: float = 0.05,
    ) -> None:
        self._backends = backends or ["gemini", "typhoon", "gemma"]
        self._use_llm_judge = use_llm_judge
        self._tracker = CostTracker(cpu_cost_per_hour_usd)
        self._judge = LLMJudge() if use_llm_judge else None
        self._pipeline = ExtractionPipeline()

    def run(
        self,
        pdf_path: Path,
        doc_type: DocType,
        document_id: str | None = None,
        first_page: int | None = None,
        last_page: int | None = None,
    ) -> dict:
        """Run full evaluation on a single document.

        Returns:
            Report dict with keys:
              document_id, doc_type, extractions, agreement, judge, costs
        """
        doc_id = document_id or pdf_path.stem

        # Preprocess once per backend type (VLM vs Tesseract) to avoid redundant work
        vlm_images: list[PILImage] | None = None
        tess_images: list[PILImage] | None = None

        backend_results: dict[str, ExtractionResult] = {}
        for backend in self._backends:
            if backend == "gemma":
                if tess_images is None:
                    tess_images = self._pipeline.load_and_preprocess(
                        pdf_path, backend="gemma",
                        first_page=first_page, last_page=last_page,
                    )
                images = tess_images
            else:
                if vlm_images is None:
                    vlm_images = self._pipeline.load_and_preprocess(
                        pdf_path, backend=backend,
                        first_page=first_page, last_page=last_page,
                    )
                images = vlm_images

            extractor = _build_extractor(backend)
            result = extractor.extract(images, doc_type)
            backend_results[backend] = result

            self._tracker.record(
                backend=backend,
                document_id=doc_id,
                doc_type=doc_type.value,
                pages=result.pages_processed,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                api_cost_usd=result.estimated_cost_usd,
                elapsed_seconds=result.elapsed_seconds,
            )

        results_as_dicts = {
            name: r.data.model_dump(exclude={"raw_markdown"})
            for name, r in backend_results.items()
        }
        agreement = compute_agreement(results_as_dicts, doc_type.value)

        judge_report: dict = {}
        if self._use_llm_judge and self._judge and "gemini" in backend_results:
            reference = results_as_dicts["gemini"]
            low_conf_fields = agreement["low_confidence_fields"]

            page_images = vlm_images or []

            for backend, cand_dict in results_as_dicts.items():
                if backend == "gemini" or not low_conf_fields:
                    continue
                verdicts = self._judge.evaluate(
                    reference=reference,
                    candidate=cand_dict,
                    page_images=page_images,
                    fields_to_evaluate=low_conf_fields,
                )
                judge_report[backend] = {k: v.value for k, v in verdicts.items()}
                judge_report[f"{backend}_score"] = (
                    sum(v.score for v in verdicts.values()) / len(verdicts)
                    if verdicts
                    else 0.0
                )

        return {
            "document_id": doc_id,
            "doc_type": doc_type.value,
            "extractions": results_as_dicts,
            "agreement": agreement,
            "judge": judge_report,
            "costs": self._tracker.summary(),
        }
