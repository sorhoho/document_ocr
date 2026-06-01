"""FastAPI service for Thai document OCR extraction.

Endpoints:
  POST /extract   — Extract one document (single backend).
  POST /evaluate  — Multi-backend evaluation with agreement + judge.
  GET  /health    — Backend health check.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from document_ocr.extractors.base import DocType
from document_ocr.pipeline import ExtractionPipeline

app = FastAPI(
    title="Thai Document OCR API",
    description="Extracts structured data from Thai scanned contracts and invoices",
    version="0.1.0",
)

_pipeline = ExtractionPipeline()

_VALID_BACKENDS = {"gemini", "typhoon-api", "typhoon", "gemma"}
_VALID_DOC_TYPES = {"contract", "invoice"}


@app.post("/extract")
async def extract_document(
    file: UploadFile = File(..., description="Scanned PDF"),
    doc_type: str = Form(default="contract", description="contract or invoice"),
    backend: str = Form(default="typhoon", description="gemini | typhoon-api | typhoon | gemma"),
    first_page: int | None = Form(default=None),
    last_page: int | None = Form(default=None),
) -> JSONResponse:
    """Extract structured fields from a Thai scanned PDF."""
    if doc_type not in _VALID_DOC_TYPES:
        raise HTTPException(400, f"doc_type must be one of: {sorted(_VALID_DOC_TYPES)}")
    if backend not in _VALID_BACKENDS:
        raise HTTPException(400, f"backend must be one of: {sorted(_VALID_BACKENDS)}")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        result = _pipeline.run(
            pdf_path=tmp_path,
            doc_type=DocType(doc_type),
            backend=backend,
            first_page=first_page,
            last_page=last_page,
        )
        return JSONResponse({
            "data": result.data.model_dump(mode="json"),
            "meta": {
                "backend": result.backend,
                "pages": result.pages_processed,
                "elapsed_seconds": round(result.elapsed_seconds, 2),
                "cost_usd": result.estimated_cost_usd,
            },
        })
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/evaluate")
async def evaluate_document(
    file: UploadFile = File(...),
    doc_type: str = Form(default="contract"),
    backends: str = Form(default="gemini,typhoon,gemma"),
    no_judge: bool = Form(default=False),
    first_page: int | None = Form(default=None),
    last_page: int | None = Form(default=None),
) -> JSONResponse:
    """Run multi-backend evaluation (slower — runs all backends in sequence)."""
    from document_ocr.evaluation.harness import EvaluationHarness

    if doc_type not in _VALID_DOC_TYPES:
        raise HTTPException(400, f"doc_type must be one of: {sorted(_VALID_DOC_TYPES)}")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        backend_list = [b.strip() for b in backends.split(",")]
        harness = EvaluationHarness(
            backends=backend_list,
            use_llm_judge=not no_judge,
        )
        report = harness.run(
            tmp_path,
            DocType(doc_type),
            first_page=first_page,
            last_page=last_page,
        )
        return JSONResponse(report)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/health")
def health_check() -> dict:
    """Return backend health status."""
    from document_ocr.extractors.gemini import GeminiExtractor
    from document_ocr.extractors.gemma import GemmaExtractor
    from document_ocr.extractors.typhoon import TyphoonExtractor
    from document_ocr.extractors.typhoon_api import TyphoonAPIExtractor

    backends = {
        "gemini": GeminiExtractor,
        "typhoon-api": TyphoonAPIExtractor,
        "typhoon": TyphoonExtractor,
        "gemma": GemmaExtractor,
    }
    status = {}
    for name, cls in backends.items():
        try:
            status[name] = cls().health_check()
        except Exception:
            status[name] = False
    return {"backends": status}
