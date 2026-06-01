# Thai Document OCR

Extract structured data from Thai scanned contracts and supplier invoices. Supports four interchangeable backends — two cloud APIs and two fully self-hosted — so you can trade cost against latency and privacy.

## Backends

| Backend | How it works | Cost | Latency/page |
|---|---|---|---|
| `gemini` | Gemini 2.5 Flash (cloud) — image → structured JSON | ~$0.001–0.005/doc | 3–8 s |
| `typhoon-api` | OpenTyphoon cloud API (free for research) → Gemma struct | $0 | 5–15 s |
| `typhoon` | Typhoon OCR 1.5 (Ollama) → Gemma 3 4B struct | $0 (CPU infra) | 60–120 s |
| `gemma` | Tesseract Thai OCR → Gemma 3 4B struct | $0 (CPU infra) | 15–30 s |

**Why two stages for `typhoon`?** Typhoon OCR 1.5 is fine-tuned for Markdown transcription, not JSON generation. Forcing JSON output degrades OCR quality. Keeping OCR (Typhoon) and structuring (Gemma) as separate passes gives best accuracy per stage.

**Accuracy benchmark:** Typhoon OCR 1.5 outperforms Gemini 2.5 Flash on Thai financial documents (BLEU 0.91 vs 0.52, ROUGE-L 0.94 vs 0.70), making the self-hosted path both cheaper and more accurate.

## Extracted Fields

### Thai Contracts (`ThaiContract`)
`contract_reference` · `parties` (name, address, role, tax_id) · `contract_date` · `effective_date` · `expiry_date` · `contract_value` · `currency` · `payment_terms` · `key_obligations` · `penalty_clauses` (description, amount, rate_per_day)

### Supplier Invoices (`SupplierInvoice`)
`vendor_name` · `vendor_address` · `vendor_tax_id` · `invoice_number` · `invoice_date` · `payment_due_date` · `line_items` (description, quantity, unit_price, total_price) · `subtotal` · `vat_amount` · `vat_rate` · `total_amount` · `buyer` (name, address, tax_id)

Buddhist Era dates are automatically converted to CE (subtract 543, e.g. `15/06/2568` → `2025-06-15`).

## Requirements

**Python:** 3.11+

**System packages:**
```bash
sudo apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-tha fonts-thai-tlwg zstd
```

**Ollama** (for `typhoon` and `gemma` backends):
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Installation

```bash
git clone https://github.com/sorhoho/document_ocr
cd document_ocr
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env with your API keys
```

## Setup

### Self-hosted models (Typhoon + Gemma)

Pull both models (~6.5 GB total):
```bash
bash scripts/setup_ollama_models.sh
```

Or manually:
```bash
ollama pull scb10x/typhoon-ocr1.5-3b   # ~3.2 GB — Typhoon OCR VLM
ollama pull gemma3:4b                   # ~3.3 GB — Gemma struct extractor
```

> **Memory note:** Typhoon OCR requires ~10 GB runtime RAM; Gemma needs ~4 GB. On a 15 GB system they cannot run simultaneously. The `typhoon` backend automatically unloads Typhoon from memory before loading Gemma.

### Cloud APIs

Set in `.env`:
```bash
GEMINI_API_KEY=your-key      # https://aistudio.google.com/app/apikey
TYPHOON_API_KEY=your-key     # https://opentyphoon.ai (free for research)
```

## CLI Usage

```bash
# Check all backends
docr health

# Extract with self-hosted Typhoon (default)
docr extract invoice.pdf --doc-type invoice --backend typhoon

# Extract with Gemini cloud
docr extract contract.pdf --doc-type contract --backend gemini --output result.json

# Extract specific pages only
docr extract contract.pdf --first-page 1 --last-page 3 --backend typhoon

# Multi-backend evaluation with cross-model agreement + LLM-as-judge
docr evaluate invoice.pdf --doc-type invoice --backends gemini,typhoon,gemma --output report.json
```

**Available backends:** `gemini` · `typhoon-api` · `typhoon` · `gemma`

**Available doc types:** `contract` · `invoice`

## REST API

```bash
uvicorn api.main:app --reload
```

```
POST /extract    — Extract one document
POST /evaluate   — Multi-backend comparison
GET  /health     — Backend status
```

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@invoice.pdf" \
  -F "doc_type=invoice" \
  -F "backend=typhoon"
```

## Evaluation

The evaluation harness runs without labelled ground truth using two strategies:

**Cross-model agreement** — pairwise similarity across all backends. Fields where ≥ 2 backends agree (score ≥ 0.8) are "high confidence." Majority vote determines the consensus value.

**LLM-as-judge** — Gemini 2.5 Flash (the production baseline) scores candidate outputs from Typhoon/Gemma using a Thai-aware rubric. Accepts Thai/romanised name equivalents, BE/CE date formats, and number formatting differences as `CORRECT`.

```bash
# Compare all three backends and write a full report
docr evaluate invoice.pdf --doc-type invoice \
  --backends gemini,typhoon,gemma \
  --output report.json
```

The report includes per-field agreement scores, judge verdicts, and cost/time per backend.

## Project Layout

```
document_ocr/
├── document_ocr/
│   ├── config.py                 # pydantic-settings (API keys, model names, DPI)
│   ├── pipeline.py               # ExtractionPipeline — top-level orchestrator
│   ├── schemas/
│   │   ├── contract.py           # ThaiContract Pydantic model
│   │   └── invoice.py            # SupplierInvoice Pydantic model
│   ├── preprocessing/
│   │   ├── pdf_converter.py      # pdf_to_images() via pdf2image (300 DPI)
│   │   └── image_cleaner.py      # deskew, denoise, resize per backend
│   ├── extractors/
│   │   ├── base.py               # BaseExtractor ABC + ExtractionResult
│   │   ├── gemini.py             # GeminiExtractor (google-genai SDK)
│   │   ├── typhoon_api.py        # TyphoonAPIExtractor (OpenTyphoon cloud)
│   │   ├── typhoon.py            # TyphoonExtractor (Ollama VLM → Gemma struct)
│   │   ├── gemma.py              # GemmaExtractor (Tesseract → Gemma struct)
│   │   ├── _struct_helper.py     # Shared Gemma JSON extraction (template-based)
│   │   ├── _schema_utils.py      # Pydantic → Ollama-safe JSON schema flattener
│   │   └── _prompts.py           # Thai-aware extraction prompts
│   └── evaluation/
│       ├── metrics.py            # exact_match, numeric_match, fuzzy_text_match
│       ├── agreement.py          # Cross-model agreement scoring
│       ├── judge.py              # LLM-as-judge (Gemini-scored rubric)
│       ├── cost_tracker.py       # API cost + CPU infra cost per extraction
│       └── harness.py            # EvaluationHarness — orchestrates full eval run
├── cli/main.py                   # Typer CLI: docr extract / evaluate / health
├── api/main.py                   # FastAPI: POST /extract, POST /evaluate, GET /health
├── scripts/
│   ├── setup_ollama_models.sh    # Pull both Ollama models
│   └── create_test_invoice.py    # Generate synthetic Thai invoice for testing
└── tests/
    ├── test_schemas.py
    ├── test_preprocessing.py
    ├── test_evaluation.py
    └── test_smoke_local.py       # End-to-end Typhoon + Gemma test (auto-skipped without Ollama)
```

## Tests

```bash
# Unit tests (no Ollama or API keys required)
pytest tests/ -v --ignore=tests/test_smoke_local.py

# Full smoke test (requires Ollama + both models pulled)
pytest tests/test_smoke_local.py -v
```

The smoke test auto-skips if Ollama is not running or the Typhoon model is not pulled.

## Configuration

All settings can be set via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Google AI Studio API key |
| `TYPHOON_API_KEY` | — | OpenTyphoon API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `TYPHOON_OCR_MODEL` | `scb10x/typhoon-ocr1.5-3b` | OCR model |
| `GEMMA_STRUCT_MODEL` | `gemma3:4b` | Struct extraction model |
| `PDF_DPI` | `300` | PDF rasterisation DPI |
| `VLM_MAX_SIDE_PX` | `1800` | Max image side for VLM backends |
| `DESKEW_ENABLED` | `true` | Auto-deskew scanned pages |
| `CPU_COST_PER_HOUR_USD` | `0.05` | CPU cost for infra cost tracking |
| `LLM_JUDGE_ENABLED` | `true` | Use Gemini as judge in evaluations |

## Notes

- **Ollama >= 0.4** is required (for structured output support).
- **GGUF files** for Typhoon should not be used directly via llama.cpp/LM Studio — the official model card warns of accuracy degradation. Use `ollama pull` instead.
- The `typhoon` backend uses `keep_alive=0` on the final OCR page to free Typhoon from memory before Gemma loads. Do not disable this on servers with ≤ 16 GB RAM.
- Thai numeral normalisation (`๐–๙` → `0–9`) is applied automatically in metric comparisons.
