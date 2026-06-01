"""Thai document OCR extraction CLI.

Commands:
  extract    Extract one document with a chosen backend.
  evaluate   Run multi-backend evaluation with agreement + judge scoring.
  health     Check connectivity to all backends.

Usage:
  docr extract invoice.pdf --backend typhoon
  docr evaluate contract.pdf --backends gemini,typhoon,gemma --output report.json
  docr health
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from document_ocr.extractors.base import DocType
from document_ocr.pipeline import ExtractionPipeline

app = typer.Typer(help="Thai document OCR extraction", no_args_is_help=True)
console = Console()


@app.command()
def extract(
    pdf: Path = typer.Argument(..., help="Path to scanned PDF"),
    doc_type: DocType = typer.Option(DocType.CONTRACT, "--doc-type", "-t", help="Document type"),
    backend: str = typer.Option(
        "typhoon",
        "--backend",
        "-b",
        help="Backend: gemini | typhoon-api | typhoon | gemma",
    ),
    first_page: Optional[int] = typer.Option(None, "--first-page", help="Start page (1-indexed)"),
    last_page: Optional[int] = typer.Option(None, "--last-page", help="End page (1-indexed)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write JSON to file"),
    pretty: bool = typer.Option(True, help="Pretty-print JSON"),
) -> None:
    """Extract structured data from a Thai scanned PDF."""
    pipeline = ExtractionPipeline()
    result = pipeline.run(
        pdf_path=pdf,
        doc_type=doc_type,
        backend=backend,
        first_page=first_page,
        last_page=last_page,
    )

    json_str = result.data.model_dump_json(indent=2 if pretty else None)

    if output:
        output.write_text(json_str, encoding="utf-8")
        console.print(f"[green]Result written to {output}[/green]")
    else:
        console.print_json(json_str)

    console.print(
        f"\n[dim]Backend: {result.backend} | "
        f"Pages: {result.pages_processed} | "
        f"Time: {result.elapsed_seconds:.1f}s | "
        f"Cost: ${result.estimated_cost_usd:.4f}[/dim]"
    )


@app.command()
def evaluate(
    pdf: Path = typer.Argument(..., help="Path to scanned PDF"),
    doc_type: DocType = typer.Option(DocType.CONTRACT, "--doc-type", "-t"),
    backends: str = typer.Option(
        "gemini,typhoon,gemma",
        "--backends",
        help="Comma-separated backends to compare",
    ),
    first_page: Optional[int] = typer.Option(None, "--first-page"),
    last_page: Optional[int] = typer.Option(None, "--last-page"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM-as-judge"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write report JSON to file"),
) -> None:
    """Run multi-backend evaluation with agreement scoring and LLM-as-judge."""
    from document_ocr.evaluation.harness import EvaluationHarness

    backend_list = [b.strip() for b in backends.split(",")]
    harness = EvaluationHarness(
        backends=backend_list,
        use_llm_judge=not no_judge,
    )
    report = harness.run(
        pdf,
        doc_type,
        first_page=first_page,
        last_page=last_page,
    )

    # Agreement table
    table = Table(title="Field Agreement Scores", show_lines=True)
    table.add_column("Field", style="cyan")
    table.add_column("Agreement", justify="right")
    table.add_column("Confidence")
    for field, score in report["agreement"]["per_field"].items():
        conf_label = "HIGH" if score >= 0.8 else "LOW"
        color = "green" if score >= 0.8 else "red"
        table.add_row(field, f"{score:.2f}", f"[{color}]{conf_label}[/{color}]")
    console.print(table)
    console.print(f"\nOverall agreement: {report['agreement']['overall']:.2f}")

    # Judge scores
    if report.get("judge"):
        console.print("\n[bold]LLM-as-Judge Scores (vs Gemini reference)[/bold]")
        for key, value in report["judge"].items():
            if key.endswith("_score"):
                backend_name = key.replace("_score", "")
                console.print(f"  {backend_name}: {value:.2f}")

    # Cost summary
    console.print("\n[bold]Cost Summary[/bold]")
    for backend_name, costs in report["costs"].items():
        console.print(
            f"  {backend_name}: avg ${costs['avg_cost_usd']:.4f}/doc, "
            f"{costs['avg_elapsed_seconds']:.1f}s/doc"
        )

    if output:
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        console.print(f"\n[green]Full report written to {output}[/green]")


@app.command()
def health() -> None:
    """Check connectivity to all backends."""
    from document_ocr.extractors.gemini import GeminiExtractor
    from document_ocr.extractors.gemma import GemmaExtractor
    from document_ocr.extractors.typhoon import TyphoonExtractor
    from document_ocr.extractors.typhoon_api import TyphoonAPIExtractor

    backends = [
        ("Gemini 2.5 Flash (cloud)", GeminiExtractor),
        ("Typhoon OCR API (cloud, free)", TyphoonAPIExtractor),
        ("Typhoon OCR 1.5 (Ollama)", TyphoonExtractor),
        ("Gemma 3 4B + Tesseract (Ollama)", GemmaExtractor),
    ]
    for name, cls in backends:
        try:
            ok = cls().health_check()
        except Exception as e:
            ok = False
            console.print(f"  {name}: [red]ERROR[/red] — {e}")
            continue
        status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"  {name}: {status}")


if __name__ == "__main__":
    app()
