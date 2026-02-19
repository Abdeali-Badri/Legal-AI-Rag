"""
CLI entry point for the Legal AI document processing pipeline.

Usage:
    python app.py <file_path>
    python app.py data/contract.pdf
    python app.py data/agreement.docx --chunk-size 800 --chunk-overlap 150
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

import argparse
import json
import logging
import sys
from pathlib import Path

from legal_ai import (
    ClauseDetector,
    RiskClassifier,
)
from parser import load_document
from utils import CitationChunk, build_chunks


def setup_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s │ %(name)s │ %(message)s",
    )


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_RISK_COLORS = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def _print_header(title: str) -> None:
    width = 60
    print()
    print("═" * width)
    print(f"  {title}")
    print("═" * width)


def _print_section(title: str) -> None:
    print(f"\n── {title} {'─' * (50 - len(title))}")


def _print_element_summary(parsed_doc) -> None:
    _print_section("Extracted Elements")
    print(f"  Total elements : {len(parsed_doc.elements)}")
    print(f"  Text blocks    : {parsed_doc.text_count}")
    print(f"  Tables         : {parsed_doc.table_count}")
    print(f"  Images         : {parsed_doc.image_count}")


def _print_clauses(clauses) -> None:
    _print_section(f"Detected Clauses ({len(clauses)})")
    if not clauses:
        print("  No legal clauses detected.")
        return

    for i, clause in enumerate(clauses, 1):
        print(f"\n  [{i}] {clause.clause_type.value.upper()}")
        print(f"      Triggers  : {clause.trigger_pattern}")
        print(f"      Confidence: {clause.confidence:.0%}")
        if clause.page_number:
            print(f"      Page      : {clause.page_number}")
        if clause.section_heading:
            print(f"      Section   : {clause.section_heading}")
        # Show truncated matched text
        text = clause.matched_text
        if len(text) > 120:
            text = text[:117] + "..."
        print(f"      Text      : \"{text}\"")


def _count_risks(risks) -> dict[str, int]:
    """Count risk levels — used by both console output and JSON report."""
    counts = {"high": 0, "medium": 0, "low": 0}
    for r in risks:
        counts[r.risk_level.value] += 1
    return counts


def _print_risks(risks) -> None:
    _print_section("Risk Assessments")
    if not risks:
        print("  No risks to assess.")
        return

    counts = _count_risks(risks)
    print(f"  🔴 HIGH: {counts['high']}  │  🟡 MEDIUM: {counts['medium']}  │  🟢 LOW: {counts['low']}")

    for i, risk in enumerate(risks, 1):
        icon = _RISK_COLORS.get(risk.risk_level.value, "⚪")
        print(f"\n  {icon} [{i}] {risk.clause_type.value.upper()} → {risk.risk_level.value.upper()}")
        for factor in risk.risk_factors:
            print(f"      • {factor}")


def _print_chunks(chunks: list[CitationChunk]) -> None:
    _print_section(f"Citation-Aware Chunks ({len(chunks)})")
    text_chunks = sum(1 for c in chunks if c.element_type == "text")
    table_chunks = sum(1 for c in chunks if c.element_type == "table")
    image_chunks = sum(1 for c in chunks if c.element_type == "image")
    annotated = sum(1 for c in chunks if c.clause_types)

    print(f"  Text chunks    : {text_chunks}")
    print(f"  Table chunks   : {table_chunks}")
    print(f"  Image chunks   : {image_chunks}")
    print(f"  Annotated      : {annotated} (carry clause/risk metadata)")

    # Show first few chunks as examples
    print("\n  Sample chunks:")
    for chunk in chunks[:3]:
        preview = chunk.content[:80].replace("\n", " ")
        if len(chunk.content) > 80:
            preview += "..."
        annotations = ""
        if chunk.clause_types:
            annotations = f" [{', '.join(chunk.clause_types)}]"
        print(f"    • [{chunk.element_type}] p.{chunk.page_number}: \"{preview}\"{annotations}")


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def _save_report(
    parsed_doc,
    clauses,
    risks,
    chunks: list[CitationChunk],
    output_path: Path,
) -> None:
    """Save structured JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "source_file": parsed_doc.source_file,
        "file_type": parsed_doc.file_type,
        "summary": {
            "total_elements": len(parsed_doc.elements),
            "text_blocks": parsed_doc.text_count,
            "tables": parsed_doc.table_count,
            "images": parsed_doc.image_count,
            "clauses_detected": len(clauses),
            "total_chunks": len(chunks),
            "risk_breakdown": _count_risks(risks),
        },
        "clauses": [c.model_dump() for c in clauses],
        "risk_assessments": [r.model_dump() for r in risks],
        "chunks": [c.model_dump() for c in chunks],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  📄 Report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    output_dir: str = "data/output",
) -> dict:
    """
    Run the full document processing pipeline.

    Returns the structured report as a dict.
    """
    _print_header("Legal AI Document Processor")
    print(f"  Input: {file_path}")

    # Step 1: Parse
    print("\n  ⏳ Parsing document...")
    parsed_doc = load_document(file_path)
    _print_element_summary(parsed_doc)

    # Step 2: Detect clauses
    print("\n  ⏳ Detecting legal clauses...")
    detector = ClauseDetector()
    clauses = detector.detect(parsed_doc)
    _print_clauses(clauses)

    # Step 3: Classify risks
    print("\n  ⏳ Classifying risks...")
    classifier = RiskClassifier()
    risks = classifier.classify(clauses)
    _print_risks(risks)

    # Step 4: Build chunks
    print("\n  ⏳ Building citation-aware chunks...")
    chunks = build_chunks(
        parsed_doc, clauses, risks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    _print_chunks(chunks)

    # Step 5: Save report
    source_name = Path(file_path).stem
    output_path = Path(output_dir) / f"{source_name}_report.json"
    _save_report(parsed_doc, clauses, risks, chunks, output_path)

    _print_header("Pipeline Complete ✓")

    return {
        "parsed_doc": parsed_doc,
        "clauses": clauses,
        "risks": risks,
        "chunks": chunks,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legal AI Document Processor — parse, detect clauses, "
                    "classify risk, and build citation-aware chunks.",
    )
    parser.add_argument(
        "file",
        help="Path to the legal document (PDF, DOCX, TXT)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Max characters per text chunk (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Character overlap between chunks (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory for the JSON report (default: data/output)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        run_pipeline(
            args.file,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.exception("Pipeline failed")
        print(f"\n  ❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
