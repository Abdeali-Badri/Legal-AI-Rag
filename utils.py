"""
Citation-aware text chunking for the Legal AI RAG pipeline.

Splits a parsed and annotated document into retrieval-ready chunks
while preserving:
  - Page numbers and source file references
  - Section headings
  - Clause type and risk level annotations
  - Table/image integrity (never split atomic elements)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from legal_ai.clause_detector import ClauseType, DetectedClause
from legal_ai.risk_classifier import RiskAssessment, RiskLevel
from parser import ElementType, ParsedDocument, ParsedElement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------

class CitationChunk(BaseModel):
    """A retrieval-ready chunk with full citation and annotation metadata."""

    chunk_id: str = Field(description="Unique identifier: <source>_chunk_<N>")
    content: str = Field(description="Chunk text, markdown table, or base64 image")
    element_type: str = Field(description="text | table | image")

    # Citation metadata
    source_file: str
    page_number: int | None = None
    section_heading: str | None = None
    element_indices: list[int] = Field(
        default_factory=list,
        description="Indices of source ParsedElements that contribute to this chunk",
    )

    # Legal annotations
    clause_types: list[str] = Field(
        default_factory=list,
        description="Clause types detected in this chunk",
    )
    risk_levels: list[str] = Field(
        default_factory=list,
        description="Risk levels associated with detected clauses",
    )
    risk_explanations: list[str] = Field(
        default_factory=list,
        description="Risk explanations for each detected clause",
    )

    # Extra
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_annotation_index(
    clauses: list[DetectedClause],
    risks: list[RiskAssessment],
) -> dict[int, list[tuple[ClauseType, RiskLevel, str]]]:
    """
    Build a mapping: element_index → list of (clause_type, risk_level, explanation).
    """
    risk_by_idx: dict[int, list[tuple[ClauseType, RiskLevel, str]]] = {}

    for i, clause in enumerate(clauses):
        if i < len(risks):
            risk = risks[i]
            entry = (clause.clause_type, risk.risk_level, risk.explanation)
        else:
            entry = (clause.clause_type, RiskLevel.MEDIUM, "Risk not assessed")
        risk_by_idx.setdefault(clause.element_index, []).append(entry)

    return risk_by_idx


def _is_atomic(element: ParsedElement) -> bool:
    """Tables and images should never be split."""
    return element.element_type in (ElementType.TABLE, ElementType.IMAGE)


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def build_chunks(
    parsed_doc: ParsedDocument,
    clauses: list[DetectedClause],
    risks: list[RiskAssessment],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[CitationChunk]:
    """
    Convert a parsed + annotated document into citation-aware chunks.

    Strategy:
        1. Tables and images are kept as **atomic chunks** (never split).
        2. Consecutive text elements in the same section are merged,
           then split with ``RecursiveCharacterTextSplitter``.
        3. Each chunk carries page numbers, section headings, clause types,
           and risk levels as metadata.

    Parameters
    ----------
    parsed_doc : ParsedDocument
        Output of ``load_document()``.
    clauses : list[DetectedClause]
        Output of ``ClauseDetector.detect()``.
    risks : list[RiskAssessment]
        Output of ``RiskClassifier.classify()``.
    chunk_size : int
        Maximum characters per text chunk.
    chunk_overlap : int
        Character overlap between consecutive text chunks.

    Returns
    -------
    list[CitationChunk]
        Retrieval-ready chunks with citation and risk metadata.
    """
    annotation_index = _build_annotation_index(clauses, risks)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )

    chunks: list[CitationChunk] = []
    chunk_counter = 0

    # Group elements into runs: atomic (table/image) vs. text blocks
    groups = _group_elements(parsed_doc.elements)

    for group in groups:
        if len(group) == 1 and _is_atomic(group[0]):
            # Atomic chunk (table or image)
            el = group[0]
            annotations = annotation_index.get(el.element_index, [])

            chunk_counter += 1
            chunks.append(
                CitationChunk(
                    chunk_id=f"{parsed_doc.source_file}_chunk_{chunk_counter}",
                    content=el.content,
                    element_type=el.element_type.value,
                    source_file=parsed_doc.source_file,
                    page_number=el.page_number,
                    section_heading=el.section_heading,
                    element_indices=[el.element_index],
                    clause_types=[a[0].value for a in annotations],
                    risk_levels=[a[1].value for a in annotations],
                    risk_explanations=[a[2] for a in annotations],
                )
            )
        else:
            # Text block: merge then split
            merged_text = "\n\n".join(el.content for el in group if el.content.strip())
            if not merged_text.strip():
                continue

            # Gather metadata from all elements in the group
            all_indices = [el.element_index for el in group]
            page_numbers = [el.page_number for el in group if el.page_number]
            section_headings = [el.section_heading for el in group if el.section_heading]

            # Split the merged text
            sub_texts = text_splitter.split_text(merged_text)

            for sub_text in sub_texts:
                chunk_counter += 1

                # Determine which element's annotations apply to this sub-chunk
                # by matching clause text against the sub-chunk content
                relevant_annotations = _match_annotations_to_subchunk(
                    sub_text, group, annotation_index
                )

                chunks.append(
                    CitationChunk(
                        chunk_id=f"{parsed_doc.source_file}_chunk_{chunk_counter}",
                        content=sub_text,
                        element_type="text",
                        source_file=parsed_doc.source_file,
                        page_number=page_numbers[0] if page_numbers else None,
                        section_heading=section_headings[-1] if section_headings else None,
                        element_indices=all_indices,
                        clause_types=[a[0].value for a in relevant_annotations],
                        risk_levels=[a[1].value for a in relevant_annotations],
                        risk_explanations=[a[2] for a in relevant_annotations],
                    )
                )

    logger.info(
        "Built %d chunks from %d elements (%d atomic, %d text-split)",
        len(chunks),
        len(parsed_doc.elements),
        sum(1 for c in chunks if c.element_type in ("table", "image")),
        sum(1 for c in chunks if c.element_type == "text"),
    )

    return chunks


# ---------------------------------------------------------------------------
# Grouping logic
# ---------------------------------------------------------------------------

def _group_elements(elements: list[ParsedElement]) -> list[list[ParsedElement]]:
    """
    Group consecutive text elements together; keep atomic elements alone.

    This ensures tables and images are never merged with surrounding text,
    while consecutive text blocks (in the same section) are combined
    for better splitting.
    """
    if not elements:
        return []

    groups: list[list[ParsedElement]] = []
    current_text_group: list[ParsedElement] = []

    for el in elements:
        if _is_atomic(el):
            # Flush any pending text group
            if current_text_group:
                groups.append(current_text_group)
                current_text_group = []
            # Add atomic element as its own group
            groups.append([el])
        else:
            current_text_group.append(el)

    # Flush final text group
    if current_text_group:
        groups.append(current_text_group)

    return groups


def _match_annotations_to_subchunk(
    sub_text: str,
    group_elements: list[ParsedElement],
    annotation_index: dict[int, list[tuple[ClauseType, RiskLevel, str]]],
) -> list[tuple[ClauseType, RiskLevel, str]]:
    """
    For a sub-chunk of text, find which annotations (from the original
    elements) are relevant by checking if any of the element's content
    overlaps with the sub-chunk.
    """
    relevant: list[tuple[ClauseType, RiskLevel, str]] = []
    seen: set[tuple[str, str]] = set()  # (clause_type, risk_level) dedup

    for el in group_elements:
        annotations = annotation_index.get(el.element_index, [])
        if not annotations:
            continue

        # Check if this element's content overlaps with the sub-chunk
        # Use a simple heuristic: first 50 chars of the element appear in the sub-chunk
        snippet = el.content[:50].strip()
        if snippet and snippet in sub_text:
            for ann in annotations:
                key = (ann[0].value, ann[1].value)
                if key not in seen:
                    seen.add(key)
                    relevant.append(ann)

    # If no overlap matched, include all annotations from the group
    # (conservative: better to over-annotate than miss a clause)
    if not relevant:
        for el in group_elements:
            for ann in annotation_index.get(el.element_index, []):
                key = (ann[0].value, ann[1].value)
                if key not in seen:
                    seen.add(key)
                    relevant.append(ann)

    return relevant
