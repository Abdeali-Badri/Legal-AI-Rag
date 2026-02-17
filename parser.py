"""
Document loader and parser for legal documents.

Uses LangChain's Unstructured integration to extract text, tables,
and images from PDF, DOCX, and TXT files. No OCR is performed —
images are preserved as-is for multimodal retrieval.
"""

from __future__ import annotations

import base64
import logging
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ElementType(str, Enum):
    """Types of elements extracted from a document."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    TITLE = "title"
    HEADER = "header"
    FOOTER = "footer"
    LIST_ITEM = "list_item"
    NARRATIVE_TEXT = "narrative_text"


class ParsedElement(BaseModel):
    """A single extracted element (text block, table, or image)."""

    element_type: ElementType
    content: str = Field(
        description="Text content, markdown table, or base64-encoded image data"
    )
    page_number: int | None = Field(
        default=None, description="1-indexed page number"
    )
    element_index: int = Field(
        description="Position of this element in the document (0-indexed)"
    )
    section_heading: str | None = Field(
        default=None,
        description="Nearest preceding heading / title element",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata from Unstructured (bounding box, etc.)",
    )


class ParsedDocument(BaseModel):
    """Complete parsed representation of a single document."""

    source_file: str
    file_type: str
    elements: list[ParsedElement] = Field(default_factory=list)

    # Convenience counts -------------------------------------------------------
    @property
    def text_count(self) -> int:
        return sum(
            1 for e in self.elements
            if e.element_type not in (ElementType.TABLE, ElementType.IMAGE)
        )

    @property
    def table_count(self) -> int:
        return sum(1 for e in self.elements if e.element_type == ElementType.TABLE)

    @property
    def image_count(self) -> int:
        return sum(1 for e in self.elements if e.element_type == ElementType.IMAGE)


# ---------------------------------------------------------------------------
# Element-type mapping from Unstructured category names
# ---------------------------------------------------------------------------

_CATEGORY_MAP: dict[str, ElementType] = {
    "Title": ElementType.TITLE,
    "Header": ElementType.HEADER,
    "Footer": ElementType.FOOTER,
    "NarrativeText": ElementType.NARRATIVE_TEXT,
    "ListItem": ElementType.LIST_ITEM,
    "Table": ElementType.TABLE,
    "Image": ElementType.IMAGE,
    "FigureCaption": ElementType.TEXT,
    "Text": ElementType.TEXT,
    "UncategorizedText": ElementType.TEXT,
    "Formula": ElementType.TEXT,
    "Address": ElementType.TEXT,
    "EmailAddress": ElementType.TEXT,
    "PageBreak": ElementType.TEXT,
}


def _resolve_element_type(category: str) -> ElementType:
    """Map an Unstructured category string to our ElementType enum."""
    return _CATEGORY_MAP.get(category, ElementType.TEXT)


# ---------------------------------------------------------------------------
# HTML table → Markdown converter
# ---------------------------------------------------------------------------

def _html_table_to_markdown(html: str) -> str:
    """Best-effort conversion of an HTML <table> to a markdown table."""
    try:
        # Use a simple regex-based approach to avoid heavy dependencies
        import re

        rows: list[list[str]] = []
        for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL):
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr_match.group(1), re.DOTALL)
            # Strip nested HTML tags from cell content
            clean_cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            rows.append(clean_cells)

        if not rows:
            return html  # fallback: return raw HTML

        # Build markdown table
        col_count = max(len(r) for r in rows)
        # Pad rows to uniform width
        for row in rows:
            while len(row) < col_count:
                row.append("")

        lines: list[str] = []
        # Header row
        lines.append("| " + " | ".join(rows[0]) + " |")
        lines.append("| " + " | ".join(["---"] * col_count) + " |")
        # Data rows
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
    except Exception:
        return html


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".rtf"}


def load_document(file_path: str | Path) -> ParsedDocument:
    """
    Load and parse a document into structured elements.

    Supports PDF, DOCX, TXT, and other formats handled by Unstructured.
    Tables are converted to markdown. Images are stored as base64.
    No OCR is performed.

    Parameters
    ----------
    file_path : str | Path
        Path to the document file.

    Returns
    -------
    ParsedDocument
        Structured representation with text, tables, and images.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file type is not supported.
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    logger.info("Loading document: %s", file_path)

    # ------------------------------------------------------------------
    # Use Unstructured to partition the document
    # ------------------------------------------------------------------
    from langchain_unstructured import UnstructuredLoader

    loader_kwargs: dict[str, Any] = {
        "mode": "elements",          # get individual elements
        "strategy": "fast",          # no OCR — fast text extraction
    }

    # PDF-specific: enable table structure inference
    if suffix == ".pdf":
        loader_kwargs["infer_table_structure"] = True

    loader = UnstructuredLoader(str(file_path), **loader_kwargs)

    try:
        raw_docs = loader.load()
    except Exception as exc:
        logger.error("Unstructured failed to parse %s: %s", file_path, exc)
        raise

    logger.info("Extracted %d raw elements from %s", len(raw_docs), file_path.name)

    # ------------------------------------------------------------------
    # Convert raw LangChain Documents → ParsedElements
    # ------------------------------------------------------------------
    elements: list[ParsedElement] = []
    current_heading: str | None = None

    for idx, doc in enumerate(raw_docs):
        meta = doc.metadata or {}
        category = meta.get("category", "Text")
        el_type = _resolve_element_type(category)

        # Track the current section heading
        if el_type in (ElementType.TITLE, ElementType.HEADER):
            current_heading = doc.page_content.strip() or current_heading

        # Process content based on type
        content = doc.page_content

        if el_type == ElementType.TABLE:
            # Convert HTML table to markdown if available
            html_table = meta.get("text_as_html", "")
            if html_table:
                content = _html_table_to_markdown(html_table)

        elif el_type == ElementType.IMAGE:
            # Store image data as base64 if available
            image_base64 = meta.get("image_base64", "")
            image_path = meta.get("image_path", "")
            if image_base64:
                content = image_base64
            elif image_path:
                try:
                    with open(image_path, "rb") as f:
                        content = base64.b64encode(f.read()).decode("utf-8")
                except Exception:
                    content = f"[Image: {image_path}]"
            else:
                content = doc.page_content or "[Embedded Image]"

        page_num = meta.get("page_number")

        # Build extra metadata dict (exclude keys we already store)
        extra_meta = {
            k: v
            for k, v in meta.items()
            if k not in {"category", "page_number", "text_as_html",
                         "image_base64", "image_path"}
        }

        elements.append(
            ParsedElement(
                element_type=el_type,
                content=content,
                page_number=page_num,
                element_index=idx,
                section_heading=current_heading,
                metadata=extra_meta,
            )
        )

    parsed = ParsedDocument(
        source_file=str(file_path),
        file_type=suffix.lstrip("."),
        elements=elements,
    )

    logger.info(
        "Parsed %s: %d text, %d tables, %d images",
        file_path.name,
        parsed.text_count,
        parsed.table_count,
        parsed.image_count,
    )

    return parsed
