"""
Legal clause detection using keyword and regex pattern matching.

Detects 6 clause types commonly found in legal contracts:
indemnity, liability, termination, non-compete, payment, and arbitration.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from parser import ParsedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clause types
# ---------------------------------------------------------------------------

class ClauseType(str, Enum):
    """Legal clause categories tracked by the system."""
    INDEMNITY = "indemnity"
    LIABILITY = "liability"
    TERMINATION = "termination"
    NON_COMPETE = "non_compete"
    PAYMENT = "payment"
    ARBITRATION = "arbitration"


# ---------------------------------------------------------------------------
# Detection result model
# ---------------------------------------------------------------------------

class DetectedClause(BaseModel):
    """A single detected clause occurrence."""

    clause_type: ClauseType
    matched_text: str = Field(
        description="The sentence or paragraph that triggered the match"
    )
    trigger_pattern: str = Field(
        description="The specific keyword or pattern that matched"
    )
    element_index: int = Field(
        description="Index of the source ParsedElement"
    )
    page_number: int | None = Field(
        default=None, description="Page where the clause was found"
    )
    section_heading: str | None = Field(
        default=None, description="Section heading context"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Detection confidence (0–1). Higher when multiple "
                    "patterns match in the same block.",
    )


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Each entry is (compiled regex, human-readable label, base confidence)
_ClausePatterns = list[tuple[re.Pattern[str], str, float]]


def _compile(patterns: list[tuple[str, str, float]]) -> _ClausePatterns:
    """Compile regex patterns with case-insensitive flag."""
    return [
        (re.compile(pat, re.IGNORECASE), label, conf)
        for pat, label, conf in patterns
    ]


CLAUSE_PATTERNS: dict[ClauseType, _ClausePatterns] = {
    ClauseType.INDEMNITY: _compile([
        (r"\bindemnif(?:y|ies|ied|ication)\b", "indemnify/indemnification", 0.9),
        (r"\bhold\s+harmless\b", "hold harmless", 0.9),
        (r"\bdefend\s+and\s+indemnify\b", "defend and indemnify", 0.95),
        (r"\bindemnit(?:y|ies)\b", "indemnity", 0.85),
        (r"\bsave\s+harmless\b", "save harmless", 0.8),
    ]),

    ClauseType.LIABILITY: _compile([
        (r"\blimitation\s+of\s+liability\b", "limitation of liability", 0.95),
        (r"\bliable\b", "liable", 0.7),
        (r"\bliability\b", "liability", 0.7),
        (r"\bdamages?\b", "damages", 0.5),
        (r"\bconsequential\s+damages?\b", "consequential damages", 0.9),
        (r"\bdirect\s+damages?\b", "direct damages", 0.85),
        (r"\baggregate\s+liability\b", "aggregate liability", 0.9),
        (r"\bexclusion\s+of\s+liability\b", "exclusion of liability", 0.95),
        (r"\bcap\s+on\s+liability\b", "cap on liability", 0.9),
    ]),

    ClauseType.TERMINATION: _compile([
        (r"\bterminat(?:e|ion|ed|ing)\b", "termination", 0.8),
        (r"\bcancell?ation\b", "cancellation", 0.8),
        (r"\bexpir(?:ation|e|es|ed)\b", "expiration", 0.7),
        (r"\bright\s+to\s+terminate\b", "right to terminate", 0.95),
        (r"\btermination\s+for\s+(?:cause|convenience)\b", "termination for cause/convenience", 0.95),
        (r"\bnotice\s+of\s+termination\b", "notice of termination", 0.9),
        (r"\bsurvival\b", "survival clause", 0.7),
    ]),

    ClauseType.NON_COMPETE: _compile([
        (r"\bnon[-\s]?compete\b", "non-compete", 0.95),
        (r"\bnon[-\s]?competition\b", "non-competition", 0.95),
        (r"\bnon[-\s]?solicitation\b", "non-solicitation", 0.9),
        (r"\brestrictive\s+covenant\b", "restrictive covenant", 0.9),
        (r"\bnon[-\s]?disclosure\b", "non-disclosure", 0.7),
        (r"\bconfidentialit(?:y|ies)\b", "confidentiality", 0.6),
        (r"\btrade\s+secret\b", "trade secret", 0.7),
        (r"\bexclusivit(?:y|ies)\b", "exclusivity", 0.8),
    ]),

    ClauseType.PAYMENT: _compile([
        (r"\bpayment\s+terms?\b", "payment terms", 0.95),
        (r"\binvoic(?:e|es|ing)\b", "invoice", 0.8),
        (r"\bnet\s+\d+\b", "net payment terms", 0.9),
        (r"\bcompensation\b", "compensation", 0.75),
        (r"\bfee(?:s)?\b", "fees", 0.5),
        (r"\breimburs(?:e|ement)\b", "reimbursement", 0.8),
        (r"\blate\s+payment\b", "late payment", 0.9),
        (r"\bpenalt(?:y|ies)\b", "penalty", 0.75),
        (r"\binterest\s+(?:rate|at)\b", "interest rate", 0.8),
        (r"\bmilestone\s+payment\b", "milestone payment", 0.9),
    ]),

    ClauseType.ARBITRATION: _compile([
        (r"\barbitrat(?:ion|e|or)\b", "arbitration", 0.95),
        (r"\bdispute\s+resolution\b", "dispute resolution", 0.9),
        (r"\bmediat(?:ion|e|or)\b", "mediation", 0.85),
        (r"\bgoverning\s+law\b", "governing law", 0.8),
        (r"\bjurisdiction\b", "jurisdiction", 0.75),
        (r"\bvenue\b", "venue", 0.6),
        (r"\bchoice\s+of\s+law\b", "choice of law", 0.85),
        (r"\bbinding\s+arbitration\b", "binding arbitration", 0.95),
        (r"\bforum\s+selection\b", "forum selection", 0.85),
    ]),
}


# ---------------------------------------------------------------------------
# Clause detector
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> list[str]:
    """Rough sentence-level splitting for granular clause matching."""
    # Split on period/semicolon followed by space + uppercase, or on newlines
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z])|(?:\n\s*\n)", text)
    return [p.strip() for p in parts if p.strip()]


class ClauseDetector:
    """
    Detects legal clauses in a parsed document using keyword / regex patterns.

    Usage::

        detector = ClauseDetector()
        clauses = detector.detect(parsed_document)
    """

    def detect(self, parsed_doc: ParsedDocument) -> list[DetectedClause]:
        """
        Scan every element in the document for clause patterns.

        Parameters
        ----------
        parsed_doc : ParsedDocument
            Output of ``load_document()``.

        Returns
        -------
        list[DetectedClause]
            All detected clause occurrences, ordered by element index.
        """
        from parser import ElementType  # avoid circular import at module level

        detected: list[DetectedClause] = []

        for element in parsed_doc.elements:
            # Skip images — nothing to match
            if element.element_type == ElementType.IMAGE:
                continue

            text = element.content
            if not text or len(text.strip()) < 10:
                continue

            # Check each clause type
            for clause_type, patterns in CLAUSE_PATTERNS.items():
                matched_patterns: list[tuple[str, float]] = []

                for regex, label, base_conf in patterns:
                    if regex.search(text):
                        matched_patterns.append((label, base_conf))

                if not matched_patterns:
                    continue

                # Confidence: max base + bonus for multiple matches
                max_conf = max(c for _, c in matched_patterns)
                bonus = min(0.05 * (len(matched_patterns) - 1), 0.1)
                confidence = min(max_conf + bonus, 1.0)

                # Get the best-matching sentence for context
                sentences = _split_into_sentences(text)
                best_sentence = text  # fallback to full text
                best_score = 0

                for sentence in sentences:
                    score = sum(
                        1 for regex, _, _ in patterns if regex.search(sentence)
                    )
                    if score > best_score:
                        best_score = score
                        best_sentence = sentence

                # Truncate very long matches for readability
                if len(best_sentence) > 500:
                    best_sentence = best_sentence[:497] + "..."

                trigger = ", ".join(label for label, _ in matched_patterns)

                detected.append(
                    DetectedClause(
                        clause_type=clause_type,
                        matched_text=best_sentence,
                        trigger_pattern=trigger,
                        element_index=element.element_index,
                        page_number=element.page_number,
                        section_heading=element.section_heading,
                        confidence=round(confidence, 2),
                    )
                )

        logger.info(
            "Detected %d clause(s) across %d type(s) in %s",
            len(detected),
            len({c.clause_type for c in detected}),
            parsed_doc.source_file,
        )

        return detected
