"""
Rule-based risk classification for detected legal clauses.

Classifies each clause as HIGH, MEDIUM, or LOW risk based on
the presence of risky language patterns and provides human-readable
explanations.
"""

from __future__ import annotations

import logging
import re
from enum import Enum

from pydantic import BaseModel, Field

from legal_ai.clause_detector import ClauseType, DetectedClause

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """Risk severity levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Risk assessment model
# ---------------------------------------------------------------------------

class RiskAssessment(BaseModel):
    """Risk assessment for a single detected clause."""

    clause_type: ClauseType
    risk_level: RiskLevel
    explanation: str = Field(
        description="Human-readable explanation of why this risk level was assigned"
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Specific language patterns that contributed to the risk score"
    )
    element_index: int
    page_number: int | None = None
    section_heading: str | None = None
    matched_text: str = ""


# ---------------------------------------------------------------------------
# Risk pattern definitions
# ---------------------------------------------------------------------------

# Each rule: (regex, risk_level, factor_description)
_RiskRule = tuple[re.Pattern[str], RiskLevel, str]


def _rules(entries: list[tuple[str, RiskLevel, str]]) -> list[_RiskRule]:
    return [(re.compile(p, re.IGNORECASE), lvl, desc) for p, lvl, desc in entries]


RISK_RULES: dict[ClauseType, list[_RiskRule]] = {
    # ------------------------------------------------------------------
    # INDEMNITY
    # ------------------------------------------------------------------
    ClauseType.INDEMNITY: _rules([
        # HIGH risk
        (r"\bunlimited\s+indemnif", RiskLevel.HIGH,
         "Unlimited indemnification obligation"),
        (r"\bsole(?:ly)?\s+(?:responsible|liability|indemnif)", RiskLevel.HIGH,
         "One-sided indemnification (sole responsibility)"),
        (r"\birrevocabl[ey]\b", RiskLevel.HIGH,
         "Irrevocable indemnification commitment"),
        (r"\bfull(?:y)?\s+indemnif", RiskLevel.HIGH,
         "Full indemnification without limitations"),
        (r"\bfirst[-\s]party\s+indemnif", RiskLevel.HIGH,
         "First-party bears all indemnification burden"),
        (r"\bincluding\s+(?:but\s+not\s+limited\s+to\s+)?attorney", RiskLevel.HIGH,
         "Indemnification includes attorney fees"),
        # MEDIUM risk
        (r"\breasonabl[ey]\b", RiskLevel.MEDIUM,
         "Reasonableness qualifier present (standard language)"),
        (r"\bsubject\s+to\s+(?:the\s+)?(?:cap|limit)", RiskLevel.MEDIUM,
         "Indemnification subject to caps/limits"),
        (r"\bexcept\s+(?:for|to\s+the\s+extent)", RiskLevel.MEDIUM,
         "Contains exceptions or carve-outs"),
        # LOW risk
        (r"\bmutual(?:ly)?\s+indemnif", RiskLevel.LOW,
         "Mutual indemnification (balanced obligation)"),
        (r"\beach\s+party\s+(?:shall|will|agrees?\s+to)\s+indemnif", RiskLevel.LOW,
         "Both parties share indemnification duties"),
    ]),

    # ------------------------------------------------------------------
    # LIABILITY
    # ------------------------------------------------------------------
    ClauseType.LIABILITY: _rules([
        # HIGH risk
        (r"\bunlimited\s+liabilit", RiskLevel.HIGH,
         "No cap on liability exposure"),
        (r"\bno\s+limit(?:ation)?\s+(?:on|of)\s+liabilit", RiskLevel.HIGH,
         "Explicit removal of liability limits"),
        (r"\bsole\s+discretion\b", RiskLevel.HIGH,
         "Sole discretion language (one-sided power)"),
        (r"\bwaiv(?:e|er|es)\s+(?:all|any)\s+(?:claim|right|liabilit)", RiskLevel.HIGH,
         "Broad waiver of claims/rights"),
        (r"\bstrict\s+liabilit", RiskLevel.HIGH,
         "Strict liability standard (no fault required)"),
        # MEDIUM risk
        (r"\baggregate\s+liabilit", RiskLevel.MEDIUM,
         "Aggregate liability cap (standard but watch amount)"),
        (r"\bnot\s+(?:to\s+)?exceed\b", RiskLevel.MEDIUM,
         "Contains a cap but review the cap amount"),
        (r"\bexclud(?:e|es|ing)\s+(?:indirect|consequential)", RiskLevel.MEDIUM,
         "Excludes consequential damages (common practice)"),
        # LOW risk
        (r"\bmutual(?:ly)?\s+(?:agree|limit)", RiskLevel.LOW,
         "Mutual liability agreement"),
        (r"\bproportionat[ey]\b", RiskLevel.LOW,
         "Proportionate liability (balanced)"),
    ]),

    # ------------------------------------------------------------------
    # TERMINATION
    # ------------------------------------------------------------------
    ClauseType.TERMINATION: _rules([
        # HIGH risk
        (r"\bimmediatel?y?\s+terminat", RiskLevel.HIGH,
         "Immediate termination without notice period"),
        (r"\bwithout\s+(?:prior\s+)?(?:notice|cause)\b", RiskLevel.HIGH,
         "Termination without notice or cause"),
        (r"\bsole\s+discretion\b", RiskLevel.HIGH,
         "Termination at sole discretion"),
        (r"\bautomatic(?:ally)?\s+terminat", RiskLevel.HIGH,
         "Automatic termination trigger"),
        (r"\bforfeit(?:ure|s)?\b", RiskLevel.HIGH,
         "Forfeiture upon termination"),
        # MEDIUM risk
        (r"\b\d+[-\s]?day(?:s)?\s+(?:prior\s+)?(?:written\s+)?notice\b", RiskLevel.MEDIUM,
         "Fixed notice period for termination"),
        (r"\btermination\s+for\s+cause\b", RiskLevel.MEDIUM,
         "Termination for cause (standard)"),
        (r"\bcure\s+period\b", RiskLevel.MEDIUM,
         "Cure period provided before termination"),
        # LOW risk
        (r"\beither\s+party\s+may\s+terminat", RiskLevel.LOW,
         "Mutual termination right"),
        (r"\bmutual(?:ly)?\s+(?:agree|terminat)", RiskLevel.LOW,
         "Termination by mutual agreement"),
        (r"\breasonabl[ey]\s+(?:prior\s+)?notice\b", RiskLevel.LOW,
         "Reasonable notice required"),
    ]),

    # ------------------------------------------------------------------
    # NON-COMPETE
    # ------------------------------------------------------------------
    ClauseType.NON_COMPETE: _rules([
        # HIGH risk
        (r"\bperpetual(?:ly)?\b", RiskLevel.HIGH,
         "Perpetual non-compete (no time limit)"),
        (r"\bindefinite(?:ly)?\b", RiskLevel.HIGH,
         "Indefinite restriction period"),
        (r"\bworldwide\b", RiskLevel.HIGH,
         "Worldwide geographic scope"),
        (r"\bglobal(?:ly)?\b", RiskLevel.HIGH,
         "Global restriction scope"),
        (r"\ball\s+(?:business|industr|activit)", RiskLevel.HIGH,
         "Unreasonably broad activity restriction"),
        (r"\b(?:5|6|7|8|9|10)\s*(?:year|yr)", RiskLevel.HIGH,
         "Restriction period of 5+ years"),
        # MEDIUM risk
        (r"\b(?:1|2|3|4)\s*(?:year|yr)", RiskLevel.MEDIUM,
         "Restriction period of 1-4 years"),
        (r"\b(?:6|12|18|24)\s*(?:month)", RiskLevel.MEDIUM,
         "Restriction period of 6-24 months"),
        (r"\bspecific\s+(?:area|region|territor)", RiskLevel.MEDIUM,
         "Geographically limited restriction"),
        # LOW risk
        (r"\breasonabl[ey]\s+(?:scope|time|period|duration|geograph)", RiskLevel.LOW,
         "Reasonably scoped restriction"),
        (r"\blimited\s+to\b", RiskLevel.LOW,
         "Contains explicit limitations"),
        (r"\bnarrowly\s+tailored\b", RiskLevel.LOW,
         "Narrowly tailored restriction"),
    ]),

    # ------------------------------------------------------------------
    # PAYMENT
    # ------------------------------------------------------------------
    ClauseType.PAYMENT: _rules([
        # HIGH risk
        (r"\b(?:penalty|penalt(?:ize|ies))\b", RiskLevel.HIGH,
         "Late payment penalties"),
        (r"\binterest\s+(?:at|of)\s+\d+\s*%", RiskLevel.HIGH,
         "Specific interest rate on overdue payments"),
        (r"\bimmediate(?:ly)?\s+(?:due|payable)\b", RiskLevel.HIGH,
         "Immediate payment obligation"),
        (r"\baccelerat(?:e|ion)\b", RiskLevel.HIGH,
         "Payment acceleration clause"),
        (r"\bnon[-\s]?refundable\b", RiskLevel.HIGH,
         "Non-refundable payment terms"),
        # MEDIUM risk
        (r"\bnet\s+(?:15|30|45)\b", RiskLevel.MEDIUM,
         "Standard net payment terms"),
        (r"\bupon\s+(?:receipt|completion|delivery)\b", RiskLevel.MEDIUM,
         "Payment upon milestone/delivery"),
        (r"\blate\s+(?:fee|charge|payment)\b", RiskLevel.MEDIUM,
         "Late payment provisions present"),
        # LOW risk
        (r"\bnet\s+(?:60|90)\b", RiskLevel.LOW,
         "Extended payment terms (net 60-90)"),
        (r"\bflexibl[ey]\s+payment\b", RiskLevel.LOW,
         "Flexible payment arrangements"),
        (r"\binstallment\b", RiskLevel.LOW,
         "Installment payment option available"),
    ]),

    # ------------------------------------------------------------------
    # ARBITRATION
    # ------------------------------------------------------------------
    ClauseType.ARBITRATION: _rules([
        # HIGH risk
        (r"\bbinding\s+arbitration\b", RiskLevel.HIGH,
         "Binding arbitration (waives right to trial)"),
        (r"\bwaiv(?:e|er)\s+(?:of\s+)?(?:right\s+to\s+)?(?:jury\s+)?trial\b", RiskLevel.HIGH,
         "Explicit waiver of trial rights"),
        (r"\bclass[-\s]?action\s+waiv", RiskLevel.HIGH,
         "Class action waiver"),
        (r"\bconfidential\s+arbitration\b", RiskLevel.HIGH,
         "Confidential arbitration (limits transparency)"),
        (r"\bprevailing\s+party\s+(?:shall|will)?\s*(?:be\s+)?(?:entitled|recover)", RiskLevel.HIGH,
         "Prevailing party attorney fees clause"),
        # MEDIUM risk
        (r"\bgoverning\s+law\b", RiskLevel.MEDIUM,
         "Governing law specified (review jurisdiction)"),
        (r"\bjurisdiction\b", RiskLevel.MEDIUM,
         "Specific jurisdiction designated"),
        (r"\bmediation\s+(?:before|prior|first)\b", RiskLevel.MEDIUM,
         "Mediation required before arbitration"),
        # LOW risk
        (r"\beither\s+party\s+may\s+(?:elect|choose|opt)\b", RiskLevel.LOW,
         "Either party can choose dispute resolution method"),
        (r"\bmutual(?:ly)?\s+(?:agree|select|chosen)\b", RiskLevel.LOW,
         "Mutual agreement on arbitration terms"),
        (r"\bnon[-\s]?binding\b", RiskLevel.LOW,
         "Non-binding arbitration (preserves trial rights)"),
    ]),
}


# ---------------------------------------------------------------------------
# Risk classifier
# ---------------------------------------------------------------------------

# Priority: HIGH > MEDIUM > LOW
_RISK_PRIORITY = {RiskLevel.HIGH: 3, RiskLevel.MEDIUM: 2, RiskLevel.LOW: 1}


class RiskClassifier:
    """
    Classify detected clauses by risk level using heuristic rules.

    Usage::

        classifier = RiskClassifier()
        risks = classifier.classify(detected_clauses)
    """

    def classify(self, clauses: list[DetectedClause]) -> list[RiskAssessment]:
        """
        Classify each detected clause as HIGH, MEDIUM, or LOW risk.

        Parameters
        ----------
        clauses : list[DetectedClause]
            Output of ``ClauseDetector.detect()``.

        Returns
        -------
        list[RiskAssessment]
            One risk assessment per input clause.
        """
        assessments: list[RiskAssessment] = []

        for clause in clauses:
            rules = RISK_RULES.get(clause.clause_type, [])
            text = clause.matched_text

            matched_factors: list[tuple[RiskLevel, str]] = []

            for regex, level, description in rules:
                if regex.search(text):
                    matched_factors.append((level, description))

            if matched_factors:
                # Determine overall risk: highest matched level wins
                overall_level = max(
                    matched_factors,
                    key=lambda x: _RISK_PRIORITY[x[0]],
                )[0]
                risk_factors = [desc for _, desc in matched_factors]
                explanation = self._build_explanation(
                    clause.clause_type, overall_level, risk_factors
                )
            else:
                # No specific risk patterns matched → default to MEDIUM
                overall_level = RiskLevel.MEDIUM
                risk_factors = ["No specific risk indicators detected"]
                explanation = (
                    f"This {clause.clause_type.value} clause was detected "
                    f"but does not contain clearly favorable or unfavorable "
                    f"language patterns. Manual review is recommended."
                )

            assessments.append(
                RiskAssessment(
                    clause_type=clause.clause_type,
                    risk_level=overall_level,
                    explanation=explanation,
                    risk_factors=risk_factors,
                    element_index=clause.element_index,
                    page_number=clause.page_number,
                    section_heading=clause.section_heading,
                    matched_text=clause.matched_text,
                )
            )

        # Log summary
        high = sum(1 for a in assessments if a.risk_level == RiskLevel.HIGH)
        med = sum(1 for a in assessments if a.risk_level == RiskLevel.MEDIUM)
        low = sum(1 for a in assessments if a.risk_level == RiskLevel.LOW)
        logger.info("Risk breakdown: %d HIGH, %d MEDIUM, %d LOW", high, med, low)

        return assessments

    @staticmethod
    def _build_explanation(
        clause_type: ClauseType,
        risk_level: RiskLevel,
        factors: list[str],
    ) -> str:
        """Build a human-readable risk explanation."""


        clause_name = clause_type.value.replace("_", "-")

        intro = {
            RiskLevel.HIGH: (
                f"⚠️ HIGH RISK: This {clause_name} clause contains language "
                f"that may be unfavorable or one-sided."
            ),
            RiskLevel.MEDIUM: (
                f"⚡ MEDIUM RISK: This {clause_name} clause contains standard "
                f"but noteworthy terms that warrant review."
            ),
            RiskLevel.LOW: (
                f"✅ LOW RISK: This {clause_name} clause appears balanced "
                f"and uses favorable language."
            ),
        }

        parts = [intro[risk_level]]
        if factors:
            parts.append("Risk factors identified:")
            for f in factors:
                parts.append(f"  • {f}")

        return "\n".join(parts)
