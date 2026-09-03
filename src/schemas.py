"""Pydantic schemas for the Clinical Trial Intelligence System.

Design principle: the hard gates from Module 1 are enforced *here*, in the type
system, rather than by a downstream lint pass. A fabricated field or an uncited
claim should be impossible to construct, not merely detectable after the fact.

    HG-1  abstention        -> ExtractedField validator
    HG-2  every claim cited -> Claim.citations has min_length=1
    HG-3  no causal language -> ReportedAssociation validator
    HG-4  human gate         -> BriefingPacket.is_final

NOT CLINICALLY VALIDATED. Public data only.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------
# Shared vocabulary
# --------------------------------------------------------------------------


class SourceType(str, Enum):
    """Metadata tag applied at ingestion so agents can filter one shared index."""

    PROTOCOL = "protocol"
    GUIDANCE = "guidance"
    LITERATURE = "literature"
    FAERS = "faers"
    SYNTHETIC = "synthetic"  # evaluation edge cases only, never a real finding


class AgentName(str, Enum):
    PROTOCOL = "protocol"
    EVIDENCE = "evidence"
    REGULATORY = "regulatory"
    SAFETY = "safety"


class FieldStatus(str, Enum):
    """Why a field has (or does not have) a value."""

    EXTRACTED = "extracted"
    NOT_SPECIFIED = "not_specified"  # genuinely absent from the source


class ReviewDecision(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"


class Severity(str, Enum):
    """Deliberately has no 'violation' or 'non-compliant' level.

    Agents flag points for a regulatory professional to review; they do not make
    regulatory determinations.
    """

    INFO = "info"
    FLAG = "flag"


class EvidenceStrength(str, Enum):
    SUPPORTED = "supported"
    LIMITED = "limited"
    CONFLICTING = "conflicting"
    NONE_FOUND = "none_found"


# --------------------------------------------------------------------------
# Citations — HG-2
# --------------------------------------------------------------------------


class Citation(BaseModel):
    """A pointer back to the source a claim came from.

    chunk_id resolves either against the KB index (Regulatory, which
    retrieves from it) or as a live-source document id — "ctgov:NCT04280705",
    "pubmed:32214230", "faers:pembrolizumab" — for the agents that fetch
    directly. The Validation layer (Module 10) checks resolution both ways
    (T-22); the schema only guarantees an id was supplied.

    This docstring originally promised the KB index for *every* citation.
    That predated Module 4 settling that only Regulatory retrieves from the
    KB, and was corrected in Module 10 rather than left to mislead.
    """

    chunk_id: str = Field(min_length=1)
    source_type: SourceType
    source_name: str = Field(min_length=1)  # e.g. "NCT04280705", "ICH E9"
    retrieved_date: date
    quoted_text: str = Field(min_length=1)


class Claim(BaseModel):
    """A substantive statement. Cannot exist without at least one citation."""

    text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)  # HG-2, enforced structurally


# --------------------------------------------------------------------------
# Abstention — HG-1
# --------------------------------------------------------------------------


class ExtractedField(BaseModel):
    """One protocol field, which may legitimately be absent.

    The validator is the point of the class: a value without a citation, or a
    NOT_SPECIFIED status carrying a value anyway, raises rather than silently
    entering the pipeline as a plausible-looking guess.
    """

    value: str | None = None
    status: FieldStatus
    citation: Citation | None = None

    @model_validator(mode="after")
    def _enforce_abstention(self) -> ExtractedField:
        if self.status is FieldStatus.EXTRACTED:
            if not self.value:
                raise ValueError("status=extracted requires a value")
            if self.citation is None:
                raise ValueError("status=extracted requires a citation (HG-2)")
        else:  # NOT_SPECIFIED
            if self.value is not None:
                raise ValueError(
                    "status=not_specified must not carry a value — "
                    "returning a guess for an absent field violates HG-1"
                )
        return self

    @classmethod
    def absent(cls) -> ExtractedField:
        """The only correct response when a field is missing from the source."""
        return cls(value=None, status=FieldStatus.NOT_SPECIFIED, citation=None)


# --------------------------------------------------------------------------
# Per-agent outputs
# --------------------------------------------------------------------------


class ProtocolExtraction(BaseModel):
    """Protocol agent. Structured fields pulled from a registry record."""

    nct_id: str = Field(pattern=r"^NCT\d{8}$")
    phase: ExtractedField
    population: ExtractedField
    primary_endpoint: ExtractedField
    sample_size: ExtractedField
    inclusion_criteria: ExtractedField
    exclusion_criteria: ExtractedField

    def completeness(self) -> float:
        """Share of fields actually grounded. Reported, never optimised toward —
        a low value is a fact about the source, not a failure of the agent."""
        fields = [
            self.phase,
            self.population,
            self.primary_endpoint,
            self.sample_size,
            self.inclusion_criteria,
            self.exclusion_criteria,
        ]
        grounded = sum(f.status is FieldStatus.EXTRACTED for f in fields)
        return grounded / len(fields)


class EvidenceSynthesis(BaseModel):
    """Evidence agent. Weak or missing evidence is reported, not smoothed over."""

    question: str = Field(min_length=1)
    claims: list[Claim]
    strength: EvidenceStrength

    @model_validator(mode="after")
    def _no_silent_gap(self) -> EvidenceSynthesis:
        if self.strength is EvidenceStrength.NONE_FOUND and self.claims:
            raise ValueError("strength=none_found cannot carry claims")
        if self.strength is not EvidenceStrength.NONE_FOUND and not self.claims:
            raise ValueError("claims are required unless strength=none_found")
        return self


class RegulatoryFinding(BaseModel):
    """Regulatory agent. Every finding names the clause it came from."""

    clause_id: str = Field(min_length=1)  # e.g. "ICH E9 5.2.3"
    finding: str = Field(min_length=1)
    severity: Severity
    citation: Citation


# Words that would assert causality from spontaneous-report data (HG-3).
_CAUSAL_PATTERN = re.compile(
    r"\b(causes?|caused|causing|causal|proves?|proven|"
    r"confirmed side effect|leads to|results in|due to the drug)\b",
    re.IGNORECASE,
)


class ReportedAssociation(BaseModel):
    """One FAERS-derived association.

    report_count is computed in Python from the API response — never asked of
    the model, which must not do arithmetic on safety counts.
    """

    adverse_event: str = Field(min_length=1)
    report_count: int = Field(ge=0)
    known_label_risk: bool  # already on the label vs. unexpected
    narrative: str = Field(min_length=1)
    citation: Citation

    @model_validator(mode="after")
    def _no_causal_language(self) -> ReportedAssociation:
        hit = _CAUSAL_PATTERN.search(self.narrative)
        if hit:
            raise ValueError(
                f"causal language {hit.group(0)!r} in FAERS-derived narrative "
                "(HG-3) — FAERS supports reported association only"
            )
        return self


class SafetyScreen(BaseModel):
    drug: str = Field(min_length=1)
    associations: list[ReportedAssociation]
    total_reports_reviewed: int = Field(ge=0)


# --------------------------------------------------------------------------
# Validation layer
# --------------------------------------------------------------------------


class FlagType(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    UNRESOLVED_CITATION = "unresolved_citation"
    CROSS_AGENT_CONFLICT = "cross_agent_conflict"
    MISSING_AGENT_OUTPUT = "missing_agent_output"


class ValidationFlag(BaseModel):
    flag_type: FlagType
    detail: str = Field(min_length=1)
    agents_involved: list[AgentName] = Field(min_length=1)


# --------------------------------------------------------------------------
# Supervisor (Module 9) — routing and aggregation, before validation
# --------------------------------------------------------------------------


class RoutingMethod(str, Enum):
    EXPLICIT = "explicit"  # caller named the agents; no model call made
    LLM_INTENT = "llm_intent"  # model narrowed within the runnable set
    CAPABILITY_ONLY = "capability_only"  # router unavailable/failed — ran everything runnable


class RoutingDecision(BaseModel):
    """Why these agents and not the others.

    `selected` is always a subset of what the deterministic capability gate
    found runnable (T-19) — the model can narrow this set, never extend it.
    """

    selected: list[AgentName] = Field(default_factory=list)
    not_runnable: dict[str, str] = Field(default_factory=dict)  # agent -> missing input
    method: RoutingMethod
    rationale: str = Field(min_length=1)


class AgentFailure(BaseModel):
    """One agent that was selected, ran, and did not produce output.

    Recorded rather than raised (T-20) so a single failure cannot destroy the
    other agents' results. This is not the same as swallowing it: a failure
    here is a visible field that the Validation layer turns into a
    MISSING_AGENT_OUTPUT flag.
    """

    agent: AgentName
    error_type: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class SupervisorRun(BaseModel):
    """What one orchestrated run produced.

    Deliberately NOT a BriefingPacket (D-06): there is no overall_confidence
    and no human_decision here, because scoring confidence is the Validation
    layer's job (Module 10) and inventing a number now would be a placeholder
    metric. A run with entries in `failures` is incomplete by definition —
    never read this as a finished result.
    """

    run_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    created_at: datetime
    routing: RoutingDecision

    protocol: ProtocolExtraction | None = None
    evidence: EvidenceSynthesis | None = None
    regulatory: list[RegulatoryFinding] = Field(default_factory=list)
    safety: SafetyScreen | None = None

    failures: list[AgentFailure] = Field(default_factory=list)
    elapsed_seconds: float = Field(ge=0.0)

    @property
    def agents_with_output(self) -> list[AgentName]:
        produced = []
        if self.protocol is not None:
            produced.append(AgentName.PROTOCOL)
        if self.evidence is not None:
            produced.append(AgentName.EVIDENCE)
        if self.regulatory:
            produced.append(AgentName.REGULATORY)
        if self.safety is not None:
            produced.append(AgentName.SAFETY)
        return produced


# --------------------------------------------------------------------------
# The packet the reviewer sees — HG-4
# --------------------------------------------------------------------------


class BriefingPacket(BaseModel):
    """Assembled output of one run.

    There is deliberately no `finalise()` method and no `final` boolean that
    code can set. The only route to is_final is a recorded human decision.
    """

    run_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    created_at: datetime

    protocol: ProtocolExtraction | None = None
    evidence: EvidenceSynthesis | None = None
    regulatory: list[RegulatoryFinding] = Field(default_factory=list)
    safety: SafetyScreen | None = None

    validation_flags: list[ValidationFlag] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)

    human_decision: ReviewDecision = ReviewDecision.PENDING
    reviewer_note: str | None = None
    decided_at: datetime | None = None

    @property
    def is_final(self) -> bool:
        """HG-4. A packet is final only once a human has ruled on it."""
        return self.human_decision is not ReviewDecision.PENDING

    @model_validator(mode="after")
    def _decision_is_recorded(self) -> BriefingPacket:
        decided = self.human_decision is not ReviewDecision.PENDING
        if decided and self.decided_at is None:
            raise ValueError("a human decision must carry decided_at (HG-5)")
        if not decided and self.decided_at is not None:
            raise ValueError("decided_at set while decision is still pending")
        if self.human_decision is ReviewDecision.REJECTED and not self.reviewer_note:
            raise ValueError("a rejection must carry a reviewer_note")
        return self


# --------------------------------------------------------------------------
# Ingestion (Module 3) — what the knowledge base consumes
# --------------------------------------------------------------------------


class RawDocument(BaseModel):
    """One retrieved document, before chunking.

    Every document carries its provenance tags so a single shared index can be
    filtered per agent (Section D.2, step 1).
    """

    doc_id: str = Field(min_length=1)
    source_type: SourceType
    source_name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    url: str = Field(min_length=1)
    retrieved_date: date
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """One indexed unit in the shared knowledge base (Module 4).

    chunk_id is the identifier Citation.chunk_id points back to, so the two
    must stay in the same format an agent used before the KB existed (e.g.
    "guidance:ICH_E9:4.5") — the Validation layer resolves one against the
    other.
    """

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    source_type: SourceType
    source_name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    retrieved_date: date
    metadata: dict = Field(default_factory=dict)


class ReactionCount(BaseModel):
    """One FAERS reaction term with FDA's own server-side count.

    The count comes from openFDA's `count` aggregation — neither this codebase
    nor the model ever tallies safety reports by hand.
    """

    term: str = Field(min_length=1)
    count: int = Field(ge=0)


class DrugLabel(BaseModel):
    """The real FDA-approved label text used to determine known_label_risk
    deterministically (Module 8) -- the model never makes this call."""

    drug: str = Field(min_length=1)
    brand_name: str = Field(min_length=1)
    generic_name: str = Field(min_length=1)
    reference_text: str = Field(min_length=1)  # boxed warning + warnings + adverse reactions
    retrieved_date: date


class FaersAggregation(BaseModel):
    drug: str = Field(min_length=1)
    total_reports: int = Field(ge=0)
    reactions: list[ReactionCount]
    last_updated: str  # openFDA meta.last_updated
    disclaimer: str  # openFDA ships this on every response; keep it attached
    retrieved_date: date
