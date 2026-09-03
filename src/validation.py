"""Validation Layer (Module 10).

Turns a SupervisorRun into the BriefingPacket a human reviews (D-06). Four
jobs, in the order they run:

  1. **Missing output** — every agent that was selected but produced nothing.
  2. **Citation resolution** (T-22) — every citation's chunk_id must resolve,
     either in the KB index or as a well-formed live-source id. This is the
     only place HG-2 is actually *verified*; the schema merely guarantees a
     citation was supplied.
  3. **Cross-agent consistency** — one genuinely deterministic check (see
     check_cross_agent). Semantic conflict detection is NOT implemented, and
     Module 10's doc says so plainly rather than faking it.
  4. **Confidence** (T-21) — the unweighted mean of three observable ratios.

On what "confidence" means here, because the name invites the wrong reading:
this is a **completeness/verifiability score**, not a calibrated probability
that the packet is correct. It answers "how much of what should be here is
present and checkable". Nothing in this project has been calibrated against
ground truth yet — that is Module 13. The field is called overall_confidence
because Module 2's schema names it that.

Nothing here is a determination. Every flag asks a human to look; the packet
is not final until one has ruled on it (HG-4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from src.kb.store import KnowledgeBase
from src.schemas import (
    AgentName,
    BriefingPacket,
    Citation,
    FlagType,
    SupervisorRun,
    ValidationFlag,
)

# ASSUMPTION A-01: the cutoff that triggers a human-review flag. Arbitrary
# until calibrated against Module 13's golden set — not a validated
# operating point, and must never be quoted as one.
CONFIDENCE_THRESHOLD = 0.6

# Live-source id schemes an agent may legitimately cite without the KB
# (T-22). Anything matching none of these resolves nowhere and is flagged.
_ID_SCHEMES = (
    re.compile(r"^ctgov:NCT\d{8}$"),
    re.compile(r"^pubmed:\d+$"),
    re.compile(r"^faers:[a-z0-9_\-]+$", re.IGNORECASE),
    re.compile(r"^guidance:[A-Za-z0-9_]+(:[\d.]+)?$"),
)


# --------------------------------------------------------------------------
# Collecting citations
# --------------------------------------------------------------------------


def collect_citations(run: SupervisorRun) -> list[tuple[AgentName, Citation]]:
    """Every citation in the run, tagged with the agent that made it."""
    found: list[tuple[AgentName, Citation]] = []

    if run.protocol is not None:
        for name in (
            "phase", "population", "primary_endpoint",
            "sample_size", "inclusion_criteria", "exclusion_criteria",
        ):
            extracted = getattr(run.protocol, name)
            if extracted.citation is not None:
                found.append((AgentName.PROTOCOL, extracted.citation))

    if run.evidence is not None:
        for claim in run.evidence.claims:
            found.extend((AgentName.EVIDENCE, c) for c in claim.citations)

    found.extend((AgentName.REGULATORY, f.citation) for f in run.regulatory)

    if run.safety is not None:
        found.extend((AgentName.SAFETY, a.citation) for a in run.safety.associations)

    return found


def resolves(citation: Citation, kb: KnowledgeBase | None) -> bool:
    """True if this citation points at something real (T-22).

    KB resolution is the strong form — the chunk is there and its text can be
    re-read. Scheme matching is the weaker form for live-fetched sources: it
    proves the id is well-formed and names a real source type, not that the
    document is still retrievable today.
    """
    if kb is not None and kb.get_chunk(citation.chunk_id) is not None:
        return True
    return any(pattern.match(citation.chunk_id) for pattern in _ID_SCHEMES)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_missing_output(run: SupervisorRun) -> list[ValidationFlag]:
    """An agent that was selected but produced nothing.

    Covers both a recorded failure and the quieter case of an agent that
    returned successfully with an empty result.
    """
    produced = set(run.agents_with_output)
    flags: list[ValidationFlag] = []

    for failure in run.failures:
        flags.append(
            ValidationFlag(
                flag_type=FlagType.MISSING_AGENT_OUTPUT,
                detail=f"{failure.agent.value} failed: {failure.error_type} — {failure.detail}",
                agents_involved=[failure.agent],
            )
        )

    failed = {f.agent for f in run.failures}
    for agent in run.routing.selected:
        if agent not in produced and agent not in failed:
            flags.append(
                ValidationFlag(
                    flag_type=FlagType.MISSING_AGENT_OUTPUT,
                    detail=(
                        f"{agent.value} was selected and did not fail, but returned no output — "
                        "an empty result is not the same as a finding"
                    ),
                    agents_involved=[agent],
                )
            )
    return flags


def check_citations(run: SupervisorRun, kb: KnowledgeBase | None) -> list[ValidationFlag]:
    """HG-2, actually verified rather than structurally assumed."""
    flags: list[ValidationFlag] = []
    for agent, citation in collect_citations(run):
        if not resolves(citation, kb):
            flags.append(
                ValidationFlag(
                    flag_type=FlagType.UNRESOLVED_CITATION,
                    detail=(
                        f"{agent.value} cited chunk_id {citation.chunk_id!r}, which resolves "
                        "neither in the knowledge base nor as a known source id"
                    ),
                    agents_involved=[agent],
                )
            )
    return flags


def check_cross_agent(run: SupervisorRun) -> list[ValidationFlag]:
    """Cross-agent consistency.

    Exactly one check is implemented, because exactly one is deterministically
    checkable with what the agents return: **the drug screened for safety
    should appear somewhere in the paired trial's extracted protocol fields**
    (ASSUMPTION A-08). It catches a real operator error — screening the wrong
    drug for the trial under review — and it fires correctly on this
    project's own demo pairing (a COVID antiviral trial screened against an
    oncology drug).

    Deliberately NOT implemented: semantic disagreement between agents (e.g.
    Regulatory flagging a field as absent that Protocol extracted). Detecting
    that reliably needs an LLM judge whose verdict could not itself be
    verified, which is the kind of unverifiable model claim this project
    rejects everywhere else. Module 10's doc states this gap plainly.
    """
    if run.safety is None or run.protocol is None:
        return []

    drug = run.safety.drug.strip().lower()
    if not drug:
        return []

    values = " ".join(
        extracted.value or ""
        for name in (
            "phase", "population", "primary_endpoint",
            "sample_size", "inclusion_criteria", "exclusion_criteria",
        )
        for extracted in [getattr(run.protocol, name)]
    ).lower()

    if drug in values:
        return []

    return [
        ValidationFlag(
            flag_type=FlagType.CROSS_AGENT_CONFLICT,
            detail=(
                f"safety screened {run.safety.drug!r}, but that drug does not appear in any "
                f"field extracted from {run.protocol.nct_id} — check the screen is for the "
                "right trial (advisory: a trial's registry fields may simply not name its drug)"
            ),
            agents_involved=[AgentName.SAFETY, AgentName.PROTOCOL],
        )
    ]


# --------------------------------------------------------------------------
# Confidence (T-21)
# --------------------------------------------------------------------------


@dataclass
class ConfidenceBreakdown:
    """The score and the parts it came from.

    Kept decomposable on purpose: a single number invites being trusted, and
    this one has never been calibrated (A-07). A reviewer should be able to
    see that a 0.55 was driven by one missing agent rather than by anything
    about the quality of the evidence.
    """

    components: dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        if not self.components:
            return 0.0
        return round(sum(self.components.values()) / len(self.components), 3)

    def explain(self) -> str:
        if not self.components:
            return "no components — nothing ran, so nothing is verified"
        parts = ", ".join(f"{name}={value:.2f}" for name, value in sorted(self.components.items()))
        # ASCII arrow deliberately: a U+2192 here crashed on the Windows
        # console (cp1252) the first time this was printed, and this string
        # is headed for the Module 11 UI and Module 13's eval output.
        return f"{parts} -> mean {self.score:.2f}"


def confidence_breakdown(run: SupervisorRun, kb: KnowledgeBase | None = None) -> ConfidenceBreakdown:
    """Only ratios of things actually counted — no invented constants (T-21).

    A component that doesn't apply to this run is left out rather than
    scored zero: a run with no protocol agent shouldn't be penalised for the
    absence of protocol completeness.
    """
    breakdown = ConfidenceBreakdown()

    selected = run.routing.selected
    if selected:
        breakdown.components["agent_coverage"] = len(run.agents_with_output) / len(selected)

    citations = collect_citations(run)
    if citations:
        resolved = sum(1 for _, c in citations if resolves(c, kb))
        breakdown.components["citation_integrity"] = resolved / len(citations)

    if run.protocol is not None:
        breakdown.components["protocol_completeness"] = run.protocol.completeness()

    return breakdown


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def validate(run: SupervisorRun, kb: KnowledgeBase | None = None) -> BriefingPacket:
    """Run every check and assemble the packet a human reviews.

    The packet comes back with human_decision=PENDING and is_final=False.
    Nothing in this layer can make a packet final — only a recorded human
    decision can (HG-4), which is Module 11's job.
    """
    flags = check_missing_output(run) + check_citations(run, kb) + check_cross_agent(run)
    breakdown = confidence_breakdown(run, kb)

    if breakdown.score < CONFIDENCE_THRESHOLD:
        flags.append(
            ValidationFlag(
                flag_type=FlagType.LOW_CONFIDENCE,
                detail=(
                    f"score {breakdown.score:.2f} is below the {CONFIDENCE_THRESHOLD} "
                    f"review threshold ({breakdown.explain()}). This is a completeness "
                    "score, not a probability that the content is correct; the threshold "
                    "itself is an uncalibrated assumption (A-01)."
                ),
                agents_involved=run.routing.selected or list(AgentName),
            )
        )

    return BriefingPacket(
        run_id=run.run_id,
        query=run.query,
        created_at=run.created_at or datetime.now(),
        protocol=run.protocol,
        evidence=run.evidence,
        regulatory=run.regulatory,
        safety=run.safety,
        validation_flags=flags,
        overall_confidence=breakdown.score,
    )
