"""Proof that the Module 1 hard gates are enforced by the schemas.

Each test names the gate it covers. If one of these starts passing when it
should fail, a governance control has been lost.
"""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from src.schemas import (
    BriefingPacket,
    Citation,
    Claim,
    EvidenceStrength,
    EvidenceSynthesis,
    ExtractedField,
    FieldStatus,
    ReportedAssociation,
    ReviewDecision,
    SourceType,
)


def _citation() -> Citation:
    return Citation(
        chunk_id="chunk_0042",
        source_type=SourceType.PROTOCOL,
        source_name="NCT04280705",
        retrieved_date=date(2026, 9, 3),
        quoted_text="This is a Phase 3, randomized, double-blind trial.",
    )


# --- HG-1: no fabricated fields -------------------------------------------


def test_hg1_absent_field_cannot_carry_a_value():
    with pytest.raises(ValidationError, match="must not carry a value"):
        ExtractedField(value="Phase 3", status=FieldStatus.NOT_SPECIFIED)


def test_hg1_absent_helper_is_valid():
    field = ExtractedField.absent()
    assert field.value is None
    assert field.status is FieldStatus.NOT_SPECIFIED


def test_hg1_extracted_field_requires_a_citation():
    with pytest.raises(ValidationError, match="requires a citation"):
        ExtractedField(value="Phase 3", status=FieldStatus.EXTRACTED)


# --- HG-2: no uncited claims ----------------------------------------------


def test_hg2_claim_without_citation_is_rejected():
    with pytest.raises(ValidationError):
        Claim(text="PFS is an accepted oncology endpoint.", citations=[])


def test_hg2_evidence_gap_must_be_declared_not_filled():
    # An agent that found nothing must say so, and carry no claims.
    empty = EvidenceSynthesis(
        question="What is the evidence for this endpoint?",
        claims=[],
        strength=EvidenceStrength.NONE_FOUND,
    )
    assert empty.strength is EvidenceStrength.NONE_FOUND

    with pytest.raises(ValidationError, match="cannot carry claims"):
        EvidenceSynthesis(
            question="q",
            claims=[Claim(text="Some claim.", citations=[_citation()])],
            strength=EvidenceStrength.NONE_FOUND,
        )


# --- HG-3: no causal language on FAERS output -----------------------------


@pytest.mark.parametrize(
    "narrative",
    [
        "The drug causes hepatotoxicity in a subset of patients.",
        "This proves a link between the product and the event.",
        "Elevated ALT results in treatment discontinuation.",
    ],
)
def test_hg3_causal_language_is_rejected(narrative):
    with pytest.raises(ValidationError, match="causal language"):
        ReportedAssociation(
            adverse_event="Hepatotoxicity",
            report_count=143,
            known_label_risk=True,
            narrative=narrative,
            citation=_citation(),
        )


def test_hg3_association_language_is_accepted():
    ok = ReportedAssociation(
        adverse_event="Hepatotoxicity",
        report_count=143,
        known_label_risk=True,
        narrative=(
            "143 spontaneous reports describe hepatotoxicity co-reported with "
            "this product. Reported association only; FAERS does not establish "
            "causality and is subject to reporting bias."
        ),
        citation=_citation(),
    )
    assert ok.report_count == 143


# --- HG-4: nothing is final without a human decision ----------------------


def _packet(**overrides) -> dict:
    base = dict(
        run_id="run_001",
        query="Review NCT04280705 for go/no-go.",
        created_at=datetime(2026, 9, 3, 10, 0, 0),
        overall_confidence=0.72,
    )
    base.update(overrides)
    return base


def test_hg4_new_packet_is_not_final():
    packet = BriefingPacket(**_packet())
    assert packet.human_decision is ReviewDecision.PENDING
    assert packet.is_final is False


def test_hg4_approval_makes_it_final_and_must_be_timestamped():
    approved = BriefingPacket(
        **_packet(
            human_decision=ReviewDecision.APPROVED,
            decided_at=datetime(2026, 9, 3, 10, 15, 0),
        )
    )
    assert approved.is_final is True

    with pytest.raises(ValidationError, match="must carry decided_at"):
        BriefingPacket(**_packet(human_decision=ReviewDecision.APPROVED))


def test_hg4_rejection_requires_a_reason():
    with pytest.raises(ValidationError, match="reviewer_note"):
        BriefingPacket(
            **_packet(
                human_decision=ReviewDecision.REJECTED,
                decided_at=datetime(2026, 9, 3, 10, 15, 0),
            )
        )
