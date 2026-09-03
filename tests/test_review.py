"""Review decision logic tests — offline, no Streamlit involved. All of
src/review.py is framework-independent, so the approve/edit/reject rules are
tested directly against BriefingPacket."""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from src.review import apply_edits, decide, editable_protocol_fields
from src.schemas import (
    BriefingPacket,
    Citation,
    ExtractedField,
    FieldStatus,
    ProtocolExtraction,
    ReviewDecision,
    SourceType,
)


def _citation() -> Citation:
    return Citation(
        chunk_id="ctgov:NCT04280705",
        source_type=SourceType.PROTOCOL,
        source_name="NCT04280705",
        retrieved_date=date(2026, 9, 3),
        quoted_text="Phase 3, randomized",
    )


def _protocol() -> ProtocolExtraction:
    return ProtocolExtraction(
        nct_id="NCT04280705",
        phase=ExtractedField(value="PHASE3", status=FieldStatus.EXTRACTED, citation=_citation()),
        population=ExtractedField.absent(),
        primary_endpoint=ExtractedField(value="Time to Recovery", status=FieldStatus.EXTRACTED, citation=_citation()),
        sample_size=ExtractedField.absent(),
        inclusion_criteria=ExtractedField.absent(),
        exclusion_criteria=ExtractedField.absent(),
    )


def _packet(**overrides) -> BriefingPacket:
    base = dict(
        run_id="run-test",
        query="a query",
        created_at=datetime(2026, 9, 3, 10, 0, 0),
        protocol=_protocol(),
        overall_confidence=0.8,
    )
    base.update(overrides)
    return BriefingPacket(**base)


# --------------------------------------------------------------------------
# What's editable
# --------------------------------------------------------------------------


def test_only_extracted_fields_are_editable():
    editable = editable_protocol_fields(_packet())
    assert set(editable) == {"phase", "primary_endpoint"}
    assert "population" not in editable  # not_specified — no citation to preserve


def test_no_protocol_means_nothing_editable():
    assert editable_protocol_fields(_packet(protocol=None)) == []


# --------------------------------------------------------------------------
# apply_edits
# --------------------------------------------------------------------------


def test_edit_changes_the_value_and_keeps_the_original_citation():
    packet = _packet()
    original_citation = packet.protocol.phase.citation

    edited = apply_edits(packet, {"phase": "Phase 3b (amended)"})

    assert edited.protocol.phase.value == "Phase 3b (amended)"
    assert edited.protocol.phase.citation == original_citation
    assert edited.protocol.phase.status is FieldStatus.EXTRACTED


def test_cannot_edit_a_not_specified_field():
    """The core design decision: no fabricating a citation to let an edit
    through. This is HG-1 enforced at the review layer, not worked around."""
    with pytest.raises(KeyError, match="never extracted with a citation"):
        apply_edits(_packet(), {"population": "adults 18-75"})


def test_editing_an_unknown_field_name_raises():
    with pytest.raises(KeyError):
        apply_edits(_packet(), {"not_a_real_field": "x"})


def test_no_edits_is_a_no_op():
    packet = _packet()
    assert apply_edits(packet, {}) is packet


def test_editing_with_no_protocol_on_the_packet_raises():
    with pytest.raises(ValueError, match="no protocol extraction"):
        apply_edits(_packet(protocol=None), {"phase": "x"})


def test_edited_extracted_field_is_still_internally_valid():
    """apply_edits reconstructs through the real constructor, not
    model_copy(update=...) — model_copy skips validation entirely, which
    would silently let a malformed ExtractedField through."""
    edited = apply_edits(_packet(), {"phase": "Phase 3b"})
    # Would raise on construction if HG-1 were violated (e.g. value with no citation).
    ExtractedField.model_validate(edited.protocol.phase.model_dump())


# --------------------------------------------------------------------------
# decide()
# --------------------------------------------------------------------------


def test_approve_finalizes_the_packet():
    packet = decide(_packet(), ReviewDecision.APPROVED, reviewer_note=None)

    assert packet.human_decision is ReviewDecision.APPROVED
    assert packet.is_final is True
    assert packet.decided_at is not None


def test_reject_without_a_note_raises_the_real_validator_error():
    """decide() must propagate ValidationError, not swallow it, so the UI
    can show the actual HG-4 message."""
    with pytest.raises(ValidationError, match="reviewer_note"):
        decide(_packet(), ReviewDecision.REJECTED, reviewer_note=None)


def test_reject_with_a_note_finalizes_the_packet():
    packet = decide(_packet(), ReviewDecision.REJECTED, reviewer_note="Sample size not credible.")
    assert packet.human_decision is ReviewDecision.REJECTED
    assert packet.is_final is True


def test_edit_decision_without_a_note_raises():
    with pytest.raises(ValidationError, match="reviewer_note"):
        decide(_packet(), ReviewDecision.EDITED, reviewer_note=None, field_edits={"phase": "Phase 3b"})


def test_edit_decision_applies_the_edit_and_finalizes():
    packet = decide(
        _packet(),
        ReviewDecision.EDITED,
        reviewer_note="Corrected phase per the amendment.",
        field_edits={"phase": "Phase 3b (amended)"},
    )

    assert packet.human_decision is ReviewDecision.EDITED
    assert packet.protocol.phase.value == "Phase 3b (amended)"
    assert packet.is_final is True
    assert packet.reviewer_note == "Corrected phase per the amendment."


def test_pending_is_not_an_acceptable_decision():
    with pytest.raises(ValueError, match="not a decision"):
        decide(_packet(), ReviewDecision.PENDING, reviewer_note=None)


def test_decide_does_not_mutate_the_original_packet():
    original = _packet()
    decide(original, ReviewDecision.APPROVED, reviewer_note=None)

    assert original.human_decision is ReviewDecision.PENDING
    assert original.is_final is False
