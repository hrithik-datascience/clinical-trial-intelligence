"""Human review decisions (Module 11) — framework-independent so the
approve/edit/reject logic is fully testable without a browser or Streamlit.

**What "edit" means here, deliberately:** only a protocol field that already
carries a citation (status=EXTRACTED) can have its value changed through
this path. `ExtractedField`'s own validator (Module 2, HG-1) requires a
citation for any value with status=EXTRACTED, and this layer must not
fabricate one just to let an edit through. A NOT_SPECIFIED field cannot be
turned into a value here — a reviewer's domain knowledge belongs in
`reviewer_note`, layered on top of the structured extraction, not merged
into it as if the source document had actually said it. The original
citation is kept unchanged after an edit: it remains honest provenance for
what the agent found, even where a human has since overridden the value.

Reconstruction always goes through the real Pydantic constructor
(`BriefingPacket(**data)`), never `model_copy(update=...)` — model_copy is
documented as skipping validation, which would silently defeat HG-4's
reviewer_note requirement (see test_review.py for the regression this
guards against).
"""

from __future__ import annotations

from datetime import datetime

from src import audit
from src.schemas import AuditEventType, BriefingPacket, FieldStatus, ReviewDecision

PROTOCOL_FIELD_NAMES = (
    "phase", "population", "primary_endpoint",
    "sample_size", "inclusion_criteria", "exclusion_criteria",
)


def editable_protocol_fields(packet: BriefingPacket) -> list[str]:
    """Field names that can be edited: only ones already carrying a
    citation. A NOT_SPECIFIED field has none to preserve, so editing it here
    would mean fabricating one."""
    if packet.protocol is None:
        return []
    return [
        name for name in PROTOCOL_FIELD_NAMES
        if getattr(packet.protocol, name).status is FieldStatus.EXTRACTED
    ]


def apply_edits(packet: BriefingPacket, field_edits: dict[str, str]) -> BriefingPacket:
    """Return a new packet with the named protocol field values overridden.

    Raises KeyError for a field name that isn't editable (see
    editable_protocol_fields) rather than silently ignoring it, and
    ValueError if the packet has no protocol extraction at all.
    """
    if not field_edits:
        return packet
    if packet.protocol is None:
        raise ValueError("no protocol extraction on this packet to edit")

    editable = set(editable_protocol_fields(packet))
    unknown = set(field_edits) - editable
    if unknown:
        raise KeyError(f"not editable (never extracted with a citation): {sorted(unknown)}")

    protocol_data = packet.protocol.model_dump()
    for name, new_value in field_edits.items():
        protocol_data[name]["value"] = new_value

    packet_data = packet.model_dump()
    packet_data["protocol"] = protocol_data
    return BriefingPacket(**packet_data)  # full re-validation, incl. ExtractedField's HG-1 check


def decide(
    packet: BriefingPacket,
    decision: ReviewDecision,
    reviewer_note: str | None,
    field_edits: dict[str, str] | None = None,
) -> BriefingPacket:
    """Apply a human decision, returning the final packet.

    Raises pydantic.ValidationError (propagated, not caught here) if the
    result is structurally invalid — e.g. REJECTED or EDITED without a
    reviewer_note — so the caller can show the real validator message
    instead of a generic failure.
    """
    if decision is ReviewDecision.PENDING:
        raise ValueError("PENDING is not a decision — it is the absence of one")

    working = apply_edits(packet, field_edits or {})
    data = working.model_dump()
    data.update(human_decision=decision, reviewer_note=reviewer_note, decided_at=datetime.now())
    final_packet = BriefingPacket(**data)

    # Module 12: the audit trail's HUMAN_DECISION record. Logged only after
    # BriefingPacket(**data) has already succeeded, so an un-explained
    # rejection or edit (HG-4) never reaches the log at all -- it never
    # became a real decision in the first place.
    audit.log_event(final_packet.run_id, AuditEventType.HUMAN_DECISION, final_packet)
    return final_packet
