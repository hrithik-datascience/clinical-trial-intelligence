"""Human Review Interface (Module 11) — Streamlit app.

Thin UI wiring only. Routing/orchestration is src/agents/supervisor.py,
checks and confidence are src/validation.py, and approve/edit/reject is
src/review.py — this file renders state and calls those, it does not
reimplement any of them.

Run:  venv/Scripts/python.exe -m streamlit run src/ui/app.py

A packet only ever reaches this page as PENDING. Nothing here can make one
final except a human explicitly submitting Approve, Edit, or Reject below
(HG-4) — the same rule the schema itself enforces, so nothing in this file
can bypass it even by accident.
"""

from __future__ import annotations

import sys
from pathlib import Path

# DEVIATION FROM PLAN, found running this under both `streamlit run` and
# AppTest: Streamlit executes this file with sys.path scoped to src/ui/, not
# the project root, so `from src...` fails with ModuleNotFoundError
# regardless of how the app is launched -- not a test-only problem. Every
# other module in this project runs through pytest, which inserts the
# project root automatically; this is the first file Streamlit's own runner
# executes directly, and it needed the same guarantee made explicit.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ruff: noqa: E402 -- every import below must come after the sys.path
# bootstrap above, not before it (that's the whole point of the bootstrap).
import streamlit as st
from pydantic import ValidationError

from src.agents.supervisor import SupervisorRequest
from src.agents.supervisor import run as run_supervisor
from src.kb.build import KB_DIR
from src.kb.store import KnowledgeBase
from src.review import PROTOCOL_FIELD_NAMES, decide, editable_protocol_fields
from src.schemas import BriefingPacket, FieldStatus, FlagType, ReviewDecision, Severity, SupervisorRun
from src.validation import confidence_breakdown, validate

st.set_page_config(page_title="Clinical Trial Intelligence — Review", layout="wide")

_DECISION_LABELS = {
    "Approve": ReviewDecision.APPROVED,
    "Edit": ReviewDecision.EDITED,
    "Reject": ReviewDecision.REJECTED,
}
_FLAG_ICON = {
    FlagType.LOW_CONFIDENCE: "🟡",
    FlagType.UNRESOLVED_CITATION: "🔴",
    FlagType.CROSS_AGENT_CONFLICT: "🟠",
    FlagType.MISSING_AGENT_OUTPUT: "🔴",
}


@st.cache_resource(show_spinner="Loading knowledge base...")
def _load_kb() -> KnowledgeBase | None:
    if not (KB_DIR / "faiss.index").exists():
        return None
    return KnowledgeBase.load(KB_DIR)


def _init_state() -> None:
    st.session_state.setdefault("supervisor_run", None)
    st.session_state.setdefault("packet", None)
    st.session_state.setdefault("decided_packet", None)


def _render_disclaimer() -> None:
    st.warning(
        "**NOT CLINICALLY VALIDATED. Public data only.** Every output below is a "
        "reported association or a flag for review — never a regulatory or clinical "
        "determination. Nothing is final until explicitly approved, edited, or "
        "rejected below.",
        icon="⚠️",
    )


def _render_intake_form(kb: KnowledgeBase | None) -> None:
    st.subheader("New review")
    if kb is None:
        st.error(f"No knowledge base at `{KB_DIR}`. Run `python -m src.kb.build` first.")

    with st.form("intake"):
        query = st.text_area(
            "Request",
            placeholder="e.g. Review NCT04280705 against statistical guidance and screen its safety signals.",
        )
        col1, col2 = st.columns(2)
        nct_id = col1.text_input("NCT id (optional — or include it in the request above)")
        drug = col2.text_input("Drug (optional — never inferred from free text, HG-1)")
        protocol_summary = st.text_area("Protocol summary (optional — skips live extraction)", height=80)
        submitted = st.form_submit_button("Run analysis", disabled=kb is None)

    if not submitted:
        return
    if not query.strip():
        st.error("A request is required.")
        return

    request = SupervisorRequest(
        query=query,
        nct_id=nct_id or None,
        drug=drug or None,
        protocol_summary=protocol_summary or None,
    )
    with st.spinner("Routing and running agents — this makes live API calls..."):
        supervisor_run = run_supervisor(request, kb=kb)
        packet = validate(supervisor_run, kb=kb)

    st.session_state["supervisor_run"] = supervisor_run
    st.session_state["packet"] = packet
    st.session_state["decided_packet"] = None
    st.rerun()


def _render_run_summary(supervisor_run: SupervisorRun) -> None:
    st.subheader("Routing")
    cols = st.columns(3)
    cols[0].metric("Selected agents", len(supervisor_run.routing.selected))
    cols[1].metric("Elapsed", f"{supervisor_run.elapsed_seconds:.1f}s")
    cols[2].metric("Method", supervisor_run.routing.method.value)
    st.caption(supervisor_run.routing.rationale)
    if supervisor_run.routing.not_runnable:
        st.caption(f"Not runnable: {supervisor_run.routing.not_runnable}")
    for failure in supervisor_run.failures:
        st.error(f"{failure.agent.value} failed: {failure.error_type} — {failure.detail}")


def _render_confidence(packet: BriefingPacket, supervisor_run: SupervisorRun, kb: KnowledgeBase | None) -> None:
    st.subheader("Confidence")
    st.metric("overall_confidence", f"{packet.overall_confidence:.2f}")
    st.caption(confidence_breakdown(supervisor_run, kb).explain())
    st.caption(
        "Completeness/verifiability score, NOT a probability of correctness (T-21). "
        "The 0.6 review threshold is an uncalibrated assumption (A-01)."
    )


def _render_flags(packet: BriefingPacket) -> None:
    if not packet.validation_flags:
        st.success("No validation flags.")
        return
    st.subheader(f"Validation flags ({len(packet.validation_flags)})")
    for flag in packet.validation_flags:
        agents = ", ".join(a.value for a in flag.agents_involved)
        st.markdown(f"{_FLAG_ICON.get(flag.flag_type, '⚠️')} **{flag.flag_type.value}** ({agents})")
        st.caption(flag.detail)


def _render_protocol(packet: BriefingPacket) -> None:
    if packet.protocol is None:
        return
    st.subheader(f"Protocol — {packet.protocol.nct_id}")
    st.caption(f"Completeness: {packet.protocol.completeness():.0%} of fields grounded")
    for name in PROTOCOL_FIELD_NAMES:
        field = getattr(packet.protocol, name)
        label = name.replace("_", " ").title()
        if field.status is FieldStatus.EXTRACTED:
            st.markdown(f"**{label}:** {field.value}")
            st.caption(f'{field.citation.source_name} — "{field.citation.quoted_text}"')
        else:
            st.markdown(f"**{label}:** _not specified in the registry record_")


def _render_regulatory(packet: BriefingPacket) -> None:
    if not packet.regulatory:
        return
    st.subheader(f"Regulatory findings ({len(packet.regulatory)})")
    for finding in packet.regulatory:
        icon = "🚩" if finding.severity is Severity.FLAG else "ℹ️"
        st.markdown(f"{icon} **{finding.clause_id}** — {finding.finding}")
        st.caption(f'"{finding.citation.quoted_text}"')


def _render_safety(packet: BriefingPacket) -> None:
    if packet.safety is None:
        return
    st.subheader(f"Safety — {packet.safety.drug}")
    st.caption(f"{packet.safety.total_reports_reviewed:,} total FAERS reports reviewed")
    for assoc in packet.safety.associations:
        tag = "Known/labeled" if assoc.known_label_risk else "Unexpected"
        st.markdown(f"**{assoc.adverse_event}** ({assoc.report_count:,} reports) — {tag}")
        st.caption(assoc.narrative)


def _render_evidence(packet: BriefingPacket) -> None:
    if packet.evidence is None:
        return
    st.subheader(f"Evidence — strength: {packet.evidence.strength.value}")
    st.caption(packet.evidence.question)
    for claim in packet.evidence.claims:
        st.markdown(f"- {claim.text}")
        for citation in claim.citations:
            st.caption(f'  {citation.source_name}: "{citation.quoted_text}"')


def _render_review(packet: BriefingPacket) -> None:
    st.subheader("Human decision")
    decision_label = st.radio("Decision", options=list(_DECISION_LABELS), horizontal=True)
    decision = _DECISION_LABELS[decision_label]

    field_edits: dict[str, str] = {}
    if decision is ReviewDecision.EDITED:
        editable = editable_protocol_fields(packet)
        if not editable:
            st.info(
                "No protocol field on this packet carries a citation to edit — "
                "put any correction in the note below."
            )
        else:
            st.caption(
                "Only fields already extracted with a citation can be edited here (HG-1) — "
                "a not-specified field's correction belongs in the reviewer note, not here, "
                "because editing it would mean fabricating a citation that was never real."
            )
            for name in editable:
                current = getattr(packet.protocol, name).value
                new_value = st.text_input(name.replace("_", " ").title(), value=current, key=f"edit_{name}")
                if new_value != current:
                    field_edits[name] = new_value

    note_required = decision in (ReviewDecision.REJECTED, ReviewDecision.EDITED)
    reviewer_note = st.text_area(f"Reviewer note{' (required)' if note_required else ' (optional)'}")

    if st.button("Submit decision", type="primary"):
        try:
            final_packet = decide(packet, decision, reviewer_note or None, field_edits)
        except (ValidationError, ValueError, KeyError) as exc:
            st.error(f"Could not record this decision: {exc}")
            return
        st.session_state["decided_packet"] = final_packet
        st.session_state["packet"] = None
        st.rerun()


def _render_decided(packet: BriefingPacket) -> None:
    st.subheader("Decision recorded")
    st.success(f"{packet.human_decision.value.upper()} — decided at {packet.decided_at}")
    if packet.reviewer_note:
        st.caption(f"Note: {packet.reviewer_note}")
    st.caption(f"is_final = {packet.is_final}")
    if st.button("Start a new review"):
        st.session_state["decided_packet"] = None
        st.rerun()


def main() -> None:
    _init_state()
    st.title("Clinical Trial Intelligence — Human Review")
    _render_disclaimer()

    kb = _load_kb()
    _render_intake_form(kb)

    if st.session_state["decided_packet"] is not None:
        _render_decided(st.session_state["decided_packet"])
        return

    packet = st.session_state["packet"]
    supervisor_run = st.session_state["supervisor_run"]
    if packet is None or supervisor_run is None:
        return

    _render_run_summary(supervisor_run)
    _render_confidence(packet, supervisor_run, kb)
    _render_flags(packet)
    _render_protocol(packet)
    _render_regulatory(packet)
    _render_safety(packet)
    _render_evidence(packet)
    _render_review(packet)


main()
