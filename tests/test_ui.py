"""Human Review Interface tests — Streamlit's AppTest, which runs the real
script and simulates real widget interaction without a browser.

Every test except the one marked live pre-seeds st.session_state with a
synthetic packet, bypassing the Supervisor entirely — this is business logic
already covered by test_supervisor.py and test_validation.py; what this file
verifies is that the UI renders it and wires decisions correctly. The one
live test is the deliberate, single real pass through the actual intake form
this project's "use the API consciously" instruction calls for.
"""

import os
from datetime import date, datetime

import pytest
from dotenv import load_dotenv
from streamlit.testing.v1 import AppTest

# Other live test files trigger this as a side effect of importing the agent
# module under test at the top of the file (which imports src.llm, which
# calls load_dotenv()). This file only imports src.schemas, which doesn't —
# so the skipif below saw an empty environment and false-skipped on the
# first run. Explicit here rather than relying on import order elsewhere.
load_dotenv()

from src.schemas import (
    AgentName,
    BriefingPacket,
    Citation,
    ExtractedField,
    FieldStatus,
    ProtocolExtraction,
    RoutingDecision,
    RoutingMethod,
    SourceType,
    SupervisorRun,
)

APP_PATH = str((__import__("pathlib").Path(__file__).parent.parent / "src" / "ui" / "app.py").resolve())
TIMEOUT = 60


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


def _run() -> SupervisorRun:
    return SupervisorRun(
        run_id="run-test",
        query="review the trial",
        created_at=datetime(2026, 9, 3, 12, 0, 0),
        routing=RoutingDecision(selected=[AgentName.PROTOCOL], method=RoutingMethod.EXPLICIT, rationale="test"),
        protocol=_protocol(),
        evidence=None,
        regulatory=[],
        safety=None,
        failures=[],
        elapsed_seconds=1.23,
    )


def _packet(**overrides) -> BriefingPacket:
    base = dict(
        run_id="run-test",
        query="review the trial",
        created_at=datetime(2026, 9, 3, 12, 0, 0),
        protocol=_protocol(),
        overall_confidence=0.9,
    )
    base.update(overrides)
    return BriefingPacket(**base)


def _seeded_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.session_state["supervisor_run"] = _run()
    at.session_state["packet"] = _packet()
    return at


def _submit_button(at: AppTest):
    return next(b for b in at.button if b.label == "Submit decision")


def _reviewer_note_widget(at: AppTest):
    return next(t for t in at.text_area if t.label.startswith("Reviewer note"))


# --------------------------------------------------------------------------
# Boot and the sys.path fix
# --------------------------------------------------------------------------


def test_app_boots_without_exceptions():
    """Regression for the ModuleNotFoundError found running this under both
    `streamlit run` and AppTest: sys.path is scoped to src/ui/, not the
    project root, by Streamlit's own script runner."""
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception


def test_disclaimer_is_always_shown():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    assert any("NOT CLINICALLY VALIDATED" in w.value for w in at.warning)


def test_intake_form_present_before_any_run():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    assert "New review" in [s.value for s in at.subheader]
    assert not at.session_state["packet"]


def test_empty_query_shows_an_error_not_a_crash():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    at.button[0].click()  # "Run analysis" with an empty query
    at.run()

    assert not at.exception
    assert any("request is required" in e.value for e in at.error)
    assert at.session_state["packet"] is None


# --------------------------------------------------------------------------
# Rendering a packet (pre-seeded, no live call)
# --------------------------------------------------------------------------


def test_packet_sections_render():
    at = _seeded_app()
    at.run()

    subheaders = [s.value for s in at.subheader]
    assert "Routing" in subheaders
    assert "Confidence" in subheaders
    assert "Protocol — NCT04280705" in subheaders
    assert "Human decision" in subheaders


def test_extracted_field_shows_value_and_citation_absent_field_does_not():
    at = _seeded_app()
    at.run()

    markdown_text = " ".join(m.value for m in at.markdown)
    assert "**Phase:** PHASE3" in markdown_text
    assert "not specified in the registry record" in markdown_text  # population


def test_confidence_is_labelled_as_not_a_probability():
    """The T-21 caveat must actually reach the reviewer, not just the docs."""
    at = _seeded_app()
    at.run()
    captions = " ".join(c.value for c in at.caption)
    assert "NOT a probability of correctness" in captions


def test_no_validation_flags_shows_success_not_a_warning():
    at = _seeded_app()
    at.run()
    assert any("No validation flags" in s.value for s in at.success)


# --------------------------------------------------------------------------
# Decisions (T-approve/edit/reject via the real widgets)
# --------------------------------------------------------------------------


def test_reject_without_a_note_shows_the_real_validator_error():
    at = _seeded_app()
    at.run()
    at.radio[0].set_value("Reject")
    at.run()
    _submit_button(at).click()
    at.run()

    assert any("must carry a reviewer_note" in e.value for e in at.error)
    assert at.session_state["decided_packet"] is None


def test_reject_with_a_note_finalizes_and_shows_the_decision():
    at = _seeded_app()
    at.run()
    at.radio[0].set_value("Reject")
    at.run()
    _reviewer_note_widget(at).set_value("Sample size not credible for this indication.")
    at.run()
    _submit_button(at).click()
    at.run()

    decided = at.session_state["decided_packet"]
    assert decided is not None
    assert decided.human_decision.value == "rejected"
    assert decided.is_final is True
    assert any("REJECTED" in s.value for s in at.success)


def test_approve_needs_no_note():
    at = _seeded_app()
    at.run()
    # "Approve" is the default radio selection — no change needed.
    _submit_button(at).click()
    at.run()

    decided = at.session_state["decided_packet"]
    assert decided is not None
    assert decided.human_decision.value == "approved"


def test_edit_without_a_note_is_rejected_even_with_a_valid_field_change():
    at = _seeded_app()
    at.run()
    at.radio[0].set_value("Edit")
    at.run()

    phase_input = next(t for t in at.text_input if t.label == "Phase")
    phase_input.set_value("Phase 3b (amended)")
    at.run()
    _submit_button(at).click()
    at.run()

    assert any("must carry a reviewer_note" in e.value for e in at.error)
    assert at.session_state["decided_packet"] is None


def test_edit_only_offers_fields_that_carry_a_citation():
    """population is NOT_SPECIFIED — HG-1 means it must not be editable here."""
    at = _seeded_app()
    at.run()
    at.radio[0].set_value("Edit")
    at.run()

    editable_labels = {t.label for t in at.text_input if t.label not in ("NCT id (optional — or include it in the request above)", "Drug (optional — never inferred from free text, HG-1)")}
    assert "Phase" in editable_labels
    assert "Population" not in editable_labels


def test_edit_with_a_note_applies_the_change_and_finalizes():
    at = _seeded_app()
    at.run()
    at.radio[0].set_value("Edit")
    at.run()

    phase_input = next(t for t in at.text_input if t.label == "Phase")
    phase_input.set_value("Phase 3b (amended)")
    at.run()
    _reviewer_note_widget(at).set_value("Corrected phase per the amendment I reviewed.")
    at.run()
    _submit_button(at).click()
    at.run()

    decided = at.session_state["decided_packet"]
    assert decided is not None
    assert decided.human_decision.value == "edited"
    assert decided.protocol.phase.value == "Phase 3b (amended)"
    assert decided.is_final is True


def test_decided_screen_offers_a_fresh_review():
    at = _seeded_app()
    at.run()
    _submit_button(at).click()
    at.run()

    assert any(b.label == "Start a new review" for b in at.button)
    start_over = next(b for b in at.button if b.label == "Start a new review")
    start_over.click()
    at.run()

    assert at.session_state["decided_packet"] is None
    assert "New review" in [s.value for s in at.subheader]


# --------------------------------------------------------------------------
# One live, end-to-end pass through the real intake form
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="no API key set — skipping the one live UI pass",
)
def test_live_intake_run_and_approve():
    """The deliberate, single real pass: type a real request into the actual
    form, click Run, wait for the real Supervisor + Validation pipeline, and
    approve the result. Everything else in this file is pre-seeded state to
    avoid spending API calls on UI plumbing that doesn't need them."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()

    at.text_area[0].set_value("Review NCT04280705 against statistical guidance.")
    at.run()
    run_button = next(b for b in at.button if b.label == "Run analysis")
    run_button.click()
    at.run(timeout=120)

    assert not at.exception
    assert at.session_state["packet"] is not None
    assert "Protocol — NCT04280705" in [s.value for s in at.subheader]

    _submit_button(at).click()
    at.run()

    decided = at.session_state["decided_packet"]
    assert decided is not None
    assert decided.human_decision.value == "approved"
    assert decided.is_final is True
