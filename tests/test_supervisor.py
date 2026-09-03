"""Supervisor tests — offline. The capability gate, summary rendering and
failure isolation are all pure Python, so they are tested without an API
call; routing uses the explicit `agents=` path, which skips the model
entirely by design."""

from datetime import date

import pytest

from src.agents.supervisor import (
    SupervisorRequest,
    capability_gate,
    route,
    run,
    summary_for_regulatory,
)
from src.schemas import (
    AgentName,
    Citation,
    ExtractedField,
    FieldStatus,
    ProtocolExtraction,
    RoutingMethod,
    SourceType,
)


def _citation() -> Citation:
    return Citation(
        chunk_id="ctgov:NCT04280705",
        source_type=SourceType.PROTOCOL,
        source_name="NCT04280705",
        retrieved_date=date(2026, 9, 3),
        quoted_text="Phase 3",
    )


def _extraction() -> ProtocolExtraction:
    """One extracted field and five absent — the realistic shape, since a
    registry record rarely states everything."""
    return ProtocolExtraction(
        nct_id="NCT04280705",
        phase=ExtractedField(value="PHASE3", status=FieldStatus.EXTRACTED, citation=_citation()),
        population=ExtractedField.absent(),
        primary_endpoint=ExtractedField.absent(),
        sample_size=ExtractedField.absent(),
        inclusion_criteria=ExtractedField.absent(),
        exclusion_criteria=ExtractedField.absent(),
    )


# --------------------------------------------------------------------------
# Capability gate
# --------------------------------------------------------------------------


def test_gate_blocks_safety_without_a_drug():
    runnable, blocked = capability_gate(SupervisorRequest(query="review this trial"))
    assert AgentName.SAFETY not in runnable
    assert "drug" in blocked[AgentName.SAFETY.value]


def test_gate_never_infers_a_drug_from_the_query_text():
    """HG-1 discipline: a drug name in prose is not a supplied input."""
    request = SupervisorRequest(query="what does FAERS say about pembrolizumab?")
    runnable, blocked = capability_gate(request)
    assert AgentName.SAFETY not in runnable
    assert AgentName.SAFETY.value in blocked


def test_gate_reads_an_nct_id_out_of_the_query():
    """An NCT id has an unambiguous format, so reading it from free text is
    safe in a way a drug name is not."""
    request = SupervisorRequest(query="please review NCT04280705 for design issues")
    assert request.resolved_nct_id() == "NCT04280705"
    runnable, _ = capability_gate(request)
    assert AgentName.PROTOCOL in runnable
    assert AgentName.REGULATORY in runnable


def test_gate_allows_regulatory_from_a_supplied_summary_alone():
    request = SupervisorRequest(query="check this design", protocol_summary="Phase 3, no interim analysis.")
    runnable, _ = capability_gate(request)
    assert AgentName.REGULATORY in runnable
    assert AgentName.PROTOCOL not in runnable  # no NCT id — nothing to extract from


def test_evidence_is_always_runnable_because_the_query_is_the_question():
    runnable, _ = capability_gate(SupervisorRequest(query="is PFS a valid surrogate?"))
    assert AgentName.EVIDENCE in runnable


# --------------------------------------------------------------------------
# Routing — the narrowing constraint (T-19)
# --------------------------------------------------------------------------


def test_explicit_routing_makes_no_model_call_and_keeps_only_runnable_agents():
    request = SupervisorRequest(query="safety please")  # no drug supplied
    decision = route(request, agents=[AgentName.SAFETY, AgentName.EVIDENCE])

    assert decision.method is RoutingMethod.EXPLICIT
    assert decision.selected == [AgentName.EVIDENCE]  # SAFETY dropped: not runnable
    assert AgentName.SAFETY.value in decision.not_runnable


def test_routing_cannot_select_an_agent_the_gate_blocked():
    """The core T-19 guarantee: selection is an intersection, never a union."""
    request = SupervisorRequest(query="everything")
    decision = route(request, agents=list(AgentName))
    assert AgentName.SAFETY not in decision.selected
    assert AgentName.PROTOCOL not in decision.selected


def test_selecting_regulatory_pulls_in_protocol_it_depends_on():
    """Found on the first real end-to-end run: the router narrowed to
    regulatory+safety for a query that didn't mention field extraction, and
    Regulatory then had no summary to review. Intent is the router's to
    narrow; a data dependency is not."""
    decision = route(
        SupervisorRequest(query="review NCT04280705 against guidance"),
        agents=[AgentName.REGULATORY],
    )

    assert AgentName.PROTOCOL in decision.selected
    assert AgentName.REGULATORY in decision.selected
    assert "added automatically" in decision.rationale


def test_no_dependency_added_when_a_summary_was_supplied_directly():
    """Regulatory can stand alone if it already has its input."""
    decision = route(
        SupervisorRequest(query="check this design", protocol_summary="Phase 3, no interim analysis."),
        agents=[AgentName.REGULATORY],
    )

    assert decision.selected == [AgentName.REGULATORY]
    assert "added automatically" not in decision.rationale


# --------------------------------------------------------------------------
# Summary rendering for the Regulatory agent
# --------------------------------------------------------------------------


def test_summary_states_absent_fields_as_absent_rather_than_omitting_them():
    summary = summary_for_regulatory(_extraction())
    assert "Phase: PHASE3" in summary
    assert "Primary endpoint: not specified in the registry record" in summary
    # It must not fabricate a value for an absent field (HG-1).
    assert "Primary endpoint: None" not in summary


# --------------------------------------------------------------------------
# Failure isolation (T-20)
# --------------------------------------------------------------------------


def test_one_agent_failing_does_not_stop_the_others(monkeypatch):
    """Protocol raises; Evidence still produces output and the run completes
    with the failure recorded rather than propagated."""
    from src.agents import evidence as evidence_mod
    from src.agents import protocol as protocol_mod
    from src.schemas import EvidenceStrength, EvidenceSynthesis

    def boom(nct_id):
        raise RuntimeError("registry exploded")

    def fake_synthesize(question, max_results=5):
        return EvidenceSynthesis(question=question, claims=[], strength=EvidenceStrength.NONE_FOUND)

    monkeypatch.setattr(protocol_mod, "extract", boom)
    monkeypatch.setattr(evidence_mod, "synthesize", fake_synthesize)

    result = run(
        SupervisorRequest(query="review NCT04280705"),
        agents=[AgentName.PROTOCOL, AgentName.EVIDENCE],
    )

    assert result.protocol is None
    assert result.evidence is not None
    assert [f.agent for f in result.failures] == [AgentName.PROTOCOL]
    assert result.failures[0].error_type == "RuntimeError"
    assert "registry exploded" in result.failures[0].detail


def test_regulatory_records_missing_input_when_the_protocol_agent_fails(monkeypatch):
    """Regulatory depends on Protocol. If that dependency produces nothing,
    Regulatory must say so, not be silently skipped."""
    from src.agents import protocol as protocol_mod

    monkeypatch.setattr(protocol_mod, "extract", lambda nct_id: (_ for _ in ()).throw(RuntimeError("no record")))

    result = run(
        SupervisorRequest(query="review NCT04280705"),
        agents=[AgentName.PROTOCOL, AgentName.REGULATORY],
    )

    by_agent = {f.agent: f for f in result.failures}
    assert by_agent[AgentName.REGULATORY].error_type == "MissingInput"
    assert result.regulatory == []


def test_run_with_nothing_selected_completes_cleanly():
    """An empty selection is a real answer, not a crash."""
    result = run(SupervisorRequest(query="what is the weather"), agents=[])

    assert result.routing.selected == []
    assert result.failures == []
    assert result.agents_with_output == []
    assert result.elapsed_seconds >= 0


def test_run_is_not_a_briefing_packet(monkeypatch):
    """D-06: the Supervisor must not invent an overall_confidence."""
    result = run(SupervisorRequest(query="nothing"), agents=[])
    assert not hasattr(result, "overall_confidence")
    assert not hasattr(result, "human_decision")


# --------------------------------------------------------------------------
# Graph shape
# --------------------------------------------------------------------------


def test_regulatory_runs_exactly_once_when_reachable_by_both_paths(monkeypatch):
    """langgraph 1.2.11 executes a node twice if it is reachable from both
    the entry fan-out and a dependency edge in the same run — verified
    directly. _fan_out exists to prevent that; this is its regression test."""
    from src.agents import protocol as protocol_mod
    from src.agents import regulatory as regulatory_mod

    calls = []

    monkeypatch.setattr(protocol_mod, "extract", lambda nct_id: _extraction())

    def counting_review(summary, kb, top_k=6):
        calls.append(summary)
        return []

    monkeypatch.setattr(regulatory_mod, "review", counting_review)

    run(
        SupervisorRequest(query="review NCT04280705"),
        kb=object(),  # never used — review() is stubbed
        agents=[AgentName.PROTOCOL, AgentName.REGULATORY],
    )

    assert len(calls) == 1
    # And it received the summary built from the Protocol agent's output.
    assert "Trial: NCT04280705" in calls[0]


def test_independent_agents_actually_run_in_parallel(monkeypatch):
    """T-04 asked whether LangGraph earns its dependency weight. This is the
    evidence: two independent branches with a 0.5s cost each complete in
    ~0.5s, not ~1.0s. Uses sleeps rather than live calls so the claim is
    verified on every run at zero API cost."""
    import time

    from src.agents import evidence as evidence_mod
    from src.agents import safety as safety_mod
    from src.schemas import EvidenceStrength, EvidenceSynthesis, SafetyScreen

    def slow_evidence(question, max_results=5):
        time.sleep(0.5)
        return EvidenceSynthesis(question=question, claims=[], strength=EvidenceStrength.NONE_FOUND)

    def slow_safety(drug, top_n=10):
        time.sleep(0.5)
        return SafetyScreen(drug=drug, associations=[], total_reports_reviewed=0)

    monkeypatch.setattr(evidence_mod, "synthesize", slow_evidence)
    monkeypatch.setattr(safety_mod, "screen", slow_safety)

    result = run(
        SupervisorRequest(query="anything", drug="pembrolizumab"),
        agents=[AgentName.EVIDENCE, AgentName.SAFETY],
    )

    assert result.evidence is not None and result.safety is not None
    assert result.failures == []
    # Sequential execution would be ~1.0s.
    assert result.elapsed_seconds < 0.85, f"branches did not run concurrently: {result.elapsed_seconds}s"


@pytest.mark.parametrize(
    "selected,expected",
    [
        ([AgentName.EVIDENCE], ["evidence"]),
        ([AgentName.PROTOCOL, AgentName.REGULATORY], ["protocol"]),
        ([AgentName.REGULATORY], ["regulatory"]),
        ([], ["__end__"]),
    ],
)
def test_fan_out_targets(selected, expected):
    from src.agents.supervisor import _fan_out

    assert _fan_out({"selected": selected}) == expected
