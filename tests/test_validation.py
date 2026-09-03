"""Validation layer tests — offline. Every check here is deterministic, so
none of this needs an API call."""

from datetime import date, datetime

import pytest

from src.schemas import (
    AgentFailure,
    AgentName,
    Citation,
    Claim,
    EvidenceStrength,
    EvidenceSynthesis,
    ExtractedField,
    FieldStatus,
    FlagType,
    ProtocolExtraction,
    ReportedAssociation,
    ReviewDecision,
    RoutingDecision,
    RoutingMethod,
    SafetyScreen,
    SourceType,
    SupervisorRun,
)
from src.validation import (
    CONFIDENCE_THRESHOLD,
    check_cross_agent,
    check_missing_output,
    collect_citations,
    confidence_breakdown,
    resolves,
    validate,
)

RETRIEVED = date(2026, 9, 3)


def _citation(chunk_id: str, source_type: SourceType = SourceType.PROTOCOL) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        source_type=source_type,
        source_name="a source",
        retrieved_date=RETRIEVED,
        quoted_text="some quoted text",
    )


def _protocol(nct_id: str = "NCT04280705", extracted_fields: int = 6) -> ProtocolExtraction:
    """A protocol extraction with `extracted_fields` of 6 fields grounded."""
    names = (
        "phase", "population", "primary_endpoint",
        "sample_size", "inclusion_criteria", "exclusion_criteria",
    )
    kwargs = {}
    for i, name in enumerate(names):
        if i < extracted_fields:
            kwargs[name] = ExtractedField(
                value=f"value for {name}",
                status=FieldStatus.EXTRACTED,
                citation=_citation(f"ctgov:{nct_id}"),
            )
        else:
            kwargs[name] = ExtractedField.absent()
    return ProtocolExtraction(nct_id=nct_id, **kwargs)


def _safety(drug: str = "pembrolizumab") -> SafetyScreen:
    return SafetyScreen(
        drug=drug,
        associations=[
            ReportedAssociation(
                adverse_event="DIARRHOEA",
                report_count=2004,
                known_label_risk=True,
                narrative="Reported in association with the product.",
                citation=_citation("faers:pembrolizumab", SourceType.FAERS),
            )
        ],
        total_reports_reviewed=35723,
    )


def _run(
    selected: list[AgentName],
    protocol=None,
    evidence=None,
    regulatory=None,
    safety=None,
    failures=None,
) -> SupervisorRun:
    return SupervisorRun(
        run_id="run-test000000",
        query="a query",
        created_at=datetime(2026, 9, 3, 12, 0, 0),
        routing=RoutingDecision(
            selected=selected,
            method=RoutingMethod.EXPLICIT,
            rationale="test",
        ),
        protocol=protocol,
        evidence=evidence,
        regulatory=regulatory or [],
        safety=safety,
        failures=failures or [],
        elapsed_seconds=1.0,
    )


# --------------------------------------------------------------------------
# Citation resolution (T-22)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunk_id",
    ["ctgov:NCT04280705", "pubmed:32214230", "faers:pembrolizumab", "guidance:ICH_E9:4.5"],
)
def test_known_live_source_ids_resolve_without_a_kb(chunk_id):
    """Three of four agents fetch live and cite document ids, not KB chunks —
    a KB-only check would flag nearly every citation (T-22)."""
    assert resolves(_citation(chunk_id), None) is True


@pytest.mark.parametrize("chunk_id", ["invented-id", "http://example.com", "ctgov:NOT_AN_NCT"])
def test_ids_matching_no_known_scheme_do_not_resolve(chunk_id):
    assert resolves(_citation(chunk_id), None) is False


def test_kb_resolution_is_used_when_available():
    class FakeKB:
        def get_chunk(self, chunk_id):
            return object() if chunk_id == "guidance:FDA_E9R1:9" else None

    assert resolves(_citation("guidance:FDA_E9R1:9"), FakeKB()) is True


def test_unresolvable_citation_raises_a_flag():
    protocol = _protocol()
    protocol.phase.citation = _citation("nonsense")
    packet = validate(_run([AgentName.PROTOCOL], protocol=protocol))

    unresolved = [f for f in packet.validation_flags if f.flag_type is FlagType.UNRESOLVED_CITATION]
    assert len(unresolved) == 1
    assert "nonsense" in unresolved[0].detail


def test_collect_citations_covers_all_four_agents():
    run = _run(
        [AgentName.PROTOCOL, AgentName.EVIDENCE, AgentName.SAFETY],
        protocol=_protocol(extracted_fields=1),
        evidence=EvidenceSynthesis(
            question="q",
            claims=[Claim(text="a claim", citations=[_citation("pubmed:1", SourceType.LITERATURE)])],
            strength=EvidenceStrength.LIMITED,
        ),
        safety=_safety(),
    )
    agents = {agent for agent, _ in collect_citations(run)}
    assert agents == {AgentName.PROTOCOL, AgentName.EVIDENCE, AgentName.SAFETY}


# --------------------------------------------------------------------------
# Missing output
# --------------------------------------------------------------------------


def test_failed_agent_becomes_a_missing_output_flag():
    run = _run(
        [AgentName.PROTOCOL, AgentName.SAFETY],
        safety=_safety(),
        failures=[AgentFailure(agent=AgentName.PROTOCOL, error_type="GroundingError", detail="bad quote")],
    )
    flags = check_missing_output(run)

    assert len(flags) == 1
    assert flags[0].flag_type is FlagType.MISSING_AGENT_OUTPUT
    assert "GroundingError" in flags[0].detail


def test_agent_that_silently_returned_nothing_is_also_flagged():
    """No exception, no output — an empty result is not a finding."""
    run = _run([AgentName.PROTOCOL, AgentName.SAFETY], safety=_safety())
    flags = check_missing_output(run)

    assert [f.agents_involved[0] for f in flags] == [AgentName.PROTOCOL]
    assert "returned no output" in flags[0].detail


def test_no_missing_flags_when_every_selected_agent_produced_output():
    run = _run([AgentName.SAFETY], safety=_safety())
    assert check_missing_output(run) == []


# --------------------------------------------------------------------------
# Cross-agent consistency (A-08)
# --------------------------------------------------------------------------


def test_safety_drug_absent_from_the_trial_is_flagged():
    """Fires on this project's own demo pairing: an oncology drug screened
    against a COVID antiviral trial."""
    run = _run(
        [AgentName.PROTOCOL, AgentName.SAFETY],
        protocol=_protocol(),  # field values never mention the drug
        safety=_safety("pembrolizumab"),
    )
    flags = check_cross_agent(run)

    assert len(flags) == 1
    assert flags[0].flag_type is FlagType.CROSS_AGENT_CONFLICT
    assert set(flags[0].agents_involved) == {AgentName.SAFETY, AgentName.PROTOCOL}


def test_no_conflict_when_the_drug_appears_in_an_extracted_field():
    protocol = _protocol()
    protocol.population.value = "adults receiving pembrolizumab"
    run = _run([AgentName.PROTOCOL, AgentName.SAFETY], protocol=protocol, safety=_safety())

    assert check_cross_agent(run) == []


def test_cross_agent_check_needs_both_agents_to_have_run():
    assert check_cross_agent(_run([AgentName.SAFETY], safety=_safety())) == []
    assert check_cross_agent(_run([AgentName.PROTOCOL], protocol=_protocol())) == []


# --------------------------------------------------------------------------
# Confidence (T-21)
# --------------------------------------------------------------------------


def test_components_are_real_ratios():
    run = _run(
        [AgentName.PROTOCOL, AgentName.SAFETY],
        protocol=_protocol(extracted_fields=3),  # 3/6 = 0.5
        safety=_safety(),
    )
    breakdown = confidence_breakdown(run)

    assert breakdown.components["agent_coverage"] == 1.0  # 2 selected, 2 produced
    assert breakdown.components["protocol_completeness"] == 0.5
    assert breakdown.components["citation_integrity"] == 1.0


def test_inapplicable_components_are_omitted_not_scored_zero():
    """A run without the Protocol agent must not be penalised for the
    absence of protocol completeness."""
    breakdown = confidence_breakdown(_run([AgentName.SAFETY], safety=_safety()))

    assert "protocol_completeness" not in breakdown.components
    assert breakdown.score == 1.0


def test_score_is_the_unweighted_mean_of_components():
    run = _run(
        [AgentName.PROTOCOL, AgentName.SAFETY],
        protocol=_protocol(extracted_fields=0),  # completeness 0.0
        failures=[AgentFailure(agent=AgentName.SAFETY, error_type="NoDataFound", detail="none")],
    )
    breakdown = confidence_breakdown(run)
    # coverage 1/2 = 0.5, completeness 0.0; no citations at all (nothing extracted)
    assert breakdown.components == {"agent_coverage": 0.5, "protocol_completeness": 0.0}
    assert breakdown.score == 0.25


def test_empty_run_scores_zero_and_says_why():
    breakdown = confidence_breakdown(_run([]))
    assert breakdown.score == 0.0
    assert "nothing ran" in breakdown.explain()


def test_explain_decomposes_the_score():
    breakdown = confidence_breakdown(_run([AgentName.SAFETY], safety=_safety()))
    text = breakdown.explain()
    assert "agent_coverage=1.00" in text
    assert "mean" in text


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def test_low_confidence_flag_carries_the_decomposition_and_its_caveat():
    run = _run(
        [AgentName.PROTOCOL, AgentName.SAFETY],
        protocol=_protocol(extracted_fields=0),
        failures=[AgentFailure(agent=AgentName.SAFETY, error_type="NoDataFound", detail="none")],
    )
    packet = validate(run)

    low = [f for f in packet.validation_flags if f.flag_type is FlagType.LOW_CONFIDENCE]
    assert len(low) == 1
    assert "agent_coverage" in low[0].detail  # decomposed, not a bare number
    assert "not a probability" in low[0].detail
    assert packet.overall_confidence < CONFIDENCE_THRESHOLD


def test_clean_run_produces_no_flags():
    protocol = _protocol()
    protocol.population.value = "adults receiving pembrolizumab"
    packet = validate(_run([AgentName.PROTOCOL, AgentName.SAFETY], protocol=protocol, safety=_safety()))

    assert packet.validation_flags == []
    assert packet.overall_confidence == 1.0


def test_validation_never_makes_a_packet_final():
    """HG-4: only a recorded human decision can, and that is Module 11."""
    packet = validate(_run([AgentName.SAFETY], safety=_safety()))

    assert packet.human_decision is ReviewDecision.PENDING
    assert packet.is_final is False
    assert packet.decided_at is None


def test_packet_carries_the_run_identity_and_agent_outputs():
    run = _run([AgentName.SAFETY], safety=_safety())
    packet = validate(run)

    assert packet.run_id == run.run_id
    assert packet.query == run.query
    assert packet.created_at == run.created_at
    assert packet.safety is run.safety
