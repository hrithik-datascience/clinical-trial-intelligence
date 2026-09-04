"""Evaluation harness tests — offline. These test the metric *math* against
fake agent outputs (monkeypatched), not the golden set's real labels against
real agents — that real run is `python -m src.eval.harness` itself, exercised
separately and documented in module_13's doc, not repeated here at API cost."""

from datetime import date, datetime

from src import audit
from src.eval.golden_set import (
    GoldenProtocolExample,
    GoldenRegulatoryExample,
    GoldenSafetyExample,
    ProtocolFieldExpectation,
)
from src.eval.harness import eval_latency_and_overrides, eval_protocol, eval_regulatory, eval_safety
from src.schemas import (
    AuditEventType,
    Citation,
    ExtractedField,
    FieldStatus,
    ProtocolExtraction,
    RegulatoryFinding,
    ReviewDecision,
    Severity,
    SourceType,
    SupervisorRun,
)


def _citation() -> Citation:
    return Citation(
        chunk_id="ctgov:NCT00000000", source_type=SourceType.PROTOCOL,
        source_name="NCT00000000", retrieved_date=date(2026, 9, 4), quoted_text="Phase 3",
    )


def _extraction(**overrides) -> ProtocolExtraction:
    base = dict(
        nct_id="NCT00000000",
        phase=ExtractedField(value="PHASE3", status=FieldStatus.EXTRACTED, citation=_citation()),
        population=ExtractedField.absent(),
        primary_endpoint=ExtractedField(value="Survival", status=FieldStatus.EXTRACTED, citation=_citation()),
        sample_size=ExtractedField(value="100", status=FieldStatus.EXTRACTED, citation=_citation()),
        inclusion_criteria=ExtractedField.absent(),
        exclusion_criteria=ExtractedField.absent(),
    )
    base.update(overrides)
    return ProtocolExtraction(**base)


# --------------------------------------------------------------------------
# Protocol precision/recall
# --------------------------------------------------------------------------


def test_perfect_agreement_gives_precision_and_recall_of_one(monkeypatch):
    from src.agents import protocol as protocol_mod

    example = GoldenProtocolExample(
        nct_id="NCT00000000",
        fields={
            "phase": ProtocolFieldExpectation(should_extract=True, value_contains="3"),
            "population": ProtocolFieldExpectation(should_extract=False),
            "primary_endpoint": ProtocolFieldExpectation(should_extract=True),
            "sample_size": ProtocolFieldExpectation(should_extract=True, value_contains="100"),
            "inclusion_criteria": ProtocolFieldExpectation(should_extract=False),
            "exclusion_criteria": ProtocolFieldExpectation(should_extract=False),
        },
    )
    monkeypatch.setattr("src.eval.harness.PROTOCOL_EXAMPLES", [example])
    monkeypatch.setattr(protocol_mod, "extract", lambda nct_id: _extraction())

    result = eval_protocol()
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.value_mismatches == []
    assert result.errors == []


def test_false_negative_lowers_recall_not_precision(monkeypatch):
    """The agent wrongly abstains on a field that should have been extracted."""
    from src.agents import protocol as protocol_mod

    example = GoldenProtocolExample(
        nct_id="NCT00000000",
        fields={"phase": ProtocolFieldExpectation(should_extract=True)},
    )
    monkeypatch.setattr("src.eval.harness.PROTOCOL_EXAMPLES", [example])
    monkeypatch.setattr(protocol_mod, "extract", lambda nct_id: _extraction(phase=ExtractedField.absent()))

    result = eval_protocol()
    assert result.field_confusion["phase"] == {"TP": 0, "FP": 0, "FN": 1, "TN": 0}
    assert result.recall == 0.0
    assert result.precision is None  # no TP+FP at all


def test_false_positive_lowers_precision_not_recall(monkeypatch):
    """The agent fabricates a value where the ground truth says it should abstain."""
    from src.agents import protocol as protocol_mod

    example = GoldenProtocolExample(
        nct_id="NCT00000000",
        fields={"population": ProtocolFieldExpectation(should_extract=False)},
    )
    monkeypatch.setattr("src.eval.harness.PROTOCOL_EXAMPLES", [example])
    monkeypatch.setattr(
        protocol_mod, "extract",
        lambda nct_id: _extraction(population=ExtractedField(value="adults", status=FieldStatus.EXTRACTED, citation=_citation())),
    )

    result = eval_protocol()
    assert result.field_confusion["population"] == {"TP": 0, "FP": 1, "FN": 0, "TN": 0}
    assert result.precision == 0.0


def test_value_mismatch_is_reported_but_still_counts_as_a_true_positive(monkeypatch):
    """Status agreement (extracted vs. not) and value correctness are tracked
    separately -- a wrong value on a correctly-extracted field is a quality
    issue, not a status-classification error."""
    from src.agents import protocol as protocol_mod

    example = GoldenProtocolExample(
        nct_id="NCT00000000",
        fields={"phase": ProtocolFieldExpectation(should_extract=True, value_contains="4")},
    )
    monkeypatch.setattr("src.eval.harness.PROTOCOL_EXAMPLES", [example])
    monkeypatch.setattr(protocol_mod, "extract", lambda nct_id: _extraction())  # value is "PHASE3", not "4"

    result = eval_protocol()
    assert result.field_confusion["phase"]["TP"] == 1
    assert len(result.value_mismatches) == 1
    assert "PHASE3" in result.value_mismatches[0]


def test_agent_exception_is_recorded_not_raised(monkeypatch):
    from src.agents import protocol as protocol_mod

    example = GoldenProtocolExample(nct_id="NCT00000000", fields={})
    monkeypatch.setattr("src.eval.harness.PROTOCOL_EXAMPLES", [example])

    def boom(nct_id):
        raise RuntimeError("no network")

    monkeypatch.setattr(protocol_mod, "extract", boom)
    result = eval_protocol()
    assert len(result.errors) == 1
    assert "RuntimeError" in result.errors[0]


# --------------------------------------------------------------------------
# Regulatory citation precision
# --------------------------------------------------------------------------


def _finding(clause_id: str) -> RegulatoryFinding:
    return RegulatoryFinding(
        clause_id=clause_id, finding="a finding", severity=Severity.FLAG,
        citation=Citation(
            chunk_id="guidance:ICH_E9:4.5", source_type=SourceType.GUIDANCE,
            source_name="ICH E9 4.5", retrieved_date=date(2026, 9, 4), quoted_text="quoted",
        ),
    )


def test_citation_precision_counts_matched_findings_over_total(monkeypatch):
    from src.agents import regulatory as regulatory_mod

    examples = [
        GoldenRegulatoryExample(id="a", protocol_summary="x", is_synthetic=True,
                                 expected_clause_ids=["ICH_E9 4.5"]),
    ]
    monkeypatch.setattr("src.eval.harness.REGULATORY_EXAMPLES", examples)
    monkeypatch.setattr(
        regulatory_mod, "review",
        lambda summary, kb, top_k=6: [_finding("ICH_E9 4.5"), _finding("ICH_E9 9.9")],
    )

    result = eval_regulatory(kb=object())
    assert result.total_findings == 2
    assert result.matched_findings == 1
    assert result.citation_precision == 0.5
    assert result.examples_with_expected == 1
    assert result.examples_with_expected_hit == 1


def test_true_negative_example_over_max_findings_is_flagged(monkeypatch):
    from src.agents import regulatory as regulatory_mod

    examples = [
        GoldenRegulatoryExample(id="clean", protocol_summary="x", is_synthetic=True,
                                 expected_clause_ids=[], max_expected_findings=1),
    ]
    monkeypatch.setattr("src.eval.harness.REGULATORY_EXAMPLES", examples)
    monkeypatch.setattr(
        regulatory_mod, "review",
        lambda summary, kb, top_k=6: [_finding("ICH_E9 1.1"), _finding("ICH_E9 2.2"), _finding("ICH_E9 3.3")],
    )

    result = eval_regulatory(kb=object())
    assert len(result.over_flag_violations) == 1
    assert "3 findings > max 1" in result.over_flag_violations[0]
    assert result.matched_findings == 0  # nothing expected, so nothing can match
    assert result.citation_precision == 0.0


def test_expected_clause_never_hit_lowers_detection_rate_not_precision(monkeypatch):
    from src.agents import regulatory as regulatory_mod

    examples = [
        GoldenRegulatoryExample(id="missed", protocol_summary="x", is_synthetic=True,
                                 expected_clause_ids=["ICH_E9 4.5"]),
    ]
    monkeypatch.setattr("src.eval.harness.REGULATORY_EXAMPLES", examples)
    monkeypatch.setattr(regulatory_mod, "review", lambda summary, kb, top_k=6: [])

    result = eval_regulatory(kb=object())
    assert result.expected_clause_detection_rate == 0.0
    assert result.citation_precision is None  # no findings at all -- undefined, not zero


# --------------------------------------------------------------------------
# Safety agreement rate
# --------------------------------------------------------------------------


def test_safety_agreement_rate(monkeypatch):

    class FakeLabel:
        reference_text = "fatigue, diarrhea, nausea"

    examples = [
        GoldenSafetyExample(drug="drugx", reaction_term="FATIGUE", expected_known_label_risk=True),
        GoldenSafetyExample(drug="drugx", reaction_term="SEIZURE", expected_known_label_risk=True),  # wrong on purpose
        GoldenSafetyExample(drug="drugx", reaction_term="RANDOM UNRELATED TERM", expected_known_label_risk=False),
    ]
    monkeypatch.setattr("src.eval.harness.SAFETY_EXAMPLES", examples)
    monkeypatch.setattr("src.eval.harness.fetch_label", lambda drug: FakeLabel())

    result = eval_safety()
    assert result.total == 3
    assert result.correct == 2
    assert result.agreement_rate == 2 / 3
    assert len(result.mismatches) == 1
    assert "SEIZURE" in result.mismatches[0]


def test_safety_fetch_label_failure_is_recorded_not_raised(monkeypatch):

    examples = [GoldenSafetyExample(drug="unknown-drug", reaction_term="X", expected_known_label_risk=True)]
    monkeypatch.setattr("src.eval.harness.SAFETY_EXAMPLES", examples)

    def boom(drug):
        raise ValueError("no label found")

    monkeypatch.setattr("src.eval.harness.fetch_label", boom)
    result = eval_safety()
    assert result.total == 0
    assert len(result.errors) == 1


# --------------------------------------------------------------------------
# Latency + human-override rate from the real audit log shape
# --------------------------------------------------------------------------


def test_latency_and_override_rate_read_from_audit_events(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LOG_PATH", tmp_path / "audit_log.jsonl")

    run1 = SupervisorRun(
        run_id="run-1", query="q", created_at=datetime(2026, 9, 4, 10, 0, 0),
        routing=__import__("src.schemas", fromlist=["RoutingDecision"]).RoutingDecision(
            selected=[], method=__import__("src.schemas", fromlist=["RoutingMethod"]).RoutingMethod.EXPLICIT,
            rationale="x",
        ),
        elapsed_seconds=10.0,
    )
    run2 = run1.model_copy(update={"run_id": "run-2", "elapsed_seconds": 20.0})
    audit.log_event("run-1", AuditEventType.RUN_COMPLETED, run1)
    audit.log_event("run-2", AuditEventType.RUN_COMPLETED, run2)

    class FakePacket:
        def model_dump(self, mode="json"):
            return {"human_decision": ReviewDecision.APPROVED.value}

    class FakeRejected:
        def model_dump(self, mode="json"):
            return {"human_decision": ReviewDecision.REJECTED.value}

    audit.log_event("run-1", AuditEventType.HUMAN_DECISION, FakePacket())
    audit.log_event("run-2", AuditEventType.HUMAN_DECISION, FakeRejected())

    result = eval_latency_and_overrides()
    assert result["n_runs"] == 2
    assert result["avg_latency_s"] == 15.0
    assert result["n_human_decisions"] == 2
    assert result["human_override_rate"] == 0.5
