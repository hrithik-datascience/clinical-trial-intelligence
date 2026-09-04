"""Audit log tests — offline. Every test points LOG_PATH at a pytest
tmp_path via monkeypatch, never the real data/audit/audit_log.jsonl, so
running the suite never pollutes the real log with test noise."""

from datetime import date, datetime

import pytest
from pydantic import BaseModel

from src import audit
from src.schemas import (
    AuditEventType,
    Citation,
    LLMCallRecord,
    ReviewDecision,
    SourceType,
)


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LOG_PATH", tmp_path / "audit_log.jsonl")


class _Payload(BaseModel):
    note: str


def test_log_path_does_not_exist_until_first_write(tmp_path):
    assert not audit.LOG_PATH.exists()
    audit.log_event("run-1", AuditEventType.QUERY_RECEIVED, _Payload(note="hi"))
    assert audit.LOG_PATH.exists()


def test_write_is_append_only_mode():
    """The literal file-open mode, not a design metaphor (T-23)."""
    audit.log_event("run-1", AuditEventType.QUERY_RECEIVED, _Payload(note="first"))
    audit.log_event("run-1", AuditEventType.QUERY_RECEIVED, _Payload(note="second"))

    lines = audit.LOG_PATH.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_read_events_round_trips_the_payload():
    audit.log_event("run-1", AuditEventType.QUERY_RECEIVED, _Payload(note="hello"))
    events = audit.read_events()

    assert len(events) == 1
    assert events[0].event_type is AuditEventType.QUERY_RECEIVED
    assert events[0].run_id == "run-1"
    assert events[0].payload == {"note": "hello"}
    assert isinstance(events[0].timestamp, datetime)


def test_read_events_filters_by_run_id():
    audit.log_event("run-1", AuditEventType.QUERY_RECEIVED, _Payload(note="a"))
    audit.log_event("run-2", AuditEventType.QUERY_RECEIVED, _Payload(note="b"))
    audit.log_event("run-1", AuditEventType.RUN_COMPLETED, _Payload(note="c"))

    run_1_events = audit.read_events(run_id="run-1")
    assert len(run_1_events) == 2
    assert all(e.run_id == "run-1" for e in run_1_events)


def test_read_events_on_a_missing_file_returns_empty_not_an_error():
    assert audit.read_events() == []


def test_corrupted_line_raises_rather_than_being_silently_skipped():
    audit.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit.LOG_PATH.write_text("not valid json at all\n", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupted audit event"):
        audit.read_events()


def test_blank_lines_are_skipped_without_error():
    audit.log_event("run-1", AuditEventType.QUERY_RECEIVED, _Payload(note="a"))
    with open(audit.LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n\n")
    audit.log_event("run-1", AuditEventType.RUN_COMPLETED, _Payload(note="b"))

    assert len(audit.read_events()) == 2


# --------------------------------------------------------------------------
# LLM call logging (T-24) and cost (A-09)
# --------------------------------------------------------------------------


def test_log_llm_call_computes_cost_from_real_token_counts():
    event = audit.log_llm_call(
        run_id="run-1", agent="protocol", model="claude-sonnet-5", effort="medium",
        latency_s=1.234, input_tokens=1000, output_tokens=500,
    )
    record = LLMCallRecord.model_validate(event.payload)

    assert record.input_tokens == 1000
    assert record.output_tokens == 500
    # 1000/1e6*3.00 + 500/1e6*15.00 = 0.003 + 0.0075 = 0.0105
    assert record.cost_usd == pytest.approx(0.0105)


def test_log_llm_call_falls_back_to_sonnet_tier_pricing_for_an_unknown_model():
    cost_known = audit.estimate_cost_usd("claude-sonnet-5", 1000, 1000)
    cost_unknown = audit.estimate_cost_usd("some-future-model-id", 1000, 1000)
    assert cost_unknown == cost_known


def test_log_llm_call_truncates_previews_to_200_chars():
    event = audit.log_llm_call(
        run_id="run-1", agent="evidence", model="claude-sonnet-5", effort="medium",
        latency_s=0.5, input_tokens=10, output_tokens=10,
        input_preview="x" * 500, output_preview="y" * 500,
    )
    record = LLMCallRecord.model_validate(event.payload)
    assert len(record.input_preview) == 200
    assert len(record.output_preview) == 200


def test_log_llm_call_carries_retrieved_chunk_ids():
    """The one observability field specific to Regulatory, the only agent
    that retrieves from the KB."""
    event = audit.log_llm_call(
        run_id="run-1", agent="regulatory", model="claude-sonnet-5", effort="high",
        latency_s=2.0, input_tokens=100, output_tokens=100,
        retrieved_chunk_ids=["guidance:ICH_E9:4.5", "guidance:ICH_E9:5.6"],
    )
    record = LLMCallRecord.model_validate(event.payload)
    assert record.retrieved_chunk_ids == ["guidance:ICH_E9:4.5", "guidance:ICH_E9:5.6"]


def test_llm_calls_for_run_and_total_cost():
    audit.log_llm_call(run_id="run-1", agent="protocol", model="claude-sonnet-5", effort="medium",
                        latency_s=1.0, input_tokens=1000, output_tokens=500)
    audit.log_llm_call(run_id="run-1", agent="regulatory", model="claude-sonnet-5", effort="high",
                        latency_s=2.0, input_tokens=2000, output_tokens=1000)
    audit.log_llm_call(run_id="run-2", agent="safety", model="claude-sonnet-5", effort="medium",
                        latency_s=1.0, input_tokens=500, output_tokens=500)

    calls = audit.llm_calls_for_run("run-1")
    assert len(calls) == 2
    assert audit.total_cost_usd("run-1") == pytest.approx(sum(c.cost_usd for c in calls))
    assert audit.total_cost_usd("run-2") < audit.total_cost_usd("run-1")


def test_llm_calls_for_run_ignores_other_event_types():
    audit.log_event("run-1", AuditEventType.ROUTING_DECISION, _Payload(note="not an llm call"))
    assert audit.llm_calls_for_run("run-1") == []
