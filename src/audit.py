"""Audit Log & Observability (Module 12).

Append-only, JSONL: one AuditEvent per line, opened in mode="a" every write
(T-23). "Append-only" is the literal file-open mode, not a design metaphor —
nothing in this module ever seeks back into the file and rewrites a line. A
human decision often arrives in a later process (Module 11's review UI) than
the run that produced the packet, so events are correlated across time by
run_id rather than requiring a per-run record to be found and mutated.

Six event types, one per real pipeline stage: a query arriving, a routing
decision, each real LLM call (the observability granularity — latency, real
measured token counts, and a cost estimate flagged as an assumption, A-09),
a completed Supervisor run, a validated packet, and a human's final decision.
Every payload is a real Pydantic model from elsewhere in this project, dumped
to a dict at the point of logging — the type safety lives at construction,
not in this file's envelope.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from src.schemas import AuditEvent, AuditEventType, LLMCallRecord

LOG_PATH = Path("data/audit/audit_log.jsonl")
_write_lock = threading.Lock()

# ASSUMPTION A-09: no verified public price sheet exists in this project for
# these specific model ids. Token counts are real measurements from the API
# response; this dollar figure is a placeholder conversion only.
_PRICE_PER_MILLION_USD = {
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-opus-5": {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
}
_DEFAULT_PRICE = {"input": 3.00, "output": 15.00}  # Sonnet-tier fallback for an unlisted model


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = _PRICE_PER_MILLION_USD.get(model, _DEFAULT_PRICE)
    return round(
        input_tokens / 1_000_000 * price["input"] + output_tokens / 1_000_000 * price["output"],
        6,
    )


def _write(event: AuditEvent) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = event.model_dump_json()
    with _write_lock:  # Module 9's parallel agent branches can log concurrently
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_event(run_id: str, event_type: AuditEventType, payload: BaseModel) -> AuditEvent:
    """Append one event. Returns it so a caller can inspect what was logged
    without a second read of the file."""
    event = AuditEvent(
        event_id=f"evt-{uuid.uuid4().hex[:12]}",
        event_type=event_type,
        run_id=run_id,
        timestamp=datetime.now(),
        payload=payload.model_dump(mode="json"),
    )
    _write(event)
    return event


def log_llm_call(
    *,
    run_id: str,
    agent: str,
    model: str,
    effort: str,
    latency_s: float,
    input_tokens: int,
    output_tokens: int,
    retrieved_chunk_ids: list[str] | None = None,
    input_preview: str = "",
    output_preview: str = "",
) -> AuditEvent:
    record = LLMCallRecord(
        agent=agent,
        run_id=run_id,
        model=model,
        effort=effort,
        latency_s=round(latency_s, 3),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=estimate_cost_usd(model, input_tokens, output_tokens),
        retrieved_chunk_ids=retrieved_chunk_ids or [],
        input_preview=input_preview[:200],
        output_preview=output_preview[:200],
    )
    return log_event(run_id, AuditEventType.LLM_CALL, record)


def read_events(run_id: str | None = None) -> list[AuditEvent]:
    """Read the log back, optionally filtered to one run_id.

    A malformed line raises rather than being silently skipped — a broken
    audit trail should be loud, not quietly incomplete. This is a linear
    scan over the whole file (T-23); fine at this project's event volume.
    """
    if not LOG_PATH.exists():
        return []

    events: list[AuditEvent] = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = AuditEvent.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"{LOG_PATH}:{line_no}: corrupted audit event") from exc
            if run_id is None or event.run_id == run_id:
                events.append(event)
    return events


def llm_calls_for_run(run_id: str) -> list[LLMCallRecord]:
    """Convenience: just the LLM_CALL events for a run, as typed records."""
    return [
        LLMCallRecord.model_validate(e.payload)
        for e in read_events(run_id)
        if e.event_type is AuditEventType.LLM_CALL
    ]


def total_cost_usd(run_id: str) -> float:
    return round(sum(c.cost_usd for c in llm_calls_for_run(run_id)), 6)
