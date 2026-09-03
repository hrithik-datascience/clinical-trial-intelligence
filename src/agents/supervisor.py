"""Supervisor Agent (Module 9) — routing, orchestration, aggregation.

Three jobs, in order:

  1. **Route** (T-19). A deterministic capability gate decides which agents
     *can* run — an agent whose required input is absent is never selectable,
     and no model output can override that. A low-effort model call then
     classifies intent and may NARROW that set. It can never extend it, so a
     router that misfires can waste a call but can never invent an input or
     invoke an agent against data that doesn't exist.

  2. **Orchestrate** (LangGraph, T-04). Protocol / Evidence / Safety are
     independent and fan out in parallel; Regulatory genuinely depends on
     Protocol, because reviewing a trial means reviewing the fields actually
     extracted from it. That dependency is a real edge in the graph, not
     ceremony — see Module 9's doc for the measured parallel speed-up.

  3. **Aggregate** (T-20). A failing agent records an AgentFailure and the
     other branches carry on. This is deliberately NOT the silent-swallow
     that T-06 forbids: the failure is a visible field in the output, which
     the Validation layer (Module 10) turns into a MISSING_AGENT_OUTPUT flag.

The output is a SupervisorRun, not a BriefingPacket (D-06). Scoring overall
confidence belongs to Module 10; inventing a number here would be exactly the
placeholder metric Section H forbids.
"""

from __future__ import annotations

import operator
import re
import time
import uuid
from datetime import datetime
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.agents import evidence, protocol, regulatory, safety
from src.kb.build import KB_DIR
from src.kb.store import KnowledgeBase
from src.llm import EFFORT_ROUTING, get_client, model_name, parse_with_retry
from src.schemas import (
    AgentFailure,
    AgentName,
    FieldStatus,
    ProtocolExtraction,
    RoutingDecision,
    RoutingMethod,
    SupervisorRun,
)

_NCT_ID = re.compile(r"\bNCT\d{8}\b")


class SupervisorRequest(BaseModel):
    """One run's inputs.

    `drug` is deliberately never inferred from the query text. An NCT id has
    an unambiguous format and is safe to read out of free text; a drug name
    is not, and guessing one would put an invented input into a safety
    screen — the HG-1 failure this project exists to avoid.
    """

    query: str = Field(min_length=1)
    nct_id: str | None = None
    drug: str | None = None
    question: str | None = None
    protocol_summary: str | None = None

    def resolved_nct_id(self) -> str | None:
        if self.nct_id:
            return self.nct_id
        found = _NCT_ID.search(self.query)
        return found.group(0) if found else None


# --------------------------------------------------------------------------
# 1. Routing
# --------------------------------------------------------------------------


def capability_gate(request: SupervisorRequest) -> tuple[list[AgentName], dict[str, str]]:
    """Which agents have the inputs they need. Pure Python, no model call.

    Returns (runnable, not_runnable) where not_runnable maps an agent name to
    the input it is missing — reported to the caller rather than silently
    dropped, so "why didn't the safety agent run?" always has an answer.
    """
    runnable: list[AgentName] = []
    blocked: dict[str, str] = {}

    nct_id = request.resolved_nct_id()

    if nct_id:
        runnable.append(AgentName.PROTOCOL)
    else:
        blocked[AgentName.PROTOCOL.value] = "no NCT id supplied or found in the query"

    # Evidence falls back to the query itself — a question is always present.
    runnable.append(AgentName.EVIDENCE)

    if nct_id or request.protocol_summary:
        runnable.append(AgentName.REGULATORY)
    else:
        blocked[AgentName.REGULATORY.value] = "no protocol summary and no NCT id to derive one from"

    if request.drug:
        runnable.append(AgentName.SAFETY)
    else:
        blocked[AgentName.SAFETY.value] = "no drug supplied (never inferred from free text)"

    return runnable, blocked


class _LlmRoute(BaseModel):
    agents: list[AgentName] = Field(
        default_factory=list,
        description="Only the agents whose output the query actually calls for.",
    )
    rationale: str = Field(min_length=1, description="One sentence, plain language.")


_ROUTER_SYSTEM_PROMPT = """\
You route a clinical trial reviewer's request to the agents that can answer \
it. You do not answer the request yourself.

The agents:
- protocol: extracts structured fields (phase, population, endpoints, sample \
size, eligibility) from a trial registry record.
- evidence: searches published literature (PubMed) and synthesizes cited \
findings on a clinical or methodological question.
- regulatory: checks a trial's design against ICH/FDA statistical guidance \
and flags points for a reviewer to examine.
- safety: reports FAERS adverse-event data for a drug, separating known \
labeled risks from unexpected signals.

Rules:
1. Choose ONLY from the runnable agents listed in the request. Anything else \
is discarded.
2. Choose every agent the request genuinely calls for, and no others. A \
broad request ("review this trial") legitimately needs several.
3. If the request calls for none of them, return an empty list and say so."""


def _llm_route(query: str, runnable: list[AgentName]) -> tuple[list[AgentName], str]:
    listing = ", ".join(a.value for a in runnable)
    response = parse_with_retry(
        get_client(),
        model=model_name(),
        max_tokens=1000,
        output_config={"effort": EFFORT_ROUTING},
        system=_ROUTER_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Request: {query}\n\nRunnable agents: {listing}\n\nWhich are needed?",
            }
        ],
        output_format=_LlmRoute,
    )
    parsed = response.parsed_output
    # The narrowing constraint (T-19): intersect, never union.
    selected = [a for a in parsed.agents if a in runnable]
    return selected, parsed.rationale


def close_dependencies(
    selected: list[AgentName], request: SupervisorRequest
) -> tuple[list[AgentName], str]:
    """Add back any agent the selection structurally depends on.

    DEVIATION FROM PLAN, found on the first real end-to-end run: the router
    was asked to "review NCT04280705 against statistical guidance, and screen
    the drug's safety signals" and sensibly narrowed to regulatory + safety —
    the query never asks for field extraction. But Regulatory reviews a
    *protocol summary*, and when none is supplied directly that summary can
    only come from the Protocol agent, so Regulatory produced nothing at all.

    The capability gate's promise ("regulatory is runnable, there's an NCT
    id") is only true if Protocol also runs. Intent is the router's to
    narrow; a data dependency is not. This runs after routing, deterministically,
    for the same reason the gate runs before it.
    """
    if AgentName.REGULATORY not in selected or request.protocol_summary is not None:
        return selected, ""
    if AgentName.PROTOCOL in selected:
        return selected, ""
    return (
        [AgentName.PROTOCOL, *selected],
        " Protocol was added automatically: Regulatory reviews an extracted "
        "protocol summary, and no summary was supplied directly.",
    )


def route(
    request: SupervisorRequest, agents: list[AgentName] | None = None
) -> RoutingDecision:
    """Decide which agents to run.

    `agents` short-circuits the model call entirely — used by the UI and by
    tests that shouldn't spend an API call to exercise orchestration.
    """
    runnable, blocked = capability_gate(request)

    if agents is not None:
        selected = [a for a in agents if a in runnable]
        selected, added = close_dependencies(selected, request)
        return RoutingDecision(
            selected=selected,
            not_runnable=blocked,
            method=RoutingMethod.EXPLICIT,
            rationale=f"Caller named {[a.value for a in agents]}; kept those that were runnable.{added}",
        )

    try:
        selected, rationale = _llm_route(request.query, runnable)
    except Exception as exc:  # router unavailable — fall OPEN, never closed
        return RoutingDecision(
            selected=runnable,
            not_runnable=blocked,
            method=RoutingMethod.CAPABILITY_ONLY,
            rationale=(
                f"Intent router failed ({type(exc).__name__}); ran every runnable agent "
                "rather than returning less information than the data supports."
            ),
        )

    selected, added = close_dependencies(selected, request)
    return RoutingDecision(
        selected=selected,
        not_runnable=blocked,
        method=RoutingMethod.LLM_INTENT,
        rationale=f"{rationale}{added}",
    )


# --------------------------------------------------------------------------
# 2. The graph
# --------------------------------------------------------------------------


class _State(TypedDict):
    request: SupervisorRequest
    selected: list[AgentName]
    kb: KnowledgeBase | None
    protocol: ProtocolExtraction | None
    evidence: object | None
    regulatory: list
    safety: object | None
    # Concurrent branches append here, so it needs a reducer.
    failures: Annotated[list[AgentFailure], operator.add]


def _failure(agent: AgentName, exc: Exception) -> AgentFailure:
    # error_type carries the real exception class — a bug in our own code
    # surfaces as e.g. TypeError rather than hiding behind a generic message.
    return AgentFailure(
        agent=agent,
        error_type=type(exc).__name__,
        detail=(str(exc) or repr(exc))[:500],
    )


def summary_for_regulatory(extraction: ProtocolExtraction) -> str:
    """Render an extraction as the protocol summary the Regulatory agent reviews.

    Absent fields are stated as absent rather than omitted. That is honest
    about the source *and* useful: a missing interim-analysis plan is exactly
    the kind of omission the Regulatory agent exists to flag (Module 7's real
    result), and it can only flag what it can see is missing.
    """
    fields = (
        ("Phase", extraction.phase),
        ("Population", extraction.population),
        ("Primary endpoint", extraction.primary_endpoint),
        ("Sample size", extraction.sample_size),
        ("Inclusion criteria", extraction.inclusion_criteria),
        ("Exclusion criteria", extraction.exclusion_criteria),
    )
    lines = [f"Trial: {extraction.nct_id}"]
    for label, field in fields:
        if field.status is FieldStatus.EXTRACTED:
            lines.append(f"{label}: {field.value}")
        else:
            lines.append(f"{label}: not specified in the registry record")
    return "\n".join(lines)


def _node_protocol(state: _State) -> dict:
    try:
        return {"protocol": protocol.extract(state["request"].resolved_nct_id())}
    except Exception as exc:
        return {"failures": [_failure(AgentName.PROTOCOL, exc)]}


def _node_evidence(state: _State) -> dict:
    request = state["request"]
    try:
        return {"evidence": evidence.synthesize(request.question or request.query)}
    except Exception as exc:
        return {"failures": [_failure(AgentName.EVIDENCE, exc)]}


def _node_safety(state: _State) -> dict:
    try:
        return {"safety": safety.screen(state["request"].drug)}
    except Exception as exc:
        return {"failures": [_failure(AgentName.SAFETY, exc)]}


def _node_regulatory(state: _State) -> dict:
    request = state["request"]
    summary = request.protocol_summary
    if summary is None and state.get("protocol") is not None:
        summary = summary_for_regulatory(state["protocol"])

    if summary is None:
        # Selected, but its input never materialised (the Protocol agent it
        # depends on failed). Recorded as a failure, not quietly dropped.
        return {
            "failures": [
                AgentFailure(
                    agent=AgentName.REGULATORY,
                    error_type="MissingInput",
                    detail="no protocol summary available — the Protocol agent it depends on produced none",
                )
            ]
        }

    try:
        kb = state["kb"] or _load_kb()
        return {"regulatory": regulatory.review(summary, kb)}
    except Exception as exc:
        return {"failures": [_failure(AgentName.REGULATORY, exc)]}


def _fan_out(state: _State) -> list[str]:
    """Entry fan-out. Regulatory is entered from here only when Protocol is
    NOT running — otherwise the protocol -> regulatory edge handles it.
    Verified against langgraph 1.2.11: a node reachable by both paths in the
    same run executes twice."""
    selected = set(state["selected"])
    targets = [
        name
        for agent, name in (
            (AgentName.PROTOCOL, "protocol"),
            (AgentName.EVIDENCE, "evidence"),
            (AgentName.SAFETY, "safety"),
        )
        if agent in selected
    ]
    if AgentName.REGULATORY in selected and AgentName.PROTOCOL not in selected:
        targets.append("regulatory")
    return targets or [END]


def _after_protocol(state: _State) -> list[str]:
    return ["regulatory"] if AgentName.REGULATORY in set(state["selected"]) else [END]


def build_graph():
    graph = StateGraph(_State)
    graph.add_node("protocol", _node_protocol)
    graph.add_node("evidence", _node_evidence)
    graph.add_node("regulatory", _node_regulatory)
    graph.add_node("safety", _node_safety)

    graph.add_conditional_edges(START, _fan_out, ["protocol", "evidence", "safety", "regulatory", END])
    graph.add_conditional_edges("protocol", _after_protocol, ["regulatory", END])
    graph.add_edge("evidence", END)
    graph.add_edge("safety", END)
    graph.add_edge("regulatory", END)
    return graph.compile()


_GRAPH = None
_KB: KnowledgeBase | None = None


def _load_kb() -> KnowledgeBase:
    """Load the persisted index. Deliberately does not build one on the fly —
    that would fire four live ingestion APIs as a side effect of a query."""
    global _KB
    if _KB is None:
        if not (KB_DIR / "faiss.index").exists():
            raise FileNotFoundError(
                f"no knowledge base at {KB_DIR} — run `python -m src.kb.build` first"
            )
        _KB = KnowledgeBase.load(KB_DIR)
    return _KB


def _compiled():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# --------------------------------------------------------------------------
# 3. Run
# --------------------------------------------------------------------------


def run(
    request: SupervisorRequest,
    kb: KnowledgeBase | None = None,
    agents: list[AgentName] | None = None,
) -> SupervisorRun:
    """Route, execute the graph, and aggregate.

    Does not raise when an agent fails — see T-20. Check `failures` on the
    result; a run with failures is incomplete, never a finished answer.
    """
    started = time.monotonic()
    decision = route(request, agents=agents)

    final = _compiled().invoke(
        {
            "request": request,
            "selected": decision.selected,
            "kb": kb,
            "protocol": None,
            "evidence": None,
            "regulatory": [],
            "safety": None,
            "failures": [],
        }
    )

    return SupervisorRun(
        run_id=f"run-{uuid.uuid4().hex[:12]}",
        query=request.query,
        created_at=datetime.now(),
        routing=decision,
        protocol=final.get("protocol"),
        evidence=final.get("evidence"),
        regulatory=final.get("regulatory") or [],
        safety=final.get("safety"),
        failures=final.get("failures") or [],
        elapsed_seconds=round(time.monotonic() - started, 2),
    )
