"""Live Supervisor test — real intent routing and a real orchestrated run.

Scoped deliberately to what Module 9 adds: the LLM intent router, and the
Protocol -> Regulatory dependency edge with its summary hand-off. The four
agents each have their own live tests already, so this does not re-run all
of them (3 live calls here, not 5).

Skipped unless ANTHROPIC_API_KEY is set. Run explicitly:
    venv/Scripts/python.exe -m pytest tests/test_supervisor_live.py -v -s
"""

import os

import pytest

from src.agents.supervisor import SupervisorRequest, route, run
from src.ingest import guidance
from src.kb.store import KnowledgeBase
from src.schemas import AgentName, RoutingMethod

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="no API key set — skipping live LLM call"
)


@pytest.fixture(scope="module")
def guidance_kb() -> KnowledgeBase:
    """Guidance-only KB built from the real ICH E9 / FDA E9(R1) PDFs, so this
    test does not depend on `python -m src.kb.build` having been run (the
    persisted index is gitignored)."""
    kb = KnowledgeBase()
    kb.add_documents(guidance.fetch_all())
    return kb


def test_router_narrows_to_the_agents_the_query_actually_asks_for():
    """All four agents are runnable here — an NCT id and a drug are both
    supplied — so anything the router leaves out, it left out on intent."""
    request = SupervisorRequest(
        query="What does the published literature say about whether PFS predicts overall survival?",
        nct_id="NCT04280705",
        drug="pembrolizumab",
    )
    decision = route(request)

    assert decision.method is RoutingMethod.LLM_INTENT
    assert AgentName.EVIDENCE in decision.selected
    # A pure literature question should not pull in a FAERS safety screen.
    assert AgentName.SAFETY not in decision.selected
    assert decision.rationale


def test_router_can_never_select_an_agent_whose_input_is_missing():
    """The T-19 guarantee, against a live model: ask directly for a safety
    screen but supply no drug. The gate must win."""
    decision = route(SupervisorRequest(query="run a FAERS safety screen on this drug"))

    assert AgentName.SAFETY not in decision.selected
    assert AgentName.SAFETY.value in decision.not_runnable


def test_end_to_end_protocol_feeds_regulatory(guidance_kb: KnowledgeBase):
    """The real dependency edge: the Protocol agent's extraction becomes the
    summary the Regulatory agent reviews, against real ICH E9 text."""
    result = run(
        SupervisorRequest(query="Review NCT04280705 against statistical guidance"),
        kb=guidance_kb,
        agents=[AgentName.PROTOCOL, AgentName.REGULATORY],  # explicit: no router call
    )

    assert result.protocol is not None
    assert result.protocol.nct_id == "NCT04280705"
    assert AgentName.PROTOCOL in result.agents_with_output
    assert result.elapsed_seconds > 0

    # What this module owns is the hand-off: Regulatory must have received a
    # real summary derived from the Protocol agent. A MissingInput failure
    # would mean the edge itself broke. A GroundingError would mean the edge
    # worked and the model then fabricated a quote — the grounding check
    # doing its job, and known to happen occasionally on live re-runs
    # (Modules 6-7), so it is not asserted away here.
    missing_input = [
        f for f in result.failures
        if f.agent is AgentName.REGULATORY and f.error_type == "MissingInput"
    ]
    assert not missing_input, "Protocol -> Regulatory hand-off failed to supply a summary"

    for finding in result.regulatory:
        assert finding.citation.quoted_text
        assert finding.clause_id
        assert "guidance:" not in finding.clause_id  # deviation 3 regression
