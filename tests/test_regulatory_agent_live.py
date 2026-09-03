"""Live Regulatory Agent test — real ICH E9 text via the shared knowledge
base (Module 4) + real API call.

Skipped unless ANTHROPIC_API_KEY is set. Run explicitly:
    venv/Scripts/python.exe -m pytest tests/test_regulatory_agent_live.py -v -s
"""

import os

import pytest

from src.agents.regulatory import review
from src.ingest import guidance
from src.kb.store import KnowledgeBase
from src.schemas import Severity

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="no API key set — skipping live LLM call"
)

PROTOCOL_SUMMARY = """\
Phase 3, randomized, double-blind, placebo-controlled trial of an antiviral
agent in hospitalized adults with COVID-19. Primary endpoint: time to
recovery, assessed via an 8-category ordinal scale. Sample size: 1062
participants, enrolled at multiple sites. No interim analysis plan or
multiplicity adjustment strategy is described in the available protocol
summary. Two secondary endpoints (mortality, clinical status at day 15) are
also assessed without a stated multiplicity correction method."""


@pytest.fixture(scope="module")
def guidance_kb() -> KnowledgeBase:
    """A guidance-only KB, built from the real ICH E9 / FDA E9(R1) PDFs --
    smaller and faster than the full four-source Module 4 build, but real
    text end to end, not a fixture."""
    kb = KnowledgeBase()
    kb.add_documents(guidance.fetch_all())
    return kb


def test_review_grounds_every_finding_in_a_real_clause(guidance_kb: KnowledgeBase):
    findings = review(PROTOCOL_SUMMARY, guidance_kb, top_k=6)

    assert len(findings) > 0
    for finding in findings:
        assert finding.citation.quoted_text
        assert finding.severity in (Severity.INFO, Severity.FLAG)
        # HG on this agent: vocabulary that would assert a determination
        # must never appear, and the schema doesn't even offer the enum
        # value to select — this checks the free-text finding too.
        lowered = finding.finding.lower()
        for banned in ("compliant", "violat", "approved", "non-compliant"):
            assert banned not in lowered


def test_review_returns_empty_for_an_unrelated_summary(guidance_kb: KnowledgeBase):
    findings = review(
        "A retrospective chart review of manufacturing cold-chain logistics "
        "for vaccine distribution in rural clinics.",
        guidance_kb,
        top_k=6,
    )
    # Not asserting == [] strictly (the model may find a tenuous link), but
    # this should be small — a sanity check retrieval isn't forcing noise.
    assert len(findings) <= 2
