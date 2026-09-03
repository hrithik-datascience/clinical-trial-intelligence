"""Live Evidence Agent test — real PubMed search + real API call.

Skipped unless ANTHROPIC_API_KEY is set. Run explicitly:
    venv/Scripts/python.exe -m pytest tests/test_evidence_agent_live.py -v -s
"""

import os

import pytest

from src.agents.evidence import synthesize
from src.schemas import EvidenceStrength

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="no API key set — skipping live LLM call"
)


def test_synthesize_grounds_every_claim_in_a_real_abstract():
    result = synthesize(
        "Is progression-free survival a valid surrogate endpoint for overall "
        "survival in oncology trials?",
        max_results=4,
    )

    assert result.strength is not EvidenceStrength.NONE_FOUND
    assert len(result.claims) > 0

    for claim in result.claims:
        assert len(claim.citations) >= 1
        for citation in claim.citations:
            # Reaching here at all is the proof the grounding check held —
            # a fabricated quote would have raised GroundingError already.
            assert citation.quoted_text
            assert citation.source_name.startswith("PMID")


def test_synthesize_reports_none_found_for_a_nonsense_query():
    result = synthesize("xyzzy quantum flibbertigibbet nonsense query 12345", max_results=3)
    assert result.strength is EvidenceStrength.NONE_FOUND
    assert result.claims == []
