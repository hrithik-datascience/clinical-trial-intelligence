"""Live Safety Agent test — real openFDA FAERS + real FDA label + real API call.

Skipped unless ANTHROPIC_API_KEY is set. Run explicitly:
    venv/Scripts/python.exe -m pytest tests/test_safety_agent_live.py -v -s
"""

import os

import pytest

from src.agents.safety import screen
from src.schemas import _CAUSAL_PATTERN

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="no API key set — skipping live LLM call"
)

# ASSUMPTION A-06 — placeholder drug pending Module 8's real demo choice,
# same one Module 3/4 already verified has real FAERS volume.
DEMO_DRUG = "pembrolizumab"


def test_screen_reports_real_counts_with_no_causal_language():
    result = screen(DEMO_DRUG, top_n=5)

    assert result.drug == DEMO_DRUG
    assert result.total_reports_reviewed > 0
    assert len(result.associations) == 5

    for assoc in result.associations:
        assert assoc.report_count > 0
        assert assoc.citation.quoted_text
        # Same word-boundary regex the schema itself enforces (Module 2's
        # HG-3 validator) -- a naive substring check on "causal" would
        # false-positive on the legitimate, careful phrasing "causally
        # attributable" (i.e. denying causation), which is real model output
        # this test hit on a live run.
        assert _CAUSAL_PATTERN.search(assoc.narrative) is None


def test_screen_distinguishes_known_from_unexpected_risk():
    result = screen(DEMO_DRUG, top_n=10)
    # A drug with substantial FAERS volume should have at least one reaction
    # that matches its own label text — otherwise the known/unexpected
    # distinction (the whole point of this module) isn't exercised at all.
    assert any(a.known_label_risk for a in result.associations)
