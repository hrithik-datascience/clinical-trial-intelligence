"""Live Protocol Agent test — real API call, real cost (a few cents).

Skipped automatically when ANTHROPIC_API_KEY is not set, so the offline suite
(tests/test_protocol_agent.py) stays free and fast. Run explicitly:

    venv/Scripts/python.exe -m pytest tests/test_protocol_agent_live.py -v -s
"""

import os

import pytest

from src.agents.protocol import extract
from src.schemas import FieldStatus

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="no API key set — skipping live LLM call"
)


def test_extract_real_study_grounds_every_extracted_field():
    """NCT04280705 is a real, stable, completed trial — safe as a fixture."""
    result = extract("NCT04280705")

    assert result.nct_id == "NCT04280705"
    # At least the fields that are unambiguously in the registry record.
    assert result.phase.status is FieldStatus.EXTRACTED
    assert result.sample_size.status is FieldStatus.EXTRACTED

    for name in (
        "phase", "population", "primary_endpoint",
        "sample_size", "inclusion_criteria", "exclusion_criteria",
    ):
        field = getattr(result, name)
        if field.status is FieldStatus.EXTRACTED:
            # Every grounded field must carry a citation whose quote is real —
            # the GroundingError path already enforces this, so reaching here
            # at all is the proof it held.
            assert field.citation is not None
            assert field.citation.quoted_text
        else:
            assert field.value is None
            assert field.citation is None
