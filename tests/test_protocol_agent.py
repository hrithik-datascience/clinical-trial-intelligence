"""Protocol Agent tests.

Offline tests exercise the grounding check and schema translation without
hitting the API. Live behaviour (does the model actually abstain correctly on
a real record) is checked by `tests/test_protocol_agent_live.py`, which is
skipped unless ANTHROPIC_API_KEY is set — see that file for why the split.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from src.agents.protocol import GroundingError, _LlmField, _to_extracted_field
from src.schemas import FieldStatus

SOURCE_TEXT = "Phase: PHASE3\nEnrollment: 1062 (ACTUAL)\nConditions: COVID-19"


def test_grounding_check_accepts_a_real_quote():
    field = _LlmField(status="extracted", value="PHASE3", quoted_text="PHASE3")
    result = _to_extracted_field(field, SOURCE_TEXT, "NCT04280705", date.today(), "doc1")
    assert result.status is FieldStatus.EXTRACTED
    assert result.citation.quoted_text == "PHASE3"


def test_grounding_check_rejects_a_fabricated_quote():
    """The core enforcement of this module: a plausible-sounding invented
    quote must be caught even though it satisfies the Module 2 schema on its
    own — Citation.quoted_text just requires min_length=1, not truth."""
    field = _LlmField(
        status="extracted", value="Phase 2", quoted_text="a phrase never in the source"
    )
    with pytest.raises(GroundingError):
        _to_extracted_field(field, SOURCE_TEXT, "NCT04280705", date.today(), "doc1")


def test_not_specified_never_reaches_the_grounding_check():
    field = _LlmField(status="not_specified")
    result = _to_extracted_field(field, SOURCE_TEXT, "NCT04280705", date.today(), "doc1")
    assert result.status is FieldStatus.NOT_SPECIFIED
    assert result.value is None
    assert result.citation is None


def test_llm_field_extracted_requires_value_and_quote():
    with pytest.raises(ValidationError):
        _LlmField(status="extracted", value="PHASE3", quoted_text=None)


def test_llm_field_not_specified_rejects_a_value():
    with pytest.raises(ValidationError):
        _LlmField(status="not_specified", value="PHASE3", quoted_text="PHASE3")
