"""Evidence Agent tests — offline logic + the whitespace-normalization fix."""

import pytest
from pydantic import ValidationError

from src.agents.errors import GroundingError, is_grounded
from src.agents.evidence import _LlmClaim, _LlmSynthesis
from src.schemas import EvidenceStrength


def test_is_grounded_accepts_exact_match():
    assert is_grounded("hello world", "say hello world now")


def test_is_grounded_ignores_unicode_thin_space_differences():
    """The real bug found running Module 6: PubMed abstracts use U+2009
    (thin space) around punctuation; the model quotes with an ordinary
    space. Both must be treated as the same text."""
    source = "median PFS was 11.3 months (95% CI = 9.2-13.4)"
    model_quote = "median PFS was 11.3 months (95% CI = 9.2-13.4)"
    assert is_grounded(model_quote, source)


def test_is_grounded_rejects_a_fabricated_quote():
    assert not is_grounded("a phrase never in the source", "completely different text")


def test_llm_synthesis_none_found_cannot_carry_claims():
    with pytest.raises(ValidationError):
        _LlmSynthesis(
            strength=EvidenceStrength.NONE_FOUND,
            claims=[_LlmClaim(text="x", doc_index=0, quoted_text="y")],
        )


def test_llm_synthesis_supported_requires_claims():
    with pytest.raises(ValidationError):
        _LlmSynthesis(strength=EvidenceStrength.SUPPORTED, claims=[])
