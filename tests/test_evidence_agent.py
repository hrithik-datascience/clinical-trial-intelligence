"""Evidence Agent tests — offline logic + the whitespace-normalization fix."""

import pytest
from pydantic import ValidationError

from src.agents.errors import is_grounded
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


def test_is_grounded_ignores_capitalization_of_a_sentence_start():
    """Second real bug found running Module 6: the model capitalizes the
    first letter of a mid-sentence fragment when presenting it as a
    standalone quoted sentence. Content is identical; only case differs."""
    source = "...and 5.7 months; the median PFS was 11.3 months in the group"
    model_quote = "The median PFS was 11.3 months in the group"
    assert is_grounded(model_quote, source)


def test_is_grounded_rejects_a_fabricated_quote():
    assert not is_grounded("a phrase never in the source", "completely different text")


def test_is_grounded_ignores_markdown_escaped_angle_brackets():
    """Real bug found running Module 13's evaluation harness: ClinicalTrials.gov's
    API returns eligibility text with literal backslash-escaped '<'/'>'
    ("PEFR \\> 50%"), verified present in real NCT00413387 and NCT04280705
    text. The model naturally quotes it unescaped -- identical content."""
    source = "FEV1 or PEFR \\> 50% and \\< 80% of the predicted normal"
    model_quote = "FEV1 or PEFR > 50% and < 80% of the predicted normal"
    assert is_grounded(model_quote, source)


def test_is_grounded_still_rejects_a_genuinely_different_number_near_a_bracket():
    """The escape fix must not become permissive about content, only about
    the backslash presentation artifact."""
    source = "PEFR \\> 50% of the predicted normal"
    assert not is_grounded("PEFR > 90% of the predicted normal", source)


def test_llm_synthesis_none_found_cannot_carry_claims():
    with pytest.raises(ValidationError):
        _LlmSynthesis(
            strength=EvidenceStrength.NONE_FOUND,
            claims=[_LlmClaim(text="x", doc_index=0, quoted_text="y")],
        )


def test_llm_synthesis_supported_requires_claims():
    with pytest.raises(ValidationError):
        _LlmSynthesis(strength=EvidenceStrength.SUPPORTED, claims=[])
