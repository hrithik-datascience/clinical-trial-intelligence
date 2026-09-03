"""Safety Agent tests — offline. is_known_label_risk is pure text matching,
so it's tested directly against a real captured label fragment (KEYTRUDA /
pembrolizumab adverse_reactions text, captured 2026-09-03), not an invented
payload."""

from src.agents.safety import _build_prompt, is_known_label_risk

# Real excerpt from the FDA-approved KEYTRUDA label's adverse_reactions
# section, captured live. US spelling, as FDA labels use.
REAL_LABEL_FRAGMENT = (
    "atients) were: KEYTRUDA as a single agent: fatigue, musculoskeletal "
    "pain, rash, diarrhea, pyrexia, cough, decreased appetite, pruritus, "
    "dyspnea, constipation, pain, abdominal pain, nausea, and hypothy"
)


def test_matches_us_spelling_directly():
    assert is_known_label_risk("diarrhea", REAL_LABEL_FRAGMENT) is True


def test_matches_case_insensitively():
    assert is_known_label_risk("DIARRHEA", REAL_LABEL_FRAGMENT) is True


def test_normalizes_meddra_british_spelling_to_match_us_label():
    """MedDRA/FAERS reaction terms use British spelling; FDA labels use US
    spelling. This is the real mismatch T-18 exists to resolve."""
    assert is_known_label_risk("DIARRHOEA", REAL_LABEL_FRAGMENT) is True


def test_returns_false_for_a_term_genuinely_absent_from_the_label():
    assert is_known_label_risk("HEPATOTOXICITY", REAL_LABEL_FRAGMENT) is False


def test_does_not_match_a_medical_synonym_not_lexically_present():
    """Deliberately conservative (T-18): a clinically-related but
    differently-worded term does not match. Documents the real limitation
    rather than hiding it."""
    assert is_known_label_risk("MALIGNANT NEOPLASM PROGRESSION", REAL_LABEL_FRAGMENT) is False


def test_build_prompt_labels_known_and_unexpected_reactions_distinctly():
    entries = [
        {"term": "DIARRHOEA", "count": 2004, "known": True},
        {"term": "RARE UNEXPECTED EVENT", "count": 3, "known": False},
    ]
    prompt = _build_prompt("pembrolizumab", entries)

    assert "[0] DIARRHOEA -- 2004 reports -- known/labeled risk" in prompt
    assert "[1] RARE UNEXPECTED EVENT -- 3 reports -- NOT on current label (unexpected)" in prompt
