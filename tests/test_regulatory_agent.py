"""Regulatory Agent tests — prompt building, offline. Clause-splitting and
selection tests moved to tests/test_kb_chunking.py and tests/test_kb_store.py
once Module 4 replaced the keyword-overlap stopgap with real KB retrieval."""

from src.agents.regulatory import _build_prompt


def test_build_prompt_includes_every_supplied_clause():
    clauses = {
        "4.5": ("Interim Analysis", "All interim analyses should be planned in advance."),
        "3.5": ("Sample Size", "The method by which sample size is calculated should be given."),
    }
    prompt = _build_prompt("A protocol summary.", clauses)

    assert "Clause 4.5 (Interim Analysis)" in prompt
    assert "Clause 3.5 (Sample Size)" in prompt
    assert "All interim analyses should be planned in advance." in prompt
    assert "A protocol summary." in prompt


def test_build_prompt_truncates_a_very_long_clause_text():
    long_text = "x" * 3000
    clauses = {"1.1": ("Title", long_text)}
    prompt = _build_prompt("summary", clauses)
    # _build_prompt caps each clause body at 1500 chars before it reaches the model.
    assert "x" * 1500 in prompt
    assert "x" * 1501 not in prompt
