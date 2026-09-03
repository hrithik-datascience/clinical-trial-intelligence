"""Regulatory Agent tests — clause parsing and selection, offline."""

from src.agents.clauses import Clause, select_relevant_clauses, split_into_clauses

# A minimal but structurally real fragment: TOC-style short entries followed
# by real numbered sections with actual body text, mirroring the shape of a
# real ICH/FDA PDF extraction.
SAMPLE_DOC = """
Table of contents
2.2.1 Population
2.2.2 Primary and Secondary Variables

Statistical Principles for Clinical Trials
2.2.1 Population
In the earlier phases of drug development the choice of subjects for a
clinical trial may be heavily influenced by the wish to maximise the chance
of observing specific clinical effects of interest, spanning well beyond the
minimum body length threshold used to distinguish real content from a bare
table-of-contents entry.

2.2.2 Primary and Secondary Variables
The primary variable should be the variable capable of providing the most
clinically relevant and convincing evidence directly related to the primary
objective of the trial, again long enough to clear the minimum body filter.
"""


def test_split_skips_toc_entries():
    clauses = split_into_clauses(SAMPLE_DOC, min_body_chars=40)
    ids = [c.clause_id for c in clauses]
    # Real sections appear once each, not twice (TOC + body).
    assert ids.count("2.2.1") == 1
    assert ids.count("2.2.2") == 1


def test_split_captures_real_body_text():
    clauses = split_into_clauses(SAMPLE_DOC, min_body_chars=40)
    pop = next(c for c in clauses if c.clause_id == "2.2.1")
    assert "choice of subjects" in pop.text


def test_select_relevant_clauses_ranks_by_keyword_overlap():
    clauses = [
        Clause("2.2.1", "Population", "subjects patients target population enrollment"),
        Clause("2.3.1", "Blinding", "double-blind placebo masking investigator"),
        Clause("9.9.9", "Unrelated", "completely different topic about manufacturing"),
    ]
    top = select_relevant_clauses(clauses, "randomized double-blind placebo trial", top_k=2)
    ids = [c.clause_id for c in top]
    assert "2.3.1" in ids
    assert "9.9.9" not in ids


def test_select_relevant_clauses_returns_empty_when_nothing_overlaps():
    clauses = [Clause("1.1", "Title", "completely unrelated manufacturing content")]
    top = select_relevant_clauses(clauses, "xyz nonsense query with no overlap", top_k=5)
    assert top == []
