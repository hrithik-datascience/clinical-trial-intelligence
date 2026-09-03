"""Regulatory Agent tests — prompt building and identifier labelling, offline.
Clause-splitting and selection tests moved to tests/test_kb_chunking.py and
tests/test_kb_store.py once Module 4 replaced the keyword-overlap stopgap with
real KB retrieval."""

from datetime import date

from src.agents.regulatory import _build_prompt, _display_label
from src.schemas import Chunk, SourceType


def _chunk(chunk_id: str, metadata: dict) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.rsplit(":", 1)[0],
        source_type=SourceType.GUIDANCE,
        source_name="ICH E9",
        title="A clause",
        text="body text",
        retrieved_date=date(2026, 9, 3),
        metadata=metadata,
    )


def test_display_label_uses_the_clause_number_when_there_is_one():
    label = _display_label(_chunk("guidance:ICH_E9:4.5", {"clause_id": "4.5", "guidance_key": "ICH_E9"}))
    assert label == "ICH_E9 4.5"


def test_display_label_never_exposes_a_namespace_prefix_for_generic_chunks():
    """Regression: the model strips a 'guidance:' prefix when asked to echo
    an identifier back, which raised a false GroundingError twice — once in
    Module 4 (clause branch) and again in Module 9 (this fallback branch,
    which the first fix missed)."""
    label = _display_label(_chunk("guidance:FDA_E9R1:9", {}))

    assert label == "FDA_E9R1 9"
    assert "guidance:" not in label
    assert ":" not in label


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
