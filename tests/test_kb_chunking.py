"""Chunking tests — clause splitting (moved from Module 7's retired
clauses.py) and generic paragraph splitting, both offline."""

from datetime import date

from src.kb.chunking import _chunk_generic, _chunk_guidance, chunk_document
from src.schemas import RawDocument, SourceType

# A minimal but structurally real fragment: TOC-style short entries followed
# by real numbered sections with actual body text, mirroring the shape of a
# real ICH/FDA PDF extraction.
SAMPLE_GUIDANCE_TEXT = """
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


def _guidance_doc(text: str = SAMPLE_GUIDANCE_TEXT) -> RawDocument:
    return RawDocument(
        doc_id="guidance:ICH_E9",
        source_type=SourceType.GUIDANCE,
        source_name="ICH E9",
        title="ICH E9: Statistical Principles for Clinical Trials",
        text=text,
        url="https://example.org/e9",
        retrieved_date=date(2026, 9, 3),
    )


def test_chunk_guidance_skips_toc_entries():
    chunks = _chunk_guidance(_guidance_doc(), min_body_chars=40)
    ids = [c.metadata["clause_id"] for c in chunks]
    # Real sections appear once each, not twice (TOC + body).
    assert ids.count("2.2.1") == 1
    assert ids.count("2.2.2") == 1


def test_chunk_guidance_captures_real_body_text_and_ids():
    chunks = _chunk_guidance(_guidance_doc(), min_body_chars=40)
    pop = next(c for c in chunks if c.metadata["clause_id"] == "2.2.1")
    assert "choice of subjects" in pop.text
    assert pop.chunk_id == "guidance:ICH_E9:2.2.1"
    assert pop.metadata["guidance_key"] == "ICH_E9"
    assert pop.source_type is SourceType.GUIDANCE


def test_chunk_document_routes_guidance_through_clause_splitting():
    chunks = chunk_document(_guidance_doc())
    assert {c.metadata["clause_id"] for c in chunks} == {"2.2.1", "2.2.2"}


def test_chunk_document_falls_back_to_generic_when_no_clause_headers():
    doc = _guidance_doc(text="Just a paragraph of guidance text with no numbered headers at all.")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "guidance:ICH_E9:0"


def _generic_doc(text: str) -> RawDocument:
    return RawDocument(
        doc_id="pubmed:99999999",
        source_type=SourceType.LITERATURE,
        source_name="PMID 99999999",
        title="A study of something",
        text=text,
        url="https://pubmed.ncbi.nlm.nih.gov/99999999/",
        retrieved_date=date(2026, 9, 3),
    )


def test_chunk_generic_keeps_a_short_document_as_one_chunk():
    chunks = _chunk_generic(_generic_doc("One short paragraph."), chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].text == "One short paragraph."


def test_chunk_generic_splits_long_document_into_multiple_chunks_with_overlap():
    paragraphs = [f"Paragraph {i} " + "word " * 40 for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = _chunk_generic(_generic_doc(text), chunk_size=300, overlap=50)

    assert len(chunks) > 1
    # Every paragraph's distinctive marker survives somewhere in the chunks.
    joined = " ".join(c.text for c in chunks)
    for i in range(10):
        assert f"Paragraph {i} " in joined
    # Chunk ids are sequential and reference the parent document.
    assert [c.chunk_id for c in chunks] == [f"pubmed:99999999:{i}" for i in range(len(chunks))]


def test_chunk_generic_hard_splits_a_single_paragraph_longer_than_chunk_size():
    huge_paragraph = "x" * 2000
    chunks = _chunk_generic(_generic_doc(huge_paragraph), chunk_size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
