"""KnowledgeBase tests — real local embedding model (sentence-transformers),
no mocked encoder and no Anthropic API key required. Model-scoped fixture so
the model loads once per test session, not once per test."""

from datetime import date

import pytest

from src.kb.store import KnowledgeBase
from src.schemas import RawDocument, SourceType

DOCS = [
    RawDocument(
        doc_id="guidance:ICH_E9",
        source_type=SourceType.GUIDANCE,
        source_name="ICH E9",
        title="ICH E9 guidance",
        text=(
            "\nStatistical Principles for Clinical Trials\n"
            "4.5 Interim Analysis\n"
            "All interim analyses should be carefully planned in advance and "
            "described in the protocol, together with multiplicity adjustment "
            "procedures for the primary endpoint and any secondary endpoints "
            "considered, spanning well past the minimum body length filter.\n\n"
            "4.6 Data Monitoring Committee\n"
            "An independent data monitoring committee may be established to "
            "assess safety and efficacy at predetermined intervals during the "
            "trial, again long enough to clear the minimum body length filter.\n"
        ),
        url="https://example.org/e9",
        retrieved_date=date(2026, 9, 3),
    ),
    RawDocument(
        doc_id="pubmed:11111111",
        source_type=SourceType.LITERATURE,
        source_name="PMID 11111111",
        title="Progression-free survival as a surrogate for overall survival",
        text=(
            "This systematic review examines whether progression-free survival "
            "is a valid surrogate endpoint for overall survival across multiple "
            "oncology indications, finding validation must be assessed per "
            "cancer type rather than assumed to generalize."
        ),
        url="https://pubmed.ncbi.nlm.nih.gov/11111111/",
        retrieved_date=date(2026, 9, 3),
    ),
    RawDocument(
        doc_id="faers:pembrolizumab",
        source_type=SourceType.FAERS,
        source_name="openFDA FAERS (pembrolizumab)",
        title="FAERS reported reactions - pembrolizumab",
        text=(
            "FAERS spontaneous adverse event reports for pembrolizumab. "
            "Most frequently co-reported reaction terms include diarrhoea and "
            "malignant neoplasm progression. These are counts of voluntary "
            "reports, not confirmed adverse reactions."
        ),
        url="https://api.fda.gov/drug/event.json",
        retrieved_date=date(2026, 9, 3),
    ),
]


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    store = KnowledgeBase()
    added = store.add_documents(DOCS)
    assert added == len(store)
    return store


def test_search_ranks_the_relevant_chunk_first(kb: KnowledgeBase):
    results = kb.search("interim analysis multiplicity adjustment", top_k=3)
    assert results
    assert results[0].chunk.metadata.get("clause_id") == "4.5"


def test_search_filters_to_the_requested_source_type(kb: KnowledgeBase):
    results = kb.search("progression-free survival overall survival", top_k=5, source_type=SourceType.LITERATURE)
    assert results
    assert all(r.chunk.source_type is SourceType.LITERATURE for r in results)


def test_search_source_type_filter_excludes_other_types(kb: KnowledgeBase):
    results = kb.search("pembrolizumab adverse reactions", top_k=5, source_type=SourceType.GUIDANCE)
    assert all(r.chunk.source_type is SourceType.GUIDANCE for r in results)
    # The FAERS chunk should not leak into a guidance-only result set.
    assert not any(r.chunk.source_type is SourceType.FAERS for r in results)


def test_empty_knowledge_base_returns_no_results():
    empty = KnowledgeBase()
    assert empty.search("anything", top_k=5) == []


def test_save_and_load_round_trip_preserves_search(kb: KnowledgeBase, tmp_path):
    kb.save(tmp_path)
    reloaded = KnowledgeBase.load(tmp_path)

    assert len(reloaded) == len(kb)
    original = kb.search("interim analysis multiplicity adjustment", top_k=1)
    restored = reloaded.search("interim analysis multiplicity adjustment", top_k=1)
    assert restored[0].chunk.chunk_id == original[0].chunk.chunk_id
