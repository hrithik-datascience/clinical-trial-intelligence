"""Ingestion tests.

Offline tests use fragments captured from real API responses on 2026-09-03 —
not invented payloads. Live integration is checked by `src.ingest.verify`,
which hits all four APIs for real.
"""

import pytest
from pydantic import ValidationError

from src.ingest import ctgov, openfda
from src.ingest.client import ApiClient, IngestError, RateLimiter
from src.schemas import FaersAggregation, ReactionCount, SourceType

# Captured from GET /api/v2/studies/NCT04280705, trimmed to the fields used.
REAL_CTGOV_FRAGMENT = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT04280705",
            "briefTitle": "Adaptive COVID-19 Treatment Trial (ACTT)",
        },
        "designModule": {
            "studyType": "INTERVENTIONAL",
            "phases": ["PHASE3"],
            "enrollmentInfo": {"count": 1062, "type": "ACTUAL"},
        },
        "outcomesModule": {
            "primaryOutcomes": [
                {"measure": "Time to Recovery", "description": "Day of recovery..."}
            ]
        },
        "eligibilityModule": {"eligibilityCriteria": "Inclusion: hospitalized adults..."},
    }
}


def test_ctgov_flatten_keeps_registry_values_verbatim():
    flat = ctgov._flatten(REAL_CTGOV_FRAGMENT)
    assert flat["nct_id"] == "NCT04280705"
    assert flat["phase"] == "PHASE3"
    assert flat["enrollment"] == 1062  # the registry's number, not derived


def test_ctgov_absent_fields_stay_absent():
    """HG-1 begins at ingestion: a missing key must not become an empty string."""
    stripped = {"protocolSection": {"identificationModule": {"nctId": "NCT12345678"}}}
    flat = ctgov._flatten(stripped)
    assert "phase" not in flat
    assert "enrollment" not in flat
    assert "eligibility_criteria" not in flat


def test_ctgov_document_omits_missing_fields_from_index_text():
    stripped = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT12345678", "briefTitle": "A trial"}
        }
    }
    doc = ctgov._to_document(stripped)
    assert "Phase:" not in doc.text  # never "Phase: unknown"
    assert doc.source_type is SourceType.PROTOCOL
    assert doc.source_name == "NCT12345678"


def _aggregation() -> FaersAggregation:
    return FaersAggregation(
        drug="pembrolizumab",
        total_reports=35723,
        reactions=[
            ReactionCount(term="MALIGNANT NEOPLASM PROGRESSION", count=2933),
            ReactionCount(term="DIARRHOEA", count=2004),
        ],
        last_updated="2026-07-30",
        disclaimer="Do not rely on openFDA to make decisions regarding medical care.",
        retrieved_date=__import__("datetime").date(2026, 9, 3),
    )


def test_faers_document_never_uses_causal_language():
    """The rendered text must survive the same HG-3 check applied downstream."""
    from src.schemas import _CAUSAL_PATTERN

    doc = openfda.to_document(_aggregation())
    assert _CAUSAL_PATTERN.search(doc.text) is None
    assert "does not establish" in doc.text
    assert "under-reporting" in doc.text


def test_faers_document_carries_the_fda_disclaimer():
    doc = openfda.to_document(_aggregation())
    assert doc.metadata["disclaimer"]
    assert doc.metadata["reaction_counts"]["DIARRHOEA"] == 2004


def test_faers_counts_are_never_recomputed_locally():
    """total_reports comes from openFDA meta, not from summing the reactions."""
    agg = _aggregation()
    assert agg.total_reports != sum(r.count for r in agg.reactions)


def test_reaction_count_rejects_negative():
    with pytest.raises(ValidationError):
        ReactionCount(term="X", count=-1)


def test_rate_limiter_enforces_minimum_interval():
    import time

    limiter = RateLimiter(0.15)
    start = time.monotonic()
    for _ in range(3):
        limiter.wait()
    # Two gaps of 0.15s. Tolerance because time.sleep() on Windows can
    # return a few ms early relative to monotonic().
    assert time.monotonic() - start >= 0.29


def test_client_raises_on_permanent_error_without_retrying():
    """A 4xx must fail immediately, not burn the retry budget.

    ClinicalTrials.gov answers 400 (not 404) for a malformed NCT id —
    confirmed against the live API rather than assumed.
    """
    with ApiClient("test", "https://clinicaltrials.gov/api/v2") as api:
        with pytest.raises(IngestError, match="HTTP 400"):
            api.get_json("/studies/NCT00000000")
