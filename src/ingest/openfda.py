"""openFDA FAERS — spontaneous adverse-event reports.

Two rules this module exists to enforce:

1. **Counts come from FDA, not from us and not from the model.** The `count`
   parameter makes openFDA do the aggregation server-side and return reaction
   terms with report totals. Nothing downstream tallies reports by hand.

2. **The disclaimer travels with the data.** Every openFDA response carries one,
   and it is attached to the aggregation object so it can reach the UI. FAERS is
   voluntary spontaneous-report data: under-reporting and reporting bias are
   structural, and a count is a count of *reports*, not of events.

openFDA answers 404 when a search matches nothing. That is a finding, not a
failure, so it is raised as NoDataFound.
"""

from __future__ import annotations

from datetime import date

from src.ingest.client import ApiClient, IngestError, NoDataFound
from src.schemas import (
    DrugLabel,
    FaersAggregation,
    RawDocument,
    ReactionCount,
    SourceType,
)

BASE_URL = "https://api.fda.gov"
EVENT_PATH = "/drug/event.json"
LABEL_PATH = "/drug/label.json"
REACTION_FIELD = "patient.reaction.reactionmeddrapt.exact"
# Sections most likely to state a known/labeled risk, in priority order.
_LABEL_SECTIONS = ("boxed_warning", "warnings", "warnings_and_cautions", "adverse_reactions")


def _client() -> ApiClient:
    # 240 requests/min per IP without a key -> 0.25s is comfortably inside it.
    return ApiClient("openFDA", BASE_URL, min_interval_s=0.25)


def _search_term(drug: str) -> str:
    escaped = drug.replace('"', "")
    return f'patient.drug.medicinalproduct:"{escaped}"'


def fetch_reaction_counts(drug: str, limit: int = 25) -> FaersAggregation:
    """Top reported reactions for a drug, counted by FDA.

    Raises NoDataFound when FAERS holds no reports for the drug — the Safety
    agent must be able to say "no reports found", which is a real answer.
    """
    with _client() as api:
        try:
            counts = api.get_json(
                EVENT_PATH,
                params={"search": _search_term(drug), "count": REACTION_FIELD,
                        "limit": limit},
            )
        except IngestError as exc:
            if "404" in str(exc):
                raise NoDataFound(f"no FAERS reports for {drug!r}") from exc
            raise

        # Second call: the total report count for the drug. count-mode responses
        # do not carry meta.results.total, so it has to be asked for separately.
        totals = api.get_json(
            EVENT_PATH, params={"search": _search_term(drug), "limit": 1}
        )

    meta = counts.get("meta", {})
    results = counts.get("results", [])
    if not results:
        raise NoDataFound(f"no FAERS reactions for {drug!r}")

    return FaersAggregation(
        drug=drug,
        total_reports=totals.get("meta", {}).get("results", {}).get("total", 0),
        reactions=[
            ReactionCount(term=r["term"], count=r["count"])
            for r in results
            if "term" in r and "count" in r
        ],
        last_updated=meta.get("last_updated", ""),
        disclaimer=meta.get("disclaimer", ""),
        retrieved_date=date.today(),
    )


def fetch_label(drug: str) -> DrugLabel:
    """The real FDA-approved label for a drug, used by the Safety agent
    (Module 8) to determine known_label_risk deterministically.

    Prefers an exact openfda.generic_name match over a partial one. Verified
    live against pembrolizumab: a partial match returns "KEYTRUDA QLEX" (a
    newer pembrolizumab + berahyaluronidase combination product) ahead of
    plain "KEYTRUDA" -- the wrong reference label for a plain-pembrolizumab
    safety screen. Falls back to a partial match only if no exact one exists
    (e.g. the input is a brand name, not a generic name).
    """
    with _client() as api:
        payload = _label_search(api, f'openfda.generic_name.exact:"{drug.upper()}"')
        if not payload.get("results"):
            payload = _label_search(api, f'openfda.generic_name:"{drug}"')

    results = payload.get("results", [])
    if not results:
        raise NoDataFound(f"no FDA label found for {drug!r}")

    result = results[0]
    ofda = result.get("openfda", {})
    sections = [
        para
        for key in _LABEL_SECTIONS
        for para in result.get(key, [])
    ]
    reference_text = "\n\n".join(sections)
    if not reference_text:
        raise NoDataFound(f"label found for {drug!r} but it has no warnings/adverse-reaction text")

    return DrugLabel(
        drug=drug,
        brand_name=", ".join(ofda.get("brand_name", [])) or drug,
        generic_name=", ".join(ofda.get("generic_name", [])) or drug,
        reference_text=reference_text,
        retrieved_date=date.today(),
    )


def _label_search(api: ApiClient, search: str) -> dict:
    try:
        return api.get_json(LABEL_PATH, params={"search": search, "limit": 1})
    except IngestError as exc:
        if "404" in str(exc):
            return {}
        raise


def to_document(aggregation: FaersAggregation) -> RawDocument:
    """Render an aggregation for the shared index.

    The wording here is load-bearing. Every line says "reports" or "co-reported",
    never anything implying the drug produced the event — the phrasing that the
    HG-3 validator on ReportedAssociation would reject downstream.
    """
    lines = [
        f"FAERS spontaneous adverse event reports for {aggregation.drug}.",
        f"Total reports in FAERS mentioning this product: {aggregation.total_reports}.",
        "",
        "Most frequently co-reported reaction terms (MedDRA preferred terms):",
    ]
    lines += [
        f"  {r.term}: {r.count} reports" for r in aggregation.reactions
    ]
    lines += [
        "",
        "These are counts of voluntary reports, not confirmed adverse reactions. "
        "FAERS is subject to under-reporting and reporting bias, and a report "
        "does not establish that the product produced the event.",
    ]

    return RawDocument(
        doc_id=f"faers:{aggregation.drug.lower().replace(' ', '_')}",
        source_type=SourceType.FAERS,
        source_name=f"openFDA FAERS ({aggregation.drug})",
        title=f"FAERS reported reactions — {aggregation.drug}",
        text="\n".join(lines),
        url=f"https://api.fda.gov/drug/event.json?search={_search_term(aggregation.drug)}",
        retrieved_date=aggregation.retrieved_date,
        metadata={
            "drug": aggregation.drug,
            "total_reports": aggregation.total_reports,
            "last_updated": aggregation.last_updated,
            "disclaimer": aggregation.disclaimer,
            "reaction_counts": {r.term: r.count for r in aggregation.reactions},
        },
    )
