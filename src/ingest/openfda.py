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
    FaersAggregation,
    RawDocument,
    ReactionCount,
    SourceType,
)

BASE_URL = "https://api.fda.gov"
EVENT_PATH = "/drug/event.json"
REACTION_FIELD = "patient.reaction.reactionmeddrapt.exact"


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
