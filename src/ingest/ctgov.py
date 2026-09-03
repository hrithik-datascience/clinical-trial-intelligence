"""ClinicalTrials.gov v2 API — protocol records.

Field paths below were read off a live response for NCT04280705, not assumed.
Anything the registry omits stays omitted: this module never substitutes a
default, because a missing field has to reach the Protocol agent as missing
(HG-1) rather than as a plausible value invented during ingestion.
"""

from __future__ import annotations

from datetime import date

from src.ingest.client import ApiClient, NoDataFound
from src.schemas import RawDocument, SourceType

BASE_URL = "https://clinicaltrials.gov/api/v2"
STUDY_URL = "https://clinicaltrials.gov/study/{nct_id}"


def _client() -> ApiClient:
    # No published hard rate limit; 0.2s keeps us obviously polite.
    return ApiClient("ClinicalTrials.gov", BASE_URL, min_interval_s=0.2)


def _join(values: list[str] | None, sep: str = ", ") -> str:
    return sep.join(v for v in (values or []) if v)


def _flatten(study: dict) -> dict:
    """Pull the fields the Protocol agent cares about into a flat dict.

    Absent keys stay absent — callers must handle a missing key, not receive
    an empty string that reads like a real answer.
    """
    section = study.get("protocolSection", {})
    ident = section.get("identificationModule", {})
    design = section.get("designModule", {})
    elig = section.get("eligibilityModule", {})
    outcomes = section.get("outcomesModule", {})
    conditions = section.get("conditionsModule", {})

    flat: dict = {"nct_id": ident.get("nctId", "")}

    if title := ident.get("briefTitle"):
        flat["brief_title"] = title
    if official := ident.get("officialTitle"):
        flat["official_title"] = official
    if phases := design.get("phases"):
        flat["phase"] = _join(phases)
    if study_type := design.get("studyType"):
        flat["study_type"] = study_type
    # enrollmentInfo.count is the registry's own number — never estimated here.
    enrollment = design.get("enrollmentInfo", {})
    if (count := enrollment.get("count")) is not None:
        flat["enrollment"] = count
        flat["enrollment_type"] = enrollment.get("type", "")
    if conds := conditions.get("conditions"):
        flat["conditions"] = _join(conds)
    if primary := outcomes.get("primaryOutcomes"):
        flat["primary_outcomes"] = [
            {"measure": o.get("measure", ""), "description": o.get("description", "")}
            for o in primary
        ]
    if criteria := elig.get("eligibilityCriteria"):
        flat["eligibility_criteria"] = criteria
    for key in ("sex", "minimumAge", "maximumAge", "healthyVolunteers"):
        if (value := elig.get(key)) is not None:
            flat[key] = value

    return flat


def _to_document(study: dict) -> RawDocument:
    flat = _flatten(study)
    nct_id = flat.get("nct_id")
    if not nct_id:
        raise NoDataFound("study record carried no nctId")

    # Human-readable text for the shared index. Only fields that were actually
    # present are rendered, so the index never contains "Phase: unknown".
    lines = [f"Title: {flat.get('brief_title', '')}"]
    for label, key in (
        ("Study type", "study_type"),
        ("Phase", "phase"),
        ("Conditions", "conditions"),
    ):
        if key in flat:
            lines.append(f"{label}: {flat[key]}")
    if "enrollment" in flat:
        lines.append(f"Enrollment: {flat['enrollment']} ({flat.get('enrollment_type','')})")
    for outcome in flat.get("primary_outcomes", []):
        lines.append(f"Primary outcome: {outcome['measure']}")
        if outcome["description"]:
            lines.append(f"  {outcome['description']}")
    if "eligibility_criteria" in flat:
        lines.append(f"Eligibility criteria:\n{flat['eligibility_criteria']}")

    return RawDocument(
        doc_id=f"ctgov:{nct_id}",
        source_type=SourceType.PROTOCOL,
        source_name=nct_id,
        title=flat.get("brief_title") or nct_id,
        text="\n".join(lines),
        url=STUDY_URL.format(nct_id=nct_id),
        retrieved_date=date.today(),
        metadata=flat,
    )


def fetch_study(nct_id: str) -> RawDocument:
    """One study by NCT id. Raises NoDataFound if the registry has no record."""
    with _client() as api:
        try:
            study = api.get_json(f"/studies/{nct_id}")
        except Exception as exc:
            if "404" in str(exc):
                raise NoDataFound(f"no ClinicalTrials.gov record for {nct_id}") from exc
            raise
    return _to_document(study)


def search_studies(condition: str, max_results: int = 10) -> list[RawDocument]:
    """Search by condition. Returns [] only when the registry itself returns none."""
    with _client() as api:
        payload = api.get_json(
            "/studies",
            params={
                "query.cond": condition,
                "pageSize": min(max_results, 100),
                "countTotal": "true",
            },
        )
    return [_to_document(s) for s in payload.get("studies", [])]
