"""PubMed E-utilities — published literature for the Evidence agent.

E-utilities is free but rate-limited to 3 requests/second without an API key
(10 with one), and NCBI asks for a contact email. Both are honoured here:
exceeding the limit gets an IP throttled, which would look like a bug much
later and in a different module.

Only metadata is fetched (esearch + esummary). Full text is frequently
paywalled, so the Evidence agent cites title/journal/year — and Module 12 must
report that this is an abstract-level, not full-text, evidence base.
"""

from __future__ import annotations

import os
from datetime import date

from src.ingest.client import ApiClient
from src.schemas import RawDocument, SourceType

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def _client() -> ApiClient:
    # 3 req/s without a key -> 0.35s interval leaves headroom.
    return ApiClient("PubMed", BASE_URL, min_interval_s=0.35)


def _common_params() -> dict:
    params = {"tool": "clinical-trial-intelligence"}
    if email := os.getenv("PUBMED_CONTACT_EMAIL"):
        params["email"] = email  # NCBI asks for this; it is not required
    if key := os.getenv("PUBMED_API_KEY"):
        params["api_key"] = key
    return params


def search(query: str, max_results: int = 10) -> list[str]:
    """Return PMIDs for a query, most relevant first."""
    with _client() as api:
        payload = api.get_json(
            "/esearch.fcgi",
            params={
                **_common_params(),
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            },
        )
    return payload.get("esearchresult", {}).get("idlist", [])


def fetch_summaries(pmids: list[str]) -> list[RawDocument]:
    """Metadata for a list of PMIDs. An empty input is an empty result."""
    if not pmids:
        return []

    with _client() as api:
        payload = api.get_json(
            "/esummary.fcgi",
            params={
                **_common_params(),
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
            },
        )

    result = payload.get("result", {})
    documents: list[RawDocument] = []

    for pmid in result.get("uids", []):
        record = result.get(pmid)
        if not record or not record.get("title"):
            continue  # skip malformed entries rather than inventing a title

        authors = [a.get("name", "") for a in record.get("authors", [])]
        author_line = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")

        text = "\n".join(
            line for line in [
                f"Title: {record['title']}",
                f"Authors: {author_line}" if author_line.strip(" ,") else "",
                f"Journal: {record.get('source', '')}",
                f"Published: {record.get('pubdate', '')}",
                f"Publication types: {', '.join(record.get('pubtype', []))}",
            ] if line
        )

        documents.append(
            RawDocument(
                doc_id=f"pubmed:{pmid}",
                source_type=SourceType.LITERATURE,
                source_name=f"PMID {pmid}",
                title=record["title"],
                text=text,
                url=ARTICLE_URL.format(pmid=pmid),
                retrieved_date=date.today(),
                metadata={
                    "pmid": pmid,
                    "journal": record.get("source", ""),
                    "pubdate": record.get("pubdate", ""),
                    "authors": authors,
                    "pubtype": record.get("pubtype", []),
                    "abstract_only": True,  # no full text — stated, not hidden
                },
            )
        )

    return documents


def search_and_fetch(query: str, max_results: int = 10) -> list[RawDocument]:
    return fetch_summaries(search(query, max_results))
