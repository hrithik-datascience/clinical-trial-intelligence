"""PubMed E-utilities — published literature for the Evidence agent.

E-utilities is free but rate-limited to 3 requests/second without an API key
(10 with one), and NCBI asks for a contact email. Both are honoured here:
exceeding the limit gets an IP throttled, which would look like a bug much
later and in a different module.

DEVIATION FROM PLAN (found while building Module 6, not Module 3): esummary
alone returns only citation metadata (title/journal/authors/date) — no
abstract text. That was fine for a document *listing*, but the Evidence agent
needs real text to ground claims against, the same way Module 5's Protocol
agent grounds quotes against source text. A citation with nothing to quote
cannot satisfy HG-2 honestly. Fixed by adding an efetch call that pulls the
actual abstract; esummary is kept for date/journal formatting, which its XML
does not expose as cleanly.

Full article text is still not available (mostly paywalled) — the Evidence
agent is therefore abstract-level, and that limitation is real and reported,
not fixed by this change.
"""

from __future__ import annotations

import os
from datetime import date
from xml.etree import ElementTree

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


def fetch_abstracts(pmids: list[str]) -> dict[str, str]:
    """Real abstract text via efetch, keyed by PMID. Missing/absent abstracts
    (some article types genuinely have none) are simply not in the returned
    dict — callers must check, not assume every PMID has an entry."""
    if not pmids:
        return {}

    with _client() as api:
        response = api.get(
            "/efetch.fcgi",
            params={
                **_common_params(),
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "xml",
            },
        )

    root = ElementTree.fromstring(response.text)
    abstracts: dict[str, str] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        parts = [
            (node.text or "")
            for node in article.findall(".//Abstract/AbstractText")
        ]
        text = " ".join(p.strip() for p in parts if p.strip())
        if text:
            abstracts[pmid_el.text] = text
    return abstracts


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
    abstracts = fetch_abstracts(pmids)  # DEVIATION: added so claims have real text to quote
    documents: list[RawDocument] = []

    for pmid in result.get("uids", []):
        record = result.get(pmid)
        if not record or not record.get("title"):
            continue  # skip malformed entries rather than inventing a title

        authors = [a.get("name", "") for a in record.get("authors", [])]
        author_line = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")
        abstract = abstracts.get(pmid, "")

        text = "\n".join(
            line for line in [
                f"Title: {record['title']}",
                f"Authors: {author_line}" if author_line.strip(" ,") else "",
                f"Journal: {record.get('source', '')}",
                f"Published: {record.get('pubdate', '')}",
                f"Publication types: {', '.join(record.get('pubtype', []))}",
                f"Abstract: {abstract}" if abstract else "",
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
                    "has_abstract": bool(abstract),
                    "abstract_only": True,  # no full article text — stated, not hidden
                },
            )
        )

    return documents


def search_and_fetch(query: str, max_results: int = 10) -> list[RawDocument]:
    return fetch_summaries(search(query, max_results))
