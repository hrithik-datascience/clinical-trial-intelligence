"""Splits an FDA/ICH guidance document into numbered clauses.

DEVIATION FROM PLAN, decided before writing this module: Module 4 (Knowledge
Base — chunking, embeddings, FAISS/BM25) does not exist yet, so the
Regulatory agent has no retrieval layer to lean on. Rather than pass all
~140K characters of ICH E9 to the model on every call (expensive, and it
buries the relevant clause in noise), this module does deterministic
keyword-based clause selection: split the guidance into real numbered
sections (regex on "2.2.1 Title" headers, verified against live text), then
score each clause's relevance to the protocol under review by keyword
overlap. This is a stopgap, not a design decision — when Module 4 exists,
`select_relevant_clauses()` should be replaced by real hybrid retrieval and
this whole file becomes unnecessary. Logged as T-12 in the Technique Log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CLAUSE_HEADER = re.compile(
    r"\n(\d\.\d+(?:\.\d+)?)\s+([A-Z][A-Za-z][A-Za-z /\-,]{2,55})\s*\n"
)
# Repeated running header/footer text that pollutes every page of the PDF
# extraction and would otherwise leak into quoted_text.
_PAGE_NOISE = re.compile(
    r"\n?Statistical Principles for Clinical Trials\s*\n?|\n\d{1,3}\s*\n(?=\S)"
)


@dataclass(frozen=True)
class Clause:
    clause_id: str
    title: str
    text: str


def _clean(text: str) -> str:
    return _PAGE_NOISE.sub(" ", text).strip()


def split_into_clauses(document_text: str, min_body_chars: int = 40) -> list[Clause]:
    """Deterministic split on numbered section headers.

    Skips the table of contents by requiring each clause to be followed by
    real body text (min_body_chars) before the next header — a TOC entry is
    immediately followed by the next TOC entry, so its "body" is ~0 chars.
    """
    matches = list(_CLAUSE_HEADER.finditer(document_text))
    clauses: list[Clause] = []

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(document_text)
        body = _clean(document_text[start:end])
        if len(body) < min_body_chars:
            continue  # TOC entry or empty section, not real content
        clauses.append(Clause(clause_id=match.group(1), title=match.group(2).strip(), text=body))

    return clauses


_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "are", "be",
    "this", "that", "with", "as", "on", "by", "should", "may", "trial", "trials",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def select_relevant_clauses(clauses: list[Clause], query_text: str, top_k: int = 6) -> list[Clause]:
    """Keyword-overlap ranking. Deliberately simple and named as a stopgap —
    see module docstring. Returns [] if nothing overlaps at all, rather than
    padding with irrelevant clauses to hit top_k."""
    query_kw = _keywords(query_text)
    if not query_kw:
        return []

    scored = [
        (len(query_kw & _keywords(f"{c.title} {c.text}")), c)
        for c in clauses
    ]
    scored = [(score, c) for score, c in scored if score > 0]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in scored[:top_k]]
