"""Splits RawDocuments into indexable Chunks (T-15).

Guidance documents (ICH/FDA PDFs) keep the clause-based split first built in
Module 7 as a stopgap (src/agents/clauses.py, now retired) -- verified
against live ICH E9 text: 50 real clauses parsed, table-of-contents entries
correctly excluded. It produces exactly the clause-numbered chunk_id that
regulatory citations already depend on, and generic fixed-size chunking would
cut across that structure for no benefit. Everything else (protocol records,
literature abstracts, FAERS aggregations) has no comparable numbered
structure, so it gets paragraph-aware recursive character chunking instead.
"""

from __future__ import annotations

import re

from src.schemas import Chunk, RawDocument, SourceType

_CLAUSE_HEADER = re.compile(
    r"\n(\d\.\d+(?:\.\d+)?)\s+([A-Z][A-Za-z][A-Za-z /\-,]{2,55})\s*\n"
)
# Repeated running header/footer text that pollutes every page of the PDF
# extraction and would otherwise leak into quoted_text.
_PAGE_NOISE = re.compile(
    r"\n?Statistical Principles for Clinical Trials\s*\n?|\n\d{1,3}\s*\n(?=\S)"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _clean_guidance_noise(text: str) -> str:
    return _PAGE_NOISE.sub(" ", text).strip()


def _chunk_guidance(doc: RawDocument, min_body_chars: int = 40) -> list[Chunk]:
    """One chunk per real numbered clause.

    Skips the table of contents by requiring real body text before the next
    header -- a TOC entry is immediately followed by the next TOC entry, so
    its "body" is ~0 chars.
    """
    matches = list(_CLAUSE_HEADER.finditer(doc.text))
    guidance_key = doc.doc_id.split(":", 1)[1] if ":" in doc.doc_id else doc.doc_id
    chunks: list[Chunk] = []

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc.text)
        body = _clean_guidance_noise(doc.text[start:end])
        if len(body) < min_body_chars:
            continue  # TOC entry or empty section, not real content

        clause_id, title = match.group(1), match.group(2).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{clause_id}",
                doc_id=doc.doc_id,
                source_type=doc.source_type,
                source_name=f"{doc.source_name} {clause_id}",
                title=title,
                text=body,
                retrieved_date=doc.retrieved_date,
                metadata={**doc.metadata, "clause_id": clause_id, "guidance_key": guidance_key},
            )
        )
    return chunks


def _chunk_generic(
    doc: RawDocument, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """Pack whole paragraphs up to chunk_size; split only a paragraph that
    alone exceeds it. `overlap` trailing chars carry into the next chunk so a
    fact split across a chunk boundary isn't lost to either side."""
    paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [doc.text.strip()]

    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = (current[-overlap:] + "\n\n" + para) if overlap else para
        else:
            # a single paragraph longer than chunk_size: hard-split it
            step = max(chunk_size - overlap, 1)
            for start in range(0, len(para), step):
                pieces.append(para[start : start + chunk_size])
            current = ""
    if current:
        pieces.append(current)

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}:{i}",
            doc_id=doc.doc_id,
            source_type=doc.source_type,
            source_name=doc.source_name,
            title=doc.title,
            text=piece,
            retrieved_date=doc.retrieved_date,
            metadata=doc.metadata,
        )
        for i, piece in enumerate(pieces)
    ]


def chunk_document(doc: RawDocument) -> list[Chunk]:
    """Route to the right chunking strategy for this document's source_type."""
    if doc.source_type is SourceType.GUIDANCE:
        clause_chunks = _chunk_guidance(doc)
        if clause_chunks:
            return clause_chunks
        # no numbered-clause structure found -- fall back rather than index nothing
    return _chunk_generic(doc)
