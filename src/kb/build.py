"""Module 4 build + verification: fetch real documents from all four live
ingestion sources, build the shared knowledge base, persist it, and run
sample queries to check retrieval quality before other modules depend on it.

Run: venv/Scripts/python.exe -m src.kb.build
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.ingest import ctgov, guidance, openfda, pubmed
from src.kb.store import KnowledgeBase
from src.schemas import RawDocument, SourceType

KB_DIR = Path("data/index")

# Same demo identifiers Module 3 already verified live -- keeps this corpus
# traceable to confirmed-working sources rather than picking new ones untested.
DEMO_NCT_IDS = ["NCT04280705"]
DEMO_DRUG = "pembrolizumab"  # ASSUMPTION A-06 -- placeholder pending Module 8
LITERATURE_QUERIES = [
    "progression-free survival surrogate endpoint oncology",
    "adaptive platform trial design",
]

# (query, source_type filter or None, at least one chunk's source_name/title
# substring expected to appear in the top result -- the honest pass/fail bar)
SAMPLE_QUERIES: list[tuple[str, SourceType | None]] = [
    ("interim analysis and multiplicity adjustment", SourceType.GUIDANCE),
    ("estimand and sensitivity analysis in clinical trials", SourceType.GUIDANCE),
    ("is progression-free survival a valid surrogate for overall survival", SourceType.LITERATURE),
    ("COVID-19 hospitalized adults time to recovery", SourceType.PROTOCOL),
    ("pembrolizumab reported adverse reactions", SourceType.FAERS),
]


def fetch_corpus() -> list[RawDocument]:
    docs: list[RawDocument] = []
    docs += guidance.fetch_all()
    docs += [ctgov.fetch_study(nct) for nct in DEMO_NCT_IDS]
    for query in LITERATURE_QUERIES:
        docs += pubmed.search_and_fetch(query, max_results=5)
    docs.append(openfda.to_document(openfda.fetch_reaction_counts(DEMO_DRUG, limit=10)))
    return docs


def build() -> tuple[KnowledgeBase, list[RawDocument]]:
    docs = fetch_corpus()
    kb = KnowledgeBase()
    kb.add_documents(docs)
    kb.save(KB_DIR)
    return kb, docs


def main() -> int:
    print("Module 4 -- building shared knowledge base from live sources")
    print("=" * 74)

    kb, docs = build()
    by_type: dict[str, int] = {}
    for d in docs:
        by_type[d.source_type.value] = by_type.get(d.source_type.value, 0) + 1
    print(f"\n{len(docs)} documents -> {len(kb)} chunks, saved to {KB_DIR}")
    for source_type, count in sorted(by_type.items()):
        print(f"  {source_type:<12} {count} document(s)")

    print("\nSample query retrieval")
    print("-" * 74)
    failures = 0
    for query, source_type in SAMPLE_QUERIES:
        results = kb.search(query, top_k=3, source_type=source_type)
        label = f"[{source_type.value}] " if source_type else ""
        print(f"\n{label}{query!r}")
        if not results:
            print("  NO RESULTS")
            failures += 1
            continue
        for r in results:
            print(f"  {r.score:.4f}  {r.chunk.chunk_id:<40} {r.chunk.title[:50]}")

    print("\n" + "=" * 74)
    print("NO RESULTS on 1+ query" if failures else "ALL SAMPLE QUERIES RETURNED RESULTS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
