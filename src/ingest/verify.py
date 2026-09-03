"""Module 3 verification: pull from all four live sources and report honestly.

Run:  venv/Scripts/python.exe -m src.ingest.verify

This is the evidence behind Module 3's STATUS row. It prints what was actually
retrieved, including anything that failed — a failed source is reported as
failed, never skipped over to keep the summary looking clean.
"""

from __future__ import annotations

import sys

from src.ingest import ctgov, guidance, openfda, pubmed
from src.ingest.client import IngestError, NoDataFound

# ACTUAL/enrolled trial used only as a known-good record for verification.
DEMO_NCT = "NCT04280705"
# ASSUMPTION: pembrolizumab is a placeholder with high FAERS volume, chosen so
# ingestion can be verified before the real demo drug is picked (Module 8).
DEMO_DRUG = "pembrolizumab"
DEMO_QUERY = "progression-free survival surrogate endpoint oncology"


def _line(label: str, status: str, detail: str) -> None:
    print(f"  {label:<24} {status:<8} {detail}")


def main() -> int:
    print("Module 3 — live ingestion check")
    print("=" * 74)
    failures = 0

    # 1. ClinicalTrials.gov
    print("\nClinicalTrials.gov v2")
    try:
        doc = ctgov.fetch_study(DEMO_NCT)
        meta = doc.metadata
        _line("record", "OK", f"{doc.source_name} — {doc.title[:44]}")
        _line("phase", "OK", meta.get("phase", "(absent)"))
        _line("enrollment", "OK", str(meta.get("enrollment", "(absent)")))
        _line("primary outcomes", "OK", str(len(meta.get("primary_outcomes", []))))
        _line("text length", "OK", f"{len(doc.text):,} chars")
    except (IngestError, NoDataFound) as exc:
        failures += 1
        _line("record", "FAILED", str(exc)[:60])

    # 2. openFDA FAERS
    print(f"\nopenFDA FAERS — {DEMO_DRUG} (placeholder drug)")
    try:
        agg = openfda.fetch_reaction_counts(DEMO_DRUG, limit=5)
        _line("total reports", "OK", f"{agg.total_reports:,}")
        _line("last updated", "OK", agg.last_updated)
        for reaction in agg.reactions[:3]:
            _line(f"  {reaction.term[:20]}", "OK", f"{reaction.count:,} reports")
        doc = openfda.to_document(agg)
        _line("disclaimer kept", "OK", "yes" if agg.disclaimer else "MISSING")
    except (IngestError, NoDataFound) as exc:
        failures += 1
        _line("aggregation", "FAILED", str(exc)[:60])
    try:
        label = openfda.fetch_label(DEMO_DRUG)
        _line("label", "OK", f"{label.brand_name} ({len(label.reference_text):,} chars)")
    except (IngestError, NoDataFound) as exc:
        failures += 1
        _line("label", "FAILED", str(exc)[:60])

    # 3. PubMed
    print("\nPubMed E-utilities")
    try:
        docs = pubmed.search_and_fetch(DEMO_QUERY, max_results=3)
        _line("articles", "OK", f"{len(docs)} retrieved")
        for d in docs:
            _line(f"  {d.source_name}", "OK",
                  f"{d.metadata['journal']} {d.metadata['pubdate']} — {d.title[:34]}")
    except (IngestError, NoDataFound) as exc:
        failures += 1
        _line("search", "FAILED", str(exc)[:60])

    # 4. FDA / ICH guidance
    print("\nFDA / ICH guidance PDFs")
    for name in guidance.GUIDANCE_SOURCES:
        try:
            doc = guidance.fetch_guidance(name)
            _line(name, "OK",
                  f"{doc.metadata['page_count']} pages, {len(doc.text):,} chars, "
                  f"{doc.metadata['version']}")
        except (IngestError, NoDataFound) as exc:
            failures += 1
            _line(name, "FAILED", str(exc)[:60])

    # Error handling actually exercised, not just written
    print("\nError handling")
    try:
        ctgov.fetch_study("NCT00000000")
        _line("bad NCT id", "FAILED", "should have raised, returned a record")
        failures += 1
    except (IngestError, NoDataFound) as exc:
        _line("bad NCT id", "OK", f"raised {type(exc).__name__}")
    try:
        openfda.fetch_reaction_counts("zzzznotadrugzzzz")
        _line("unknown drug", "FAILED", "should have raised")
        failures += 1
    except NoDataFound:
        _line("unknown drug", "OK", "raised NoDataFound (a finding, not an error)")
    except IngestError as exc:
        _line("unknown drug", "OK", f"raised IngestError: {str(exc)[:34]}")

    print("\n" + "=" * 74)
    print(f"{'ALL SOURCES OK' if not failures else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
