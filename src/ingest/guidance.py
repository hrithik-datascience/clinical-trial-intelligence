"""FDA / ICH guidance documents — the Regulatory agent's rubric source.

These are stable, versioned public PDFs. Guidance version and date are
themselves metadata worth keeping (Section D.3): a finding cited against E9 is
only meaningful if you know which E9.

PDFs are cached to data/raw/guidance/ so repeat runs do not re-download, and
so an offline demo still works.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pypdf import PdfReader

from src.ingest.client import ApiClient, IngestError
from src.schemas import RawDocument, SourceType

CACHE_DIR = Path("data/raw/guidance")

# Verified reachable 2026-09-03. Each entry records the version we pinned.
GUIDANCE_SOURCES: dict[str, dict[str, str]] = {
    "ICH_E9": {
        "url": "https://database.ich.org/sites/default/files/E9_Guideline.pdf",
        "title": "ICH E9: Statistical Principles for Clinical Trials",
        "version": "Step 4, 1998-02-05",
    },
    "FDA_E9R1": {
        "url": "https://www.fda.gov/media/148473/download",
        "title": "E9(R1) Statistical Principles for Clinical Trials: Estimands "
                 "and Sensitivity Analysis in Clinical Trials",
        "version": "FDA final guidance, May 2021",
    },
}


def _download(name: str, url: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.pdf"
    if path.exists() and path.stat().st_size > 0:
        return path

    with ApiClient("Guidance", min_interval_s=0.5, timeout_s=60.0) as api:
        response = api.get(url)

    content_type = response.headers.get("content-type", "")
    if "pdf" not in content_type.lower():
        raise IngestError(f"{name}: expected a PDF, got {content_type!r}")

    path.write_bytes(response.content)
    return path


def _extract_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise IngestError(f"{path.name}: PDF produced no extractable text")
    return text


def fetch_guidance(name: str) -> RawDocument:
    """Download (or reuse cached) guidance and extract its text."""
    if name not in GUIDANCE_SOURCES:
        raise IngestError(f"unknown guidance {name!r}; known: {list(GUIDANCE_SOURCES)}")

    source = GUIDANCE_SOURCES[name]
    path = _download(name, source["url"])
    text = _extract_text(path)

    return RawDocument(
        doc_id=f"guidance:{name}",
        source_type=SourceType.GUIDANCE,
        source_name=name.replace("_", " "),
        title=source["title"],
        text=text,
        url=source["url"],
        retrieved_date=date.today(),
        metadata={
            "version": source["version"],  # which E9 a finding was cited against
            "page_count": len(PdfReader(str(path)).pages),
            "cached_path": str(path),
        },
    )


def fetch_all() -> list[RawDocument]:
    return [fetch_guidance(name) for name in GUIDANCE_SOURCES]
