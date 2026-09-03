"""Protocol Agent (Module 5).

Extracts six structured fields from a ClinicalTrials.gov record: phase,
population, primary endpoint, sample size, inclusion criteria, exclusion
criteria.

Two-schema design, and it is the point of this file:

  1. The model is asked for a SIMPLE schema (`_LlmField`) — a value, a status,
     and the exact quote it drew the value from. Nothing about Citation
     objects, chunk ids, or retrieval — the model cannot know those.
  2. Python then builds the real `ProtocolExtraction` (Module 2's schema),
     constructing each `Citation` deterministically from the source document
     we already have, and REJECTS any field whose quoted_text does not
     literally appear in the source text.

That second step is a grounding check, and it is the actual enforcement of
HG-1/HG-2 for this agent: the schema validators (Module 2) stop a field from
being malformed, but only this check stops the model from inventing a quote
that sounds plausible but isn't in the document.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.agents.errors import GroundingError, is_grounded
from src.ingest.ctgov import fetch_study
from src.llm import EFFORT_EXTRACTION, get_client, model_name, parse_with_retry
from src.schemas import Citation, ExtractedField, FieldStatus, ProtocolExtraction, SourceType

__all__ = ["GroundingError", "extract"]


# --------------------------------------------------------------------------
# What we ask the model for — deliberately simpler than the real schema
# --------------------------------------------------------------------------


class _LlmField(BaseModel):
    status: Literal["extracted", "not_specified"]
    value: str | None = Field(
        default=None,
        description="The extracted value. Omit (null) if status is not_specified.",
    )
    quoted_text: str | None = Field(
        default=None,
        description=(
            "The exact substring from the source document that supports this "
            "value, copied verbatim. Omit if status is not_specified."
        ),
    )

    @model_validator(mode="after")
    def _shape(self) -> "_LlmField":
        if self.status == "extracted" and not (self.value and self.quoted_text):
            raise ValueError("extracted requires both value and quoted_text")
        if self.status == "not_specified" and (self.value or self.quoted_text):
            raise ValueError("not_specified must not carry a value or quote")
        return self


class _LlmExtraction(BaseModel):
    phase: _LlmField
    population: _LlmField
    primary_endpoint: _LlmField
    sample_size: _LlmField
    inclusion_criteria: _LlmField
    exclusion_criteria: _LlmField


_SYSTEM_PROMPT = """\
You extract structured fields from a clinical trial protocol record for a \
regulatory reviewer. You are extracting, not summarizing or inferring.

Rules, no exceptions:
1. Every value you return MUST be quoted verbatim from the source text \
provided. Copy the exact substring into quoted_text.
2. If a field is genuinely not present in the source text, you MUST return \
status="not_specified" with value=null and quoted_text=null. Do NOT guess, \
infer from context, or use general medical knowledge to fill a gap.
3. Never paraphrase a quote to make it fit — if you cannot find an exact \
substring, the field is not_specified.

A missing field reported honestly is correct behaviour. A plausible-sounding \
guess is the single worst thing you can do here."""


def _build_prompt(doc_text: str, nct_id: str) -> str:
    return (
        f"Source document for {nct_id} (from ClinicalTrials.gov):\n\n"
        f"{doc_text}\n\n"
        "Extract: phase, population, primary_endpoint, sample_size, "
        "inclusion_criteria, exclusion_criteria."
    )


def _to_extracted_field(
    llm_field: _LlmField, doc_text: str, source_name: str, retrieved: date, doc_id: str,
) -> ExtractedField:
    if llm_field.status == "not_specified":
        return ExtractedField.absent()

    # The grounding check: the model's quote must actually be in the source
    # (whitespace-normalized — see src/agents/errors.py for why).
    if not is_grounded(llm_field.quoted_text, doc_text):
        raise GroundingError(
            f"model claimed a quote not present in the source document: "
            f"{llm_field.quoted_text!r}"
        )

    citation = Citation(
        # Whole-document id, not a KB chunk id: this agent extracts from one
        # registry record it fetched live, so there is no retrieval step and
        # nothing to point a chunk id at. Module 10's citation check resolves
        # ids of this shape by source scheme (T-22).
        chunk_id=doc_id,
        source_type=SourceType.PROTOCOL,
        source_name=source_name,
        retrieved_date=retrieved,
        quoted_text=llm_field.quoted_text,
    )
    return ExtractedField(value=llm_field.value, status=FieldStatus.EXTRACTED, citation=citation)


def extract(nct_id: str) -> ProtocolExtraction:
    """Fetch a study and extract its structured fields.

    Raises NoDataFound (from ctgov) if the registry has no such study, and
    GroundingError if the model's response cannot be reconciled with the
    source document. Neither is caught here — the caller decides how to
    surface a failed extraction.
    """
    doc = fetch_study(nct_id)
    client = get_client()

    response = parse_with_retry(
        client,
        model=model_name(),
        max_tokens=4000,
        output_config={"effort": EFFORT_EXTRACTION},
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(doc.text, nct_id)}],
        output_format=_LlmExtraction,
    )
    llm_result = response.parsed_output

    kwargs = {
        field: _to_extracted_field(
            getattr(llm_result, field), doc.text, doc.source_name, doc.retrieved_date, doc.doc_id
        )
        for field in (
            "phase", "population", "primary_endpoint",
            "sample_size", "inclusion_criteria", "exclusion_criteria",
        )
    }
    return ProtocolExtraction(nct_id=nct_id, **kwargs)
