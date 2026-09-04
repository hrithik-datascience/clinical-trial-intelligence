"""Safety Agent (Module 8).

Given a drug name, retrieves real FAERS-reported adverse-event counts from
openFDA -- report_count is FDA's own server-side aggregation; this code and
the model never recompute or estimate it -- and checks each reaction term
against the drug's real FDA-approved label to determine known_label_risk
deterministically in Python (see is_known_label_risk). The model's only job
is to write a plain-language narrative per reported association for a
reviewer: it never touches the adverse_event name, the count, or the
known/unexpected determination, and Module 2's HG-3 schema validator on
ReportedAssociation rejects any narrative that asserts the drug caused the
event -- FAERS is spontaneous-report data, never confirmed causality.

Unlike Modules 5-7, there is no free-text grounding check here: every field
except narrative is computed directly from the API response, not asserted by
the model, so there is nothing for the model to fabricate a quote about.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, ValidationError

from src.ingest.openfda import fetch_label, fetch_reaction_counts
from src.llm import EFFORT_EXTRACTION, get_client, model_name, parse_with_retry
from src.schemas import Citation, ReportedAssociation, SafetyScreen, SourceType

# MedDRA reaction terms use international/British spelling ("diarrhoea",
# "anaemia", "oedema"); FDA label text uses US spelling ("diarrhea",
# "anemia", "edema"). Verified live: a naive substring check false-negatives
# on real, correctly-labeled risks purely over spelling. A blanket digraph
# collapse (ae/oe -> e) resolves the common cases without a synonym table.
_BRITISH_DIGRAPHS = re.compile(r"ae|oe", re.IGNORECASE)


def _normalize_for_label_match(text: str) -> str:
    return _BRITISH_DIGRAPHS.sub("e", text.lower())


def is_known_label_risk(term: str, label_reference_text: str) -> bool:
    """True if the reaction term (or its US-spelling equivalent) appears
    verbatim in the drug's real FDA label text.

    Deliberately conservative: only spelling is normalized, not medical
    synonyms -- "malignant neoplasm progression" will not match "disease
    progression" even though a clinician might treat them as related (see
    Module 8's Limitations). A false "unexpected" is the safer failure mode
    for this project's purpose than a false "known".
    """
    return _normalize_for_label_match(term) in _normalize_for_label_match(label_reference_text)


class _LlmNarrative(BaseModel):
    index: int = Field(description="Index into the supplied reaction list this narrative is for.")
    narrative: str = Field(
        min_length=1,
        description=(
            "Plain-language description of this reported association for a reviewer. "
            "Report association only -- never state or imply the drug caused the event."
        ),
    )


class _LlmNarratives(BaseModel):
    narratives: list[_LlmNarrative]


_SYSTEM_PROMPT = """\
You write plain-language safety narratives from FAERS (FDA Adverse Event \
Reporting System) data for a clinical trial reviewer. FAERS is voluntary, \
spontaneous-report data: a report does not establish that the product \
caused the event.

Rules, no exceptions:
1. Write exactly one narrative per supplied reaction, referencing it by its \
index.
2. Never use language that asserts or implies causation -- no "causes", \
"caused by", "causing", "leads to", "results in", "confirmed side effect", \
or similar. This applies EQUALLY to severe outcomes like death or disease \
progression -- do not write "N reports of death caused by the drug"; write \
"N reports of death were recorded in association with the drug" instead. \
Severity of the outcome is never a reason to relax this rule.
3. If a reaction is marked as a known/labeled risk, you may note that. If \
marked NOT on the current label, note it is an unexpected signal worth a \
closer look -- not a confirmed new risk.
4. Do not invent or restate a different report count than the one given."""


def _fallback_narrative(term: str, count: int, known: bool) -> str:
    """A deterministic, always-HG-3-compliant narrative, used only when the
    model's own narrative gets rejected (see screen()). Never asks the
    model again -- real safety data reaching the reviewer matters more than
    a second attempt at prose, and API calls are used deliberately in this
    project, not spent retrying for style."""
    status = (
        "a known, labeled risk" if known
        else "not on the current FDA label — an unexpected signal worth a closer look"
    )
    return (
        f"FAERS recorded {count} report(s) of '{term}' in association with this product. "
        f"This is {status}."
    )


def _build_prompt(drug: str, entries: list[dict]) -> str:
    listing = "\n".join(
        f"[{i}] {e['term']} -- {e['count']} reports -- "
        f"{'known/labeled risk' if e['known'] else 'NOT on current label (unexpected)'}"
        for i, e in enumerate(entries)
    )
    return f"Drug: {drug}\n\nReported reactions:\n{listing}"


def screen(drug: str, top_n: int = 10, run_id: str | None = None) -> SafetyScreen:
    """Real FAERS counts + a real label check + a model-written narrative per
    association.

    Raises NoDataFound (from src.ingest.openfda) if FAERS has no reports for
    the drug, or if no FDA label exists for it. Raises ValueError if the
    model omits a narrative for a supplied reaction -- every reaction has
    real data behind it, so a missing narrative is a bug to surface, not a
    gap to paper over. run_id correlates this call in the audit log
    (Module 12) with its Supervisor run; omit it outside one.
    """
    agg = fetch_reaction_counts(drug, limit=top_n)
    label = fetch_label(drug)

    entries = [
        {"term": r.term, "count": r.count, "known": is_known_label_risk(r.term, label.reference_text)}
        for r in agg.reactions
    ]

    client = get_client()
    response = parse_with_retry(
        client,
        agent="safety",
        run_id=run_id,
        model=model_name(),
        max_tokens=4000,
        output_config={"effort": EFFORT_EXTRACTION},
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(drug, entries)}],
        output_format=_LlmNarratives,
    )

    by_index = {n.index: n.narrative for n in response.parsed_output.narratives}
    missing = [i for i in range(len(entries)) if i not in by_index]
    if missing:
        raise ValueError(f"model omitted narratives for reaction indices {missing}")

    associations: list[ReportedAssociation] = []
    for i, entry in enumerate(entries):
        citation = Citation(
            chunk_id=f"faers:{drug.lower().replace(' ', '_')}",
            source_type=SourceType.FAERS,
            source_name=f"openFDA FAERS ({drug})",
            retrieved_date=agg.retrieved_date,
            quoted_text=f"{entry['term']}: {entry['count']} reports",
        )
        try:
            associations.append(
                ReportedAssociation(
                    adverse_event=entry["term"],
                    report_count=entry["count"],
                    known_label_risk=entry["known"],
                    narrative=by_index[i],
                    citation=citation,
                )
            )
        except ValidationError:
            # DEVIATION FROM PLAN, found on the first live run: HG-3's
            # causal-language check (correctly) rejected the model's
            # narrative for DEATH and MALIGNANT NEOPLASM PROGRESSION --
            # severe outcomes the model described as "caused by" the drug
            # despite the explicit instruction not to. This is real safety
            # data; letting one bad narrative crash the whole screen (or
            # silently drop that association) would hide exactly the
            # findings a reviewer most needs to see. Falls back to a
            # deterministic, always-compliant narrative instead of a
            # second model call -- consistent with using the API
            # deliberately, not as a retry loop.
            associations.append(
                ReportedAssociation(
                    adverse_event=entry["term"],
                    report_count=entry["count"],
                    known_label_risk=entry["known"],
                    narrative=_fallback_narrative(entry["term"], entry["count"], entry["known"]),
                    citation=citation,
                )
            )

    return SafetyScreen(drug=drug, associations=associations, total_reports_reviewed=agg.total_reports)
