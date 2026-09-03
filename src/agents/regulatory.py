"""Regulatory Agent (Module 7).

Given a protocol summary, selects relevant ICH E9 / FDA E9(R1) clauses
(src/agents/clauses.py — a keyword stopgap until Module 4 exists) and asks
the model to raise findings against them. Same grounding discipline as
Modules 5-6: the model must quote the clause it's citing, and Python verifies
the quote before trusting it.

The vocabulary constraint from Module 2 does the rest of the work here.
Severity only has INFO and FLAG — there is no VIOLATION or NON_COMPLIANT to
even select, so the model cannot make a regulatory determination even if
asked to. That's enforced by the schema, not by a prompt instruction alone.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents.clauses import Clause, select_relevant_clauses, split_into_clauses
from src.agents.errors import GroundingError, is_grounded
from src.ingest.guidance import GUIDANCE_SOURCES, fetch_guidance
from src.llm import EFFORT_REASONING, get_client, model_name, parse_with_retry
from src.schemas import Citation, RegulatoryFinding, Severity, SourceType


class _LlmFinding(BaseModel):
    clause_id: str = Field(min_length=1, description="Must match one of the supplied clause ids exactly.")
    finding: str = Field(min_length=1, description="What the reviewer should look at, in plain language.")
    severity: Severity
    quoted_text: str = Field(min_length=1, description="Exact substring from that clause's text.")


class _LlmFindings(BaseModel):
    findings: list[_LlmFinding] = Field(default_factory=list)


_SYSTEM_PROMPT = """\
You are flagging points in a clinical trial protocol summary for a \
regulatory reviewer to check against ICH/FDA statistical guidance. You do \
NOT determine compliance and you do NOT approve or reject anything — you \
only point out where the protocol summary touches on something the guidance \
discusses, so a human reviewer can look closer.

Rules, no exceptions:
1. Every finding must cite one of the supplied clause_ids exactly as given.
2. quoted_text must be copied VERBATIM from that clause's text — an exact \
substring, not a paraphrase or summary.
3. Never use words like "compliant", "violates", "non-compliant", or \
"approved" — you flag topics for review, you do not rule on them.
4. If the protocol summary doesn't clearly relate to a supplied clause, \
don't force a finding just to have output. An empty findings list is a \
correct answer when nothing is genuinely worth flagging.
5. severity="flag" only for something a reviewer should actively check; \
severity="info" for background context worth noting but not urgent."""


def _build_prompt(protocol_summary: str, clauses: list[Clause]) -> str:
    listing = "\n\n".join(
        f"Clause {c.clause_id} ({c.title}):\n{c.text[:1500]}" for c in clauses
    )
    return (
        f"Protocol summary:\n{protocol_summary}\n\n"
        f"Relevant guidance clauses:\n\n{listing}\n\n"
        "Raise findings only where genuinely relevant. Cite clause_id exactly."
    )


def review(protocol_summary: str, guidance_name: str = "ICH_E9", top_k: int = 6) -> list[RegulatoryFinding]:
    """Review a protocol summary against one guidance document.

    Raises GroundingError if the model cites a clause_id we didn't supply, or
    a quote that isn't a literal (whitespace-normalized) substring of that
    clause's actual text.
    """
    doc = fetch_guidance(guidance_name)
    all_clauses = split_into_clauses(doc.text)
    relevant = select_relevant_clauses(all_clauses, protocol_summary, top_k=top_k)

    if not relevant:
        return []  # honest: nothing in this guidance matched, don't force a call

    by_id = {c.clause_id: c for c in relevant}
    client = get_client()
    response = parse_with_retry(
        client,
        model=model_name(),
        # DEVIATION FROM PLAN: first live run truncated mid-JSON at 4000
        # tokens (Invalid JSON: EOF while parsing a string). effort=high on
        # this agent means longer reasoning + several multi-clause findings;
        # 4000 wasn't enough headroom for the actual output. Raised to 8000.
        max_tokens=8000,
        output_config={"effort": EFFORT_REASONING},
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(protocol_summary, relevant)}],
        output_format=_LlmFindings,
    )

    findings: list[RegulatoryFinding] = []
    for llm_finding in response.parsed_output.findings:
        clause = by_id.get(llm_finding.clause_id)
        if clause is None:
            raise GroundingError(
                f"model cited clause_id {llm_finding.clause_id!r}, not among "
                f"the {len(relevant)} clauses supplied"
            )
        if not is_grounded(llm_finding.quoted_text, clause.text):
            raise GroundingError(
                f"model claimed a quote not present in clause {clause.clause_id}: "
                f"{llm_finding.quoted_text!r}"
            )
        citation = Citation(
            chunk_id=f"guidance:{guidance_name}:{clause.clause_id}",
            source_type=SourceType.GUIDANCE,
            source_name=f"{GUIDANCE_SOURCES[guidance_name]['title']} — {clause.clause_id}",
            retrieved_date=doc.retrieved_date,
            quoted_text=llm_finding.quoted_text,
        )
        findings.append(
            RegulatoryFinding(
                clause_id=f"{guidance_name} {clause.clause_id}",
                finding=llm_finding.finding,
                severity=llm_finding.severity,
                citation=citation,
            )
        )
    return findings
