"""Regulatory Agent (Module 7). Retrieval upgraded in Module 4.

Given a protocol summary, retrieves relevant guidance clauses from the shared
knowledge base (hybrid vector + BM25 search, T-17) and asks the model to
raise findings against them. Module 7 originally built a keyword-overlap
stopgap (src/agents/clauses.py, since retired) because no knowledge base
existed yet; that file's clause-splitting logic now lives in
src/kb/chunking.py, and selection is real hybrid retrieval instead of
keyword overlap alone.

Same grounding discipline as Modules 5-6: the model must quote the clause
it's citing, and Python verifies the quote before trusting it.

The vocabulary constraint from Module 2 does the rest of the work here.
Severity only has INFO and FLAG -- there is no VIOLATION or NON_COMPLIANT to
even select, so the model cannot make a regulatory determination even if
asked to. That's enforced by the schema, not by a prompt instruction alone.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents.errors import GroundingError, is_grounded
from src.kb.store import KnowledgeBase
from src.llm import EFFORT_REASONING, get_client, model_name, parse_with_retry
from src.schemas import Chunk, Citation, RegulatoryFinding, Severity, SourceType


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
2. quoted_text must be copied VERBATIM from THE CLAUSE you are citing — an \
exact substring of the guidance text, not a paraphrase or summary. Never \
quote the protocol summary here: the protocol summary is the thing being \
reviewed, and the clause is the guidance you are citing against it. If a \
clause has no sentence that supports your point, drop the finding rather \
than quoting the protocol back.
3. Never use words like "compliant", "violates", "non-compliant", or \
"approved" — you flag topics for review, you do not rule on them.
4. If the protocol summary doesn't clearly relate to a supplied clause, \
don't force a finding just to have output. An empty findings list is a \
correct answer when nothing is genuinely worth flagging.
5. severity="flag" only for something a reviewer should actively check; \
severity="info" for background context worth noting but not urgent."""


def _display_label(chunk: Chunk) -> str:
    """A short, model-facing identifier: "ICH_E9 4.5" when the source has a
    real numbered clause; the chunk_id itself otherwise.

    DEVIATION FROM PLAN, found running this module live in two stages:

    1. Against both real guidance documents together: ICH E9 uses
       decimal-numbered headers ("2.2.1 Title") that src/kb/chunking.py's
       clause splitter parses cleanly (50 real clauses). FDA E9(R1) uses a
       different scheme entirely — Roman numerals with lettered
       sub-sections and a parenthetical annex number
       ("III. ESTIMANDS (A.3)") — which the same regex does not match, so
       it falls back to generic paragraph chunking with no clause_id in its
       metadata. Building a second regex for one document's idiosyncratic
       numbering was rejected as scope creep with no reuse value; findings
       against FDA E9(R1) are labeled by chunk_id instead of a clause
       number. Retrieval and grounding are unaffected — this only changes
       how the citation displays.

    2. First live run using the *raw* chunk_id ("guidance:ICH_E9:4.5") as
       the identifier the model must echo back verbatim raised
       GroundingError: the model returned "ICH_E9:4.5", dropping the
       "guidance:" prefix as redundant. Not a fabrication — an identifier
       reformatting, the same class of issue as the whitespace/case
       grounding fixes in Modules 6-7. Fixed by using this short label
       (what Module 7 originally used, before the KB existed) as the
       identifier shown to and expected back from the model, reserving the
       raw chunk_id for internal bookkeeping only.

    3. Module 9 found that fix was only half applied. Deviation 2 changed
       the clause-structured branch but left the fallback branch returning
       the raw chunk_id — so an FDA E9(R1) chunk still asked the model to
       echo "guidance:FDA_E9R1:9", and it still stripped the prefix,
       failing identically ("model cited clause_id 'FDA_E9R1:9'"). It only
       surfaced once a run actually retrieved a generic-chunked passage.
       Both branches now produce the same short "<doc> <position>" shape,
       with no namespace prefix for the model to tidy away.
    """
    clause_id = chunk.metadata.get("clause_id")
    if clause_id:
        return f"{chunk.metadata['guidance_key']} {clause_id}"
    # Generic-chunked guidance: "guidance:FDA_E9R1:9" -> "FDA_E9R1 9".
    return f"{chunk.doc_id.split(':')[-1]} {chunk.chunk_id.rsplit(':', 1)[-1]}"


def _build_prompt(protocol_summary: str, clauses: dict[str, tuple[str, str]]) -> str:
    listing = "\n\n".join(
        f"Clause {clause_id} ({title}):\n{text[:1500]}"
        for clause_id, (title, text) in clauses.items()
    )
    return (
        f"Protocol summary:\n{protocol_summary}\n\n"
        f"Relevant guidance clauses:\n\n{listing}\n\n"
        "Raise findings only where genuinely relevant. Cite clause_id exactly."
    )


def review(protocol_summary: str, kb: KnowledgeBase, top_k: int = 6) -> list[RegulatoryFinding]:
    """Review a protocol summary against the guidance subset of the shared
    knowledge base.

    Raises GroundingError if the model cites a clause_id we didn't supply, or
    a quote that isn't a literal (whitespace/case-normalized) substring of
    that clause's actual text.
    """
    results = kb.search(protocol_summary, top_k=top_k, source_type=SourceType.GUIDANCE)
    if not results:
        return []  # honest: nothing in the guidance corpus matched, don't force a call

    # Keyed by the short display label (see _display_label) -- unique across
    # one search's results, and the identifier shape the model reproduces
    # most reliably (deviation 2 in _display_label's docstring).
    by_id = {_display_label(r.chunk): r.chunk for r in results}
    prompt_clauses = {cid: (chunk.title, chunk.text) for cid, chunk in by_id.items()}

    client = get_client()
    response = parse_with_retry(
        client,
        model=model_name(),
        # Truncation observed in Module 7's first live run at max_tokens=4000
        # under effort=high with several multi-clause findings. Raised to 8000.
        max_tokens=8000,
        output_config={"effort": EFFORT_REASONING},
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(protocol_summary, prompt_clauses)}],
        output_format=_LlmFindings,
    )

    findings: list[RegulatoryFinding] = []
    for llm_finding in response.parsed_output.findings:
        chunk = by_id.get(llm_finding.clause_id)
        if chunk is None:
            raise GroundingError(
                f"model cited clause_id {llm_finding.clause_id!r}, not among "
                f"the {len(results)} clauses supplied"
            )
        if not is_grounded(llm_finding.quoted_text, chunk.text):
            raise GroundingError(
                f"model claimed a quote not present in clause {llm_finding.clause_id}: "
                f"{llm_finding.quoted_text!r}"
            )
        citation = Citation(
            chunk_id=chunk.chunk_id,
            source_type=chunk.source_type,
            source_name=chunk.source_name,
            retrieved_date=chunk.retrieved_date,
            quoted_text=llm_finding.quoted_text,
        )
        findings.append(
            RegulatoryFinding(
                clause_id=_display_label(chunk),
                finding=llm_finding.finding,
                severity=llm_finding.severity,
                citation=citation,
            )
        )
    return findings
